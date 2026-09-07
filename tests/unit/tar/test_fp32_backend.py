"""The experimental auxiliary precision boundary keeps the reference path explicit."""

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.tar import TabUTARModel, TARConfig, score
from tabu_lab.models.tar.attention import log_presence, presence
from tabu_lab.models.tar.verification import mixed_fixture


def test_precision_guard():
    with pytest.raises(ValueError, match="numerical_backend"):
        TARConfig(numerical_backend="automatic")
    with pytest.raises(ValueError, match="requires float32"):
        TabUTARModel(TARConfig(numerical_backend="experimental_fp32"), dtype=torch.float64)
    with pytest.raises(ValueError, match="MPS requires"):
        TabUTARModel(device="mps")


def test_fp32_fixed_weights_predictions_and_gradients():
    torch.set_num_threads(1)
    cfg = TARConfig(width=16, heads=4, ff_width=32, blocks=2, semantic_slots=3, inducing_slots=8)
    ref = TabUTARModel(cfg)
    candidate = TabUTARModel(replace(cfg, numerical_backend="experimental_fp32"))
    e, truth = mixed_fixture()
    for a, b in zip(ref.parameters(), candidate.parameters(), strict=True):
        assert torch.equal(a, b)
    outputs = [m(e) for m in (ref, candidate)]
    losses = [score(o, truth) for o in outputs]
    assert losses[0].dtype == torch.float64
    assert losses[1].dtype == torch.float32
    torch.testing.assert_close(losses[1].double(), losses[0], atol=2e-5, rtol=2e-5)
    for loss in losses:
        loss.backward()
    ga, gb = [
        torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
        for m in (ref, candidate)
    ]
    assert torch.isfinite(gb).all()
    assert (ga - gb).norm() / ga.norm() < 1e-3
    assert not outputs[1].carriers[4, 2].any()
    assert not outputs[1].carriers[5:, 3:].any()


def test_fp32_presence_small_and_large_finite_carriers():
    x = torch.tensor([[1e-10, -2e-10], [1e10, -2e10], [0.0, 0.0]], requires_grad=True)
    gate = presence(x, fp32=True)
    torch.testing.assert_close(gate, presence(x), rtol=1e-5, atol=1e-25)
    assert gate[0] > 0 and gate[2] == 0
    gate.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(
        log_presence(x[:2], fp32=True), log_presence(x[:2]), rtol=1e-5, atol=1e-5
    )


def test_classification_truth_validated_before_fp32_cast():
    cfg = TARConfig(
        numerical_backend="experimental_fp32",
        width=16,
        heads=4,
        ff_width=32,
        blocks=2,
        semantic_slots=3,
        inducing_slots=8,
    )
    e, truth = mixed_fixture()
    truth[(4, 1)] = 1.00000001
    with pytest.raises(ValueError, match="out-of-domain"):
        score(TabUTARModel(cfg)(e), truth)
