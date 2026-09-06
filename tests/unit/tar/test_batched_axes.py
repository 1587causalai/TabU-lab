"""Numerical equivalence to the frozen serial implementation, including gradients."""

from dataclasses import replace

import pytest
import torch
from serial_reference import SerialAxisBlock, SerialOMAB

from tabu_lab.models.tar import TabUTARModel, TARConfig, score
from tabu_lab.models.tar.attention import TAROMAB
from tabu_lab.models.tar.training import TARTrainer, TARTrainingConfig
from tabu_lab.models.tar.verification import mixed_fixture


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**kw):
    return TARConfig(
        width=16, heads=4, ff_width=32, blocks=2, semantic_slots=3, inducing_slots=8, **kw
    )


def grad(value, parameter):
    return torch.zeros_like(parameter) if value is None else value


@pytest.mark.parametrize("chunk", [1, 3, 2048])
def test_batched_masked_omab_matches_serial_values_and_gradients(chunk):
    torch.manual_seed(52)
    cfg = config(source_chunk=chunk, receiver_chunk_rows=2)
    new, old = TAROMAB(cfg, dtype=torch.float64), SerialOMAB(cfg, dtype=torch.float64)
    old.load_state_dict(new.state_dict())
    x = torch.randn(4, 5, 16, dtype=torch.float64, requires_grad=True)
    raw = torch.randn(4, 7, 16, dtype=torch.float64)
    raw[0, 0] = 0
    raw[1, 2] = 3e-8
    s = raw.requires_grad_()
    xo, so = x.detach().clone().requires_grad_(), s.detach().clone().requires_grad_()
    visible = torch.ones(4, 7, dtype=torch.bool)
    visible[1, :2] = False  # Entire first chunks can be empty.
    visible[2] = False  # Entire group has no sources.
    null = torch.zeros(4, 5, dtype=torch.bool)
    null[0, 1] = True
    null[3] = True  # Another group has only Null receivers.
    y = new(x, s, source_mask=visible, null_mask=null)
    expected = torch.stack(
        [old(xo[i], so[i], source_mask=visible[i], null_mask=null[i]) for i in range(4)]
    )
    torch.testing.assert_close(y, expected, rtol=1e-12, atol=1e-12)
    weights = torch.randn_like(y)
    (y * weights).sum().backward()
    (expected * weights).sum().backward()
    for a, b in [(x, xo), (s, so), *zip(new.parameters(), old.parameters(), strict=True)]:
        torch.testing.assert_close(grad(a.grad, a), grad(b.grad, b), rtol=1e-10, atol=1e-11)
    assert torch.count_nonzero(s.grad[~visible]) == 0
    assert torch.count_nonzero(x.grad[null]) == 0
    changed = s.detach().clone()
    changed[1:] *= 100
    torch.testing.assert_close(
        new(x, changed, source_mask=visible, null_mask=null)[0], y[0], rtol=0, atol=0
    )


@pytest.mark.parametrize("inducing", [False, True])
@pytest.mark.parametrize("columns_per_chunk", [1, 4, 64])
def test_complete_model_matches_serial(inducing, columns_per_chunk):
    cfg = config(inducing_enabled=inducing, collect_column_chunk=columns_per_chunk)
    new, old = TabUTARModel(cfg, dtype=torch.float64), TabUTARModel(cfg, dtype=torch.float64)
    old.layers = torch.nn.ModuleList(
        [SerialAxisBlock(cfg, dtype=torch.float64) for _ in range(cfg.blocks)]
    )
    old.load_state_dict(new.state_dict())
    ep, truth = mixed_fixture()
    y, expected = new(ep), old(ep)
    torch.testing.assert_close(y.carriers, expected.carriers, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(y.responses, expected.responses, rtol=1e-11, atol=1e-12)
    for a, b in zip(y.predictions, expected.predictions, strict=True):
        torch.testing.assert_close(a.value, b.value, rtol=1e-10, atol=1e-11)
        if a.probabilities is not None:
            torch.testing.assert_close(a.probabilities, b.probabilities, rtol=1e-11, atol=1e-12)
    loss, old_loss = score(y, truth), score(expected, truth)
    torch.testing.assert_close(loss, old_loss, rtol=1e-10, atol=1e-11)
    loss.backward()
    old_loss.backward()
    for (name, a), (other, b) in zip(new.named_parameters(), old.named_parameters(), strict=True):
        assert name == other
        torch.testing.assert_close(grad(a.grad, a), grad(b.grad, b), rtol=1e-8, atol=1e-10)
    training = TARTrainingConfig(effective_episode_batch=1, warmup_steps=0, optimizer_steps=2)
    nt, ot = TARTrainer(new, training), TARTrainer(old, training)
    nt.train_step([(ep, truth)])
    ot.train_step([(ep, truth)])
    for a, b in zip(new.parameters(), old.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


def test_empty_source_axis_and_multiple_batch_axes():
    op = TAROMAB(config(), dtype=torch.float64)
    x = torch.randn(2, 3, 4, 16, dtype=torch.float64, requires_grad=True)
    s = torch.empty(2, 3, 0, 16, dtype=torch.float64)
    y = op(x, s)
    expected = torch.stack(
        [torch.stack([op(x[i, j], s[i, j]) for j in range(3)]) for i in range(2)]
    )
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_empty_chunk_does_not_rescale_negative_logits():
    cfg = config(source_chunk=1)
    full, chunked = (
        TAROMAB(replace(cfg, source_chunk=100), dtype=torch.float64),
        TAROMAB(cfg, dtype=torch.float64),
    )
    with torch.no_grad():
        full.q.weight.zero_()
        full.q.bias.fill_(100)
        full.k.weight.zero_()
        full.k.bias.fill_(-100)
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(2, 3, 16, dtype=torch.float64)
    s = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[False, True, False, True], [False, False, False, False]])
    y = chunked(x, s, source_mask=mask)
    torch.testing.assert_close(y, full(x, s, source_mask=mask), rtol=1e-12, atol=1e-12)
    y.sum().backward()
    assert torch.isfinite(s.grad).all()
