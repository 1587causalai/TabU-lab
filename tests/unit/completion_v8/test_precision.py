"""Scoped precision preserves checkpoints, reference replay and finite gradients."""

import pytest
import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    configure_precision,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import V7Config, V7Model
from tabu_lab.models.restoration_v7.readout import column_shared_ll


@pytest.fixture(autouse=True)
def runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def model_and_data():
    torch.manual_seed(129)
    spec = V7Config.v73(
        backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
        unit_layers=0,
        coupling_blocks=1,
        coupling_hidden=(16,),
        center_chunk_size=4,
    )
    model = V8CompletionModel(V8Config(spec)).float()
    x = torch.linspace(-1, 1, 9)
    values = (x, x.square() + 0.1 * x)
    observed = torch.ones(9, 2, dtype=torch.bool)
    query = torch.zeros_like(observed)
    query[-3:, -1] = True
    inputs, _, truth = make_episode(
        (ColumnSchema("x", "numeric"), ColumnSchema("y", "numeric")),
        values, observed, query, code_seed=13,
    )
    return model, inputs, truth


def test_fast_precision_is_scoped_and_restores_reference(tmp_path):
    model, inputs, _ = model_and_data()
    other, _, _ = model_and_data()
    model.eval()
    configure_precision(model, mode="reference")
    configure_precision(other, mode="reference")
    baseline = model(prepare_completion(inputs, model.config), decode=False).states[-1].detach()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    receipt = configure_precision(model)
    probe = torch.ones(2, 128)
    assert receipt['modules'] > 0
    assert model.backbone.layers[0].row.accumulation_dtype(probe) == torch.float32
    assert other.backbone.layers[0].row.accumulation_dtype(probe) == torch.float64
    assert column_shared_ll.__globals__['solve_dtype'](probe) == torch.float64
    fast = model(prepare_completion(inputs, model.config), decode=False).states[-1].detach()
    torch.testing.assert_close(fast, baseline, rtol=2e-3, atol=2e-4)
    for key, value in model.state_dict().items():
        assert torch.equal(value, state[key])
    checkpoint = tmp_path / 'weights.pt'
    save_weights(model, checkpoint)
    loaded = load_weights(checkpoint)
    assert loaded.backbone.layers[0].row.accumulation_dtype(probe) == torch.float32
    assert loaded._ll_compute_dtype == torch.float32
    replay_loaded = load_weights(checkpoint, precision="reference")
    assert replay_loaded._ll_compute_dtype is None
    torch.testing.assert_close(
        replay_loaded(prepare_completion(inputs, replay_loaded.config), decode=False).states[-1],
        baseline, rtol=0, atol=0,
    )
    configure_precision(model, mode='reference')
    replay = model(prepare_completion(inputs, model.config), decode=False).states[-1].detach()
    torch.testing.assert_close(replay, baseline, rtol=0, atol=0)


def test_fast_precision_completion_backward_and_inverse():
    model, inputs, truth = model_and_data()
    configure_precision(model)
    ep = prepare_completion(inputs, model.config)
    loss = completion_loss(model(ep, decode=False), ep, truth)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.query_seed.grad.abs().sum() > 0
    codes = torch.randn(5, 3, 64)
    valid = torch.ones(5, 3, dtype=torch.bool)
    restored = model.encoder.inverse(model.encoder(codes, valid), valid)
    torch.testing.assert_close(restored, codes, rtol=5e-5, atol=5e-5)


def test_precision_rejects_implicit_conversion_and_other_models():
    model, _, _ = model_and_data()
    with pytest.raises(ValueError, match='FP32'):
        configure_precision(model.double())
    with pytest.raises(ValueError, match='mode'):
        configure_precision(model, mode='float16')
    with pytest.raises(TypeError, match='V8'):
        configure_precision(V7Model(model.config.inherited))


def test_ll_precision_default_and_explicit_exception():
    model, _, _ = model_and_data()
    assert model.execution_precision["mode"] == "fast_fp32"
    assert model._ll_compute_dtype == torch.float32
    units = torch.randn(9, 4)
    features = torch.randn(9, 8)
    responses = torch.randn(6, 3)
    args = (units, torch.arange(6), responses, features, torch.arange(6, 9))
    fast = column_shared_ll(*args, ridge=0.01, bandwidth=1., compute_dtype=torch.float32)
    ref = column_shared_ll(*args, ridge=0.01, bandwidth=1.)
    assert fast.slope.dtype == fast.eval_weights.dtype == torch.float32
    assert ref.slope.dtype == torch.float64
    torch.testing.assert_close(fast.recovered.double(), ref.recovered, rtol=2e-3, atol=2e-4)
    with pytest.raises(ValueError, match="reason"):
        configure_precision(model, mode="fp32_attention_fp64_ll")
    configure_precision(model, mode="fp32_attention_fp64_ll", reason="historical mixed replay")
    assert model._ll_compute_dtype is None
    assert model.backbone.layers[0].row.accumulation_dtype(units) == torch.float32
    model.double()
    assert model.execution_precision["mode"] == "reference"
    assert model.backbone.layers[0].row.accumulation_dtype(units) == torch.float64
    model.float()
    assert model.execution_precision["mode"] == "fast_fp32"
    configure_precision(model, mode="reference")
    model.to("cpu")
    assert model.execution_precision["mode"] == "reference"
