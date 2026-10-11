"""Optional readout contracts and dependency checks against inherited dynamics."""

import pytest
import torch
from torch.nn import functional as F

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    from_v7,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.completion_v8.readout import PositiveLinearReadout, SignedLinearReadout
from tabu_lab.models.restoration_v7 import V7Model

from .test_completion import data, spec
from .test_completion import runtime as runtime


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_factored_readout_and_response_contract(dtype):
    head = PositiveLinearReadout().to(dtype)
    q, k, v = [
        torch.randn(*shape, dtype=dtype, requires_grad=True)
        for shape in [(4, 128), (7, 128), (7, 64)]
    ]
    weights = (F.softplus(head.q(q)) + 1e-4) @ (F.softplus(head.k(k)) + 1e-4).T
    weights = weights / weights.sum(-1, keepdim=True)
    actual = head(q, k, v)
    torch.testing.assert_close(actual, weights @ v)
    perm = torch.randperm(7)
    torch.testing.assert_close(actual, head(q, k[perm], v[perm]))
    torch.testing.assert_close(head(q, k, v[:1].expand(7, -1)), v[:1].expand(4, -1))
    torch.testing.assert_close(head(q, k[:1], v[:1]), v[:1].expand(4, -1))
    actual.square().sum().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    with pytest.raises(ValueError, match="nonempty"):
        head(q, k[:0], v[:0])


@pytest.mark.parametrize("readout", ["positive_linear_attention", "signed_linear_attention"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_optional_end_to_end_no_solver_leakage_reload(tmp_path, monkeypatch, device, readout):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    cfg = V8Config(spec(), rounds=2, readout=readout)
    model = V8CompletionModel(cfg).to(device)

    def forbidden(*a, **kw):
        raise AssertionError("optional readout called LL")

    monkeypatch.setattr("tabu_lab.models.completion_v8.model.column_shared_ll", forbidden)
    inputs, truth = data(torch.float32, device)
    changed, _ = data(torch.float32, device, changed=True)
    ep = prepare_completion(inputs, cfg)
    out = model(ep)
    assert torch.equal(out.completed_codes, model(prepare_completion(changed, cfg)).completed_codes)
    assert torch.equal(
        out.completed_codes[inputs.visible], ep.observed.to(out.completed_codes)[inputs.visible]
    )
    completion_loss(out, ep, truth).backward()
    for p in model.attention_readout.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    path = tmp_path / "optional.pt"
    save_weights(model, path)
    restored = load_weights(path, device=device)
    assert restored.config == cfg
    assert torch.equal(out.completed_codes, restored(ep).completed_codes)


@pytest.mark.parametrize("readout", ["positive_linear_attention", "signed_linear_attention"])
def test_default_serialization_and_migration(readout):
    old = V8Config(spec())
    assert set(old.as_dict()) == {"inherited", "rounds"}
    assert V8Config.from_dict(old.as_dict()) == old
    assert not any("attention_readout" in k for k in V8CompletionModel(old).state_dict())
    model = from_v7(V7Model(spec()), readout=readout)
    assert "attention_readout.q.weight" in model.migration_receipt["new_keys"]
    assert V8Config.from_dict(model.config.as_dict()) == model.config
    with pytest.raises(ValueError):
        V8Config(spec(), readout="unknown")


def test_unit_geometry_is_not_a_source_for_cell_only_readout():
    # Unit/Feature receivers summarize cells; their states are not cell sources
    # in this inherited implementation. Unit-only terminal layers feed LL only.
    cfg = V8Config(spec(), readout="positive_linear_attention")
    model = V8CompletionModel(cfg).double()
    inputs, _ = data()
    ep = prepare_completion(inputs, cfg)
    baseline = model(ep).completed_codes
    with torch.no_grad():
        model.unit_seed.add_(3)
        model.feature_seed.sub_(2)
        for parameter in model.backbone.unit_blocks.parameters():
            parameter.mul_(2)
    assert torch.equal(baseline, model(ep).completed_codes)


@pytest.mark.parametrize("kind,row_slots", [("inducing", 0), ("inducing", 2)])
@pytest.mark.parametrize("query_source", [True, False])
def test_cell_only_matches_inherited_cell_outputs_and_gradients(kind, row_slots, query_source):
    from dataclasses import replace

    from tabu_lab.models.completion_v8.cell_dynamics import cell_only_dynamics

    inherited = spec()
    inherited = replace(
        inherited, backbone=replace(inherited.backbone, kind=kind, row_slots=row_slots)
    )
    model = V8CompletionModel(V8Config(inherited)).double()
    inputs, _ = data()
    cells = torch.randn(7, 3, 128, dtype=torch.float64, requires_grad=True)
    valid = inputs.visible | inputs.query
    sources = valid if query_source else inputs.visible
    full = torch.cat(
        (
            torch.cat((cells, model.unit_seed.expand(7, 1, -1)), 1),
            torch.cat((model.feature_seed.expand(3, -1), cells.new_zeros(1, 128)))[None],
        ),
        0,
    )
    expected = model.backbone(full, sources, inputs.query)[:7, :3]
    actual = cell_only_dynamics(model.backbone, cells, sources, inputs.query)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    g1 = torch.autograd.grad(expected.square().sum(), cells, retain_graph=True)[0]
    g2 = torch.autograd.grad(actual.square().sum(), cells)[0]
    torch.testing.assert_close(g1, g2, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_signed_factored_values_and_derivatives_match_explicit_weights(dtype):
    head = SignedLinearReadout().to(dtype)
    query, support, values = [
        torch.randn(*shape, dtype=dtype, requires_grad=True)
        for shape in [(3, 128), (513, 128), (513, 64)]
    ]
    q, k = F.silu(head.q(query)), F.silu(head.k(support))
    expected = ((q @ k.T) @ values) / 32**0.5
    actual = head(query, support, values)
    tol = 3e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    parameters = (query, support, values, *head.parameters())
    expected_grad = torch.autograd.grad(expected.square().mean(), parameters, retain_graph=True)
    actual_grad = torch.autograd.grad(actual.square().mean(), parameters)
    for x, y in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(x, y, atol=tol, rtol=tol)
    perm = torch.randperm(513)
    torch.testing.assert_close(actual, head(query, support[perm], values[perm]), atol=tol, rtol=tol)


def test_signed_coefficients_are_not_convex_and_no_layer_norm(monkeypatch):
    head = SignedLinearReadout().double()
    with torch.no_grad():
        head.q.weight.zero_()
        head.k.weight.zero_()
        head.q.weight[0, 0] = 32
        head.k.weight[0, 0] = -1

    def forbidden(*a, **kw):
        raise AssertionError("readout must not add LayerNorm")

    monkeypatch.setattr(F, "layer_norm", forbidden)
    h = torch.zeros(1, 128, dtype=torch.float64)
    h[0, 0] = 1
    values = torch.ones(1, 64, dtype=torch.float64)
    out = head(h, h, values)
    assert bool((out < -1).all())  # Signed coefficients can also exceed unit magnitude.
    torch.testing.assert_close(head(h, h.expand(2, -1), values.expand(2, -1)), 2 * out)
    with pytest.raises(ValueError, match="nonempty"):
        head(h, h[:0], values[:0])
