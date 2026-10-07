"""Optional variable-axis inducing preserves source roles and table symmetries."""

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.backbone import AxialBackbone, AxialLayer, BackboneConfig


@pytest.fixture(autouse=True)
def deterministic_cpu():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(37)
        yield


def config(**overrides):
    return BackboneConfig(
        **(dict(width=8, heads=2, layers=2, ff_width=12, slots=3, row_slots=2) | overrides)
    )


@pytest.mark.parametrize("invalid", [-1, 1.5, True, None, "2"])
def test_row_slots_requires_a_nonnegative_integer(invalid):
    with pytest.raises(ValueError, match="row_slots"):
        config(row_slots=invalid)


def test_default_disables_row_slots_without_parameters_or_random_draws():
    cfg = BackboneConfig(width=8, heads=2, layers=2, ff_width=12, slots=3)
    before = torch.get_rng_state()
    default = AxialBackbone(cfg)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    explicit = AxialBackbone(replace(cfg, row_slots=0))
    assert torch.equal(torch.get_rng_state(), after)
    assert all(
        layer.row_collect is None and layer.row_slot_seed is None for layer in default.layers
    )
    assert not any("row_slot" in k or "row_collect" in k for k in default.state_dict())
    explicit.load_state_dict(default.state_dict(), strict=True)


def row_layer():
    layer = AxialLayer(config(kind="direct")).double()
    # Isolate row communication: make the preceding column stage identity.
    with torch.no_grad():
        layer.column.out.weight.zero_()
        layer.column.ff[-1].weight.zero_()
    return layer


@pytest.mark.parametrize("empty_by_projection", [False, True])
def test_empty_rows_cannot_read_seed_residuals_and_keep_local_updates(empty_by_projection):
    layer = row_layer()
    h = torch.randn(4, 5, 8, dtype=torch.float64)
    sources = torch.zeros(4, 5, dtype=torch.bool)
    if empty_by_projection:
        sources[:3, :4] = True
        with torch.no_grad():
            layer.row_collect.attention_presence.weight.zero_()
    nulls = torch.zeros_like(sources)
    nulls[-1, -1] = True
    h = h.masked_fill(nulls[..., None], 0)
    expected = layer.row.batched(h, torch.zeros_like(h), sources)
    actual = layer(h, sources, nulls)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.no_grad():
        layer.row_slot_seed.add_(100)
    changed = layer(h, sources, nulls)
    torch.testing.assert_close(changed, actual, rtol=0, atol=0)
    changed.sum().backward()
    assert torch.equal(layer.row_slot_seed.grad, torch.zeros_like(layer.row_slot_seed))


def test_source_mask_controls_communication_to_unit_and_feature_stays_local():
    layer = row_layer()
    h = torch.randn(3, 4, 8, dtype=torch.float64)
    sources = torch.zeros(3, 4, dtype=torch.bool)
    sources[:2, 0] = True
    nulls = torch.zeros_like(sources)
    nulls[-1, -1] = True
    changed = h.clone()
    changed[0, 1] += 3
    original = layer(h, sources, nulls)
    excluded = layer(changed, sources, nulls)
    assert torch.equal(original[0, -1], excluded[0, -1])
    sources[0, 1] = True  # Query-as-source can explicitly admit the same Cell.
    admitted = layer(h, sources, nulls)
    admitted_changed = layer(changed, sources, nulls)
    assert not torch.allclose(admitted[0, -1], admitted_changed[0, -1])
    assert torch.equal(admitted[1:], admitted_changed[1:])
    assert torch.equal(original[-1], admitted[-1])
    assert not bool(admitted[-1, -1].any())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dual_axis_inducing_is_equivariant_null_closed_and_differentiable(dtype):
    model = AxialBackbone(config()).to(dtype=dtype)
    h = torch.randn(5, 6, 8, dtype=dtype, requires_grad=True)
    sources = torch.rand(4, 5) > 0.4
    queries = ~sources & (torch.rand(4, 5) > 0.5)
    output = model(h, sources, queries)
    assert output.shape == h.shape and torch.isfinite(output).all()
    assert not bool(output[:4, :5][~(sources | queries)].any())
    assert not bool(output[-1, -1].any())
    rp = torch.tensor([2, 0, 3, 1, 4])
    cp = torch.tensor([3, 1, 4, 0, 2, 5])
    permuted = model(
        h[rp][:, cp], sources[rp[:-1]][:, cp[:-1]], queries[rp[:-1]][:, cp[:-1]]
    )
    torch.testing.assert_close(permuted, output[rp][:, cp])
    output.square().sum().backward()
    assert torch.isfinite(h.grad).all()
    for layer in model.layers:
        for parameter in (
            layer.row_slot_seed, layer.row_collect.k.weight, layer.row_collect.v.weight
        ):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert bool(parameter.grad.abs().sum() > 0)
