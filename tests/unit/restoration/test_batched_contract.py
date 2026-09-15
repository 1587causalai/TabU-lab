"""Exact-zero source deletion also precedes content projection in batched OMAB."""

import pytest
import torch

from tabu_lab.models.restoration.backbone import OMAB, BackboneConfig


def test_projected_zero_sources_cannot_overflow_content_before_deletion():
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    with torch.no_grad():
        module.attention_presence.weight.zero_()
        module.q.weight.copy_(torch.eye(4))
        module.k.weight.copy_(torch.eye(4))
    receivers = torch.full((2, 3, 4), 1e120, dtype=torch.float64)
    sources = torch.full((2, 2, 4), 1e200, dtype=torch.float64, requires_grad=True)
    eligible = torch.ones(2, 2, dtype=torch.bool)
    expected = module.batched(receivers, torch.zeros_like(sources), eligible)
    actual = module.batched(receivers, sources, eligible)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    gradient, = torch.autograd.grad(actual.sum(), sources)
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_nonfinite_receivers_fail_at_the_omab_output_stage():
    """Stage-granularity checks: no per-tensor guards, output stays explicit."""
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    receivers = torch.full((1, 2, 4), float("nan"), dtype=torch.float64)
    sources = torch.zeros(1, 1, 4, dtype=torch.float64)
    eligible = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="OMAB output"):
        module.batched(receivers, sources, eligible)


def test_overflowing_presence_projection_is_not_treated_as_empty_source():
    """Finite inputs whose presence projection overflows must fail closed."""
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    with torch.no_grad():
        module.attention_presence.weight.zero_()
        module.attention_presence.weight[0, 0] = 2.0
    receivers = torch.zeros(1, 2, 4, dtype=torch.float64)
    sources = torch.tensor([[[1e308, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    eligible = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="OMAB output"):
        module.batched(receivers, sources, eligible)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_presence_projection_is_not_treated_as_empty_source(value):
    """NaN/Inf projections must not become exact-zero presence."""
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    receivers = torch.zeros(1, 2, 4, dtype=torch.float64)
    sources = torch.tensor([[[value, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    eligible = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="OMAB output"):
        module.batched(receivers, sources, eligible)


def test_nan_presence_weight_is_not_treated_as_empty_source():
    """A finite source with a nonfinite presence projection must fail closed."""
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    with torch.no_grad():
        module.attention_presence.weight.zero_()
        module.attention_presence.weight[0, 0] = float("nan")
    receivers = torch.zeros(1, 2, 4, dtype=torch.float64)
    sources = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    eligible = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(FloatingPointError, match="OMAB output"):
        module.batched(receivers, sources, eligible)


def test_ineligible_nan_source_payloads_stay_deleted_without_entering_checks():
    """NaN payloads in deleted sources never enter a projection, by design."""
    module = OMAB(BackboneConfig(width=4, heads=1, ff_width=8)).double()
    receivers = torch.zeros(1, 2, 4, dtype=torch.float64)
    sources = torch.zeros(1, 2, 4, dtype=torch.float64)
    sources[0, 1] = float("nan")
    eligible = torch.tensor([[True, False]])
    expected = module.batched(receivers, sources[:, :1], eligible[:, :1])
    actual = module.batched(receivers, sources, eligible)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
