"""Exact-zero source deletion also precedes content projection in batched OMAB."""

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
