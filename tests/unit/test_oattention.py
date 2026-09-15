from __future__ import annotations

import pytest
import torch

from tabu_lab.primitives import OMAB, presence_gate


def test_presence_gate_preserves_small_nonzero_float32_receivers() -> None:
    value = torch.full((1, 32), 3.0e-8, dtype=torch.float32)

    gate = presence_gate(value)

    assert gate.item() > 0.0
    assert gate.item() == pytest.approx(2.88e-8, rel=1.0e-5)


def test_presence_gate_keeps_exact_zero_and_large_values_well_defined() -> None:
    zero = torch.zeros((1, 32), dtype=torch.float32)
    large = torch.full((1, 32), 1.0e20, dtype=torch.float32)

    assert presence_gate(zero).item() == 0.0
    assert presence_gate(large).item() == 1.0


def test_omab_preserves_small_nonzero_receiver_residual_and_gradient() -> None:
    block = OMAB(
        d_model=4,
        n_heads=1,
        d_ff=8,
        dropout=0.0,
        presence_tau=1.0e-6,
    )
    receiver = torch.full(
        (1, 1, 4),
        3.0e-8,
        dtype=torch.float32,
        requires_grad=True,
    )
    source = torch.full((1, 1, 4), 0.1, dtype=torch.float32)

    output = block(receiver, source).state
    output.sum().backward()

    assert torch.any(output != 0.0)
    assert receiver.grad is not None
    assert torch.isfinite(receiver.grad).all()
    assert receiver.grad.abs().sum().item() > 0.0


def test_omab_keeps_exact_zero_receiver_zero_without_hard_writeback_mask() -> None:
    block = OMAB(
        d_model=4,
        n_heads=1,
        d_ff=8,
        dropout=0.0,
        presence_tau=1.0e-6,
    )
    receiver = torch.zeros((1, 1, 4), dtype=torch.float32)
    source = torch.full((1, 1, 4), 0.1, dtype=torch.float32)

    output = block(receiver, source).state

    assert torch.equal(output, torch.zeros_like(output))
