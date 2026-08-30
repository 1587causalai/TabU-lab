from __future__ import annotations

import torch

from tabu_lab.experiments import make_linear_world_batch, run_synthetic_fit


def test_synthetic_world_keeps_masked_truth_out_of_forward_values() -> None:
    batch = make_linear_world_batch(seed=1730)
    hidden_response = batch.target_mask
    assert torch.equal(
        batch.model_input.values[:, :, 2][hidden_response],
        torch.zeros_like(batch.response_truth[hidden_response]),
    )
    assert torch.equal(batch.model_input.target_mask[:, :, 2], hidden_response)
    assert torch.isfinite(batch.response_truth).all()


def test_tabubase_passes_bounded_synthetic_fit_gate() -> None:
    result = run_synthetic_fit(seed=1729, steps=20)
    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.final_train_loss < result.initial_train_loss
    assert torch.isfinite(torch.tensor(result.final_validation_loss))
