from __future__ import annotations

import torch

from tabu_lab.experiments import (
    make_linear_world_batch,
    make_query_row_synthetic_episode,
    run_query_row_fixed_world_fit,
    run_query_row_frozen_icl,
    run_query_row_multi_world_fit,
    run_synthetic_fit,
)


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


def test_tabur_fixed_world_keeps_masked_truth_out_of_public_evidence() -> None:
    episode = make_query_row_synthetic_episode(seed=1730, rows=12, row_token_count=4)
    hidden = episode.sidecar.target_mask
    assert torch.equal(
        episode.evidence.forward_values[hidden],
        torch.zeros_like(episode.sidecar.target_values[hidden]),
    )
    assert episode.evidence.target_mask.equal(hidden)
    assert episode.sidecar.target_count > 0


def test_tabur_fixed_world_f0_fit_gate_passes() -> None:
    result = run_query_row_fixed_world_fit(seed=1729, steps=20, rows=32, row_token_count=4)
    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.final_train_loss < result.initial_train_loss
    assert torch.isfinite(torch.tensor(result.final_validation_loss))


def test_tabur_multi_world_s1_fit_gate_passes_without_real_data() -> None:
    result = run_query_row_multi_world_fit(
        seed=1729,
        steps=20,
        train_worlds=4,
        validation_worlds=2,
        rows=24,
        row_token_count=4,
    )
    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.final_train_loss < result.initial_train_loss
    assert torch.isfinite(torch.tensor(result.final_validation_loss))


def test_tabur_stage5_frozen_icl_has_no_optimizer_or_parameter_mutation() -> None:
    result = run_query_row_frozen_icl(
        seed=1729,
        pretrain_steps=8,
        pretrain_worlds=2,
        context_rows=(2, 4),
        rows=12,
        row_token_count=4,
    )
    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.records
    assert all(record.parameter_hash_unchanged for record in result.records)
    assert all(not record.optimizer_created for record in result.records)
