from __future__ import annotations

import math

from tabu_lab.experiments import run_query_row_finetune_lift


def test_tabur_stage6_paired_finetune_lift_is_finite_and_profile_compatible() -> None:
    result = run_query_row_finetune_lift(
        dataset_ids=("iris", "diabetes"),
        label_budget=32,
        updates=2,
        pretrain_steps=3,
        pretrain_worlds=2,
        test_limit=8,
    )
    assert result.status == "pass"
    assert result.execution_status == "succeeded"
    assert result.capability_gate == "not_applicable"
    assert result.evidence_status == "local_unissued"
    assert result.profile_id == "supervised.label_broadcast.v1"
    assert math.isfinite(result.pretrain_final_loss)
    assert len(result.records) == 2
    for record in result.records:
        assert math.isfinite(record.scratch_loss)
        assert math.isfinite(record.pretrained_loss)
        assert math.isfinite(record.gain_scratch_minus_pretrained)
        assert all(math.isfinite(value) for value in record.scratch_metrics.values())
        assert all(math.isfinite(value) for value in record.pretrained_metrics.values())
        assert record.exact_same_init
        assert (
            record.scratch_initial_parameter_sha256
            == record.pretrain_initial_parameter_sha256
        )
        assert (
            record.pretrained_initial_parameter_sha256
            == record.pretrained_checkpoint_parameter_sha256
        )
        assert record.scratch_episode_schedule_sha256 == record.pretrained_episode_schedule_sha256
