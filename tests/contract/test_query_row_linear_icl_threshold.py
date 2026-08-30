from __future__ import annotations

import math

from tabu_lab.experiments import (
    LINEAR_REGRESSION_BASELINE_ID,
    run_query_row_linear_icl_threshold,
)


def test_query_row_linear_threshold_is_bounded_and_context_bucketed() -> None:
    result = run_query_row_linear_icl_threshold(
        seed=1729,
        pretrain_rows=8,
        pretrain_worlds=2,
        pretrain_steps=4,
        eval_rows=16,
        eval_worlds=2,
        context_rows=(4, 8),
        device="cpu",
    )

    assert result.status in {"pass", "continue"}
    assert result.threshold_met is (result.status == "pass")
    assert result.evidence_status == "local_unissued"
    assert result.baseline_id == LINEAR_REGRESSION_BASELINE_ID
    assert result.threshold_ratio == 1.0
    assert result.parameter_hash_unchanged
    assert result.context_rows == (4, 8)
    assert tuple(summary.context_rows for summary in result.context_summaries) == (4, 8)
    assert len(result.records) == 4
    assert all(record.target_count > 0 for record in result.records)
    assert all(
        math.isfinite(record.pretrained_mse)
        and math.isfinite(record.linear_regression_mse)
        for record in result.records
    )


def test_query_row_linear_threshold_keeps_truth_out_of_ols_context_fit() -> None:
    result = run_query_row_linear_icl_threshold(
        seed=1730,
        pretrain_rows=8,
        pretrain_worlds=2,
        pretrain_steps=2,
        eval_rows=16,
        eval_worlds=1,
        context_rows=(8,),
        device="cpu",
    )

    # The baseline is defined over context evidence, not TruthSidecar target
    # values.  This is a structural contract check, not a capability claim.
    assert result.context_summaries[0].target_count > 0
    assert result.context_summaries[0].target_count == result.records[0].target_count
    assert result.claim_boundary.startswith("TabUR frozen synthetic ICL")
