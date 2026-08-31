from __future__ import annotations

import math

import pytest

from tabu_lab.experiments import (
    CLASSICAL_ICL_BASELINE_IDS,
    run_query_row_classical_icl_benchmark,
)


def test_query_row_classical_icl_requires_optional_xgboost() -> None:
    pytest.importorskip("xgboost")
    result = run_query_row_classical_icl_benchmark(
        seed=1729,
        pretrain_rows=8,
        pretrain_worlds=2,
        pretrain_steps=2,
        eval_rows=16,
        eval_worlds=1,
        context_rows=(8,),
        device="cpu",
    )

    assert result.baseline_ids == CLASSICAL_ICL_BASELINE_IDS
    assert result.status in {"pass", "continue"}
    assert result.threshold_met is (result.status == "pass")
    assert result.evidence_status == "local_unissued"
    assert result.contract_version == "0.2.0"
    assert result.row_readout_mode == "anchored"
    assert result.row_readout_identity["mode"] == "anchored"
    assert len(result.variant_hash) == 64
    assert result.parameter_hash_unchanged
    assert len(result.records) == 1
    assert result.records[0].target_count > 0
    assert all(
        math.isfinite(value)
        for value in (
            result.tabur_mse,
            result.linear_regression_mse,
            result.mlp_mse,
            result.xgboost_mse,
        )
    )


def test_query_row_classical_icl_declares_no_accepted_claim() -> None:
    pytest.importorskip("xgboost")
    result = run_query_row_classical_icl_benchmark(
        seed=1730,
        pretrain_rows=8,
        pretrain_worlds=2,
        pretrain_steps=1,
        eval_rows=16,
        eval_worlds=1,
        context_rows=(8,),
        device="cpu",
    )

    assert "no real-data transfer" in result.claim_boundary
    assert result.pretraining.evidence_status == "local_unissued"
