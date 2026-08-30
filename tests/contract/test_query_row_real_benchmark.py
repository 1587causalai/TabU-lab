from __future__ import annotations

import math

from tabu_lab.experiments import run_query_row_real_scratch_benchmark


def test_tabur_stage4_is_scratch_only_and_reports_declared_baselines() -> None:
    result = run_query_row_real_scratch_benchmark(
        dataset_ids=("iris", "diabetes"),
        label_budget=32,
        updates=4,
        test_limit=16,
        row_token_count=4,
    )
    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.model_id == "tabu.query.row"
    assert all(record.status == "pass" for record in result.datasets)
    iris = next(record for record in result.datasets if record.dataset_id == "iris")
    diabetes = next(record for record in result.datasets if record.dataset_id == "diabetes")
    assert set(iris.baseline_metrics) == {"majority", "uniform"}
    assert set(diabetes.baseline_metrics) == {"train_mean"}
    assert all(math.isfinite(value) for value in iris.model_metrics.values())
    assert all(math.isfinite(value) for value in diabetes.model_metrics.values())
