"""Small, reproducible R1 real-data execution wedge.

The runner consumes an already prepared evaluator bundle.  It never downloads
data and never passes the retained target column to a model adapter.  A model
checkpoint is optional for the CPU smoke path; absent checkpoints are recorded
as a failed/local-unissued model assessment rather than hidden by tuning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from tabu_lab.adapters.eval_data_workflow import load_prepared_eval_bundle
from tabu_lab.evaluation.foundry import (
    BaselineAdapter,
    BlindExample,
    load_suite,
    score_predictions,
)
from tabu_lab.evidence.schemas import EvidenceSchema


class R1SeedMetrics(EvidenceSchema):
    schema_version: Literal["tabu.r1-seed-metrics.v1"] = "tabu.r1-seed-metrics.v1"
    seed: int = Field(ge=0)
    baseline_id: str
    metrics: dict[str, float]
    coverage: float = Field(ge=0.0, le=1.0)
    counts: dict[str, int]
    predictions: tuple[dict[str, Any], ...] = ()


class R1RunReceipt(EvidenceSchema):
    schema_version: Literal["tabu.r1-run-receipt.v1"] = "tabu.r1-run-receipt.v1"
    experiment_id: Literal["R1-001-tabul-sklearn-diabetes-regression-v1"] = (
        "R1-001-tabul-sklearn-diabetes-regression-v1"
    )
    contract_id: Literal["tabul"] = "tabul"
    contract_version: str
    model_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prepared_bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: tuple[int, ...] = (1729, 2718, 31415)
    baseline_metrics: tuple[R1SeedMetrics, ...] = Field(min_length=1)
    model_metrics: tuple[R1SeedMetrics, ...] = ()
    outcome: Literal["failed", "passed", "blocked"]
    evidence_level: Literal["local_unissued"] = "local_unissued"
    checkpoint_ref: str | None = None
    failure_reason: str | None = None
    claim_boundary: str = (
        "R1 is a bounded Diabetes regression evaluation; it is not benchmark, supported-model, "
        "foundation-model, or causal evidence."
    )


_R1_SCENARIO_ID = "sklearn-diabetes-regression-micro"


def _validate_prepared_r1_bundle(bundle: Any) -> None:
    """Keep the R1 runner bound to the frozen Diabetes preparation contract."""

    prepared = bundle.prepared
    if prepared.scenario_id != _R1_SCENARIO_ID:
        raise ValueError("R1 requires the frozen sklearn Diabetes regression scenario")
    if prepared.binding.dataset_id != "sklearn-diabetes":
        raise ValueError("R1 requires the sklearn-diabetes dataset binding")
    if prepared.binding.partition_counts != {"train": 256, "validation": 64, "test": 122}:
        raise ValueError("R1 requires the frozen 256/64/122 train/validation/test split")
    if prepared.preparation.preprocessing.get("fit_partition") != "train":
        raise ValueError("R1 preprocessing must be fitted on the train partition")
    if not prepared.binding.test_truth_isolated or prepared.binding.adapter_receives_test_truth:
        raise ValueError("R1 prepared data must isolate test truth from the adapter")


def validate_r1_binding(spec: Any) -> None:
    """Validate the frozen R1 vertical-slice identity after generic parsing."""

    if (
        spec.stage.value != "R1"
        or spec.experiment_id != "R1-001-tabul-sklearn-diabetes-regression-v1"
    ):
        raise ValueError("R1 binding is reserved for the TabUL Diabetes regression slice")
    if spec.contract_id != "tabul" or spec.dataset.dataset_id != "sklearn-diabetes":
        raise ValueError("R1-001 is bound to TabUL and sklearn-diabetes")
    if spec.dataset.origin.value != "classic":
        raise ValueError("R1 dataset origin must be classic")
    if (
        spec.training.learning_rate != 1.0e-3
        or spec.training.max_updates > 10_000
        or spec.training.max_epochs is None
        or spec.training.max_epochs > 100
    ):
        raise ValueError("R1 training budget drifted from the frozen limits")
    names = tuple(partition.name for partition in spec.split.partitions)
    if names != ("train", "validation", "test"):
        raise ValueError("R1 requires explicit train/validation/test partitions")


def _metric_record(
    seed: int,
    baseline_id: str,
    scored: Any,
    predictions: Any,
) -> R1SeedMetrics:
    metrics = {key: float(value) for key, value in scored.metrics.items()}
    squared_errors = [
        float(item.metrics["squared_error"])
        for item in scored.per_example
        if item.scored and "squared_error" in item.metrics
    ]
    if squared_errors:
        metrics["rmse"] = math.sqrt(sum(squared_errors) / len(squared_errors))
    if "regression_nrmse" in metrics:
        metrics["nrmse"] = metrics["regression_nrmse"]
    prediction_rows = tuple(
        {
            "abstained": prediction.abstained,
            "example_id": prediction.example_id,
            "failure_category": (
                prediction.failure_category.value
                if prediction.failure_category is not None
                else None
            ),
            "failure_code": prediction.failure_code,
            "probabilities": prediction.probabilities,
            "value": prediction.value,
        }
        for prediction in predictions
    )
    return R1SeedMetrics(
        seed=seed,
        baseline_id=baseline_id,
        metrics=metrics,
        coverage=float(scored.coverage),
        counts={key: int(value) for key, value in scored.counts.items()},
        predictions=prediction_rows,
    )


def run_r1(
    prepared_bundle: str | Path,
    *,
    output: str | Path,
    model_spec_hash: str,
    contract_version: str = "0.1.0",
    checkpoint_ref: str | None = None,
) -> R1RunReceipt:
    """Score the frozen train/test split against the two declared baselines."""

    bundle = load_prepared_eval_bundle(prepared_bundle)
    _validate_prepared_r1_bundle(bundle)
    scenario = next(
        scenario
        for scenario in load_suite("table-supervised-micro-v0").scenarios
        if scenario.scenario_id == _R1_SCENARIO_ID
    )
    prepared = bundle.prepared
    records: list[R1SeedMetrics] = []
    for seed in (1729, 2718, 31415):
        for baseline in scenario.baselines:
            if baseline.family not in {"mean", "standardized_ridge"}:
                continue
            adapter = BaselineAdapter(baseline)
            predictions = adapter.predict(
                scenario=scenario,
                fit_examples=prepared.train,
                examples=tuple(
                    BlindExample(
                        example_id=item.example_id,
                        target_kind=item.target_kind,
                        target_family=item.target_family,
                        features=item.features,
                        context=item.context,
                    )
                    for item in prepared.test
                ),
                seed=seed,
            )
            scored = score_predictions(
                scenario=scenario,
                fit_examples=prepared.train,
                truth=prepared.test,
                predictions=predictions,
            )
            records.append(_metric_record(seed, baseline.baseline_id, scored, predictions))
    reason = (
        "checkpoint scorer is not registered; baseline metrics are retained but no model "
        "prediction is issued"
        if checkpoint_ref
        else "no model checkpoint supplied; baseline-only R1 smoke"
    )
    receipt = R1RunReceipt(
        contract_version=contract_version,
        model_spec_hash=model_spec_hash,
        prepared_bundle_hash=bundle.content_hash,
        split_hash=prepared.binding.split_sha256,
        baseline_metrics=tuple(records),
        outcome="failed",
        checkpoint_ref=checkpoint_ref,
        failure_reason=reason,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return receipt


__all__ = ["R1RunReceipt", "R1SeedMetrics", "run_r1", "validate_r1_binding"]
