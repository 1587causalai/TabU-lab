from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tabu_lab.evaluation.foundry import (
    ComparisonReport,
    DatasetSnapshotBinding,
    EvalProducerBinding,
    EvalResult,
    EvalSuiteSpec,
    ProducerProvenance,
    list_suite_ids,
    load_suite,
    validate_suite,
)


def test_local_unissued_receipted_model_producer_is_valid_but_non_public() -> None:
    producer = EvalProducerBinding(
        provenance="receipted_run",
        run_id="local-unissued-model-run",
        receipt_sha256="d" * 64,
        receipt_pointer="local-unissued/runs/local-unissued-model-run/receipt.json",
        publication_eligible=False,
    )

    assert producer.provenance is ProducerProvenance.RECEIPTED_RUN
    assert producer.publication_eligible is False

ROOT = Path(__file__).resolve().parents[2]


def test_all_v0_and_tabubase_v1_suites_load_without_fetching_data() -> None:
    expected_v0 = {
        "table-supervised-micro-v0",
        "table-completion-micro-v0",
        "graph-completion-micro-v0",
        "recsys-completion-micro-v0",
    }
    expected_v1 = {
        "table-supervised-micro-v1",
        "table-completion-micro-v1",
    }

    assert set(list_suite_ids()) == expected_v0 | expected_v1
    for suite_id in sorted(expected_v0 | expected_v1):
        suite = load_suite(suite_id)
        report = validate_suite(suite)

        assert report.valid, report.issues
        assert suite.schema_version == "tabu.eval-suite.v1"
        assert suite.budget.model_seeds == (1729, 2718, 31415)
        assert suite.composite_score is False
        assert suite.test_sweeps_allowed is False
        assert suite.variable_under_test == "model_contract_or_artifact"
        if suite_id in expected_v0:
            assert all("tabu4do" not in item.applicable_contracts for item in suite.scenarios)
        else:
            assert all(item.applicable_contracts == ("tabu.cell.base",) for item in suite.scenarios)


def test_suite_protocols_freeze_requested_task_specific_boundaries() -> None:
    supervised = load_suite("table-supervised-micro-v0")
    completion = load_suite("table-completion-micro-v0")
    graph = load_suite("graph-completion-micro-v0")
    recsys = load_suite("recsys-completion-micro-v0")

    assert {scenario.dataset.dataset_id for scenario in supervised.scenarios} == {
        "openml-adult-v2-task-7592",
        "sklearn-diabetes",
    }
    assert all(
        scenario.mask and scenario.mask.fraction == 0.15 for scenario in completion.scenarios
    )
    assert graph.scenarios[0].topology_contract_checks == (
        "topology_perturbation_pass",
        "locality_contract_pass",
    )
    assert recsys.scenarios[0].selection.users == 64
    assert recsys.scenarios[0].selection.items == 128
    assert not any("rank" in metric.metric_id.lower() for metric in recsys.scenarios[0].metrics)


def test_suite_validation_fails_closed_on_leakage_and_composite_score() -> None:
    suite = load_suite("table-completion-micro-v0")
    payload = suite.model_dump(mode="python")
    payload["scenarios"][0]["mask"]["applied_after_split"] = False
    with pytest.raises(ValidationError, match="True"):
        EvalSuiteSpec.model_validate(payload)

    payload = suite.model_dump(mode="python")
    payload["composite_score"] = True
    with pytest.raises(ValidationError, match="False"):
        EvalSuiteSpec.model_validate(payload)

    with pytest.raises(ValidationError, match="train"):
        DatasetSnapshotBinding(
            dataset_id="dataset",
            source_sha256="a" * 64,
            split_sha256="b" * 64,
            recipe_sha256="c" * 64,
            partition_counts={"train": 1, "validation": 1, "test": 1},
            preprocessing_fit_partition="validation",
        )


def test_checked_in_eval_schemas_match_runtime_generation() -> None:
    schema_types = {
        "eval-suite": EvalSuiteSpec,
        "eval-result": EvalResult,
        "eval-comparison": ComparisonReport,
    }
    for name, schema_type in schema_types.items():
        checked_in = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())
        generated = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://research.wehub.us/schemas/{name}.schema.json",
            **schema_type.model_json_schema(),
        }

        assert checked_in == generated
