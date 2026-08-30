from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments.tabubase_full_context_comparison import (
    BASELINE_SCHEMA,
    COMPARISON_SCHEMA,
    FROZEN_SCHEMA,
    PILOT_COMPARISON_SCHEMA,
    FullContextComparisonValidationError,
    compare_full_context_pilot_receipts,
    compare_full_context_receipts,
)

CHECKPOINT_SEEDS = (101, 202, 303)
SPLIT_SEEDS = (11, 22)
DATASETS = ("classification-fixture", "regression-fixture")
DATASET_HASHES = {
    "classification-fixture": "a" * 64,
    "regression-fixture": "b" * 64,
}


def _manifest(dataset_id: str, split_seed: int) -> dict[str, Any]:
    suffix = f"{len(dataset_id)}{split_seed}"
    return {
        "dataset_id": dataset_id,
        "dataset_sha256": DATASET_HASHES[dataset_id],
        "split_seed": split_seed,
        "train_rows": 12 + split_seed,
        "query_rows": 5 + split_seed,
        "train_indices_sha256": ("1" * 60 + suffix)[-64:],
        "context_order_sha256": ("2" * 60 + suffix)[-64:],
        "query_indices_sha256": ("3" * 60 + suffix)[-64:],
        "feature_indices_sha256": ("4" * 60 + suffix)[-64:],
        "feature_indices": [0, 1, 2],
        "target_scale": 1.0 if dataset_id.startswith("classification") else 2.0,
    }


def _metrics(task: str, value: float) -> dict[str, float]:
    if task == "classification":
        return {
            "accuracy": value,
            "balanced_accuracy": value - 0.01,
            "log_loss": 1.0 - value,
            "macro_f1": value - 0.02,
            "normalized_nll": 1.1 - value,
            "roc_auc_ovr_macro": value + 0.01,
        }
    return {
        "mae": value + 1.0,
        "r2": 0.5 - value / 10.0,
        "rmse": value + 2.0,
        "scaled_mae": value / 2.0,
        "scaled_rmse": value,
    }


def _fixture_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    manifests = {
        dataset_id: {
            str(split_seed): _manifest(dataset_id, split_seed) for split_seed in SPLIT_SEEDS
        }
        for dataset_id in DATASETS
    }
    frozen_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []
    for dataset_id in DATASETS:
        task = "classification" if dataset_id.startswith("classification") else "regression"
        classes = 2 if task == "classification" else None
        for checkpoint_index, checkpoint_seed in enumerate(CHECKPOINT_SEEDS):
            for split_index, split_seed in enumerate(SPLIT_SEEDS):
                manifest = manifests[dataset_id][str(split_seed)]
                for arm_index, arm in enumerate(
                    ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
                ):
                    value = 0.60 + checkpoint_index * 0.10 + split_index * 0.02
                    value += arm_index * 0.01
                    frozen_records.append(
                        {
                            "dataset_id": dataset_id,
                            "dataset_sha256": DATASET_HASHES[dataset_id],
                            "dataset_source": "fixture",
                            "task": task,
                            "classes": classes,
                            "checkpoint_seed": checkpoint_seed,
                            "split_seed": split_seed,
                            "arm": arm,
                            "context_size": manifest["train_rows"],
                            "context_policy": "full_train",
                            "train_rows_total": manifest["train_rows"],
                            "full_context": True,
                            "query_rows": manifest["query_rows"],
                            "predictor_count": 3,
                            "selected_feature_indices": [0, 1, 2],
                            "split_manifest": manifest,
                            "context_class_count": classes or 0,
                            "metrics": _metrics(task, value),
                        }
                    )
        for split_index, split_seed in enumerate(SPLIT_SEEDS):
            manifest = manifests[dataset_id][str(split_seed)]
            for estimator_index, estimator in enumerate(("xgboost", "mlp")):
                value = 0.70 + split_index * 0.04 + estimator_index * 0.02
                baseline_records.append(
                    {
                        "dataset_id": dataset_id,
                        "task": task,
                        "classes": classes,
                        "split_seed": split_seed,
                        "estimator_seed": split_seed,
                        "estimator": estimator,
                        "train_rows": manifest["train_rows"],
                        "query_rows": manifest["query_rows"],
                        "predictor_count": 3,
                        "split_manifest": manifest,
                        "split_manifest_sha256": canonical_hash(manifest),
                        "metrics": _metrics(task, value),
                        "fit": {
                            "estimator": estimator,
                            "estimator_seed": split_seed,
                            "fit_rows": manifest["train_rows"],
                            "query_rows": manifest["query_rows"],
                        },
                    }
                )

    frozen_hashes = {
        str(seed): {
            arm: {
                "before": f"{seed % 10}" * 64,
                "after": f"{seed % 10}" * 64,
                "unchanged": True,
            }
            for arm in ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
        }
        for seed in CHECKPOINT_SEEDS
    }
    frozen = {
        "schema_version": FROZEN_SCHEMA,
        "status": "local_unissued",
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "context_policy": "full_train",
        "context_sizes": None,
        "query_limit": None,
        "query_policy": "all_heldout_rows",
        "query_chunk_semantics": (
            "response_readout_only_after_one_full_transductive_evidence_episode"
        ),
        "query_evidence_policy": "all_heldout_predictors_single_transductive_episode",
        "arms": ["pretrained_frozen", "random_init_frozen", "pretrained_shuffled"],
        "datasets": list(DATASETS),
        "dataset_hashes": DATASET_HASHES,
        "panel_manifest": None,
        "checkpoint_seeds": list(CHECKPOINT_SEEDS),
        "split_seeds": list(SPLIT_SEEDS),
        "split_manifests": manifests,
        "context_rows_by_dataset_split": {
            dataset_id: {
                str(seed): manifests[dataset_id][str(seed)]["train_rows"] for seed in SPLIT_SEEDS
            }
            for dataset_id in DATASETS
        },
        "optimizer_created": False,
        "frozen_arm_optimizer_created": False,
        "all_frozen_arm_parameter_hashes_unchanged": True,
        "all_parameter_hashes_unchanged": True,
        "per_arm_parameter_hashes": frozen_hashes,
        "checkpoints": [
            {"seed": seed, "path": f"checkpoint-{seed}.safetensors", "sha256": "c" * 64}
            for seed in CHECKPOINT_SEEDS
        ],
        "records": frozen_records,
    }
    baseline = {
        "schema_version": BASELINE_SCHEMA,
        "status": "local_unissued",
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "context_policy": "full_train",
        "train_policy": "all_train_partition_rows",
        "query_policy": "all_heldout_rows",
        "datasets": list(DATASETS),
        "dataset_hashes": DATASET_HASHES,
        "panel_manifest": None,
        "split_seeds": list(SPLIT_SEEDS),
        "estimator_seed_policy": "estimator_seed_equals_split_seed",
        "estimators": ["xgboost", "mlp"],
        "split_manifests": manifests,
        "split_manifest_sha256": {
            dataset_id: {
                str(seed): canonical_hash(manifests[dataset_id][str(seed)]) for seed in SPLIT_SEEDS
            }
            for dataset_id in DATASETS
        },
        "records": baseline_records,
    }
    return frozen, baseline


def _write_receipts(
    tmp_path: Path, frozen: dict[str, Any], baseline: dict[str, Any]
) -> tuple[Path, Path]:
    frozen_path = tmp_path / "frozen.json"
    baseline_path = tmp_path / "baseline.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return frozen_path, baseline_path


def _retain_one_checkpoint(frozen: dict[str, Any], checkpoint_seed: int) -> None:
    frozen["checkpoint_seeds"] = [checkpoint_seed]
    frozen["checkpoints"] = [
        row for row in frozen["checkpoints"] if row["seed"] == checkpoint_seed
    ]
    frozen["per_arm_parameter_hashes"] = {
        str(checkpoint_seed): frozen["per_arm_parameter_hashes"][str(checkpoint_seed)]
    }
    frozen["records"] = [
        row for row in frozen["records"] if row["checkpoint_seed"] == checkpoint_seed
    ]


def test_comparison_reports_checkpoint_means_pooled_mean_and_inductive_baselines(
    tmp_path: Path,
) -> None:
    frozen, baseline = _fixture_receipts()
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)
    output_path = tmp_path / "comparison.json"

    result = compare_full_context_receipts(
        frozen_path,
        baseline_path,
        output_path=output_path,
    )

    assert result["schema_version"] == COMPARISON_SCHEMA
    assert result["all_compatibility_and_frozen_gates_passed"] is True
    assert result["compatibility"]["all_heldout_query_rows_identical"] is True
    assert result["compatibility"]["frozen_optimizer_created"] is False
    assert result["compatibility"]["producer_source_identity_equal"] is True
    dataset = result["datasets"]["classification-fixture"]
    assert dataset["common_metrics"] == [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "log_loss",
        "normalized_nll",
        "roc_auc_ovr_macro",
    ]
    assert dataset["tabubase_pretrained"]["by_checkpoint_seed"]["101"]["metrics_mean"][
        "accuracy"
    ] == pytest.approx(0.61)
    assert dataset["tabubase_pretrained"]["by_checkpoint_seed"]["303"]["metrics_mean"][
        "accuracy"
    ] == pytest.approx(0.81)
    assert dataset["tabubase_pretrained"]["descriptive_pooled_mean"]["observation_count"] == 6
    assert dataset["tabubase_pretrained"]["descriptive_pooled_mean"]["metrics_mean"][
        "accuracy"
    ] == pytest.approx(0.71)
    assert dataset["classical_baselines"]["estimators"]["xgboost"]["metrics_mean"][
        "accuracy"
    ] == pytest.approx(0.72)
    assert "transductive" in dataset["tabubase_pretrained"]["evaluation_semantics"]
    assert "inductive" in dataset["classical_baselines"]["evaluation_semantics"]
    assert output_path.is_file()
    assert result["result_path"] == str(output_path)
    assert len(result["result_sha256"]) == 64


def test_single_checkpoint_pilot_preserves_gates_without_claiming_robustness(
    tmp_path: Path,
) -> None:
    frozen, baseline = _fixture_receipts()
    _retain_one_checkpoint(frozen, 202)
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="exactly three"):
        compare_full_context_receipts(frozen_path, baseline_path)

    result = compare_full_context_pilot_receipts(
        frozen_path,
        baseline_path,
        checkpoint_seed=202,
    )

    assert result["schema_version"] == PILOT_COMPARISON_SCHEMA
    assert result["compatibility"]["checkpoint_seeds"] == [202]
    assert result["datasets"]["classification-fixture"]["tabubase_pretrained"][
        "checkpoint_seed_count"
    ] == 1
    assert "single-checkpoint" in result["claim_boundary"]


def test_single_checkpoint_pilot_rejects_wrong_declared_seed(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    _retain_one_checkpoint(frozen, 202)
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="pilot panel"):
        compare_full_context_pilot_receipts(
            frozen_path,
            baseline_path,
            checkpoint_seed=101,
        )


def test_comparison_allows_only_panel_manifest_snapshot_path_drift(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    identity = {
        "file_sha256": "d" * 64,
        "canonical_payload_sha256": "e" * 64,
        "materialization_manifest_sha256": "f" * 64,
    }
    frozen["panel_manifest"] = identity | {"path": "/snapshot-a/panel.yaml"}
    baseline["panel_manifest"] = identity | {"path": "/snapshot-b/panel.yaml"}
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    result = compare_full_context_receipts(frozen_path, baseline_path)

    assert result["compatibility"]["panel_manifest_content_identity_equal"] is True
    assert result["compatibility"]["panel_manifest_paths"] == {
        "frozen": "/snapshot-a/panel.yaml",
        "baselines": "/snapshot-b/panel.yaml",
    }


def test_comparison_rejects_panel_manifest_content_identity_drift(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    frozen["panel_manifest"] = {
        "path": "/snapshot-a/panel.yaml",
        "file_sha256": "d" * 64,
    }
    baseline["panel_manifest"] = {
        "path": "/snapshot-b/panel.yaml",
        "file_sha256": "e" * 64,
    }
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="panel_manifest_identity"):
        compare_full_context_receipts(frozen_path, baseline_path)


def test_comparison_rejects_producer_source_identity_drift(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    baseline["source_tree_sha256"] = "c" * 64
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="producer_source_identity"):
        compare_full_context_receipts(frozen_path, baseline_path)


def test_comparison_rejects_missing_producer_source_identity(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    del frozen["git_commit"]
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="frozen/git_commit"):
        compare_full_context_receipts(frozen_path, baseline_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda frozen, _baseline: frozen.__setitem__("schema_version", "wrong"), "schema"),
        (
            lambda _frozen, baseline: baseline["dataset_hashes"].__setitem__(
                "classification-fixture", "d" * 64
            ),
            "dataset_sha256 mismatch",
        ),
        (
            lambda frozen, _baseline: frozen.__setitem__("optimizer_created", True),
            "optimizer_created",
        ),
        (
            lambda frozen, _baseline: frozen.__setitem__(
                "all_frozen_arm_parameter_hashes_unchanged", False
            ),
            "all_frozen_arm_parameter_hashes_unchanged",
        ),
        (
            lambda frozen, _baseline: frozen["per_arm_parameter_hashes"]["101"][
                "pretrained_frozen"
            ].__setitem__("after", "e" * 64),
            "hash gate failed",
        ),
    ),
)
def test_comparison_rejects_schema_source_and_frozen_gate_drift(
    tmp_path: Path, mutation: Any, message: str
) -> None:
    frozen, baseline = _fixture_receipts()
    mutation(frozen, baseline)
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match=message):
        compare_full_context_receipts(frozen_path, baseline_path)


def test_comparison_rejects_changed_heldout_query_manifest(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    baseline = copy.deepcopy(baseline)
    manifest = baseline["split_manifests"]["classification-fixture"]["11"]
    manifest["query_indices_sha256"] = "f" * 64
    baseline["split_manifest_sha256"]["classification-fixture"]["11"] = canonical_hash(manifest)
    for row in baseline["records"]:
        if row["dataset_id"] == "classification-fixture" and row["split_seed"] == 11:
            row["split_manifest"] = manifest
            row["split_manifest_sha256"] = canonical_hash(manifest)
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="split_manifests"):
        compare_full_context_receipts(frozen_path, baseline_path)


def test_comparison_rejects_missing_full_heldout_record(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    baseline["records"].pop()
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="incomplete"):
        compare_full_context_receipts(frozen_path, baseline_path)


def test_comparison_rejects_non_transductive_frozen_semantics(tmp_path: Path) -> None:
    frozen, baseline = _fixture_receipts()
    frozen["query_evidence_policy"] = "independent_query_chunks"
    frozen_path, baseline_path = _write_receipts(tmp_path, frozen, baseline)

    with pytest.raises(FullContextComparisonValidationError, match="transductive"):
        compare_full_context_receipts(frozen_path, baseline_path)
