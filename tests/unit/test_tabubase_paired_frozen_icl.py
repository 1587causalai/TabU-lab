from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from tabu_lab.experiments.tabubase_paired_frozen_icl import (
    NEW6_PANEL_ID,
    OLD6_DATASETS,
    OLD6_PANEL_ID,
    REAL_FULL_CONTEXT_SCHEMA,
    ComparisonValidationError,
    compare_paired_frozen_icl_results,
)

K_GRID = (0, 1, 2, 4, 8, 16, 32)
ARMS = ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")


def _arm_hashes(character: str) -> dict[str, dict[str, str | bool]]:
    value = character * 64
    return {
        arm: {"before": value, "after": value, "unchanged": True}
        for arm in ARMS
    }


def _synthetic_receipt(*, expanded: bool) -> dict[str, Any]:
    offset = -0.2 if expanded else 0.0
    observations = {
        arm: {
            modality: {
                str(k): [
                    1.0 + 0.01 * k + 0.001 * world + offset
                    for world in range(4)
                ]
                for k in K_GRID
            }
            for modality in ("classification", "regression")
        }
        for arm in ARMS
    }
    return {
        "schema_version": "tabu.transfer-base-frozen-icl-local-unissued.v1",
        "status": "local_unissued",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
        "tokenizer_version": "cell-tokenizer.v2",
        "nominal_codebook_size": 100,
        "nominal_codebook_seed": 1729,
        "seed": 1729,
        "checkpoint_seed": 1729,
        "checkpoint": "/mock/expanded.safetensors" if expanded else "/mock/old.safetensors",
        "checkpoint_sha256": ("b" if expanded else "a") * 64,
        "world_seed": 1729,
        "world_scope": "heldout",
        "generator_family": "heteroscedastic_missingness_shift",
        "heldout_worlds": 8,
        "worlds_per_modality": 4,
        "context_sizes": list(K_GRID),
        "query_rows_per_world": 32,
        "arms_executed": list(ARMS),
        "k0_semantics": {
            "classification": "uniform_over_declared_classes",
            "regression": "zero_in_generator_standardized_space",
        },
        "git_commit": "3502fdd80539f2a8b9703cc4e4546fd01f3826ce",
        "frozen_arm_optimizer_created": False,
        "per_arm_parameter_hashes": _arm_hashes("b" if expanded else "a"),
        "all_frozen_arm_parameter_hashes_unchanged": True,
        "gates": {"all_frozen_parameter_hashes_unchanged": True},
        "observations": observations,
    }


_TASKS = {
    "iris": ("classification", 3),
    "wine": ("classification", 3),
    "breast_cancer": ("classification", 2),
    "digits": ("classification", 10),
    "diabetes": ("regression", None),
    "california_housing": ("regression", None),
}


def _real_receipt(*, expanded: bool) -> dict[str, Any]:
    split_seeds = (1729, 2718)
    hashes = {
        dataset_id: f"{index + 1:064x}"
        for index, dataset_id in enumerate(OLD6_DATASETS)
    }
    records: list[dict[str, Any]] = []
    for dataset_id in OLD6_DATASETS:
        task, classes = _TASKS[dataset_id]
        for split_seed in split_seeds:
            for arm in ARMS:
                for k in K_GRID:
                    primary = 1.0 + 0.01 * k + 0.001 * (split_seed % 10)
                    accuracy = 0.4 + 0.005 * k
                    if expanded and arm == "pretrained_frozen":
                        primary -= 0.1
                        accuracy += 0.05
                    metrics = (
                        {"normalized_nll": primary, "accuracy": accuracy}
                        if task == "classification"
                        else {"scaled_rmse": primary, "r2": 0.0}
                    )
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "dataset_sha256": hashes[dataset_id],
                            "dataset_source": f"mock://{dataset_id}",
                            "task": task,
                            "classes": classes,
                            "checkpoint_seed": 1729,
                            "split_seed": split_seed,
                            "arm": arm,
                            "context_size": k,
                            "query_rows": 64,
                            "predictor_count": 4,
                            "selected_feature_indices": [0, 1, 2, 3],
                            "context_class_count": min(k, classes or 0),
                            "metrics": metrics,
                        }
                    )
    return {
        "schema_version": "tabu.transfer-base-real-frozen-icl-local-unissued.v1",
        "status": "local_unissued",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
        "tokenizer_version": "cell-tokenizer.v2",
        "nominal_codebook_size": 100,
        "nominal_codebook_seed": 1729,
        "datasets": list(OLD6_DATASETS),
        "dataset_hashes": hashes,
        "checkpoint_seeds": [1729],
        "split_seeds": list(split_seeds),
        "context_sizes": list(K_GRID),
        "query_limit": 256,
        "query_chunk_rows": 64,
        "arms": list(ARMS),
        "optimizer_created": False,
        "frozen_arm_optimizer_created": False,
        "per_arm_parameter_hashes": {
            "1729": _arm_hashes("b" if expanded else "a")
        },
        "all_frozen_arm_parameter_hashes_unchanged": True,
        "all_parameter_hashes_unchanged": True,
        "checkpoints": [
            {
                "seed": 1729,
                "path": "/mock/expanded.safetensors" if expanded else "/mock/old.safetensors",
                "sha256": ("b" if expanded else "a") * 64,
            }
        ],
        "source_tree_sha256": "c" * 64,
        "records": records,
    }


def _real_full_context_receipt(*, expanded: bool) -> dict[str, Any]:
    split_seeds = (1729, 2718)
    hashes = {
        dataset_id: f"{index + 1:064x}"
        for index, dataset_id in enumerate(OLD6_DATASETS)
    }
    context_rows_by_dataset_split: dict[str, dict[str, int]] = {}
    records: list[dict[str, Any]] = []
    for dataset_index, dataset_id in enumerate(OLD6_DATASETS):
        task, classes = _TASKS[dataset_id]
        context_rows_by_dataset_split[dataset_id] = {}
        for split_seed in split_seeds:
            context_rows = 100 + dataset_index + (split_seed % 10)
            context_rows_by_dataset_split[dataset_id][str(split_seed)] = context_rows
            for arm in ARMS:
                primary = 1.0 + 0.001 * (split_seed % 10)
                accuracy = 0.4
                scaled_mae = 0.8
                r2 = -0.1
                if expanded and arm == "pretrained_frozen":
                    primary -= 0.1
                    accuracy += 0.05
                    scaled_mae -= 0.08
                    r2 += 0.2
                metrics = (
                    {"normalized_nll": primary, "accuracy": accuracy}
                    if task == "classification"
                    else {
                        "scaled_rmse": primary,
                        "scaled_mae": scaled_mae,
                        "r2": r2,
                    }
                )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset_sha256": hashes[dataset_id],
                        "dataset_source": f"mock://{dataset_id}",
                        "task": task,
                        "classes": classes,
                        "checkpoint_seed": 1729,
                        "split_seed": split_seed,
                        "arm": arm,
                        "context_size": context_rows,
                        "context_policy": "full_train",
                        "train_rows_total": context_rows,
                        "full_context": True,
                        "query_rows": 54,
                        "predictor_count": 4,
                        "selected_feature_indices": [0, 1, 2, 3],
                        "context_class_count": classes or 0,
                        "metrics": metrics,
                    }
                )
    return {
        "schema_version": REAL_FULL_CONTEXT_SCHEMA,
        "status": "local_unissued",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
        "tokenizer_version": "cell-tokenizer.v2",
        "nominal_codebook_size": 100,
        "nominal_codebook_seed": 1729,
        "datasets": list(OLD6_DATASETS),
        "dataset_hashes": hashes,
        "checkpoint_seeds": [1729],
        "split_seeds": list(split_seeds),
        "context_policy": "full_train",
        "context_sizes": None,
        "context_rows_by_dataset_split": context_rows_by_dataset_split,
        "query_limit": None,
        "query_policy": "all_heldout_rows",
        "query_chunk_rows": 64,
        "query_chunk_semantics": (
            "response_readout_only_after_one_full_transductive_evidence_episode"
        ),
        "arms": list(ARMS),
        "optimizer_created": False,
        "frozen_arm_optimizer_created": False,
        "per_arm_parameter_hashes": {
            "1729": _arm_hashes("b" if expanded else "a")
        },
        "all_frozen_arm_parameter_hashes_unchanged": True,
        "all_parameter_hashes_unchanged": True,
        "checkpoints": [
            {
                "seed": 1729,
                "path": "/mock/expanded.safetensors" if expanded else "/mock/old.safetensors",
                "sha256": ("b" if expanded else "a") * 64,
            }
        ],
        "source_tree_sha256": "c" * 64,
        "records": records,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_synthetic_direct_pair_accepts_equal_checkpoint_seed_and_bootstraps_worlds(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old.json"
    expanded_path = tmp_path / "expanded.json"
    output_path = tmp_path / "comparison.json"
    _write(old_path, _synthetic_receipt(expanded=False))
    _write(expanded_path, _synthetic_receipt(expanded=True))

    result = compare_paired_frozen_icl_results(
        old_path,
        expanded_path,
        output_path=output_path,
        bootstrap_replicates=100,
    )

    classification = result["comparison"]["modalities"]["classification"]
    aulc = classification["pretrained_expanded_vs_old_aulc_gain"]
    assert result["comparison_kind"] == "synthetic"
    assert result["compatibility"]["checkpoint_seed"] == 1729
    assert aulc["mean"] == pytest.approx(0.2)
    assert aulc["paired_wins"] == 4
    assert aulc["pairing_unit"] == "matched_world_index"
    assert classification["accuracy_delta"]["status"] == "unavailable_in_synthetic_receipt"
    assert output_path.is_file()


def test_synthetic_comparison_rejects_optimizer_or_failed_per_arm_hash(tmp_path: Path) -> None:
    old = _synthetic_receipt(expanded=False)
    expanded = _synthetic_receipt(expanded=True)
    expanded["frozen_arm_optimizer_created"] = True
    old_path = tmp_path / "old.json"
    expanded_path = tmp_path / "expanded.json"
    _write(old_path, old)
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="optimizer"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)

    expanded = _synthetic_receipt(expanded=True)
    expanded["per_arm_parameter_hashes"]["pretrained_frozen"]["after"] = "c" * 64
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="hash gate failed"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)


def test_synthetic_comparison_rejects_world_panel_drift(tmp_path: Path) -> None:
    old = _synthetic_receipt(expanded=False)
    expanded = _synthetic_receipt(expanded=True)
    expanded["world_seed"] = 2718
    old_path = tmp_path / "old.json"
    expanded_path = tmp_path / "expanded.json"
    _write(old_path, old)
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="world_seed"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)


def test_real_old6_pair_reports_loss_gains_accuracy_deltas_and_one_panel_macro(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old-real.json"
    expanded_path = tmp_path / "expanded-real.json"
    _write(old_path, _real_receipt(expanded=False))
    _write(expanded_path, _real_receipt(expanded=True))

    result = compare_paired_frozen_icl_results(
        old_path,
        expanded_path,
        bootstrap_replicates=100,
    )

    comparison = result["comparison"]
    iris = comparison["datasets"]["iris"]
    diabetes = comparison["datasets"]["diabetes"]
    assert comparison["panel_id"] == OLD6_PANEL_ID
    assert set(comparison["macro_by_panel"]) == {OLD6_PANEL_ID}
    assert NEW6_PANEL_ID not in comparison["macro_by_panel"]
    assert iris["eligible_context_sizes"] == [4, 8, 16, 32]
    assert iris["pretrained_expanded_vs_old_primary_aulc_gain"]["mean"] == pytest.approx(
        0.1
    )
    assert iris["classification_accuracy"][
        "pretrained_expanded_vs_old_accuracy_k32_delta"
    ]["mean"] == pytest.approx(0.05)
    assert "classification_accuracy" not in diabetes
    assert diabetes["eligible_context_sizes"] == [1, 2, 4, 8, 16, 32]


def test_real_comparison_rejects_source_or_record_panel_drift(tmp_path: Path) -> None:
    old = _real_receipt(expanded=False)
    expanded = _real_receipt(expanded=True)
    expanded["source_tree_sha256"] = "d" * 64
    old_path = tmp_path / "old.json"
    expanded_path = tmp_path / "expanded.json"
    _write(old_path, old)
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="source_tree_sha256"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)

    expanded = _real_receipt(expanded=True)
    expanded["records"] = expanded["records"][:-1]
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="incomplete"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)


def test_real_comparison_requires_one_equal_seed_per_line(tmp_path: Path) -> None:
    old = _real_receipt(expanded=False)
    expanded = _real_receipt(expanded=True)
    expanded["checkpoint_seeds"] = [2718]
    expanded["checkpoints"][0]["seed"] = 2718
    for row in expanded["records"]:
        row["checkpoint_seed"] = 2718
    expanded["per_arm_parameter_hashes"] = {
        "2718": expanded["per_arm_parameter_hashes"].pop("1729")
    }
    old_path = tmp_path / "old.json"
    expanded_path = tmp_path / "expanded.json"
    _write(old_path, old)
    _write(expanded_path, expanded)
    with pytest.raises(ComparisonValidationError, match="checkpoint_seeds"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)

    multiple = copy.deepcopy(old)
    multiple["checkpoint_seeds"] = [1729, 2718]
    _write(old_path, multiple)
    with pytest.raises(ComparisonValidationError, match="exactly one checkpoint seed"):
        compare_paired_frozen_icl_results(old_path, expanded_path, bootstrap_replicates=100)


def test_real_full_context_pair_reports_accuracy_r2_and_mae_without_aulc(
    tmp_path: Path,
) -> None:
    old_path = tmp_path / "old-real-full.json"
    expanded_path = tmp_path / "expanded-real-full.json"
    _write(old_path, _real_full_context_receipt(expanded=False))
    _write(expanded_path, _real_full_context_receipt(expanded=True))

    result = compare_paired_frozen_icl_results(
        old_path,
        expanded_path,
        bootstrap_replicates=100,
    )

    comparison = result["comparison"]
    iris = comparison["datasets"]["iris"]
    diabetes = comparison["datasets"]["diabetes"]
    assert comparison["context_policy"] == "full_train"
    assert "eligible_context_sizes" not in iris
    assert "pretrained_expanded_vs_old_primary_aulc_gain" not in iris
    assert iris["pretrained_expanded_vs_old_primary_loss_gain"]["mean"] == pytest.approx(
        0.1
    )
    assert iris["pretrained_expanded_vs_old_accuracy_delta"]["mean"] == pytest.approx(
        0.05
    )
    assert diabetes["pretrained_expanded_vs_old_r2_delta"]["mean"] == pytest.approx(0.2)
    assert diabetes["pretrained_expanded_vs_old_scaled_mae_gain"]["mean"] == pytest.approx(
        0.08
    )


def test_real_full_context_comparison_rejects_truncated_context(tmp_path: Path) -> None:
    old = _real_full_context_receipt(expanded=False)
    expanded = _real_full_context_receipt(expanded=True)
    expanded["records"][0]["context_size"] -= 1
    old_path = tmp_path / "old-real-full.json"
    expanded_path = tmp_path / "expanded-real-full.json"
    _write(old_path, old)
    _write(expanded_path, expanded)

    with pytest.raises(ComparisonValidationError, match="every train row"):
        compare_paired_frozen_icl_results(
            old_path,
            expanded_path,
            bootstrap_replicates=100,
        )
