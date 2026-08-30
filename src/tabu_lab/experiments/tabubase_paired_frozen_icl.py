"""Fail-closed paired comparison of two hardened frozen-ICL result lines.

This module compares already-produced receipts.  It never constructs a model,
loads a checkpoint, creates an optimizer, or calls the multi-checkpoint
aggregator.  In particular, the old and expanded lines may both use checkpoint
seed 1729: their role in the comparison comes from the explicit input position,
not from requiring distinct seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

K_GRID = (0, 1, 2, 4, 8, 16, 32)
FROZEN_ARMS = ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
SYNTHETIC_SCHEMA = "tabu.transfer-base-frozen-icl-local-unissued.v1"
REAL_SCHEMA = "tabu.transfer-base-real-frozen-icl-local-unissued.v1"
REAL_FULL_CONTEXT_SCHEMA = (
    "tabu.transfer-base-real-full-context-frozen-icl-local-unissued.v1"
)
COMPARISON_SCHEMA = "tabu.tabubase-paired-frozen-icl-comparison.v1"

OLD6_DATASETS = (
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "diabetes",
    "california_housing",
)
NEW6_DATASETS = (
    "banknote_authentication",
    "segment",
    "spambase",
    "airfoil_self_noise",
    "concrete_compressive_strength",
    "qsar_fish_toxicity",
)
OLD6_PANEL_ID = "tabubase-real-frozen-icl-sklearn-old6-v1"
NEW6_PANEL_ID = "tabubase-real-frozen-icl-openml-numeric-nomissing-new6-v1"

ComparisonKind = Literal["auto", "synthetic", "real"]
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ComparisonValidationError(ValueError):
    """Raised when a receipt cannot support a fail-closed paired comparison."""


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ComparisonValidationError(f"result does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonValidationError(f"cannot read result JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ComparisonValidationError(f"result root must be an object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_bool(receipt: Mapping[str, Any], key: str, expected: bool) -> None:
    if receipt.get(key) is not expected:
        raise ComparisonValidationError(f"{key} must be exactly {expected}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ComparisonValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ComparisonValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonValidationError(f"{label} must be finite")
    return result


def _require_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonValidationError(f"{label} must be a non-empty string")
    return value


def _require_single_checkpoint_seed(receipt: Mapping[str, Any], *, real: bool) -> int:
    if real:
        seeds = receipt.get("checkpoint_seeds")
        if not isinstance(seeds, list) or len(seeds) != 1:
            raise ComparisonValidationError(
                "a direct real comparison requires exactly one checkpoint seed per line"
            )
        seed = seeds[0]
    else:
        seed = receipt.get("checkpoint_seed")
        if receipt.get("seed") != seed:
            raise ComparisonValidationError("synthetic seed and checkpoint_seed must agree")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ComparisonValidationError("checkpoint seed must be a non-negative integer")
    return seed


def _validate_frozen_hash_gates(
    receipt: Mapping[str, Any], *, checkpoint_seeds: Sequence[int] | None
) -> None:
    _require_exact_bool(receipt, "frozen_arm_optimizer_created", False)
    if "optimizer_created" in receipt:
        _require_exact_bool(receipt, "optimizer_created", False)
    _require_exact_bool(receipt, "all_frozen_arm_parameter_hashes_unchanged", True)
    hashes = receipt.get("per_arm_parameter_hashes")
    if not isinstance(hashes, dict):
        raise ComparisonValidationError("per_arm_parameter_hashes must be present")
    if checkpoint_seeds is None:
        groups: Mapping[str, Any] = {"synthetic": hashes}
    else:
        expected = {str(seed) for seed in checkpoint_seeds}
        if set(hashes) != expected:
            raise ComparisonValidationError(
                "real per-arm hash groups do not match checkpoint seeds"
            )
        groups = hashes
    for group_name, group in groups.items():
        if not isinstance(group, dict) or set(group) != set(FROZEN_ARMS):
            raise ComparisonValidationError(f"{group_name} must hash every frozen arm exactly once")
        for arm in FROZEN_ARMS:
            arm_hashes = group[arm]
            if not isinstance(arm_hashes, dict):
                raise ComparisonValidationError(f"{group_name}/{arm} hashes must be an object")
            before = _require_sha256(
                arm_hashes.get("before"), label=f"{group_name}/{arm}/before"
            )
            after = _require_sha256(
                arm_hashes.get("after"), label=f"{group_name}/{arm}/after"
            )
            if arm_hashes.get("unchanged") is not True or before != after:
                raise ComparisonValidationError(f"{group_name}/{arm} frozen hash gate failed")


def _require_equal_fields(
    old: Mapping[str, Any], expanded: Mapping[str, Any], fields: Sequence[str]
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for field in fields:
        if field not in old or field not in expanded:
            raise ComparisonValidationError(f"compatibility field is missing: {field}")
        if old[field] != expanded[field]:
            raise ComparisonValidationError(
                f"paired results differ at compatibility field: {field}"
            )
        snapshot[field] = old[field]
    return snapshot


def _stable_seed(seed: int, tag: str) -> int:
    digest = hashlib.sha256(f"{seed}:{tag}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _paired_summary(
    deltas: Sequence[float],
    *,
    replicates: int,
    seed: int,
    tag: str,
    direction: str,
    unit: str,
) -> dict[str, Any]:
    if replicates < 100:
        raise ComparisonValidationError("paired bootstrap requires at least 100 replicates")
    if not deltas or any(not math.isfinite(value) for value in deltas):
        raise ComparisonValidationError("paired deltas must be non-empty and finite")
    values = np.asarray(deltas, dtype=np.float64)
    generator = np.random.default_rng(_stable_seed(seed, tag))
    indices = generator.integers(0, len(values), size=(replicates, len(values)))
    samples = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "paired_wins": int(np.count_nonzero(values > 0.0)),
        "paired_ties": int(np.count_nonzero(values == 0.0)),
        "paired_total": len(values),
        "pairing_unit": unit,
        "direction": direction,
    }


def _synthetic_aulc(values: Sequence[float]) -> float:
    if len(values) != len(K_GRID):
        raise ComparisonValidationError("synthetic AULC requires the complete frozen K grid")
    return float(sum((left + right) * 0.5 for left, right in pairwise(values)) / 6.0)


def _log2_aulc(values: Sequence[float], context_sizes: Sequence[int]) -> float:
    if len(values) != len(context_sizes) or not values:
        raise ComparisonValidationError("real AULC inputs must be non-empty and aligned")
    if any(k <= 0 for k in context_sizes):
        raise ComparisonValidationError("real eligible AULC context sizes must be positive")
    if len(values) == 1:
        return float(values[0])
    x = np.log2(np.asarray(context_sizes, dtype=np.float64))
    return float(np.trapezoid(np.asarray(values, dtype=np.float64), x=x) / (x[-1] - x[0]))


def _validate_synthetic_observations(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    observations = receipt.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(FROZEN_ARMS):
        raise ComparisonValidationError(f"{label} synthetic observations have wrong arms")
    world_count = receipt.get("worlds_per_modality")
    if isinstance(world_count, bool) or not isinstance(world_count, int) or world_count < 1:
        raise ComparisonValidationError(f"{label} worlds_per_modality is invalid")
    expected_k = {str(k) for k in K_GRID}
    for arm in FROZEN_ARMS:
        arm_data = observations[arm]
        if not isinstance(arm_data, dict) or set(arm_data) != {"classification", "regression"}:
            raise ComparisonValidationError(f"{label}/{arm} modalities are incomplete")
        for modality in ("classification", "regression"):
            curves = arm_data[modality]
            if not isinstance(curves, dict) or set(curves) != expected_k:
                raise ComparisonValidationError(f"{label}/{arm}/{modality} K grid is incomplete")
            for k in K_GRID:
                values = curves[str(k)]
                if not isinstance(values, list) or len(values) != world_count:
                    raise ComparisonValidationError(
                        f"{label}/{arm}/{modality}/K={k} world panel is incomplete"
                    )
                for index, value in enumerate(values):
                    _require_finite(value, label=f"{label}/{arm}/{modality}/K={k}/W={index}")
    return observations


def _validate_synthetic(receipt: Mapping[str, Any], *, label: str) -> int:
    if receipt.get("schema_version") != SYNTHETIC_SCHEMA:
        raise ComparisonValidationError(f"{label} is not a hardened synthetic result")
    if receipt.get("status") != "local_unissued":
        raise ComparisonValidationError(f"{label} synthetic status must be local_unissued")
    seed = _require_single_checkpoint_seed(receipt, real=False)
    _validate_frozen_hash_gates(receipt, checkpoint_seeds=None)
    _require_sha256(receipt.get("checkpoint_sha256"), label=f"{label}/checkpoint_sha256")
    _require_string(receipt.get("checkpoint"), label=f"{label}/checkpoint")
    _require_string(receipt.get("git_commit"), label=f"{label}/git_commit")
    if receipt.get("arms_executed") != list(FROZEN_ARMS):
        raise ComparisonValidationError(f"{label} synthetic frozen arms are incomplete")
    if receipt.get("heldout_worlds") != 2 * int(receipt.get("worlds_per_modality", 0)):
        raise ComparisonValidationError(f"{label} synthetic world counts disagree")
    gates = receipt.get("gates")
    if (
        not isinstance(gates, dict)
        or gates.get("all_frozen_parameter_hashes_unchanged") is not True
    ):
        raise ComparisonValidationError(f"{label} synthetic aggregate frozen hash gate failed")
    _validate_synthetic_observations(receipt, label=label)
    return seed


def _compare_synthetic(
    old: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_seed = _validate_synthetic(old, label="old")
    expanded_seed = _validate_synthetic(expanded, label="expanded")
    compatibility = _require_equal_fields(
        old,
        expanded,
        (
            "contract_id",
            "contract_version",
            "profile_id",
            "tokenizer_version",
            "nominal_codebook_size",
            "nominal_codebook_seed",
            "checkpoint_seed",
            "world_seed",
            "world_scope",
            "generator_family",
            "heldout_worlds",
            "worlds_per_modality",
            "context_sizes",
            "query_rows_per_world",
            "arms_executed",
            "k0_semantics",
            "git_commit",
        ),
    )
    if old_seed != expanded_seed:
        raise ComparisonValidationError("old and expanded checkpoint seeds differ")
    if tuple(old["context_sizes"]) != K_GRID:
        raise ComparisonValidationError("synthetic context grid is not the frozen K grid")
    old_observations = old["observations"]["pretrained_frozen"]
    expanded_observations = expanded["observations"]["pretrained_frozen"]
    world_count = int(old["worlds_per_modality"])
    modalities: dict[str, Any] = {}
    for modality in ("classification", "regression"):
        old_curves = [
            [float(old_observations[modality][str(k)][world]) for k in K_GRID]
            for world in range(world_count)
        ]
        expanded_curves = [
            [float(expanded_observations[modality][str(k)][world]) for k in K_GRID]
            for world in range(world_count)
        ]
        aulc_gain = [
            _synthetic_aulc(old_curve) - _synthetic_aulc(expanded_curve)
            for old_curve, expanded_curve in zip(old_curves, expanded_curves, strict=True)
        ]
        k32_gain = [
            old_curve[-1] - expanded_curve[-1]
            for old_curve, expanded_curve in zip(old_curves, expanded_curves, strict=True)
        ]
        metric = "normalized_nll" if modality == "classification" else "scaled_rmse"
        modalities[modality] = {
            "primary_loss_metric": metric,
            "pretrained_expanded_vs_old_aulc_gain": _paired_summary(
                aulc_gain,
                replicates=replicates,
                seed=seed,
                tag=f"synthetic:{modality}:aulc",
                direction="positive means expanded has lower loss than old",
                unit="matched_world_index",
            ),
            "pretrained_expanded_vs_old_k32_gain": _paired_summary(
                k32_gain,
                replicates=replicates,
                seed=seed,
                tag=f"synthetic:{modality}:k32",
                direction="positive means expanded has lower loss than old",
                unit="matched_world_index",
            ),
        }
        if modality == "classification":
            modalities[modality]["accuracy_delta"] = {
                "status": "unavailable_in_synthetic_receipt",
                "direction": "positive would mean expanded has higher accuracy than old",
            }
    return {
        "pairing": {
            "unit": "world_index",
            "world_identity": "world_seed + world_scope + modality + deterministic index",
            "matched_worlds_per_modality": world_count,
        },
        "modalities": modalities,
    }, compatibility


def _panel_id(receipt: Mapping[str, Any]) -> str:
    datasets = tuple(receipt.get("datasets", ()))
    inferred = None
    if datasets == OLD6_DATASETS:
        inferred = OLD6_PANEL_ID
    elif datasets == NEW6_DATASETS:
        inferred = NEW6_PANEL_ID
    declared = receipt.get("panel_id")
    if declared is not None:
        declared = _require_string(declared, label="panel_id")
        if inferred is not None and declared != inferred:
            raise ComparisonValidationError("declared real panel_id conflicts with its dataset set")
        return declared
    if inferred is not None:
        return inferred
    digest = hashlib.sha256(json.dumps(datasets).encode()).hexdigest()[:16]
    return f"custom-real-panel-{digest}"


def _validate_real(
    receipt: Mapping[str, Any], *, label: str
) -> tuple[int, dict[Any, Any], str]:
    schema = receipt.get("schema_version")
    if schema not in {REAL_SCHEMA, REAL_FULL_CONTEXT_SCHEMA}:
        raise ComparisonValidationError(f"{label} is not a hardened real frozen-ICL result")
    if receipt.get("status") != "local_unissued":
        raise ComparisonValidationError(f"{label} real status must be local_unissued")
    seed = _require_single_checkpoint_seed(receipt, real=True)
    _validate_frozen_hash_gates(receipt, checkpoint_seeds=(seed,))
    _require_exact_bool(receipt, "all_parameter_hashes_unchanged", True)
    datasets = receipt.get("datasets")
    split_seeds = receipt.get("split_seeds")
    context_sizes = receipt.get("context_sizes")
    context_policy = receipt.get("context_policy")
    arms = receipt.get("arms")
    if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets):
        raise ComparisonValidationError(f"{label} datasets must be non-empty and unique")
    if not isinstance(split_seeds, list) or not split_seeds or len(set(split_seeds)) != len(
        split_seeds
    ):
        raise ComparisonValidationError(f"{label} split seeds must be non-empty and unique")
    if schema == REAL_SCHEMA:
        if context_policy not in {None, "low_shot_grid"}:
            raise ComparisonValidationError(f"{label} low-shot context policy is invalid")
        context_policy = "low_shot_grid"
        if tuple(context_sizes or ()) != K_GRID:
            raise ComparisonValidationError(f"{label} real context grid is not the frozen K grid")
    else:
        if context_policy != "full_train":
            raise ComparisonValidationError(f"{label} full-context policy is invalid")
        if context_sizes is not None:
            raise ComparisonValidationError(
                f"{label} full-context receipt must use dataset-specific train-row counts"
            )
        if receipt.get("query_limit") is not None or receipt.get("query_policy") != (
            "all_heldout_rows"
        ):
            raise ComparisonValidationError(
                f"{label} full-context receipt must evaluate every held-out row"
            )
        if receipt.get("query_chunk_semantics") != (
            "response_readout_only_after_one_full_transductive_evidence_episode"
        ):
            raise ComparisonValidationError(
                f"{label} full-context receipt has incompatible query chunk semantics"
            )
    if arms != list(FROZEN_ARMS):
        raise ComparisonValidationError(f"{label} real frozen arms are incomplete")
    dataset_hashes = receipt.get("dataset_hashes")
    if not isinstance(dataset_hashes, dict) or set(dataset_hashes) != set(datasets):
        raise ComparisonValidationError(f"{label} dataset hashes do not match the panel")
    for dataset_id, value in dataset_hashes.items():
        _require_sha256(value, label=f"{label}/dataset_hashes/{dataset_id}")
    _require_sha256(receipt.get("source_tree_sha256"), label=f"{label}/source_tree_sha256")
    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise ComparisonValidationError(f"{label} must identify exactly one checkpoint")
    checkpoint = checkpoints[0]
    if not isinstance(checkpoint, dict) or checkpoint.get("seed") != seed:
        raise ComparisonValidationError(f"{label} checkpoint identity has a wrong seed")
    _require_string(checkpoint.get("path"), label=f"{label}/checkpoint/path")
    _require_sha256(checkpoint.get("sha256"), label=f"{label}/checkpoint/sha256")
    records = receipt.get("records")
    if not isinstance(records, list):
        raise ComparisonValidationError(f"{label} records must be a list")
    index: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for record_index, row in enumerate(records):
        if not isinstance(row, dict):
            raise ComparisonValidationError(f"{label} record {record_index} is not an object")
        dataset_id = row.get("dataset_id")
        split_seed = row.get("split_seed")
        arm = row.get("arm")
        context_size = row.get("context_size")
        if dataset_id not in datasets or split_seed not in split_seeds:
            raise ComparisonValidationError(f"{label} record is outside the declared panel")
        if row.get("checkpoint_seed") != seed or arm not in FROZEN_ARMS:
            raise ComparisonValidationError(f"{label} record has a wrong seed or arm")
        if context_policy == "low_shot_grid":
            if context_size not in K_GRID:
                raise ComparisonValidationError(f"{label} record has a wrong context size")
        else:
            train_rows_total = row.get("train_rows_total")
            if (
                isinstance(train_rows_total, bool)
                or not isinstance(train_rows_total, int)
                or train_rows_total < 1
                or context_size != train_rows_total
                or row.get("full_context") is not True
                or row.get("context_policy") != "full_train"
            ):
                raise ComparisonValidationError(
                    f"{label} record does not expose every train row as context"
                )
        if row.get("dataset_sha256") != dataset_hashes[dataset_id]:
            raise ComparisonValidationError(f"{label}/{dataset_id} record source hash mismatch")
        _require_string(row.get("dataset_source"), label=f"{label}/{dataset_id}/dataset_source")
        task = row.get("task")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or task not in {"classification", "regression"}:
            raise ComparisonValidationError(f"{label} record task or metrics are invalid")
        primary = "normalized_nll" if task == "classification" else "scaled_rmse"
        _require_finite(metrics.get(primary), label=f"{label}/record/{primary}")
        if task == "classification":
            accuracy = _require_finite(metrics.get("accuracy"), label=f"{label}/record/accuracy")
            if not 0.0 <= accuracy <= 1.0:
                raise ComparisonValidationError(f"{label} classification accuracy is outside [0,1]")
        elif context_policy == "full_train":
            _require_finite(metrics.get("scaled_mae"), label=f"{label}/record/scaled_mae")
            _require_finite(metrics.get("r2"), label=f"{label}/record/r2")
        key = (str(dataset_id), int(split_seed), str(arm), int(context_size))
        if key in index:
            raise ComparisonValidationError(f"{label} has duplicate real record: {key}")
        index[key] = row
    expected_context_count = len(K_GRID) if context_policy == "low_shot_grid" else 1
    expected = len(datasets) * len(split_seeds) * len(FROZEN_ARMS) * expected_context_count
    if len(index) != expected:
        raise ComparisonValidationError(f"{label} real panel is incomplete")
    return seed, index, str(context_policy)


def _real_dataset_summary(
    dataset_id: str,
    *,
    old_index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    expanded_index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    split_seeds: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    example = old_index[(dataset_id, split_seeds[0], "pretrained_frozen", K_GRID[0])]
    task = str(example["task"])
    classes = example.get("classes")
    if task == "classification":
        if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
            raise ComparisonValidationError(f"{dataset_id} classification classes are invalid")
        eligible = [k for k in K_GRID if k >= classes]
        primary = "normalized_nll"
    else:
        if classes is not None:
            raise ComparisonValidationError(f"{dataset_id} regression classes must be null")
        eligible = [k for k in K_GRID if k > 0]
        primary = "scaled_rmse"
    if not eligible or 32 not in eligible:
        raise ComparisonValidationError(f"{dataset_id} has no eligible K=32 endpoint")
    paired_units: list[dict[str, Any]] = []
    primary_aulc_gains: list[float] = []
    primary_k32_gains: list[float] = []
    accuracy_aulc_deltas: list[float] = []
    accuracy_k32_deltas: list[float] = []
    for split_seed in split_seeds:
        old_rows = [old_index[(dataset_id, split_seed, "pretrained_frozen", k)] for k in eligible]
        expanded_rows = [
            expanded_index[(dataset_id, split_seed, "pretrained_frozen", k)] for k in eligible
        ]
        old_values = [float(row["metrics"][primary]) for row in old_rows]
        expanded_values = [float(row["metrics"][primary]) for row in expanded_rows]
        old_aulc = _log2_aulc(old_values, eligible)
        expanded_aulc = _log2_aulc(expanded_values, eligible)
        old_k32 = float(
            old_index[(dataset_id, split_seed, "pretrained_frozen", 32)]["metrics"][primary]
        )
        expanded_k32 = float(
            expanded_index[(dataset_id, split_seed, "pretrained_frozen", 32)]["metrics"][primary]
        )
        unit: dict[str, Any] = {
            "split_seed": split_seed,
            "old_primary_aulc": old_aulc,
            "expanded_primary_aulc": expanded_aulc,
            "primary_aulc_gain": old_aulc - expanded_aulc,
            "old_primary_k32": old_k32,
            "expanded_primary_k32": expanded_k32,
            "primary_k32_gain": old_k32 - expanded_k32,
        }
        primary_aulc_gains.append(unit["primary_aulc_gain"])
        primary_k32_gains.append(unit["primary_k32_gain"])
        if task == "classification":
            old_accuracy = [float(row["metrics"]["accuracy"]) for row in old_rows]
            expanded_accuracy = [float(row["metrics"]["accuracy"]) for row in expanded_rows]
            old_accuracy_aulc = _log2_aulc(old_accuracy, eligible)
            expanded_accuracy_aulc = _log2_aulc(expanded_accuracy, eligible)
            old_accuracy_k32 = float(
                old_index[(dataset_id, split_seed, "pretrained_frozen", 32)]["metrics"][
                    "accuracy"
                ]
            )
            expanded_accuracy_k32 = float(
                expanded_index[(dataset_id, split_seed, "pretrained_frozen", 32)]["metrics"][
                    "accuracy"
                ]
            )
            unit.update(
                {
                    "old_accuracy_aulc": old_accuracy_aulc,
                    "expanded_accuracy_aulc": expanded_accuracy_aulc,
                    "accuracy_aulc_delta": expanded_accuracy_aulc - old_accuracy_aulc,
                    "old_accuracy_k32": old_accuracy_k32,
                    "expanded_accuracy_k32": expanded_accuracy_k32,
                    "accuracy_k32_delta": expanded_accuracy_k32 - old_accuracy_k32,
                }
            )
            accuracy_aulc_deltas.append(unit["accuracy_aulc_delta"])
            accuracy_k32_deltas.append(unit["accuracy_k32_delta"])
        paired_units.append(unit)
    result: dict[str, Any] = {
        "task": task,
        "classes": classes,
        "primary_loss_metric": primary,
        "eligible_context_sizes": eligible,
        "paired_split_units": paired_units,
        "pretrained_expanded_vs_old_primary_aulc_gain": _paired_summary(
            primary_aulc_gains,
            replicates=replicates,
            seed=seed,
            tag=f"real:{dataset_id}:primary-aulc",
            direction="positive means expanded has lower loss than old",
            unit="split_seed",
        ),
        "pretrained_expanded_vs_old_k32_gain": _paired_summary(
            primary_k32_gains,
            replicates=replicates,
            seed=seed,
            tag=f"real:{dataset_id}:primary-k32",
            direction="positive means expanded has lower loss than old",
            unit="split_seed",
        ),
    }
    if task == "classification":
        result["classification_accuracy"] = {
            "pretrained_expanded_vs_old_accuracy_aulc_delta": _paired_summary(
                accuracy_aulc_deltas,
                replicates=replicates,
                seed=seed,
                tag=f"real:{dataset_id}:accuracy-aulc",
                direction="positive means expanded has higher accuracy than old",
                unit="split_seed",
            ),
            "pretrained_expanded_vs_old_accuracy_k32_delta": _paired_summary(
                accuracy_k32_deltas,
                replicates=replicates,
                seed=seed,
                tag=f"real:{dataset_id}:accuracy-k32",
                direction="positive means expanded has higher accuracy than old",
                unit="split_seed",
            ),
        }
    return result


def _full_context_row(
    index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    *,
    dataset_id: str,
    split_seed: int,
) -> Mapping[str, Any]:
    matches = [
        row
        for (row_dataset, row_split, arm, _context_size), row in index.items()
        if row_dataset == dataset_id
        and row_split == split_seed
        and arm == "pretrained_frozen"
    ]
    if len(matches) != 1:
        raise ComparisonValidationError(
            f"{dataset_id}/{split_seed} must have one full-context pretrained record"
        )
    return matches[0]


def _real_full_context_dataset_summary(
    dataset_id: str,
    *,
    old_index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    expanded_index: Mapping[tuple[str, int, str, int], Mapping[str, Any]],
    split_seeds: Sequence[int],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    paired_units: list[dict[str, Any]] = []
    primary_gains: list[float] = []
    accuracy_deltas: list[float] = []
    r2_deltas: list[float] = []
    scaled_mae_gains: list[float] = []
    task: str | None = None
    classes: Any = None
    primary: str | None = None
    for split_seed in split_seeds:
        old_row = _full_context_row(
            old_index, dataset_id=dataset_id, split_seed=split_seed
        )
        expanded_row = _full_context_row(
            expanded_index, dataset_id=dataset_id, split_seed=split_seed
        )
        task = str(old_row["task"])
        classes = old_row.get("classes")
        primary = "normalized_nll" if task == "classification" else "scaled_rmse"
        old_primary = float(old_row["metrics"][primary])
        expanded_primary = float(expanded_row["metrics"][primary])
        primary_gain = old_primary - expanded_primary
        primary_gains.append(primary_gain)
        unit: dict[str, Any] = {
            "split_seed": split_seed,
            "context_rows": int(old_row["context_size"]),
            "query_rows": int(old_row["query_rows"]),
            "old_primary": old_primary,
            "expanded_primary": expanded_primary,
            "primary_loss_gain": primary_gain,
        }
        if task == "classification":
            old_accuracy = float(old_row["metrics"]["accuracy"])
            expanded_accuracy = float(expanded_row["metrics"]["accuracy"])
            delta = expanded_accuracy - old_accuracy
            accuracy_deltas.append(delta)
            unit.update(
                {
                    "old_accuracy": old_accuracy,
                    "expanded_accuracy": expanded_accuracy,
                    "accuracy_delta": delta,
                }
            )
        else:
            old_r2 = float(old_row["metrics"]["r2"])
            expanded_r2 = float(expanded_row["metrics"]["r2"])
            r2_delta = expanded_r2 - old_r2
            old_scaled_mae = float(old_row["metrics"]["scaled_mae"])
            expanded_scaled_mae = float(expanded_row["metrics"]["scaled_mae"])
            scaled_mae_gain = old_scaled_mae - expanded_scaled_mae
            r2_deltas.append(r2_delta)
            scaled_mae_gains.append(scaled_mae_gain)
            unit.update(
                {
                    "old_r2": old_r2,
                    "expanded_r2": expanded_r2,
                    "r2_delta": r2_delta,
                    "old_scaled_mae": old_scaled_mae,
                    "expanded_scaled_mae": expanded_scaled_mae,
                    "scaled_mae_gain": scaled_mae_gain,
                }
            )
        paired_units.append(unit)
    assert task is not None and primary is not None
    result: dict[str, Any] = {
        "task": task,
        "classes": classes,
        "evaluation_scope": "all_train_partition_rows_as_context",
        "primary_loss_metric": primary,
        "paired_split_units": paired_units,
        "pretrained_expanded_vs_old_primary_loss_gain": _paired_summary(
            primary_gains,
            replicates=replicates,
            seed=seed,
            tag=f"real-full:{dataset_id}:primary-loss",
            direction="positive means expanded has lower full-context loss than old",
            unit="split_seed",
        ),
    }
    if task == "classification":
        result["pretrained_expanded_vs_old_accuracy_delta"] = _paired_summary(
            accuracy_deltas,
            replicates=replicates,
            seed=seed,
            tag=f"real-full:{dataset_id}:accuracy",
            direction="positive means expanded has higher full-context accuracy than old",
            unit="split_seed",
        )
    else:
        result["pretrained_expanded_vs_old_r2_delta"] = _paired_summary(
            r2_deltas,
            replicates=replicates,
            seed=seed,
            tag=f"real-full:{dataset_id}:r2",
            direction="positive means expanded has higher full-context R2 than old",
            unit="split_seed",
        )
        result["pretrained_expanded_vs_old_scaled_mae_gain"] = _paired_summary(
            scaled_mae_gains,
            replicates=replicates,
            seed=seed,
            tag=f"real-full:{dataset_id}:scaled-mae",
            direction="positive means expanded has lower full-context scaled MAE than old",
            unit="split_seed",
        )
    return result


def _real_full_context_macro(
    dataset_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    macro: dict[str, Any] = {}
    for task in ("classification", "regression"):
        rows = [row for row in dataset_results.values() if row["task"] == task]
        if not rows:
            macro[task] = {"dataset_count": 0, "status": "not_run"}
            continue
        primary = [
            row["pretrained_expanded_vs_old_primary_loss_gain"]["mean"] for row in rows
        ]
        task_macro: dict[str, Any] = {
            "dataset_count": len(rows),
            "aggregation_unit": "dataset_equal_weight",
            "mean_primary_loss_gain": float(np.mean(primary)),
            "datasets_with_positive_primary_loss_gain": sum(
                value > 0.0 for value in primary
            ),
        }
        if task == "classification":
            accuracy = [
                row["pretrained_expanded_vs_old_accuracy_delta"]["mean"] for row in rows
            ]
            task_macro.update(
                {
                    "mean_accuracy_delta": float(np.mean(accuracy)),
                    "datasets_with_positive_accuracy_delta": sum(
                        value > 0.0 for value in accuracy
                    ),
                }
            )
        else:
            r2 = [row["pretrained_expanded_vs_old_r2_delta"]["mean"] for row in rows]
            scaled_mae = [
                row["pretrained_expanded_vs_old_scaled_mae_gain"]["mean"]
                for row in rows
            ]
            task_macro.update(
                {
                    "mean_r2_delta": float(np.mean(r2)),
                    "datasets_with_positive_r2_delta": sum(value > 0.0 for value in r2),
                    "mean_scaled_mae_gain": float(np.mean(scaled_mae)),
                    "datasets_with_positive_scaled_mae_gain": sum(
                        value > 0.0 for value in scaled_mae
                    ),
                }
            )
        macro[task] = task_macro
    return macro


def _real_macro(dataset_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    macro: dict[str, Any] = {}
    for task in ("classification", "regression"):
        rows = [row for row in dataset_results.values() if row["task"] == task]
        if not rows:
            macro[task] = {"dataset_count": 0, "status": "not_run"}
            continue
        aulc = [row["pretrained_expanded_vs_old_primary_aulc_gain"]["mean"] for row in rows]
        k32 = [row["pretrained_expanded_vs_old_k32_gain"]["mean"] for row in rows]
        task_macro: dict[str, Any] = {
            "dataset_count": len(rows),
            "aggregation_unit": "dataset_equal_weight",
            "mean_primary_aulc_gain": float(np.mean(aulc)),
            "datasets_with_positive_primary_aulc_gain": sum(value > 0.0 for value in aulc),
            "mean_primary_k32_gain": float(np.mean(k32)),
            "datasets_with_positive_primary_k32_gain": sum(value > 0.0 for value in k32),
        }
        if task == "classification":
            accuracy_aulc = [
                row["classification_accuracy"][
                    "pretrained_expanded_vs_old_accuracy_aulc_delta"
                ]["mean"]
                for row in rows
            ]
            accuracy_k32 = [
                row["classification_accuracy"][
                    "pretrained_expanded_vs_old_accuracy_k32_delta"
                ]["mean"]
                for row in rows
            ]
            task_macro.update(
                {
                    "mean_accuracy_aulc_delta": float(np.mean(accuracy_aulc)),
                    "datasets_with_positive_accuracy_aulc_delta": sum(
                        value > 0.0 for value in accuracy_aulc
                    ),
                    "mean_accuracy_k32_delta": float(np.mean(accuracy_k32)),
                    "datasets_with_positive_accuracy_k32_delta": sum(
                        value > 0.0 for value in accuracy_k32
                    ),
                }
            )
        macro[task] = task_macro
    return macro


def _compare_real(
    old: Mapping[str, Any],
    expanded: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    old_seed, old_index, old_context_policy = _validate_real(old, label="old")
    expanded_seed, expanded_index, expanded_context_policy = _validate_real(
        expanded, label="expanded"
    )
    compatibility = _require_equal_fields(
        old,
        expanded,
        (
            "contract_id",
            "contract_version",
            "profile_id",
            "tokenizer_version",
            "nominal_codebook_size",
            "nominal_codebook_seed",
            "datasets",
            "dataset_hashes",
            "checkpoint_seeds",
            "split_seeds",
            "context_sizes",
            "query_limit",
            "query_chunk_rows",
            "arms",
            "source_tree_sha256",
        ),
    )
    if old_seed != expanded_seed:
        raise ComparisonValidationError("old and expanded checkpoint seeds differ")
    if old_context_policy != expanded_context_policy:
        raise ComparisonValidationError("old and expanded real context policies differ")
    compatibility["context_policy"] = old_context_policy
    if old_context_policy == "full_train":
        compatibility.update(
            _require_equal_fields(
                old,
                expanded,
                (
                    "context_policy",
                    "context_rows_by_dataset_split",
                    "query_policy",
                    "query_chunk_semantics",
                ),
            )
        )
    old_panel = _panel_id(old)
    expanded_panel = _panel_id(expanded)
    if old_panel != expanded_panel:
        raise ComparisonValidationError("old and expanded real panel IDs differ")
    for optional_source_field in (
        "panel_manifest_sha256",
        "dataset_source_manifests",
        "dataset_source_manifest_sha256s",
    ):
        if optional_source_field in old or optional_source_field in expanded:
            compatibility.update(
                _require_equal_fields(old, expanded, (optional_source_field,))
            )
    record_source_fields: tuple[str, ...] = (
        "dataset_sha256",
        "dataset_source",
        "task",
        "classes",
        "split_seed",
        "context_size",
        "query_rows",
        "predictor_count",
        "selected_feature_indices",
        "context_class_count",
    )
    if old_context_policy == "full_train":
        record_source_fields += (
            "context_policy",
            "train_rows_total",
            "full_context",
        )
    if set(old_index) != set(expanded_index):
        raise ComparisonValidationError("old and expanded real record panels differ")
    for key in old_index:
        _require_equal_fields(old_index[key], expanded_index[key], record_source_fields)
    split_seeds = [int(value) for value in old["split_seeds"]]
    if old_context_policy == "full_train":
        dataset_results = {
            dataset_id: _real_full_context_dataset_summary(
                dataset_id,
                old_index=old_index,
                expanded_index=expanded_index,
                split_seeds=split_seeds,
                replicates=replicates,
                seed=seed,
            )
            for dataset_id in old["datasets"]
        }
        macro = _real_full_context_macro(dataset_results)
    else:
        dataset_results = {
            dataset_id: _real_dataset_summary(
                dataset_id,
                old_index=old_index,
                expanded_index=expanded_index,
                split_seeds=split_seeds,
                replicates=replicates,
                seed=seed,
            )
            for dataset_id in old["datasets"]
        }
        macro = _real_macro(dataset_results)
    compatibility["panel_id"] = old_panel
    return {
        "panel_id": old_panel,
        "context_policy": old_context_policy,
        "pairing": {
            "unit": "dataset_id + split_seed",
            "checkpoint_seed": old_seed,
            "matched_split_count_per_dataset": len(split_seeds),
        },
        "datasets": dataset_results,
        "macro_by_panel": {
            old_panel: {
                "rule": "this macro contains one real panel only; old6 and new6 are never mixed",
                **macro,
            }
        },
    }, compatibility


def compare_paired_frozen_icl_results(
    old_result_path: Path,
    expanded_result_path: Path,
    *,
    kind: ComparisonKind = "auto",
    output_path: Path | None = None,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Compare one old and one expanded hardened receipt on an identical panel."""

    if old_result_path.resolve() == expanded_result_path.resolve():
        raise ComparisonValidationError("old and expanded results must be distinct files")
    old = _load_json_object(old_result_path)
    expanded = _load_json_object(expanded_result_path)
    if kind == "auto":
        schemas = {old.get("schema_version"), expanded.get("schema_version")}
        if schemas == {SYNTHETIC_SCHEMA}:
            resolved_kind: Literal["synthetic", "real"] = "synthetic"
        elif schemas in ({REAL_SCHEMA}, {REAL_FULL_CONTEXT_SCHEMA}):
            resolved_kind = "real"
        else:
            raise ComparisonValidationError("cannot infer one compatible result kind")
    elif kind in {"synthetic", "real"}:
        resolved_kind = kind
    else:
        raise ComparisonValidationError(f"unsupported comparison kind: {kind}")
    if resolved_kind == "synthetic":
        comparison, compatibility = _compare_synthetic(
            old,
            expanded,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    else:
        comparison, compatibility = _compare_real(
            old,
            expanded,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    if resolved_kind == "synthetic":
        old_checkpoint = {
            "seed": old["checkpoint_seed"],
            "path": old["checkpoint"],
            "sha256": old["checkpoint_sha256"],
        }
        expanded_checkpoint = {
            "seed": expanded["checkpoint_seed"],
            "path": expanded["checkpoint"],
            "sha256": expanded["checkpoint_sha256"],
        }
    else:
        old_checkpoint = old["checkpoints"][0]
        expanded_checkpoint = expanded["checkpoints"][0]
    receipt: dict[str, Any] = {
        "schema_version": COMPARISON_SCHEMA,
        "status": "local_unissued",
        "comparison_kind": resolved_kind,
        "line_roles": {
            "old": {
                "result_path": str(old_result_path),
                "result_sha256": _sha256_file(old_result_path),
                "checkpoint": old_checkpoint,
            },
            "expanded": {
                "result_path": str(expanded_result_path),
                "result_sha256": _sha256_file(expanded_result_path),
                "checkpoint": expanded_checkpoint,
            },
        },
        "checkpoint_seed_policy": (
            "line roles are positional; equal checkpoint seeds are required and do not invoke "
            "the distinct-seed aggregator"
        ),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "interval": "paired percentile 95%",
        },
        "compatibility": compatibility,
        "all_compatibility_and_frozen_gates_passed": True,
        "comparison": comparison,
        "claim_boundary": (
            "direct paired local-unissued comparison only; positive loss gain means old loss "
            "minus expanded loss, while positive classification accuracy delta means expanded "
            "accuracy minus old accuracy; real full-context receipts expose every train row, "
            "while historical K<=32 receipts remain low-shot diagnostics; no fine-tuning, "
            "formal receipt, broad benchmark, foundation-model, or causal claim"
        ),
    }
    if output_path is None:
        return receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt | {
        "result_path": str(output_path),
        "result_sha256": _sha256_file(output_path),
    }


__all__ = [
    "COMPARISON_SCHEMA",
    "NEW6_PANEL_ID",
    "OLD6_PANEL_ID",
    "REAL_FULL_CONTEXT_SCHEMA",
    "ComparisonValidationError",
    "compare_paired_frozen_icl_results",
]
