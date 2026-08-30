"""Fail-closed descriptive summary for full-context TabUBase and classical baselines.

The two input receipts have deliberately different evaluation semantics.  The
frozen TabUBase line evaluates all held-out predictor rows in one transductive
evidence episode, while MLP and XGBoost are fitted inductively on the train
partition for each split.  This module verifies that both lines use exactly the
same data and held-out rows, then reports their metrics side by side.  It does
not turn those semantically different lines into a paired superiority test.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tabu_lab.contracts import canonical_hash

FROZEN_SCHEMA = "tabu.transfer-base-real-full-context-frozen-icl-local-unissued.v1"
BASELINE_SCHEMA = "tabu.transfer-base-real-full-context-baselines-local-unissued.v1"
COMPARISON_SCHEMA = "tabu.tabubase-full-context-descriptive-comparison.v1"
PILOT_COMPARISON_SCHEMA = (
    "tabu.tabubase-full-context-single-checkpoint-pilot-comparison.v1"
)

FROZEN_ARMS = ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
BASELINE_ESTIMATORS = ("xgboost", "mlp")
CLASSIFICATION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "log_loss",
    "normalized_nll",
    "roc_auc_ovr_macro",
)
REGRESSION_METRICS = ("rmse", "mae", "scaled_rmse", "scaled_mae", "r2")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}")


class FullContextComparisonValidationError(ValueError):
    """Raised when two receipts cannot support the descriptive comparison."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FullContextComparisonValidationError(f"{label} receipt does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FullContextComparisonValidationError(
            f"cannot read {label} receipt JSON: {path}"
        ) from exc
    if not isinstance(payload, dict):
        raise FullContextComparisonValidationError(f"{label} receipt root must be an object")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_bool(
    receipt: Mapping[str, Any], key: str, expected: bool, *, label: str
) -> None:
    if receipt.get(key) is not expected:
        raise FullContextComparisonValidationError(f"{label}/{key} must be exactly {expected}")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FullContextComparisonValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _require_git_commit(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise FullContextComparisonValidationError(
            f"{label} must be a lowercase Git commit id"
        )
    return value


def _validate_producer_source_identity(
    receipt: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    """Require the evaluator source identity before accepting a comparison input."""

    return {
        "git_commit": _require_git_commit(
            receipt.get("git_commit"), label=f"{label}/git_commit"
        ),
        "source_tree_sha256": _require_sha256(
            receipt.get("source_tree_sha256"), label=f"{label}/source_tree_sha256"
        ),
    }


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise FullContextComparisonValidationError(f"{label} must be a positive integer")
    return value


def _require_seed_list(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise FullContextComparisonValidationError(f"{label} must be a non-empty list")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FullContextComparisonValidationError(
                f"{label}[{index}] must be a non-negative integer"
            )
        seeds.append(seed)
    if len(set(seeds)) != len(seeds):
        raise FullContextComparisonValidationError(f"{label} must be unique")
    return tuple(seeds)


def _is_declared_seed(value: Any, declared: Sequence[int]) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value in declared


def _require_datasets(value: Any, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise FullContextComparisonValidationError(f"{label} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise FullContextComparisonValidationError(f"{label} must contain non-empty strings")
    if len(set(value)) != len(value):
        raise FullContextComparisonValidationError(f"{label} must be unique")
    return tuple(value)


def _require_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FullContextComparisonValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise FullContextComparisonValidationError(f"{label} must be finite")
    return result


def _metric_names(task: str) -> tuple[str, ...]:
    if task == "classification":
        return CLASSIFICATION_METRICS
    if task == "regression":
        return REGRESSION_METRICS
    raise FullContextComparisonValidationError(f"unsupported task: {task!r}")


def _validate_metrics(value: Any, *, task: str, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(_metric_names(task)):
        raise FullContextComparisonValidationError(
            f"{label} must contain the complete {task} metric set"
        )
    metrics = {
        metric: _require_finite(value[metric], label=f"{label}/{metric}")
        for metric in _metric_names(task)
    }
    if task == "classification":
        for metric in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro"):
            if not 0.0 <= metrics[metric] <= 1.0:
                raise FullContextComparisonValidationError(
                    f"{label}/{metric} must be inside [0, 1]"
                )
        for metric in ("log_loss", "normalized_nll"):
            if metrics[metric] < 0.0:
                raise FullContextComparisonValidationError(f"{label}/{metric} must be non-negative")
    else:
        for metric in ("mae", "rmse", "scaled_mae", "scaled_rmse"):
            if metrics[metric] < 0.0:
                raise FullContextComparisonValidationError(f"{label}/{metric} must be non-negative")
    return metrics


def _validate_dataset_hashes(value: Any, *, datasets: Sequence[str], label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(datasets):
        raise FullContextComparisonValidationError(
            f"{label} must identify every declared dataset exactly once"
        )
    return {
        dataset_id: _require_sha256(value[dataset_id], label=f"{label}/{dataset_id}")
        for dataset_id in datasets
    }


def _validate_split_manifest(
    value: Any,
    *,
    dataset_id: str,
    dataset_hash: str,
    split_seed: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FullContextComparisonValidationError(f"{label} must be an object")
    if value.get("dataset_id") != dataset_id:
        raise FullContextComparisonValidationError(f"{label}/dataset_id mismatch")
    if value.get("dataset_sha256") != dataset_hash:
        raise FullContextComparisonValidationError(f"{label}/dataset_sha256 mismatch")
    if value.get("split_seed") != split_seed:
        raise FullContextComparisonValidationError(f"{label}/split_seed mismatch")
    _require_positive_int(value.get("train_rows"), label=f"{label}/train_rows")
    _require_positive_int(value.get("query_rows"), label=f"{label}/query_rows")
    for field in (
        "train_indices_sha256",
        "context_order_sha256",
        "query_indices_sha256",
        "feature_indices_sha256",
    ):
        _require_sha256(value.get(field), label=f"{label}/{field}")
    features = value.get("feature_indices")
    if (
        not isinstance(features, list)
        or not features
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in features)
        or len(set(features)) != len(features)
    ):
        raise FullContextComparisonValidationError(
            f"{label}/feature_indices must be non-empty, unique non-negative integers"
        )
    target_scale = _require_finite(value.get("target_scale"), label=f"{label}/target_scale")
    if target_scale <= 0.0:
        raise FullContextComparisonValidationError(f"{label}/target_scale must be positive")
    return value


def _validate_split_manifests(
    value: Any,
    *,
    datasets: Sequence[str],
    dataset_hashes: Mapping[str, str],
    split_seeds: Sequence[int],
    label: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(value, dict) or set(value) != set(datasets):
        raise FullContextComparisonValidationError(
            f"{label} must identify every declared dataset exactly once"
        )
    expected_seed_keys = {str(seed) for seed in split_seeds}
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_id in datasets:
        dataset_manifests = value[dataset_id]
        if not isinstance(dataset_manifests, dict) or set(dataset_manifests) != expected_seed_keys:
            raise FullContextComparisonValidationError(
                f"{label}/{dataset_id} must identify every split seed exactly once"
            )
        result[dataset_id] = {
            str(seed): _validate_split_manifest(
                dataset_manifests[str(seed)],
                dataset_id=dataset_id,
                dataset_hash=dataset_hashes[dataset_id],
                split_seed=seed,
                label=f"{label}/{dataset_id}/{seed}",
            )
            for seed in split_seeds
        }
    return result


def _validate_frozen_hashes(receipt: Mapping[str, Any], checkpoint_seeds: Sequence[int]) -> None:
    _require_exact_bool(receipt, "optimizer_created", False, label="frozen")
    _require_exact_bool(receipt, "frozen_arm_optimizer_created", False, label="frozen")
    _require_exact_bool(receipt, "all_frozen_arm_parameter_hashes_unchanged", True, label="frozen")
    _require_exact_bool(receipt, "all_parameter_hashes_unchanged", True, label="frozen")
    hashes = receipt.get("per_arm_parameter_hashes")
    expected_seed_keys = {str(seed) for seed in checkpoint_seeds}
    if not isinstance(hashes, dict) or set(hashes) != expected_seed_keys:
        raise FullContextComparisonValidationError(
            "frozen/per_arm_parameter_hashes must cover every checkpoint seed"
        )
    for seed in checkpoint_seeds:
        seed_hashes = hashes[str(seed)]
        if not isinstance(seed_hashes, dict) or set(seed_hashes) != set(FROZEN_ARMS):
            raise FullContextComparisonValidationError(
                f"frozen/per_arm_parameter_hashes/{seed} must cover every frozen arm"
            )
        for arm in FROZEN_ARMS:
            arm_hashes = seed_hashes[arm]
            if not isinstance(arm_hashes, dict):
                raise FullContextComparisonValidationError(
                    f"frozen/per_arm_parameter_hashes/{seed}/{arm} must be an object"
                )
            before = _require_sha256(
                arm_hashes.get("before"), label=f"frozen/hashes/{seed}/{arm}/before"
            )
            after = _require_sha256(
                arm_hashes.get("after"), label=f"frozen/hashes/{seed}/{arm}/after"
            )
            if arm_hashes.get("unchanged") is not True or before != after:
                raise FullContextComparisonValidationError(
                    f"frozen hash gate failed for checkpoint {seed}, arm {arm}"
                )


def _validate_frozen_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_checkpoint_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    if receipt.get("schema_version") != FROZEN_SCHEMA:
        raise FullContextComparisonValidationError("frozen receipt schema is not full-context v1")
    if receipt.get("status") != "local_unissued":
        raise FullContextComparisonValidationError("frozen receipt status must be local_unissued")
    producer_source_identity = _validate_producer_source_identity(receipt, label="frozen")
    if (
        receipt.get("context_policy") != "full_train"
        or receipt.get("context_sizes") is not None
        or receipt.get("query_limit") is not None
        or receipt.get("query_policy") != "all_heldout_rows"
    ):
        raise FullContextComparisonValidationError(
            "frozen receipt must expose all train rows and all held-out rows"
        )
    if receipt.get("query_chunk_semantics") != (
        "response_readout_only_after_one_full_transductive_evidence_episode"
    ) or receipt.get("query_evidence_policy") != (
        "all_heldout_predictors_single_transductive_episode"
    ):
        raise FullContextComparisonValidationError(
            "frozen receipt must declare the transductive all-query evidence semantics"
        )
    if receipt.get("arms") != list(FROZEN_ARMS):
        raise FullContextComparisonValidationError("frozen receipt arms are incomplete")

    datasets = _require_datasets(receipt.get("datasets"), label="frozen/datasets")
    hashes = _validate_dataset_hashes(
        receipt.get("dataset_hashes"), datasets=datasets, label="frozen/dataset_hashes"
    )
    checkpoint_seeds = _require_seed_list(
        receipt.get("checkpoint_seeds"), label="frozen/checkpoint_seeds"
    )
    if expected_checkpoint_seeds is None and len(checkpoint_seeds) != 3:
        raise FullContextComparisonValidationError(
            "frozen receipt must contain exactly three checkpoint seeds"
        )
    if expected_checkpoint_seeds is not None:
        expected = tuple(expected_checkpoint_seeds)
        if not expected or checkpoint_seeds != expected:
            raise FullContextComparisonValidationError(
                "frozen pilot receipt checkpoint seeds do not match the declared pilot panel"
            )
    split_seeds = _require_seed_list(receipt.get("split_seeds"), label="frozen/split_seeds")
    manifests = _validate_split_manifests(
        receipt.get("split_manifests"),
        datasets=datasets,
        dataset_hashes=hashes,
        split_seeds=split_seeds,
        label="frozen/split_manifests",
    )
    _validate_frozen_hashes(receipt, checkpoint_seeds)

    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(checkpoint_seeds):
        raise FullContextComparisonValidationError(
            "frozen/checkpoints must identify every checkpoint seed exactly once"
        )
    checkpoint_index: dict[int, Mapping[str, Any]] = {}
    for item in checkpoints:
        if not isinstance(item, dict):
            raise FullContextComparisonValidationError("frozen checkpoint must be an object")
        seed = item.get("seed")
        if not _is_declared_seed(seed, checkpoint_seeds) or seed in checkpoint_index:
            raise FullContextComparisonValidationError("frozen checkpoint seed is invalid")
        if not isinstance(item.get("path"), str) or not item["path"]:
            raise FullContextComparisonValidationError("frozen checkpoint path is invalid")
        _require_sha256(item.get("sha256"), label=f"frozen/checkpoints/{seed}/sha256")
        checkpoint_index[int(seed)] = item
    if set(checkpoint_index) != set(checkpoint_seeds):
        raise FullContextComparisonValidationError("frozen checkpoint panel is incomplete")

    context_rows = receipt.get("context_rows_by_dataset_split")
    if not isinstance(context_rows, dict) or set(context_rows) != set(datasets):
        raise FullContextComparisonValidationError(
            "frozen/context_rows_by_dataset_split does not match datasets"
        )
    for dataset_id in datasets:
        expected = {
            str(seed): manifests[dataset_id][str(seed)]["train_rows"] for seed in split_seeds
        }
        if context_rows[dataset_id] != expected:
            raise FullContextComparisonValidationError(
                f"frozen context row ledger disagrees for {dataset_id}"
            )

    records = receipt.get("records")
    if not isinstance(records, list):
        raise FullContextComparisonValidationError("frozen/records must be a list")
    index: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    dataset_contracts: dict[str, tuple[str, int | None, int]] = {}
    for position, row in enumerate(records):
        if not isinstance(row, dict):
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} must be an object"
            )
        dataset_id = row.get("dataset_id")
        checkpoint_seed = row.get("checkpoint_seed")
        split_seed = row.get("split_seed")
        arm = row.get("arm")
        if (
            not isinstance(dataset_id, str)
            or dataset_id not in datasets
            or not _is_declared_seed(checkpoint_seed, checkpoint_seeds)
            or not _is_declared_seed(split_seed, split_seeds)
            or not isinstance(arm, str)
            or arm not in FROZEN_ARMS
        ):
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} is outside the declared panel"
            )
        manifest = manifests[str(dataset_id)][str(split_seed)]
        if row.get("split_manifest") != manifest:
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} split manifest mismatch"
            )
        if (
            row.get("dataset_sha256") != hashes[str(dataset_id)]
            or row.get("context_policy") != "full_train"
            or row.get("full_context") is not True
            or row.get("context_size") != manifest["train_rows"]
            or row.get("train_rows_total") != manifest["train_rows"]
            or row.get("query_rows") != manifest["query_rows"]
        ):
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} does not use the complete registered split"
            )
        predictor_count = _require_positive_int(
            row.get("predictor_count"), label=f"frozen/records/{position}/predictor_count"
        )
        if predictor_count != len(manifest["feature_indices"]):
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} predictor count disagrees with split manifest"
            )
        if row.get("selected_feature_indices") != manifest["feature_indices"]:
            raise FullContextComparisonValidationError(
                f"frozen/records/{position} selected features disagree with split manifest"
            )
        task = row.get("task")
        classes = row.get("classes")
        if task == "classification":
            if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
                raise FullContextComparisonValidationError(
                    f"frozen/records/{position} classification classes are invalid"
                )
        elif task == "regression":
            if classes is not None:
                raise FullContextComparisonValidationError(
                    f"frozen/records/{position} regression classes must be null"
                )
        else:
            raise FullContextComparisonValidationError(f"frozen/records/{position} task is invalid")
        metrics = _validate_metrics(
            row.get("metrics"), task=str(task), label=f"frozen/records/{position}/metrics"
        )
        contract = (str(task), classes, predictor_count)
        existing = dataset_contracts.setdefault(str(dataset_id), contract)
        if existing != contract:
            raise FullContextComparisonValidationError(
                f"frozen dataset contract drifts for {dataset_id}"
            )
        key = (str(dataset_id), int(checkpoint_seed), int(split_seed), str(arm))
        if key in index:
            raise FullContextComparisonValidationError(f"duplicate frozen record: {key}")
        index[key] = row | {"metrics": metrics}

    expected_count = len(datasets) * len(checkpoint_seeds) * len(split_seeds) * len(FROZEN_ARMS)
    if len(index) != expected_count:
        raise FullContextComparisonValidationError("frozen record panel is incomplete")
    return {
        "datasets": datasets,
        "dataset_hashes": hashes,
        "checkpoint_seeds": checkpoint_seeds,
        "split_seeds": split_seeds,
        "split_manifests": manifests,
        "panel_manifest": receipt.get("panel_manifest"),
        "producer_source_identity": producer_source_identity,
        "records": index,
        "dataset_contracts": dataset_contracts,
    }


def _validate_baseline_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if receipt.get("schema_version") != BASELINE_SCHEMA:
        raise FullContextComparisonValidationError("baseline receipt schema is not full-context v1")
    if receipt.get("status") != "local_unissued":
        raise FullContextComparisonValidationError("baseline receipt status must be local_unissued")
    producer_source_identity = _validate_producer_source_identity(receipt, label="baseline")
    if (
        receipt.get("context_policy") != "full_train"
        or receipt.get("train_policy") != "all_train_partition_rows"
        or receipt.get("query_policy") != "all_heldout_rows"
    ):
        raise FullContextComparisonValidationError(
            "baseline receipt must fit all train rows and score all held-out rows"
        )
    if receipt.get("estimators") != list(BASELINE_ESTIMATORS):
        raise FullContextComparisonValidationError("baseline estimator panel is incomplete")
    if receipt.get("estimator_seed_policy") != "estimator_seed_equals_split_seed":
        raise FullContextComparisonValidationError("baseline estimator seed policy is invalid")

    datasets = _require_datasets(receipt.get("datasets"), label="baseline/datasets")
    hashes = _validate_dataset_hashes(
        receipt.get("dataset_hashes"), datasets=datasets, label="baseline/dataset_hashes"
    )
    split_seeds = _require_seed_list(receipt.get("split_seeds"), label="baseline/split_seeds")
    manifests = _validate_split_manifests(
        receipt.get("split_manifests"),
        datasets=datasets,
        dataset_hashes=hashes,
        split_seeds=split_seeds,
        label="baseline/split_manifests",
    )
    manifest_hashes = receipt.get("split_manifest_sha256")
    if not isinstance(manifest_hashes, dict) or set(manifest_hashes) != set(datasets):
        raise FullContextComparisonValidationError(
            "baseline/split_manifest_sha256 does not match datasets"
        )
    for dataset_id in datasets:
        expected = {
            str(seed): canonical_hash(manifests[dataset_id][str(seed)]) for seed in split_seeds
        }
        if manifest_hashes[dataset_id] != expected:
            raise FullContextComparisonValidationError(
                f"baseline split manifest hash ledger disagrees for {dataset_id}"
            )

    records = receipt.get("records")
    if not isinstance(records, list):
        raise FullContextComparisonValidationError("baseline/records must be a list")
    index: dict[tuple[str, int, str], dict[str, Any]] = {}
    dataset_contracts: dict[str, tuple[str, int | None, int]] = {}
    for position, row in enumerate(records):
        if not isinstance(row, dict):
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} must be an object"
            )
        dataset_id = row.get("dataset_id")
        split_seed = row.get("split_seed")
        estimator = row.get("estimator")
        if (
            not isinstance(dataset_id, str)
            or dataset_id not in datasets
            or not _is_declared_seed(split_seed, split_seeds)
            or not isinstance(estimator, str)
            or estimator not in BASELINE_ESTIMATORS
            or row.get("estimator_seed") != split_seed
        ):
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} is outside the declared panel"
            )
        manifest = manifests[str(dataset_id)][str(split_seed)]
        if row.get("split_manifest") != manifest:
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} split manifest mismatch"
            )
        expected_manifest_hash = canonical_hash(manifest)
        if row.get("split_manifest_sha256") != expected_manifest_hash:
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} split manifest hash mismatch"
            )
        if (
            row.get("train_rows") != manifest["train_rows"]
            or row.get("query_rows") != manifest["query_rows"]
        ):
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} does not use the complete registered split"
            )
        predictor_count = _require_positive_int(
            row.get("predictor_count"), label=f"baseline/records/{position}/predictor_count"
        )
        if predictor_count != len(manifest["feature_indices"]):
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} predictor count disagrees with split manifest"
            )
        task = row.get("task")
        classes = row.get("classes")
        if task == "classification":
            if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
                raise FullContextComparisonValidationError(
                    f"baseline/records/{position} classification classes are invalid"
                )
        elif task == "regression":
            if classes is not None:
                raise FullContextComparisonValidationError(
                    f"baseline/records/{position} regression classes must be null"
                )
        else:
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} task is invalid"
            )
        metrics = _validate_metrics(
            row.get("metrics"), task=str(task), label=f"baseline/records/{position}/metrics"
        )
        fit = row.get("fit")
        if (
            not isinstance(fit, dict)
            or fit.get("estimator") != estimator
            or fit.get("estimator_seed") != split_seed
            or fit.get("fit_rows") != manifest["train_rows"]
            or fit.get("query_rows") != manifest["query_rows"]
        ):
            raise FullContextComparisonValidationError(
                f"baseline/records/{position} fit ledger does not use the complete split"
            )
        contract = (str(task), classes, predictor_count)
        existing = dataset_contracts.setdefault(str(dataset_id), contract)
        if existing != contract:
            raise FullContextComparisonValidationError(
                f"baseline dataset contract drifts for {dataset_id}"
            )
        key = (str(dataset_id), int(split_seed), str(estimator))
        if key in index:
            raise FullContextComparisonValidationError(f"duplicate baseline record: {key}")
        index[key] = row | {"metrics": metrics}

    expected_count = len(datasets) * len(split_seeds) * len(BASELINE_ESTIMATORS)
    if len(index) != expected_count:
        raise FullContextComparisonValidationError("baseline record panel is incomplete")
    return {
        "datasets": datasets,
        "dataset_hashes": hashes,
        "split_seeds": split_seeds,
        "split_manifests": manifests,
        "panel_manifest": receipt.get("panel_manifest"),
        "producer_source_identity": producer_source_identity,
        "records": index,
        "dataset_contracts": dataset_contracts,
    }


def _panel_manifest_identity(
    value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Separate a content-bound panel identity from its snapshot-local path."""

    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise FullContextComparisonValidationError(f"{label} must be null or an object")
    path = value.get("path")
    if not isinstance(path, str) or not path:
        raise FullContextComparisonValidationError(f"{label}/path must be a non-empty string")
    identity = {key: item for key, item in value.items() if key != "path"}
    if not identity:
        raise FullContextComparisonValidationError(
            f"{label} must contain content identity fields in addition to path"
        )
    return identity, path


def _mean_metrics(rows: Sequence[Mapping[str, Any]], *, task: str) -> dict[str, float]:
    if not rows:
        raise FullContextComparisonValidationError("cannot summarize an empty metric panel")
    return {
        metric: float(math.fsum(float(row["metrics"][metric]) for row in rows) / len(rows))
        for metric in _metric_names(task)
    }


def _summarize(frozen: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    split_seeds: tuple[int, ...] = frozen["split_seeds"]
    checkpoint_seeds: tuple[int, ...] = frozen["checkpoint_seeds"]
    for dataset_id in frozen["datasets"]:
        task, classes, _ = frozen["dataset_contracts"][dataset_id]
        by_checkpoint: dict[str, Any] = {}
        pooled: list[Mapping[str, Any]] = []
        for checkpoint_seed in checkpoint_seeds:
            rows = [
                frozen["records"][(dataset_id, checkpoint_seed, split_seed, "pretrained_frozen")]
                for split_seed in split_seeds
            ]
            pooled.extend(rows)
            by_checkpoint[str(checkpoint_seed)] = {
                "split_count": len(rows),
                "aggregation_unit": "split_seed_equal_weight",
                "metrics_mean": _mean_metrics(rows, task=task),
            }
        baseline_summaries: dict[str, Any] = {}
        for estimator in BASELINE_ESTIMATORS:
            rows = [
                baseline["records"][(dataset_id, split_seed, estimator)]
                for split_seed in split_seeds
            ]
            baseline_summaries[estimator] = {
                "split_count": len(rows),
                "aggregation_unit": "split_seed_equal_weight",
                "metrics_mean": _mean_metrics(rows, task=task),
            }
        summaries[dataset_id] = {
            "task": task,
            "classes": classes,
            "common_metrics": list(_metric_names(task)),
            "tabubase_pretrained": {
                "evaluation_semantics": (
                    "transductive_all_heldout_predictors_in_one_frozen_evidence_episode"
                ),
                "checkpoint_seed_count": len(checkpoint_seeds),
                "by_checkpoint_seed": by_checkpoint,
                "descriptive_pooled_mean": {
                    "observation_count": len(pooled),
                    "aggregation_unit": "checkpoint_seed_x_split_seed_equal_weight",
                    "metrics_mean": _mean_metrics(pooled, task=task),
                    "interpretation": (
                        "descriptive pooling only; checkpoint seeds are reported separately above"
                    ),
                },
            },
            "classical_baselines": {
                "evaluation_semantics": (
                    "inductive_fit_on_train_partition_then_predict_complete_heldout_partition"
                ),
                "estimators": baseline_summaries,
            },
        }
    return summaries


def _compare_full_context_receipts(
    frozen_receipt_path: Path,
    baseline_receipt_path: Path,
    *,
    output_path: Path | None = None,
    expected_checkpoint_seeds: Sequence[int] | None = None,
    schema_version: str,
    comparison_scope: str,
    claim_boundary: str,
) -> dict[str, Any]:
    """Validate and descriptively summarize aligned full-context receipts."""

    frozen_path = frozen_receipt_path.resolve()
    baseline_path = baseline_receipt_path.resolve()
    if frozen_path == baseline_path:
        raise FullContextComparisonValidationError(
            "frozen and baseline receipts must be distinct files"
        )
    if output_path is not None and output_path.resolve() in {frozen_path, baseline_path}:
        raise FullContextComparisonValidationError(
            "output path must not overwrite an input receipt"
        )

    frozen_payload = _load_json_object(frozen_receipt_path, label="frozen")
    baseline_payload = _load_json_object(baseline_receipt_path, label="baseline")
    frozen = _validate_frozen_receipt(
        frozen_payload,
        expected_checkpoint_seeds=expected_checkpoint_seeds,
    )
    baseline = _validate_baseline_receipt(baseline_payload)

    compatibility_fields = (
        "datasets",
        "dataset_hashes",
        "split_seeds",
        "split_manifests",
        "dataset_contracts",
    )
    for field in compatibility_fields:
        if frozen[field] != baseline[field]:
            raise FullContextComparisonValidationError(
                f"frozen and baseline receipts differ at compatibility field: {field}"
            )
    frozen_panel_identity, frozen_panel_path = _panel_manifest_identity(
        frozen["panel_manifest"], label="frozen/panel_manifest"
    )
    baseline_panel_identity, baseline_panel_path = _panel_manifest_identity(
        baseline["panel_manifest"], label="baseline/panel_manifest"
    )
    if frozen_panel_identity != baseline_panel_identity:
        raise FullContextComparisonValidationError(
            "frozen and baseline receipts differ at compatibility field: panel_manifest_identity"
        )
    if frozen["producer_source_identity"] != baseline["producer_source_identity"]:
        raise FullContextComparisonValidationError(
            "frozen and baseline receipts differ at compatibility field: producer_source_identity"
        )

    split_manifest_hashes = {
        dataset_id: {
            str(seed): canonical_hash(frozen["split_manifests"][dataset_id][str(seed)])
            for seed in frozen["split_seeds"]
        }
        for dataset_id in frozen["datasets"]
    }
    receipt: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "local_unissued",
        "inputs": {
            "frozen": {
                "path": str(frozen_receipt_path),
                "sha256": _sha256_file(frozen_receipt_path),
                "schema_version": FROZEN_SCHEMA,
            },
            "baselines": {
                "path": str(baseline_receipt_path),
                "sha256": _sha256_file(baseline_receipt_path),
                "schema_version": BASELINE_SCHEMA,
            },
        },
        "compatibility": {
            "datasets": list(frozen["datasets"]),
            "dataset_hashes": frozen["dataset_hashes"],
            "checkpoint_seeds": list(frozen["checkpoint_seeds"]),
            "split_seeds": list(frozen["split_seeds"]),
            "split_manifest_sha256": split_manifest_hashes,
            "panel_manifest": frozen["panel_manifest"],
            "panel_manifest_content_identity_equal": True,
            "panel_manifest_paths": {
                "frozen": frozen_panel_path,
                "baselines": baseline_panel_path,
            },
            "panel_manifest_path_is_nonsemantic": True,
            "producer_source_identity": frozen["producer_source_identity"],
            "producer_source_identity_equal": True,
            "all_heldout_query_rows_identical": True,
            "all_train_context_rows_identical": True,
            "frozen_optimizer_created": False,
            "all_frozen_arm_parameter_hashes_unchanged": True,
        },
        "evaluation_semantics": {
            "tabubase_pretrained": (
                "frozen transductive episode containing all held-out predictor rows; "
                "only response readout is chunked"
            ),
            "mlp_xgboost": (
                "inductive estimator fitted independently on each train partition and then "
                "applied to the complete held-out partition"
            ),
            "comparison_scope": comparison_scope,
        },
        "datasets": _summarize(frozen, baseline),
        "all_compatibility_and_frozen_gates_passed": True,
        "claim_boundary": claim_boundary,
    }
    if output_path is None:
        return receipt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt | {
        "result_path": str(output_path),
        "result_sha256": _sha256_file(output_path),
    }


def compare_full_context_receipts(
    frozen_receipt_path: Path,
    baseline_receipt_path: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare the registered three-checkpoint panel without weakening its gate."""

    return _compare_full_context_receipts(
        frozen_receipt_path,
        baseline_receipt_path,
        output_path=output_path,
        schema_version=COMPARISON_SCHEMA,
        comparison_scope=(
            "side-by-side descriptive metric summary on identical rows; no paired "
            "superiority inference across the different evaluation semantics"
        ),
        claim_boundary=(
            "local-unissued descriptive comparison only; TabUBase is optimizer-free and "
            "transductive over all held-out predictors, whereas MLP/XGBoost are inductive "
            "per-split fitted baselines; no fine-tuning of TabUBase, formal receipt, SOTA, "
            "foundation-model, causal, or broad benchmark claim"
        ),
    )


def compare_full_context_pilot_receipts(
    frozen_receipt_path: Path,
    baseline_receipt_path: Path,
    *,
    checkpoint_seed: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Compare one checkpoint as a local pilot, not a robustness panel."""

    if (
        isinstance(checkpoint_seed, bool)
        or not isinstance(checkpoint_seed, int)
        or checkpoint_seed < 0
    ):
        raise FullContextComparisonValidationError(
            "pilot checkpoint seed must be a non-negative integer"
        )
    return _compare_full_context_receipts(
        frozen_receipt_path,
        baseline_receipt_path,
        output_path=output_path,
        expected_checkpoint_seeds=(checkpoint_seed,),
        schema_version=PILOT_COMPARISON_SCHEMA,
        comparison_scope=(
            "single-checkpoint side-by-side descriptive pilot on identical rows; no "
            "checkpoint-robustness or paired superiority inference across the different "
            "evaluation semantics"
        ),
        claim_boundary=(
            "local-unissued single-checkpoint pilot only; it does not establish robustness "
            "across pretrained checkpoints. TabUBase is optimizer-free and transductive over "
            "all held-out predictors, whereas MLP/XGBoost are inductive per-split fitted "
            "baselines; no fine-tuning of TabUBase, formal receipt, SOTA, foundation-model, "
            "causal, or broad benchmark claim"
        ),
    )


__all__ = [
    "BASELINE_SCHEMA",
    "COMPARISON_SCHEMA",
    "FROZEN_SCHEMA",
    "PILOT_COMPARISON_SCHEMA",
    "FullContextComparisonValidationError",
    "compare_full_context_pilot_receipts",
    "compare_full_context_receipts",
]
