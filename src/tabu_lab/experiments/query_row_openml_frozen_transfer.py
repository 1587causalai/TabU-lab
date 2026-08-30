"""R4-v2 TabUR frozen transfer on a pinned OpenML panel.

The runner is deliberately separate from the Axis-B OpenML evaluators.  It
loads only ``tabu.query.row@0.1.0`` checkpoints, builds the same low-shot
episodes for frozen TabUR and context-only classical fits, and records source
and split identities without promoting a local run into a formal claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any
from dataclasses import replace

import numpy as np
import torch
import yaml

from tabu_lab.contracts import canonical_hash
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_pretraining import load_query_row_pretrain_checkpoint
from .query_row_r5_classical_icl import _state_hash
from .tabubase_openml_new6 import (
    OPENML_NEW6_BY_ID,
    OPENML_NEW6_SPECS,
    fetch_openml_new6_dataset,
)
from .tabubase_real_benchmark import _source_tree_hash
from .tabubase_real_icl import (
    LOW_SHOT_CONTEXT_POLICY,
    build_real_icl_episode,
    prepare_real_icl_split,
    real_icl_split_manifest,
)
from .tabubase_real_metrics import classification_metrics, regression_metrics
from .tabubase_scale import ROOT_SEEDS, resolve_device

QUERY_OPENML_PANEL_SCHEMA = "tabu.query-row.openml-new6-panel.v1"
QUERY_OPENML_RESULT_SCHEMA = "tabu.query-row.openml-frozen-classical-result.v1"
QUERY_OPENML_K_GRID = (0, 1, 2, 4, 8, 16, 32)
QUERY_OPENML_PANEL_ID = "tabur-query-row-openml-new6-2026-08-31"
QUERY_BASE_MODEL_SPEC_HASH = "24f9ddae70b116b2ac88d2ccc833870ca351de8148774fe8b9fbf72a0c58d1c0"

BASELINE_IDS = (
    "ordinary_least_squares.context_only.v1",
    "mlp.context_only.v1",
    "xgboost.context_only.v1",
)
BASELINE_CONFIG: dict[str, Any] = {
    "linear_regression": {"ridge": 1.0e-4},
    "logistic_regression": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
    "mlp": {
        "hidden_layer_sizes": (64, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1.0e-4,
        "learning_rate_init": 1.0e-3,
        "max_iter": 500,
        "early_stopping": False,
        "shuffle": True,
        "tol": 1.0e-4,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 8,
        "tree_method": "hist",
    },
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _require_equal(field: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"query OpenML panel drift at {field}: expected {expected!r}, got {observed!r}")


def load_query_openml_panel_manifest(path: Path) -> dict[str, Any]:
    """Validate the query-specific OpenML data manifest without Axis-B assumptions."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing query OpenML panel manifest: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("query OpenML panel manifest must be a mapping")
    _require_equal("schema_version", QUERY_OPENML_PANEL_SCHEMA, payload.get("schema_version"))
    _require_equal("panel_id", QUERY_OPENML_PANEL_ID, payload.get("panel_id"))
    status = payload.get("status")
    if not isinstance(status, dict):
        raise RuntimeError("query OpenML panel status must be a mapping")
    _require_equal("status.registration", "candidate_preregistered", status.get("registration"))
    _require_equal("status.execution", "not_run", status.get("execution"))
    _require_equal("status.empirical_claim", "none", status.get("empirical_claim"))
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("query OpenML panel model must be a mapping")
    _require_equal("model.contract_id", "tabu.query.row", model.get("contract_id"))
    _require_equal("model.contract_version", "0.1.0", str(model.get("contract_version")))
    _require_equal("model.model_spec_hash", QUERY_BASE_MODEL_SPEC_HASH, model.get("model_spec_hash"))
    _require_equal("model.profile_id", "supervised.label_broadcast.v1", model.get("profile_id"))
    _require_equal("model.row_token_count", 4, model.get("row_token_count"))
    source = payload.get("source_contract")
    if not isinstance(source, dict):
        raise RuntimeError("query OpenML panel source_contract must be a mapping")
    for key, expected in {
        "provider": "OpenML",
        "api": "sklearn.datasets.fetch_openml",
        "identity_key": "data_id",
        "target_column": "default-target",
        "as_frame": False,
        "parser": "liac-arff",
        "cache": True,
    }.items():
        _require_equal(f"source_contract.{key}", expected, source.get(key))
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise RuntimeError("query OpenML panel datasets must be a list")
    by_id = {entry.get("dataset_id"): entry for entry in entries if isinstance(entry, dict)}
    expected_ids = tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
    if set(by_id) != set(expected_ids):
        raise RuntimeError(f"query OpenML panel must contain exactly {expected_ids!r}")
    for spec in OPENML_NEW6_SPECS:
        entry = by_id[spec.dataset_id]
        for key, expected in {
            "openml_name": spec.openml_name,
            "data_id": spec.data_id,
            "version": spec.version,
            "upstream_md5": spec.upstream_md5,
            "license": spec.license,
            "task": spec.task,
            "rows": spec.rows,
            "predictors": spec.predictors,
            "classes": spec.classes,
        }.items():
            _require_equal(f"datasets.{spec.dataset_id}.{key}", expected, entry.get(key))
    evaluation = payload.get("evaluation_design")
    if not isinstance(evaluation, dict):
        raise RuntimeError("query OpenML panel evaluation_design must be a mapping")
    for key, expected in {
        "checkpoint_seeds": list(ROOT_SEEDS),
        "split_seeds": list(ROOT_SEEDS),
        "context_policy": "low_shot_grid",
        "context_sizes": list(QUERY_OPENML_K_GRID),
        "query_limit": 256,
        "query_chunk_rows": 64,
    }.items():
        _require_equal(f"evaluation_design.{key}", expected, evaluation.get(key))
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "canonical_payload_sha256": _canonical_sha256(payload),
        "payload": payload,
        "dataset_ids": expected_ids,
    }


def _source_commit() -> str | None:
    explicit = os.environ.get("TABU_SOURCE_COMMIT")
    if explicit:
        return explicit
    result = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def _aligned_probabilities(model: Any, features: np.ndarray, *, classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    labels = np.asarray(model.classes_)
    try:
        labels = labels.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("classifier classes must be integer labels") from exc
    if raw.shape != (len(features), len(labels)) or not np.array_equal(np.sort(labels), np.arange(classes)):
        raise RuntimeError("classifier probability columns do not cover the declared domain")
    probabilities = np.zeros((len(features), classes), dtype=np.float64)
    for column, label in enumerate(labels.tolist()):
        probabilities[:, int(label)] = raw[:, column]
    if not np.isfinite(probabilities).all() or bool((probabilities < 0).any()):
        raise RuntimeError("classifier emitted invalid probabilities")
    probabilities = np.clip(probabilities, 1.0e-12, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _no_context_metrics(split: Any) -> dict[str, float]:
    truth = split.response[split.query_indices]
    if split.dataset.task == "classification":
        assert split.classes is not None
        probabilities = np.full((len(truth), split.classes), 1.0 / split.classes, dtype=np.float64)
        return classification_metrics(truth, probabilities, classes=split.classes)
    return regression_metrics(truth, np.zeros(len(truth), dtype=np.float64), target_scale=split.target_scale)


def _fit_baseline(split: Any, *, context_size: int, estimator: str, seed: int) -> dict[str, Any]:
    truth = split.response[split.query_indices]
    if context_size == 0:
        return {"status": "not_applicable", "metrics": _no_context_metrics(split), "fit": {"fit_rows": 0}}
    if split.dataset.task == "classification" and split.classes is not None and context_size < split.classes:
        return {
            "status": "not_applicable",
            "metrics": None,
            "fit": {"fit_rows": context_size, "reason": "context_below_declared_class_count"},
        }
    try:
        import sklearn
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("OpenML baselines require scikit-learn and xgboost") from exc
    context = split.context_order[:context_size]
    x_train = np.asarray(split.features[context], dtype=np.float32)
    x_query = np.asarray(split.features[split.query_indices], dtype=np.float32)
    y_train = np.asarray(split.response[context])
    fit: dict[str, Any] = {
        "estimator": estimator,
        "estimator_seed": seed,
        "fit_rows": len(context),
        "query_rows": len(split.query_indices),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if split.dataset.task == "classification":
            assert split.classes is not None
            if estimator == "linear":
                model = LogisticRegression(**BASELINE_CONFIG["logistic_regression"], random_state=seed)
                train_features, query_features = x_train, x_query
                fit["estimator_config"] = BASELINE_CONFIG["logistic_regression"] | {"random_state": seed}
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features, query_features = scaler.transform(x_train), scaler.transform(x_query)
                config = BASELINE_CONFIG["mlp"]
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
                    activation=str(config["activation"]),
                    solver=str(config["solver"]),
                    alpha=float(config["alpha"]),
                    learning_rate_init=float(config["learning_rate_init"]),
                    max_iter=int(config["max_iter"]),
                    early_stopping=bool(config["early_stopping"]),
                    shuffle=bool(config["shuffle"]),
                    tol=float(config["tol"]),
                    batch_size=min(64, len(y_train)),
                    random_state=seed,
                )
                fit["preprocessing"] = "train_only_standard_scaler"
            elif estimator == "xgboost":
                model = xgb.XGBClassifier(
                    **BASELINE_CONFIG["xgboost"],
                    random_state=seed,
                    eval_metric="logloss",
                )
                train_features, query_features = x_train, x_query
                fit["estimator_config"] = BASELINE_CONFIG["xgboost"] | {
                    "random_state": seed,
                    "eval_metric": "logloss",
                }
            else:
                raise ValueError(f"unknown OpenML baseline: {estimator}")
            model.fit(train_features, y_train)
            metrics = classification_metrics(
                truth,
                _aligned_probabilities(model, query_features, classes=split.classes),
                classes=split.classes,
            )
        else:
            if estimator == "linear":
                model = Ridge(alpha=float(BASELINE_CONFIG["linear_regression"]["ridge"]))
                train_features, query_features, train_target = x_train, x_query, y_train
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features, query_features = scaler.transform(x_train), scaler.transform(x_query)
                target_mean = float(y_train.mean())
                target_scale = max(float(y_train.std()), 1.0e-6)
                train_target = (y_train - target_mean) / target_scale
                config = BASELINE_CONFIG["mlp"]
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(config["hidden_layer_sizes"]),
                    activation=str(config["activation"]),
                    solver=str(config["solver"]),
                    alpha=float(config["alpha"]),
                    learning_rate_init=float(config["learning_rate_init"]),
                    max_iter=int(config["max_iter"]),
                    early_stopping=bool(config["early_stopping"]),
                    shuffle=bool(config["shuffle"]),
                    tol=float(config["tol"]),
                    batch_size=min(64, len(y_train)),
                    random_state=seed,
                )
                fit["target_standardization"] = {"mean": target_mean, "scale": target_scale}
            elif estimator == "xgboost":
                model = xgb.XGBRegressor(
                    **BASELINE_CONFIG["xgboost"],
                    random_state=seed,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                )
                train_features, query_features, train_target = x_train, x_query, y_train
            else:
                raise ValueError(f"unknown OpenML baseline: {estimator}")
            model.fit(train_features, train_target)
            predicted = np.asarray(model.predict(query_features), dtype=np.float64)
            if estimator == "mlp":
                predicted = predicted * target_scale + target_mean
            metrics = regression_metrics(truth, predicted, target_scale=split.target_scale)
    fit["warnings"] = [str(item.message) for item in caught]
    if estimator == "mlp":
        fit["n_iter"] = int(model.n_iter_)
        fit["converged_before_max_iter"] = int(model.n_iter_) < int(BASELINE_CONFIG["mlp"]["max_iter"])
    return {"status": "passed", "metrics": metrics, "fit": fit}


def _build_model_from_checkpoint(path: Path, *, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    identity_path = path.with_suffix(".identity.json")
    if not identity_path.is_file():
        raise FileNotFoundError(f"checkpoint identity sidecar is required: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    model_identity = identity["model_identity"]
    ref = model_identity["reference_config"]
    config = ReferenceConfig(
        d_model=int(ref["d_model"]),
        n_heads=int(ref["n_heads"]),
        d_ff=int(ref["d_ff"]),
        n_blocks=int(ref["n_blocks"]),
        inducing_slots=int(ref["inducing_slots"]),
        matched_slots=int(ref["matched_slots"]),
        max_features=int(ref["max_features"]),
    )
    model = build_model(
        "tabu.query.row",
        config=config,
        profile=str(model_identity["profile_id"]),
        row_token_count=int(model_identity["row_token_count"]),
    ).to(device)
    load_query_row_pretrain_checkpoint(model, path)
    model.eval()
    model.requires_grad_(False)
    return model, identity


def _build_random_model(identity: dict[str, Any], *, seed: int, device: torch.device) -> torch.nn.Module:
    model_identity = identity["model_identity"]
    ref = model_identity["reference_config"]
    config = ReferenceConfig(
        d_model=int(ref["d_model"]),
        n_heads=int(ref["n_heads"]),
        d_ff=int(ref["d_ff"]),
        n_blocks=int(ref["n_blocks"]),
        inducing_slots=int(ref["inducing_slots"]),
        matched_slots=int(ref["matched_slots"]),
        max_features=int(ref["max_features"]),
    )
    torch.manual_seed(seed)
    model = build_model(
        "tabu.query.row",
        config=config,
        profile=str(model_identity["profile_id"]),
        row_token_count=int(model_identity["row_token_count"]),
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _tabur_outputs(model: torch.nn.Module, evidence: Any, split: Any, *, context_size: int) -> tuple[dict[str, float], torch.Tensor]:
    """Return metrics and the public query prediction slice for one episode."""

    with torch.inference_mode():
        prediction = model(evidence)
    truth = split.response[split.query_indices]
    if split.dataset.task == "classification":
        assert split.classes is not None
        values = prediction.entries["distribution"].values
        if values is None:
            raise RuntimeError("TabUR query OpenML classification returned no distribution")
        if values.ndim == 4:
            values = values[0]
        probabilities = values[context_size:, -1, : split.classes].detach().cpu().numpy()
        probabilities = np.clip(probabilities.astype(np.float64), 1.0e-12, None)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return classification_metrics(truth, probabilities, classes=split.classes), values[context_size:, -1, : split.classes].detach().cpu()
    raw = prediction.auxiliaries["numeric_raw_prediction"]
    if raw.ndim == 3:
        raw = raw[0]
    predicted = raw[context_size:, -1].detach().cpu().numpy().astype(np.float64)
    return regression_metrics(truth, predicted, target_scale=split.target_scale), raw[context_size:, -1].detach().cpu()


def _tabur_metrics(model: torch.nn.Module, evidence: Any, split: Any, *, context_size: int) -> dict[str, float]:
    return _tabur_outputs(model, evidence, split, context_size=context_size)[0]


def _metric_names(task: str) -> tuple[str, ...]:
    return (
        ("normalized_nll", "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "roc_auc_ovr_macro")
        if task == "classification"
        else ("scaled_rmse", "scaled_mae", "r2", "rmse", "mae")
    )


def _mean_metrics(rows: list[dict[str, Any]], *, metric_names: tuple[str, ...]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row["metrics"][metric]) for row in rows]))
        for metric in metric_names
        if row_has_metric(rows, metric)
    }


def row_has_metric(rows: list[dict[str, Any]], metric: str) -> bool:
    return bool(rows) and all(
        isinstance(row.get("metrics"), dict)
        and metric in row["metrics"]
        and math.isfinite(float(row["metrics"][metric]))
        for row in rows
    )


def _summarize(
    baseline_records: list[dict[str, Any]],
    frozen_records: list[dict[str, Any]],
    *,
    dataset_ids: tuple[str, ...],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        task = str(next(row["task"] for row in baseline_records if row["dataset_id"] == dataset_id))
        metric_names = _metric_names(task)
        dataset_summary: dict[str, Any] = {"task": task, "primary_metric": "normalized_nll" if task == "classification" else "scaled_rmse", "contexts": {}}
        for context_size in QUERY_OPENML_K_GRID:
            base = [row for row in baseline_records if row["dataset_id"] == dataset_id and row["context_size"] == context_size and row["metrics"] is not None]
            frozen = [row for row in frozen_records if row["dataset_id"] == dataset_id and row["context_size"] == context_size]
            by_arm = {
                arm: [row for row in frozen if row["arm"] == arm and row["metrics"] is not None]
                for arm in ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
            }
            estimators = {
                estimator: [row for row in base if row["estimator"] == estimator]
                for estimator in ("linear", "mlp", "xgboost")
            }
            dataset_summary["contexts"][str(context_size)] = {
                "frozen": {arm: _mean_metrics(rows, metric_names=metric_names) for arm, rows in by_arm.items() if rows},
                "baselines": {estimator: _mean_metrics(rows, metric_names=metric_names) for estimator, rows in estimators.items() if rows},
                "eligible_baseline_rows": len(base),
            }
        summary[dataset_id] = dataset_summary
    for task in ("classification", "regression"):
        selected = [summary[dataset_id] for dataset_id in dataset_ids if summary[dataset_id]["task"] == task]
        primary = "normalized_nll" if task == "classification" else "scaled_rmse"
        macro: dict[str, Any] = {"dataset_count": len(selected), "primary_metric": primary}
        for arm in ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled", "linear", "mlp", "xgboost"):
            curves: list[float] = []
            for item in selected:
                vals = [item["contexts"][str(k)]["frozen" if arm.endswith("frozen") else "baselines"].get(arm, {}).get(primary) for k in QUERY_OPENML_K_GRID if k > 0 and (item["contexts"][str(k)]["frozen" if arm.endswith("frozen") else "baselines"].get(arm, {}).get(primary) is not None)]
                if vals:
                    curves.append(float(np.mean(vals)))
            macro[arm] = {"dataset_macro_mean_primary": float(np.mean(curves)) if curves else None}
        macro["datasets_with_finite_pretrained"] = sum(
            any(item["contexts"][str(k)]["frozen"].get("pretrained_frozen", {}).get(primary) is not None for k in QUERY_OPENML_K_GRID[1:])
            for item in selected
        )
        summary[f"_{task}_macro"] = macro
    return summary


def run_query_row_openml_frozen_transfer(
    *,
    panel_manifest: Path,
    checkpoint_paths: tuple[Path, ...],
    dataset_ids: tuple[str, ...] | None = None,
    checkpoint_seeds: tuple[int, ...] = ROOT_SEEDS,
    split_seeds: tuple[int, ...] = ROOT_SEEDS,
    query_limit: int = 256,
    device: str | torch.device = "cuda",
    openml_data_home: Path | None = None,
) -> dict[str, Any]:
    """Run the query-contract OpenML low-shot frozen-transfer comparison."""

    if not checkpoint_paths:
        raise ValueError("at least one TabUR checkpoint is required")
    if tuple(checkpoint_seeds) != ROOT_SEEDS or tuple(split_seeds) != ROOT_SEEDS:
        raise ValueError("query OpenML panel requires the preregistered three seeds")
    if query_limit != 256:
        raise ValueError("query OpenML panel requires query_limit=256")
    panel = load_query_openml_panel_manifest(panel_manifest)
    selected_ids = tuple(dataset_ids or panel["dataset_ids"])
    if not selected_ids or not set(selected_ids).issubset(set(panel["dataset_ids"])):
        raise ValueError("dataset_ids must be a non-empty subset of the query OpenML panel")
    resolved_device = resolve_device(str(device))
    started = time.monotonic()
    fetched = []
    for dataset_id in selected_ids:
        fetched.append(
            fetch_openml_new6_dataset(
                dataset_id,
                cache=True,
                data_home=openml_data_home,
            )
        )
    datasets = {item.spec.dataset_id: item.dataset for item in fetched}
    provenance = {item.spec.dataset_id: {"source_manifest": item.source_manifest, "source_manifest_sha256": item.source_manifest_sha256} for item in fetched}
    splits = {
        (dataset_id, split_seed): prepare_real_icl_split(
            datasets[dataset_id], split_seed=split_seed, query_limit=query_limit
        )
        for dataset_id in selected_ids
        for split_seed in split_seeds
    }
    split_manifests = {
        dataset_id: {str(seed): real_icl_split_manifest(splits[(dataset_id, seed)]) for seed in split_seeds}
        for dataset_id in selected_ids
    }
    baseline_records: list[dict[str, Any]] = []
    baseline_cache: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for dataset_id in selected_ids:
        for split_seed in split_seeds:
            split = splits[(dataset_id, split_seed)]
            for context_size in QUERY_OPENML_K_GRID:
                for estimator in ("linear", "mlp", "xgboost"):
                    fit = _fit_baseline(
                        split,
                        context_size=context_size,
                        estimator=estimator,
                        seed=split_seed,
                    )
                    baseline_cache[(dataset_id, split_seed, context_size, estimator)] = fit
                    baseline_records.append(
                        {
                            "dataset_id": dataset_id,
                            "task": split.dataset.task,
                            "split_seed": split_seed,
                            "context_size": context_size,
                            "train_rows_total": len(split.train_indices),
                            "query_rows": len(split.query_indices),
                            "predictor_count": split.features.shape[1],
                            "split_manifest": split_manifests[dataset_id][str(split_seed)],
                            "estimator": estimator,
                            "status": fit["status"],
                            "metrics": fit["metrics"],
                            "fit": fit["fit"],
                        }
                    )
    frozen_records: list[dict[str, Any]] = []
    checkpoint_results: list[dict[str, Any]] = []
    frozen_controls: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing TabUR checkpoint: {checkpoint_path}")
        pretrained, identity = _build_model_from_checkpoint(checkpoint_path, device=resolved_device)
        shuffled = _build_model_from_checkpoint(checkpoint_path, device=resolved_device)[0]
        random_model = _build_random_model(identity, seed=int(identity["metadata"]["root_seed"]) + 900_000, device=resolved_device)
        models = {
            "pretrained_frozen": pretrained,
            "random_init_frozen": random_model,
            "pretrained_shuffled": shuffled,
        }
        before = {arm: _state_hash(model) for arm, model in models.items()}
        substitution_ok = True
        for dataset_id in selected_ids:
            for split_seed in split_seeds:
                split = splits[(dataset_id, split_seed)]
                for context_size in QUERY_OPENML_K_GRID:
                    metadata = {
                        "dataset_id": dataset_id,
                        "task": split.dataset.task,
                        "checkpoint_seed": int(identity["metadata"]["root_seed"]),
                        "split_seed": split_seed,
                        "context_size": context_size,
                        "query_rows": len(split.query_indices),
                        "train_rows_total": len(split.train_indices),
                        "predictor_count": split.features.shape[1],
                        "split_manifest": split_manifests[dataset_id][str(split_seed)],
                    }
                    if context_size == 0:
                        metrics = _no_context_metrics(split)
                        for arm in models:
                            frozen_records.append(metadata | {"arm": arm, "status": "not_applicable", "metrics": metrics})
                        continue
                    if (
                        split.dataset.task == "classification"
                        and split.classes is not None
                        and context_size < split.classes
                    ):
                        # A categorical query-row terminal has no complete
                        # declared-domain support until every class is present
                        # in context.  This is an expected diagnostic gap, not
                        # a model failure; the preregistered primary curve
                        # excludes these K values.
                        for arm in models:
                            frozen_records.append(
                                metadata
                                | {
                                    "arm": arm,
                                    "status": "not_applicable",
                                    "metrics": None,
                                    "reason": "context_below_declared_class_count",
                                }
                            )
                        continue
                    evidence, truth = build_real_icl_episode(
                        split,
                        context_size=context_size,
                        query_indices=split.query_indices,
                        shuffled_context=False,
                        context_policy=LOW_SHOT_CONTEXT_POLICY,
                    )
                    shuffled_evidence, _ = build_real_icl_episode(
                        split,
                        context_size=context_size,
                        query_indices=split.query_indices,
                        shuffled_context=True,
                        context_policy=LOW_SHOT_CONTEXT_POLICY,
                    )
                    # Keep the reusable real-ICL episode builder but explicitly
                    # mark the resulting evidence as Axis-C query-row input.
                    evidence = replace(
                        evidence,
                        metadata={
                            **dict(evidence.metadata),
                            "model_contract": "tabu.query.row@0.1.0",
                            "query_family": "TabUR",
                        },
                    )
                    shuffled_evidence = replace(
                        shuffled_evidence,
                        metadata={
                            **dict(shuffled_evidence.metadata),
                            "model_contract": "tabu.query.row@0.1.0",
                            "query_family": "TabUR",
                        },
                    )
                    del truth
                    pretrained_metrics, pretrained_prediction = _tabur_outputs(pretrained, evidence.to(resolved_device), split, context_size=context_size)
                    random_metrics, _ = _tabur_outputs(random_model, evidence.to(resolved_device), split, context_size=context_size)
                    shuffled_metrics, _ = _tabur_outputs(shuffled, shuffled_evidence.to(resolved_device), split, context_size=context_size)
                    if dataset_id == selected_ids[0] and split_seed == split_seeds[0] and context_size == 1:
                        substituted_evidence, substituted_truth = build_real_icl_episode(
                            split,
                            context_size=context_size,
                            query_indices=split.query_indices,
                            shuffled_context=False,
                            context_policy=LOW_SHOT_CONTEXT_POLICY,
                        )
                        # Truth substitution happens only in the sidecar; Steps
                        # 1-4 receive identical evidence and therefore must be
                        # prediction-invariant.  Compare the public prediction
                        # slice, not merely the input tensor.
                        substituted_truth = replace(
                            substituted_truth,
                            target_values=substituted_truth.target_values + 123.456,
                        )
                        substituted_prediction = _tabur_outputs(
                            pretrained,
                            substituted_evidence.to(resolved_device),
                            split,
                            context_size=context_size,
                        )[1]
                        substitution_ok = substitution_ok and bool(
                            torch.equal(pretrained_prediction, substituted_prediction)
                        )
                    for arm, metrics in (
                        ("pretrained_frozen", pretrained_metrics),
                        ("random_init_frozen", random_metrics),
                        ("pretrained_shuffled", shuffled_metrics),
                    ):
                        frozen_records.append(metadata | {"arm": arm, "status": "passed", "metrics": metrics})
        after = {arm: _state_hash(model) for arm, model in models.items()}
        frozen_controls.append(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _file_sha256(checkpoint_path),
                "identity": str(checkpoint_path.with_suffix(".identity.json")),
                "identity_sha256": _file_sha256(checkpoint_path.with_suffix(".identity.json")),
                "rung": identity["metadata"]["rung"],
                "root_seed": identity["metadata"]["root_seed"],
                "parameter_hashes": {
                    arm: {"before": before[arm], "after": after[arm], "unchanged": before[arm] == after[arm]}
                    for arm in models
                },
                "optimizer_created": False,
                "parameter_update_attempted": False,
                "truth_substitution_prediction_unchanged": substitution_ok,
                "status": "passed" if all(before[arm] == after[arm] for arm in models) and substitution_ok else "failed",
            }
        )
        checkpoint_results.append(frozen_controls[-1])
    result = {
        "schema_version": QUERY_OPENML_RESULT_SCHEMA,
        "status": "passed" if all(item["status"] == "passed" for item in frozen_controls) else "failed",
        "evidence_status": "local_unissued",
        "claim_boundary": (
            "R4 v2 synthetic-pretrained TabUR query-row frozen OpenML diagnostic versus "
            "context-only Linear/MLP/XGBoost; no formal receipt, benchmark, SOTA, causal, "
            "or accepted capability claim"
        ),
        "contract_id": "tabu.query.row",
        "contract_version": "0.1.0",
        "model_spec_hash": QUERY_BASE_MODEL_SPEC_HASH,
        "profile_id": "supervised.label_broadcast.v1",
        "generator_id": "tabur.supervised-query-row-diverse-v2",
        "panel_id": QUERY_OPENML_PANEL_ID,
        "panel_manifest": panel,
        "datasets": list(selected_ids),
        "dataset_provenance": provenance,
        "dataset_hashes": {dataset_id: datasets[dataset_id].content_hash for dataset_id in selected_ids},
        "checkpoint_seeds": list(checkpoint_seeds),
        "split_seeds": list(split_seeds),
        "context_policy": LOW_SHOT_CONTEXT_POLICY,
        "context_sizes": list(QUERY_OPENML_K_GRID),
        "query_limit": query_limit,
        "query_chunk_rows": 64,
        "baseline_ids": list(BASELINE_IDS),
        "baseline_config": BASELINE_CONFIG,
        "baseline_config_hash": canonical_hash(BASELINE_CONFIG),
        "split_manifests": split_manifests,
        "checkpoints": checkpoint_results,
        "frozen_controls": frozen_controls,
        "baseline_records": baseline_records,
        "frozen_records": frozen_records,
        "summary": _summarize(baseline_records, frozen_records, dataset_ids=selected_ids),
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(resolved_device),
            "cuda": torch.version.cuda,
            "runtime_backend": os.environ.get("WEHUB_RUNTIME_BACKEND"),
            "runtime_image": os.environ.get("WEHUB_RUNTIME_IMAGE"),
        },
        "source_commit": _source_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "elapsed_seconds": time.monotonic() - started,
    }
    return result


__all__ = [
    "BASELINE_CONFIG",
    "BASELINE_IDS",
    "QUERY_OPENML_K_GRID",
    "QUERY_OPENML_PANEL_SCHEMA",
    "QUERY_OPENML_RESULT_SCHEMA",
    "load_query_openml_panel_manifest",
    "run_query_row_openml_frozen_transfer",
]
