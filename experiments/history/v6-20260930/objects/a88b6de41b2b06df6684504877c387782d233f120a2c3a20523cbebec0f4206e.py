"""Shared query-row transfer primitives used by the full-context evaluator.

This module deliberately contains no context-size policy.  The public real-data
protocol lives in :mod:`query_row_openml_full_context`; keeping checkpoint
loading and matched baselines here prevents a diagnostic K-grid runner from
becoming an accidental default.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_identity import require_query_row_readout_identity
from .query_row_pretraining import (
    load_query_row_pretrain_checkpoint,
    read_query_row_pretrain_checkpoint_identity,
)
from .tabubase_real_metrics import classification_metrics, regression_metrics

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


def source_commit() -> str | None:
    explicit = os.environ.get("TABU_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() or None


def aligned_probabilities(model: Any, features: np.ndarray, *, classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    labels = np.asarray(model.classes_)
    try:
        labels = labels.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("classifier classes must be integer labels") from exc
    if raw.shape != (len(features), len(labels)) or not np.array_equal(
        np.sort(labels), np.arange(classes)
    ):
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
        probabilities = np.full((len(truth), split.classes), 1.0 / split.classes)
        return classification_metrics(truth, probabilities, classes=split.classes)
    return regression_metrics(
        truth, np.zeros(len(truth), dtype=np.float64), target_scale=split.target_scale
    )


def fit_baseline(split: Any, *, context_size: int, estimator: str, seed: int) -> dict[str, Any]:
    truth = split.response[split.query_indices]
    if context_size == 0:
        return {
            "status": "not_applicable",
            "metrics": _no_context_metrics(split),
            "fit": {"fit_rows": 0},
        }
    if (
        split.dataset.task == "classification"
        and split.classes is not None
        and context_size < split.classes
    ):
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
                model = LogisticRegression(
                    **BASELINE_CONFIG["logistic_regression"], random_state=seed
                )
                train_features, query_features = x_train, x_query
                fit["estimator_config"] = BASELINE_CONFIG["logistic_regression"] | {
                    "random_state": seed
                }
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features, query_features = (
                    scaler.transform(x_train),
                    scaler.transform(x_query),
                )
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
                    **BASELINE_CONFIG["xgboost"], random_state=seed, eval_metric="logloss"
                )
                train_features, query_features = x_train, x_query
                fit["estimator_config"] = BASELINE_CONFIG["xgboost"] | {
                    "random_state": seed, "eval_metric": "logloss"
                }
            else:
                raise ValueError(f"unknown OpenML baseline: {estimator}")
            model.fit(train_features, y_train)
            metrics = classification_metrics(
                truth,
                aligned_probabilities(model, query_features, classes=split.classes),
                classes=split.classes,
            )
        else:
            if estimator == "linear":
                model = Ridge(alpha=float(BASELINE_CONFIG["linear_regression"]["ridge"]))
                train_features, query_features, train_target = x_train, x_query, y_train
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features, query_features = (
                    scaler.transform(x_train),
                    scaler.transform(x_query),
                )
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
        fit["converged_before_max_iter"] = int(model.n_iter_) < int(
            BASELINE_CONFIG["mlp"]["max_iter"]
        )
    return {"status": "passed", "metrics": metrics, "fit": fit}


def build_model_from_checkpoint(
    path: Path, *, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    identity = read_query_row_pretrain_checkpoint_identity(path)
    model_identity = identity["model_identity"]
    readout = require_query_row_readout_identity(model_identity)
    config = _reference_config_from_identity(model_identity)
    model = build_model(
        "tabu.query.row",
        config=config,
        profile=str(model_identity["profile_id"]),
        row_token_count=int(model_identity["row_token_count"]),
        row_readout_mode=str(readout["mode"]),
        anchored_gamma_initial=float(readout["anchored_gamma_initial"]),
    ).to(device)
    load_query_row_pretrain_checkpoint(model, path)
    model.eval()
    model.requires_grad_(False)
    return model, identity


def build_random_model(
    identity: dict[str, Any], *, seed: int, device: torch.device
) -> torch.nn.Module:
    model_identity = identity["model_identity"]
    readout = require_query_row_readout_identity(model_identity)
    config = _reference_config_from_identity(model_identity)
    torch.manual_seed(seed)
    model = build_model(
        "tabu.query.row",
        config=config,
        profile=str(model_identity["profile_id"]),
        row_token_count=int(model_identity["row_token_count"]),
        row_readout_mode=str(readout["mode"]),
        anchored_gamma_initial=float(readout["anchored_gamma_initial"]),
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _reference_config_from_identity(model_identity: dict[str, Any]) -> ReferenceConfig:
    reference = model_identity.get("reference_config")
    if not isinstance(reference, dict):
        raise ValueError("checkpoint model_identity.reference_config is required")
    expected = set(ReferenceConfig.__dataclass_fields__)
    missing = sorted(expected - set(reference))
    unexpected = sorted(set(reference) - expected)
    if missing or unexpected:
        raise ValueError(
            "checkpoint reference_config must be exact; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return ReferenceConfig(**reference)


__all__ = [
    "BASELINE_CONFIG",
    "BASELINE_IDS",
    "QUERY_BASE_MODEL_SPEC_HASH",
    "aligned_probabilities",
    "build_model_from_checkpoint",
    "build_random_model",
    "fit_baseline",
    "source_commit",
]
