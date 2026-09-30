"""Exact-split MLP/XGBoost baselines for TabUBase full-context evaluation."""

from __future__ import annotations

import json
import os
import platform
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tabu_lab.contracts import canonical_hash

from .tabubase_openml_cached import (
    CACHED_OPENML_BY_ID,
    is_cached_openml_panel_manifest,
    load_cached_openml_panel_manifest,
)
from .tabubase_openml_new6 import (
    OPENML_NEW6_BY_ID,
    FetchOpenML,
    load_openml_new6_panel_manifest,
)
from .tabubase_real_benchmark import RealDataset, _source_tree_hash
from .tabubase_real_icl import (
    DEFAULT_REAL_ICL_DATASETS,
    FULL_CONTEXT_POLICY,
    RealIclConfig,
    _load_real_icl_panel,
    _source_commit,
    prepare_real_icl_split,
    real_icl_split_manifest,
)
from .tabubase_real_metrics import classification_metrics, regression_metrics
from .tabubase_scale import ROOT_SEEDS, _sha256_file

BASELINE_SCHEMA = "tabu.transfer-base-real-full-context-baselines-local-unissued.v1"
BASELINE_ESTIMATORS = ("xgboost", "mlp")
XGBOOST_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 8,
    "tree_method": "hist",
}
MLP_CONFIG: dict[str, Any] = {
    "hidden_layer_sizes": [64, 64],
    "activation": "relu",
    "solver": "adam",
    "alpha": 1.0e-4,
    "learning_rate_init": 1.0e-3,
    "max_iter": 500,
    "early_stopping": False,
    "shuffle": True,
    "tol": 1.0e-4,
}


@dataclass(frozen=True, slots=True)
class FullContextBaselineConfig:
    output_path: Path
    dataset_ids: tuple[str, ...] = DEFAULT_REAL_ICL_DATASETS
    split_seeds: tuple[int, ...] = ROOT_SEEDS
    panel_manifest_path: Path | None = None
    openml_cache: bool = True
    openml_data_home: Path | None = None

    def validate(self) -> FullContextBaselineConfig:
        if not self.dataset_ids or len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError("baseline datasets must be non-empty and unique")
        if not self.split_seeds or len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("baseline split seeds must be non-empty and unique")
        if self.panel_manifest_path is not None:
            if is_cached_openml_panel_manifest(self.panel_manifest_path):
                panel = load_cached_openml_panel_manifest(self.panel_manifest_path)
                allowed_ids = set(CACHED_OPENML_BY_ID)
                panel_name = "cached OpenML"
            else:
                panel = load_openml_new6_panel_manifest(self.panel_manifest_path)
                allowed_ids = set(OPENML_NEW6_BY_ID)
                panel_name = "OpenML new6"
            if panel.context_policy != FULL_CONTEXT_POLICY:
                raise ValueError("full-context baselines require a full_train panel manifest")
            if (
                set(self.dataset_ids) - set(panel.dataset_ids)
                or set(self.dataset_ids) - allowed_ids
            ):
                raise ValueError("requested baseline dataset is absent from the panel manifest")
            if self.split_seeds != ROOT_SEEDS:
                raise ValueError(f"{panel_name} baselines require the preregistered split seeds")
            if not self.openml_cache:
                raise ValueError(f"{panel_name} baselines require cache=true")
            if (
                not is_cached_openml_panel_manifest(self.panel_manifest_path)
                and self.openml_data_home is not None
                and not self.openml_data_home.is_dir()
            ):
                raise ValueError("OpenML new6 data_home must be an existing directory")
        return self


def _aligned_probabilities(model: Any, features: np.ndarray, *, classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    declared_classes = np.asarray(model.classes_)
    try:
        model_classes = declared_classes.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("classifier classes_ must contain integer response labels") from exc
    if not np.array_equal(declared_classes, model_classes):
        raise RuntimeError("classifier classes_ must contain integer response labels")
    if raw.ndim != 2 or raw.shape != (len(features), len(model_classes)):
        raise RuntimeError("classifier probability columns do not match model.classes_")
    if not np.array_equal(np.sort(model_classes), np.arange(classes)):
        raise RuntimeError("classifier probability columns do not cover every declared class")
    if not np.isfinite(raw).all() or bool((raw < 0.0).any()):
        raise RuntimeError("classifier emitted invalid probabilities")
    if bool((raw.sum(axis=1) <= 0.0).any()):
        raise RuntimeError("classifier emitted a probability row with no mass")
    probabilities = np.zeros((len(features), classes), dtype=np.float64)
    for column, label in enumerate(model_classes.tolist()):
        if label < 0 or label >= classes:
            raise RuntimeError("classifier emitted a probability for an unknown class")
        probabilities[:, label] = raw[:, column]
    return probabilities


def _fit_baseline(
    split: Any,
    *,
    estimator: str,
    estimator_seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    try:
        import sklearn
        import xgboost as xgb
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - optional runtime dependencies
        raise RuntimeError("full-context baselines require scikit-learn and xgboost") from exc

    x_train = np.asarray(split.features[split.train_indices], dtype=np.float32)
    x_query = np.asarray(split.features[split.query_indices], dtype=np.float32)
    y_train = np.asarray(split.response[split.train_indices])
    y_query = np.asarray(split.response[split.query_indices])
    if len(x_train) != len(split.train_indices) or len(x_query) != len(split.query_indices):
        raise RuntimeError("baseline row materialization drifted from the frozen split")

    fit_details: dict[str, Any] = {
        "estimator": estimator,
        "estimator_seed": estimator_seed,
        "fit_rows": len(x_train),
        "query_rows": len(x_query),
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if split.dataset.task == "classification":
            assert split.classes is not None
            if estimator == "xgboost":
                model = xgb.XGBClassifier(
                    **XGBOOST_CONFIG,
                    random_state=estimator_seed,
                    eval_metric="logloss",
                )
                train_features, query_features = x_train, x_query
                fit_details["preprocessing"] = "none_selected_predictors_only"
                fit_details["estimator_config"] = XGBOOST_CONFIG | {
                    "random_state": estimator_seed,
                    "eval_metric": "logloss",
                }
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features = scaler.transform(x_train)
                query_features = scaler.transform(x_query)
                model = MLPClassifier(
                    hidden_layer_sizes=tuple(MLP_CONFIG["hidden_layer_sizes"]),
                    activation=str(MLP_CONFIG["activation"]),
                    solver=str(MLP_CONFIG["solver"]),
                    alpha=float(MLP_CONFIG["alpha"]),
                    batch_size=min(64, len(y_train)),
                    learning_rate_init=float(MLP_CONFIG["learning_rate_init"]),
                    max_iter=int(MLP_CONFIG["max_iter"]),
                    early_stopping=bool(MLP_CONFIG["early_stopping"]),
                    shuffle=bool(MLP_CONFIG["shuffle"]),
                    tol=float(MLP_CONFIG["tol"]),
                    random_state=estimator_seed,
                )
                fit_details["preprocessing"] = "train_only_standard_scaler"
                fit_details["predictor_scaler"] = {
                    "fit_scope": "train_indices_only",
                    "fit_rows": len(x_train),
                    "mean_sha256": canonical_hash(np.asarray(scaler.mean_).tolist()),
                    "scale_sha256": canonical_hash(np.asarray(scaler.scale_).tolist()),
                }
                fit_details["estimator_config"] = MLP_CONFIG | {
                    "batch_size": min(64, len(y_train)),
                    "random_state": estimator_seed,
                }
            else:
                raise ValueError(f"unknown baseline estimator: {estimator}")
            model.fit(train_features, y_train)
            probabilities = _aligned_probabilities(
                model,
                query_features,
                classes=split.classes,
            )
            metrics = classification_metrics(y_query, probabilities, classes=split.classes)
        else:
            if estimator == "xgboost":
                model = xgb.XGBRegressor(
                    **XGBOOST_CONFIG,
                    random_state=estimator_seed,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                )
                train_features, query_features = x_train, x_query
                train_target = y_train
                fit_details["preprocessing"] = "none_selected_predictors_only"
                fit_details["estimator_config"] = XGBOOST_CONFIG | {
                    "random_state": estimator_seed,
                    "objective": "reg:squarederror",
                    "eval_metric": "rmse",
                }
                target_mean, target_scale = 0.0, 1.0
            elif estimator == "mlp":
                scaler = StandardScaler().fit(x_train)
                train_features = scaler.transform(x_train)
                query_features = scaler.transform(x_query)
                target_mean = float(y_train.mean())
                target_scale = max(float(y_train.std()), 1.0e-6)
                train_target = (y_train - target_mean) / target_scale
                model = MLPRegressor(
                    hidden_layer_sizes=tuple(MLP_CONFIG["hidden_layer_sizes"]),
                    activation=str(MLP_CONFIG["activation"]),
                    solver=str(MLP_CONFIG["solver"]),
                    alpha=float(MLP_CONFIG["alpha"]),
                    batch_size=min(64, len(y_train)),
                    learning_rate_init=float(MLP_CONFIG["learning_rate_init"]),
                    max_iter=int(MLP_CONFIG["max_iter"]),
                    early_stopping=bool(MLP_CONFIG["early_stopping"]),
                    shuffle=bool(MLP_CONFIG["shuffle"]),
                    tol=float(MLP_CONFIG["tol"]),
                    random_state=estimator_seed,
                )
                fit_details["preprocessing"] = "train_only_predictor_and_target_standard_scalers"
                fit_details["predictor_scaler"] = {
                    "fit_scope": "train_indices_only",
                    "fit_rows": len(x_train),
                    "mean_sha256": canonical_hash(np.asarray(scaler.mean_).tolist()),
                    "scale_sha256": canonical_hash(np.asarray(scaler.scale_).tolist()),
                }
                fit_details["target_scaler"] = {
                    "fit_scope": "train_indices_only",
                    "fit_rows": len(y_train),
                    "mean": target_mean,
                    "scale": target_scale,
                    "prediction_inverse_transform": "prediction * scale + mean",
                }
                fit_details["estimator_config"] = MLP_CONFIG | {
                    "batch_size": min(64, len(y_train)),
                    "random_state": estimator_seed,
                }
            else:
                raise ValueError(f"unknown baseline estimator: {estimator}")
            model.fit(train_features, train_target)
            predicted = np.asarray(model.predict(query_features), dtype=np.float64)
            if estimator == "mlp":
                predicted = predicted * target_scale + target_mean
            metrics = regression_metrics(y_query, predicted, target_scale=split.target_scale)

    fit_details["warnings"] = [str(item.message) for item in caught]
    if estimator == "mlp":
        fit_details["n_iter"] = int(model.n_iter_)
        fit_details["converged_before_max_iter"] = int(model.n_iter_) < int(MLP_CONFIG["max_iter"])
    return metrics, fit_details


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset_id in sorted({str(row["dataset_id"]) for row in records}):
        dataset_rows = [row for row in records if row["dataset_id"] == dataset_id]
        task = str(dataset_rows[0]["task"])
        estimator_summaries: dict[str, Any] = {}
        for estimator in BASELINE_ESTIMATORS:
            selected = [row for row in dataset_rows if row["estimator"] == estimator]
            metric_names = sorted(selected[0]["metrics"])
            estimator_summaries[estimator] = {
                "split_count": len(selected),
                "metrics": {
                    metric: {
                        "mean": float(np.mean([row["metrics"][metric] for row in selected])),
                        "min": float(np.min([row["metrics"][metric] for row in selected])),
                        "max": float(np.max([row["metrics"][metric] for row in selected])),
                    }
                    for metric in metric_names
                },
            }
        summary[dataset_id] = {"task": task, "estimators": estimator_summaries}
    return summary


def run_full_context_baselines(
    config: FullContextBaselineConfig,
    *,
    openml_fetcher: FetchOpenML | None = None,
    openml_sklearn_version: str | None = None,
    dataset_loader: Callable[[str], RealDataset] | None = None,
) -> dict[str, Any]:
    """Fit baselines on every frozen train row and score every held-out row."""

    config.validate()
    started = time.monotonic()
    loader_config = RealIclConfig(
        checkpoint_root=Path("."),
        output_path=config.output_path,
        dataset_ids=config.dataset_ids,
        checkpoint_seeds=(ROOT_SEEDS[0],),
        split_seeds=config.split_seeds,
        context_policy=FULL_CONTEXT_POLICY,
        query_limit=None,
        panel_manifest_path=config.panel_manifest_path,
        openml_cache=config.openml_cache,
        openml_data_home=config.openml_data_home,
    )
    if dataset_loader is not None:
        if config.panel_manifest_path is not None:
            raise ValueError("an injected dataset loader cannot replace an OpenML panel")
        datasets = {dataset_id: dataset_loader(dataset_id) for dataset_id in config.dataset_ids}
        panel_manifest = None
        dataset_provenance = {
            dataset_id: {
                "provider": "injected_dataset_loader",
                "real_dataset_content_sha256": dataset.content_hash,
            }
            for dataset_id, dataset in datasets.items()
        }
    else:
        datasets, panel_manifest, dataset_provenance = _load_real_icl_panel(
            loader_config,
            openml_fetcher=openml_fetcher,
            openml_sklearn_version=openml_sklearn_version,
        )
    splits = {
        (dataset_id, split_seed): prepare_real_icl_split(
            dataset,
            split_seed=split_seed,
            query_limit=None,
        )
        for dataset_id, dataset in datasets.items()
        for split_seed in config.split_seeds
    }
    split_manifests = {
        dataset_id: {
            str(split_seed): real_icl_split_manifest(splits[(dataset_id, split_seed)])
            for split_seed in config.split_seeds
        }
        for dataset_id in config.dataset_ids
    }
    records: list[dict[str, Any]] = []
    for dataset_id in config.dataset_ids:
        for split_seed in config.split_seeds:
            split = splits[(dataset_id, split_seed)]
            split_manifest = split_manifests[dataset_id][str(split_seed)]
            for estimator in BASELINE_ESTIMATORS:
                metrics, fit = _fit_baseline(
                    split,
                    estimator=estimator,
                    estimator_seed=split_seed,
                )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "task": split.dataset.task,
                        "classes": split.classes,
                        "split_seed": split_seed,
                        "estimator_seed": split_seed,
                        "estimator": estimator,
                        "train_rows": len(split.train_indices),
                        "query_rows": len(split.query_indices),
                        "predictor_count": split.features.shape[1],
                        "split_manifest": split_manifest,
                        "split_manifest_sha256": canonical_hash(split_manifest),
                        "metrics": metrics,
                        "fit": fit,
                    }
                )

    receipt: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA,
        "status": "local_unissued",
        "context_policy": FULL_CONTEXT_POLICY,
        "train_policy": "all_train_partition_rows",
        "query_policy": "all_heldout_rows",
        "datasets": list(config.dataset_ids),
        "dataset_hashes": {key: value.content_hash for key, value in datasets.items()},
        "dataset_provenance": dataset_provenance,
        "panel_manifest": panel_manifest,
        "openml_data_home": (
            str(config.openml_data_home.expanduser().resolve())
            if config.openml_data_home is not None
            else None
        ),
        "split_seeds": list(config.split_seeds),
        "estimator_seed_policy": "estimator_seed_equals_split_seed",
        "estimators": list(BASELINE_ESTIMATORS),
        "estimator_contracts": {
            "xgboost": XGBOOST_CONFIG,
            "mlp": MLP_CONFIG,
        },
        "split_manifests": split_manifests,
        "split_manifest_sha256": {
            dataset_id: {
                str(split_seed): canonical_hash(split_manifests[dataset_id][str(split_seed)])
                for split_seed in config.split_seeds
            }
            for dataset_id in config.dataset_ids
        },
        "records": records,
        "summary": _summarize(records),
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "git_commit": _source_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "source_status": "local_unissued_mirrored_worktree_not_clean_commit",
        "claim_boundary": (
            "fixed-hyperparameter classical baselines on the exact frozen full-context split; "
            "no validation tuning, benchmark, SOTA, or formal receipt claim"
        ),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt | {
        "result_path": str(config.output_path),
        "result_sha256": _sha256_file(config.output_path),
    }


__all__ = [
    "BASELINE_ESTIMATORS",
    "BASELINE_SCHEMA",
    "MLP_CONFIG",
    "XGBOOST_CONFIG",
    "FullContextBaselineConfig",
    "run_full_context_baselines",
]
