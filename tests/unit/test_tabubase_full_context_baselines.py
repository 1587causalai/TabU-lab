from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import tabu_lab.experiments.tabubase_full_context_baselines as baseline_module
from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments.tabubase_full_context_baselines import (
    FullContextBaselineConfig,
    _aligned_probabilities,
    _fit_baseline,
    run_full_context_baselines,
)
from tabu_lab.experiments.tabubase_real_benchmark import RealDataset
from tabu_lab.experiments.tabubase_real_icl import (
    PreparedRealIclSplit,
    prepare_real_icl_split,
    real_icl_split_manifest,
)
from tabu_lab.experiments.tabubase_real_metrics import regression_metrics


def _split(
    *,
    task: str,
    features: np.ndarray,
    response: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    classes: int | None,
) -> PreparedRealIclSplit:
    dataset = RealDataset(
        dataset_id=f"{task}-fixture",
        task=task,  # type: ignore[arg-type]
        features=features,
        response=response,
        source="fixture",
    )
    target_scale = (
        1.0 if task == "classification" else max(float(response[train_indices].std()), 1.0e-6)
    )
    return PreparedRealIclSplit(
        dataset=dataset,
        split_seed=1729,
        features=features,
        response=response,
        train_indices=train_indices,
        query_indices=query_indices,
        context_order=train_indices[::-1].copy(),
        feature_indices=np.arange(features.shape[1], dtype=np.int64),
        classes=classes,
        target_scale=target_scale,
    )


def _install_fake_ml_stack(
    monkeypatch: pytest.MonkeyPatch,
    events: dict[str, list[Any]],
    *,
    regression_predictions: np.ndarray | None = None,
) -> None:
    sklearn = ModuleType("sklearn")
    sklearn.__version__ = "test-sklearn"  # type: ignore[attr-defined]
    neural_network = ModuleType("sklearn.neural_network")
    preprocessing = ModuleType("sklearn.preprocessing")
    xgboost = ModuleType("xgboost")
    xgboost.__version__ = "test-xgboost"  # type: ignore[attr-defined]

    class FakeStandardScaler:
        def fit(self, values: np.ndarray) -> FakeStandardScaler:
            materialized = np.asarray(values, dtype=np.float64).copy()
            events.setdefault("scaler_fit", []).append(materialized)
            self.mean_ = materialized.mean(axis=0)
            self.scale_ = materialized.std(axis=0)
            self.scale_[self.scale_ == 0.0] = 1.0
            return self

        def transform(self, values: np.ndarray) -> np.ndarray:
            materialized = np.asarray(values, dtype=np.float64).copy()
            events.setdefault("scaler_transform", []).append(materialized)
            return (materialized - self.mean_) / self.scale_

    class FakeMLPClassifier:
        def __init__(self, **kwargs: Any) -> None:
            events.setdefault("classifier_init", []).append(kwargs)
            self.n_iter_ = 7
            self.classes_ = np.asarray([0, 1, 2], dtype=np.int64)

        def fit(self, features: np.ndarray, target: np.ndarray) -> FakeMLPClassifier:
            events.setdefault("classifier_fit", []).append(
                (np.asarray(features).copy(), np.asarray(target).copy())
            )
            return self

        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            events.setdefault("classifier_predict", []).append(np.asarray(features).copy())
            return np.full((len(features), 3), 1.0 / 3.0, dtype=np.float64)

    class FakeMLPRegressor:
        def __init__(self, **kwargs: Any) -> None:
            events.setdefault("regressor_init", []).append(kwargs)
            self.n_iter_ = 9

        def fit(self, features: np.ndarray, target: np.ndarray) -> FakeMLPRegressor:
            events.setdefault("regressor_fit", []).append(
                (np.asarray(features).copy(), np.asarray(target).copy())
            )
            return self

        def predict(self, features: np.ndarray) -> np.ndarray:
            events.setdefault("regressor_predict", []).append(np.asarray(features).copy())
            assert regression_predictions is not None
            return regression_predictions.copy()

    class _UnusedXGBoost:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("the MLP tests must not instantiate XGBoost")

    neural_network.MLPClassifier = FakeMLPClassifier  # type: ignore[attr-defined]
    neural_network.MLPRegressor = FakeMLPRegressor  # type: ignore[attr-defined]
    preprocessing.StandardScaler = FakeStandardScaler  # type: ignore[attr-defined]
    xgboost.XGBClassifier = _UnusedXGBoost  # type: ignore[attr-defined]
    xgboost.XGBRegressor = _UnusedXGBoost  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.neural_network", neural_network)
    monkeypatch.setitem(sys.modules, "sklearn.preprocessing", preprocessing)
    monkeypatch.setitem(sys.modules, "xgboost", xgboost)


def test_probability_columns_are_aligned_and_require_complete_class_coverage() -> None:
    class PermutedClassifier:
        classes_ = np.asarray([2, 0, 1], dtype=np.int64)

        @staticmethod
        def predict_proba(_features: np.ndarray) -> np.ndarray:
            return np.asarray([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=np.float64)

    features = np.zeros((2, 1), dtype=np.float32)
    aligned = _aligned_probabilities(PermutedClassifier(), features, classes=3)
    np.testing.assert_allclose(aligned, [[0.2, 0.1, 0.7], [0.6, 0.3, 0.1]])

    class MissingClassClassifier(PermutedClassifier):
        classes_ = np.asarray([0, 2], dtype=np.int64)

        @staticmethod
        def predict_proba(_features: np.ndarray) -> np.ndarray:
            return np.asarray([[0.4, 0.6], [0.5, 0.5]], dtype=np.float64)

    with pytest.raises(RuntimeError, match="cover every declared class"):
        _aligned_probabilities(MissingClassClassifier(), features, classes=3)


def test_mlp_classifier_scaler_is_fit_on_train_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: dict[str, list[Any]] = {}
    _install_fake_ml_stack(monkeypatch, events)
    features = np.asarray(
        [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0], [1000.0, -900.0]],
        dtype=np.float32,
    )
    response = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    train = np.asarray([0, 1, 2, 3], dtype=np.int64)
    query = np.asarray([4], dtype=np.int64)
    split = _split(
        task="classification",
        features=features,
        response=response,
        train_indices=train,
        query_indices=query,
        classes=3,
    )

    metrics, fit = _fit_baseline(split, estimator="mlp", estimator_seed=2718)

    np.testing.assert_array_equal(events["scaler_fit"][0], features[train])
    assert len(events["scaler_fit"]) == 1
    np.testing.assert_array_equal(events["scaler_transform"][0], features[train])
    np.testing.assert_array_equal(events["scaler_transform"][1], features[query])
    scaled_train = events["classifier_fit"][0][0]
    np.testing.assert_allclose(scaled_train.mean(axis=0), np.zeros(2), atol=1.0e-12)
    np.testing.assert_allclose(scaled_train.std(axis=0), np.ones(2), atol=1.0e-12)
    assert fit["fit_rows"] == len(train)
    assert fit["query_rows"] == len(query)
    assert fit["predictor_scaler"]["fit_scope"] == "train_indices_only"
    assert fit["predictor_scaler"]["fit_rows"] == len(train)
    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "macro_f1",
        "normalized_nll",
        "roc_auc_ovr_macro",
    }


def test_mlp_regression_standardizes_train_target_and_inverse_transforms_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standardized_predictions = np.asarray([0.0, 1.0], dtype=np.float64)
    events: dict[str, list[Any]] = {}
    _install_fake_ml_stack(
        monkeypatch,
        events,
        regression_predictions=standardized_predictions,
    )
    features = np.asarray([[0.0], [1.0], [2.0], [3.0], [500.0], [700.0]], dtype=np.float32)
    response = np.asarray([10.0, 20.0, 30.0, 40.0, 25.0, 35.0], dtype=np.float32)
    train = np.asarray([0, 1, 2, 3], dtype=np.int64)
    query = np.asarray([4, 5], dtype=np.int64)
    split = _split(
        task="regression",
        features=features,
        response=response,
        train_indices=train,
        query_indices=query,
        classes=None,
    )

    metrics, fit = _fit_baseline(split, estimator="mlp", estimator_seed=31415)

    fitted_target = events["regressor_fit"][0][1]
    assert float(fitted_target.mean()) == pytest.approx(0.0, abs=1.0e-7)
    assert float(fitted_target.std()) == pytest.approx(1.0, abs=1.0e-7)
    target_mean = float(response[train].mean())
    target_scale = float(response[train].std())
    expected_predictions = standardized_predictions * target_scale + target_mean
    expected_metrics = regression_metrics(
        response[query],
        expected_predictions,
        target_scale=split.target_scale,
    )
    assert metrics == pytest.approx(expected_metrics)
    assert fit["target_scaler"] == {
        "fit_scope": "train_indices_only",
        "fit_rows": len(train),
        "mean": target_mean,
        "scale": target_scale,
        "prediction_inverse_transform": "prediction * scale + mean",
    }


def test_runner_reuses_frozen_split_hashes_and_does_not_truncate_query(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(19)
    rows = 2_000
    dataset = RealDataset(
        dataset_id="large-regression-fixture",
        task="regression",
        features=rng.normal(size=(rows, 70)).astype(np.float32),
        response=rng.normal(size=rows).astype(np.float32),
        source="fixture",
    )
    expected_split = prepare_real_icl_split(dataset, split_seed=1729, query_limit=None)
    expected_manifest = real_icl_split_manifest(expected_split)
    observed_query_rows: list[int] = []

    def fake_fit(
        split: PreparedRealIclSplit, *, estimator: str, estimator_seed: int
    ) -> tuple[dict[str, float], dict[str, Any]]:
        assert estimator in {"xgboost", "mlp"}
        assert estimator_seed == 1729
        observed_query_rows.append(len(split.query_indices))
        return (
            {
                "rmse": 1.0,
                "mae": 0.8,
                "scaled_rmse": 1.1,
                "scaled_mae": 0.9,
                "r2": 0.0,
            },
            {"fit_rows": len(split.train_indices), "query_rows": len(split.query_indices)},
        )

    monkeypatch.setattr(baseline_module, "_fit_baseline", fake_fit)
    monkeypatch.setattr(baseline_module, "_source_tree_hash", lambda: "source-fixture")
    output_path = tmp_path / "baseline.json"
    receipt = run_full_context_baselines(
        FullContextBaselineConfig(
            output_path=output_path,
            dataset_ids=(dataset.dataset_id,),
            split_seeds=(1729,),
        ),
        dataset_loader=lambda _dataset_id: dataset,
    )

    assert len(expected_split.query_indices) == 600
    assert observed_query_rows == [600, 600]
    assert receipt["query_policy"] == "all_heldout_rows"
    assert receipt["split_manifests"][dataset.dataset_id]["1729"] == expected_manifest
    expected_hash = canonical_hash(expected_manifest)
    assert receipt["split_manifest_sha256"][dataset.dataset_id]["1729"] == expected_hash
    for record in receipt["records"]:
        assert record["train_rows"] == len(expected_split.train_indices)
        assert record["query_rows"] == len(expected_split.query_indices)
        assert record["predictor_count"] == 63
        assert record["split_manifest"] == expected_manifest
        assert record["split_manifest_sha256"] == expected_hash
        assert record["split_manifest"]["train_indices_sha256"] == canonical_hash(
            expected_split.train_indices.tolist()
        )
        assert record["split_manifest"]["query_indices_sha256"] == canonical_hash(
            expected_split.query_indices.tolist()
        )
        assert record["split_manifest"]["feature_indices_sha256"] == canonical_hash(
            expected_split.feature_indices.tolist()
        )
