"""Truth-isolated model protocol and deterministic baseline adapters."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import (
    AdapterKind,
    AdapterSpec,
    BaselineSpec,
    BlindExample,
    FailureCategory,
    PreparedExample,
    RawPrediction,
    ScenarioSpec,
    TargetKind,
)


@runtime_checkable
class ModelAdapter(Protocol):
    """Minimal evaluation adapter; test examples are structurally truth-free."""

    @property
    def spec(self) -> AdapterSpec: ...

    def predict(
        self,
        *,
        scenario: ScenarioSpec,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]: ...


def _numeric_vector(features: dict[str, object], keys: tuple[str, ...]) -> np.ndarray:
    values: list[float] = []
    for key in keys:
        value = features.get(key)
        values.append(float(value) if isinstance(value, int | float) else 0.0)
    return np.asarray(values, dtype=np.float64)


def _label(value: object) -> str:
    return str(value)


def _abstain(example: BlindExample, code: str) -> RawPrediction:
    return RawPrediction(
        example_id=example.example_id,
        abstained=True,
        failure_category=FailureCategory.EVALUATOR,
        failure_code=code,
    )


class BaselineAdapter:
    """Pure NumPy deterministic implementations of the frozen v0 baselines."""

    def __init__(
        self,
        baseline: BaselineSpec | Mapping[str, object],
        train_only_fitted_state: Mapping[str, object] | None = None,
    ) -> None:
        baseline = BaselineSpec.model_validate(baseline)
        self.baseline = baseline
        self._categorical_domains: dict[str, tuple[str, ...]] = {}
        if train_only_fitted_state is not None:
            raw_families = train_only_fitted_state.get("families")
            if not isinstance(raw_families, Mapping):
                raise ValueError("baseline fitted state has no family map")
            for family_id, raw_family in raw_families.items():
                if not isinstance(family_id, str) or not isinstance(raw_family, Mapping):
                    raise ValueError("baseline fitted-state family map is malformed")
                if raw_family.get("kind") != "categorical":
                    continue
                raw_domain = raw_family.get("domain")
                if (
                    not isinstance(raw_domain, list)
                    or not raw_domain
                    or any(not isinstance(value, str) or not value for value in raw_domain)
                    or len(raw_domain) != len(set(raw_domain))
                ):
                    raise ValueError("baseline categorical fitted-state domain is malformed")
                self._categorical_domains[family_id] = tuple(sorted(raw_domain))
        iterative_defaults = {
            "ohe_logistic": 250,
            "user_item_bias": 10,
        }
        fit_iterations = int(
            baseline.hyperparameters.get("iterations", iterative_defaults.get(baseline.family, 0))
        )
        self._spec = AdapterSpec(
            adapter_id=baseline.baseline_id,
            adapter_version="1.0.0",
            kind=AdapterKind.BASELINE,
            fit_iterations=fit_iterations,
            device_class="single_device",
            deterministic=True,
            baseline_family=baseline.family,
        )

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    def predict(
        self,
        *,
        scenario: ScenarioSpec,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del seed  # every baseline is deliberately deterministic
        family = self.baseline.family
        if family in {"majority", "global_mode"}:
            return self._mode(fit_examples, examples)
        if family in {"mean", "global_mean"}:
            return self._mean(fit_examples, examples)
        if family == "mean_mode":
            return self._mean_mode(fit_examples, examples)
        if family == "numeric_knn":
            return self._knn(fit_examples, examples)
        if family == "standardized_ridge":
            return self._ridge(fit_examples, examples)
        if family == "ohe_logistic":
            return self._logistic(fit_examples, examples)
        if family == "neighbor_mode":
            return self._neighbor_mode(fit_examples, examples)
        if family == "user_item_bias":
            return self._user_item_bias(fit_examples, examples)
        raise ValueError(f"unsupported baseline family: {family}")

    def _mean(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        by_family: defaultdict[str, list[float]] = defaultdict(list)
        for item in fit:
            if item.target_kind is TargetKind.NUMERIC:
                by_family[item.target_family].append(float(item.target))
        predictions: list[RawPrediction] = []
        for item in examples:
            if item.target_kind is not TargetKind.NUMERIC:
                predictions.append(_abstain(item, "baseline_not_applicable_to_target_kind"))
                continue
            targets = by_family.get(item.target_family, [])
            if not targets:
                predictions.append(_abstain(item, "no_numeric_fit_targets_for_family"))
                continue
            predictions.append(
                RawPrediction(
                    example_id=item.example_id,
                    value=sum(targets) / len(targets),
                )
            )
        return tuple(predictions)

    def _mode(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for item in fit:
            if item.target_kind is TargetKind.CATEGORICAL:
                by_family[item.target_family][_label(item.target)] += 1
        predictions: list[RawPrediction] = []
        for item in examples:
            if item.target_kind is not TargetKind.CATEGORICAL:
                predictions.append(_abstain(item, "baseline_not_applicable_to_target_kind"))
                continue
            counts = by_family.get(item.target_family)
            if not counts:
                predictions.append(_abstain(item, "no_categorical_fit_targets_for_family"))
                continue
            labels = self._categorical_domains.get(
                item.target_family,
                tuple(sorted(counts)),
            )
            if set(counts) - set(labels):
                raise ValueError("baseline fit target lies outside the train-only domain")
            alpha = float(self.baseline.hyperparameters.get("laplace_alpha", 0.0))
            if not np.isfinite(alpha) or alpha < 0.0:
                raise ValueError("laplace_alpha must be finite and non-negative")
            total = sum(counts.values()) + alpha * len(labels)
            mode = min(labels, key=lambda label: (-counts[label], label))
            predictions.append(
                RawPrediction(
                    example_id=item.example_id,
                    value=mode,
                    probabilities={
                        label: (counts[label] + alpha) / total for label in labels
                    },
                )
            )
        return tuple(predictions)

    def _mean_mode(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        numeric = self._mean(fit, examples)
        categorical = self._mode(fit, examples)
        return tuple(
            num if example.target_kind is TargetKind.NUMERIC else cat
            for example, num, cat in zip(examples, numeric, categorical, strict=True)
        )

    def _knn(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        by_family: defaultdict[str, list[PreparedExample]] = defaultdict(list)
        for item in fit:
            if item.target_kind is TargetKind.NUMERIC:
                by_family[item.target_family].append(item)
        categorical = self._mode(fit, examples)
        predictions: list[RawPrediction] = []
        for example, categorical_prediction in zip(examples, categorical, strict=True):
            if example.target_kind is not TargetKind.NUMERIC:
                predictions.append(categorical_prediction)
                continue
            numeric_fit = by_family.get(example.target_family, [])
            if not numeric_fit:
                predictions.append(_abstain(example, "no_numeric_fit_targets_for_family"))
                continue
            keys = tuple(
                sorted(
                    {
                        key
                        for item in numeric_fit
                        for key, value in item.features.items()
                        if isinstance(value, int | float)
                    }
                )
            )
            if not keys:
                value = sum(float(item.target) for item in numeric_fit) / len(numeric_fit)
                predictions.append(RawPrediction(example_id=example.example_id, value=value))
                continue
            matrix = np.stack([_numeric_vector(item.features, keys) for item in numeric_fit])
            means = matrix.mean(axis=0)
            scales = matrix.std(axis=0)
            scales[scales == 0] = 1.0
            matrix = (matrix - means) / scales
            targets = np.asarray([float(item.target) for item in numeric_fit], dtype=np.float64)
            k = min(int(self.baseline.hyperparameters.get("k", 5)), len(numeric_fit))
            vector = (_numeric_vector(example.features, keys) - means) / scales
            distances = ((matrix - vector) ** 2).sum(axis=1)
            neighbors = np.argsort(distances, kind="stable")[:k]
            predictions.append(
                RawPrediction(example_id=example.example_id, value=float(targets[neighbors].mean()))
            )
        return tuple(predictions)

    @staticmethod
    def _numeric_design(
        fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = tuple(
            sorted(
                {
                    key
                    for item in fit
                    for key, value in item.features.items()
                    if isinstance(value, int | float)
                }
            )
        )
        if not keys:
            fit_matrix = np.zeros((len(fit), 0), dtype=np.float64)
            eval_matrix = np.zeros((len(examples), 0), dtype=np.float64)
        else:
            fit_matrix = np.stack([_numeric_vector(item.features, keys) for item in fit])
            eval_matrix = np.stack([_numeric_vector(item.features, keys) for item in examples])
        means = fit_matrix.mean(axis=0) if fit_matrix.shape[1] else np.zeros(0)
        scales = fit_matrix.std(axis=0) if fit_matrix.shape[1] else np.ones(0)
        scales[scales == 0] = 1.0
        fit_scaled = (fit_matrix - means) / scales
        eval_scaled = (eval_matrix - means) / scales
        return (
            fit_scaled,
            eval_scaled,
            np.asarray([float(item.target) for item in fit], dtype=np.float64),
        )

    def _ridge(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        numeric_fit = [item for item in fit if item.target_kind is TargetKind.NUMERIC]
        numeric_eval = [item for item in examples if item.target_kind is TargetKind.NUMERIC]
        if not numeric_fit:
            return tuple(_abstain(item, "no_numeric_fit_targets") for item in examples)
        x_fit, x_eval, y = self._numeric_design(numeric_fit, numeric_eval)
        x_fit = np.column_stack([np.ones(len(x_fit)), x_fit])
        x_eval = np.column_stack([np.ones(len(x_eval)), x_eval])
        alpha = float(self.baseline.hyperparameters.get("alpha", 1.0))
        penalty = np.eye(x_fit.shape[1]) * alpha
        penalty[0, 0] = 0.0
        weights = np.linalg.pinv(x_fit.T @ x_fit + penalty) @ x_fit.T @ y
        values = iter((x_eval @ weights).tolist())
        return tuple(
            RawPrediction(example_id=item.example_id, value=float(next(values)))
            if item.target_kind is TargetKind.NUMERIC
            else _abstain(item, "baseline_not_applicable_to_target_kind")
            for item in examples
        )

    @staticmethod
    def _one_hot_design(
        fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[np.ndarray, np.ndarray]:
        keys = tuple(sorted({key for item in fit for key in item.features}))
        columns: list[tuple[str, str]] = []
        numeric_keys: list[str] = []
        for key in keys:
            values = [item.features.get(key) for item in fit]
            if all(value is None or isinstance(value, int | float) for value in values):
                numeric_keys.append(key)
            else:
                columns.extend(
                    (key, str(value)) for value in sorted({str(value) for value in values})
                )
        numeric_matrix = np.asarray(
            [[float(item.features.get(key, 0.0) or 0.0) for key in numeric_keys] for item in fit],
            dtype=np.float64,
        )
        eval_numeric = np.asarray(
            [
                [float(item.features.get(key, 0.0) or 0.0) for key in numeric_keys]
                for item in examples
            ],
            dtype=np.float64,
        )
        if numeric_keys:
            means = numeric_matrix.mean(axis=0)
            scales = numeric_matrix.std(axis=0)
            scales[scales == 0] = 1.0
            numeric_matrix = (numeric_matrix - means) / scales
            eval_numeric = (eval_numeric - means) / scales
        fit_rows = []
        eval_rows = []
        for index, item in enumerate(fit):
            fit_rows.append(
                [1.0, *numeric_matrix[index].tolist()]
                + [float(str(item.features.get(key)) == value) for key, value in columns]
            )
        for index, item in enumerate(examples):
            eval_rows.append(
                [1.0, *eval_numeric[index].tolist()]
                + [float(str(item.features.get(key)) == value) for key, value in columns]
            )
        return np.asarray(fit_rows, dtype=np.float64), np.asarray(eval_rows, dtype=np.float64)

    def _logistic(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        categorical_fit = [item for item in fit if item.target_kind is TargetKind.CATEGORICAL]
        categorical_eval = [item for item in examples if item.target_kind is TargetKind.CATEGORICAL]
        labels = sorted({_label(item.target) for item in categorical_fit})
        if len(labels) < 2:
            return self._mode(fit, examples)
        x_fit, x_eval = self._one_hot_design(categorical_fit, categorical_eval)
        y = np.asarray([labels.index(_label(item.target)) for item in categorical_fit])
        weights = np.zeros((x_fit.shape[1], len(labels)), dtype=np.float64)
        learning_rate = float(self.baseline.hyperparameters.get("learning_rate", 0.1))
        iterations = int(self.baseline.hyperparameters.get("iterations", 250))
        regularization = float(self.baseline.hyperparameters.get("l2", 1.0e-3))
        one_hot = np.eye(len(labels), dtype=np.float64)[y]
        for _ in range(iterations):
            logits = x_fit @ weights
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = x_fit.T @ (probabilities - one_hot) / len(x_fit)
            gradient[1:] += regularization * weights[1:]
            weights -= learning_rate * gradient
        logits = x_eval @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        categorical_predictions = iter(probabilities)
        result: list[RawPrediction] = []
        for item in examples:
            if item.target_kind is not TargetKind.CATEGORICAL:
                result.append(_abstain(item, "baseline_not_applicable_to_target_kind"))
                continue
            row = next(categorical_predictions)
            winner = labels[int(row.argmax())]
            result.append(
                RawPrediction(
                    example_id=item.example_id,
                    value=winner,
                    probabilities={label: float(row[index]) for index, label in enumerate(labels)},
                )
            )
        return tuple(result)

    def _neighbor_mode(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        fallback = self._mode(fit, examples)
        support_by_family: defaultdict[str, set[str]] = defaultdict(set)
        for row in fit:
            if row.target_kind is TargetKind.CATEGORICAL:
                support_by_family[row.target_family].add(str(row.target))
        smoothing = float(self.baseline.hyperparameters.get("laplace_smoothing", 1.0))
        if not np.isfinite(smoothing) or smoothing <= 0.0:
            raise ValueError("neighbor-mode Laplace smoothing must be positive and finite")
        predictions: list[RawPrediction] = []
        for item, default in zip(examples, fallback, strict=True):
            raw_labels = item.context.get("neighbor_labels")
            if not isinstance(raw_labels, list) or not raw_labels:
                predictions.append(default)
                continue
            counts = Counter(str(label) for label in raw_labels)
            labels = sorted(support_by_family.get(item.target_family, set()))
            if not labels:
                predictions.append(default)
                continue
            if set(counts) - set(labels):
                raise ValueError("neighbor labels lie outside train-only target support")
            total = sum(counts.values()) + smoothing * len(labels)
            winner = min(labels, key=lambda label: (-counts[label], label))
            predictions.append(
                RawPrediction(
                    example_id=item.example_id,
                    value=winner,
                    probabilities={label: (counts[label] + smoothing) / total for label in labels},
                )
            )
        return tuple(predictions)

    def _user_item_bias(
        self, fit: Sequence[PreparedExample], examples: Sequence[BlindExample]
    ) -> tuple[RawPrediction, ...]:
        numeric_fit = [item for item in fit if item.target_kind is TargetKind.NUMERIC]
        if not numeric_fit:
            return tuple(_abstain(item, "no_numeric_fit_targets") for item in examples)
        global_mean = sum(float(item.target) for item in numeric_fit) / len(numeric_fit)
        regularization = float(self.baseline.hyperparameters.get("regularization", 10.0))
        iterations = int(self.baseline.hyperparameters.get("iterations", 10))
        user_bias: defaultdict[str, float] = defaultdict(float)
        item_bias: defaultdict[str, float] = defaultdict(float)
        by_user: defaultdict[str, list[PreparedExample]] = defaultdict(list)
        by_item: defaultdict[str, list[PreparedExample]] = defaultdict(list)
        for item in numeric_fit:
            by_user[str(item.context.get("user_id"))].append(item)
            by_item[str(item.context.get("item_id"))].append(item)
        for _ in range(iterations):
            for user in sorted(by_user):
                rows = by_user[user]
                residual = sum(
                    float(row.target) - global_mean - item_bias[str(row.context.get("item_id"))]
                    for row in rows
                )
                user_bias[user] = residual / (regularization + len(rows))
            for item_id in sorted(by_item):
                rows = by_item[item_id]
                residual = sum(
                    float(row.target) - global_mean - user_bias[str(row.context.get("user_id"))]
                    for row in rows
                )
                item_bias[item_id] = residual / (regularization + len(rows))
        return tuple(
            RawPrediction(
                example_id=item.example_id,
                value=float(
                    global_mean
                    + user_bias[str(item.context.get("user_id"))]
                    + item_bias[str(item.context.get("item_id"))]
                ),
            )
            if item.target_kind is TargetKind.NUMERIC
            else _abstain(item, "baseline_not_applicable_to_target_kind")
            for item in examples
        )
