"""Sample-rescorable metrics with train-only normalization."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .contracts import (
    FailureCategory,
    PerExampleScore,
    PreparedExample,
    RawPrediction,
    ScenarioSpec,
    TargetKind,
)


@dataclass(frozen=True)
class ScoredPredictions:
    per_example: tuple[PerExampleScore, ...]
    metrics: dict[str, float]
    counts: dict[str, int]
    failure_counts: dict[FailureCategory, int]
    coverage: float


def _binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def score_predictions(
    *,
    scenario: ScenarioSpec,
    fit_examples: Sequence[PreparedExample],
    truth: Sequence[PreparedExample],
    predictions: Sequence[RawPrediction],
    train_only_fitted_state: Mapping[str, object] | None = None,
) -> ScoredPredictions:
    """Score retained predictions without exposing truth to an adapter.

    ``fit_examples`` may be a bounded evaluator selection while a materializer
    fits train-only statistics/codebooks on the complete declared train
    partition.  When that contract is explicit, callers pass the retained
    fitted state so scoring uses the same train-only normalization and category
    support instead of treating the selected subset as the authority.
    """

    fit_examples = tuple(
        PreparedExample.model_validate(
            item.model_dump(mode="python") if isinstance(item, PreparedExample) else item
        )
        for item in fit_examples
    )
    truth = tuple(
        PreparedExample.model_validate(
            item.model_dump(mode="python") if isinstance(item, PreparedExample) else item
        )
        for item in truth
    )
    predictions = tuple(
        RawPrediction.model_validate(
            item.model_dump(mode="python") if isinstance(item, RawPrediction) else item
        )
        for item in predictions
    )

    truth_by_id = {item.example_id: item for item in truth}
    prediction_by_id = {item.example_id: item for item in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("raw prediction ids must be unique")
    if set(prediction_by_id) != set(truth_by_id):
        missing = sorted(set(truth_by_id) - set(prediction_by_id))
        extra = sorted(set(prediction_by_id) - set(truth_by_id))
        raise ValueError(f"predictions must cover exact test ids; missing={missing}, extra={extra}")

    numeric_fit_by_family: dict[str, list[float]] = {}
    categorical_fit_by_family: dict[str, set[str]] = {}
    for item in fit_examples:
        if item.target_kind is TargetKind.NUMERIC:
            numeric_fit_by_family.setdefault(item.target_family, []).append(float(item.target))
        else:
            categorical_fit_by_family.setdefault(item.target_family, set()).add(str(item.target))
    numeric_scales: dict[str, float] = {}
    for family, values in numeric_fit_by_family.items():
        mean = sum(values) / len(values)
        scale = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        if scale == 0:
            scale = max(values) - min(values)
        numeric_scales[family] = scale or 1.0

    if train_only_fitted_state is not None:
        raw_families = train_only_fitted_state.get("families")
        if not isinstance(raw_families, Mapping):
            raise ValueError("train-only fitted state has no family map")
        for family_id, raw_family in raw_families.items():
            if not isinstance(family_id, str) or not isinstance(raw_family, Mapping):
                raise ValueError("train-only fitted state family map is malformed")
            kind = raw_family.get("kind")
            if kind == "numeric":
                mean = raw_family.get("mean")
                scale = raw_family.get("scale")
                if (
                    not isinstance(mean, int | float)
                    or isinstance(mean, bool)
                    or not isinstance(scale, int | float)
                    or isinstance(scale, bool)
                    or not math.isfinite(float(mean))
                    or not math.isfinite(float(scale))
                    or float(scale) <= 0.0
                ):
                    raise ValueError("numeric train-only fitted state is malformed")
                numeric_scales[family_id] = float(scale)
            elif kind == "categorical":
                domain = raw_family.get("domain")
                if (
                    not isinstance(domain, list)
                    or not domain
                    or any(not isinstance(value, str) or not value for value in domain)
                ):
                    raise ValueError("categorical train-only fitted state is malformed")
                categorical_fit_by_family[family_id] = set(domain)
            else:
                raise ValueError("train-only fitted state has an unsupported family kind")
    nll_required = any(
        metric.metric_id in {"classification_nll", "categorical_nll"} for metric in scenario.metrics
    )

    per_example: list[PerExampleScore] = []
    numeric_truth: list[float] = []
    numeric_prediction: list[float] = []
    numeric_truth_by_id: dict[str, float] = {}
    numeric_normalized_squared_errors: list[float] = []
    categorical_truth: list[str] = []
    categorical_prediction: list[str] = []
    categorical_nll: list[float] = []
    auc_labels: list[int] = []
    auc_scores: list[float] = []
    failures: Counter[FailureCategory] = Counter()

    for item in truth:
        prediction = prediction_by_id[item.example_id]
        numeric_scale: float | None = None
        categorical_support: tuple[str, ...] = ()
        if item.target_kind is TargetKind.NUMERIC:
            numeric_scale = numeric_scales.get(item.target_family)
            if numeric_scale is None:
                raise ValueError(
                    "numeric target family has no train-only normalization scale: "
                    f"{item.target_family}"
                )
            truth_value: float | str = float(item.target)
        else:
            family_labels = categorical_fit_by_family.get(item.target_family)
            if not family_labels:
                raise ValueError(
                    "categorical target family has no train-only label support: "
                    f"{item.target_family}"
                )
            categorical_support = tuple(sorted(family_labels))
            truth_value = str(item.target)
            if truth_value not in family_labels:
                raise ValueError(
                    "categorical truth lies outside train-only label support for family "
                    f"{item.target_family}"
                )
        if prediction.abstained:
            category = prediction.failure_category or FailureCategory.EVALUATOR
            failures[category] += 1
            per_example.append(
                PerExampleScore(
                    example_id=item.example_id,
                    prediction_sha256=prediction.content_hash,
                    target_kind=item.target_kind,
                    target_family=item.target_family,
                    truth=truth_value,
                    normalization_scale=numeric_scale,
                    categorical_support=categorical_support,
                    scored=False,
                    failure_category=category,
                )
            )
            continue

        metrics: dict[str, float] = {}
        if item.target_kind is TargetKind.NUMERIC:
            assert numeric_scale is not None
            if (
                prediction.value is None
                or isinstance(prediction.value, str)
                or prediction.probabilities is not None
            ):
                raise ValueError("numeric targets require a numeric prediction value")
            predicted = float(prediction.value)
            actual = float(truth_value)
            error = predicted - actual
            metrics = {
                "absolute_error": abs(error),
                "squared_error": error**2,
            }
            normalized_squared_error = (error / numeric_scale) ** 2
            metrics["normalized_squared_error"] = normalized_squared_error
            numeric_normalized_squared_errors.append(normalized_squared_error)
            numeric_truth.append(actual)
            numeric_prediction.append(predicted)
            numeric_truth_by_id[item.example_id] = actual
        else:
            family_labels = set(categorical_support)
            if prediction.value is None:
                if prediction.probabilities is None:
                    raise ValueError("categorical targets require a value or distribution")
                predicted_label = min(
                    prediction.probabilities,
                    key=lambda label: (-prediction.probabilities[label], label),
                )
            else:
                predicted_label = str(prediction.value)
            actual_label = str(truth_value)
            if predicted_label not in family_labels:
                raise ValueError(
                    "categorical prediction lies outside train-only label support for family "
                    f"{item.target_family}"
                )
            metrics["correct"] = float(predicted_label == actual_label)
            categorical_truth.append(actual_label)
            categorical_prediction.append(predicted_label)
            if prediction.probabilities is not None:
                probability_labels = set(prediction.probabilities)
                if probability_labels != family_labels:
                    raise ValueError(
                        "categorical probability support must exactly match the train-only "
                        f"labels for target family {item.target_family}"
                    )
                probability = prediction.probabilities[actual_label]
                if probability <= 0.0:
                    raise ValueError("categorical NLL requires positive truth probability")
                nll = -math.log(probability)
                metrics["negative_log_likelihood"] = nll
                categorical_nll.append(nll)

                distribution_label = min(
                    prediction.probabilities,
                    key=lambda label: (-prediction.probabilities[label], label),
                )
                if prediction.value is not None and predicted_label != distribution_label:
                    raise ValueError(
                        "categorical value must agree with the retained probability distribution"
                    )

                labels = sorted(prediction.probabilities)
                if len(labels) == 2 and actual_label in labels:
                    positive = labels[-1]
                    auc_label = int(actual_label == positive)
                    auc_score = prediction.probabilities[positive]
                    metrics["auc_label"] = float(auc_label)
                    metrics["auc_score"] = auc_score
                    auc_labels.append(auc_label)
                    auc_scores.append(auc_score)
            elif nll_required:
                raise ValueError(
                    "categorical NLL requires a complete probability distribution for every "
                    "scored categorical target"
                )

        per_example.append(
            PerExampleScore(
                example_id=item.example_id,
                prediction_sha256=prediction.content_hash,
                target_kind=item.target_kind,
                target_family=item.target_family,
                truth=truth_value,
                normalization_scale=numeric_scale,
                categorical_support=categorical_support,
                scored=True,
                metrics=metrics,
            )
        )

    total = len(truth)
    scored = len(numeric_truth) + len(categorical_truth)
    coverage = scored / total
    aggregates: dict[str, float] = {
        "coverage": coverage,
        "abstention": 1.0 - coverage,
    }
    if numeric_truth:
        errors = [
            predicted - actual
            for predicted, actual in zip(numeric_prediction, numeric_truth, strict=True)
        ]
        mse = sum(error**2 for error in errors) / len(errors)
        aggregates.update(
            {
                "rmse": mse**0.5,
                "mae": sum(abs(error) for error in errors) / len(errors),
            }
        )
        aggregates["nrmse"] = (
            sum(numeric_normalized_squared_errors) / len(numeric_normalized_squared_errors)
        ) ** 0.5
        truth_mean = sum(numeric_truth) / len(numeric_truth)
        for index, score in enumerate(per_example):
            actual = numeric_truth_by_id.get(score.example_id)
            if actual is None or not score.scored:
                continue
            contributions = dict(score.metrics)
            contributions["truth_centered_squared"] = (actual - truth_mean) ** 2
            per_example[index] = score.model_copy(update={"metrics": contributions})
        denominator = sum((value - truth_mean) ** 2 for value in numeric_truth)
        if denominator:
            aggregates["r2"] = 1.0 - sum(error**2 for error in errors) / denominator
    if categorical_truth:
        aggregates["accuracy"] = sum(
            predicted == actual
            for predicted, actual in zip(categorical_prediction, categorical_truth, strict=True)
        ) / len(categorical_truth)
    if categorical_nll:
        if len(categorical_nll) != len(categorical_truth):
            raise ValueError("categorical NLL is incomplete for scored categorical targets")
        aggregates["nll"] = sum(categorical_nll) / len(categorical_nll)
    auc = _binary_auc(auc_labels, auc_scores)
    if auc is not None:
        aggregates["auroc"] = auc
    aliases = {
        "classification_nll": "nll",
        "categorical_nll": "nll",
        "categorical_accuracy": "accuracy",
        "regression_nrmse": "nrmse",
        "numeric_nrmse": "nrmse",
        "numeric_mae": "mae",
    }
    requested: dict[str, float] = {}
    for metric in scenario.metrics:
        source_name = aliases.get(metric.metric_id, metric.metric_id)
        if source_name not in aggregates:
            if metric.primary:
                raise ValueError(f"primary metric could not be scored: {metric.metric_id}")
            continue
        requested[metric.metric_id] = aggregates[source_name]

    counts = {
        "targets": total,
        "scored": scored,
        "abstained": total - scored,
        "numeric_targets": sum(item.target_kind is TargetKind.NUMERIC for item in truth),
        "categorical_targets": sum(item.target_kind is TargetKind.CATEGORICAL for item in truth),
    }
    return ScoredPredictions(
        per_example=tuple(per_example),
        metrics=requested,
        counts=counts,
        failure_counts=dict(sorted(failures.items(), key=lambda item: item[0].value)),
        coverage=coverage,
    )


def rescore_result_samples(
    *,
    scenario: ScenarioSpec,
    fit_examples: Sequence[PreparedExample],
    truth: Sequence[PreparedExample],
    predictions: Sequence[RawPrediction],
    train_only_fitted_state: Mapping[str, object] | None = None,
) -> ScoredPredictions:
    """Public alias emphasizing that raw outputs are sufficient for rescoring."""

    return score_predictions(
        scenario=scenario,
        fit_examples=fit_examples,
        truth=truth,
        predictions=predictions,
        train_only_fitted_state=train_only_fitted_state,
    )
