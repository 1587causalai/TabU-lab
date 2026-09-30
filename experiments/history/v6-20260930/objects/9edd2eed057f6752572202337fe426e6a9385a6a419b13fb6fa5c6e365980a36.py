"""Shared NumPy metrics for frozen TabUBase and classical real-data baselines."""

from __future__ import annotations

import math

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Ranks are one-indexed in the Mann--Whitney representation of AUC.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels.astype(bool)
    positives = int(positive.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    ranks = _average_ranks(np.asarray(scores, dtype=np.float64))
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def classification_metrics(
    truth: np.ndarray,
    probabilities: np.ndarray,
    *,
    classes: int,
) -> dict[str, float]:
    """Return common classification metrics from one shared probability matrix."""

    labels = np.asarray(truth, dtype=np.int64)
    values = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or values.shape != (len(labels), classes) or classes < 2:
        raise ValueError("classification metrics require labels [N] and probabilities [N,C]")
    if bool(((labels < 0) | (labels >= classes)).any()) or not np.isfinite(values).all():
        raise ValueError("classification labels/probabilities are outside the declared domain")
    values = np.clip(values, 1.0e-12, None)
    values /= values.sum(axis=1, keepdims=True)
    predicted = values.argmax(axis=1)
    log_loss = float(-np.log(values[np.arange(len(labels)), labels]).mean())

    recalls: list[float] = []
    f1_values: list[float] = []
    for label in range(classes):
        true_positive = int(((predicted == label) & (labels == label)).sum())
        false_positive = int(((predicted == label) & (labels != label)).sum())
        false_negative = int(((predicted != label) & (labels == label)).sum())
        support = true_positive + false_negative
        recalls.append(true_positive / support if support else 0.0)
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)

    if classes == 2:
        roc_auc = _binary_roc_auc(labels == 1, values[:, 1])
    else:
        roc_auc = float(
            np.mean(
                [_binary_roc_auc(labels == label, values[:, label]) for label in range(classes)]
            )
        )
    return {
        "accuracy": float((predicted == labels).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "log_loss": log_loss,
        "normalized_nll": log_loss / math.log(classes),
        "roc_auc_ovr_macro": roc_auc,
    }


def regression_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    target_scale: float,
) -> dict[str, float]:
    """Return raw and train-scale-normalized regression metrics."""

    observed = np.asarray(truth, dtype=np.float64)
    values = np.asarray(predicted, dtype=np.float64)
    if observed.ndim != 1 or values.shape != observed.shape:
        raise ValueError("regression metrics require aligned one-dimensional arrays")
    if not np.isfinite(observed).all() or not np.isfinite(values).all():
        raise ValueError("regression metrics require finite truth and predictions")
    scale = float(target_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("target scale must be finite and positive")
    residual = values - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum((observed - observed.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else math.nan
    return {
        "rmse": rmse,
        "mae": mae,
        "scaled_rmse": rmse / scale,
        "scaled_mae": mae / scale,
        "r2": r2,
    }


__all__ = ["classification_metrics", "regression_metrics"]
