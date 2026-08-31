"""Query-row real-regression coordinate and loss boundaries.

Real tasks have a task-level response coordinate while each evidence episode
has its own context normalization.  This module keeps that conversion local to
the query-row real-data runners instead of changing the global Objective.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tabu_lab.contracts import LossBundle, PredictionBundle, TruthSidecar


def _batch(value: Tensor, *, name: str) -> Tensor:
    """Return a dense ``[batch, rows, features]`` tensor."""

    if value.ndim == 2:
        return value.unsqueeze(0)
    if value.ndim == 3:
        return value
    raise ValueError(f"{name} must have rank 2 or 3, got {tuple(value.shape)}")


def numeric_raw_prediction(prediction: PredictionBundle) -> Tensor:
    """Return task-scale ``y^(g)`` predictions from a public prediction bundle."""

    raw = prediction.auxiliaries.get("numeric_raw_prediction")
    if raw is None:
        raise ValueError("query-row real regression requires numeric_raw_prediction")
    return raw


def numeric_raw_prediction_from_public(prediction: Any) -> Tensor:
    """Read the explicit raw auxiliary from either a bundle or mapping facade."""

    if isinstance(prediction, PredictionBundle):
        return numeric_raw_prediction(prediction)
    try:
        raw = prediction["numeric_raw_prediction"]
    except (KeyError, TypeError) as exc:
        raise ValueError("query-row real regression requires numeric_raw_prediction") from exc
    return torch.as_tensor(raw)


def task_scale_to_raw(
    prediction_task_scale: Any,
    *,
    response_mean: float,
    response_scale: float,
) -> Any:
    """Apply the real-task inverse exactly once after context inversion."""

    return prediction_task_scale * float(response_scale) + float(response_mean)


def query_row_real_regression_loss(
    prediction: PredictionBundle,
    truth: TruthSidecar,
) -> LossBundle:
    """Score task-scale predictions against the stable real-task sidecar.

    ``TruthSidecar`` carries ``y^(g)``.  The model's public ``numeric`` entry is
    episode-context standardized, but ``numeric_raw_prediction`` is already in
    the task-scale coordinate.  This objective therefore performs no second
    context or task transform.
    """

    if prediction.episode_id != truth.episode_id:
        raise ValueError("prediction and TruthSidecar episode ids must match")
    raw = _batch(numeric_raw_prediction(prediction), name="numeric_raw_prediction")
    model_targets_value = prediction.auxiliaries.get("numeric_target_mask")
    if model_targets_value is None:
        model_targets_value = prediction.auxiliaries.get("target_mask")
    support_value = prediction.auxiliaries.get("numeric_support_available")
    if support_value is None:
        support_value = prediction.auxiliaries.get("support_available")
    if model_targets_value is None or support_value is None:
        raise ValueError("prediction is missing numeric target/support masks")
    model_targets = _batch(model_targets_value, name="numeric_target_mask").to(
        dtype=torch.bool, device=raw.device
    )
    support = _batch(support_value, name="numeric_support_available").to(
        dtype=torch.bool, device=raw.device
    )
    truth_values = _batch(truth.target_values.to(device=raw.device), name="truth.target_values")
    truth_targets = _batch(truth.target_mask.to(device=raw.device), name="truth.target_mask").to(
        dtype=torch.bool
    )
    if not (
        raw.shape
        == model_targets.shape
        == support.shape
        == truth_values.shape
        == truth_targets.shape
    ):
        raise ValueError("query-row real regression tensors must have identical dense shapes")
    if bool((truth_targets & ~model_targets).any()):
        raise ValueError("TruthSidecar target_mask must be a subset of model targets")
    scored = truth_targets & model_targets & support
    error = raw - truth_values.to(dtype=raw.dtype)
    zero = raw.sum() * 0.0
    count = scored.sum().to(dtype=raw.dtype)
    mse = torch.where(scored, error.square(), torch.zeros_like(error)).sum() / count.clamp_min(1)
    total = torch.where(count > 0, mse, zero)
    return LossBundle(
        episode_id=prediction.episode_id,
        total=total,
        components={"mse": mse},
        counts={
            "targets": int(truth_targets.sum().item()),
            "scored_targets": int(scored.sum().item()),
            "abstained_targets": int((truth_targets & ~scored).sum().item()),
        },
        metadata={
            "objective": "query_row_real_task_scale_numeric",
            "prediction_coordinate": "numeric_raw_prediction=y^(g)",
            "truth_coordinate": "TruthSidecar=y^(g)",
            "context_normalization_applied": False,
            "task_inverse_applied": False,
        },
    )


__all__ = [
    "numeric_raw_prediction",
    "numeric_raw_prediction_from_public",
    "query_row_real_regression_loss",
    "task_scale_to_raw",
]
