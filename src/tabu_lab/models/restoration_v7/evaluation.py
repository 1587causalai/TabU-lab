"""Frozen V7 trajectories with the donor and support-marginal baselines.

Evaluation runs the same forward as training under inference admission; only
the metrics below read the isolated truth sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

from ..restoration._validation import finite
from .codec import G64ColumnCodec
from .model import V7Episode, V7Model, V7Output, prepare_episode
from .runner import V7Task
from .training import reference_values, state_loss


@dataclass(frozen=True)
class V7Trajectory:
    target: int
    kind: str
    status: str  # "ok" or "no-answer-code" (code losses then undefined)
    code_losses: tuple[float, ...] | None  # L_0 (initial Query state) .. L_K
    state_changes: tuple[float, ...]  # mean row ||C^(t) - C^(t-1)||, t = 1..K
    predictions: tuple[Tensor, ...]  # decoded values of rounds 1..K
    donor: Tensor  # sampled-support baseline, distinct from the default seed initial state
    metrics: dict[str, float]  # final round, original value units
    baselines: dict[str, dict[str, float]]


@dataclass(frozen=True)
class V7JointTrajectory:
    columns: dict[int, V7Trajectory]


def value_metrics(column: G64ColumnCodec, prediction: Tensor, truth: Tensor) -> dict[str, float]:
    prediction, truth = prediction.detach().cpu(), truth.detach().cpu()
    if column.kind == "numeric":
        prediction, truth = prediction.to(torch.float64), truth.to(torch.float64)
        finite(prediction, "V7 numeric metric predictions")
        finite(truth, "V7 numeric metric references")
        error = prediction - truth
        magnitude = error.abs().amax()
        if not bool(torch.isfinite(magnitude)):
            # Opposite finite extremes can overflow an individual difference
            # even when the aggregate MAE/RMSE is still representable.
            magnitude = torch.maximum(prediction.abs().amax(), truth.abs().amax())
            scaled = prediction / magnitude - truth / magnitude
        elif float(magnitude) == 0.0:
            scaled = torch.zeros_like(error)
        else:
            scaled = error / magnitude
        mae = scaled.abs().mean() * magnitude
        rmse = scaled.square().mean().sqrt() * magnitude
        standardized_rmse = rmse / column.scale
        finite(torch.stack((mae, rmse, standardized_rmse)), "V7 numeric metrics")
        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "standardized_rmse": float(standardized_rmse),
        }
    metrics = {"accuracy": float((prediction == truth).to(torch.float64).mean())}
    if column.kind == "ordinal":
        # Domain indices are category identities, not necessarily rank order.
        # Codecs constructed through the former API default to identity order.
        if column.rank_positions is None:
            distances = (prediction - truth).abs()
        else:
            positions = column.rank_positions.detach().cpu()
            distances = (positions[prediction] - positions[truth]).abs()
        metrics["mean_rank_distance"] = float(distances.to(torch.float64).mean())
    return metrics


def _marginal(column: G64ColumnCodec, labels: Tensor, count: int) -> Tensor:
    labels = labels.detach().cpu()
    if column.kind == "numeric":
        # The codec already computed the same visible-support mean stably.
        return labels.new_full((count,), column.mean, dtype=torch.float64)
    counts = torch.bincount(labels, minlength=column.domain_size)
    # argmax returns the first maximum: ties break toward the lowest domain index.
    return counts.argmax().expand(count)


def _baselines(episode: V7Episode, truth: Tensor, codes: Tensor | None, chi: float) -> dict:
    column = episode.codec.columns[episode.target]
    labels = episode.inputs.values[episode.target][episode.support_rows]
    count = len(episode.query_rows)
    donor_values = episode.inputs.values[episode.target][episode.donor_rows]
    marginal = value_metrics(column, _marginal(column, labels, count), truth)
    donor = value_metrics(column, donor_values, truth)
    if codes is not None:
        support_codes = episode.observed[episode.support_rows, episode.target]
        mean_code = support_codes.mean(0).expand(count, -1).to(codes)
        marginal["code_mean_loss"] = float(state_loss(mean_code, codes, chi))
        donor_codes = episode.observed[episode.donor_rows, episode.target].to(codes)
        donor["code_loss"] = float(state_loss(donor_codes, codes, chi))
    return {"donor": donor, "support_marginal": marginal}


@torch.no_grad()
def evaluate_task(model: V7Model, task: V7Task) -> V7Trajectory | V7JointTrajectory:
    if int(task.inputs.query.any(0).sum()) > 1:
        return evaluate_joint_task(model, task)
    model.eval()
    config = model.config
    episode = prepare_episode(
        task.inputs,
        donor_seed=task.donor_seed,
        code_dim=config.code_dim,
        epsilon=config.epsilon,
        codec=config.codec,
        admission="inference",
    )
    output = model(episode, decode=False)
    return _trajectory(episode, output, task, config)


def _trajectory(episode, output, task, config):
    column = episode.codec.columns[episode.target]
    truth = reference_values(episode, task.truth)
    chi = config.chi_numeric if column.kind == "numeric" else config.chi_discrete
    encodable = bool(column.encodable(truth).all())
    codes = column.encode(truth).to(output.initial) if encodable else None
    trail = (output.initial, *output.states)
    losses = tuple(float(state_loss(s, codes, chi)) for s in trail) if encodable else None
    changes = tuple(
        float((after - before).norm(dim=-1).mean()) for before, after in pairwise(trail)
    )
    predictions = tuple(column.decode(state) for state in output.states)
    return V7Trajectory(
        target=episode.target,
        kind=column.kind,
        status="ok" if encodable else "no-answer-code",
        code_losses=losses,
        state_changes=changes,
        predictions=predictions,
        donor=column.decode(episode.observed[episode.donor_rows, episode.target]),
        metrics=value_metrics(column, predictions[-1], truth),
        baselines=_baselines(episode, truth, codes, chi),
    )


@torch.no_grad()
def evaluate_joint_task(model: V7Model, task: V7Task) -> V7JointTrajectory:
    """Report every queried column from the same joint forward, including all rounds."""
    from .joint import prepare_joint_episode

    model.eval()
    episode = prepare_joint_episode(
        task.inputs, donor_seed=task.donor_seed, config=model.config, admission="inference"
    )
    output = model(episode, decode=False)
    reports = {}
    for part in episode.columns:
        column_episode = V7Episode(
            episode.inputs,
            episode.codec,
            part.column,
            part.supports,
            part.rows,
            part.donors,
            episode.observed,
        )
        column_output = V7Output(
            output.initial[part.positions],
            tuple(s[part.positions] for s in output.states),
            tuple(s[part.column] for s in output.slopes),
            None,
        )
        reports[part.column] = _trajectory(column_episode, column_output, task, model.config)
    return V7JointTrajectory(reports)


__all__ = [
    "V7JointTrajectory",
    "V7Trajectory",
    "evaluate_joint_task",
    "evaluate_task",
    "value_metrics",
]
