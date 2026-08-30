"""A context-only ordinary-linear-regression threshold for TabUR frozen ICL."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tabu_lab.contracts import canonical_hash

from .query_row_frozen_icl import _state_hash
from .query_row_pretraining import (
    QueryRowPretrainingResult,
    train_query_row_synthetic_pretraining_model,
)
from .query_row_synthetic_fit import QueryRowSyntheticEpisode, make_query_row_synthetic_episode
from .tabubase_scale import resolve_device

LINEAR_REGRESSION_BASELINE_ID = "ordinary_least_squares.context_only.v1"
LINEAR_REGRESSION_BASELINE_SPEC = {
    "baseline_id": LINEAR_REGRESSION_BASELINE_ID,
    "fit_scope": "context_rows_only",
    "features": "all_other_visible_columns",
    "missing_predictor_policy": "context_mean",
    "ridge": 1.0e-4,
    "metric": "context_standardized_target_mse",
}


def _context_standardization(episode: QueryRowSyntheticEpisode) -> tuple[Tensor, Tensor]:
    """Match the model's completion-profile numeric context statistics."""

    values = episode.evidence.forward_values.to(dtype=torch.float32)
    visible = ~episode.sidecar.target_mask
    visible_values = torch.where(visible, values, torch.zeros_like(values))
    count = visible.sum(dim=0).to(values.dtype).clamp_min(1.0)
    max_abs = visible_values.abs().amax(dim=0).clamp_min(1.0)
    scaled_values = values / max_abs
    scaled_mean = torch.where(visible, scaled_values, torch.zeros_like(scaled_values)).sum(dim=0)
    scaled_mean = scaled_mean / count
    centered = torch.where(
        visible,
        scaled_values - scaled_mean,
        torch.zeros_like(scaled_values),
    )
    scaled_std = (centered.square().sum(dim=0) / count).sqrt()
    mean = scaled_mean * max_abs
    scale = scaled_std * max_abs + 1.0e-6
    return mean, scale


def _linear_regression_predictions(
    episode: QueryRowSyntheticEpisode,
    *,
    context_rows: int,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Fit one OLS response law per feature using context rows only."""

    values = episode.evidence.forward_values.to(dtype=torch.float32).cpu()
    target = episode.sidecar.target_mask.to(dtype=torch.bool).cpu()
    rows, features = values.shape
    if not 1 <= context_rows < rows:
        raise ValueError("context_rows must be in [1, rows-1]")
    if bool(target[:context_rows].any()):
        raise ValueError("linear baseline context rows must contain no target cells")
    mean, scale = _context_standardization(episode)
    mean = mean.cpu()
    prediction = torch.zeros_like(values)
    for feature in range(features):
        predictors = [index for index in range(features) if index != feature]
        design = values[:context_rows, predictors]
        response = values[:context_rows, feature]
        design = torch.cat((torch.ones(context_rows, 1), design), dim=1)
        regularizer = torch.eye(design.shape[1]) * float(ridge)
        coefficients = torch.linalg.solve(
            design.transpose(0, 1) @ design + regularizer,
            design.transpose(0, 1) @ response,
        )
        query = values[context_rows:, predictors].clone()
        query_target = target[context_rows:, predictors]
        query = torch.where(query_target, mean[predictors].view(1, -1), query)
        query_design = torch.cat((torch.ones(rows - context_rows, 1), query), dim=1)
        prediction[context_rows:, feature] = query_design @ coefficients
    return prediction, scale


def _linear_regression_mse(
    episode: QueryRowSyntheticEpisode,
    *,
    context_rows: int,
    ridge: float = 1.0e-4,
) -> tuple[float, int]:
    prediction, scale = _linear_regression_predictions(
        episode,
        context_rows=context_rows,
        ridge=ridge,
    )
    target = episode.sidecar.target_mask.to(dtype=torch.bool).cpu()
    truth = episode.sidecar.target_values.to(dtype=torch.float32).cpu()
    error = (prediction - truth) / scale.view(1, -1)
    scored = error.square()[target]
    if not scored.numel():
        raise RuntimeError("linear baseline episode has no target cells")
    return float(scored.mean().item()), int(scored.numel())


def _model_mse(model: torch.nn.Module, episode: QueryRowSyntheticEpisode) -> tuple[float, int]:
    prediction = model(episode.evidence)
    predicted = prediction["numeric"]  # type: ignore[index]
    support = prediction["numeric_support_available"].to(torch.bool)  # type: ignore[index]
    mean = prediction["numeric_context_mean"]  # type: ignore[index]
    scale = prediction["numeric_context_scale"]  # type: ignore[index]
    if predicted.ndim == 2:
        predicted = predicted.unsqueeze(0)
        support = support.unsqueeze(0)
        mean = mean.unsqueeze(0)
        scale = scale.unsqueeze(0)
    truth = episode.sidecar.target_values.to(device=predicted.device)
    target = episode.sidecar.target_mask.to(device=predicted.device)
    scored = target.unsqueeze(0) & support
    if not bool(scored.any()):
        raise RuntimeError("TabUR frozen ICL episode has no supported target cells")
    standardized_truth = (truth.unsqueeze(0) - mean) / scale.clamp_min(1.0e-8)
    error = (predicted - standardized_truth).square()[scored]
    return float(error.mean().detach().cpu().item()), int(error.numel())


@dataclass(frozen=True, slots=True)
class QueryRowLinearICLRecord:
    world_id: str
    world_family: str
    context_rows: int
    target_count: int
    pretrained_mse: float
    linear_regression_mse: float

    @property
    def pretrained_at_or_below_linear(self) -> bool:
        return self.pretrained_mse <= self.linear_regression_mse

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "pretrained_at_or_below_linear": self.pretrained_at_or_below_linear,
        }


@dataclass(frozen=True, slots=True)
class QueryRowLinearICLContextSummary:
    """Target-cell-weighted result for one frozen-ICL context budget."""

    context_rows: int
    target_count: int
    pretrained_mse: float
    linear_regression_mse: float
    threshold_ratio: float

    @property
    def threshold_met(self) -> bool:
        return (
            math.isfinite(self.pretrained_mse)
            and math.isfinite(self.linear_regression_mse)
            and self.pretrained_mse <= self.threshold_ratio * self.linear_regression_mse
        )

    @property
    def relative_margin(self) -> float:
        return (self.linear_regression_mse - self.pretrained_mse) / max(
            abs(self.linear_regression_mse), 1.0e-8
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "threshold_met": self.threshold_met,
            "relative_margin": self.relative_margin,
        }


@dataclass(frozen=True, slots=True)
class QueryRowLinearICLThresholdResult:
    status: str
    threshold_met: bool
    evidence_status: str
    claim_boundary: str
    baseline_id: str
    baseline_spec_hash: str
    model_id: str
    contract_version: str
    model_spec_hash: str
    profile_id: str
    device: str
    seed: int
    pretrain_rows: int
    pretrain_worlds: int
    pretrain_steps: int
    eval_rows: int
    eval_worlds: int
    context_rows: tuple[int, ...]
    threshold_ratio: float
    pretrained_mse: float
    linear_regression_mse: float
    relative_margin: float
    parameter_hash_before: str
    parameter_hash_after: str
    parameter_hash_unchanged: bool
    pretraining: QueryRowPretrainingResult
    context_summaries: tuple[QueryRowLinearICLContextSummary, ...]
    records: tuple[QueryRowLinearICLRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pretraining"] = self.pretraining.as_dict()
        payload["records"] = [record.as_dict() for record in self.records]
        payload["context_summaries"] = [
            summary.as_dict() for summary in self.context_summaries
        ]
        payload["context_rows"] = list(self.context_rows)
        return payload


def run_query_row_linear_icl_threshold(
    *,
    seed: int = 1729,
    pretrain_rows: int = 64,
    pretrain_worlds: int = 64,
    pretrain_steps: int = 300,
    eval_rows: int = 32,
    eval_worlds: int = 12,
    context_rows: tuple[int, ...] = (8, 16, 32),
    row_token_count: int = 4,
    learning_rate: float = 1.0e-2,
    device: str | torch.device = "cpu",
    checkpoint: Path | None = None,
) -> QueryRowLinearICLThresholdResult:
    """Train TabUR and test whether frozen ICL reaches an OLS baseline.

    The held-out worlds and context sizes are fixed by ``seed``.  A threshold
    pass means both the target-cell-weighted aggregate and every context bucket
    are no larger than the same metric from ordinary least squares fit on
    context rows only.  The exact ratio is deliberately strict (``1.0``).
    """

    if pretrain_rows < 3 or pretrain_worlds <= 0 or pretrain_steps <= 0:
        raise ValueError("pretrain rows, worlds and steps must be positive")
    if eval_rows < 3 or eval_worlds <= 0:
        raise ValueError("eval rows and worlds must be positive")
    if not context_rows or any(size < 1 or size >= eval_rows for size in context_rows):
        raise ValueError("context_rows must be non-empty and smaller than eval_rows")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    threshold_ratio = 1.0
    resolved_device = resolve_device(str(device))
    model, pretraining = train_query_row_synthetic_pretraining_model(
        profile="completion.artificial_mask.v1",
        seed=seed,
        rows=pretrain_rows,
        worlds=pretrain_worlds,
        steps=pretrain_steps,
        learning_rate=learning_rate,
        row_token_count=row_token_count,
        device=resolved_device,
        output=checkpoint,
    )
    model.eval()
    heldout: list[tuple[QueryRowSyntheticEpisode, int, str]] = []
    families = ("row_latent_linear", "row_latent_periodic", "row_latent_polynomial")
    for world_index in range(eval_worlds):
        family = families[world_index % len(families)]
        for context_size in context_rows:
            episode = make_query_row_synthetic_episode(
                seed=seed + 100_000 + world_index * 101,
                rows=eval_rows,
                row_token_count=row_token_count,
                context_rows=context_size,
                world_id=f"heldout-linear-threshold-{world_index}-{context_size}",
                world_family=family,
            )
            heldout.append((episode, context_size, family))

    parameter_hash_before = _state_hash(model)
    records: list[QueryRowLinearICLRecord] = []
    weighted_pretrained = 0.0
    weighted_linear = 0.0
    total_targets = 0
    bucket_totals: dict[int, list[float]] = {
        context_size: [0.0, 0.0, 0.0] for context_size in context_rows
    }
    with torch.no_grad():
        for episode, context_size, family in heldout:
            pretrained_mse, pretrained_count = _model_mse(model, episode)
            linear_mse, linear_count = _linear_regression_mse(
                episode,
                context_rows=context_size,
            )
            if pretrained_count != linear_count:
                raise RuntimeError("TabUR and linear baseline scored different target counts")
            records.append(
                QueryRowLinearICLRecord(
                    world_id=episode.world_id,
                    world_family=family,
                    context_rows=context_size,
                    target_count=pretrained_count,
                    pretrained_mse=pretrained_mse,
                    linear_regression_mse=linear_mse,
                )
            )
            weighted_pretrained += pretrained_mse * pretrained_count
            weighted_linear += linear_mse * linear_count
            total_targets += pretrained_count
            bucket_totals[context_size][0] += pretrained_mse * pretrained_count
            bucket_totals[context_size][1] += linear_mse * linear_count
            bucket_totals[context_size][2] += pretrained_count
    parameter_hash_after = _state_hash(model)
    pretrained_mse = weighted_pretrained / total_targets
    linear_mse = weighted_linear / total_targets
    context_summaries = tuple(
        QueryRowLinearICLContextSummary(
            context_rows=context_size,
            target_count=int(totals[2]),
            pretrained_mse=totals[0] / totals[2],
            linear_regression_mse=totals[1] / totals[2],
            threshold_ratio=threshold_ratio,
        )
        for context_size, totals in sorted(bucket_totals.items())
    )
    threshold_met = (
        parameter_hash_before == parameter_hash_after
        and math.isfinite(pretrained_mse)
        and math.isfinite(linear_mse)
        and pretrained_mse <= threshold_ratio * linear_mse
        and all(summary.threshold_met for summary in context_summaries)
    )
    return QueryRowLinearICLThresholdResult(
        status="pass" if threshold_met else "continue",
        threshold_met=threshold_met,
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR frozen synthetic ICL versus a context-only ordinary-linear-regression "
            "diagnostic; no real-data transfer, benchmark, formal receipt, or accepted claim"
        ),
        baseline_id=LINEAR_REGRESSION_BASELINE_ID,
        baseline_spec_hash=canonical_hash(LINEAR_REGRESSION_BASELINE_SPEC),
        model_id=model.model_id,
        contract_version=model.contract_version,
        model_spec_hash=model.model_spec_hash,
        profile_id="completion.artificial_mask.v1",
        device=str(resolved_device),
        seed=seed,
        pretrain_rows=pretrain_rows,
        pretrain_worlds=pretrain_worlds,
        pretrain_steps=pretrain_steps,
        eval_rows=eval_rows,
        eval_worlds=eval_worlds,
        context_rows=context_rows,
        threshold_ratio=threshold_ratio,
        pretrained_mse=pretrained_mse,
        linear_regression_mse=linear_mse,
        relative_margin=(linear_mse - pretrained_mse) / max(abs(linear_mse), 1.0e-8),
        parameter_hash_before=parameter_hash_before,
        parameter_hash_after=parameter_hash_after,
        parameter_hash_unchanged=parameter_hash_before == parameter_hash_after,
        pretraining=pretraining,
        context_summaries=context_summaries,
        records=tuple(records),
    )


__all__ = [
    "LINEAR_REGRESSION_BASELINE_ID",
    "LINEAR_REGRESSION_BASELINE_SPEC",
    "QueryRowLinearICLContextSummary",
    "QueryRowLinearICLRecord",
    "QueryRowLinearICLThresholdResult",
    "run_query_row_linear_icl_threshold",
]
