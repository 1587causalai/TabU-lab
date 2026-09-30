"""Profile-compatible supervised synthetic episodes for TabUR Stage 6.

The generator deliberately uses the same ``supervised.label_broadcast.v1``
contract as the real-task runner.  This makes the pretraining checkpoint
transfer explicit and avoids silently loading a completion-profile checkpoint
into a supervised model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    TruthSidecar,
    canonical_hash,
)


@dataclass(frozen=True, slots=True)
class QueryRowSupervisedSyntheticEpisode:
    evidence: EvidenceEpisode
    sidecar: TruthSidecar
    generator_id: str = "tabur.supervised-row-latent-linear.v1"
    world_id: str = "fixed-world"


def make_query_row_supervised_synthetic_episode(
    *,
    seed: int,
    rows: int = 32,
    context_rows: int | None = None,
    row_token_count: int = 4,
    noise_scale: float = 0.05,
    world_id: str = "fixed-world",
    world_family: str = "row_latent_linear",
) -> QueryRowSupervisedSyntheticEpisode:
    """Create a numeric supervised episode with a query-only response column."""

    if rows < 3:
        raise ValueError("rows must be at least 3")
    if row_token_count <= 0:
        raise ValueError("row_token_count must be positive")
    if context_rows is None:
        context_rows = max(2, rows // 2)
    if context_rows < 1 or context_rows >= rows:
        raise ValueError("context_rows must be in [1, rows-1]")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    if world_family not in {"row_latent_linear", "row_latent_periodic", "row_latent_polynomial"}:
        raise ValueError("unknown TabUR supervised synthetic world family")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(rows, max(4, row_token_count), generator=generator)
    x0, x1, x2, x3 = latent.unbind(dim=1)
    predictors = torch.stack((x0, x1, x2, x3), dim=1)
    if world_family == "row_latent_linear":
        response = 0.8 * x0 + 0.35 * x1 - 0.2 * x2 + 0.1 * x3
    elif world_family == "row_latent_periodic":
        response = torch.sin(x0) + 0.25 * torch.cos(x1) + 0.15 * x2
    else:
        response = x0.square() + 0.25 * x1 * x2 - 0.1 * x3
    response = response + noise_scale * torch.randn(rows, generator=generator)
    values = torch.cat((predictors, response.unsqueeze(1)), dim=1)

    query_start = context_rows
    forward_values = values.clone()
    forward_values[query_start:, -1] = 0.0
    feature_names = (*tuple(f"feature_{index}" for index in range(predictors.shape[1])), "response")
    feature_specs = (
        *tuple(
        FeatureSpec(name=name, kind=FeatureKind.NUMERIC, role=FeatureRole.PREDICTOR)
        for name in feature_names[:-1]
        ),
        FeatureSpec(name="response", kind=FeatureKind.NUMERIC, role=FeatureRole.RESPONSE),
    )

    origins: list[list[OriginState]] = []
    roles: list[list[ForwardRole]] = []
    target_mask = torch.zeros(rows, values.shape[1], dtype=torch.bool)
    target_mask[query_start:, -1] = True
    for row in range(rows):
        row_origins = [OriginState.OBSERVED] * values.shape[1]
        row_roles = [ForwardRole.RECEIVER | ForwardRole.SOURCE] * values.shape[1]
        if row >= query_start:
            row_origins[-1] = OriginState.QUERY
            row_roles[-1] = ForwardRole.RECEIVER | ForwardRole.TARGET
        origins.append(row_origins)
        roles.append(row_roles)

    episode_id = f"tabur-supervised-{world_id}-{seed}"
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised",
        source_partition="synthetic_context",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"r{index}" for index in range(rows)),
        feature_names=feature_names,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        feature_specs=feature_specs,
        metadata={
            "generator_id": "tabur.supervised-row-latent-linear.v1",
            "world_id": world_id,
            "world_family": world_family,
            "row_token_count": row_token_count,
            "truth_boundary": "sidecar_only",
            "profile_id": "supervised.label_broadcast.v1",
        },
    )
    target_values = torch.zeros_like(values)
    target_values[target_mask] = values[target_mask]
    sidecar = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.recipe.v1",
                "generator_id": "tabur.supervised-row-latent-linear.v1",
                "world_id": world_id,
                "world_family": world_family,
                "seed": seed,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=feature_names,
        target_values=target_values,
        target_mask=target_mask,
        metadata={"truth_scope": "loss_only", "profile_id": "supervised.label_broadcast.v1"},
    )
    return QueryRowSupervisedSyntheticEpisode(evidence=evidence, sidecar=sidecar, world_id=world_id)


def supervised_synthetic_episode_loss(
    model: object,
    episode: QueryRowSupervisedSyntheticEpisode,
) -> Tensor:
    """Compute the numeric query-response loss through the public model output."""

    prediction = model(episode.evidence)  # type: ignore[operator]
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
    standardized_truth = (truth.unsqueeze(0) - mean) / scale.clamp_min(1.0e-8)
    scored = target.unsqueeze(0) & support
    if not bool(scored.any()):
        raise RuntimeError("supervised synthetic episode has no supported query target")
    error = predicted - standardized_truth
    numerator = torch.where(scored, error.square(), torch.zeros_like(error)).sum()
    return numerator / scored.sum().clamp_min(1)


__all__ = [
    "QueryRowSupervisedSyntheticEpisode",
    "make_query_row_supervised_synthetic_episode",
    "supervised_synthetic_episode_loss",
]
