"""Bounded Stage-3 synthetic fit for the executable TabUR row family.

The generator makes one deterministic row-heterogeneous numeric world.  The
world's latent row state is generator-only; the model receives only a masked,
truth-free ``EvidenceEpisode``.  This module is an F0 realizability harness,
not a multi-world pretraining result or a formal evidence receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

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
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_identity import query_row_result_identity
from .tabubase_scale import resolve_device


@dataclass(frozen=True, slots=True)
class QueryRowSyntheticEpisode:
    evidence: EvidenceEpisode
    sidecar: TruthSidecar
    generator_id: str = "tabur.row-latent-linear-world.v1"
    world_id: str = "fixed-world"


@dataclass(frozen=True, slots=True)
class QueryRowSyntheticFitResult:
    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    profile_id: str
    model_spec_hash: str
    variant_hash: str
    row_readout_mode: str
    row_readout_identity: dict[str, Any]
    generator_id: str
    world_id: str
    row_token_count: int
    seed: int
    steps: int
    learning_rate: float
    device: str
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRowSyntheticMultiWorldFitResult:
    """Bounded S1 result over disjoint synthetic worlds."""

    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    profile_id: str
    model_spec_hash: str
    variant_hash: str
    row_readout_mode: str
    row_readout_identity: dict[str, Any]
    generator_id: str
    train_worlds: int
    validation_worlds: int
    row_token_count: int
    seed: int
    steps: int
    learning_rate: float
    device: str
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mask_with_support(
    generator: torch.Generator,
    *,
    rows: int,
    features: int,
    mask_ratio: float,
) -> Tensor:
    mask = torch.rand(rows, features, generator=generator) < mask_ratio
    # Keep one visible and one target value for every feature.  This is a
    # generator validity condition, not a model-side imputation shortcut.
    mask[0, :] = False
    mask[1, :] = True
    return mask


def make_query_row_synthetic_episode(
    *,
    seed: int,
    rows: int = 32,
    row_token_count: int = 4,
    mask_ratio: float = 0.25,
    noise_scale: float = 0.05,
    world_id: str = "fixed-world",
    world_family: str = "row_latent_linear",
    context_rows: int | None = None,
) -> QueryRowSyntheticEpisode:
    """Create a deterministic row-heterogeneous numeric completion episode."""

    if rows < 3:
        raise ValueError("rows must be at least 3 to preserve visible and target support")
    if row_token_count <= 0:
        raise ValueError("row_token_count must be positive")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if world_family not in {"row_latent_linear", "row_latent_periodic", "row_latent_polynomial"}:
        raise ValueError("unknown TabUR synthetic world family")
    latent = torch.randn(rows, max(4, row_token_count), generator=generator)
    noise = noise_scale * torch.randn(rows, 4, generator=generator)
    q0, q1, q2, q3 = latent.unbind(dim=1)
    if world_family == "row_latent_linear":
        columns = (
            q0 + noise[:, 0],
            (0.5 + 0.5 * torch.tanh(q1)) * q0 + 0.25 * q2 + noise[:, 1],
            q1 - 0.35 * q2 + 0.15 * q3 + noise[:, 2],
            (1.0 + 0.3 * torch.tanh(q0)) * q3 + 0.2 * q1 + noise[:, 3],
        )
    elif world_family == "row_latent_periodic":
        columns = (
            torch.sin(q0) + noise[:, 0],
            torch.cos(q1) * q0 + 0.25 * q2 + noise[:, 1],
            torch.sin(q1 + q2) + 0.15 * q3 + noise[:, 2],
            torch.cos(q3 + 0.3 * q0) + 0.2 * q1 + noise[:, 3],
        )
    else:
        columns = (
            q0.square() + noise[:, 0],
            q0 * q1 + 0.25 * q2 + noise[:, 1],
            q1.square() - 0.35 * q2 + noise[:, 2],
            q3 * q3 * torch.sign(q0) + 0.2 * q1 + noise[:, 3],
        )
    values = torch.stack(columns, dim=1)
    target_mask = _mask_with_support(
        generator,
        rows=rows,
        features=values.shape[1],
        mask_ratio=mask_ratio,
    )
    if context_rows is not None:
        if context_rows < 1 or context_rows >= rows:
            raise ValueError("context_rows must be in [1, rows-1]")
        target_mask[:context_rows, :] = False
        if not bool(target_mask[context_rows:, :].any()):
            target_mask[context_rows, 0] = True
    visible_mask = ~target_mask
    forward_values = values.masked_fill(target_mask, 0.0)
    feature_names = tuple(f"x{index}" for index in range(values.shape[1]))
    feature_specs = tuple(
        FeatureSpec(
            name=name,
            kind=FeatureKind.NUMERIC,
            role=FeatureRole.PREDICTOR,
        )
        for name in feature_names
    )
    origins = [
        [
            OriginState.OBSERVED if visible_mask[row, col] else OriginState.ARTIFICIAL_MASK
            for col in range(values.shape[1])
        ]
        for row in range(rows)
    ]
    roles = [
        [
            (ForwardRole.RECEIVER | ForwardRole.SOURCE)
            if visible_mask[row, col]
            else (ForwardRole.RECEIVER | ForwardRole.TARGET)
            for col in range(values.shape[1])
        ]
        for row in range(rows)
    ]
    evidence = EvidenceEpisode(
        episode_id=f"tabur-{world_id}-{seed}",
        dataset_id="tabur-synthetic",
        source_partition="synthetic_context",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"r{index}" for index in range(rows)),
        feature_names=feature_names,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        feature_specs=feature_specs,
        metadata={
            "generator_id": "tabur.row-latent-linear-world.v1",
            "world_id": world_id,
            "world_family": world_family,
            "row_token_count": row_token_count,
            "truth_boundary": "sidecar_only",
        },
    )
    sidecar = TruthSidecar(
        episode_id=evidence.episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.synthetic.recipe.v1",
                "generator_id": "tabur.row-latent-linear-world.v1",
                "world_id": world_id,
                "world_family": world_family,
                "seed": seed,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=feature_names,
        target_values=values.masked_fill(~target_mask, 0.0),
        target_mask=target_mask,
        metadata={"truth_scope": "loss_only"},
    )
    return QueryRowSyntheticEpisode(evidence=evidence, sidecar=sidecar, world_id=world_id)


def _episode_loss(model: Any, episode: QueryRowSyntheticEpisode) -> Tensor:
    prediction = model(episode.evidence)
    predicted = prediction["numeric"]
    support = prediction["numeric_support_available"].to(torch.bool)
    mean = prediction["numeric_context_mean"]
    scale = prediction["numeric_context_scale"]
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
        raise RuntimeError("TabUR synthetic episode has no supported target cells")
    error = predicted - standardized_truth
    numerator = torch.where(scored, error.square(), torch.zeros_like(error)).sum()
    return numerator / scored.sum().clamp_min(1)


def run_query_row_fixed_world_fit(
    *,
    seed: int = 1729,
    steps: int = 20,
    learning_rate: float = 1.0e-2,
    rows: int = 32,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
    config: ReferenceConfig | None = None,
) -> QueryRowSyntheticFitResult:
    """Run the bounded TabUR F0 fit gate on one fixed synthetic world."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    torch.manual_seed(seed)
    train = make_query_row_synthetic_episode(
        seed=seed + 1,
        rows=rows,
        row_token_count=row_token_count,
        world_id="train-fixed-world",
    )
    validation = make_query_row_synthetic_episode(
        seed=seed + 2,
        rows=rows,
        row_token_count=row_token_count,
        world_id="validation-fixed-world",
    )
    model = build_model(
        "tabu.query.row",
        config=config
        or ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=row_token_count,
            max_features=4,
        ),
        profile="completion.artificial_mask.v1",
        row_token_count=row_token_count,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_validation = float(_episode_loss(model, validation).item())
    initial_train: float | None = None
    final_train = float("nan")
    for _ in range(steps):
        loss = _episode_loss(model, train)
        if initial_train is None:
            initial_train = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_train = float(loss.item())
    model.eval()
    with torch.no_grad():
        final_validation = float(_episode_loss(model, validation).item())
    status = (
        "pass"
        if torch.isfinite(torch.tensor(final_train)) and final_train < initial_train
        else "kill"
    )
    result_identity = query_row_result_identity(model.checkpoint_identity())
    return QueryRowSyntheticFitResult(
        status=status,
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR fixed-world synthetic realizability only; no multi-world fit, "
            "real-data, frozen ICL, or fine-tuning claim"
        ),
        **result_identity,
        generator_id=train.generator_id,
        world_id=train.world_id,
        row_token_count=row_token_count,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        device=str(resolved_device),
        initial_train_loss=float(initial_train),
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
    )


def run_query_row_multi_world_fit(
    *,
    seed: int = 1729,
    steps: int = 20,
    learning_rate: float = 1.0e-2,
    train_worlds: int = 8,
    validation_worlds: int = 4,
    rows: int = 24,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
    config: ReferenceConfig | None = None,
) -> QueryRowSyntheticMultiWorldFitResult:
    """Run bounded S1 fitting on train worlds and disjoint validation worlds."""

    if steps <= 0 or train_worlds <= 0 or validation_worlds <= 0:
        raise ValueError("steps and world counts must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    torch.manual_seed(seed)
    train_episodes = tuple(
        make_query_row_synthetic_episode(
            seed=seed + 10 + index,
            rows=rows,
            row_token_count=row_token_count,
            world_id=f"train-world-{index}",
            world_family=("row_latent_linear" if index % 2 == 0 else "row_latent_polynomial"),
        )
        for index in range(train_worlds)
    )
    validation_episodes = tuple(
        make_query_row_synthetic_episode(
            seed=seed + 100 + index,
            rows=rows,
            row_token_count=row_token_count,
            world_id=f"heldout-world-{index}",
            world_family="row_latent_periodic",
        )
        for index in range(validation_worlds)
    )
    model = build_model(
        "tabu.query.row",
        config=config
        or ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=row_token_count,
            max_features=4,
        ),
        profile="completion.artificial_mask.v1",
        row_token_count=row_token_count,
    ).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def mean_loss(episodes: tuple[QueryRowSyntheticEpisode, ...]) -> Tensor:
        losses = [_episode_loss(model, episode) for episode in episodes]
        return torch.stack(losses).mean()

    model.train()
    with torch.no_grad():
        initial_validation = float(mean_loss(validation_episodes).item())
    initial_train = float(mean_loss(train_episodes).item())
    final_train = float("nan")
    for step in range(steps):
        loss = _episode_loss(model, train_episodes[step % len(train_episodes)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_train = float(loss.item())
    model.eval()
    with torch.no_grad():
        final_train = float(mean_loss(train_episodes).item())
        final_validation = float(mean_loss(validation_episodes).item())
    status = (
        "pass"
        if torch.isfinite(torch.tensor(final_train))
        and final_train < initial_train
        and torch.isfinite(torch.tensor(final_validation))
        else "kill"
    )
    result_identity = query_row_result_identity(model.checkpoint_identity())
    return QueryRowSyntheticMultiWorldFitResult(
        status=status,
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR bounded multi-world synthetic fit only; no real-data, frozen ICL, "
            "or fine-tuning claim"
        ),
        **result_identity,
        generator_id="tabur.row-latent-multi-world.v1",
        train_worlds=train_worlds,
        validation_worlds=validation_worlds,
        row_token_count=row_token_count,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        device=str(resolved_device),
        initial_train_loss=initial_train,
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
    )


__all__ = [
    "QueryRowSyntheticEpisode",
    "QueryRowSyntheticFitResult",
    "QueryRowSyntheticMultiWorldFitResult",
    "make_query_row_synthetic_episode",
    "run_query_row_fixed_world_fit",
    "run_query_row_multi_world_fit",
]
