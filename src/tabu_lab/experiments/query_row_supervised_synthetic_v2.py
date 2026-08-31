"""Diverse supervised synthetic prior v2 for TabUR transfer experiments.

World parameters are sampled before row generation and are addressed by an
explicit ``partition/world_id`` pair.  The model receives only a masked
``EvidenceEpisode``; generated query truth lives exclusively in the sidecar.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

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

from .query_row_identity import query_row_result_identity

GeneratorPartition = Literal["train", "validation"]
GENERATOR_ID = "tabur.supervised-query-row-diverse-v2"
WORLD_FAMILIES = (
    "sparse_additive",
    "sparse_dag_scm",
    "tree_threshold",
    "latent_factor",
    "polynomial_interaction",
    "periodic_saturating",
    "mixture_subgroup",
)
PREDICTOR_REGIMES = ("gaussian", "heavy_tailed", "skewed", "mixture", "quantized", "bounded")
WIDTH_BUCKETS = (6, 8, 9, 11, 17, 21, 32)
NOISE_LEVELS = ("low", "medium", "high")
CONTEXT_ANCHORS = (8, 16, 32, 64, 128, 256, 512)


@dataclass(frozen=True, slots=True)
class QueryRowSupervisedSyntheticV2Episode:
    evidence: EvidenceEpisode
    sidecar: TruthSidecar
    world_id: str
    partition: GeneratorPartition
    family: str
    width: int
    predictor_regime: str
    noise_level: str
    context_rows: int
    generator_id: str = GENERATOR_ID


def _seed(root_seed: int, *parts: object) -> int:
    encoded = "|".join((str(root_seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def _generator(root_seed: int, *parts: object) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(_seed(root_seed, *parts))


def _draw_predictors(width: int, rows: int, regime: str, generator: torch.Generator) -> Tensor:
    if regime == "gaussian":
        values = torch.randn(rows, width, generator=generator)
    elif regime == "heavy_tailed":
        numerator = torch.randn(rows, width, generator=generator)
        denominator = torch.rand(rows, width, generator=generator).clamp_min(1.0e-4).sqrt()
        values = numerator / denominator
    elif regime == "skewed":
        values = torch.exp(torch.randn(rows, width, generator=generator).clamp(-3.0, 3.0)) - 1.0
    elif regime == "mixture":
        base = torch.randn(rows, width, generator=generator)
        shift = (torch.rand(rows, width, generator=generator) > 0.7).to(base.dtype) * 2.0
        values = base + shift
    elif regime == "quantized":
        values = torch.randn(rows, width, generator=generator).mul(2.0).round().div(2.0)
    elif regime == "bounded":
        values = torch.tanh(torch.randn(rows, width, generator=generator))
    else:
        raise ValueError(f"unknown v2 predictor regime: {regime!r}")
    return values.to(dtype=torch.float32)


def _response(
    x: Tensor,
    *,
    family: str,
    generator: torch.Generator,
) -> Tensor:
    width = x.shape[1]
    active_count = min(width, int(torch.randint(2, min(6, width + 1), (1,), generator=generator)))
    active = torch.randperm(width, generator=generator)[:active_count]
    coefficients = torch.randn(active_count, generator=generator) / math.sqrt(active_count)
    xa = x[:, active]
    if family == "sparse_additive":
        response = (xa * coefficients).sum(dim=1)
    elif family == "sparse_dag_scm":
        a = x[:, 0]
        b = x[:, 1] + 0.45 * a
        c = x[:, 2] - 0.35 * b
        response = 0.7 * a + 0.4 * b - 0.25 * c
    elif family == "tree_threshold":
        response = torch.where(x[:, 0] > 0.0, 0.8 + 0.25 * x[:, 1], -0.7 + 0.2 * x[:, 2])
        response = response + torch.where(x[:, 3] > 0.5, 0.35, -0.15)
    elif family == "latent_factor":
        latent = torch.randn(x.shape[0], generator=generator)
        loading = torch.randn(width, generator=generator) / math.sqrt(width)
        response = 0.75 * latent + 0.25 * (x * loading).sum(dim=1)
    elif family == "polynomial_interaction":
        response = xa[:, 0].square() + 0.35 * xa[:, 0] * xa[:, 1]
        response = response - 0.15 * xa[:, -1].square()
    elif family == "periodic_saturating":
        response = torch.sin(xa[:, 0]) + 0.4 * torch.cos(xa[:, 1]) + 0.3 * torch.tanh(xa[:, -1])
    elif family == "mixture_subgroup":
        subgroup = x[:, 0] > 0.0
        left = 0.8 * xa[:, 0] - 0.25 * xa[:, 1]
        right = -0.5 * xa[:, 0] + 0.65 * xa[:, 1] + 0.7
        response = torch.where(subgroup, left, right)
    else:
        raise ValueError(f"unknown v2 world family: {family!r}")
    transform = int(torch.randint(0, 3, (1,), generator=generator))
    if transform == 1:
        response = torch.tanh(response)
    elif transform == 2:
        response = torch.sign(response) * response.abs().sqrt()
    return response


def _resolve_plan(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition,
    width: int | None,
    family: str | None,
    predictor_regime: str | None,
    noise_level: str | None,
    context_rows: int | None,
) -> tuple[int, str, str, str, int]:
    index_seed = _seed(root_seed, partition, world_id, "plan")
    index = index_seed % len(WORLD_FAMILIES)
    resolved_family = family or WORLD_FAMILIES[index]
    resolved_width = width or WIDTH_BUCKETS[index_seed % len(WIDTH_BUCKETS)]
    resolved_regime = (
        predictor_regime or PREDICTOR_REGIMES[(index_seed // 7) % len(PREDICTOR_REGIMES)]
    )
    resolved_noise = noise_level or NOISE_LEVELS[(index_seed // 17) % len(NOISE_LEVELS)]
    resolved_context = context_rows or CONTEXT_ANCHORS[(index_seed // 31) % len(CONTEXT_ANCHORS)]
    if resolved_family not in WORLD_FAMILIES:
        raise ValueError(f"unknown v2 world family: {resolved_family!r}")
    if resolved_width not in WIDTH_BUCKETS:
        raise ValueError(f"v2 width must be one of {WIDTH_BUCKETS}")
    if resolved_regime not in PREDICTOR_REGIMES:
        raise ValueError(f"unknown v2 predictor regime: {resolved_regime!r}")
    if resolved_noise not in NOISE_LEVELS:
        raise ValueError(f"unknown v2 noise level: {resolved_noise!r}")
    if resolved_context not in CONTEXT_ANCHORS:
        raise ValueError(f"v2 context_rows must be one of {CONTEXT_ANCHORS}")
    return resolved_width, resolved_family, resolved_regime, resolved_noise, resolved_context


def make_query_row_supervised_synthetic_v2_episode(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition = "train",
    width: int | None = None,
    family: str | None = None,
    predictor_regime: str | None = None,
    noise_level: str | None = None,
    context_rows: int | None = None,
    rows: int | None = None,
) -> QueryRowSupervisedSyntheticV2Episode:
    """Generate one deterministic, masked supervised query-row world."""

    if not world_id.strip():
        raise ValueError("world_id must not be empty")
    width, family, predictor_regime, noise_level, context_rows = _resolve_plan(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
        width=width,
        family=family,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=context_rows,
    )
    rows = rows or context_rows * 2
    if rows <= context_rows or context_rows < 1:
        raise ValueError("rows must be greater than positive context_rows")
    world_generator = _generator(root_seed, partition, world_id, "world")
    row_generator = _generator(root_seed, partition, world_id, "rows")
    predictors = _draw_predictors(width, rows, predictor_regime, row_generator)
    response = _response(predictors, family=family, generator=world_generator)
    signal_scale = response.std().clamp_min(1.0e-6)
    snr = {"low": 3.0, "medium": 10.0, "high": 30.0}[noise_level]
    response = response + signal_scale / snr * torch.randn(rows, generator=row_generator)
    permutation = torch.randperm(width, generator=world_generator)
    predictors = predictors[:, permutation]

    values = torch.cat((predictors, response.unsqueeze(1)), dim=1)
    target_mask = torch.zeros_like(values, dtype=torch.bool)
    target_mask[context_rows:, -1] = True
    forward_values = values.masked_fill(target_mask, 0.0)
    feature_names = (*tuple(f"feature_{index:03d}" for index in range(width)), "response")
    feature_specs = (
        *tuple(
            FeatureSpec(name=name, kind=FeatureKind.NUMERIC, role=FeatureRole.PREDICTOR)
            for name in feature_names[:-1]
        ),
        FeatureSpec(name="response", kind=FeatureKind.NUMERIC, role=FeatureRole.RESPONSE),
    )
    origins = [
        [
            OriginState.QUERY if target_mask[row, col] else OriginState.OBSERVED
            for col in range(width + 1)
        ]
        for row in range(rows)
    ]
    roles = [
        [
            ForwardRole.RECEIVER
            | (ForwardRole.TARGET if target_mask[row, col] else ForwardRole.SOURCE)
            for col in range(width + 1)
        ]
        for row in range(rows)
    ]
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v2",
        source_partition=f"synthetic_{partition}",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"{world_id}-row-{index}" for index in range(rows)),
        feature_names=feature_names,
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_id": GENERATOR_ID,
            "partition": partition,
            "world_id": world_id,
            "world_family": family,
            "predictor_regime": predictor_regime,
            "width": width,
            "noise_level": noise_level,
            "context_rows": context_rows,
            "truth_boundary": "sidecar_only",
            "feature_permutation": permutation.tolist(),
        },
    )
    sidecar = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.v2.recipe.v1",
                "generator_id": GENERATOR_ID,
                "partition": partition,
                "world_id": world_id,
                "family": family,
                "predictor_regime": predictor_regime,
                "width": width,
                "noise_level": noise_level,
                "context_rows": context_rows,
                "root_seed": root_seed,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=feature_names,
        target_values=values.masked_fill(~target_mask, 0.0),
        target_mask=target_mask,
        metadata={
            "truth_scope": "loss_only",
            "generator_id": GENERATOR_ID,
            "partition": partition,
            "world_id": world_id,
        },
    )
    return QueryRowSupervisedSyntheticV2Episode(
        evidence=evidence,
        sidecar=sidecar,
        world_id=world_id,
        partition=partition,
        family=family,
        width=width,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=context_rows,
    )


def build_query_row_supervised_synthetic_v2_plan(
    *,
    root_seed: int,
    worlds: int,
    partition: GeneratorPartition,
) -> tuple[dict[str, Any], ...]:
    """Freeze world assignments before any rows are generated."""

    if worlds <= 0:
        raise ValueError("worlds must be positive")
    plan: list[dict[str, Any]] = []
    partition_offset = 3 if partition == "validation" else 0
    for index in range(worlds):
        world_id = f"{partition}-world-{index:06d}"
        family = WORLD_FAMILIES[(index + partition_offset) % len(WORLD_FAMILIES)]
        width = WIDTH_BUCKETS[(3 * index + partition_offset) % len(WIDTH_BUCKETS)]
        regime = PREDICTOR_REGIMES[(5 * index + partition_offset) % len(PREDICTOR_REGIMES)]
        noise = NOISE_LEVELS[(index + partition_offset) % len(NOISE_LEVELS)]
        context = CONTEXT_ANCHORS[(11 * index + partition_offset) % len(CONTEXT_ANCHORS)]
        plan.append(
            {
                "world_id": world_id,
                "partition": partition,
                "family": family,
                "width": width,
                "predictor_regime": regime,
                "noise_level": noise,
                "context_rows": context,
            }
        )
    return tuple(plan)


def validate_query_row_supervised_synthetic_v2(
    *,
    root_seed: int = 1729,
    worlds: int = 512,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run the R4 generator exits on CPU float32 (or an explicit device)."""

    if worlds <= 0:
        raise ValueError("worlds must be positive")
    train_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed, worlds=worlds, partition="train"
    )
    validation_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed, worlds=worlds, partition="validation"
    )
    train_ids = {item["world_id"] for item in train_plan}
    validation_ids = {item["world_id"] for item in validation_plan}
    coverage = {
        "families": sorted({item["family"] for item in train_plan}),
        "widths": sorted({item["width"] for item in train_plan}),
        "predictor_regimes": sorted({item["predictor_regime"] for item in train_plan}),
        "noise_levels": sorted({item["noise_level"] for item in train_plan}),
        "context_rows": sorted({item["context_rows"] for item in train_plan}),
    }
    coverage_ok = (
        set(coverage["families"]) == set(WORLD_FAMILIES)
        and set(coverage["widths"]) == set(WIDTH_BUCKETS)
        and set(coverage["predictor_regimes"]) == set(PREDICTOR_REGIMES)
        and set(coverage["noise_levels"]) == set(NOISE_LEVELS)
        and set(coverage["context_rows"]) == set(CONTEXT_ANCHORS)
    )
    sample_spec = train_plan[0]
    sample = make_query_row_supervised_synthetic_v2_episode(
        root_seed=root_seed,
        world_id=sample_spec["world_id"],
        partition="train",
        width=sample_spec["width"],
        family=sample_spec["family"],
        predictor_regime=sample_spec["predictor_regime"],
        noise_level=sample_spec["noise_level"],
        context_rows=sample_spec["context_rows"],
    )
    replay = make_query_row_supervised_synthetic_v2_episode(
        root_seed=root_seed,
        world_id=sample_spec["world_id"],
        partition="train",
        width=sample_spec["width"],
        family=sample_spec["family"],
        predictor_regime=sample_spec["predictor_regime"],
        noise_level=sample_spec["noise_level"],
        context_rows=sample_spec["context_rows"],
    )
    substituted = substitute_query_truth(sample, value=123.0)
    replay_ok = sample.evidence.evidence_hash == replay.evidence.evidence_hash
    truth_isolated = (
        sample.evidence.evidence_hash == substituted.evidence.evidence_hash
        and torch.equal(sample.evidence.forward_values, substituted.evidence.forward_values)
        and not torch.equal(sample.sidecar.target_values, substituted.sidecar.target_values)
        and bool((sample.evidence.forward_values[sample.evidence.target_mask] == 0).all())
    )
    from tabu_lab.models import build_model
    from tabu_lab.models.types import ReferenceConfig

    resolved_device = torch.device(device)
    model = build_model(
        "tabu.query.row",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=row_token_count,
            max_features=256,
        ),
        profile="supervised.label_broadcast.v1",
        row_token_count=row_token_count,
    ).to(resolved_device)
    loss = supervised_synthetic_v2_episode_loss(model, sample)
    model.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    with torch.no_grad():
        public_prediction = model(sample.evidence)
        dense_prediction = model._forward_dense(
            sample.evidence.to(resolved_device), emit_trace=False
        )
    dense_parity = torch.allclose(
        public_prediction["numeric_raw_prediction"],
        dense_prediction["numeric_raw_prediction"],
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    oracle_error = (
        torch.where(
            sample.sidecar.target_mask,
            sample.sidecar.target_values - sample.sidecar.target_values,
            torch.zeros_like(sample.sidecar.target_values),
        )
        .square()
        .sum()
        .item()
    )
    exits = {
        "deterministic_replay": replay_ok,
        "truth_substitution_isolation": truth_isolated,
        "coverage": coverage_ok,
        "known_reference_oracle": oracle_error == 0.0,
        "dense_chunked_prediction_parity": bool(dense_parity),
        "finite_forward_backward": bool(torch.isfinite(loss)) and finite_gradients,
        "disjoint_train_validation_world_ids": train_ids.isdisjoint(validation_ids),
    }
    result_identity = query_row_result_identity(model.checkpoint_identity())
    return {
        **result_identity,
        "schema_version": "tabu.query-row.supervised-synthetic-v2-validation.v2",
        "generator_id": GENERATOR_ID,
        "root_seed": root_seed,
        "worlds_per_partition": worlds,
        "coverage": coverage,
        "exits": exits,
        "status": "passed" if all(exits.values()) else "failed",
        "evidence_status": "local_unissued",
        "claim_boundary": "R4 generator validation only; no pretraining or capability claim",
    }


def supervised_synthetic_v2_episode_loss(
    model: Any,
    episode: QueryRowSupervisedSyntheticV2Episode,
) -> Tensor:
    """Score the explicit context-inverted raw auxiliary against sidecar truth."""

    prediction = model(episode.evidence)
    raw = prediction["numeric_raw_prediction"]
    support = prediction["numeric_support_available"].to(torch.bool)
    if raw.ndim == 2:
        raw = raw.unsqueeze(0)
        support = support.unsqueeze(0)
    truth = episode.sidecar.target_values.to(device=raw.device).unsqueeze(0)
    target = episode.sidecar.target_mask.to(device=raw.device).unsqueeze(0)
    scored = target & support
    if not bool(scored.any()):
        raise RuntimeError("v2 episode has no supported supervised target")
    error = raw - truth.to(dtype=raw.dtype)
    return torch.where(
        scored, error.square(), torch.zeros_like(error)
    ).sum() / scored.sum().clamp_min(1)


def substitute_query_truth(
    episode: QueryRowSupervisedSyntheticV2Episode,
    *,
    value: float,
) -> QueryRowSupervisedSyntheticV2Episode:
    """Return a truth-substituted sidecar while preserving evidence byte identity."""

    values = episode.sidecar.target_values.clone()
    values[episode.sidecar.target_mask] = value
    return replace(episode, sidecar=replace(episode.sidecar, target_values=values))


__all__ = [
    "CONTEXT_ANCHORS",
    "GENERATOR_ID",
    "NOISE_LEVELS",
    "PREDICTOR_REGIMES",
    "QueryRowSupervisedSyntheticV2Episode",
    "WORLD_FAMILIES",
    "WIDTH_BUCKETS",
    "build_query_row_supervised_synthetic_v2_plan",
    "make_query_row_supervised_synthetic_v2_episode",
    "substitute_query_truth",
    "supervised_synthetic_v2_episode_loss",
    "validate_query_row_supervised_synthetic_v2",
]
