"""Executable frozen-ICL diagnostic for TabUBase 0.2.0.

The frozen arms never construct an optimizer.  Held-out worlds use the
heteroscedastic/missingness family excluded from synthetic pretraining, and
all paired arms see identical world ids, context sizes, and query rows.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    ForwardRole,
    OriginState,
    TruthSidecar,
    canonical_hash,
    origin_code,
)
from tabu_lab.models.components import CellTokenizer
from tabu_lab.models.types import DenseModelInput

from .tabubase_scale import (
    _sha256_file,
    _stable_seed,
    _standardize_from_context,
    _state_hash,
    build_tabubase_scale_model,
    load_pretrain_checkpoint,
)

K_GRID = (0, 1, 2, 4, 8, 16, 32)
FROZEN_ARMS = ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
HELDOUT_FAMILY = "heteroscedastic_missingness_shift"
_MAX_CONTEXT = max(K_GRID)
_N_PREDICTORS = 8
_RESPONSE_FEATURE = _N_PREDICTORS


@dataclass(frozen=True, slots=True)
class HeldoutIclCase:
    world_id: str
    modality: Literal["classification", "regression"]
    target_type: Literal["binary", "categorical", "numeric"]
    context_size: int
    episode: EvidenceEpisode
    truth: TruthSidecar
    query_values: torch.Tensor
    classes: int | None
    target_scale: float


@dataclass(frozen=True, slots=True)
class FrozenIclConfig:
    checkpoint: Path
    output_dir: Path
    seed: int = 1729
    heldout_worlds: int = 512
    query_rows: int = 32
    batch_size: int = 32
    bootstrap_replicates: int = 2_000
    nominal_codebook_size: int = 100
    nominal_codebook_seed: int = 1729
    world_scope: Literal["heldout", "train_mixture"] = "heldout"
    world_seed: int = 1729

    def validate(self) -> FrozenIclConfig:
        if not self.checkpoint.is_file():
            raise ValueError("pretrained checkpoint does not exist")
        if self.heldout_worlds < 2 or self.heldout_worlds % 2:
            raise ValueError("heldout_worlds must be an even integer of at least two")
        if self.query_rows < 1 or self.batch_size < 1:
            raise ValueError("query_rows and batch_size must be positive")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if self.nominal_codebook_size != 100:
            raise ValueError("the frozen v2 ICL diagnostic requires a 100-code codebook")
        if self.world_scope not in {"heldout", "train_mixture"}:
            raise ValueError("world_scope must be heldout or train_mixture")
        if self.seed < 0 or self.world_seed < 0:
            raise ValueError("checkpoint and world seeds must be non-negative")
        return self


def _feature_specs(target_type: str) -> tuple[Any, ...]:
    from tabu_lab.contracts import FeatureSpec

    predictors = tuple(FeatureSpec(name=f"numeric_{index}") for index in range(4))
    predictors += tuple(
        FeatureSpec(
            name=f"ordinal_{index}",
            kind=FeatureKind.ORDINAL,
            domain=("low", "middle", "high"),
            codebook_id=f"tabubase-icl-heldout-ordinal-{index}-v1",
        )
        for index in range(2)
    )
    predictors += tuple(
        FeatureSpec(
            name=f"nominal_{index}",
            kind=FeatureKind.CATEGORICAL,
            domain=("n0", "n1", "n2", "n3"),
            codebook_id=f"tabubase-icl-heldout-nominal-{index}-v1",
        )
        for index in range(2)
    )
    if target_type == "numeric":
        response = FeatureSpec(name="response", role=FeatureRole.RESPONSE)
    else:
        classes = 2 if target_type == "binary" else 4
        response = FeatureSpec(
            name="response",
            kind=FeatureKind.CATEGORICAL,
            domain=tuple(f"class_{index}" for index in range(classes)),
            codebook_id=f"tabubase-icl-heldout-response-{classes}-v1",
            role=FeatureRole.RESPONSE,
        )
    return (*predictors, response)


def _context_standardize(values: torch.Tensor, context_rows: int) -> torch.Tensor:
    # K=0 and K=1 cannot identify a variance.  Identity scaling is the frozen,
    # query-blind fallback; no query/population statistics are borrowed.
    if context_rows < 2:
        return values
    return _standardize_from_context(values, context_rows)


def build_heldout_icl_case(
    *,
    root_seed: int,
    world_index: int,
    modality: Literal["classification", "regression"],
    context_size: int,
    query_rows: int = 32,
    shuffled_context: bool = False,
    world_scope: Literal["heldout", "train_mixture"] = "heldout",
) -> HeldoutIclCase:
    """Compile one nested-context world without query-statistic leakage."""

    if context_size not in K_GRID or world_index < 0 or query_rows < 1:
        raise ValueError("invalid held-out ICL world arguments")
    if world_scope not in {"heldout", "train_mixture"}:
        raise ValueError("world_scope must be heldout or train_mixture")
    world_id = f"tabubase-icl-{world_scope}-{modality}-{world_index:04d}"
    generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(root_seed, f"heldout:{world_id}")
    )
    total_rows = _MAX_CONTEXT + query_rows
    latent = torch.randn((total_rows, 4), generator=generator, dtype=torch.float32)
    raw_numeric = torch.stack(
        (
            latent[:, 0],
            latent[:, 1],
            latent[:, 2] + 0.3 * latent[:, 0],
            latent[:, 3] + 0.25 * latent[:, 1].square(),
        ),
        dim=1,
    )
    ordinal = torch.stack(
        (
            torch.bucketize(latent[:, 0].contiguous(), torch.tensor((-0.5, 0.5))),
            torch.bucketize(
                (latent[:, 1] + 0.25 * latent[:, 2]).contiguous(),
                torch.tensor((-0.5, 0.5)),
            ),
        ),
        dim=1,
    ).to(torch.float32)
    nominal = torch.stack(
        (
            (latent[:, 2] > 0).to(torch.int64) + 2 * (latent[:, 3] > 0).to(torch.int64),
            torch.remainder(
                torch.bucketize(
                    (latent[:, 0] - latent[:, 1]).contiguous(),
                    torch.tensor((-0.7, 0.0, 0.7)),
                ),
                4,
            ),
        ),
        dim=1,
    ).to(torch.float32)
    if world_scope == "heldout":
        family = HELDOUT_FAMILY
        heteroscedastic_scale = 0.08 + 0.22 * latent[:, 0].abs()
        noise = torch.randn(total_rows, generator=generator) * heteroscedastic_scale
        score = 0.75 * latent[:, 0] - 0.35 * latent[:, 1].square()
        score = score + 0.3 * torch.sin(latent[:, 2] + latent[:, 0])
        score = score + 0.2 * (nominal[:, 0] == 3).to(torch.float32) + noise
        target_scale = math.sqrt(0.75**2 + 2.0 * 0.35**2 + 0.3**2 + 0.2**2 + 0.2**2)
        score = score / target_scale
    else:
        family = ("sparse_scm", "tree_threshold", "latent_factor")[world_index % 3]
        train_numeric = _context_standardize(raw_numeric, context_size)
        if family == "sparse_scm":
            score = 0.9 * train_numeric[:, 0] - 0.45 * train_numeric[:, 1].square()
            score = score + 0.35 * torch.tanh(train_numeric[:, 2])
            score = score + 0.2 * (nominal[:, 0] == 3).to(torch.float32)
        elif family == "tree_threshold":
            score = 0.8 * (train_numeric[:, 0] > 0).to(torch.float32)
            score = score - 0.6 * (train_numeric[:, 1] > 0.5).to(torch.float32)
            score = score + 0.35 * ordinal[:, 0] + 0.15 * train_numeric[:, 3]
        else:
            factor = 0.7 * train_numeric[:, 0] + 0.5 * train_numeric[:, 1]
            score = factor + 0.35 * torch.sin(train_numeric[:, 2] + factor)
            score = score + 0.15 * nominal[:, 1]
        # The train generator standardizes numeric targets from context.  The
        # paired ICL diagnostic uses the same query-blind rule and K<2 fallback.
        score = _context_standardize(score.unsqueeze(1), context_size).squeeze(1)

    if modality == "regression":
        target_type: Literal["binary", "categorical", "numeric"] = "numeric"
        response = score
        classes = None
    elif world_index % 2 == 0:
        target_type = "binary"
        response = (score > 0.0).to(torch.float32)
        classes = 2
    else:
        target_type = "categorical"
        response = torch.bucketize(score.contiguous(), torch.tensor((-0.75, 0.0, 0.75))).to(
            torch.float32
        )
        classes = 4

    chosen = torch.cat(
        (
            torch.arange(context_size),
            torch.arange(_MAX_CONTEXT, total_rows),
        )
    )
    numeric = _context_standardize(raw_numeric[chosen], context_size)
    predictors = torch.cat((numeric, ordinal[chosen], nominal[chosen]), dim=1)
    selected_response = response[chosen].clone()
    if shuffled_context and context_size > 1:
        permutation = torch.randperm(
            context_size,
            generator=torch.Generator(device="cpu").manual_seed(
                _stable_seed(root_seed, f"shuffle:{world_id}:k={context_size}")
            ),
        )
        selected_response[:context_size] = selected_response[:context_size][permutation]
    values = torch.cat((predictors, selected_response.unsqueeze(1)), dim=1)
    rows = context_size + query_rows
    roles = torch.full(
        (rows, _N_PREDICTORS + 1),
        int(ForwardRole.RECEIVER | ForwardRole.SOURCE),
        dtype=torch.int64,
    )
    origins = torch.full(
        roles.shape,
        origin_code(OriginState.OBSERVED),
        dtype=torch.int64,
    )

    # Missingness belongs only to predictors.  It is generated before episode
    # selection and remains paired across K and arms.
    missing_generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(root_seed, f"missing:{world_id}")
    )
    base_missing = torch.rand((total_rows, _N_PREDICTORS), generator=missing_generator)
    missing_probability = 0.04 + 0.08 * torch.sigmoid(latent[:, :1])
    missing = (
        (base_missing < missing_probability)[chosen]
        if world_scope == "heldout"
        else torch.zeros((rows, _N_PREDICTORS), dtype=torch.bool)
    )
    roles[:, :_N_PREDICTORS][missing] = int(ForwardRole.RECEIVER)
    origins[:, :_N_PREDICTORS][missing] = origin_code(OriginState.NATURAL_MISSING)
    values[:, :_N_PREDICTORS][missing] = 0.0

    roles[context_size:, _RESPONSE_FEATURE] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    origins[context_size:, _RESPONSE_FEATURE] = origin_code(OriginState.QUERY)
    values[context_size:, _RESPONSE_FEATURE] = 0.0
    episode_id = f"{world_id}-k{context_size}-{'shuffled' if shuffled_context else 'normal'}"
    row_ids = tuple(f"{episode_id}-row-{index:03d}" for index in range(rows))
    specs = _feature_specs(target_type)
    episode = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id=(
            "tabubase-synthetic-heldout-heteroscedastic-missingness-v1"
            if world_scope == "heldout"
            else "tabubase-synthetic-train-family-replay-v1"
        ),
        source_partition="heldout_family" if world_scope == "heldout" else "train_family_replay",
        fit_partition="train",
        row_ids=row_ids,
        feature_names=tuple(spec.name for spec in specs),
        feature_specs=specs,
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_family": family,
            "world_scope": world_scope,
            "response_family": target_type,
            "statistics_scope": "context_only_with_identity_fallback_for_k_lt_2",
            "world_id": world_id,
            "context_size": context_size,
            "context_shuffled": shuffled_context,
        },
    )
    truth_values = torch.zeros_like(values)
    query_values = response[_MAX_CONTEXT:].clone()
    truth_values[context_size:, _RESPONSE_FEATURE] = query_values
    truth = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabubase-heldout-icl-recipe.v1",
                "root_seed": root_seed,
                "world_id": world_id,
                "context_size": context_size,
                "query_rows": query_rows,
                "shuffled_context": shuffled_context,
                "world_scope": world_scope,
            }
        ),
        row_ids=row_ids,
        feature_names=episode.feature_names,
        target_values=truth_values,
        target_mask=episode.target_mask,
    )
    return HeldoutIclCase(
        world_id=world_id,
        modality=modality,
        target_type=target_type,
        context_size=context_size,
        episode=episode,
        truth=truth,
        query_values=query_values,
        classes=classes,
        target_scale=1.0,
    )


def _batch_input(cases: Sequence[HeldoutIclCase], device: torch.device) -> DenseModelInput:
    if not cases:
        raise ValueError("cannot batch an empty ICL case sequence")
    dense = tuple(DenseModelInput.from_any(case.episode) for case in cases)
    first = dense[0]
    if any(item.feature_specs != first.feature_specs for item in dense[1:]):
        raise ValueError("batched ICL cases must share one feature schema")
    return DenseModelInput(
        values=torch.cat(tuple(item.values for item in dense), dim=0),
        visible_mask=torch.cat(tuple(item.visible_mask for item in dense), dim=0),
        target_mask=torch.cat(tuple(item.target_mask for item in dense), dim=0),
        natural_missing_mask=torch.cat(tuple(item.natural_missing_mask for item in dense), dim=0),
        artificial_target_mask=torch.cat(
            tuple(
                item.artificial_target_mask
                for item in dense
                if item.artificial_target_mask is not None
            ),
            dim=0,
        ),
        query_target_mask=torch.cat(
            tuple(item.query_target_mask for item in dense if item.query_target_mask is not None),
            dim=0,
        ),
        unsupported_target_mask=torch.cat(
            tuple(
                item.unsupported_target_mask
                for item in dense
                if item.unsupported_target_mask is not None
            ),
            dim=0,
        ),
        feature_specs=first.feature_specs,
        row_ids=first.row_ids,
        target_feature=_RESPONSE_FEATURE,
        episode_id=f"tabubase-icl-batch-{cases[0].target_type}-k{cases[0].context_size}",
        metadata={"profile_id": "supervised.label_broadcast.v1"},
        squeezed_batch=False,
    ).to(device)


def _score_batch(
    model: torch.nn.Module,
    cases: Sequence[HeldoutIclCase],
    *,
    device: torch.device,
) -> list[float]:
    if cases[0].context_size == 0:
        return [
            1.0
            if case.modality == "classification"
            else float(case.query_values.square().mean().sqrt())
            for case in cases
        ]
    inputs = _batch_input(cases, device)
    with torch.inference_mode():
        prediction = model._forward_dense(inputs, emit_trace=False)
    query = slice(cases[0].context_size, None)
    if cases[0].modality == "regression":
        values = prediction.entries["numeric"].values
        if values is None:
            raise RuntimeError("numeric frozen ICL arm returned no values")
        predicted = values[:, query, _RESPONSE_FEATURE].detach().cpu()
        return [
            float((predicted[index] - case.query_values).square().mean().sqrt() / case.target_scale)
            for index, case in enumerate(cases)
        ]
    probabilities = prediction.entries["distribution"].values
    if probabilities is None:
        raise RuntimeError("categorical frozen ICL arm returned no distribution")
    probabilities = probabilities[:, query, _RESPONSE_FEATURE].detach().cpu()
    metrics: list[float] = []
    for index, case in enumerate(cases):
        assert case.classes is not None
        codes = case.query_values.round().to(torch.int64)
        selected = probabilities[index, :, : case.classes].gather(1, codes.unsqueeze(1)).squeeze(1)
        nll = -selected.clamp_min(1.0e-8).log().mean()
        metrics.append(float(nll / math.log(case.classes)))
    return metrics


def paired_aulc(values: Sequence[float]) -> float:
    if len(values) != len(K_GRID) or any(not math.isfinite(value) for value in values):
        raise ValueError("AULC requires one finite value at every frozen K")
    return float(sum((left + right) * 0.5 for left, right in pairwise(values)) / (len(values) - 1))


def paired_world_bootstrap(
    gains: Sequence[float], *, replicates: int = 2_000, seed: int = 1729
) -> tuple[float, float, float]:
    if len(gains) == 0 or any(not math.isfinite(value) for value in gains):
        raise ValueError("paired bootstrap requires finite world gains")
    if replicates < 100:
        raise ValueError("paired bootstrap requires at least 100 replicates")
    values = np.asarray(gains, dtype=np.float64)
    generator = np.random.default_rng(_stable_seed(seed, "icl-world-bootstrap"))
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        samples[index] = values[generator.integers(0, len(values), len(values))].mean()
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _evaluate_arm(
    model: torch.nn.Module,
    *,
    root_seed: int,
    worlds_per_modality: int,
    query_rows: int,
    batch_size: int,
    device: torch.device,
    shuffled_context: bool,
    world_scope: Literal["heldout", "train_mixture"],
) -> dict[str, dict[str, list[float]]]:
    observations: dict[str, dict[str, list[float]]] = {
        "classification": {},
        "regression": {},
    }
    model.eval()
    for modality in ("classification", "regression"):
        for context_size in K_GRID:
            cases = [
                build_heldout_icl_case(
                    root_seed=root_seed,
                    world_index=index,
                    modality=modality,
                    context_size=context_size,
                    query_rows=query_rows,
                    shuffled_context=shuffled_context,
                    world_scope=world_scope,
                )
                for index in range(worlds_per_modality)
            ]
            metrics = [0.0] * len(cases)
            # Binary and four-class worlds have distinct response schemas.
            for target_type in sorted({case.target_type for case in cases}):
                indices = [
                    index for index, case in enumerate(cases) if case.target_type == target_type
                ]
                for offset in range(0, len(indices), batch_size):
                    selected = indices[offset : offset + batch_size]
                    scores = _score_batch(
                        model, [cases[index] for index in selected], device=device
                    )
                    for index, score in zip(selected, scores, strict=True):
                        metrics[index] = score
            observations[modality][str(context_size)] = metrics
    return observations


def _run_frozen_arm(
    model: torch.nn.Module,
    *,
    root_seed: int,
    worlds_per_modality: int,
    query_rows: int,
    batch_size: int,
    device: torch.device,
    shuffled_context: bool,
    world_scope: Literal["heldout", "train_mixture"],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, str | bool]]:
    """Evaluate one frozen arm with state hashes immediately around its invocation."""

    model.requires_grad_(False)
    model.eval()
    before = _state_hash(model)
    with torch.inference_mode():
        observations = _evaluate_arm(
            model,
            root_seed=root_seed,
            worlds_per_modality=worlds_per_modality,
            query_rows=query_rows,
            batch_size=batch_size,
            device=device,
            shuffled_context=shuffled_context,
            world_scope=world_scope,
        )
    after = _state_hash(model)
    return observations, {"before": before, "after": after, "unchanged": before == after}


def _summarize(
    observations: dict[str, dict[str, dict[str, list[float]]]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for modality in ("classification", "regression"):
        world_count = len(observations["pretrained_frozen"][modality]["0"])
        curves = {
            arm: [
                [observations[arm][modality][str(k)][world] for k in K_GRID]
                for world in range(world_count)
            ]
            for arm in FROZEN_ARMS
        }
        aulc = {
            arm: [paired_aulc(curve) for curve in arm_curves] for arm, arm_curves in curves.items()
        }
        random_gain = [
            random - pretrained
            for random, pretrained in zip(
                aulc["random_init_frozen"], aulc["pretrained_frozen"], strict=True
            )
        ]
        shuffled_gain = [
            shuffled - pretrained
            for shuffled, pretrained in zip(
                aulc["pretrained_shuffled"], aulc["pretrained_frozen"], strict=True
            )
        ]
        random_ci = paired_world_bootstrap(random_gain, replicates=bootstrap_replicates, seed=seed)
        shuffled_ci = paired_world_bootstrap(
            shuffled_gain, replicates=bootstrap_replicates, seed=seed + 1
        )
        means = {
            arm: {str(k): float(np.mean(observations[arm][modality][str(k)])) for k in K_GRID}
            for arm in FROZEN_ARMS
        }
        summary[modality] = {
            "metric": "normalized_nll" if modality == "classification" else "scaled_rmse",
            "mean_curve": means,
            "mean_aulc": {arm: float(np.mean(values)) for arm, values in aulc.items()},
            "pretrained_vs_random_aulc_gain": {
                "mean": random_ci[0],
                "lower_95": random_ci[1],
                "upper_95": random_ci[2],
            },
            "normal_vs_shuffled_aulc_gain": {
                "mean": shuffled_ci[0],
                "lower_95": shuffled_ci[1],
                "upper_95": shuffled_ci[2],
            },
            "endpoint_k32": {arm: means[arm]["32"] for arm in FROZEN_ARMS},
            "gate": {
                "pretrained_vs_random_lower_95_gt_zero": random_ci[1] > 0.0,
                "normal_vs_shuffled_lower_95_gt_zero": shuffled_ci[1] > 0.0,
            },
        }
    return summary


def run_frozen_icl(config: FrozenIclConfig, *, device: torch.device) -> dict[str, Any]:
    """Run the three optimizer-free Link-5 arms and emit local-unissued evidence."""

    config.validate()
    started = time.monotonic()
    models = {
        arm: build_tabubase_scale_model(
            seed=config.seed,
            device=device,
            nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
            nominal_codebook_size=config.nominal_codebook_size,
            nominal_codebook_seed=config.nominal_codebook_seed,
        )
        for arm in FROZEN_ARMS
    }
    load_pretrain_checkpoint(models["pretrained_frozen"], config.checkpoint)
    load_pretrain_checkpoint(models["pretrained_shuffled"], config.checkpoint)
    identity = models["pretrained_frozen"].checkpoint_identity()
    if identity.get("tokenizer_version") != "cell-tokenizer.v2":
        raise ValueError("frozen ICL checkpoint is not tokenizer v2")
    if identity.get("nominal_codebook_size") != config.nominal_codebook_size:
        raise ValueError("frozen ICL checkpoint codebook size differs from the run contract")
    worlds_per_modality = config.heldout_worlds // 2
    observations: dict[str, dict[str, dict[str, list[float]]]] = {}
    per_arm_parameter_hashes: dict[str, dict[str, str | bool]] = {}
    for arm, shuffled_context in (
        ("pretrained_frozen", False),
        ("random_init_frozen", False),
        ("pretrained_shuffled", True),
    ):
        observations[arm], per_arm_parameter_hashes[arm] = _run_frozen_arm(
            models[arm],
            root_seed=config.world_seed,
            worlds_per_modality=worlds_per_modality,
            query_rows=config.query_rows,
            batch_size=config.batch_size,
            device=device,
            shuffled_context=shuffled_context,
            world_scope=config.world_scope,
        )
    before = {arm: str(per_arm_parameter_hashes[arm]["before"]) for arm in FROZEN_ARMS}
    after = {arm: str(per_arm_parameter_hashes[arm]["after"]) for arm in FROZEN_ARMS}
    parameter_hash_unchanged = {
        arm: bool(per_arm_parameter_hashes[arm]["unchanged"]) for arm in FROZEN_ARMS
    }
    all_frozen_arm_parameter_hashes_unchanged = all(parameter_hash_unchanged.values())
    summary = _summarize(
        observations,
        bootstrap_replicates=config.bootstrap_replicates,
        seed=config.world_seed,
    )
    full_scale = config.heldout_worlds == 512
    gates = {
        "full_512_world_panel": full_scale,
        "all_frozen_parameter_hashes_unchanged": all_frozen_arm_parameter_hashes_unchanged,
        "classification_pretrained_vs_random": summary["classification"]["gate"][
            "pretrained_vs_random_lower_95_gt_zero"
        ],
        "classification_normal_vs_shuffled": summary["classification"]["gate"][
            "normal_vs_shuffled_lower_95_gt_zero"
        ],
        "regression_pretrained_vs_random": summary["regression"]["gate"][
            "pretrained_vs_random_lower_95_gt_zero"
        ],
        "regression_normal_vs_shuffled": summary["regression"]["gate"][
            "normal_vs_shuffled_lower_95_gt_zero"
        ],
    }
    receipt: dict[str, Any] = {
        "schema_version": "tabu.transfer-base-frozen-icl-local-unissued.v1",
        "status": "local_unissued",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
        "tokenizer_version": "cell-tokenizer.v2",
        "nominal_codebook_size": config.nominal_codebook_size,
        "nominal_codebook_seed": config.nominal_codebook_seed,
        "checkpoint": str(config.checkpoint),
        "checkpoint_sha256": _sha256_file(config.checkpoint),
        "selection_rule": "lowest PT-S1 validation loss before held-out ICL evaluation",
        "seed": config.seed,
        "checkpoint_seed": config.seed,
        "world_seed": config.world_seed,
        "world_scope": config.world_scope,
        "generator_family": (
            HELDOUT_FAMILY
            if config.world_scope == "heldout"
            else "training_mixture_sparse_scm_tree_threshold_latent_factor"
        ),
        "heldout_worlds": config.heldout_worlds,
        "worlds_per_modality": worlds_per_modality,
        "context_sizes": list(K_GRID),
        "query_rows_per_world": config.query_rows,
        "arms_declared": [*FROZEN_ARMS, "scratch_finetune"],
        "arms_executed": list(FROZEN_ARMS),
        "scratch_finetune_status": "pending_separate_non_frozen_reference",
        "frozen_arm_optimizer_created": False,
        "per_arm_parameter_hashes": per_arm_parameter_hashes,
        "all_frozen_arm_parameter_hashes_unchanged": (all_frozen_arm_parameter_hashes_unchanged),
        "parameter_hash_before": before,
        "parameter_hash_after": after,
        "parameter_hash_unchanged": parameter_hash_unchanged,
        "k0_semantics": {
            "classification": "uniform_over_declared_classes",
            "regression": "zero_in_generator_standardized_space",
        },
        "summary": summary,
        "gates": gates,
        "passed_primary_frozen_gate": all(gates.values()),
        "observations": observations,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "runtime_backend": os.environ.get("WEHUB_RUNTIME_BACKEND"),
            "runtime_image": os.environ.get("WEHUB_RUNTIME_IMAGE"),
        },
        "git_commit": os.environ.get("TABU_SOURCE_COMMIT")
        or subprocess.run(
            ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=False
        ).stdout.strip()
        or None,
        "claim_boundary": (
            "optimizer-free synthetic ICL diagnostic only; scratch-finetune reference "
            "and independent replay remain open; not a formal receipt or foundation-model claim"
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "result.json"
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt | {"result_path": str(output_path), "result_sha256": _sha256_file(output_path)}


def aggregate_common_world_results(
    result_paths: Sequence[Path],
    *,
    output_path: Path,
    bootstrap_replicates: int = 2_000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Aggregate checkpoint seeds while clustering repeated observations by world id."""

    if len(result_paths) < 2 or len(set(result_paths)) != len(result_paths):
        raise ValueError("common-world aggregation requires distinct result paths")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    world_seeds = {int(item.get("world_seed", item.get("seed", -1))) for item in results}
    checkpoint_seeds = tuple(int(item.get("checkpoint_seed", item["seed"])) for item in results)
    if len(world_seeds) != 1 or len(set(checkpoint_seeds)) != len(checkpoint_seeds):
        raise ValueError("results must share one world seed and use distinct checkpoint seeds")
    if any(tuple(item["context_sizes"]) != K_GRID for item in results):
        raise ValueError("results do not share the frozen K grid")
    worlds_per_modality = {int(item["worlds_per_modality"]) for item in results}
    if len(worlds_per_modality) != 1:
        raise ValueError("results do not share one world panel")
    world_count = next(iter(worlds_per_modality))
    modalities: dict[str, Any] = {}
    for modality_index, modality in enumerate(("classification", "regression")):
        random_gains_by_seed: list[list[float]] = []
        shuffled_gains_by_seed: list[list[float]] = []
        for result in results:
            observations = result["observations"]
            random_gains: list[float] = []
            shuffled_gains: list[float] = []
            for world in range(world_count):
                pretrained = paired_aulc(
                    [observations["pretrained_frozen"][modality][str(k)][world] for k in K_GRID]
                )
                random = paired_aulc(
                    [observations["random_init_frozen"][modality][str(k)][world] for k in K_GRID]
                )
                shuffled = paired_aulc(
                    [observations["pretrained_shuffled"][modality][str(k)][world] for k in K_GRID]
                )
                random_gains.append(random - pretrained)
                shuffled_gains.append(shuffled - pretrained)
            random_gains_by_seed.append(random_gains)
            shuffled_gains_by_seed.append(shuffled_gains)
        # Each world is one cluster; checkpoint seeds are repeated measures
        # inside that cluster and are averaged before resampling worlds.
        random_cluster_gains = np.mean(np.asarray(random_gains_by_seed), axis=0)
        shuffled_cluster_gains = np.mean(np.asarray(shuffled_gains_by_seed), axis=0)
        random_ci = paired_world_bootstrap(
            random_cluster_gains,
            replicates=bootstrap_replicates,
            seed=seed + 2 * modality_index,
        )
        shuffled_ci = paired_world_bootstrap(
            shuffled_cluster_gains,
            replicates=bootstrap_replicates,
            seed=seed + 2 * modality_index + 1,
        )
        modalities[modality] = {
            "pretrained_vs_random_aulc_gain": {
                "mean": random_ci[0],
                "lower_95": random_ci[1],
                "upper_95": random_ci[2],
            },
            "normal_vs_shuffled_aulc_gain": {
                "mean": shuffled_ci[0],
                "lower_95": shuffled_ci[1],
                "upper_95": shuffled_ci[2],
            },
            "gate": random_ci[1] > 0.0 and shuffled_ci[1] > 0.0,
        }
    payload: dict[str, Any] = {
        "schema_version": "tabu.transfer-base-icl-three-seed-clustered-summary.v1",
        "status": "local_unissued",
        "checkpoint_seeds": list(checkpoint_seeds),
        "world_seed": next(iter(world_seeds)),
        "worlds_per_modality": world_count,
        "bootstrap_replicates": bootstrap_replicates,
        "cluster_unit": "world_id",
        "checkpoint_seed_treatment": "repeated_measures_averaged_inside_world_cluster",
        "input_results": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in result_paths
        ],
        "modalities": modalities,
        "passed_clustered_gate": all(item["gate"] for item in modalities.values()),
        "claim_boundary": "clustered local-unissued diagnostic; not a formal Link-5 receipt",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload | {"result_path": str(output_path), "result_sha256": _sha256_file(output_path)}


__all__ = [
    "FROZEN_ARMS",
    "HELDOUT_FAMILY",
    "K_GRID",
    "FrozenIclConfig",
    "HeldoutIclCase",
    "aggregate_common_world_results",
    "build_heldout_icl_case",
    "paired_aulc",
    "paired_world_bootstrap",
    "run_frozen_icl",
]
