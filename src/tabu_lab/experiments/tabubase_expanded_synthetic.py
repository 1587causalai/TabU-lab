"""Conservative Stage-A synthetic prior for frozen-ICL pretraining research.

This module freezes only a generator contract for ``tabu.cell.base@0.2.0``
with ``supervised.label_broadcast.v1``.  It does not start training, construct
an optimizer, or establish a model claim.  A world manifest is sampled before
rows and before context/query compilation, so partitions identify different
task-generating mechanisms rather than different views of one compiled batch.

The built-in audit covers G-D0 deterministic replay, G-D1 authority/leakage,
and G-D2 distribution coverage.  G-D3 through G-D5 are deliberately outside
that Stage-A audit.  A small, optional closed-form ridge diagnostic is exposed
separately as a selected-world G-D3 development aid; it is not silently folded
into the generator acceptance result.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, Literal

import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    TruthSidecar,
    canonical_hash,
    origin_code,
)

EXPANDED_SYNTHETIC_GENERATOR_VERSION = "tabubase.expanded-synthetic.v4"
GENERATOR_VERSION = EXPANDED_SYNTHETIC_GENERATOR_VERSION
MODEL_CONTRACT = "tabu.cell.base@0.2.0"
PROFILE_ID = "supervised.label_broadcast.v1"
NOMINAL_CODEBOOK_PLAN = "source_scoped_frozen_codebook.v2"
NOMINAL_CODEBOOK_SIZE = 100
NOMINAL_SOURCE_SCOPE_ID = "tabubase.expanded-synthetic.source-codebook.v1"

TRAIN_FAMILIES = (
    "sparse_glm",
    "sparse_dag_scm",
    "tree_threshold",
    "latent_factor",
    "polynomial_interaction",
    "periodic_monotone",
    "categorical_interaction",
)
HELDOUT_FAMILIES = ("mixture_of_experts",)
WIDTHS = (4, 8, 16, 32)
SCHEMA_PROFILES = ("numeric_only", "ordinal_only", "nominal_only", "mixed")
MISSINGNESS_REGIMES = ("none", "mcar", "mar")
RESPONSE_MODALITIES = ("numeric", "binary", "ordinal", "categorical")
FROZEN_CONTEXT_ROWS_SCHEDULE = (2, 4, 8, 16, 32, 64)
LONG_CONTEXT_ROWS_SCHEDULE = (2, 4, 8, 16, 32, 64, 128, 256, 512)
RESPONSE_CALIBRATION_ROWS = 256
CONTEXT_CANDIDATE_INITIAL_ROWS = 64
CONTEXT_CANDIDATE_MAX_ROWS = 1_024
LONG_CONTEXT_CANDIDATE_ROWS = 512

Partition = Literal["train", "validation", "test", "heldout_family"]
FamilyScope = Literal["training", "heldout"]


def _stable_seed(root_seed: int, namespace: str) -> int:
    if type(root_seed) is not int or root_seed < 0:
        raise ValueError("root_seed must be a non-negative integer")
    if not namespace.strip():
        raise ValueError("seed namespace cannot be empty")
    payload = f"{GENERATOR_VERSION}|{root_seed}|{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


def expanded_eligible_context_rows(
    *, world_index: int, context_rows_schedule: tuple[int, ...]
) -> tuple[int, ...]:
    """Return only K values whose context can support the declared response.

    Four-class ordinal and categorical worlds cannot be realized at K=2 by the
    same-column categorical terminal: a class absent from context has exactly
    zero support.  Those worlds therefore begin at K=4.  Numeric and binary
    worlds retain the complete frozen schedule.
    """

    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a non-negative integer")
    if not context_rows_schedule:
        raise ValueError("context_rows_schedule cannot be empty")
    modality = RESPONSE_MODALITIES[world_index % len(RESPONSE_MODALITIES)]
    minimum = 4 if modality in {"ordinal", "categorical"} else 2
    eligible = tuple(value for value in context_rows_schedule if value >= minimum)
    if not eligible:
        raise ValueError(f"no context size can support {modality} response")
    return eligible


def expanded_training_context_rows(
    *, world_index: int, context_rows_schedule: tuple[int, ...]
) -> int:
    """Choose a deterministic eligible K without response/K parity locking."""

    eligible = expanded_eligible_context_rows(
        world_index=world_index,
        context_rows_schedule=context_rows_schedule,
    )
    block, response_slot = divmod(world_index, len(RESPONSE_MODALITIES))
    return eligible[(block + response_slot) % len(eligible)]


def _schema_kinds(profile: str, width: int) -> tuple[str, ...]:
    if profile == "numeric_only":
        return (FeatureKind.NUMERIC.value,) * width
    if profile == "ordinal_only":
        return (FeatureKind.ORDINAL.value,) * width
    if profile == "nominal_only":
        return (FeatureKind.CATEGORICAL.value,) * width
    if profile != "mixed":
        raise ValueError(f"unknown schema profile: {profile!r}")
    cycle = (
        FeatureKind.NUMERIC.value,
        FeatureKind.ORDINAL.value,
        FeatureKind.CATEGORICAL.value,
    )
    return tuple(cycle[index % len(cycle)] for index in range(width))


@dataclass(frozen=True, slots=True)
class WorldManifest:
    """Fully replayable parameters for one task-generating world."""

    generator_version: str
    model_contract: str
    profile_id: str
    root_seed: int
    partition: str
    family_scope: str
    world_index: int
    world_id: str
    family: str
    predictor_width: int
    schema_profile: str
    predictor_kinds: tuple[str, ...]
    predictor_cardinalities: tuple[int, ...]
    predictor_permutation: tuple[int, ...]
    response_modality: str
    response_cardinality: int
    missingness_regime: str
    latent_dimension: int
    parameter_seed: int
    row_seed: int
    noise_seed: int
    missingness_seed: int
    response_calibration_seed: int
    response_calibration_noise_seed: int
    response_calibration_rows: int
    numeric_distributions: tuple[str, ...]
    locations: tuple[float, ...]
    scales: tuple[float, ...]
    category_skews: tuple[float, ...]
    latent_loadings: tuple[tuple[float, ...], ...]
    primary_coefficients: tuple[float, ...]
    secondary_coefficients: tuple[float, ...]
    bias: float
    noise_scale: float
    missingness_probability: float
    missingness_slope: float
    response_thresholds: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError("world manifest uses an unknown generator version")
        if self.model_contract != MODEL_CONTRACT or self.profile_id != PROFILE_ID:
            raise ValueError("world manifest changed the frozen model/profile contract")
        if self.partition not in {"train", "validation", "test", "heldout_family"}:
            raise ValueError("unknown world partition")
        if self.family_scope not in {"training", "heldout"}:
            raise ValueError("unknown family scope")
        expected_families = TRAIN_FAMILIES if self.family_scope == "training" else HELDOUT_FAMILIES
        if self.family not in expected_families:
            raise ValueError("world family is outside its declared family partition")
        if self.family_scope == "heldout" and self.partition != "heldout_family":
            raise ValueError("held-out families require the heldout_family partition")
        if self.partition == "heldout_family" and self.family_scope != "heldout":
            raise ValueError("heldout_family cannot contain an optimization family")
        if self.predictor_width not in WIDTHS or self.predictor_width + 1 > 64:
            raise ValueError("predictor width is outside the frozen Stage-A range")
        width = self.predictor_width
        sized_fields = (
            self.predictor_kinds,
            self.predictor_cardinalities,
            self.predictor_permutation,
            self.numeric_distributions,
            self.locations,
            self.scales,
            self.category_skews,
            self.latent_loadings,
            self.primary_coefficients,
            self.secondary_coefficients,
        )
        if any(len(field) != width for field in sized_fields):
            raise ValueError("world predictor parameters must match predictor_width")
        if set(self.predictor_permutation) != set(range(width)):
            raise ValueError("predictor_permutation must contain every source predictor once")
        if self.schema_profile not in SCHEMA_PROFILES:
            raise ValueError("unknown schema profile")
        if self.missingness_regime not in MISSINGNESS_REGIMES:
            raise ValueError("unknown missingness regime")
        if self.response_modality not in RESPONSE_MODALITIES:
            raise ValueError("unknown response modality")
        if any(len(row) != self.latent_dimension for row in self.latent_loadings):
            raise ValueError("latent loading rows must match latent_dimension")
        if self.response_calibration_rows != RESPONSE_CALIBRATION_ROWS:
            raise ValueError("response calibration row count changed from the v4 contract")
        for kind, cardinality in zip(
            self.predictor_kinds, self.predictor_cardinalities, strict=True
        ):
            resolved = FeatureKind(kind)
            if resolved is FeatureKind.NUMERIC and cardinality != 0:
                raise ValueError("numeric predictors cannot declare a domain cardinality")
            if resolved is FeatureKind.ORDINAL and not 3 <= cardinality <= 20:
                raise ValueError("ordinal predictor cardinality must be in [3, 20]")
            if resolved is FeatureKind.CATEGORICAL and not 2 <= cardinality <= 100:
                raise ValueError("nominal predictor cardinality must be in [2, 100]")
        if self.response_modality == "numeric":
            if self.response_cardinality != 0 or self.response_thresholds:
                raise ValueError("numeric responses cannot declare a categorical domain")
        else:
            if not 2 <= self.response_cardinality <= NOMINAL_CODEBOOK_SIZE:
                raise ValueError("response cardinality exceeds the frozen codebook")
            if len(self.response_thresholds) != self.response_cardinality - 1:
                raise ValueError("response thresholds must separate every declared class")
        if any(not math.isfinite(value) for value in self._float_parameters()):
            raise ValueError("world manifest floating-point parameters must be finite")

    def _float_parameters(self) -> tuple[float, ...]:
        flattened_loadings = tuple(value for row in self.latent_loadings for value in row)
        return (
            *self.locations,
            *self.scales,
            *self.category_skews,
            *flattened_loadings,
            *self.primary_coefficients,
            *self.secondary_coefficients,
            self.bias,
            self.noise_scale,
            self.missingness_probability,
            self.missingness_slope,
            *self.response_thresholds,
        )

    @property
    def manifest_hash(self) -> str:
        return canonical_hash({"schema": "tabubase-expanded-world-manifest.v4", "world": self})

    @property
    def content_hash(self) -> str:
        return self.manifest_hash


def sample_expanded_world_manifest(
    *,
    root_seed: int,
    world_index: int,
    partition: Partition = "train",
    family_scope: FamilyScope | None = None,
) -> WorldManifest:
    """Sample a world manifest before any row or episode is materialized."""

    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a non-negative integer")
    if partition not in {"train", "validation", "test", "heldout_family"}:
        raise ValueError("partition must be train, validation, test, or heldout_family")
    if family_scope is None:
        family_scope = "heldout" if partition == "heldout_family" else "training"
    if family_scope not in {"training", "heldout"}:
        raise ValueError("family_scope must be training or heldout")
    if (partition == "heldout_family") != (family_scope == "heldout"):
        raise ValueError("held-out family scope and heldout_family partition must agree")

    family_pool = TRAIN_FAMILIES if family_scope == "training" else HELDOUT_FAMILIES
    family = family_pool[world_index % len(family_pool)]
    response_modality = RESPONSE_MODALITIES[world_index % len(RESPONSE_MODALITIES)]
    predictor_width = WIDTHS[(world_index // 4) % len(WIDTHS)]
    schema_profile = SCHEMA_PROFILES[(world_index // 16) % len(SCHEMA_PROFILES)]
    missingness_regime = MISSINGNESS_REGIMES[
        (world_index // 64) % len(MISSINGNESS_REGIMES)
    ]
    predictor_kinds = _schema_kinds(schema_profile, predictor_width)
    parameter_seed = _stable_seed(
        root_seed, f"manifest:{partition}:{family_scope}:{world_index}:{family}"
    )
    generator = torch.Generator(device="cpu").manual_seed(parameter_seed)

    cardinalities: list[int] = []
    for kind in predictor_kinds:
        if kind == FeatureKind.NUMERIC.value:
            cardinalities.append(0)
        elif kind == FeatureKind.ORDINAL.value:
            cardinalities.append(int(torch.randint(3, 21, (), generator=generator)))
        else:
            cardinalities.append(int(torch.randint(2, 101, (), generator=generator)))

    permutation = tuple(
        int(value) for value in torch.randperm(predictor_width, generator=generator)
    )
    if predictor_width > 1 and permutation == tuple(range(predictor_width)):
        permutation = (*permutation[1:], permutation[0])
    latent_dimension = 1 + int(
        torch.randint(0, min(8, predictor_width), (), generator=generator)
    )
    distributions = ("normal", "heavy_tail", "positive_skew", "mixture")
    distribution_offset = int(torch.randint(0, len(distributions), (), generator=generator))
    numeric_distributions = tuple(
        distributions[(source + distribution_offset) % len(distributions)]
        if kind == FeatureKind.NUMERIC.value
        else "not_numeric"
        for source, kind in enumerate(predictor_kinds)
    )
    locations = tuple(_uniform(generator, -1.0, 1.0) for _ in range(predictor_width))
    scales = tuple(_uniform(generator, 0.55, 1.8) for _ in range(predictor_width))
    category_skews = tuple(_uniform(generator, 0.0, 2.0) for _ in range(predictor_width))
    loading_tensor = torch.randn(
        predictor_width, latent_dimension, generator=generator, dtype=torch.float32
    ) / math.sqrt(float(latent_dimension))
    latent_loadings = tuple(tuple(float(value) for value in row) for row in loading_tensor)

    primary = torch.randn(predictor_width, generator=generator, dtype=torch.float32)
    primary = primary / primary.norm().clamp_min(1.0e-6)
    if family == "sparse_glm":
        active = max(2, math.ceil(math.sqrt(predictor_width)))
        active_indices = torch.randperm(predictor_width, generator=generator)[:active]
        sparse = torch.zeros_like(primary)
        sparse[active_indices] = primary[active_indices]
        primary = sparse / sparse.norm().clamp_min(1.0e-6)
    secondary = torch.randn(predictor_width, generator=generator, dtype=torch.float32)
    secondary = secondary / secondary.norm().clamp_min(1.0e-6)

    if response_modality == "numeric":
        response_cardinality = 0
    elif response_modality == "binary":
        response_cardinality = 2
    elif response_modality == "ordinal":
        response_cardinality = 4
    else:
        response_cardinality = 4
    response_thresholds = (
        ()
        if response_cardinality == 0
        else tuple(
            float(value)
            for value in torch.linspace(-1.25, 1.25, response_cardinality - 1)
        )
    )
    missingness_probability = (
        0.0 if missingness_regime == "none" else _uniform(generator, 0.04, 0.16)
    )
    missingness_slope = (
        _uniform(generator, 0.6, 1.4) if missingness_regime == "mar" else 0.0
    )
    noise_scale = (
        _uniform(generator, 0.025, 0.07)
        if family == "sparse_glm"
        else _uniform(generator, 0.06, 0.22)
    )
    world_id = (
        f"tabubase-expanded-v4-{partition}-{family_scope}-s{root_seed}-w{world_index:08d}"
    )
    provisional = WorldManifest(
        generator_version=GENERATOR_VERSION,
        model_contract=MODEL_CONTRACT,
        profile_id=PROFILE_ID,
        root_seed=root_seed,
        partition=partition,
        family_scope=family_scope,
        world_index=world_index,
        world_id=world_id,
        family=family,
        predictor_width=predictor_width,
        schema_profile=schema_profile,
        predictor_kinds=predictor_kinds,
        predictor_cardinalities=tuple(cardinalities),
        predictor_permutation=permutation,
        response_modality=response_modality,
        response_cardinality=response_cardinality,
        missingness_regime=missingness_regime,
        latent_dimension=latent_dimension,
        parameter_seed=parameter_seed,
        row_seed=_stable_seed(root_seed, f"rows:{world_id}"),
        noise_seed=_stable_seed(root_seed, f"noise:{world_id}"),
        missingness_seed=_stable_seed(root_seed, f"missingness:{world_id}"),
        response_calibration_seed=_stable_seed(root_seed, f"response-calibration:{world_id}"),
        response_calibration_noise_seed=_stable_seed(
            root_seed, f"response-calibration-noise:{world_id}"
        ),
        response_calibration_rows=RESPONSE_CALIBRATION_ROWS,
        numeric_distributions=numeric_distributions,
        locations=locations,
        scales=scales,
        category_skews=category_skews,
        latent_loadings=latent_loadings,
        primary_coefficients=tuple(float(value) for value in primary),
        secondary_coefficients=tuple(float(value) for value in secondary),
        bias=_uniform(generator, -0.35, 0.35),
        noise_scale=noise_scale,
        missingness_probability=missingness_probability,
        missingness_slope=missingness_slope,
        response_thresholds=response_thresholds,
    )
    if response_cardinality == 0:
        return provisional

    # Thresholds are world parameters estimated from a dedicated calibration
    # bank.  The bank has independent row/noise seeds and is materialized before
    # context/query compilation, so no query response can influence class
    # support or balance.
    calibration_manifest = replace(
        provisional,
        row_seed=provisional.response_calibration_seed,
        noise_seed=provisional.response_calibration_noise_seed,
    )
    calibration_predictors, calibration_latent = _materialize_predictors(
        calibration_manifest,
        RESPONSE_CALIBRATION_ROWS,
    )
    calibration_score = _materialize_response(
        calibration_manifest,
        calibration_predictors,
        calibration_latent,
    )
    quantiles = torch.arange(1, response_cardinality, dtype=torch.float32) / float(
        response_cardinality
    )
    calibrated_thresholds = tuple(
        float(value) for value in torch.quantile(calibration_score, quantiles)
    )
    if any(
        right <= left
        for left, right in pairwise(calibrated_thresholds)
    ):
        raise RuntimeError("response calibration produced non-increasing thresholds")
    return replace(provisional, response_thresholds=calibrated_thresholds)


def _materialize_predictors(
    manifest: WorldManifest,
    rows: int,
    *,
    row_seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    resolved_seed = manifest.row_seed if row_seed is None else row_seed
    latent_generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(resolved_seed, "materialize-latent")
    )
    idiosyncratic_generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(resolved_seed, "materialize-idiosyncratic")
    )
    latent = torch.randn(
        rows,
        manifest.latent_dimension,
        generator=latent_generator,
        dtype=torch.float32,
    )
    idiosyncratic = torch.randn(
        rows,
        manifest.predictor_width,
        generator=idiosyncratic_generator,
        dtype=torch.float32,
    )
    loadings = torch.tensor(manifest.latent_loadings, dtype=torch.float32)
    base = latent @ loadings.transpose(0, 1) + 0.4 * idiosyncratic
    values = torch.empty(rows, manifest.predictor_width, dtype=torch.float32)
    for feature, (kind, cardinality) in enumerate(
        zip(manifest.predictor_kinds, manifest.predictor_cardinalities, strict=True)
    ):
        location = manifest.locations[feature]
        scale = manifest.scales[feature]
        feature_base = base[:, feature]
        if kind == FeatureKind.NUMERIC.value:
            distribution = manifest.numeric_distributions[feature]
            if distribution == "normal":
                transformed = feature_base
            elif distribution == "heavy_tail":
                transformed = (feature_base + 0.18 * feature_base.pow(3)).clamp(-8.0, 8.0)
            elif distribution == "positive_skew":
                transformed = torch.exp((0.45 * feature_base).clamp(-4.0, 4.0)) - 1.0
            elif distribution == "mixture":
                transformed = feature_base + 0.9 * (latent[:, 0] > 0).to(torch.float32)
            else:
                raise RuntimeError("unknown numeric distribution in validated manifest")
            values[:, feature] = location + scale * transformed
        elif kind == FeatureKind.ORDINAL.value:
            thresholds = torch.linspace(-1.5, 1.5, cardinality - 1)
            values[:, feature] = torch.bucketize(
                (feature_base + 0.15 * location).contiguous(), thresholds
            ).to(torch.float32)
        else:
            probability_generator = torch.Generator(device="cpu").manual_seed(
                _stable_seed(
                    manifest.parameter_seed,
                    f"category-probability-permutation-{feature}",
                )
            )
            sampling_generator = torch.Generator(device="cpu").manual_seed(
                _stable_seed(resolved_seed, f"materialize-category-{feature}")
            )
            labels = torch.arange(1, cardinality + 1, dtype=torch.float32)
            probabilities = labels.pow(-manifest.category_skews[feature]).clamp_min(1.0e-12)
            probabilities = probabilities / probabilities.sum()
            label_permutation = torch.randperm(
                cardinality,
                generator=probability_generator,
            )
            probabilities = probabilities[label_permutation]
            values[:, feature] = torch.multinomial(
                probabilities,
                rows,
                replacement=True,
                generator=sampling_generator,
            ).to(torch.float32)
    return values, latent


def _response_design(manifest: WorldManifest, predictors: torch.Tensor) -> torch.Tensor:
    columns: list[torch.Tensor] = []
    for feature, (kind, cardinality) in enumerate(
        zip(manifest.predictor_kinds, manifest.predictor_cardinalities, strict=True)
    ):
        column = predictors[:, feature]
        if kind == FeatureKind.NUMERIC.value:
            column = (column - manifest.locations[feature]) / manifest.scales[feature]
        else:
            column = 2.0 * column / float(max(1, cardinality - 1)) - 1.0
        columns.append(column)
    return torch.stack(columns, dim=1)


def _materialize_response(
    manifest: WorldManifest,
    predictors: torch.Tensor,
    latent: torch.Tensor,
    *,
    noise_seed: int | None = None,
) -> torch.Tensor:
    design = _response_design(manifest, predictors)
    primary = torch.tensor(manifest.primary_coefficients, dtype=torch.float32)
    secondary = torch.tensor(manifest.secondary_coefficients, dtype=torch.float32)
    linear = design @ primary
    if manifest.family == "sparse_glm":
        score = linear
    elif manifest.family == "sparse_dag_scm":
        score = linear + 0.35 * torch.tanh(design[:, 0])
        if manifest.predictor_width > 1:
            score = score + 0.2 * design[:, 0] * torch.tanh(design[:, 1])
    elif manifest.family == "tree_threshold":
        score = 0.9 * (design[:, 0] > 0).to(torch.float32)
        score = score - 0.7 * (design[:, 1] > 0.35).to(torch.float32)
        score = score + 0.25 * linear
    elif manifest.family == "latent_factor":
        factor = latent[:, 0]
        if manifest.latent_dimension > 1:
            factor = factor + 0.45 * latent[:, 1]
        score = 0.75 * factor + 0.3 * torch.sin(factor + design[:, 0]) + 0.2 * linear
    elif manifest.family == "polynomial_interaction":
        score = 0.45 * linear + 0.65 * design[:, 0] * design[:, 1]
        if manifest.predictor_width > 2:
            score = score + 0.25 * design[:, 2].square()
    elif manifest.family == "periodic_monotone":
        score = 0.65 * torch.sin(math.pi * design[:, 0])
        score = score + 0.55 * torch.tanh(design[:, 1]) + 0.2 * linear
    elif manifest.family == "categorical_interaction":
        interaction = (design[:, 0] > 0).to(torch.float32)
        interaction = interaction * (design[:, 1] < 0).to(torch.float32)
        score = 0.45 * linear + 0.9 * interaction
    elif manifest.family == "mixture_of_experts":
        gate = design[:, 0] + 0.25 * latent[:, 0] > 0
        score = torch.where(gate, design @ primary, design @ secondary)
        score = score + gate.to(torch.float32) * 0.35
    else:
        raise RuntimeError("unknown response family in validated manifest")
    generator = torch.Generator(device="cpu").manual_seed(
        manifest.noise_seed if noise_seed is None else noise_seed
    )
    noise = torch.randn(score.shape, generator=generator, dtype=torch.float32)
    if manifest.family == "mixture_of_experts":
        noise = noise * (0.6 + 0.4 * design[:, 0].abs())
    return score + manifest.bias + manifest.noise_scale * noise


def _sample_missingness(manifest: WorldManifest, predictors: torch.Tensor) -> torch.Tensor:
    if manifest.missingness_regime == "none":
        return torch.zeros_like(predictors, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(manifest.missingness_seed)
    uniforms = torch.rand(predictors.shape, generator=generator)
    if manifest.missingness_regime == "mcar":
        probabilities = torch.full_like(predictors, manifest.missingness_probability)
    else:
        driver = (predictors[:, 0] - manifest.locations[0]) / manifest.scales[0]
        base_probability = manifest.missingness_probability
        base_logit = math.log(base_probability / (1.0 - base_probability))
        probabilities = torch.sigmoid(
            torch.tensor(base_logit) + manifest.missingness_slope * driver
        ).unsqueeze(1).expand_as(predictors)
    missing = uniforms < probabilities
    if manifest.missingness_regime == "mar":
        # The declared MAR driver remains observed.  The mask depends only on
        # pre-response predictor values and an independent RNG namespace.
        missing[:, 0] = False
    return missing


def _context_standardize(
    values: torch.Tensor, *, available: torch.Tensor, context_rows: int
) -> torch.Tensor:
    """Apply statistics estimated only from legal context cells.

    K=0, K=1, or fewer than two observed context values uses an identity
    transform.  This is query-blind and avoids unstable one-point variance.
    """

    result = values.clone()
    if context_rows < 2:
        return result
    legal = available[:context_rows]
    evidence = values[:context_rows][legal]
    if evidence.numel() < 2:
        return result
    mean = evidence.mean()
    scale = evidence.std(unbiased=False).clamp_min(1.0e-5)
    return (result - mean) / scale


def _bucket_response(manifest: WorldManifest, raw_response: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor(manifest.response_thresholds, dtype=torch.float32)
    return torch.bucketize(raw_response.contiguous(), thresholds).to(torch.float32)


def _class_balanced_context_order(
    labels: torch.Tensor,
    *,
    class_count: int,
    required_rows: int,
) -> tuple[int, ...]:
    buckets = [
        [index for index, value in enumerate(labels.tolist()) if int(value) == class_index]
        for class_index in range(class_count)
    ]
    if any(not bucket for bucket in buckets):
        raise RuntimeError("context candidate pool does not contain every response class")
    offsets = [0] * class_count
    order: list[int] = []
    while len(order) < required_rows:
        progressed = False
        for class_index, bucket in enumerate(buckets):
            offset = offsets[class_index]
            if offset >= len(bucket):
                continue
            order.append(bucket[offset])
            offsets[class_index] += 1
            progressed = True
            if len(order) == required_rows:
                break
        if not progressed:
            break
    if len(order) != required_rows:
        raise RuntimeError("context candidate pool cannot fill the frozen context bank")
    return tuple(order)


def _materialize_episode_bank(
    manifest: WorldManifest,
    *,
    query_rows: int,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    tuple[int, ...],
]:
    """Materialize a fixed context candidate pool plus an independent query bank."""

    if (
        type(context_candidate_rows) is not int
        or not 1 <= context_candidate_rows <= CONTEXT_CANDIDATE_MAX_ROWS
    ):
        raise ValueError(
            "context_candidate_rows must be an integer in the frozen range [1, 1024]"
        )
    pool_rows = context_candidate_rows
    while True:
        raw_predictors, latent = _materialize_predictors(
            manifest,
            pool_rows + query_rows,
        )
        raw_response = _materialize_response(manifest, raw_predictors, latent)
        if manifest.response_modality == "numeric":
            response = raw_response
            context_order = tuple(range(context_candidate_rows))
            break
        response = _bucket_response(manifest, raw_response)
        try:
            context_order = _class_balanced_context_order(
                response[:pool_rows],
                class_count=manifest.response_cardinality,
                required_rows=context_candidate_rows,
            )
            break
        except RuntimeError:
            if pool_rows >= CONTEXT_CANDIDATE_MAX_ROWS:
                raise
            pool_rows = min(pool_rows * 2, CONTEXT_CANDIDATE_MAX_ROWS)
    missing = _sample_missingness(manifest, raw_predictors)
    return raw_predictors, response, missing, pool_rows, context_order


def _domain(kind: str, cardinality: int) -> tuple[str, ...]:
    prefix = "level" if kind == FeatureKind.ORDINAL.value else "category"
    return tuple(f"{prefix}-{index:03d}" for index in range(cardinality))


def _feature_specs(manifest: WorldManifest) -> tuple[FeatureSpec, ...]:
    source_specs: list[FeatureSpec] = []
    for source, (kind, cardinality) in enumerate(
        zip(manifest.predictor_kinds, manifest.predictor_cardinalities, strict=True)
    ):
        name = f"{kind}_source_{source:02d}"
        if kind == FeatureKind.NUMERIC.value:
            source_specs.append(FeatureSpec(name=name))
        else:
            source_specs.append(
                FeatureSpec(
                    name=name,
                    kind=FeatureKind(kind),
                    domain=_domain(kind, cardinality),
                    # Hold the existing source-scoped tokenizer treatment fixed:
                    # the same declared source/domain label maps to the same frozen
                    # code across worlds.  A world-scoped assignment is a separate
                    # future ablation and must not be smuggled into this prior change.
                    codebook_id=f"{NOMINAL_SOURCE_SCOPE_ID}:predictor-source-{source:02d}",
                )
            )
    predictors = tuple(source_specs[index] for index in manifest.predictor_permutation)
    if manifest.response_modality == "numeric":
        response = FeatureSpec(name="response", role=FeatureRole.RESPONSE)
    else:
        response_kind = (
            FeatureKind.ORDINAL
            if manifest.response_modality == "ordinal"
            else FeatureKind.CATEGORICAL
        )
        response = FeatureSpec(
            name="response",
            kind=response_kind,
            domain=tuple(
                f"response-class-{index:03d}" for index in range(manifest.response_cardinality)
            ),
            codebook_id=f"{NOMINAL_SOURCE_SCOPE_ID}:response-source",
            role=FeatureRole.RESPONSE,
        )
    return (*predictors, response)


def build_expanded_synthetic_episode(
    *,
    root_seed: int,
    world_index: int,
    partition: Partition = "train",
    family_scope: FamilyScope | None = None,
    context_rows: int = 64,
    query_rows: int = 64,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
) -> tuple[EvidenceEpisode, TruthSidecar, dict[str, Any]]:
    """Build one deterministic expanded-prior supervised episode.

    Query response cells are physical zero in the :class:`EvidenceEpisode`.
    Their values exist only at the matching coordinates of the returned
    :class:`TruthSidecar`.
    """

    if (
        type(context_candidate_rows) is not int
        or not CONTEXT_CANDIDATE_INITIAL_ROWS
        <= context_candidate_rows
        <= CONTEXT_CANDIDATE_MAX_ROWS
    ):
        raise ValueError(
            "context_candidate_rows must be an integer in the frozen range [64, 1024]"
        )
    if (
        type(context_rows) is not int
        or not 0 <= context_rows <= context_candidate_rows
    ):
        raise ValueError(
            "context_rows must be an integer between zero and context_candidate_rows"
        )
    if type(query_rows) is not int or query_rows <= 0:
        raise ValueError("query_rows must be a positive integer")
    manifest = sample_expanded_world_manifest(
        root_seed=root_seed,
        world_index=world_index,
        partition=partition,
        family_scope=family_scope,
    )
    if manifest.response_cardinality and context_rows < manifest.response_cardinality:
        raise ValueError(
            "classification context is smaller than its declared response support"
        )
    (
        bank_predictors,
        bank_response,
        bank_missing,
        materialized_candidate_rows,
        context_order,
    ) = _materialize_episode_bank(
        manifest,
        query_rows=query_rows,
        context_candidate_rows=context_candidate_rows,
    )
    context_indices = context_order[:context_rows]
    query_indices = tuple(
        range(materialized_candidate_rows, materialized_candidate_rows + query_rows)
    )
    selected_indices = (*context_indices, *query_indices)
    index_tensor = torch.tensor(selected_indices, dtype=torch.int64)
    raw_predictors = bank_predictors[index_tensor]
    missing_source_order = bank_missing[index_tensor]
    raw_response = bank_response[index_tensor]
    rows = context_rows + query_rows

    transformed = raw_predictors.clone()
    for feature, kind in enumerate(manifest.predictor_kinds):
        if kind != FeatureKind.NUMERIC.value:
            continue
        transformed[:, feature] = _context_standardize(
            raw_predictors[:, feature],
            available=~missing_source_order[:, feature],
            context_rows=context_rows,
        )
    if manifest.response_modality == "numeric":
        response = _context_standardize(
            raw_response,
            available=torch.ones(rows, dtype=torch.bool),
            context_rows=context_rows,
        )
    else:
        response = raw_response
        context_support = {int(value) for value in response[:context_rows].tolist()}
        query_support = {int(value) for value in response[context_rows:].tolist()}
        if not query_support <= context_support:
            raise RuntimeError("query response class is absent from the legal context support")

    order = list(manifest.predictor_permutation)
    predictors = transformed[:, order]
    missing = missing_source_order[:, order]
    values = torch.cat((predictors, response.unsqueeze(1)), dim=1)
    roles = torch.full(
        values.shape,
        int(ForwardRole.RECEIVER | ForwardRole.SOURCE),
        dtype=torch.int64,
    )
    origins = torch.full(
        values.shape,
        origin_code(OriginState.OBSERVED),
        dtype=torch.int64,
    )
    roles[:, :-1][missing] = int(ForwardRole.RECEIVER)
    origins[:, :-1][missing] = origin_code(OriginState.NATURAL_MISSING)
    values[:, :-1][missing] = 0.0
    roles[context_rows:, -1] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    origins[context_rows:, -1] = origin_code(OriginState.QUERY)
    values[context_rows:, -1] = 0.0

    episode_id = f"{manifest.world_id}-k{context_rows}-q{query_rows}"
    if context_candidate_rows != CONTEXT_CANDIDATE_INITIAL_ROWS:
        episode_id += f"-cb{context_candidate_rows}"
    row_ids = tuple(
        f"{manifest.world_id}-bank-row-{index:06d}" for index in selected_indices
    )
    specs = _feature_specs(manifest)
    episode_metadata: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "world_manifest_hash": manifest.manifest_hash,
        "world_id": manifest.world_id,
        "world_index": manifest.world_index,
        "family": manifest.family,
        "family_scope": manifest.family_scope,
        "model_contract": MODEL_CONTRACT,
        "profile_id": PROFILE_ID,
        "predictor_width": manifest.predictor_width,
        "predictor_permutation": manifest.predictor_permutation,
        "schema_profile": manifest.schema_profile,
        "response_modality": manifest.response_modality,
        "response_cardinality": manifest.response_cardinality,
        "response_calibration_rows": manifest.response_calibration_rows,
        "response_threshold_source": "independent_world_calibration_bank_before_episode",
        "missingness_regime": manifest.missingness_regime,
        "missingness_mask_inputs": "predictors_only_before_response_generation",
        "missingness_uses_response": False,
        "statistics_scope": "context_only_with_identity_fallback_for_fewer_than_two_values",
        "nominal_codebook_plan": NOMINAL_CODEBOOK_PLAN,
        "nominal_codebook_size": NOMINAL_CODEBOOK_SIZE,
        "nominal_codebook_scope": "source_codebook_id_and_domain_label",
        "context_selection": (
            "class_balanced_context_candidate_pool_only"
            if manifest.response_cardinality
            else "fixed_context_bank_prefix"
        ),
        "context_candidate_rows": materialized_candidate_rows,
        "context_source_indices": context_indices,
        "query_bank_start": materialized_candidate_rows,
        "query_response_used_for_context_selection": False,
        "context_rows": context_rows,
        "query_rows": query_rows,
    }
    if context_candidate_rows != CONTEXT_CANDIDATE_INITIAL_ROWS:
        episode_metadata["frozen_context_bank_rows"] = context_candidate_rows
    episode = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabubase-expanded-synthetic-pretraining-stage-a-v4",
        source_partition=manifest.partition,
        fit_partition="train",
        row_ids=row_ids,
        feature_names=tuple(spec.name for spec in specs),
        feature_specs=specs,
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
        metadata=episode_metadata,
    )
    truth_values = torch.zeros_like(values)
    truth_values[context_rows:, -1] = response[context_rows:]
    recipe = {
        "schema": "tabubase-expanded-synthetic-episode-recipe.v4",
        "generator_version": GENERATOR_VERSION,
        "world_manifest_hash": manifest.manifest_hash,
        "context_rows": context_rows,
        "query_rows": query_rows,
        "context_candidate_rows": materialized_candidate_rows,
        "context_source_indices": context_indices,
        "query_source_indices": query_indices,
    }
    if context_candidate_rows != CONTEXT_CANDIDATE_INITIAL_ROWS:
        recipe["frozen_context_bank_rows"] = context_candidate_rows
    recipe_hash = canonical_hash(recipe)
    truth = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=recipe_hash,
        row_ids=row_ids,
        feature_names=episode.feature_names,
        target_values=truth_values,
        target_mask=episode.target_mask,
        metadata={"world_manifest_hash": manifest.manifest_hash},
    )
    return episode, truth, dict(episode_metadata)


def expanded_synthetic_coverage(
    manifests: Sequence[WorldManifest] | None = None,
    *,
    root_seed: int = 1729,
    world_count: int = 192,
    include_heldout: bool = True,
) -> dict[str, Any]:
    """Report empirical G-D2 coverage without materializing any rows."""

    if manifests is None:
        if type(world_count) is not int or world_count < 1:
            raise ValueError("world_count must be a positive integer")
        sampled: list[WorldManifest] = [
            sample_expanded_world_manifest(root_seed=root_seed, world_index=index)
            for index in range(world_count)
        ]
        if include_heldout:
            heldout_count = max(1, min(world_count, 64))
            sampled.extend(
                sample_expanded_world_manifest(
                    root_seed=root_seed,
                    world_index=index,
                    partition="heldout_family",
                )
                for index in range(heldout_count)
            )
        manifests = tuple(sampled)
    else:
        manifests = tuple(manifests)
        if not manifests:
            raise ValueError("coverage requires at least one world manifest")
        if any(not isinstance(manifest, WorldManifest) for manifest in manifests):
            raise TypeError("coverage entries must be WorldManifest instances")

    def counts(values: Sequence[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    all_cardinalities = [
        cardinality
        for manifest in manifests
        for cardinality in (
            *manifest.predictor_cardinalities,
            manifest.response_cardinality,
        )
        if cardinality > 0
    ]
    family_counts = counts([manifest.family for manifest in manifests])
    modality_counts = counts([manifest.response_modality for manifest in manifests])
    width_counts = counts([str(manifest.predictor_width) for manifest in manifests])
    schema_counts = counts([manifest.schema_profile for manifest in manifests])
    missingness_counts = counts([manifest.missingness_regime for manifest in manifests])
    kind_counts = counts(
        [kind for manifest in manifests for kind in manifest.predictor_kinds]
    )
    training_families = {
        manifest.family for manifest in manifests if manifest.family_scope == "training"
    }
    heldout_families = {
        manifest.family for manifest in manifests if manifest.family_scope == "heldout"
    }
    checks = {
        "all_training_families": training_families == set(TRAIN_FAMILIES),
        "complete_heldout_family": heldout_families == set(HELDOUT_FAMILIES),
        "family_partitions_disjoint": not training_families & heldout_families,
        "all_response_modalities": set(modality_counts) == set(RESPONSE_MODALITIES),
        "all_widths": {int(width) for width in width_counts} == set(WIDTHS),
        "homogeneous_and_mixed_schemas": set(schema_counts) == set(SCHEMA_PROFILES),
        "all_predictor_kinds": set(kind_counts)
        == {
            FeatureKind.NUMERIC.value,
            FeatureKind.ORDINAL.value,
            FeatureKind.CATEGORICAL.value,
        },
        "all_missingness_regimes": set(missingness_counts) == set(MISSINGNESS_REGIMES),
        "cardinality_at_most_100": bool(all_cardinalities)
        and max(all_cardinalities) <= NOMINAL_CODEBOOK_SIZE,
        "response_after_width_at_most_64": all(
            manifest.predictor_width + 1 <= 64 for manifest in manifests
        ),
    }
    return {
        "schema": "tabubase-expanded-synthetic-coverage.v4",
        "generator_version": GENERATOR_VERSION,
        "world_count": len(manifests),
        "family_counts": family_counts,
        "response_modality_counts": modality_counts,
        "width_counts": width_counts,
        "schema_profile_counts": schema_counts,
        "predictor_kind_counts": kind_counts,
        "missingness_counts": missingness_counts,
        "maximum_declared_cardinality": max(all_cardinalities, default=0),
        "checks": checks,
        "passed": all(checks.values()),
    }


def audit_expanded_training_episode_universe(
    *,
    root_seed: int = 1729,
    world_count: int = 20_000,
    context_rows_schedule: tuple[int, ...] = FROZEN_CONTEXT_ROWS_SCHEDULE,
    query_rows: int = 8,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
) -> dict[str, Any]:
    """Compile every declared training world and verify legal class support."""

    if world_count < 1 or query_rows < 1:
        raise ValueError("training-universe audit requires positive worlds and query rows")
    modality_k_counts: Counter[str] = Counter()
    candidate_pool_counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for world_index in range(world_count):
        context_rows = expanded_training_context_rows(
            world_index=world_index,
            context_rows_schedule=context_rows_schedule,
        )
        try:
            episode, truth, metadata = build_expanded_synthetic_episode(
                root_seed=root_seed,
                world_index=world_index,
                context_rows=context_rows,
                query_rows=query_rows,
                context_candidate_rows=context_candidate_rows,
            )
            support_ok = True
            if metadata["response_cardinality"]:
                context_support = {
                    int(value)
                    for value in episode.forward_values[:context_rows, -1].tolist()
                }
                query_support = {
                    int(value)
                    for value in truth.target_values[context_rows:, -1].tolist()
                }
                support_ok = query_support <= context_support
            if not support_ok:
                raise RuntimeError("compiled query response lacks legal context support")
            modality_k_counts[f"{metadata['response_modality']}|K={context_rows}"] += 1
            candidate_pool_counts[str(metadata["context_candidate_rows"])] += 1
        except Exception as exc:
            if len(failures) < 20:
                failures.append(
                    {
                        "world_index": world_index,
                        "context_rows": context_rows,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    expected_pairs = {
        f"{modality}|K={context_rows}"
        for response_slot, modality in enumerate(RESPONSE_MODALITIES)
        for context_rows in expanded_eligible_context_rows(
            world_index=response_slot,
            context_rows_schedule=context_rows_schedule,
        )
    }
    observed_pairs = set(modality_k_counts)
    checks = {
        "all_worlds_compile": not failures,
        "every_support_realizable_modality_K_pair_present": observed_pairs == expected_pairs,
        "candidate_pool_never_exceeds_frozen_max": all(
            int(value) <= CONTEXT_CANDIDATE_MAX_ROWS for value in candidate_pool_counts
        ),
    }
    return {
        "schema": "tabubase-expanded-training-universe-audit.v4",
        "generator_version": GENERATOR_VERSION,
        "root_seed": root_seed,
        "world_count": world_count,
        "context_rows_schedule": list(context_rows_schedule),
        "frozen_context_bank_rows": context_candidate_rows,
        "query_rows": query_rows,
        "modality_K_counts": dict(sorted(modality_k_counts.items())),
        "context_candidate_pool_counts": dict(sorted(candidate_pool_counts.items())),
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _find_manifest(
    *,
    root_seed: int,
    predicate: Any,
    start: int = 0,
    limit: int = 10_000,
) -> WorldManifest:
    for world_index in range(start, start + limit):
        manifest = sample_expanded_world_manifest(
            root_seed=root_seed, world_index=world_index
        )
        if predicate(manifest):
            return manifest
    raise RuntimeError("could not find a requested deterministic world stratum")


def audit_expanded_synthetic_generator(
    *,
    root_seed: int = 1729,
    coverage_worlds: int = 192,
    context_rows: int = 16,
    query_rows: int = 8,
    context_rows_schedule: tuple[int, ...] = FROZEN_CONTEXT_ROWS_SCHEDULE,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
) -> dict[str, Any]:
    """Run only the Stage-A G-D0/G-D1/G-D2 generator gates."""

    if context_rows < 2 or query_rows < 1:
        raise ValueError("audit requires at least two context rows and one query row")
    replay_cases = (
        (0, "train"),
        (17, "validation"),
        (5, "heldout_family"),
    )
    replay_checks: list[bool] = []
    for world_index, partition in replay_cases:
        first_episode, first_truth, first_metadata = build_expanded_synthetic_episode(
            root_seed=root_seed,
            world_index=world_index,
            partition=partition,  # type: ignore[arg-type]
            context_rows=context_rows,
            query_rows=query_rows,
            context_candidate_rows=context_candidate_rows,
        )
        replay_episode, replay_truth, replay_metadata = build_expanded_synthetic_episode(
            root_seed=root_seed,
            world_index=world_index,
            partition=partition,  # type: ignore[arg-type]
            context_rows=context_rows,
            query_rows=query_rows,
            context_candidate_rows=context_candidate_rows,
        )
        replay_checks.append(
            first_episode.evidence_hash == replay_episode.evidence_hash
            and first_truth.truth_hash == replay_truth.truth_hash
            and first_metadata == replay_metadata
        )
    gd0_details = {
        "byte_canonical_hash_replay_cases": len(replay_cases),
        "all_replays_match": all(replay_checks),
    }

    numeric_manifest = _find_manifest(
        root_seed=root_seed,
        predicate=lambda item: item.schema_profile == "numeric_only"
        and item.response_modality == "numeric"
        and item.missingness_regime == "none",
    )
    episode, truth, metadata = build_expanded_synthetic_episode(
        root_seed=root_seed,
        world_index=numeric_manifest.world_index,
        context_rows=context_rows,
        query_rows=query_rows,
        context_candidate_rows=context_candidate_rows,
    )
    context = episode.forward_values[:context_rows]
    context_means = context.mean(dim=0)
    context_scales = context.std(dim=0, unbiased=False)
    standardization_matches = bool(
        (context_means.abs() < 2.0e-5).all()
        and ((context_scales - 1.0).abs() < 2.0e-4).all()
    )
    target_coordinates_only = bool(
        episode.target_mask[:, :-1].sum() == 0
        and episode.target_mask[context_rows:, -1].all()
        and not episode.target_mask[:context_rows, -1].any()
    )
    query_truth_isolated = bool(
        (episode.forward_values[episode.target_mask] == 0).all()
        and truth.target_count == query_rows
        and (truth.target_values[~truth.target_mask] == 0).all()
    )
    evidence_hash_before_substitution = episode.evidence_hash
    substituted_values = truth.target_values.clone()
    substituted_values[truth.target_mask] += 17.0
    substituted_truth = TruthSidecar(
        episode_id=truth.episode_id,
        recipe_hash=truth.recipe_hash,
        row_ids=truth.row_ids,
        feature_names=truth.feature_names,
        target_values=substituted_values,
        target_mask=truth.target_mask,
        metadata=truth.metadata,
    )
    truth_substitution_is_separate = bool(
        substituted_truth.truth_hash != truth.truth_hash
        and episode.evidence_hash == evidence_hash_before_substitution
    )

    missing_manifest = _find_manifest(
        root_seed=root_seed,
        predicate=lambda item: item.missingness_regime == "mar",
    )
    missing_episode, _, missing_metadata = build_expanded_synthetic_episode(
        root_seed=root_seed,
        world_index=missing_manifest.world_index,
        context_rows=max(context_rows, 32),
        query_rows=max(query_rows, 16),
        context_candidate_rows=context_candidate_rows,
    )
    natural_missing = missing_episode.origin_states == origin_code(OriginState.NATURAL_MISSING)
    source = (missing_episode.forward_roles & int(ForwardRole.SOURCE)) != 0
    legal_missingness = bool(
        not natural_missing[:, -1].any()
        and not source[natural_missing].any()
        and (missing_episode.forward_values[natural_missing] == 0).all()
        and missing_metadata["missingness_uses_response"] is False
    )
    partition_manifests = tuple(
        sample_expanded_world_manifest(
            root_seed=root_seed,
            world_index=11,
            partition=partition,
        )
        for partition in ("train", "validation", "test")
    )
    partition_separation = (
        len({item.world_id for item in partition_manifests}) == len(partition_manifests)
        and len({item.manifest_hash for item in partition_manifests})
        == len(partition_manifests)
    )
    zero_context_episode, zero_context_truth, _ = build_expanded_synthetic_episode(
        root_seed=root_seed,
        world_index=numeric_manifest.world_index,
        context_rows=0,
        query_rows=query_rows,
        context_candidate_rows=context_candidate_rows,
    )
    zero_context_supported = bool(
        zero_context_truth.target_count == query_rows
        and zero_context_episode.target_mask[:, -1].all()
        and (zero_context_episode.forward_values[:, -1] == 0).all()
    )
    support_checks: list[bool] = []
    nested_query_checks: list[bool] = []
    for modality in ("binary", "ordinal", "categorical"):
        support_manifest = _find_manifest(
            root_seed=root_seed,
            predicate=lambda item, expected=modality: item.response_modality == expected,
        )
        query_row_banks: list[tuple[str, ...]] = []
        for support_context_rows in expanded_eligible_context_rows(
            world_index=support_manifest.world_index,
            context_rows_schedule=context_rows_schedule,
        ):
            support_episode, support_truth, support_metadata = (
                build_expanded_synthetic_episode(
                    root_seed=root_seed,
                    world_index=support_manifest.world_index,
                    context_rows=support_context_rows,
                    query_rows=max(query_rows, 16),
                    context_candidate_rows=context_candidate_rows,
                )
            )
            context_labels = {
                int(value)
                for value in support_episode.forward_values[
                    :support_context_rows, -1
                ].tolist()
            }
            query_labels = {
                int(value)
                for value in support_truth.target_values[
                    support_context_rows:, -1
                ].tolist()
            }
            support_checks.append(
                query_labels <= context_labels
                and support_metadata["query_response_used_for_context_selection"] is False
            )
            query_row_banks.append(support_episode.row_ids[support_context_rows:])
        nested_query_checks.append(len(set(query_row_banks)) == 1)
    unsupported_four_class_k2_fails_closed = True
    for modality in ("ordinal", "categorical"):
        manifest = _find_manifest(
            root_seed=root_seed,
            predicate=lambda item, expected=modality: item.response_modality == expected,
        )
        try:
            build_expanded_synthetic_episode(
                root_seed=root_seed,
                world_index=manifest.world_index,
                context_rows=2,
                query_rows=max(query_rows, 16),
                context_candidate_rows=context_candidate_rows,
            )
        except ValueError:
            continue
        unsupported_four_class_k2_fails_closed = False
    gd1_details = {
        "partition_precedes_compilation": partition_separation,
        "query_response_truth_isolated": query_truth_isolated,
        "truth_substitution_keeps_model_input_unchanged": truth_substitution_is_separate,
        "query_targets_only_response_cells": target_coordinates_only,
        "context_only_numeric_and_response_standardization": standardization_matches,
        "missingness_is_predictor_only_and_response_blind": legal_missingness,
        "zero_context_condition_supported": zero_context_supported,
        "classification_query_labels_are_context_supported": all(support_checks),
        "classification_context_selection_is_query_blind": all(support_checks),
        "nested_K_uses_identical_query_row_bank": all(nested_query_checks),
        "unsupported_four_class_K2_fails_closed": unsupported_four_class_k2_fails_closed,
        "manifest_hash_bound_in_episode": metadata["world_manifest_hash"]
        == numeric_manifest.manifest_hash,
    }
    coverage = expanded_synthetic_coverage(
        root_seed=root_seed,
        world_count=coverage_worlds,
        include_heldout=True,
    )
    gates = {
        "G-D0": {"passed": all(replay_checks), "details": gd0_details},
        "G-D1": {"passed": all(gd1_details.values()), "details": gd1_details},
        "G-D2": {"passed": coverage["passed"], "details": coverage},
    }
    return {
        "schema": "tabubase-expanded-synthetic-generator-audit.v4",
        "generator_version": GENERATOR_VERSION,
        "context_rows_schedule": list(context_rows_schedule),
        "frozen_context_bank_rows": context_candidate_rows,
        "scope": "generator_only_gd0_through_gd2",
        "gates": gates,
        "not_evaluated_gates": ("G-D3", "G-D4", "G-D5"),
        "not_a_model_or_training_claim": True,
        "passed": all(bool(gate["passed"]) for gate in gates.values()),
    }


def _closed_form_ridge(
    predictors: torch.Tensor, response: torch.Tensor, *, penalty: float
) -> torch.Tensor:
    design = torch.cat(
        (torch.ones(predictors.shape[0], 1, dtype=predictors.dtype), predictors), dim=1
    ).to(torch.float64)
    values = response.to(torch.float64)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * penalty
    regularizer[0, 0] = 0.0
    return torch.linalg.solve(design.transpose(0, 1) @ design + regularizer, design.T @ values)


def evaluate_selected_world_ridge_reference_gate(
    *,
    root_seed: int = 1729,
    selected_worlds: int = 4,
    small_context: int = 8,
    large_context: int = 64,
    query_rows: int = 64,
    ridge_penalty: float = 1.0e-3,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Run an optional no-sklearn selected-world G-D3 development check.

    Worlds are automatically restricted to ``sparse_glm``, ``numeric_only``,
    numeric-response, no-missing strata.  Both fits use prefixes from the same
    large-context episode and score the identical query bank.  This function is
    separate from :func:`audit_expanded_synthetic_generator` by design.
    """

    if selected_worlds < 1:
        raise ValueError("selected_worlds must be positive")
    if not 2 <= small_context < large_context or query_rows < 1:
        raise ValueError("ridge gate requires 2 <= small_context < large_context")
    if not math.isfinite(ridge_penalty) or ridge_penalty <= 0.0:
        raise ValueError("ridge_penalty must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")

    chosen: list[WorldManifest] = []
    cursor = 0
    while len(chosen) < selected_worlds:
        manifest = _find_manifest(
            root_seed=root_seed,
            start=cursor,
            predicate=lambda item: item.family == "sparse_glm"
            and item.schema_profile == "numeric_only"
            and item.response_modality == "numeric"
            and item.missingness_regime == "none",
        )
        chosen.append(manifest)
        cursor = manifest.world_index + 1

    per_world: list[dict[str, Any]] = []
    for manifest in chosen:
        raw_predictors, latent = _materialize_predictors(
            manifest,
            large_context + query_rows,
        )
        raw_response = _materialize_response(manifest, raw_predictors, latent)
        predictors = raw_predictors.clone()
        for feature in range(manifest.predictor_width):
            predictors[:, feature] = _context_standardize(
                raw_predictors[:, feature],
                available=torch.ones(large_context + query_rows, dtype=torch.bool),
                context_rows=large_context,
            )
        predictors = predictors[:, list(manifest.predictor_permutation)]
        standardized_response = _context_standardize(
            raw_response,
            available=torch.ones(large_context + query_rows, dtype=torch.bool),
            context_rows=large_context,
        )
        response = standardized_response[:large_context]
        query_predictors = predictors[large_context:]
        query_response = standardized_response[large_context:].to(torch.float64)
        small_coefficients = _closed_form_ridge(
            predictors[:small_context],
            response[:small_context],
            penalty=ridge_penalty,
        )
        large_coefficients = _closed_form_ridge(
            predictors[:large_context],
            response,
            penalty=ridge_penalty,
        )
        query_design = torch.cat(
            (
                torch.ones(query_rows, 1, dtype=query_predictors.dtype),
                query_predictors,
            ),
            dim=1,
        ).to(torch.float64)
        small_rmse = float(
            torch.sqrt(torch.mean((query_design @ small_coefficients - query_response).square()))
        )
        large_rmse = float(
            torch.sqrt(torch.mean((query_design @ large_coefficients - query_response).square()))
        )
        per_world.append(
            {
                "world_id": manifest.world_id,
                "world_manifest_hash": manifest.manifest_hash,
                "predictor_width": manifest.predictor_width,
                "small_rmse": small_rmse,
                "large_rmse": large_rmse,
                "rmse_gain": small_rmse - large_rmse,
            }
        )
    small_mean = sum(item["small_rmse"] for item in per_world) / len(per_world)
    large_mean = sum(item["large_rmse"] for item in per_world) / len(per_world)
    finite = all(
        math.isfinite(float(item[key]))
        for item in per_world
        for key in ("small_rmse", "large_rmse", "rmse_gain")
    )
    return {
        "schema": "tabubase-expanded-selected-world-ridge-gd3.v4",
        "gate": "G-D3",
        "included_in_stage_a_audit": False,
        "selected_stratum": {
            "family": "sparse_glm",
            "schema_profile": "numeric_only",
            "response_modality": "numeric",
            "missingness_regime": "none",
        },
        "small_context": small_context,
        "large_context": large_context,
        "query_rows": query_rows,
        "small_rmse": small_mean,
        "large_rmse": large_mean,
        "rmse_gain": small_mean - large_mean,
        "per_world": per_world,
        "finite": finite,
        "passed": finite and large_mean <= small_mean + tolerance,
    }


__all__ = [
    "CONTEXT_CANDIDATE_INITIAL_ROWS",
    "EXPANDED_SYNTHETIC_GENERATOR_VERSION",
    "FROZEN_CONTEXT_ROWS_SCHEDULE",
    "GENERATOR_VERSION",
    "HELDOUT_FAMILIES",
    "LONG_CONTEXT_CANDIDATE_ROWS",
    "LONG_CONTEXT_ROWS_SCHEDULE",
    "MISSINGNESS_REGIMES",
    "MODEL_CONTRACT",
    "NOMINAL_CODEBOOK_PLAN",
    "NOMINAL_CODEBOOK_SIZE",
    "NOMINAL_SOURCE_SCOPE_ID",
    "PROFILE_ID",
    "RESPONSE_MODALITIES",
    "SCHEMA_PROFILES",
    "TRAIN_FAMILIES",
    "WIDTHS",
    "WorldManifest",
    "audit_expanded_synthetic_generator",
    "audit_expanded_training_episode_universe",
    "build_expanded_synthetic_episode",
    "evaluate_selected_world_ridge_reference_gate",
    "expanded_eligible_context_rows",
    "expanded_synthetic_coverage",
    "expanded_training_context_rows",
    "sample_expanded_world_manifest",
]
