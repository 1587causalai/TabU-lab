"""Broad supervised synthetic prior v3 with SCM and DiscoSCM families.

The DiscoSCM family is a compact TabUR adaptation of the MIT-licensed
``tabuf-episode-api`` generator (generator.py blob
``70285476b0dec8222f097721762127ee36afe3c8``).  It retains the important
population / feature-token / typed-cell construction while projecting the
generated table into TabUR's supervised query-row envelope.  Latent Units,
feature tokens, and the sampled DAG never enter model-facing evidence.
Episode shapes come from a continuous compute-bounded prior rather than the
fixed width/context buckets retained by v2.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
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

from .query_row_supervised_synthetic_v2 import (
    NOISE_LEVELS,
    PREDICTOR_REGIMES,
    _draw_predictors,
    _generator,
    _response,
)
from .query_row_supervised_synthetic_v2 import (
    WORLD_FAMILIES as V2_WORLD_FAMILIES,
)

GeneratorPartition = Literal["train", "validation"]
GENERATOR_ID = "tabur.supervised-query-row-broad-v3"
DISCOSCM_FAMILY = "discoscm"
STRUCTURED_SCM_FAMILY = "sparse_dag_scm"
WORLD_FAMILIES = (*V2_WORLD_FAMILIES, DISCOSCM_FAMILY)
DISCOSCM_SOURCE_BLOB = "70285476b0dec8222f097721762127ee36afe3c8"
DISCOSCM_TYPE_WEIGHTS = {
    "numeric": 70.0,
    "ordinal": 5.0,
    "binary": 10.0,
    "categorical": 5.0,
    "high_cardinality": 5.0,
}
# Missingness remains available as an explicit generator control.  The v3
# TabUR training default is zero until the current query-row null-mask path can
# consume missingness-expanded token layouts without a shape mismatch.
DISCOSCM_DEFAULT_MISSING_FRAC = 0.0
DISCOSCM_TOKEN_HERITABILITY = 0.75
DISCOSCM_MAX_CARDINALITY = 64
STRUCTURED_SCM_MECHANISMS = (
    "linear",
    "tanh",
    "periodic",
    "threshold",
    "interaction",
)
BROAD_SCALE_PRIOR_ID = "tabur.broad-scale-prior.v1"
BROAD_MIN_FEATURES = 4
BROAD_MAX_FEATURES = 256
BROAD_MIN_ROWS = 16
BROAD_MAX_ROWS = 8192
BROAD_MIN_CELLS = 8192
BROAD_MAX_CELLS = 65536
# This is the v3 TabUR lane capacity, not a QueryBase-wide default.  ``width``
# counts predictors while the model also receives one response column.
TABUR_V3_MODEL_MAX_FEATURES = 1024


@dataclass(frozen=True, slots=True)
class QueryRowSupervisedSyntheticV3Episode:
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


@dataclass(frozen=True, slots=True)
class _StructuredSCMWorld:
    values: np.ndarray
    parents: tuple[tuple[int, ...], ...]
    topology: tuple[int, ...]
    mechanisms: tuple[str, ...]
    graph_family: str


@dataclass(frozen=True, slots=True)
class BroadEpisodeShape:
    width: int
    rows: int
    context_rows: int
    query_rows: int
    cell_budget: int
    scale_band: str


def _seed(root_seed: int, *parts: object) -> int:
    encoded = "|".join((str(root_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def _rng(root_seed: int, *parts: object) -> np.random.Generator:
    return np.random.default_rng(_seed(root_seed, *parts))


def _log_uniform_int(
    rng: np.random.Generator, *, lower: int, upper: int
) -> int:
    return round(np.exp(rng.uniform(np.log(lower), np.log(upper))))


def sample_broad_episode_shape(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition,
    max_features: int = BROAD_MAX_FEATURES,
    max_cells: int = BROAD_MAX_CELLS,
) -> BroadEpisodeShape:
    """Draw a continuous, compute-bounded episode shape independently of family."""

    if max_features < BROAD_MIN_FEATURES:
        raise ValueError(f"max_features must be at least {BROAD_MIN_FEATURES}")
    if max_cells < BROAD_MIN_CELLS:
        raise ValueError(f"max_cells must be at least {BROAD_MIN_CELLS}")
    rng = _rng(root_seed, partition, world_id, "broad-scale")
    width_draw = float(rng.random())
    if width_draw < 0.65:
        scale_band = "typical"
        width = int(np.clip(round(np.exp(rng.normal(np.log(20.0), 0.65))), 4, 64))
    elif width_draw < 0.90:
        scale_band = "medium"
        width = _log_uniform_int(rng, lower=32, upper=128)
    else:
        scale_band = "wide"
        width = _log_uniform_int(rng, lower=128, upper=512)
    width = int(np.clip(width, BROAD_MIN_FEATURES, min(max_features, BROAD_MAX_FEATURES)))

    row_draw = float(rng.random())
    if row_draw < 0.60:
        requested_rows = int(
            np.clip(round(np.exp(rng.normal(np.log(256.0), 0.8))), 16, 1024)
        )
    elif row_draw < 0.90:
        requested_rows = _log_uniform_int(rng, lower=512, upper=4096)
    else:
        requested_rows = _log_uniform_int(rng, lower=4096, upper=BROAD_MAX_ROWS)
    sampled_budget = _log_uniform_int(
        rng, lower=BROAD_MIN_CELLS, upper=min(max_cells, BROAD_MAX_CELLS)
    )
    budget_rows = max(2, sampled_budget // (width + 1))
    rows = int(np.clip(min(requested_rows, budget_rows), BROAD_MIN_ROWS, BROAD_MAX_ROWS))
    train_fraction = float(np.clip(rng.beta(8.0, 2.0), 0.5, 0.95))
    context_rows = int(np.clip(round(rows * train_fraction), 1, rows - 1))
    return BroadEpisodeShape(
        width=width,
        rows=rows,
        context_rows=context_rows,
        query_rows=rows - context_rows,
        cell_budget=sampled_budget,
        scale_band=scale_band,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1.0e-8)


def _sample_population(
    rng: np.random.Generator, *, rows: int, unit_dim: int
) -> tuple[np.ndarray, int]:
    maximum = max(1, int(math.sqrt(rows)))
    choices = np.arange(1, maximum + 1)
    probability = 1.0 / choices
    components = int(rng.choice(choices, p=probability / probability.sum()))
    weights = rng.dirichlet(np.ones(components))
    assignment = rng.choice(components, size=rows, p=weights)
    units = np.zeros((rows, unit_dim), dtype=np.float64)
    for component in range(components):
        selected = assignment == component
        count = int(selected.sum())
        if count == 0:
            continue
        if components == 1:
            mean = np.zeros(unit_dim)
            scale = 1.0
        else:
            mean = float(rng.uniform(1.5, 4.0)) * _normalize(rng.standard_normal(unit_dim))
            scale = float(np.clip(np.exp(rng.normal(-0.2, 0.45)), 0.15, 4.0))
        units[selected] = mean + scale * rng.standard_normal((count, unit_dim))
    return units, components


def _sample_dag(
    rng: np.random.Generator, *, columns: int, independent: np.ndarray
) -> tuple[list[list[int]], list[int], str]:
    response = columns - 1
    predictors = [index for index in range(response) if not independent[index]]
    rng.shuffle(predictors)
    topology = [*predictors, response]
    parents: list[list[int]] = [[] for _ in range(columns)]
    for position, child in enumerate(topology):
        available = topology[:position]
        if not available:
            continue
        parent_count = min(len(available), 6, int(rng.poisson(2.2)))
        if child == response:
            parent_count = max(1, parent_count)
        if parent_count:
            out_degree = np.asarray(
                [1 + sum(parent in group for group in parents) for parent in available],
                dtype=np.float64,
            )
            chosen = rng.choice(
                np.asarray(available),
                size=parent_count,
                replace=False,
                p=out_degree / out_degree.sum(),
            )
            parents[child] = [int(item) for item in np.atleast_1d(chosen)]
    graph_family = "sparse"
    if len(topology) >= 8 and rng.random() < 0.15:
        graph_family = "star"
        hub = topology[int(rng.integers(0, max(1, len(topology) // 3)))]
        for child in topology[topology.index(hub) + 1 :]:
            if rng.random() < 0.6 and hub not in parents[child]:
                parents[child].append(hub)
    return parents, topology, graph_family


def _activation(value: np.ndarray, name: str) -> np.ndarray:
    if name == "tanh":
        return np.tanh(value)
    if name == "leaky_relu":
        return np.where(value >= 0.0, value, 0.1 * value)
    if name == "sin":
        return np.sin(value)
    return value


def _sample_tokens(
    rng: np.random.Generator,
    *,
    columns: int,
    unit_dim: int,
    parents: list[list[int]],
    topology: list[int],
) -> list[np.ndarray | None]:
    tokens: list[np.ndarray | None] = [None] * columns
    alpha = DISCOSCM_TOKEN_HERITABILITY
    innovation_weight = math.sqrt(1.0 - alpha * alpha)
    for child in topology:
        if not parents[child]:
            tokens[child] = _normalize(rng.standard_normal(unit_dim))
            continue
        weights = rng.uniform(0.5, 2.0, len(parents[child]))
        weights *= rng.choice((-1.0, 1.0), len(parents[child]))
        weights /= np.abs(weights).sum()
        signal = sum(
            weight * tokens[parent]
            for weight, parent in zip(weights, parents[child], strict=True)
        )
        activation = str(
            rng.choice(
                ("identity", "tanh", "leaky_relu", "sin"),
                p=(0.5, 0.2, 0.2, 0.1),
            )
        )
        signal = _normalize(_activation(np.asarray(signal), activation))
        innovation = rng.standard_normal(unit_dim)
        innovation -= float(innovation @ signal) * signal
        innovation = _normalize(innovation)
        tokens[child] = _normalize(alpha * signal + innovation_weight * innovation)
    return tokens


def _class_count(kind: str, rows: int, rng: np.random.Generator) -> int | None:
    if kind == "numeric":
        return None
    if kind == "binary":
        return 2
    if kind == "ordinal":
        return int(rng.integers(3, 9))
    if kind == "categorical":
        return int(rng.integers(8, 33))
    upper = min(DISCOSCM_MAX_CARDINALITY, max(33, rows))
    return int(rng.integers(32, upper + 1))


def _realize_column(
    rng: np.random.Generator,
    *,
    units: np.ndarray,
    token: np.ndarray | None,
    kind: str,
    classes: int | None,
    independent: bool,
    sigma: float,
) -> np.ndarray:
    rows, unit_dim = units.shape
    if kind == "numeric":
        if independent:
            return rng.normal(rng.normal(), np.exp(rng.normal(-0.2, 0.4)), rows)
        assert token is not None
        return (
            np.exp(rng.normal(0.0, 0.35)) * (units @ token)
            + rng.normal(0.0, 0.5)
            + sigma * rng.standard_normal(rows)
        )
    assert classes is not None
    if independent:
        probability = rng.dirichlet(np.full(classes, 0.9))
        return rng.choice(classes, size=rows, p=probability).astype(np.float64)
    assert token is not None
    score = units @ token + sigma * rng.standard_normal(rows)
    if kind == "binary":
        return (score > 0.0).astype(np.float64)
    if kind == "ordinal":
        cuts = np.sort(rng.normal(size=classes - 1))
        return np.sum(score[:, None] > cuts[None, :], axis=1).astype(np.float64)
    residual = rng.standard_normal((classes, unit_dim))
    residual -= np.outer(residual @ token, token)
    class_scale = rng.standard_normal(classes)
    if kind == "high_cardinality":
        scale = np.exp(rng.normal(0.0, 0.9, classes))
        class_scale *= scale
        residual *= scale[:, None]
    embeddings = np.outer(class_scale, token) + 0.25 * residual
    logits = units @ embeddings.T
    logits -= logits.max(axis=1, keepdims=True)
    probability = np.exp(np.clip(logits, -40.0, 40.0))
    probability /= probability.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probability, axis=1)
    cumulative[:, -1] = 1.0
    return (rng.random(rows)[:, None] <= cumulative).argmax(axis=1).astype(np.float64)


def _feature_spec(index: int, kind: str, classes: int | None) -> FeatureSpec:
    name = f"feature_{index:03d}"
    if kind == "numeric":
        return FeatureSpec(name=name)
    feature_kind = FeatureKind.ORDINAL if kind == "ordinal" else FeatureKind.CATEGORICAL
    prefix = "level" if feature_kind is FeatureKind.ORDINAL else "category"
    assert classes is not None
    return FeatureSpec(
        name=name,
        kind=feature_kind,
        domain=tuple(f"{prefix}-{item:03d}" for item in range(classes)),
        codebook_id=f"tabur.discoscm.v1:predictor-{index:03d}",
    )


def _draw_scm_exogenous(
    rng: np.random.Generator, *, rows: int, regime: str
) -> np.ndarray:
    if regime == "gaussian":
        values = rng.standard_normal(rows)
    elif regime == "heavy_tailed":
        values = rng.standard_t(df=3, size=rows)
    elif regime == "skewed":
        values = np.exp(np.clip(rng.standard_normal(rows), -3.0, 3.0)) - 1.0
    elif regime == "mixture":
        values = rng.standard_normal(rows) + 2.0 * (rng.random(rows) > 0.7)
    elif regime == "quantized":
        values = np.round(2.0 * rng.standard_normal(rows)) / 2.0
    elif regime == "bounded":
        values = np.tanh(rng.standard_normal(rows))
    else:
        raise ValueError(f"unknown structured SCM exogenous regime: {regime!r}")
    return np.clip(values, -20.0, 20.0).astype(np.float64)


def _sample_structured_scm_dag(
    rng: np.random.Generator, *, columns: int
) -> tuple[list[list[int]], list[int], str]:
    response = columns - 1
    predictors = list(range(response))
    rng.shuffle(predictors)
    topology = [*predictors, response]
    parents: list[list[int]] = [[] for _ in range(columns)]
    for position, child in enumerate(topology):
        available = topology[:position]
        if not available:
            continue
        parent_count = min(len(available), 4, int(rng.poisson(1.8)))
        if child == response:
            parent_count = max(1, parent_count)
        if parent_count:
            chosen = rng.choice(
                np.asarray(available), size=parent_count, replace=False
            )
            parents[child] = [int(item) for item in np.atleast_1d(chosen)]
    if len(predictors) > 1 and not any(parents[index] for index in predictors):
        parents[topology[1]] = [topology[0]]
    graph_family = "sparse"
    if columns >= 8 and rng.random() < 0.2:
        graph_family = "star"
        hub = topology[int(rng.integers(0, max(1, len(topology) // 3)))]
        for child in topology[topology.index(hub) + 1 :]:
            if rng.random() < 0.55 and hub not in parents[child]:
                parents[child].append(hub)
    return parents, topology, graph_family


def _apply_structural_equation(
    rng: np.random.Generator,
    *,
    parent_values: np.ndarray,
    mechanism: str,
    noise_scale: float,
    exogenous_regime: str,
) -> np.ndarray:
    parent_count = parent_values.shape[1]
    weights = rng.uniform(0.5, 2.0, parent_count)
    weights *= rng.choice((-1.0, 1.0), parent_count)
    weights /= np.abs(weights).sum()
    linear_index = parent_values @ weights
    if mechanism == "linear":
        signal = linear_index
    elif mechanism == "tanh":
        signal = np.tanh(1.5 * linear_index)
    elif mechanism == "periodic":
        signal = np.sin(linear_index) + 0.25 * np.cos(2.0 * linear_index)
    elif mechanism == "threshold":
        signal = np.where(linear_index >= 0.0, 0.75, -0.75) + 0.25 * linear_index
    elif mechanism == "interaction" and parent_count >= 2:
        signal = linear_index + 0.35 * np.tanh(parent_values[:, 0] * parent_values[:, 1])
    else:
        signal = np.tanh(linear_index)
    noise = _draw_scm_exogenous(
        rng, rows=parent_values.shape[0], regime=exogenous_regime
    )
    return np.clip(signal + noise_scale * noise, -20.0, 20.0)


def _sample_structured_scm_world(
    rng: np.random.Generator,
    *,
    rows: int,
    width: int,
    predictor_regime: str,
    noise_level: str,
) -> _StructuredSCMWorld:
    columns = width + 1
    parents, topology, graph_family = _sample_structured_scm_dag(
        rng, columns=columns
    )
    noise_scale = {"low": 0.08, "medium": 0.2, "high": 0.5}[noise_level]
    values = np.zeros((rows, columns), dtype=np.float64)
    mechanisms = ["root_exogenous"] * columns
    for child in topology:
        if not parents[child]:
            values[:, child] = _draw_scm_exogenous(
                rng, rows=rows, regime=predictor_regime
            )
            continue
        mechanism = str(rng.choice(STRUCTURED_SCM_MECHANISMS))
        mechanisms[child] = mechanism
        values[:, child] = _apply_structural_equation(
            rng,
            parent_values=values[:, parents[child]],
            mechanism=mechanism,
            noise_scale=noise_scale,
            exogenous_regime=predictor_regime,
        )
    return _StructuredSCMWorld(
        values=values,
        parents=tuple(tuple(group) for group in parents),
        topology=tuple(topology),
        mechanisms=tuple(mechanisms),
        graph_family=graph_family,
    )


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
    shape = sample_broad_episode_shape(
        root_seed=root_seed, world_id=world_id, partition=partition
    )
    family_rng = _rng(root_seed, partition, world_id, "family")
    regime_rng = _rng(root_seed, partition, world_id, "predictor-regime")
    noise_rng = _rng(root_seed, partition, world_id, "noise-level")
    resolved_family = family or str(family_rng.choice(WORLD_FAMILIES))
    resolved_width = width or shape.width
    if resolved_family == DISCOSCM_FAMILY:
        if predictor_regime not in (None, "mixed_type_disco"):
            raise ValueError("DiscoSCM v3 uses the mixed_type_disco predictor regime")
        resolved_regime = "mixed_type_disco"
    else:
        resolved_regime = predictor_regime or str(regime_rng.choice(PREDICTOR_REGIMES))
    resolved_noise = noise_level or str(noise_rng.choice(NOISE_LEVELS))
    resolved_context = context_rows or shape.context_rows
    if resolved_family not in WORLD_FAMILIES:
        raise ValueError(f"unknown v3 world family: {resolved_family!r}")
    if not BROAD_MIN_FEATURES <= resolved_width <= BROAD_MAX_FEATURES:
        raise ValueError(
            f"v3 width must be in [{BROAD_MIN_FEATURES}, {BROAD_MAX_FEATURES}]"
        )
    if resolved_family != DISCOSCM_FAMILY and resolved_regime not in PREDICTOR_REGIMES:
        raise ValueError(f"unknown v3 predictor regime: {resolved_regime!r}")
    if resolved_noise not in NOISE_LEVELS:
        raise ValueError(f"unknown v3 noise level: {resolved_noise!r}")
    if not 1 <= resolved_context < BROAD_MAX_ROWS:
        raise ValueError(f"v3 context_rows must be in [1, {BROAD_MAX_ROWS - 1}]")
    return resolved_width, resolved_family, resolved_regime, resolved_noise, resolved_context


def _wrap_v2_episode(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition,
    width: int,
    family: str,
    predictor_regime: str,
    noise_level: str,
    context_rows: int,
    rows: int | None,
) -> QueryRowSupervisedSyntheticV3Episode:
    total_rows = rows or context_rows * 2
    if total_rows <= context_rows or context_rows < 1:
        raise ValueError("rows must be greater than positive context_rows")
    world_generator = _generator(root_seed, partition, world_id, "v3-world")
    row_generator = _generator(root_seed, partition, world_id, "v3-rows")
    predictors = _draw_predictors(width, total_rows, predictor_regime, row_generator)
    response = _response(predictors, family=family, generator=world_generator)
    signal_scale = response.std().clamp_min(1.0e-6)
    snr = {"low": 3.0, "medium": 10.0, "high": 30.0}[noise_level]
    response = response + signal_scale / snr * torch.randn(
        total_rows, generator=row_generator
    )
    permutation = torch.randperm(width, generator=world_generator)
    values = torch.cat((predictors[:, permutation], response.unsqueeze(1)), dim=1)
    target_mask = torch.zeros_like(values, dtype=torch.bool)
    target_mask[context_rows:, -1] = True
    forward_values = values.masked_fill(target_mask, 0.0)
    feature_names = (*tuple(f"feature_{index:03d}" for index in range(width)), "response")
    feature_specs = (
        *(
            FeatureSpec(name=name, role=FeatureRole.PREDICTOR)
            for name in feature_names[:-1]
        ),
        FeatureSpec(name="response", role=FeatureRole.RESPONSE),
    )
    origins = [
        [
            OriginState.QUERY if target_mask[row, column] else OriginState.OBSERVED
            for column in range(width + 1)
        ]
        for row in range(total_rows)
    ]
    roles = [
        [
            ForwardRole.RECEIVER | ForwardRole.TARGET
            if target_mask[row, column]
            else ForwardRole.RECEIVER | ForwardRole.SOURCE
            for column in range(width + 1)
        ]
        for row in range(total_rows)
    ]
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v3",
        source_partition=f"synthetic_{partition}",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"{world_id}-row-{index}" for index in range(total_rows)),
        feature_names=feature_names,
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_id": GENERATOR_ID,
            "generator_version": "v3",
            "scale_prior_id": BROAD_SCALE_PRIOR_ID,
            "partition": partition,
            "world_id": world_id,
            "world_family": family,
            "predictor_regime": predictor_regime,
            "width": width,
            "rows": total_rows,
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
                "schema": "tabur.supervised.synthetic.v3.recipe.v1",
                "generator_id": GENERATOR_ID,
                "scale_prior_id": BROAD_SCALE_PRIOR_ID,
                "root_seed": root_seed,
                "partition": partition,
                "world_id": world_id,
                "family": family,
                "predictor_regime": predictor_regime,
                "width": width,
                "rows": total_rows,
                "noise_level": noise_level,
                "context_rows": context_rows,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=evidence.feature_names,
        target_values=values.masked_fill(~target_mask, 0.0),
        target_mask=target_mask,
        metadata={"truth_scope": "loss_only", "generator_id": GENERATOR_ID},
    )
    return QueryRowSupervisedSyntheticV3Episode(
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


def _make_structured_scm_episode(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition,
    width: int,
    predictor_regime: str,
    noise_level: str,
    context_rows: int,
    rows: int | None,
) -> QueryRowSupervisedSyntheticV3Episode:
    total_rows = rows or context_rows * 2
    if total_rows <= context_rows or context_rows < 1:
        raise ValueError("rows must be greater than positive context_rows")
    rng = _rng(root_seed, partition, world_id, "structured-scm-world")
    world = _sample_structured_scm_world(
        rng,
        rows=total_rows,
        width=width,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
    )
    values = torch.as_tensor(world.values, dtype=torch.float32)
    target_mask = torch.zeros_like(values, dtype=torch.bool)
    target_mask[context_rows:, -1] = True
    forward_values = values.masked_fill(target_mask, 0.0)
    origins = [
        [
            OriginState.QUERY if target_mask[row, column] else OriginState.OBSERVED
            for column in range(width + 1)
        ]
        for row in range(total_rows)
    ]
    roles = [
        [
            ForwardRole.RECEIVER | ForwardRole.TARGET
            if target_mask[row, column]
            else ForwardRole.RECEIVER | ForwardRole.SOURCE
            for column in range(width + 1)
        ]
        for row in range(total_rows)
    ]
    feature_specs = (
        *(FeatureSpec(name=f"feature_{index:03d}") for index in range(width)),
        FeatureSpec(name="response", role=FeatureRole.RESPONSE),
    )
    feature_names = tuple(spec.name for spec in feature_specs)
    graph_hash = canonical_hash(
        {
            "schema": "tabur.structured-scm.graph.v1",
            "parents": world.parents,
            "topology": world.topology,
            "mechanisms": world.mechanisms,
        }
    )
    mechanism_counts = {
        mechanism: world.mechanisms.count(mechanism)
        for mechanism in sorted(set(world.mechanisms))
    }
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v3",
        source_partition=f"synthetic_{partition}",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"{world_id}-row-{index}" for index in range(total_rows)),
        feature_names=feature_names,
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_id": GENERATOR_ID,
            "generator_version": "v3",
            "scale_prior_id": BROAD_SCALE_PRIOR_ID,
            "partition": partition,
            "world_id": world_id,
            "world_family": STRUCTURED_SCM_FAMILY,
            "predictor_regime": predictor_regime,
            "width": width,
            "noise_level": noise_level,
            "context_rows": context_rows,
            "truth_boundary": "sidecar_only",
            "scm_scope": "observed_columns",
            "row_sampling": "iid_from_shared_structural_equations",
            "latent_unit_representation": "absent",
            "graph_family": world.graph_family,
            "graph_hash": graph_hash,
            "edge_count": sum(len(group) for group in world.parents),
            "non_root_predictor_count": sum(
                bool(world.parents[index]) for index in range(width)
            ),
            "response_parent_count": len(world.parents[-1]),
            "mechanism_counts": mechanism_counts,
        },
    )
    sidecar = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.v3.recipe.v1",
                "generator_id": GENERATOR_ID,
                "scale_prior_id": BROAD_SCALE_PRIOR_ID,
                "root_seed": root_seed,
                "partition": partition,
                "world_id": world_id,
                "family": STRUCTURED_SCM_FAMILY,
                "width": width,
                "predictor_regime": predictor_regime,
                "noise_level": noise_level,
                "context_rows": context_rows,
                "rows": total_rows,
                "graph_hash": graph_hash,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=feature_names,
        target_values=values.masked_fill(~target_mask, 0.0),
        target_mask=target_mask,
        metadata={"truth_scope": "loss_only", "generator_id": GENERATOR_ID},
    )
    return QueryRowSupervisedSyntheticV3Episode(
        evidence=evidence,
        sidecar=sidecar,
        world_id=world_id,
        partition=partition,
        family=STRUCTURED_SCM_FAMILY,
        width=width,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=context_rows,
    )


def _make_discoscm_episode(
    *,
    root_seed: int,
    world_id: str,
    partition: GeneratorPartition,
    width: int,
    noise_level: str,
    context_rows: int,
    rows: int | None,
    missing_frac: float,
) -> QueryRowSupervisedSyntheticV3Episode:
    total_rows = rows or context_rows * 2
    if total_rows <= context_rows or context_rows < 1:
        raise ValueError("rows must be greater than positive context_rows")
    rng = _rng(root_seed, partition, world_id, "discoscm-world")
    unit_dim = int(np.clip(round(np.exp(rng.normal(np.log(16.0), 0.45))), 2, 64))
    units, mixture_components = _sample_population(rng, rows=total_rows, unit_dim=unit_dim)
    type_names = tuple(DISCOSCM_TYPE_WEIGHTS)
    type_probability = np.asarray(tuple(DISCOSCM_TYPE_WEIGHTS.values()), dtype=np.float64)
    type_probability /= type_probability.sum()
    predictor_kinds = [str(item) for item in rng.choice(type_names, width, p=type_probability)]
    kinds = [*predictor_kinds, "numeric"]
    classes = [_class_count(kind, total_rows, rng) for kind in kinds]
    independent = rng.random(width + 1) < 0.05
    independent[-1] = False
    parents, topology, graph_family = _sample_dag(
        rng, columns=width + 1, independent=independent
    )
    tokens = _sample_tokens(
        rng,
        columns=width + 1,
        unit_dim=unit_dim,
        parents=parents,
        topology=topology,
    )
    sigma = {"low": 0.1, "medium": 0.3, "high": 0.7}[noise_level]
    columns = [
        _realize_column(
            rng,
            units=units,
            token=tokens[index],
            kind=kind,
            classes=classes[index],
            independent=bool(independent[index]),
            sigma=sigma,
        )
        for index, kind in enumerate(kinds)
    ]
    values = torch.as_tensor(np.column_stack(columns), dtype=torch.float32)
    missing_frac = float(np.clip(missing_frac, 0.0, 0.95))
    missing = torch.zeros_like(values, dtype=torch.bool)
    missing[:, :width] = torch.as_tensor(
        rng.random((total_rows, width)) < missing_frac
    )
    target_mask = torch.zeros_like(values, dtype=torch.bool)
    target_mask[context_rows:, -1] = True
    forward_values = values.masked_fill(missing | target_mask, 0.0)
    origins: list[list[OriginState]] = []
    roles: list[list[ForwardRole]] = []
    for row in range(total_rows):
        origin_row: list[OriginState] = []
        role_row: list[ForwardRole] = []
        for column in range(width + 1):
            if target_mask[row, column]:
                origin_row.append(OriginState.QUERY)
                role_row.append(ForwardRole.RECEIVER | ForwardRole.TARGET)
            elif missing[row, column]:
                origin_row.append(OriginState.NATURAL_MISSING)
                role_row.append(ForwardRole.RECEIVER)
            else:
                origin_row.append(OriginState.OBSERVED)
                role_row.append(ForwardRole.RECEIVER | ForwardRole.SOURCE)
        origins.append(origin_row)
        roles.append(role_row)
    feature_specs = (
        *(
            _feature_spec(index, kind, classes[index])
            for index, kind in enumerate(predictor_kinds)
        ),
        FeatureSpec(name="response", role=FeatureRole.RESPONSE),
    )
    feature_names = tuple(spec.name for spec in feature_specs)
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v3",
        source_partition=f"synthetic_{partition}",
        fit_partition="synthetic_fit",
        row_ids=tuple(f"{world_id}-row-{index}" for index in range(total_rows)),
        feature_names=feature_names,
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_id": GENERATOR_ID,
            "generator_version": "v3",
            "scale_prior_id": BROAD_SCALE_PRIOR_ID,
            "partition": partition,
            "world_id": world_id,
            "world_family": DISCOSCM_FAMILY,
            "predictor_regime": "mixed_type_disco",
            "width": width,
            "noise_level": noise_level,
            "context_rows": context_rows,
            "truth_boundary": "sidecar_only",
            "discoscm_source_blob": DISCOSCM_SOURCE_BLOB,
            "unit_dim": unit_dim,
            "population_components": mixture_components,
            "graph_family": graph_family,
            "edge_count": sum(len(item) for item in parents),
            "feature_kinds": kinds,
            "missing_fraction": missing_frac,
            "token_heritability": DISCOSCM_TOKEN_HERITABILITY,
            "latent_mechanism_visibility": "sidecar_omitted",
        },
    )
    sidecar = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.v3.recipe.v1",
                "generator_id": GENERATOR_ID,
                "scale_prior_id": BROAD_SCALE_PRIOR_ID,
                "root_seed": root_seed,
                "partition": partition,
                "world_id": world_id,
                "family": DISCOSCM_FAMILY,
                "width": width,
                "noise_level": noise_level,
                "context_rows": context_rows,
                "rows": total_rows,
                "missing_fraction": missing_frac,
                "discoscm_source_blob": DISCOSCM_SOURCE_BLOB,
            }
        ),
        row_ids=evidence.row_ids,
        feature_names=feature_names,
        target_values=values.masked_fill(~target_mask, 0.0),
        target_mask=target_mask,
        metadata={"truth_scope": "loss_only", "generator_id": GENERATOR_ID},
    )
    return QueryRowSupervisedSyntheticV3Episode(
        evidence=evidence,
        sidecar=sidecar,
        world_id=world_id,
        partition=partition,
        family=DISCOSCM_FAMILY,
        width=width,
        predictor_regime="mixed_type_disco",
        noise_level=noise_level,
        context_rows=context_rows,
    )


def make_query_row_supervised_synthetic_v3_episode(
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
    missing_frac: float = DISCOSCM_DEFAULT_MISSING_FRAC,
) -> QueryRowSupervisedSyntheticV3Episode:
    """Generate one deterministic v3 world from the broad family mixture."""

    if not world_id.strip():
        raise ValueError("world_id must not be empty")
    default_shape = sample_broad_episode_shape(
        root_seed=root_seed, world_id=world_id, partition=partition
    )
    width = width or default_shape.width
    context_rows = context_rows or default_shape.context_rows
    rows = rows or default_shape.rows
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
    if family == DISCOSCM_FAMILY:
        return _make_discoscm_episode(
            root_seed=root_seed,
            world_id=world_id,
            partition=partition,
            width=width,
            noise_level=noise_level,
            context_rows=context_rows,
            rows=rows,
            missing_frac=missing_frac,
        )
    if family == STRUCTURED_SCM_FAMILY:
        if missing_frac != DISCOSCM_DEFAULT_MISSING_FRAC:
            raise ValueError("missing_frac is currently a DiscoSCM-only v3 control")
        return _make_structured_scm_episode(
            root_seed=root_seed,
            world_id=world_id,
            partition=partition,
            width=width,
            predictor_regime=predictor_regime,
            noise_level=noise_level,
            context_rows=context_rows,
            rows=rows,
        )
    if missing_frac != DISCOSCM_DEFAULT_MISSING_FRAC:
        raise ValueError("missing_frac is currently a DiscoSCM-only v3 control")
    return _wrap_v2_episode(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
        width=width,
        family=family,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=context_rows,
        rows=rows,
    )


def build_query_row_supervised_synthetic_v3_plan(
    *, root_seed: int, worlds: int, partition: GeneratorPartition
) -> tuple[dict[str, Any], ...]:
    """Freeze a balanced v3 family plan before materializing any rows."""

    if worlds <= 0:
        raise ValueError("worlds must be positive")
    plan = []
    for index in range(worlds):
        world_id = f"{partition}-world-{index:06d}"
        block = index // len(WORLD_FAMILIES)
        position = index % len(WORLD_FAMILIES)
        family_order = _rng(
            root_seed, partition, block, "family-coverage-block"
        ).permutation(WORLD_FAMILIES)
        family = str(family_order[position])
        shape = sample_broad_episode_shape(
            root_seed=root_seed, world_id=world_id, partition=partition
        )
        regime = (
            "mixed_type_disco"
            if family == DISCOSCM_FAMILY
            else str(
                _rng(root_seed, partition, world_id, "predictor-regime").choice(
                    PREDICTOR_REGIMES
                )
            )
        )
        noise = str(
            _rng(root_seed, partition, world_id, "noise-level").choice(NOISE_LEVELS)
        )
        plan.append(
            {
                "world_id": world_id,
                "partition": partition,
                "family": family,
                "width": shape.width,
                "predictor_regime": regime,
                "noise_level": noise,
                "context_rows": shape.context_rows,
                "rows": shape.rows,
            }
        )
    return tuple(plan)


def substitute_query_truth(
    episode: QueryRowSupervisedSyntheticV3Episode, *, value: float
) -> QueryRowSupervisedSyntheticV3Episode:
    values = episode.sidecar.target_values.clone()
    values[episode.sidecar.target_mask] = value
    return replace(episode, sidecar=replace(episode.sidecar, target_values=values))


def supervised_synthetic_v3_episode_loss(
    model: Any, episode: QueryRowSupervisedSyntheticV3Episode
) -> Tensor:
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
        raise RuntimeError("v3 episode has no supported supervised target")
    error = raw - truth.to(dtype=raw.dtype)
    return torch.where(scored, error.square(), torch.zeros_like(error)).sum() / scored.sum()


def validate_query_row_supervised_synthetic_v3(
    *, root_seed: int = 1729, worlds: int = 64
) -> dict[str, Any]:
    train = build_query_row_supervised_synthetic_v3_plan(
        root_seed=root_seed, worlds=worlds, partition="train"
    )
    validation = build_query_row_supervised_synthetic_v3_plan(
        root_seed=root_seed, worlds=worlds, partition="validation"
    )
    sample = make_query_row_supervised_synthetic_v3_episode(
        root_seed=root_seed,
        world_id="train-discoscm-smoke",
        family=DISCOSCM_FAMILY,
        width=8,
        noise_level="medium",
        context_rows=8,
        rows=16,
    )
    replay = make_query_row_supervised_synthetic_v3_episode(
        root_seed=root_seed,
        world_id="train-discoscm-smoke",
        family=DISCOSCM_FAMILY,
        width=8,
        noise_level="medium",
        context_rows=8,
        rows=16,
    )
    structured = make_query_row_supervised_synthetic_v3_episode(
        root_seed=root_seed,
        world_id="train-structured-scm-smoke",
        family=STRUCTURED_SCM_FAMILY,
        predictor_regime="gaussian",
        width=8,
        noise_level="medium",
        context_rows=8,
        rows=16,
    )
    structured_replay = make_query_row_supervised_synthetic_v3_episode(
        root_seed=root_seed,
        world_id="train-structured-scm-smoke",
        family=STRUCTURED_SCM_FAMILY,
        predictor_regime="gaussian",
        width=8,
        noise_level="medium",
        context_rows=8,
        rows=16,
    )
    substituted = substitute_query_truth(sample, value=123.0)
    widths = [int(item["width"]) for item in train]
    rows = [int(item["rows"]) for item in train]
    contexts = [int(item["context_rows"]) for item in train]
    cells = [
        (int(item["width"]) + 1) * int(item["rows"])
        for item in train
    ]
    legacy_widths = {6, 8, 9, 11, 17, 21, 32}
    legacy_contexts = {8, 16, 32, 64, 128, 256, 512}
    exits = {
        "deterministic_replay": sample.evidence.evidence_hash == replay.evidence.evidence_hash,
        "truth_substitution_isolation": (
            sample.evidence.evidence_hash == substituted.evidence.evidence_hash
            and not torch.equal(sample.sidecar.target_values, substituted.sidecar.target_values)
        ),
        "all_families_planned": {item["family"] for item in train} == set(WORLD_FAMILIES),
        "discoscm_planned": any(item["family"] == DISCOSCM_FAMILY for item in train),
        "broad_continuous_shapes": (
            max(widths) > 32
            and any(width not in legacy_widths for width in widths)
            and any(context not in legacy_contexts for context in contexts)
        ),
        "compute_bounded_shapes": (
            all(BROAD_MIN_FEATURES <= width <= BROAD_MAX_FEATURES for width in widths)
            and all(BROAD_MIN_ROWS <= row_count <= BROAD_MAX_ROWS for row_count in rows)
            and all(cell_count <= BROAD_MAX_CELLS for cell_count in cells)
        ),
        "structured_scm_deterministic_replay": (
            structured.evidence.evidence_hash == structured_replay.evidence.evidence_hash
        ),
        "structured_scm_table_wide_dag": (
            structured.evidence.metadata["scm_scope"] == "observed_columns"
            and structured.evidence.metadata["non_root_predictor_count"] > 0
            and structured.evidence.metadata["response_parent_count"] > 0
        ),
        "structured_scm_has_no_latent_units": (
            structured.evidence.metadata["latent_unit_representation"] == "absent"
        ),
        "query_truth_physically_hidden": bool(
            (sample.evidence.forward_values[sample.sidecar.target_mask] == 0).all()
        ),
        "latent_mechanism_not_exposed": not any(
            key in sample.evidence.metadata for key in ("units", "tokens", "parents", "dag")
        ),
        "disjoint_train_validation_world_ids": {
            item["world_id"] for item in train
        }.isdisjoint({item["world_id"] for item in validation}),
    }
    return {
        "schema_version": "tabu.query-row.supervised-synthetic-v3-validation.v1",
        "generator_id": GENERATOR_ID,
        "worlds_per_partition": worlds,
        "families": list(WORLD_FAMILIES),
        "scale_prior_id": BROAD_SCALE_PRIOR_ID,
        "shape_coverage": {
            "width_min": min(widths),
            "width_max": max(widths),
            "width_unique": len(set(widths)),
            "rows_min": min(rows),
            "rows_max": max(rows),
            "rows_unique": len(set(rows)),
            "context_rows_min": min(contexts),
            "context_rows_max": max(contexts),
            "context_rows_unique": len(set(contexts)),
            "cells_max": max(cells),
        },
        "exits": exits,
        "status": "passed" if all(exits.values()) else "failed",
        "evidence_status": "local_unissued",
        "claim_boundary": "generator validation only; no pretraining or capability claim",
    }


__all__ = [
    "BROAD_MAX_CELLS",
    "BROAD_MAX_FEATURES",
    "BROAD_MAX_ROWS",
    "BROAD_MIN_CELLS",
    "BROAD_MIN_FEATURES",
    "BROAD_MIN_ROWS",
    "BROAD_SCALE_PRIOR_ID",
    "DISCOSCM_DEFAULT_MISSING_FRAC",
    "DISCOSCM_FAMILY",
    "DISCOSCM_SOURCE_BLOB",
    "GENERATOR_ID",
    "STRUCTURED_SCM_FAMILY",
    "STRUCTURED_SCM_MECHANISMS",
    "TABUR_V3_MODEL_MAX_FEATURES",
    "WORLD_FAMILIES",
    "BroadEpisodeShape",
    "QueryRowSupervisedSyntheticV3Episode",
    "build_query_row_supervised_synthetic_v3_plan",
    "make_query_row_supervised_synthetic_v3_episode",
    "sample_broad_episode_shape",
    "substitute_query_truth",
    "supervised_synthetic_v3_episode_loss",
    "validate_query_row_supervised_synthetic_v3",
]
