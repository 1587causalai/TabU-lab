"""Categorical-response capsule over the immutable compute-capped v3.1 prior.

The parent generator remains numeric-response-only.  This successor preserves
its world, shape, predictor, and routing-budget identities, then discretizes
the response using thresholds derived exclusively from visible context rows.
Query response values remain physically absent from model-facing evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import torch
from torch import Tensor

from tabu_lab.contracts import FeatureKind, FeatureRole, FeatureSpec, canonical_hash

from .query_row_supervised_synthetic_v3 import QueryRowSupervisedSyntheticV3Episode
from .query_row_supervised_synthetic_v3_capped import (
    DEFAULT_MAX_ROUTING_PAIRS,
    make_query_row_supervised_synthetic_v3_capped_episode,
)

GENERATOR_ID = "tabur.supervised-query-row-broad-v3-categorical-response"
GENERATOR_VERSION = "3.2.0"
RESPONSE_SCHEMA_ID = "tabur.context-quantile-categorical-response.v1"
MIN_RESPONSE_CLASSES = 2
MAX_RESPONSE_CLASSES = 8


def _schema_draw(root_seed: int, partition: str, world_id: str) -> tuple[str, int]:
    digest = hashlib.sha256(
        f"{root_seed}|{partition}|{world_id}|categorical-response-schema".encode()
    ).digest()
    if digest[0] < 128:
        return "binary", 2
    return "categorical", 3 + digest[1] % (MAX_RESPONSE_CLASSES - 2)


def _context_split_thresholds(values: Tensor, requested_classes: int) -> Tensor:
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("categorical response requires at least two context values")
    ordered = values.detach().to(dtype=torch.float64, device="cpu").sort().values
    if int(torch.unique_consecutive(ordered).numel()) < 2:
        raise ValueError("categorical response requires non-constant context values")
    candidates: list[Tensor] = []
    count = int(ordered.numel())
    for split in range(1, requested_classes):
        index = min(count - 1, max(1, split * count // requested_classes))
        left = ordered[index - 1]
        right = ordered[index]
        if not bool(right > left):
            continue
        threshold = left + (right - left) * 0.5
        if not candidates or bool(threshold > candidates[-1]):
            candidates.append(threshold)
    if not candidates:
        unique = torch.unique_consecutive(ordered)
        index = max(1, int(unique.numel()) // 2)
        left = unique[index - 1]
        right = unique[index]
        candidates.append(left + (right - left) * 0.5)
    return torch.stack(candidates).to(dtype=values.dtype)


def make_query_row_supervised_synthetic_v3_categorical_episode(
    *,
    root_seed: int,
    world_id: str,
    partition: str = "train",
    width: int | None = None,
    family: str | None = None,
    predictor_regime: str | None = None,
    noise_level: str | None = None,
    context_rows: int | None = None,
    rows: int | None = None,
    missing_frac: float = 0.0,
    scm_missingness_family: str | None = None,
    scm_missingness_rate: float | None = None,
    max_routing_pairs: int = DEFAULT_MAX_ROUTING_PAIRS,
) -> QueryRowSupervisedSyntheticV3Episode:
    """Generate one typed categorical-response episode without query-truth leakage."""

    parent = make_query_row_supervised_synthetic_v3_capped_episode(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
        width=width,
        family=family,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=context_rows,
        rows=rows,
        missing_frac=missing_frac,
        scm_missingness_family=scm_missingness_family,
        scm_missingness_rate=scm_missingness_rate,
        max_routing_pairs=max_routing_pairs,
    )
    target_mask = parent.sidecar.target_mask
    response_target_mask = target_mask[:, -1]
    if not bool(response_target_mask.any()) or bool(target_mask[:, :-1].any()):
        raise ValueError("categorical response capsule requires response-only query targets")

    response = parent.evidence.forward_values[:, -1].clone()
    response[response_target_mask] = parent.sidecar.target_values[response_target_mask, -1]
    _, requested_classes = _schema_draw(root_seed, partition, world_id)
    thresholds = _context_split_thresholds(
        response[: parent.context_rows],
        requested_classes=requested_classes,
    )
    codes = torch.bucketize(response, thresholds).to(dtype=torch.float32)
    class_count = int(thresholds.numel()) + 1
    response_kind = "binary" if class_count == 2 else "categorical"
    threshold_hash = canonical_hash(tuple(float(value) for value in thresholds))
    codebook_id = f"{RESPONSE_SCHEMA_ID}:{threshold_hash[:24]}"
    domain = tuple(f"class-{index:03d}" for index in range(class_count))

    forward_values = parent.evidence.forward_values.clone()
    forward_values[: parent.context_rows, -1] = codes[: parent.context_rows]
    forward_values[response_target_mask, -1] = 0.0
    feature_specs = (
        *parent.evidence.feature_specs[:-1],
        FeatureSpec(
            name=parent.evidence.feature_names[-1],
            kind=FeatureKind.CATEGORICAL,
            domain=domain,
            codebook_id=codebook_id,
            role=FeatureRole.RESPONSE,
        ),
    )
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    response_metadata = {
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "parent_generator_id": parent.generator_id,
        "response_value_family": "categorical",
        "response_kind": response_kind,
        "response_class_count": class_count,
        "response_schema_id": RESPONSE_SCHEMA_ID,
        "response_codebook_id": codebook_id,
        "response_threshold_hash": threshold_hash,
        "response_schema_source": "visible_context_only",
    }
    evidence = replace(
        parent.evidence,
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v3-categorical-response",
        forward_values=forward_values,
        feature_specs=feature_specs,
        metadata={**dict(parent.evidence.metadata), **response_metadata},
    )
    target_values = torch.zeros_like(parent.sidecar.target_values)
    target_values[response_target_mask, -1] = codes[response_target_mask]
    sidecar = replace(
        parent.sidecar,
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.v3.categorical-response.recipe.v1",
                "parent_recipe_hash": parent.sidecar.recipe_hash,
                "generator_id": GENERATOR_ID,
                "response_schema_id": RESPONSE_SCHEMA_ID,
                "response_kind": response_kind,
                "response_class_count": class_count,
                "response_threshold_hash": threshold_hash,
            }
        ),
        target_values=target_values,
        metadata={**dict(parent.sidecar.metadata), **response_metadata},
    )
    return replace(
        parent,
        evidence=evidence,
        sidecar=sidecar,
        generator_id=GENERATOR_ID,
    )


__all__ = [
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "MAX_RESPONSE_CLASSES",
    "MIN_RESPONSE_CLASSES",
    "RESPONSE_SCHEMA_ID",
    "make_query_row_supervised_synthetic_v3_categorical_episode",
]
