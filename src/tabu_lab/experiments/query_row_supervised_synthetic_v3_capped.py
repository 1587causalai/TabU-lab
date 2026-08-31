"""Compute-capped successor to the immutable broad supervised-v3 prior.

The v3.0 prior bounds materialized cells, but the current same-column numeric
terminal materializes a routing ledger proportional to ``rows**2 * features``.
This generator keeps v3.0's world families and continuous width prior while
adding an explicit routing-pair budget.  It is a new generator identity: the
v3.0 implementation and every run that selected it remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from tabu_lab.contracts import canonical_hash

from .query_row_supervised_synthetic_v3 import (
    BROAD_MIN_ROWS,
    QueryRowSupervisedSyntheticV3Episode,
    build_query_row_supervised_synthetic_v3_plan,
    make_query_row_supervised_synthetic_v3_episode,
    sample_broad_episode_shape,
)
from .query_row_supervised_synthetic_v3 import (
    GENERATOR_ID as PARENT_GENERATOR_ID,
)

GENERATOR_ID = "tabur.supervised-query-row-broad-v3-compute-capped"
GENERATOR_VERSION = "3.1.0"
ROUTING_BUDGET_ID = "tabur.same-column-routing-pair-budget.v1"
DEFAULT_MAX_ROUTING_PAIRS = 8_000_000


@dataclass(frozen=True, slots=True)
class ComputeCappedShape:
    """Episode shape plus the quadratic cost used to admit it."""

    width: int
    rows: int
    context_rows: int
    query_rows: int
    requested_rows: int
    requested_context_rows: int
    routing_pairs: int
    max_routing_pairs: int
    was_capped: bool


def cap_same_column_routing_shape(
    *,
    width: int,
    rows: int,
    context_rows: int,
    max_routing_pairs: int = DEFAULT_MAX_ROUTING_PAIRS,
) -> ComputeCappedShape:
    """Cap rows so the dense same-column routing ledger has a fixed ceiling."""

    if width < 1:
        raise ValueError("width must be positive")
    if rows <= context_rows or context_rows < 1:
        raise ValueError("rows must be greater than positive context_rows")
    minimum_cost = (width + 1) * BROAD_MIN_ROWS * BROAD_MIN_ROWS
    if max_routing_pairs < minimum_cost:
        raise ValueError(
            "max_routing_pairs cannot admit the minimum broad-prior episode shape"
        )
    maximum_rows = math.isqrt(max_routing_pairs // (width + 1))
    capped_rows = min(rows, maximum_rows)
    context_fraction = context_rows / rows
    capped_context = min(
        capped_rows - 1,
        max(1, round(capped_rows * context_fraction)),
    )
    routing_pairs = (width + 1) * capped_rows * capped_rows
    return ComputeCappedShape(
        width=width,
        rows=capped_rows,
        context_rows=capped_context,
        query_rows=capped_rows - capped_context,
        requested_rows=rows,
        requested_context_rows=context_rows,
        routing_pairs=routing_pairs,
        max_routing_pairs=max_routing_pairs,
        was_capped=capped_rows != rows,
    )


def _resolved_shape(
    *,
    root_seed: int,
    world_id: str,
    partition: str,
    width: int | None,
    rows: int | None,
    context_rows: int | None,
    max_routing_pairs: int,
) -> ComputeCappedShape:
    sampled = sample_broad_episode_shape(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
    )
    resolved_width = sampled.width if width is None else width
    requested_rows = sampled.rows if rows is None else rows
    if context_rows is None:
        requested_context = round(
            requested_rows * sampled.context_rows / sampled.rows
        )
        requested_context = min(requested_rows - 1, max(1, requested_context))
    else:
        requested_context = context_rows
    return cap_same_column_routing_shape(
        width=resolved_width,
        rows=requested_rows,
        context_rows=requested_context,
        max_routing_pairs=max_routing_pairs,
    )


def make_query_row_supervised_synthetic_v3_capped_episode(
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
    """Generate a v3-family episode admitted by the quadratic routing budget."""

    shape = _resolved_shape(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
        width=width,
        rows=rows,
        context_rows=context_rows,
        max_routing_pairs=max_routing_pairs,
    )
    parent = make_query_row_supervised_synthetic_v3_episode(
        root_seed=root_seed,
        world_id=world_id,
        partition=partition,
        width=shape.width,
        family=family,
        predictor_regime=predictor_regime,
        noise_level=noise_level,
        context_rows=shape.context_rows,
        rows=shape.rows,
        missing_frac=missing_frac,
        scm_missingness_family=scm_missingness_family,
        scm_missingness_rate=scm_missingness_rate,
    )
    episode_id = f"{GENERATOR_ID}-{partition}-{world_id}"
    budget_metadata: dict[str, Any] = {
        "parent_generator_id": PARENT_GENERATOR_ID,
        "generator_id": GENERATOR_ID,
        "generator_version": GENERATOR_VERSION,
        "routing_budget_id": ROUTING_BUDGET_ID,
        "requested_rows": shape.requested_rows,
        "requested_context_rows": shape.requested_context_rows,
        "rows": shape.rows,
        "context_rows": shape.context_rows,
        "routing_pairs": shape.routing_pairs,
        "max_routing_pairs": shape.max_routing_pairs,
        "routing_shape_was_capped": shape.was_capped,
    }
    evidence = replace(
        parent.evidence,
        episode_id=episode_id,
        dataset_id="tabur-synthetic-supervised-v3-compute-capped",
        metadata={**dict(parent.evidence.metadata), **budget_metadata},
    )
    sidecar = replace(
        parent.sidecar,
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabur.supervised.synthetic.v3.compute-capped.recipe.v1",
                "parent_recipe_hash": parent.sidecar.recipe_hash,
                "generator_id": GENERATOR_ID,
                "routing_budget_id": ROUTING_BUDGET_ID,
                "max_routing_pairs": shape.max_routing_pairs,
                "routing_pairs": shape.routing_pairs,
                "requested_rows": shape.requested_rows,
                "rows": shape.rows,
                "context_rows": shape.context_rows,
            }
        ),
        metadata={
            **dict(parent.sidecar.metadata),
            "generator_id": GENERATOR_ID,
            "parent_generator_id": PARENT_GENERATOR_ID,
            "routing_budget_id": ROUTING_BUDGET_ID,
        },
    )
    return replace(
        parent,
        evidence=evidence,
        sidecar=sidecar,
        context_rows=shape.context_rows,
        generator_id=GENERATOR_ID,
    )


def build_query_row_supervised_synthetic_v3_capped_plan(
    *,
    root_seed: int,
    worlds: int,
    partition: str,
    max_routing_pairs: int = DEFAULT_MAX_ROUTING_PAIRS,
) -> tuple[dict[str, Any], ...]:
    """Project the immutable v3 plan through the quadratic admission budget."""

    parent_plan = build_query_row_supervised_synthetic_v3_plan(
        root_seed=root_seed,
        worlds=worlds,
        partition=partition,
    )
    projected: list[dict[str, Any]] = []
    for item in parent_plan:
        shape = cap_same_column_routing_shape(
            width=int(item["width"]),
            rows=int(item["rows"]),
            context_rows=int(item["context_rows"]),
            max_routing_pairs=max_routing_pairs,
        )
        projected.append(
            {
                **item,
                "rows": shape.rows,
                "context_rows": shape.context_rows,
                "requested_rows": shape.requested_rows,
                "requested_context_rows": shape.requested_context_rows,
                "routing_pairs": shape.routing_pairs,
                "max_routing_pairs": shape.max_routing_pairs,
                "routing_shape_was_capped": shape.was_capped,
            }
        )
    return tuple(projected)


def validate_query_row_supervised_synthetic_v3_capped(
    *, root_seed: int = 1729, worlds: int = 64
) -> dict[str, Any]:
    """Return deterministic non-evidentiary admission diagnostics."""

    train = build_query_row_supervised_synthetic_v3_capped_plan(
        root_seed=root_seed,
        worlds=worlds,
        partition="train",
    )
    replay = build_query_row_supervised_synthetic_v3_capped_plan(
        root_seed=root_seed,
        worlds=worlds,
        partition="train",
    )
    costs = [int(item["routing_pairs"]) for item in train]
    exits = {
        "deterministic_replay": train == replay,
        "quadratic_budget_respected": all(
            cost <= DEFAULT_MAX_ROUTING_PAIRS for cost in costs
        ),
        "positive_query_support": all(
            1 <= int(item["context_rows"]) < int(item["rows"])
            for item in train
        ),
        "v3_family_plan_retained": all("family" in item for item in train),
    }
    return {
        "schema": "tabur.supervised.synthetic.v3.compute-capped.validation.v1",
        "generator_id": GENERATOR_ID,
        "parent_generator_id": PARENT_GENERATOR_ID,
        "routing_budget_id": ROUTING_BUDGET_ID,
        "worlds": worlds,
        "max_observed_routing_pairs": max(costs),
        "capped_worlds": sum(bool(item["routing_shape_was_capped"]) for item in train),
        "exits": exits,
        "status": "passed" if all(exits.values()) else "failed",
    }


__all__ = [
    "DEFAULT_MAX_ROUTING_PAIRS",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "ROUTING_BUDGET_ID",
    "ComputeCappedShape",
    "build_query_row_supervised_synthetic_v3_capped_plan",
    "cap_same_column_routing_shape",
    "make_query_row_supervised_synthetic_v3_capped_episode",
    "validate_query_row_supervised_synthetic_v3_capped",
]
