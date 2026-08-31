from __future__ import annotations

from tabu_lab.experiments.query_row_supervised_synthetic_v3 import (
    GENERATOR_ID as PARENT_GENERATOR_ID,
)
from tabu_lab.experiments.query_row_supervised_synthetic_v3_capped import (
    DEFAULT_MAX_ROUTING_PAIRS,
    GENERATOR_ID,
    build_query_row_supervised_synthetic_v3_capped_plan,
    cap_same_column_routing_shape,
    make_query_row_supervised_synthetic_v3_capped_episode,
    validate_query_row_supervised_synthetic_v3_capped,
)


def test_quadratic_cap_preserves_context_fraction_and_admits_minimum_shape() -> None:
    shape = cap_same_column_routing_shape(
        width=4,
        rows=2048,
        context_rows=1536,
    )

    assert shape.was_capped
    assert shape.rows < shape.requested_rows
    assert shape.routing_pairs <= DEFAULT_MAX_ROUTING_PAIRS
    assert 1 <= shape.context_rows < shape.rows
    assert abs(shape.context_rows / shape.rows - 0.75) < 0.01


def test_compute_capped_plan_retains_v3_worlds_under_quadratic_budget() -> None:
    plan = build_query_row_supervised_synthetic_v3_capped_plan(
        root_seed=1729,
        worlds=256,
        partition="train",
    )

    assert any(item["routing_shape_was_capped"] for item in plan)
    assert all(
        (int(item["width"]) + 1) * int(item["rows"]) ** 2
        == int(item["routing_pairs"])
        <= DEFAULT_MAX_ROUTING_PAIRS
        for item in plan
    )
    assert all("family" in item for item in plan)


def test_compute_capped_episode_has_distinct_generator_and_recipe_identity() -> None:
    episode = make_query_row_supervised_synthetic_v3_capped_episode(
        root_seed=2718,
        world_id="explicit-large-row-world",
        width=4,
        rows=2048,
        context_rows=1536,
        family="sparse_additive",
        predictor_regime="gaussian",
        noise_level="medium",
    )
    replay = make_query_row_supervised_synthetic_v3_capped_episode(
        root_seed=2718,
        world_id="explicit-large-row-world",
        width=4,
        rows=2048,
        context_rows=1536,
        family="sparse_additive",
        predictor_regime="gaussian",
        noise_level="medium",
    )

    metadata = episode.evidence.metadata
    assert episode.generator_id == GENERATOR_ID
    assert metadata["generator_id"] == GENERATOR_ID
    assert metadata["parent_generator_id"] == PARENT_GENERATOR_ID
    assert metadata["routing_shape_was_capped"] is True
    assert metadata["routing_pairs"] <= DEFAULT_MAX_ROUTING_PAIRS
    assert episode.evidence.evidence_hash == replay.evidence.evidence_hash
    assert episode.sidecar.truth_hash == replay.sidecar.truth_hash
    assert episode.evidence.episode_id == episode.sidecar.episode_id


def test_compute_capped_generator_validation_passes() -> None:
    result = validate_query_row_supervised_synthetic_v3_capped(
        root_seed=1618,
        worlds=64,
    )

    assert result["status"] == "passed"
    assert all(result["exits"].values())
