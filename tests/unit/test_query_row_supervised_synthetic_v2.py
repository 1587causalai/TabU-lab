from __future__ import annotations

import torch

from tabu_lab.experiments import (
    WIDTH_BUCKETS,
    WORLD_FAMILIES,
    build_query_row_supervised_synthetic_v2_plan,
    make_query_row_supervised_synthetic_v2_episode,
    substitute_query_truth,
    validate_query_row_supervised_synthetic_v2,
)


def test_v2_plan_is_deterministic_and_train_validation_ids_are_disjoint() -> None:
    train = build_query_row_supervised_synthetic_v2_plan(
        root_seed=1729, worlds=64, partition="train"
    )
    replay = build_query_row_supervised_synthetic_v2_plan(
        root_seed=1729, worlds=64, partition="train"
    )
    validation = build_query_row_supervised_synthetic_v2_plan(
        root_seed=1729, worlds=64, partition="validation"
    )
    assert train == replay
    assert {item["world_id"] for item in train}.isdisjoint(
        {item["world_id"] for item in validation}
    )
    assert {item["family"] for item in train} == set(WORLD_FAMILIES)
    assert {item["width"] for item in train} == set(WIDTH_BUCKETS)


def test_v2_truth_substitution_preserves_model_evidence() -> None:
    episode = make_query_row_supervised_synthetic_v2_episode(
        root_seed=2718,
        world_id="train-world-explicit",
        width=32,
        family="polynomial_interaction",
        predictor_regime="heavy_tailed",
        noise_level="high",
        context_rows=16,
    )
    substituted = substitute_query_truth(episode, value=99.0)
    assert episode.evidence.evidence_hash == substituted.evidence.evidence_hash
    assert torch.equal(episode.evidence.forward_values, substituted.evidence.forward_values)
    assert not torch.equal(episode.sidecar.target_values, substituted.sidecar.target_values)


def test_v2_generator_exits_pass_on_cpu() -> None:
    result = validate_query_row_supervised_synthetic_v2(root_seed=31415, worlds=32)
    assert result["status"] == "passed"
    assert result["contract_version"] == "0.2.0"
    assert result["row_readout_mode"] == "anchored"
    assert result["row_readout_identity"]["mode"] == "anchored"
    assert len(result["variant_hash"]) == 64
    assert all(result["exits"].values())
