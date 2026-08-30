from __future__ import annotations

import math

import torch

from tabu_lab.contracts import FeatureKind, FeatureRole, ForwardRole, OriginState, origin_code
from tabu_lab.experiments.tabubase_expanded_synthetic import (
    EXPANDED_SYNTHETIC_GENERATOR_VERSION,
    HELDOUT_FAMILIES,
    LONG_CONTEXT_CANDIDATE_ROWS,
    LONG_CONTEXT_ROWS_SCHEDULE,
    MISSINGNESS_REGIMES,
    MODEL_CONTRACT,
    PROFILE_ID,
    RESPONSE_MODALITIES,
    SCHEMA_PROFILES,
    TRAIN_FAMILIES,
    WIDTHS,
    audit_expanded_synthetic_generator,
    audit_expanded_training_episode_universe,
    build_expanded_synthetic_episode,
    evaluate_selected_world_ridge_reference_gate,
    expanded_eligible_context_rows,
    expanded_synthetic_coverage,
    expanded_training_context_rows,
    sample_expanded_world_manifest,
)


def test_expanded_world_manifest_replays_and_partitions_before_compilation() -> None:
    first = sample_expanded_world_manifest(root_seed=1729, world_index=37)
    replay = sample_expanded_world_manifest(root_seed=1729, world_index=37)
    validation = sample_expanded_world_manifest(
        root_seed=1729,
        world_index=37,
        partition="validation",
    )
    heldout = sample_expanded_world_manifest(
        root_seed=1729,
        world_index=37,
        partition="heldout_family",
    )
    assert first == replay
    assert first.manifest_hash == replay.manifest_hash
    assert first.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION
    assert first.model_contract == MODEL_CONTRACT == "tabu.cell.base@0.2.0"
    assert first.profile_id == PROFILE_ID == "supervised.label_broadcast.v1"
    assert first.world_id != validation.world_id
    assert first.manifest_hash != validation.manifest_hash
    assert first.family in TRAIN_FAMILIES
    assert heldout.family in HELDOUT_FAMILIES
    assert not set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES)


def test_expanded_episode_replays_and_physically_isolates_query_response() -> None:
    episode, truth, metadata = build_expanded_synthetic_episode(
        root_seed=2718,
        world_index=73,
        context_rows=12,
        query_rows=9,
    )
    replay, replay_truth, replay_metadata = build_expanded_synthetic_episode(
        root_seed=2718,
        world_index=73,
        context_rows=12,
        query_rows=9,
    )
    assert episode.evidence_hash == replay.evidence_hash
    assert truth.truth_hash == replay_truth.truth_hash
    assert metadata == replay_metadata
    assert episode.feature_specs[-1].role is FeatureRole.RESPONSE
    assert not episode.target_mask[:, :-1].any()
    assert episode.target_mask[12:, -1].all()
    assert torch.count_nonzero(episode.forward_values[episode.target_mask]) == 0
    assert truth.target_count == 9
    assert torch.count_nonzero(truth.target_values[~truth.target_mask]) == 0
    assert metadata["world_manifest_hash"]
    assert metadata["missingness_uses_response"] is False


def test_expanded_episode_supports_zero_context_and_source_scoped_codebooks() -> None:
    episode, truth, metadata = build_expanded_synthetic_episode(
        root_seed=31415,
        world_index=48,  # mixed schema
        context_rows=0,
        query_rows=11,
    )
    assert episode.forward_values.shape == (11, metadata["predictor_width"] + 1)
    assert episode.target_mask[:, -1].all()
    assert truth.target_count == 11
    assert torch.count_nonzero(episode.forward_values[:, -1]) == 0
    nonnumeric = [
        spec for spec in episode.feature_specs if spec.kind is not FeatureKind.NUMERIC
    ]
    assert nonnumeric
    assert all(spec.codebook_id for spec in nonnumeric)
    assert all(metadata["world_id"] not in str(spec.codebook_id) for spec in nonnumeric)
    assert all(len(spec.domain) <= 100 for spec in nonnumeric)

    other, _, _ = build_expanded_synthetic_episode(
        root_seed=31415,
        world_index=49,
        context_rows=2,
        query_rows=11,
    )
    ids = {spec.name: spec.codebook_id for spec in episode.feature_specs if spec.codebook_id}
    other_ids = {spec.name: spec.codebook_id for spec in other.feature_specs if spec.codebook_id}
    for name in ids.keys() & other_ids.keys():
        assert ids[name] == other_ids[name]


def test_expanded_predictor_permutation_and_context_standardization_are_deterministic() -> None:
    manifest = sample_expanded_world_manifest(root_seed=1729, world_index=0)
    episode, _, metadata = build_expanded_synthetic_episode(
        root_seed=1729,
        world_index=0,
        context_rows=32,
        query_rows=8,
    )
    assert manifest.schema_profile == "numeric_only"
    assert manifest.response_modality == "numeric"
    assert manifest.missingness_regime == "none"
    assert tuple(metadata["predictor_permutation"]) == manifest.predictor_permutation
    source_order = tuple(
        int(spec.name.rsplit("_", 1)[-1]) for spec in episode.feature_specs[:-1]
    )
    assert source_order == manifest.predictor_permutation
    context = episode.forward_values[:32]
    torch.testing.assert_close(
        context.mean(dim=0), torch.zeros(context.shape[1]), atol=2e-5, rtol=0
    )
    torch.testing.assert_close(
        context.std(dim=0, unbiased=False),
        torch.ones(context.shape[1]),
        atol=2e-4,
        rtol=0,
    )


def test_expanded_mcar_and_mar_masks_are_legal_and_response_blind() -> None:
    for world_index, expected_regime in ((64, "mcar"), (128, "mar")):
        episode, _, metadata = build_expanded_synthetic_episode(
            root_seed=1729,
            world_index=world_index,
            context_rows=48,
            query_rows=32,
        )
        assert metadata["missingness_regime"] == expected_regime
        missing = episode.origin_states == origin_code(OriginState.NATURAL_MISSING)
        source = (episode.forward_roles & int(ForwardRole.SOURCE)) != 0
        assert missing.any()
        assert not missing[:, -1].any()
        assert not source[missing].any()
        assert torch.count_nonzero(episode.forward_values[missing]) == 0
        assert metadata["missingness_mask_inputs"] == (
            "predictors_only_before_response_generation"
        )
        assert metadata["missingness_uses_response"] is False


def test_expanded_coverage_contains_all_frozen_stage_a_strata() -> None:
    coverage = expanded_synthetic_coverage(root_seed=1729, world_count=192)
    assert coverage["passed"]
    assert set(coverage["family_counts"]) == set(TRAIN_FAMILIES) | set(HELDOUT_FAMILIES)
    assert set(coverage["response_modality_counts"]) == set(RESPONSE_MODALITIES)
    assert {int(width) for width in coverage["width_counts"]} == set(WIDTHS)
    assert set(coverage["schema_profile_counts"]) == set(SCHEMA_PROFILES)
    assert set(coverage["missingness_counts"]) == set(MISSINGNESS_REGIMES)
    assert coverage["maximum_declared_cardinality"] <= 100
    assert coverage["checks"]["complete_heldout_family"]
    assert coverage["checks"]["family_partitions_disjoint"]


def test_expanded_generator_audit_passes_only_gd0_through_gd2() -> None:
    audit = audit_expanded_synthetic_generator(
        root_seed=1729,
        coverage_worlds=192,
        context_rows=16,
        query_rows=8,
    )
    assert audit["passed"]
    assert set(audit["gates"]) == {"G-D0", "G-D1", "G-D2"}
    assert all(gate["passed"] for gate in audit["gates"].values())
    assert audit["not_evaluated_gates"] == ("G-D3", "G-D4", "G-D5")
    assert audit["not_a_model_or_training_claim"] is True


def test_expanded_training_universe_audit_compiles_and_covers_eligible_K() -> None:
    audit = audit_expanded_training_episode_universe(
        root_seed=1729,
        world_count=96,
        query_rows=4,
    )
    assert audit["passed"]
    assert audit["failure_count"] == 0
    assert audit["checks"]["every_support_realizable_modality_K_pair_present"]


def test_selected_world_closed_form_ridge_gd3_is_finite_and_large_not_worse() -> None:
    result = evaluate_selected_world_ridge_reference_gate(
        root_seed=1729,
        selected_worlds=2,
        small_context=8,
        large_context=48,
        query_rows=32,
    )
    assert result["gate"] == "G-D3"
    assert result["included_in_stage_a_audit"] is False
    assert result["finite"]
    assert result["passed"]
    assert math.isfinite(result["rmse_gain"])
    assert result["large_rmse"] <= result["small_rmse"]
    assert all(item["rmse_gain"] > 0.0 for item in result["per_world"])


def test_expanded_context_schedule_is_support_realizable_and_breaks_parity_lock() -> None:
    schedule = (2, 4, 8, 16, 32, 64)
    observed = {
        response_slot: {
            expanded_training_context_rows(
                world_index=world_index,
                context_rows_schedule=schedule,
            )
            for world_index in range(2_048)
            if world_index % len(RESPONSE_MODALITIES) == response_slot
        }
        for response_slot in range(len(RESPONSE_MODALITIES))
    }
    assert observed[0] == observed[1] == set(schedule)
    assert observed[2] == observed[3] == set(schedule[1:])
    for world_index in range(32):
        selected = expanded_training_context_rows(
            world_index=world_index,
            context_rows_schedule=schedule,
        )
        assert selected in expanded_eligible_context_rows(
            world_index=world_index,
            context_rows_schedule=schedule,
        )


def test_expanded_classification_context_supports_every_query_label_and_is_nested() -> None:
    schedule = (2, 4, 8, 16, 32, 64)
    for world_index in (1, 2, 3):
        query_banks: list[tuple[str, ...]] = []
        for context_rows in expanded_eligible_context_rows(
            world_index=world_index,
            context_rows_schedule=schedule,
        ):
            episode, truth, metadata = build_expanded_synthetic_episode(
                root_seed=1729,
                world_index=world_index,
                context_rows=context_rows,
                query_rows=16,
            )
            context_labels = {
                int(value) for value in episode.forward_values[:context_rows, -1].tolist()
            }
            query_labels = {
                int(value) for value in truth.target_values[context_rows:, -1].tolist()
            }
            assert query_labels <= context_labels
            assert metadata["query_response_used_for_context_selection"] is False
            query_banks.append(episode.row_ids[context_rows:])
        assert len(set(query_banks)) == 1


def test_expanded_four_class_world_fails_closed_at_k2() -> None:
    for world_index in (2, 3):
        try:
            build_expanded_synthetic_episode(
                root_seed=1729,
                world_index=world_index,
                context_rows=2,
                query_rows=8,
            )
        except ValueError as exc:
            assert "response support" in str(exc)
        else:
            raise AssertionError("four-class world accepted structurally unrealizable K=2")


def test_long_context_bank_is_nested_and_keeps_one_query_bank() -> None:
    episodes = []
    for context_rows in (128, 256, 512):
        episode, truth, metadata = build_expanded_synthetic_episode(
            root_seed=1729,
            world_index=3,
            context_rows=context_rows,
            query_rows=16,
            context_candidate_rows=LONG_CONTEXT_CANDIDATE_ROWS,
        )
        assert episode.forward_values.shape[0] == context_rows + 16
        assert truth.target_count == 16
        assert metadata["frozen_context_bank_rows"] == LONG_CONTEXT_CANDIDATE_ROWS
        assert metadata["query_bank_start"] >= LONG_CONTEXT_CANDIDATE_ROWS
        assert episode.episode_id.endswith("-cb512")
        episodes.append((context_rows, episode))

    assert episodes[0][1].row_ids[:128] == episodes[1][1].row_ids[:128]
    assert episodes[1][1].row_ids[:256] == episodes[2][1].row_ids[:256]
    assert len({episode.row_ids[context_rows:] for context_rows, episode in episodes}) == 1


def test_long_context_training_universe_covers_every_eligible_modality_k_pair() -> None:
    audit = audit_expanded_training_episode_universe(
        root_seed=1729,
        world_count=96,
        context_rows_schedule=LONG_CONTEXT_ROWS_SCHEDULE,
        query_rows=4,
        context_candidate_rows=LONG_CONTEXT_CANDIDATE_ROWS,
    )
    assert audit["passed"]
    assert audit["failure_count"] == 0
    assert audit["frozen_context_bank_rows"] == 512
    assert audit["checks"]["every_support_realizable_modality_K_pair_present"]
