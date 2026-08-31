from __future__ import annotations

import torch

from tabu_lab.contracts import FeatureKind, OriginState, origin_mask
from tabu_lab.experiments.query_row_supervised_synthetic_v3 import (
    BROAD_MAX_CELLS,
    BROAD_MAX_FEATURES,
    BROAD_MAX_ROWS,
    DISCOSCM_FAMILY,
    STRUCTURED_SCM_FAMILY,
    TABUR_V3_MODEL_MAX_FEATURES,
    WORLD_FAMILIES,
    build_query_row_supervised_synthetic_v3_plan,
    make_query_row_supervised_synthetic_v3_episode,
    substitute_query_truth,
    validate_query_row_supervised_synthetic_v3,
)


def test_v3_plan_keeps_existing_families_and_adds_discoscm() -> None:
    train = build_query_row_supervised_synthetic_v3_plan(
        root_seed=1729, worlds=len(WORLD_FAMILIES), partition="train"
    )
    replay = build_query_row_supervised_synthetic_v3_plan(
        root_seed=1729, worlds=len(WORLD_FAMILIES), partition="train"
    )
    assert train == replay
    assert {item["family"] for item in train} == set(WORLD_FAMILIES)
    assert DISCOSCM_FAMILY in WORLD_FAMILIES
    disco_spec = next(item for item in train if item["family"] == DISCOSCM_FAMILY)
    episode = make_query_row_supervised_synthetic_v3_episode(root_seed=1729, **disco_spec)
    assert episode.family == DISCOSCM_FAMILY


def test_v3_broad_scale_prior_is_continuous_bounded_and_family_balanced() -> None:
    plan = build_query_row_supervised_synthetic_v3_plan(
        root_seed=1729, worlds=100, partition="train"
    )
    widths = {item["width"] for item in plan}
    contexts = {item["context_rows"] for item in plan}
    family_counts = {
        family: sum(item["family"] == family for item in plan)
        for family in WORLD_FAMILIES
    }
    assert max(widths) > 32
    assert any(width not in {6, 8, 9, 11, 17, 21, 32} for width in widths)
    assert any(context not in {8, 16, 32, 64, 128, 256, 512} for context in contexts)
    assert max(family_counts.values()) - min(family_counts.values()) <= 1
    assert all(4 <= item["width"] <= BROAD_MAX_FEATURES for item in plan)
    assert all(16 <= item["rows"] <= BROAD_MAX_ROWS for item in plan)
    assert all(1 <= item["context_rows"] < item["rows"] for item in plan)
    assert all((item["width"] + 1) * item["rows"] <= BROAD_MAX_CELLS for item in plan)
    scm_items = [item for item in plan if item["family"] == STRUCTURED_SCM_FAMILY]
    assert all("scm_missingness_family" in item for item in scm_items)
    assert all(0.01 <= item["scm_missingness_rate"] <= 0.35 for item in scm_items)
    assert {item["scm_missingness_family"] for item in scm_items} == {
        "mcar",
        "mar",
        "mnar",
        "block",
        "censoring",
    }


def test_discoscm_episode_is_typed_masked_and_deterministic() -> None:
    episode = make_query_row_supervised_synthetic_v3_episode(
        root_seed=2718,
        world_id="discoscm-explicit",
        family=DISCOSCM_FAMILY,
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
    )
    replay = make_query_row_supervised_synthetic_v3_episode(
        root_seed=2718,
        world_id="discoscm-explicit",
        family=DISCOSCM_FAMILY,
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
    )
    assert episode.evidence.evidence_hash == replay.evidence.evidence_hash
    assert any(spec.kind is not FeatureKind.NUMERIC for spec in episode.evidence.feature_specs[:-1])
    assert bool((episode.evidence.forward_values[episode.sidecar.target_mask] == 0).all())
    assert not any(
        key in episode.evidence.metadata for key in ("units", "tokens", "parents", "dag")
    )


def test_v3_truth_substitution_preserves_evidence() -> None:
    episode = make_query_row_supervised_synthetic_v3_episode(
        root_seed=31415,
        world_id="discoscm-truth-boundary",
        family=DISCOSCM_FAMILY,
        width=8,
        noise_level="low",
        context_rows=8,
        rows=16,
    )
    substituted = substitute_query_truth(episode, value=99.0)
    assert episode.evidence.evidence_hash == substituted.evidence.evidence_hash
    assert torch.equal(episode.evidence.forward_values, substituted.evidence.forward_values)
    assert not torch.equal(episode.sidecar.target_values, substituted.sidecar.target_values)


def test_structured_scm_generates_observed_columns_from_a_dag_without_units() -> None:
    episode = make_query_row_supervised_synthetic_v3_episode(
        root_seed=57721,
        world_id="structured-scm-explicit",
        family=STRUCTURED_SCM_FAMILY,
        predictor_regime="gaussian",
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
    )
    replay = make_query_row_supervised_synthetic_v3_episode(
        root_seed=57721,
        world_id="structured-scm-explicit",
        family=STRUCTURED_SCM_FAMILY,
        predictor_regime="gaussian",
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
    )
    metadata = episode.evidence.metadata
    assert episode.evidence.evidence_hash == replay.evidence.evidence_hash
    assert metadata["scm_scope"] == "observed_columns"
    assert metadata["latent_unit_representation"] == "absent"
    assert metadata["non_root_predictor_count"] > 0
    assert metadata["response_parent_count"] > 0
    assert metadata["edge_count"] >= metadata["response_parent_count"]
    natural_missing = origin_mask(
        episode.evidence.origin_states, OriginState.NATURAL_MISSING
    )
    assert bool(natural_missing.any())
    assert bool((episode.evidence.forward_values[natural_missing] == 0).all())
    assert metadata["missingness_component_id"] == "tabur.scm-missingness.v1"
    assert metadata["raw_missing_representation"] == "nan_before_evidence_compilation"
    assert bool((episode.evidence.forward_values[episode.sidecar.target_mask] == 0).all())


def test_v3_generator_exits_pass_on_cpu() -> None:
    result = validate_query_row_supervised_synthetic_v3(root_seed=1618, worlds=16)
    assert result["status"] == "passed"
    assert all(result["exits"].values())


def test_discoscm_mixed_type_episode_runs_public_forward_and_backward() -> None:
    from tabu_lab.models import build_model
    from tabu_lab.models.types import ReferenceConfig

    episode = make_query_row_supervised_synthetic_v3_episode(
        root_seed=2718,
        world_id="discoscm-public-forward",
        family=DISCOSCM_FAMILY,
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
    )
    model = build_model(
        "tabu.query.row",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=4,
            max_features=TABUR_V3_MODEL_MAX_FEATURES,
        ),
        profile="supervised.label_broadcast.v1",
        row_token_count=4,
    )
    prediction = model(episode.evidence)
    raw = prediction["numeric_raw_prediction"]
    support = prediction["numeric_support_available"].to(torch.bool)
    truth = episode.sidecar.target_values.unsqueeze(0)
    target = episode.sidecar.target_mask.unsqueeze(0)
    scored = target & support
    loss = torch.where(scored, (raw - truth).square(), torch.zeros_like(raw)).sum() / scored.sum()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_structured_scm_natural_missing_runs_public_forward_and_backward() -> None:
    from tabu_lab.models import build_model
    from tabu_lab.models.types import ReferenceConfig

    episode = make_query_row_supervised_synthetic_v3_episode(
        root_seed=57721,
        world_id="structured-scm-missing-public-forward",
        family=STRUCTURED_SCM_FAMILY,
        predictor_regime="gaussian",
        width=17,
        noise_level="medium",
        context_rows=16,
        rows=32,
        scm_missingness_family="mnar",
        scm_missingness_rate=0.2,
    )
    model = build_model(
        "tabu.query.row",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=4,
            max_features=256,
        ),
        profile="supervised.label_broadcast.v1",
        row_token_count=4,
    )
    prediction = model(episode.evidence)
    raw = prediction["numeric_raw_prediction"]
    support = prediction["numeric_support_available"].to(torch.bool)
    truth = episode.sidecar.target_values.unsqueeze(0)
    target = episode.sidecar.target_mask.unsqueeze(0)
    scored = target & support
    loss = torch.where(scored, (raw - truth).square(), torch.zeros_like(raw)).sum() / scored.sum()
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_v3_model_capacity_covers_the_broad_prior_plus_response() -> None:
    assert TABUR_V3_MODEL_MAX_FEATURES == 1024
    assert BROAD_MAX_FEATURES + 1 <= TABUR_V3_MODEL_MAX_FEATURES
