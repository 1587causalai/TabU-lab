from __future__ import annotations

from pathlib import Path

from tabu_lab.evolution import EvolutionRepository, ImpactDisposition, impact_report

ROOT = Path(__file__).resolve().parents[2]
BASE = "tabu.pretraining.query-base@1.0.0"
PROJECTABLE_BASE = (
    "tabu.pretraining.query-base-generator-v2-projectable@1.0.0-exercise"
)


def _actions(
    target: str,
    *,
    source: str = BASE,
) -> dict[str, ImpactDisposition]:
    repository = EvolutionRepository.load(ROOT)
    report = impact_report(
        repository,
        repository.resolve(source),
        repository.resolve(target),
    )
    return {action.object_kind: action.disposition for action in report.actions}


def test_math_contract_change_retrains_only_the_affected_model_lane() -> None:
    actions = _actions("tabu.pretraining.query-base-math-exercise@1.1.0-exercise")

    assert actions["slot:model_contract"] is ImpactDisposition.RETRAIN
    assert actions["slot:component_graph"] is ImpactDisposition.RETRAIN
    assert actions["slot:world_mixture"] is ImpactDisposition.UNCHANGED
    assert actions["initialization"] is ImpactDisposition.BLOCKED
    assert actions["training_run"] is ImpactDisposition.RETRAIN
    assert actions["predictions"] is ImpactDisposition.RERUN_INFERENCE


def test_generator_change_preserves_model_and_offers_only_explicit_warm_start() -> None:
    actions = _actions(
        "tabu.pretraining.query-base-generator-v3@1.1.0-exercise",
        source=PROJECTABLE_BASE,
    )

    assert actions["slot:model_contract"] is ImpactDisposition.UNCHANGED
    assert actions["slot:component_graph"] is ImpactDisposition.UNCHANGED
    assert actions["slot:world_mixture"] is ImpactDisposition.RETRAIN
    assert actions["initialization"] is ImpactDisposition.WARM_START_AVAILABLE
    assert actions["training_run"] is ImpactDisposition.RETRAIN


def test_component_replacement_does_not_invalidate_data_or_evaluation_specs() -> None:
    actions = _actions(
        "tabu.pretraining.query-base-component-adapter@1.1.0-exercise"
    )

    assert actions["slot:component_graph"] is ImpactDisposition.RETRAIN
    assert actions["slot:world_mixture"] is ImpactDisposition.UNCHANGED
    assert actions["slot:evaluation_protocol"] is ImpactDisposition.UNCHANGED
    assert actions["initialization"] is ImpactDisposition.BLOCKED


def test_evaluation_change_rescores_without_retraining_or_reinference() -> None:
    actions = _actions("tabu.pretraining.query-base-eval-v2@1.1.0-exercise")

    assert actions["checkpoint"] is ImpactDisposition.REUSE_EXACT
    assert actions["predictions"] is ImpactDisposition.REUSE_EXACT
    assert actions["evaluation"] is ImpactDisposition.RESCORE
    assert "training_run" not in actions


def test_identical_snapshot_reuses_every_artifact_exactly() -> None:
    repository = EvolutionRepository.load(ROOT)
    resolved = repository.resolve(BASE)
    report = impact_report(repository, resolved, resolved)
    actions = {action.object_kind: action.disposition for action in report.actions}

    assert actions["checkpoint"] is ImpactDisposition.REUSE_EXACT
    assert actions["predictions"] is ImpactDisposition.REUSE_EXACT
    assert actions["evaluation"] is ImpactDisposition.REUSE_EXACT


def test_gpu_pilot_recipe_change_retrains_without_changing_model_or_data() -> None:
    actions = _actions("tabu.pretraining.query-base@1.1.0")

    assert actions["slot:model_contract"] is ImpactDisposition.UNCHANGED
    assert actions["slot:component_graph"] is ImpactDisposition.UNCHANGED
    assert actions["slot:world_mixture"] is ImpactDisposition.UNCHANGED
    assert actions["slot:training_recipe"] is ImpactDisposition.RETRAIN
    assert actions["training_run"] is ImpactDisposition.RETRAIN


def test_v3_scale_and_generator_change_requires_new_runs_but_allows_warm_start() -> None:
    for source, target in (
        (
            "tabu.pretraining.query-base@1.1.0",
            "tabu.pretraining.query-base@1.2.0",
        ),
        (
            "tabu.pretraining.query-row@1.1.0",
            "tabu.pretraining.query-row@1.2.0",
        ),
    ):
        actions = _actions(target, source=source)
        assert actions["slot:model_contract"] is ImpactDisposition.UNCHANGED
        assert actions["slot:component_graph"] is ImpactDisposition.RETRAIN
        assert actions["slot:world_mixture"] is ImpactDisposition.RETRAIN
        assert actions["slot:training_recipe"] is ImpactDisposition.UNCHANGED
        assert actions["initialization"] is ImpactDisposition.WARM_START_AVAILABLE
        assert actions["training_run"] is ImpactDisposition.RETRAIN


def test_v3_compute_cap_and_recovery_recipe_restart_without_model_change() -> None:
    for source, target in (
        (
            "tabu.pretraining.query-base@1.2.0",
            "tabu.pretraining.query-base@1.3.0",
        ),
        (
            "tabu.pretraining.query-row@1.2.0",
            "tabu.pretraining.query-row@1.3.0",
        ),
    ):
        actions = _actions(target, source=source)
        assert actions["slot:model_contract"] is ImpactDisposition.UNCHANGED
        assert actions["slot:component_graph"] is ImpactDisposition.UNCHANGED
        assert actions["slot:world_mixture"] is ImpactDisposition.RETRAIN
        assert actions["slot:training_recipe"] is ImpactDisposition.RETRAIN
        assert actions["training_run"] is ImpactDisposition.RETRAIN


def test_v3_long_continuation_uses_explicit_identity_warm_start() -> None:
    for source, target in (
        (
            "tabu.pretraining.query-base@1.3.0",
            "tabu.pretraining.query-base@1.4.0",
        ),
        (
            "tabu.pretraining.query-row@1.3.0",
            "tabu.pretraining.query-row@1.4.0",
        ),
    ):
        actions = _actions(target, source=source)
        assert actions["slot:model_contract"] is ImpactDisposition.UNCHANGED
        assert actions["slot:component_graph"] is ImpactDisposition.UNCHANGED
        assert actions["slot:world_mixture"] is ImpactDisposition.UNCHANGED
        assert actions["slot:training_recipe"] is ImpactDisposition.RETRAIN
        assert actions["initialization"] is ImpactDisposition.WARM_START_AVAILABLE
        assert actions["training_run"] is ImpactDisposition.RETRAIN
