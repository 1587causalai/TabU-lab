from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml

import tabu_lab.experiments.runner as runner_module
from tabu_lab.catalog import (
    CatalogObjectKind,
    CatalogSourceRevision,
    EvidencePointer,
    ExperimentRecord,
    ExperimentStatus,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    StatusEvent,
    build_catalog,
)
from tabu_lab.contracts import (
    EvaluationBundle,
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    TruthSidecar,
    canonical_hash,
    canonical_json,
    origin_code,
)
from tabu_lab.evidence import ReceiptStatus, read_receipt
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationSummary,
)
from tabu_lab.evidence.source_identity import SourceIdentity, distribution_source_identity
from tabu_lab.experiments import (
    AugmentedReadoutGeometry,
    FitDevice,
    FitEvaluationBundle,
    FitEvidenceMode,
    FitExperimentSpec,
    FitFamilyMetrics,
    FitMetricKind,
    FitStage,
    FitTargetFamily,
    GraphUnitReceiverPlan,
    LabelAddressPlan,
    RecommendationAddressPlan,
    ReferenceBackendConfig,
)
from tabu_lab.experiments.fixture_registry import build_registered_f0_fixture
from tabu_lab.experiments.fixtures import build_f0_fixture
from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.experiments.runner import (
    ExperimentAggregateArtifacts,
    FitExperimentError,
    _aggregate_verdict,
    _assert_formal_output_root_safe,
    _authorization_safe_command,
    _build_model,
    _forward_in_eval,
    _gate_reasons,
    _mechanism_gradient_probe,
    _seed_verdict,
    _typed_family_metrics,
    _write_experiment_aggregate,
    compiler_binding_manifest,
    run_fit_experiment,
    source_tree_hash,
    source_tree_manifest,
    validate_f0_binding,
)
from tabu_lab.registry import get_model_spec
from tabu_lab.training import Objective, Trainer


def _spec(contract_id: str = "tabuf") -> FitExperimentSpec:
    return build_f0_preregistration(contract_id, device=FitDevice.CPU)


def test_unit_row_v2_preregisters_router_local_rms_plan() -> None:
    spec = build_f0_preregistration(
        "tabu.unit_row",
        device=FitDevice.CPU,
        fixture_version="v2",
    )

    assert spec.experiment_id == "F0-009-tabu-unit-row-identifiable-v2"
    assert spec.semantic.reference.geometry_normalization == "rms_unit"
    assert spec.semantic.reference.routing_bandwidth == 2.5
    assert spec.dataset.adapter.adapter_version == "2.0.0"


def test_canonical_f0_builder_declares_revision_but_custom_diagnostic_does_not() -> None:
    canonical = build_f0_preregistration(
        "tabu.unit_row",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    diagnostic = build_f0_preregistration(
        "tabu.unit_row",
        device=FitDevice.CPU,
        fixture_version="v2",
        experiment_id="F0-009-tabu-unit-row-identifiable-v2-diagnostic",
    )

    assert canonical.supersedes_experiment_ids == ("F0-002-tabu-unit-row-v1",)
    assert canonical.revision_rationale is not None
    assert diagnostic.supersedes_experiment_ids == ()
    assert diagnostic.revision_rationale is None


def test_graph_v2_versions_the_row_unit_receiver_repair() -> None:
    legacy = build_f0_preregistration(
        "tabu4graph",
        device=FitDevice.CPU,
        fixture_version="v1",
    )
    repaired = build_f0_preregistration(
        "tabu4graph",
        device=FitDevice.CPU,
        fixture_version="v2",
    )

    assert legacy.experiment_id == "F0-006-tabu4graph-v1"
    assert repaired.experiment_id == "F0-011-tabu4graph-row-unit-v2"
    assert legacy.semantic.graph_unit_receiver_plan is GraphUnitReceiverPlan.LEGACY_GRAPH_UNITS_ONLY
    assert (
        repaired.semantic.graph_unit_receiver_plan is GraphUnitReceiverPlan.SAME_ROW_VISIBLE_CELLS
    )
    assert repaired.dataset.dataset_hash == legacy.dataset.dataset_hash
    assert repaired.episode_schedule == legacy.episode_schedule


def test_graph_v2_trace_declares_repaired_receiver_and_typed_terminal() -> None:
    spec = build_f0_preregistration(
        "tabu4graph",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture("tabu4graph", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))

    assert prediction.trace.metadata["graph_unit_receiver_plan"] == ("same_row_visible_cells")
    readout_event = next(event for event in prediction.trace.events if event.name == "readout")
    assert readout_event.metadata["terminal"] == "typed_nw"
    assert readout_event.metadata["operation_trace"] == (
        "global_feature_prototype_for_readout",
        "typed_nw",
    )


def test_rec_v2_preregisters_axis_address_semantics() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
    )

    assert spec.experiment_id == "F0-014-tabu4rec-axis-address-v2"
    assert (
        spec.semantic.recommendation_address_plan
        is RecommendationAddressPlan.AXIS_ADDRESS_BOOTSTRAP_V1
    )
    assert spec.semantic.rec_axis_summary_dim == 2
    assert spec.semantic.rec_matched_residual_scale == pytest.approx(0.1)


def test_nondeterministic_mps_requires_explicit_diagnostic_identity() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.MPS,
        fixture_version="v2",
        experiment_id="F0-014-tabu4rec-axis-address-v2-mps-diagnostic",
        deterministic_algorithms=False,
        evidence_mode=FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC,
        exact_resume=False,
    )

    assert spec.execution.deterministic_algorithms is False
    assert spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
    assert spec.training.exact_resume is False

    with pytest.raises(ValueError, match="explicit diagnostic"):
        build_f0_preregistration(
            "tabu4rec",
            device=FitDevice.MPS,
            fixture_version="v2",
            deterministic_algorithms=False,
        )
    with pytest.raises(ValueError, match="cannot claim exact resume"):
        build_f0_preregistration(
            "tabu4rec",
            device=FitDevice.MPS,
            fixture_version="v2",
            deterministic_algorithms=False,
            evidence_mode=FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC,
            exact_resume=True,
        )


def test_rec_v2_trace_and_support_keep_both_arms_active() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture("tabu4rec", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))
    targets = fixture.truth.target_mask

    assert prediction.trace.metadata["recommendation_address_plan"] == ("axis_address_bootstrap_v1")
    assert prediction.auxiliaries["coordinates"].shape[-1] == 8
    assert bool(prediction.auxiliaries["rec_user_arm_support_available"][targets].all())
    assert bool(prediction.auxiliaries["rec_item_arm_support_available"][targets].all())
    assert torch.allclose(
        prediction.auxiliaries["rec_arm_weights"][targets],
        torch.full(
            (int(targets.sum()), 2),
            0.5,
            dtype=prediction.auxiliaries["rec_arm_weights"].dtype,
        ),
    )
    axis_event = next(
        event for event in prediction.trace.events if event.name == "recommendation_axis_address"
    )
    assert axis_event.metadata["uses_identifiers"] is False
    assert axis_event.metadata["truth_not_available"] is True


def test_rec_v2_mechanism_probe_requires_support_and_gradient_in_both_arms() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture("tabu4rec", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    source_counts, active_target_counts, gradient_norms, scored_target_count = (
        _mechanism_gradient_probe(
            model,
            fixture.evidence,
            fixture.truth,
            contract_id="tabu4rec",
            device=torch.device("cpu"),
        )
    )

    assert source_counts["rec_user_arm"] > 0
    assert source_counts["rec_item_arm"] > 0
    assert scored_target_count == fixture.truth.target_count
    assert active_target_counts == {
        "rec_user_arm": scored_target_count,
        "rec_item_arm": scored_target_count,
    }
    assert gradient_norms["rec_user_arm"] > 0.0
    assert gradient_norms["rec_item_arm"] > 0.0


def test_rec_mechanism_probe_backpropagates_categorical_arm_nll() -> None:
    raw = torch.tensor(
        (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0, 0.0),
        )
    )
    forward_values = raw.clone()
    forward_values[0, 0] = 0.0
    origin_states = torch.full(
        raw.shape,
        origin_code(OriginState.OBSERVED),
        dtype=torch.uint8,
    )
    origin_states[0, 0] = origin_code(OriginState.ARTIFICIAL_MASK)
    source_role = int(ForwardRole.RECEIVER | ForwardRole.SOURCE)
    target_role = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    forward_roles = torch.full(raw.shape, source_role, dtype=torch.uint8)
    forward_roles[0, 0] = target_role
    feature_specs = tuple(
        FeatureSpec(
            name=f"item-{index}",
            kind=FeatureKind.CATEGORICAL,
            domain=("negative", "positive"),
            codebook_id="tiny-rec-binary-v1",
            role=FeatureRole.RESPONSE,
        )
        for index in range(raw.shape[1])
    )
    evidence = EvidenceEpisode(
        episode_id="tiny-categorical-rec",
        dataset_id="tiny-categorical-rec",
        source_partition="train",
        fit_partition="train",
        row_ids=tuple(f"user-{index}" for index in range(raw.shape[0])),
        feature_names=tuple(spec.name for spec in feature_specs),
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origin_states,
        forward_roles=forward_roles,
    )
    target_mask = torch.zeros_like(raw, dtype=torch.bool)
    target_mask[0, 0] = True
    target_values = torch.zeros_like(raw)
    target_values[0, 0] = raw[0, 0]
    truth = TruthSidecar(
        episode_id=evidence.episode_id,
        recipe_hash=hashlib.sha256(b"tiny-categorical-rec").hexdigest(),
        row_ids=evidence.row_ids,
        feature_names=evidence.feature_names,
        target_values=target_values,
        target_mask=target_mask,
    )
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    source_counts, active_counts, gradient_norms, scored_count = (
        _mechanism_gradient_probe(
            model,
            evidence,
            truth,
            contract_id="tabu4rec",
            device=torch.device("cpu"),
        )
    )

    assert scored_count == 1
    assert source_counts == {"rec_user_arm": 3, "rec_item_arm": 3}
    assert active_counts == {"rec_user_arm": 1, "rec_item_arm": 1}
    assert gradient_norms["rec_user_arm"] > 0.0
    assert gradient_norms["rec_item_arm"] > 0.0


@pytest.mark.parametrize(
    ("contract_id", "experiment_id"),
    (
        ("tabul", "F0-012-tabul-predictor-address-v2"),
        ("tabufl", "F0-013-tabufl-independent-ledgers-v2"),
    ),
)
def test_supervised_v2_preregisters_predictor_only_label_address(
    contract_id: str,
    experiment_id: str,
) -> None:
    spec = build_f0_preregistration(
        contract_id,
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture(contract_id, fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))

    assert spec.experiment_id == experiment_id
    assert spec.semantic.label_address_plan is LabelAddressPlan.PREDICTOR_ONLY_PER_LABEL_V1
    assert spec.semantic.label_columns == (6, 7)
    assert prediction.trace.metadata["label_address_plan"] == ("predictor_only_per_label_v1")
    event = next(
        event for event in prediction.trace.events if event.name == "predictor_only_label_address"
    )
    assert event.metadata["response_tokens_excluded"] is True
    assert event.metadata["truth_not_available"] is True


@pytest.mark.parametrize(
    ("contract_id", "experiment_id"),
    (
        ("tabul", "F0-015-tabul-unit-linked-address-v3"),
        ("tabufl", "F0-016-tabufl-independent-dynamics-v3"),
    ),
)
def test_supervised_v3_uses_unit_linked_independent_label_dynamics(
    contract_id: str,
    experiment_id: str,
) -> None:
    spec = build_f0_preregistration(
        contract_id,
        device=FitDevice.CPU,
        fixture_version="v2",
        supervised_label_address_plan=(LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2),
    )
    fixture = build_f0_fixture(contract_id, fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))
    trainer = Trainer(model, learning_rate=1.0e-2, max_gradient_norm=1.0)
    step = trainer.train_step(fixture.evidence, fixture.truth)

    assert spec.experiment_id == experiment_id
    assert spec.semantic.label_address_plan is LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
    assert prediction.trace.metadata["label_address_plan"] == ("predictor_unit_linked_per_label_v2")
    dynamics_event = next(
        event
        for event in prediction.trace.events
        if event.name == "predictor_unit_address_dynamics"
    )
    readout_event = next(
        event
        for event in prediction.trace.events
        if event.name == "predictor_unit_linked_label_address"
    )
    assert dynamics_event.metadata["shared_query_unit"] is True
    assert dynamics_event.metadata["response_tokens_excluded"] is True
    assert readout_event.metadata["response_tokens_excluded"] is True
    assert step.gradient_norms["dynamics"] > 0.0
    assert step.gradient_norms["readout"] > 0.0


def test_tabufl_v4_preregisters_balanced_joint_fixture_without_architecture_drift() -> None:
    spec = build_f0_preregistration(
        "tabufl",
        device=FitDevice.CPU,
        fixture_version="v4",
    )
    fixture = build_f0_fixture("tabufl", fixture_version="v4")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))

    assert spec.experiment_id == "F0-017-tabufl-balanced-joint-v4"
    assert spec.dataset.adapter.adapter_version == "4.0.0"
    assert spec.semantic.label_address_plan is LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
    assert spec.semantic.label_columns == (6, 7)
    assert int(fixture.target_family_masks["F"].sum()) == 12
    assert int(fixture.target_family_masks["L"].sum()) == 32
    assert prediction.trace.metadata["label_address_plan"] == (
        "predictor_unit_linked_per_label_v2"
    )
    event_names = {event.name for event in prediction.trace.events}
    assert "predictor_unit_address_dynamics" in event_names
    assert "predictor_unit_linked_label_address" in event_names


def test_tabufl_v5_preregisters_frozen_16f_joint_fixture() -> None:
    spec = build_f0_preregistration(
        "tabufl",
        device=FitDevice.CPU,
        fixture_version="v5",
    )
    fixture = build_registered_f0_fixture("tabufl", fixture_version="v5")

    assert spec.experiment_id == "F0-018-tabufl-balanced-16f-v5"
    assert spec.dataset.adapter.adapter_version == "5.0.0"
    assert spec.dataset.source_uri == "pkg://tabu_lab.experiments.fixtures_v5"
    assert spec.episode_schedule.targets_per_episode == 48
    assert int(fixture.target_family_masks["F"].sum()) == 16
    assert int(fixture.target_family_masks["L"].sum()) == 32
    assert spec.semantic.label_address_plan is LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
    assert validate_f0_binding(spec).fixture_hash == fixture.fixture_hash


def test_forward_exposes_finite_truth_free_routing_diagnostics() -> None:
    spec = build_f0_preregistration(
        "tabuf",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture("tabuf", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))

    diagnostics = prediction.trace.metadata["routing_diagnostics"]
    assert diagnostics["available_query_count"] > 0
    assert diagnostics["query_count"] >= diagnostics["available_query_count"]
    assert diagnostics["coordinate_rms"] > 0.0
    assert diagnostics["source_count_minimum"] >= 1.0
    assert diagnostics["entropy_maximum"] > 0.0
    assert diagnostics["effective_support_size_minimum"] >= 1.0
    assert diagnostics["max_weight_maximum"] <= 1.0 + 1.0e-7
    for name in (
        "routing_source_count",
        "routing_entropy",
        "routing_effective_support_size",
        "routing_max_weight",
        "routing_log_weight_span",
    ):
        assert name in prediction.auxiliaries
        assert torch.isfinite(prediction.auxiliaries[name]).all()


def test_train_step_reports_parameter_group_gradient_norms() -> None:
    spec = build_f0_preregistration(
        "tabu.unit_row",
        device=FitDevice.CPU,
        fixture_version="v2",
    )
    fixture = build_f0_fixture("tabu.unit_row", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))
    trainer = Trainer(model, learning_rate=1.0e-2)

    step = trainer.train_step(fixture.evidence, fixture.truth)

    assert set(step.gradient_norms) == {
        "carrier",
        "tokenizer",
        "dynamics",
        "readout",
        "other",
    }
    assert step.gradient_norms["carrier"] > 0.0
    assert step.gradient_norms["dynamics"] > 0.0
    assert step.gradient_norms["readout"] > 0.0
    grouped_norm = sum(value**2 for value in step.gradient_norms.values()) ** 0.5
    assert grouped_norm == pytest.approx(step.gradient_norm, rel=1.0e-6, abs=1.0e-8)


def _mutated_spec(
    spec: FitExperimentSpec,
    mutate: Callable[[dict[str, Any]], None],
) -> FitExperimentSpec:
    payload = spec.model_dump(mode="python")
    mutate(payload)
    return FitExperimentSpec.model_validate(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda payload: payload["dataset"].__setitem__("source_sha256", "f" * 64),
            "generator source hash",
        ),
        (
            lambda payload: (
                payload["dataset"].__setitem__("dataset_hash", "f" * 64),
                payload["split"].__setitem__("dataset_hash", "f" * 64),
            ),
            "dataset hash",
        ),
        (
            lambda payload: payload["episode_schedule"].__setitem__("recipe_hashes", ("f" * 64,)),
            "recipe hash",
        ),
        (
            lambda payload: payload["episode_schedule"].__setitem__(
                "targets_per_episode",
                payload["episode_schedule"]["targets_per_episode"] + 1,
            ),
            "episode schedule",
        ),
    ),
)
def test_f0_binding_rejects_source_dataset_recipe_and_schedule_drift(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    spec = _mutated_spec(_spec(), mutate)

    with pytest.raises(FitExperimentError, match=message):
        validate_f0_binding(spec)


@pytest.mark.parametrize(
    ("contract_id", "member_key"),
    (
        ("tabuf", "row_ids"),
        ("tabu4graph", "elements"),
        ("tabu4rec", "interactions"),
    ),
)
def test_f0_binding_rejects_reordered_typed_split_members(
    contract_id: str,
    member_key: str,
) -> None:
    def reverse_members(payload: dict[str, Any]) -> None:
        members = payload["split"]["partitions"][0][member_key]
        payload["split"]["partitions"][0][member_key] = tuple(reversed(members))

    spec = _mutated_spec(_spec(contract_id), reverse_members)

    with pytest.raises(FitExperimentError, match="ordered compiler members"):
        validate_f0_binding(spec)


@pytest.mark.parametrize(
    ("field_name", "message"),
    (
        ("split_manifest_hash", "SplitManifest"),
        ("source_view_hash", "source SplitView"),
        ("fit_view_hash", "fit SplitView"),
        ("recipe_hash", "recipe hash"),
        ("numeric_normalizer_hash", "numeric normalizer"),
    ),
)
def test_f0_binding_rejects_compiler_provenance_drift(
    field_name: str,
    message: str,
) -> None:
    spec = _spec()
    fixture = build_f0_fixture(spec.contract_id)
    object.__setattr__(fixture.compilation.provenance, field_name, "f" * 64)

    with pytest.raises(FitExperimentError, match=message):
        validate_f0_binding(spec, fixture)


def test_compiler_binding_manifest_closes_typed_projection() -> None:
    spec = _spec("tabu4rec")
    fixture = validate_f0_binding(spec)

    manifest = compiler_binding_manifest(spec, fixture)

    assert set(manifest) == {
        "schema",
        "typed_split_hash",
        "typed_split_kind",
        "fit_partition",
        "compiler_provenance",
        "compiler_provenance_hash",
        "numeric_normalizer",
        "projection",
    }
    assert manifest["typed_split_hash"] == spec.split.content_hash
    provenance = manifest["compiler_provenance"]
    assert provenance["dataset_hash"] == fixture.dataset.dataset_hash
    assert provenance["recipe_hash"] == fixture.recipe.recipe_hash
    assert provenance["split_manifest_hash"] == fixture.split_manifest.manifest_hash
    normalizer = manifest["numeric_normalizer"]
    assert normalizer["artifact_hash"] == fixture.numeric_normalizer.artifact_hash
    assert normalizer["fit_value_mask_hash"] == (
        fixture.numeric_normalizer.statistics.fit_value_mask_hash
    )
    assert normalizer["shared_numeric_groups"] == (fixture.numeric_normalizer.shared_numeric_groups)
    assert torch.equal(normalizer["counts"], fixture.numeric_normalizer.statistics.counts)
    assert torch.equal(normalizer["means"], fixture.numeric_normalizer.statistics.means)
    assert torch.equal(normalizer["scales"], fixture.numeric_normalizer.statistics.scales)
    assert normalizer["feature_kinds"] == (fixture.numeric_normalizer.statistics.feature_kinds)


def test_source_tree_manifest_records_repository_preimage() -> None:
    repository = Path(__file__).resolve().parents[2]

    manifest = source_tree_manifest(repository)

    paths = {entry["path"] for entry in manifest["files"]}
    assert manifest["mode"] == "repository"
    assert "src/tabu_lab/experiments/runner.py" in paths
    assert "pyproject.toml" in paths
    assert source_tree_hash(repository) == canonical_hash(manifest)


def test_source_tree_manifest_supports_clean_installed_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "site-packages" / "tabu_lab"
    runner_path = package / "experiments" / "runner.py"
    runner_path.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("# installed package\n", encoding="utf-8")
    runner_path.write_text("# installed runner\n", encoding="utf-8")
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"not provenance")
    monkeypatch.setattr(runner_module, "__file__", str(runner_path))

    manifest = source_tree_manifest()

    assert manifest["mode"] == "installed_package"
    assert manifest["root_label"] == "tabu_lab_package"
    assert [entry["path"] for entry in manifest["files"]] == [
        "__init__.py",
        "experiments/runner.py",
    ]
    assert source_tree_hash() == canonical_hash(manifest)


def test_source_reviewed_boolean_cannot_self_authorize_formal_run(
    tmp_path: Path,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)

    with pytest.raises(FitExperimentError, match="authorization_catalog"):
        run_fit_experiment(
            preregistration,
            output_root=tmp_path / "runs",
            formal=True,
            source_reviewed=True,
        )

    assert not (tmp_path / "runs").exists()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _reviewed_fit_repository(
    tmp_path: Path,
    spec: FitExperimentSpec,
) -> tuple[Path, Path, str]:
    repository = tmp_path / "reviewed-repository"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "TabU Runner Test")
    _git(repository, "config", "user.email", "tabu-runner@example.test")
    preregistration = repository / "experiments" / "F0" / "preregistration.yaml"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    source = repository / "src" / "tabu_lab" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = 'runner-test'\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "reviewed fit source")
    _git(repository, "remote", "add", "origin", "https://example.test/wehub/tabu-lab.git")
    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repository, "branch", "--set-upstream-to=origin/main", "main")
    return repository, preregistration, _git(repository, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "field_name",
    (
        "commit",
        "git_tree_oid",
        "source_tree_hash",
        "preregistration_blob_hash",
        "remote_ref",
        "repository_uri",
        "repository_subdirectory",
        "lock_hash",
    ),
)
def test_formal_runner_revalidates_injected_git_identity_against_live_source(
    tmp_path: Path,
    field_name: str,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    authorization_catalog, canonical_preregistration, manifest = (
        _write_authorization_catalog(tmp_path, spec, preregistration)
    )
    repository = authorization_catalog.parent
    live = SourceIdentity.model_validate(manifest["source_identity"])
    assert live.issuance_status == "formal"
    payload = live.model_dump(mode="python")
    if field_name in {
        "commit",
        "git_tree_oid",
        "source_tree_hash",
        "preregistration_blob_hash",
        "lock_hash",
    }:
        current = str(payload[field_name])
        payload[field_name] = ("f" if set(current) != {"f"} else "e") * len(current)
    elif field_name == "remote_ref":
        payload[field_name] = "refs/remotes/origin/forged"
    elif field_name == "repository_subdirectory":
        payload[field_name] = "forged/subdirectory"
    else:
        payload[field_name] = "https://evil.example/forged/tabu-lab.git"
    forged = SourceIdentity.model_validate(payload)

    with pytest.raises(FitExperimentError, match="formal receipt refused"):
        run_fit_experiment(
            canonical_preregistration,
            output_root=tmp_path / "runs",
            repository=repository,
            formal=True,
            source_reviewed=True,
            authorization_catalog=authorization_catalog,
            source_identity=forged,
        )

    assert not (tmp_path / "runs").exists()


def _wheel_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tabu_lab/__init__.py", "__version__ = 'test'\n")
        archive.writestr("tabu_lab/experiments/runner.py", "# installed runner\n")
        archive.writestr(
            "tabu_lab-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "tabu_lab-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: tabu-lab\nVersion: 0.1.0\n",
        )
    return buffer.getvalue()


def test_installed_manifest_reverifies_distribution_and_lock_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "site-packages" / "tabu_lab"
    runner_path = package / "experiments" / "runner.py"
    runner_path.parent.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    runner_path.write_text("# installed runner\n", encoding="utf-8")
    monkeypatch.setattr(runner_module, "__file__", str(runner_path))
    wheel = _wheel_bytes()
    lock = b"version = 1\n"
    wheel_path = tmp_path / "retrieved" / "tabu_lab-0.1.0-py3-none-any.whl"
    wheel_path.parent.mkdir()
    wheel_path.write_bytes(wheel)
    lock_path = tmp_path / "retrieved" / "uv.lock"
    lock_path.write_bytes(lock)
    installed_files = tuple(
        {
            "path": path.relative_to(package).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(candidate for candidate in package.rglob("*") if candidate.is_file())
    )
    installed_tree_hash = canonical_hash(
        {
            "schema_version": "tabu.source-tree-preimage.v1",
            "mode": "installed_package",
            "root_label": "tabu_lab_package",
            "files": installed_files,
        }
    )
    expected = distribution_source_identity(
        uri="https://example.test/releases/tabu_lab-0.1.0-py3-none-any.whl",
        sha256=hashlib.sha256(wheel).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=wheel,
        retrieved_lock=lock,
        source_tree_hash=installed_tree_hash,
        live_source_root=package,
    )

    formal_manifest = source_tree_manifest(
        request_formal=True,
        reviewed=True,
        source_identity=expected,
        distribution_artifact=wheel_path,
        distribution_lock=lock_path,
    )
    missing_bytes_manifest = source_tree_manifest(
        request_formal=True,
        reviewed=True,
        source_identity=expected,
    )

    formal_identity = SourceIdentity.model_validate(formal_manifest["source_identity"])
    missing_bytes_identity = SourceIdentity.model_validate(
        missing_bytes_manifest["source_identity"]
    )
    assert formal_identity.issuance_status == "formal"
    assert formal_identity.source_tree_hash is not None
    assert str(tmp_path) not in formal_identity.model_dump_json()
    assert missing_bytes_identity.issuance_status == "local_unissued"
    assert "distribution_bytes_not_verified" in missing_bytes_identity.reasons
    assert "dependency_lock_bytes_not_verified" in missing_bytes_identity.reasons


def _fit_evaluation(gradient_step: int | None) -> FitEvaluationBundle:
    return FitEvaluationBundle(
        evaluation_id="fit-runner-gradient",
        experiment_id="F0-001-tabuf-v1",
        stage=FitStage.F0,
        model_seed=1729,
        targets=1,
        scored_targets=1,
        coverage=1.0,
        families=(
            FitFamilyMetrics(
                family=FitTargetFamily.COMPLETION,
                kind=FitMetricKind.NUMERIC,
                targets=1,
                scored_targets=1,
                initial_loss=1.0,
                final_loss=0.0,
                mse=0.0,
            ),
        ),
        gradient_nonzero_by_step=gradient_step,
        parameter_delta_norm=1.0,
        checkpoint_reloaded=True,
    )


def _rec_fit_evaluation(
    *,
    user_active_targets: int,
    item_active_targets: int,
    mechanism_scored_targets: int = 2,
) -> FitEvaluationBundle:
    return FitEvaluationBundle(
        evaluation_id="fit-runner-rec-mechanisms",
        experiment_id="F0-014-tabu4rec-axis-address-v2",
        stage=FitStage.F0,
        model_seed=1729,
        targets=2,
        scored_targets=2,
        coverage=1.0,
        families=(
            FitFamilyMetrics(
                family=FitTargetFamily.COMPLETION,
                kind=FitMetricKind.NUMERIC,
                targets=2,
                scored_targets=2,
                initial_loss=1.0,
                final_loss=0.0,
                trivial_baseline_loss=1.0,
                mse=0.0,
            ),
        ),
        gradient_nonzero_by_step=1,
        gradient_group_nonzero_by_step={"dynamics": 1, "readout": 1},
        mechanism_source_counts={"rec_user_arm": 2, "rec_item_arm": 2},
        mechanism_active_target_counts={
            "rec_user_arm": user_active_targets,
            "rec_item_arm": item_active_targets,
        },
        mechanism_scored_target_count=mechanism_scored_targets,
        mechanism_gradient_norms={"rec_user_arm": 1.0, "rec_item_arm": 1.0},
        parameter_delta_norm=1.0,
        checkpoint_reloaded=True,
    )


def test_gradient_gate_requires_nonzero_gradient_by_frozen_deadline() -> None:
    spec = _spec()

    on_time = _gate_reasons(
        spec,
        _fit_evaluation(10),
        initial_objective=1.0,
        final_objective=0.0,
    )
    late = _gate_reasons(
        spec,
        _fit_evaluation(11),
        initial_objective=1.0,
        final_objective=0.0,
    )
    absent = _gate_reasons(
        spec,
        _fit_evaluation(None),
        initial_objective=1.0,
        final_objective=0.0,
    )

    assert "nonzero_gradient_after_required_step" not in on_time
    assert "nonzero_gradient_after_required_step" in late
    assert "no_nonzero_gradient_by_required_step" in absent


def test_rec_gate_requires_both_arms_for_every_scored_target() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
    )

    partial = _gate_reasons(
        spec,
        _rec_fit_evaluation(user_active_targets=1, item_active_targets=2),
        initial_objective=1.0,
        final_objective=0.0,
    )
    complete = _gate_reasons(
        spec,
        _rec_fit_evaluation(user_active_targets=2, item_active_targets=2),
        initial_objective=1.0,
        final_objective=0.0,
    )
    mismatched_count = _gate_reasons(
        spec,
        _rec_fit_evaluation(
            user_active_targets=1,
            item_active_targets=1,
            mechanism_scored_targets=1,
        ),
        initial_objective=1.0,
        final_objective=0.0,
    )

    assert "rec_user_arm_not_active_for_every_scored_target" in partial
    assert "rec_item_arm_not_active_for_every_scored_target" not in partial
    assert not any("rec_" in reason for reason in complete)
    assert "rec_mechanism_scored_target_count_mismatch" in mismatched_count


def test_fit_evaluation_rejects_active_mechanism_count_above_scored_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _rec_fit_evaluation(
            user_active_targets=2,
            item_active_targets=2,
            mechanism_scored_targets=1,
        )


def test_model_build_is_float32_independent_of_process_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model = _build_model(_spec(), seed=1729, device=torch.device("cpu"))
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(previous)

    floating = (
        value
        for _, value in (*model.named_parameters(), *model.named_buffers())
        if value.is_floating_point()
    )
    assert all(value.dtype is torch.float32 for value in floating)


def test_model_build_rejects_non_float32_execution_even_after_model_mutation() -> None:
    spec = _spec()
    object.__setattr__(spec.execution, "dtype", "float64")

    with pytest.raises(FitExperimentError, match="only float32"):
        _build_model(spec, seed=1729, device=torch.device("cpu"))


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Apple MPS is required for this deterministic backward regression",
)
def test_mps_float32_objective_backward_is_deterministic_kernel_safe() -> None:
    spec = build_f0_preregistration("tabuf", device=FitDevice.MPS)
    fixture = build_f0_fixture("tabuf")
    device = torch.device("mps")
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = _build_model(spec, seed=1729, device=device)
        prediction = model(fixture.evidence.to(device))
        loss = Objective()(prediction, fixture.truth.to(device)).total
        loss.backward()
        torch.mps.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous)

    gradients = tuple(
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    )
    assert loss.dtype is torch.float32
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)


@pytest.mark.parametrize("contract_id", ("tabuf", "tabul", "tabufl", "tabu4rec"))
def test_model_build_consumes_semantic_augmented_readout_geometry(
    contract_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_f0_preregistration(
        contract_id,
        device=FitDevice.CPU,
        augmented_readout_geometry=AugmentedReadoutGeometry.MATCHED_UFC,
    )
    captured: dict[str, Any] = {}

    def fake_builder(model_id: str, **kwargs: Any) -> torch.nn.Module:
        captured["model_id"] = model_id
        captured.update(kwargs)
        return torch.nn.Linear(1, 1)

    monkeypatch.setattr(runner_module, "build_model", fake_builder)

    _build_model(spec, seed=1729, device=torch.device("cpu"))

    assert captured["model_id"] == contract_id
    assert captured["readout_geometry"] == "matched_ufc"


def test_metric_forward_uses_eval_mode_and_restores_training_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = build_f0_preregistration(
        "tabuf",
        device=FitDevice.CPU,
        reference=ReferenceBackendConfig(dropout=0.5),
    )
    fixture = build_f0_fixture("tabuf")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))
    observed_modes: list[bool] = []
    original_forward = model.forward

    def observe_mode(inputs: Any):  # type: ignore[no-untyped-def]
        observed_modes.append(model.training)
        return original_forward(inputs)

    monkeypatch.setattr(model, "forward", observe_mode)
    model.train()

    first = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))
    torch.manual_seed(999)
    second = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))

    assert observed_modes == [False, False]
    assert model.training
    assert first.prediction_hash == second.prediction_hash


def test_family_metrics_prefer_stable_categorical_log_probabilities() -> None:
    spec = _spec()
    fixture = build_f0_fixture("tabuf")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))
    prediction = _forward_in_eval(model, fixture.evidence, device=torch.device("cpu"))
    categorical_targets = prediction.auxiliaries["categorical_target_mask"].to(torch.bool)
    log_probabilities = prediction.auxiliaries["categorical_log_probabilities"].clone()
    log_probabilities[categorical_targets] = -200.0
    projected = replace(
        prediction,
        auxiliaries={
            **prediction.auxiliaries,
            "categorical_log_probabilities": log_probabilities,
        },
    )

    metrics = _typed_family_metrics(
        initial=projected,
        final=projected,
        truth=fixture.truth,
        baseline={"families": {"completion": {"numeric_mse": 1.0, "categorical_nll": 1.0}}},
    )

    categorical = next(item for item in metrics if item.kind is FitMetricKind.CATEGORICAL)
    assert categorical.initial_loss == 200.0
    assert categorical.final_loss == 200.0
    assert categorical.nll == 200.0


def test_zero_coverage_is_an_invalid_seed_verdict() -> None:
    evaluation = FitEvaluationBundle(
        evaluation_id="fit-runner-zero-coverage",
        experiment_id="F0-001-tabuf-v1",
        stage=FitStage.F0,
        model_seed=1729,
        targets=1,
        scored_targets=0,
        coverage=0.0,
        families=(
            FitFamilyMetrics(
                family=FitTargetFamily.COMPLETION,
                kind=FitMetricKind.NUMERIC,
                targets=1,
                scored_targets=0,
                initial_loss=1.0,
                final_loss=0.0,
                mse=0.0,
            ),
        ),
        gradient_nonzero_by_step=1,
        parameter_delta_norm=1.0,
        checkpoint_reloaded=True,
    )
    reasons = _gate_reasons(
        _spec(),
        evaluation,
        initial_objective=1.0,
        final_objective=0.0,
    )

    assert "not_all_targets_scored" in reasons
    assert "coverage_below_required" in reasons
    assert _seed_verdict(evaluation, reasons) == "invalid"


@pytest.mark.parametrize(
    ("seed_verdicts", "expected"),
    (
        (("pass", "pass", "pass"), "pass"),
        (("pass", "pass", "failed"), "unstable"),
        (("pass", "invalid", "failed"), "failed"),
        (
            ("diagnostic_pass", "diagnostic_pass", "diagnostic_pass"),
            "diagnostic_pass",
        ),
        (
            ("diagnostic_pass", "diagnostic_pass", "failed"),
            "diagnostic_unstable",
        ),
    ),
)
def test_three_seed_aggregate_verdicts_are_explicit(
    seed_verdicts: tuple[str, str, str],
    expected: str,
) -> None:
    results = tuple(SimpleNamespace(verdict=verdict) for verdict in seed_verdicts)

    assert _aggregate_verdict(results) == expected


def _quick_spec() -> FitExperimentSpec:
    def mutate(payload: dict[str, Any]) -> None:
        payload["training"]["max_updates"] = 1
        payload["supersedes_experiment_ids"] = []
        payload["revision_rationale"] = None

    return _mutated_spec(_spec(), mutate)


def _write_preregistration(path: Path, spec: FitExperimentSpec) -> Path:
    target = path / "preregistration.yaml"
    target.write_text(
        yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return target


_TEST_PUBLIC_REPOSITORY = "https://github.com/wehub-community/tabu-lab.git"


def _write_authorization_catalog(
    directory: Path,
    spec: FitExperimentSpec,
    preregistration: Path,
    preregistration_sha256: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    authorization_repository = directory / "authorization-repo"
    remote_repository = directory / "authorization-remote.git"
    remote_repository.mkdir()
    _git(remote_repository, "init", "--bare", "-b", "main")
    (authorization_repository / "specs/models").mkdir(parents=True)
    canonical_preregistration = (
        authorization_repository
        / "experiments/fit-first/F0"
        / spec.experiment_id
        / "preregistration.yaml"
    )
    canonical_preregistration.parent.mkdir(parents=True)
    canonical_preregistration.write_text(
        preregistration.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (authorization_repository / "specs/models" / f"{spec.contract_id}.yaml").write_text(
        yaml.safe_dump(
            get_model_spec(spec.contract_id).model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = authorization_repository / "src/tabu_lab/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = 'formal-authority-test'\n", encoding="utf-8")
    (authorization_repository / "pyproject.toml").write_text(
        "[project]\nname = 'tabu-lab-formal-test'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (authorization_repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(authorization_repository, "init", "-b", "main")
    _git(authorization_repository, "config", "user.email", "tests@example.test")
    _git(authorization_repository, "config", "user.name", "TabU Tests")
    _git(authorization_repository, "remote", "add", "origin", _TEST_PUBLIC_REPOSITORY)
    _git(
        authorization_repository,
        "config",
        f"url.file://{remote_repository.resolve()}.insteadOf",
        _TEST_PUBLIC_REPOSITORY,
    )
    _git(
        authorization_repository,
        "config",
        "tabu.tests.bareRemote",
        str(remote_repository.resolve()),
    )
    _git(authorization_repository, "add", ".")
    _git(authorization_repository, "commit", "-m", "freeze executable source")
    _git(authorization_repository, "push", "-u", "origin", "main")
    manifest = source_tree_manifest(
        authorization_repository,
        preregistration=canonical_preregistration,
        request_formal=True,
        reviewed=True,
    )
    source_identity = SourceIdentity.model_validate(manifest["source_identity"])
    assert source_identity.issuance_status == "formal"
    preregistration_hash = preregistration_sha256 or canonical_hash(
        yaml.safe_load(preregistration.read_text(encoding="utf-8"))
    )
    evidence = authorization_repository / "authorization-evidence"
    evidence.mkdir()
    report_payload = {"review": spec.experiment_id}
    report_path = evidence / "review-report.json"
    report_path.write_text(canonical_json(report_payload) + "\n", encoding="utf-8")
    gong_payload = {"approval": spec.experiment_id, "decision": "approved"}
    gong_path = evidence / "gong-approval.json"
    gong_path.write_text(canonical_json(gong_payload) + "\n", encoding="utf-8")
    source_path = evidence / "source-identity.json"
    source_path.write_text(canonical_json(source_identity) + "\n", encoding="utf-8")
    review_report = EvidencePointer(
        uri=report_path.relative_to(authorization_repository).as_posix(),
        sha256=canonical_hash(report_payload),
    )
    review = ReviewRecord(
        review_id=f"review-{spec.experiment_id}",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id=spec.experiment_id,
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=review_report,
        gong_approval=EvidencePointer(
            uri=gong_path.relative_to(authorization_repository).as_posix(),
            sha256=canonical_hash(gong_payload),
        ),
    )
    source_pointer = EvidencePointer(
        uri=source_path.relative_to(authorization_repository).as_posix(),
        sha256=canonical_hash(source_identity),
    )
    experiment = ExperimentRecord(
        experiment_id=spec.experiment_id,
        contract_id=spec.contract_id,
        hypothesis="bounded formal authorization fixture",
        claim_boundary="F0 fit only",
        status=ExperimentStatus.RUNNABLE,
        status_history=(
            StatusEvent(status=ExperimentStatus.DRAFT.value),
            StatusEvent(
                status=ExperimentStatus.PREREGISTERED.value,
                evidence_hashes=(preregistration_hash, review_report.sha256),
            ),
            StatusEvent(
                status=ExperimentStatus.RUNNABLE.value,
                evidence_hashes=(source_pointer.sha256,),
            ),
        ),
        preregistration=EvidencePointer(
            uri=canonical_preregistration.relative_to(authorization_repository).as_posix(),
            sha256=preregistration_hash,
            media_type="application/yaml",
        ),
        preregistration_review=review_report,
        source_identity=source_pointer,
        review_ids=(review.review_id,),
        supersedes_experiment_ids=spec.supersedes_experiment_ids,
        revision_rationale=spec.revision_rationale,
    )
    record_path = (
        authorization_repository / "experiments/records" / f"{spec.experiment_id}.json"
    )
    record_path.parent.mkdir()
    record_path.write_text(canonical_json(experiment) + "\n", encoding="utf-8")
    review_path = authorization_repository / "reviews" / f"{review.review_id}.json"
    review_path.parent.mkdir()
    review_path.write_text(canonical_json(review) + "\n", encoding="utf-8")
    _git(authorization_repository, "add", ".")
    _git(authorization_repository, "commit", "-m", "review authorization")
    source_revision_commit = _git(authorization_repository, "rev-parse", "HEAD")
    draft = build_catalog(authorization_repository)
    source_revision = CatalogSourceRevision(
        repository_uri=_TEST_PUBLIC_REPOSITORY,
        commit=source_revision_commit,
        catalog_source_tree_hash=draft.source_tree_hash,
    )
    target = authorization_repository / "catalog.json"
    build_catalog(
        authorization_repository,
        target,
        source_revision=source_revision,
    )
    _git(authorization_repository, "add", "catalog.json")
    _git(authorization_repository, "commit", "-m", "freeze catalog")
    _git(authorization_repository, "push", "origin", "main")
    return target, canonical_preregistration, manifest


def _formal_source_identity(
    token: str = "a",
    *,
    source_tree_hash: str | None = None,
    preregistration_blob_hash: str | None = None,
) -> SourceIdentity:
    return SourceIdentity(
        source_kind="git",
        issuance_status="formal",
        reviewed=True,
        repository_uri="https://example.test/wehub/tabu-lab.git",
        repository_subdirectory=".",
        commit=token * 40,
        remote_ref="refs/remotes/origin/main",
        git_tree_oid=token * 40,
        source_tree_hash=source_tree_hash or token * 64,
        preregistration_blob_hash=preregistration_blob_hash or token * 64,
        lock_hash=token * 64,
    )


def test_formal_output_root_inside_repository_must_be_git_ignored(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    (repository / ".gitignore").write_text(".local-runs/\n", encoding="utf-8")

    _assert_formal_output_root_safe(
        repository / ".local-runs" / "formal-staging",
        repository=repository,
    )
    _assert_formal_output_root_safe(
        tmp_path / "external-formal-staging",
        repository=repository,
    )
    with pytest.raises(FitExperimentError, match="must be Git ignored"):
        _assert_formal_output_root_safe(
            repository / "formal-staging",
            repository=repository,
        )


def test_formal_recorded_command_redacts_catalog_and_output_paths() -> None:
    authorization = FormalAuthorizationSummary(
        canonical_commit="d" * 40,
        catalog_hash="a" * 64,
        catalog_source_tree_hash="e" * 64,
        experiment_id="F0-test",
        experiment_status="runnable",
        preregistration_sha256="b" * 64,
        source_identity_sha256="c" * 64,
        review_ids=("review-F0-test",),
        review_report_sha256s=("f" * 64,),
        gong_approval_sha256s=("1" * 64,),
    )

    command = _authorization_safe_command(
        (
            "tabu-lab",
            "experiments",
            "run",
            "experiments/F0-test/preregistration.yaml",
            "--output-root",
            "/Users/researcher/formal-staging",
            "--formal",
            "--authorization-catalog",
            "/Users/researcher/reviewed-catalog.json",
        ),
        authorization,
        output_root="/Users/researcher/formal-staging",
    )

    assert "/Users/" not in " ".join(command)
    assert "formal-staging://output" in command
    assert f"sha256:{authorization.catalog_hash}" in command


def test_formal_authorization_rejects_preregistration_canonical_hash_drift(
    tmp_path: Path,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    catalog, _, _ = _write_authorization_catalog(tmp_path, spec, preregistration)
    drifted = yaml.safe_load(preregistration.read_text(encoding="utf-8"))
    drifted["experiment_id"] = "F0-drifted"

    with pytest.raises(FitExperimentError, match="authorization replay failed"):
        runner_module._resolve_formal_authorization(
            catalog,
            spec=spec,
            preregistration_path=preregistration,
            preregistration_text=yaml.safe_dump(drifted, sort_keys=False),
            repository=None,
        )


def test_formal_preflight_binds_live_source_and_passes_path_free_receipt_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    catalog, canonical_preregistration, manifest = _write_authorization_catalog(
        tmp_path, spec, preregistration
    )
    monkeypatch.setattr(
        runner_module,
        "source_tree_manifest",
        lambda *args, **kwargs: manifest,
    )
    observed: list[dict[str, Any]] = []

    def capture_seed(**kwargs: Any) -> SimpleNamespace:
        observed.append(kwargs)
        return SimpleNamespace(model_seed=kwargs["seed"], verdict="pass")

    monkeypatch.setattr(runner_module, "_run_f0_seed", capture_seed)
    monkeypatch.setattr(
        runner_module,
        "_write_experiment_aggregate",
        lambda **kwargs: ExperimentAggregateArtifacts(
            directory=tmp_path / "aggregate",
            summary=tmp_path / "aggregate" / "summary.json",
            checksums=tmp_path / "aggregate" / "artifacts.sha256",
            aggregate_hash="d" * 64,
            verdict="pass",
        ),
    )
    private_output = tmp_path / "formal-runs"
    result = run_fit_experiment(
        canonical_preregistration,
        output_root=private_output,
        formal=True,
        authorization_catalog=catalog,
        command=(
            "tabu-lab",
            "experiments",
            "run",
            "preregistration.yaml",
            "--output-root",
            str(private_output),
            "--formal",
            "--authorization-catalog",
            str(catalog),
        ),
    )

    assert result.passed is True
    assert len(observed) == 3
    for seed_call in observed:
        recorded = " ".join(seed_call["command"])
        assert str(private_output) not in recorded
        assert str(catalog) not in recorded
        assert "formal-staging://output" in recorded
        authorization = seed_call["formal_authorization"]
        assert isinstance(authorization, FormalAuthorizationContext)
        assert authorization.catalog == catalog.resolve()
        assert authorization.experiment_id == spec.experiment_id


def test_formal_preflight_rejects_live_source_identity_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    catalog, canonical_preregistration, manifest = _write_authorization_catalog(
        tmp_path, spec, preregistration
    )
    authorized_identity = SourceIdentity.model_validate(manifest["source_identity"])
    live_payload = authorized_identity.model_dump(mode="python")
    live_payload["source_tree_hash"] = "b" * 64
    live_identity = SourceIdentity.model_validate(live_payload)
    monkeypatch.setattr(
        runner_module,
        "source_tree_manifest",
        lambda *args, **kwargs: {
            "schema_version": "tabu.source-tree.v3",
            "mode": "repository",
            "root_label": "repository",
            "files": (),
            "source_identity": live_identity.model_dump(mode="json"),
        },
    )

    with pytest.raises(FitExperimentError, match="does not match canonical authorization"):
        run_fit_experiment(
            canonical_preregistration,
            output_root=tmp_path / "formal-runs",
            formal=True,
            authorization_catalog=catalog,
        )


def test_formal_runtime_failure_redacts_private_exception_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    catalog, canonical_preregistration, manifest = _write_authorization_catalog(
        tmp_path, spec, preregistration
    )
    monkeypatch.setattr(
        runner_module,
        "source_tree_manifest",
        lambda *args, **kwargs: manifest,
    )
    monkeypatch.setattr(
        runner_module,
        "_build_model",
        lambda *args, **kwargs: _raise(
            RuntimeError("failed under /mnt/private/run; api_key=must-not-leak")
        ),
    )

    result = run_fit_experiment(
        canonical_preregistration,
        output_root=tmp_path / "formal-runs",
        formal=True,
        authorization_catalog=catalog,
    )

    assert len(result.seed_results) == 3
    for seed_result in result.seed_results:
        assert "/mnt/private" not in (seed_result.error or "")
        receipt = read_receipt(seed_result.artifacts.receipt)
        assert receipt.error is not None
        assert "details withheld" in receipt.error
        serialized = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in seed_result.artifacts.directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".md", ".yaml"}
        )
        assert "/mnt/private" not in serialized
        assert "must-not-leak" not in serialized


def _raise(error: Exception):  # type: ignore[no-untyped-def]
    raise error


@pytest.mark.parametrize(
    ("failure_case", "expected_phase", "expected_code"),
    (
        ("build", "build", "execution_error"),
        ("oom", "build", "out_of_memory"),
        ("train", "train", "execution_error"),
        ("nonfinite", "train", "nonfinite"),
        ("evaluation", "final_evaluation", "execution_error"),
        ("checkpoint", "checkpoint", "execution_error"),
        ("artifact", "artifact", "execution_error"),
    ),
)
def test_runtime_failures_publish_immutable_typed_receipts_and_continue_all_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
    expected_phase: str,
    expected_code: str,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(runner_module, "_checkpoint_roundtrip", lambda *args, **kwargs: True)

    if failure_case == "build":
        monkeypatch.setattr(
            runner_module,
            "_build_model",
            lambda *args, **kwargs: _raise(RuntimeError("injected build failure")),
        )
    elif failure_case == "oom":
        monkeypatch.setattr(
            runner_module,
            "_build_model",
            lambda *args, **kwargs: _raise(torch.OutOfMemoryError("CUDA out of memory")),
        )
    elif failure_case == "train":
        monkeypatch.setattr(
            runner_module,
            "_trainer",
            lambda *args, **kwargs: SimpleNamespace(
                train_step=lambda *step_args, **step_kwargs: _raise(
                    RuntimeError("injected train failure")
                )
            ),
        )
    elif failure_case == "nonfinite":
        nonfinite_step = SimpleNamespace(
            loss=SimpleNamespace(total=torch.tensor(float("nan"))),
            gradient_norm=1.0,
            step=1,
        )
        monkeypatch.setattr(
            runner_module,
            "_trainer",
            lambda *args, **kwargs: SimpleNamespace(
                train_step=lambda *step_args, **step_kwargs: nonfinite_step
            ),
        )
    elif failure_case == "evaluation":
        monkeypatch.setattr(
            runner_module.Evaluator,
            "evaluate",
            lambda *args, **kwargs: _raise(RuntimeError("injected evaluation failure")),
        )
    elif failure_case == "checkpoint":
        monkeypatch.setattr(
            runner_module,
            "_checkpoint_roundtrip",
            lambda *args, **kwargs: _raise(RuntimeError("injected checkpoint failure")),
        )
    else:
        monkeypatch.setattr(
            runner_module,
            "write_fit_attempt_artifacts",
            lambda *args, **kwargs: _raise(RuntimeError("injected artifact failure")),
        )

    result = run_fit_experiment(
        preregistration,
        output_root=tmp_path / "runs",
        repository=repository,
    )

    assert result.aggregate.verdict == "failed"
    assert tuple(item.model_seed for item in result.seed_results) == spec.seeds.model_seeds
    assert all(item.failure_phase == expected_phase for item in result.seed_results)
    for item in result.seed_results:
        assert item.verdict == "failed"
        assert item.fit_evaluation is None
        receipt = read_receipt(item.artifacts.receipt)
        assert receipt.status is ReceiptStatus.FAILED
        assert receipt.metadata["issuance_status"] == "local_unissued"
        assert isinstance(receipt.metadata["source_identity_hash"], str)
        assert receipt.metadata["failure_phase"] == expected_phase
        assert receipt.metadata["failure_code"] == expected_code
        failure = json.loads(
            (item.artifacts.directory / "failure.json").read_text(encoding="utf-8")
        )
        assert failure["phase"] == expected_phase
        assert failure["code"] == expected_code
        for line in item.artifacts.checksums.read_text(encoding="utf-8").splitlines():
            expected_sha256, relative = line.split("  ", 1)
            assert (
                hashlib.sha256((item.artifacts.directory / relative).read_bytes()).hexdigest()
                == expected_sha256
            )

    aggregate = json.loads(result.aggregate.summary.read_text(encoding="utf-8"))
    assert aggregate["aggregate_hash"] == canonical_hash(aggregate["aggregate"])
    assert len(aggregate["aggregate"]["seed_attempts"]) == 3
    with pytest.raises(FileExistsError, match="immutable aggregate"):
        _write_experiment_aggregate(
            output_root=tmp_path / "runs",
            spec=spec,
            results=result.seed_results,
        )


def test_identity_free_preflight_failure_is_an_explicit_schema_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    monkeypatch.setattr(
        runner_module,
        "_training_and_execution_configs",
        lambda *args, **kwargs: _raise(RuntimeError("injected environment failure")),
    )

    with pytest.raises(FitExperimentError, match="before RunIdentity formation"):
        run_fit_experiment(
            preregistration,
            output_root=tmp_path / "runs",
            repository=Path(__file__).resolve().parents[2],
        )

    assert not (tmp_path / "runs").exists()


def test_zero_coverage_attempt_receipt_is_failed_but_fit_verdict_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _quick_spec()
    preregistration = _write_preregistration(tmp_path, spec)
    repository = Path(__file__).resolve().parents[2]
    monkeypatch.setattr(runner_module, "_checkpoint_roundtrip", lambda *args, **kwargs: True)

    def zero_coverage_evaluation(
        _evaluator: Any,
        predictions: tuple[Any, ...],
        _truth: tuple[Any, ...],
        *,
        evaluation_id: str,
    ) -> EvaluationBundle:
        return EvaluationBundle(
            evaluation_id=evaluation_id,
            episode_ids=(predictions[0].episode_id,),
            metrics={"coverage": 0.0},
            counts={"targets": 1, "scored_targets": 0},
        )

    monkeypatch.setattr(runner_module.Evaluator, "evaluate", zero_coverage_evaluation)
    monkeypatch.setattr(
        runner_module,
        "_typed_family_metrics",
        lambda **kwargs: (
            FitFamilyMetrics(
                family=FitTargetFamily.COMPLETION,
                kind=FitMetricKind.NUMERIC,
                targets=1,
                scored_targets=0,
                initial_loss=1.0,
                final_loss=1.0,
                mse=1.0,
            ),
        ),
    )
    result = run_fit_experiment(
        preregistration,
        output_root=tmp_path / "runs",
        repository=repository,
    )

    assert all(item.verdict == "invalid" for item in result.seed_results)
    for item in result.seed_results:
        assert item.fit_evaluation is not None
        assert not item.fit_evaluation.count_validation.ready
        receipt = read_receipt(item.artifacts.receipt)
        assert receipt.status is ReceiptStatus.FAILED
        run_bundle = json.loads(item.artifacts.run_bundle.read_text(encoding="utf-8"))
        assert run_bundle["metadata"]["attempt_verdict"] == "invalid"
