from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    canonical_hash,
)
from tabu_lab.models import (
    AxisMode,
    QueryComponentRole,
    QueryComponentSpec,
    QueryFamilyModelBase,
    QueryFamilyPlan,
    QueryRowGeometryAdapter,
    QueryTerminalAdapter,
    RowReadoutMode,
    TabUQueryBaseModel,
    TabUQueryRowModel,
    build_model,
    canonical_query_base_manifest,
    canonical_query_row_manifest,
    factory_dependency_hash,
    implementation_source_identity,
)
from tabu_lab.registry import (
    BuildStatus,
    get_model_spec,
    model_spec_identity_payload,
)
from tabu_lab.registry import (
    build_model as build_contract,
)
from tabu_lab.verification import (
    QueryEvaluationStage,
    QueryEvidenceLevel,
    QueryHarnessStatus,
    QueryRunStatus,
    TabUQueryEvaluationLadder,
    assess_query_family_growth,
    assess_query_runtime_growth,
    query_family_probe,
    verify_tabu_query_base_component_correctness,
    verify_tabu_query_row_component_correctness,
    verify_tabu_query_row_component_evolvability,
)


def _config():
    from tabu_lab.models.types import ReferenceConfig

    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=4,
    )


def _completion_episode() -> EvidenceEpisode:
    values = torch.tensor([[1.0, 2.0], [2.0, 0.0], [3.0, 6.0]])
    origins = [
        [OriginState.OBSERVED, OriginState.OBSERVED],
        [OriginState.OBSERVED, OriginState.ARTIFICIAL_MASK],
        [OriginState.OBSERVED, OriginState.OBSERVED],
    ]
    roles = [
        [ForwardRole.RECEIVER | ForwardRole.SOURCE] * 2,
        [ForwardRole.RECEIVER | ForwardRole.SOURCE, ForwardRole.RECEIVER | ForwardRole.TARGET],
        [ForwardRole.RECEIVER | ForwardRole.SOURCE] * 2,
    ]
    return EvidenceEpisode(
        episode_id="query-base-completion",
        dataset_id="query-test",
        source_partition="context",
        fit_partition="fit",
        row_ids=("r0", "r1", "r2"),
        feature_names=("x", "y"),
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
        feature_specs=(
            FeatureSpec(name="x", kind=FeatureKind.NUMERIC, role=FeatureRole.PREDICTOR),
            FeatureSpec(name="y", kind=FeatureKind.NUMERIC, role=FeatureRole.PREDICTOR),
        ),
    )


def _supervised_episode() -> EvidenceEpisode:
    values = torch.tensor([[1.0, 10.0], [2.0, 0.0], [3.0, 30.0]])
    origins = [
        [OriginState.OBSERVED, OriginState.OBSERVED],
        [OriginState.OBSERVED, OriginState.QUERY],
        [OriginState.OBSERVED, OriginState.OBSERVED],
    ]
    roles = [
        [ForwardRole.RECEIVER | ForwardRole.SOURCE] * 2,
        [ForwardRole.RECEIVER | ForwardRole.SOURCE, ForwardRole.RECEIVER | ForwardRole.TARGET],
        [ForwardRole.RECEIVER | ForwardRole.SOURCE] * 2,
    ]
    return EvidenceEpisode(
        episode_id="query-base-supervised",
        dataset_id="query-test",
        source_partition="context",
        fit_partition="fit",
        row_ids=("r0", "r1", "r2"),
        feature_names=("x", "y"),
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
        feature_specs=(
            FeatureSpec(name="x", kind=FeatureKind.NUMERIC, role=FeatureRole.PREDICTOR),
            FeatureSpec(name="y", kind=FeatureKind.NUMERIC, role=FeatureRole.RESPONSE),
        ),
    )


def test_query_spec_and_builder_are_independent() -> None:
    spec = get_model_spec("tabu.query.base", "0.1.0")
    old = get_model_spec("tabu.cell.base", "0.2.0")
    assert spec.contract_id == "tabu.query.base"
    assert spec.carrier["cell_role"] == "query"
    assert spec.carrier["unit_semantics"] == "abstract_axis_roles"
    assert old.contract_id == "tabu.cell.base"
    assert canonical_hash(model_spec_identity_payload(spec)) != canonical_hash(
        model_spec_identity_payload(old)
    )
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    assert isinstance(model, TabUQueryBaseModel)
    assert model.model_id == "tabu.query.base"
    assert model.component_composition is not None
    assert all(
        "tabu.cell" not in ref["component_id"]
        for ref in model.component_manifest.as_dict().values()
        if isinstance(ref, dict) and "component_id" in ref
    )


@pytest.mark.parametrize("contract_id", ("tabu.query.column", "tabu.query.row_column"))
def test_query_sibling_contracts_remain_design_open_until_runtime_freeze(contract_id: str) -> None:
    spec = get_model_spec(contract_id, "0.1.0")
    assert spec.maturity.stage.value == "design_open"
    assert spec.maturity.build_state.value == "design_open"
    result = build_contract(contract_id, profile="completion.artificial_mask.v1")
    assert result.status is BuildStatus.DESIGN_OPEN
    assert not result.executable


def test_query_row_contract_builds_augmented_carrier_and_anchored_readout() -> None:
    result = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    )
    assert result.executable
    assert isinstance(result.model, TabUQueryRowModel)
    assert result.model.contract_version == "0.2.0"
    assert result.model.geometry.geometry == "row_readout"
    assert result.model.row_readout_mode is RowReadoutMode.ANCHORED
    assert result.model.family_plan.response_mechanism == "row_readout"
    assert result.model.row_token_count == 2
    assert result.model.component_manifest.geometry.component_id == (
        "tabu.query.geometry.row_readout"
    )
    assert result.model.component_manifest.geometry.component_version == "0.2.0"
    assert result.model.component_manifest.dynamics.component_id == "tabu.query.row.dynamics"
    assert result.model.component_manifest.dynamics.component_version == "0.2.0"
    assert all(
        "tabu.cell" not in ref["component_id"]
        for ref in result.model.component_manifest.as_dict().values()
        if isinstance(ref, dict) and "component_id" in ref
    )


def test_query_row_model_spec_and_source_manifest_are_byte_and_hash_bound() -> None:
    root = Path(__file__).resolve().parents[2]
    public_spec = root / "specs" / "models" / "tabu.query.row.yaml"
    packaged_spec = root / "src" / "tabu_lab" / "specs" / "models" / public_spec.name
    public_manifest = root / "specs" / "model-factory-source-manifest.json"
    packaged_manifest = root / "src" / "tabu_lab" / "specs" / public_manifest.name
    assert public_spec.read_bytes() == packaged_spec.read_bytes()
    assert public_manifest.read_bytes() == packaged_manifest.read_bytes()
    spec = get_model_spec("tabu.query.row", "0.2.0")
    source = json.loads(public_manifest.read_text())["contracts"]["tabu.query.row"]
    closure = json.dumps(
        source["semantic_source_closure"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert source["entrypoint_sha256"] == spec.upstream.sha256
    assert hashlib.sha256(closure).hexdigest() == spec.upstream.semantic_source_tree_sha256


def test_query_row_explicit_manifest_cannot_omit_readout_identity() -> None:
    manifest = canonical_query_row_manifest(token_count=2)
    incomplete_geometry = type(manifest.geometry)(
        manifest.geometry.component_id,
        manifest.geometry.component_version,
        manifest.geometry.role,
        {"token_count": 2},
    )
    with pytest.raises(ValueError, match="missing required readout config"):
        build_model(
            "tabu.query.row",
            config=_config(),
            profile="completion.artificial_mask.v1",
            row_token_count=2,
            component_manifest=replace(manifest, geometry=incomplete_geometry),
        )


def test_query_row_forward_uses_only_ordinary_cells_for_readout() -> None:
    result = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    )
    output = result.model(_completion_episode())
    assert output.model_id == "tabu.query.row"
    assert output["numeric"].shape == (3, 2)
    assert output.trace is not None
    assert output.trace.metadata["geometry"] == "row_readout"
    assert output.trace.metadata["response_mechanism"] == "row_readout"
    assert output.trace.metadata["row_readout"]["mode"] == "anchored"
    assert output.trace.metadata["row_readout"]["global_w_rows"] == 2
    assert output.trace.metadata["unit_semantics"] == "abstract_axis_roles_with_row_unit_tokens"
    dynamics_event = next(event for event in output.trace.events if event.name == "dynamics_plan")
    assert dynamics_event.metadata["shape"] == (1, 3, 4, 8)
    assert dynamics_event.metadata["source_count"] == 5


def test_query_row_geometry_implements_all_three_tex_formulas() -> None:
    config = _config()
    carrier = torch.arange(1, 1 + 1 * 1 * 4 * 8, dtype=torch.float32).reshape(1, 1, 4, 8)
    cells = carrier[:, :, :2, :]
    row_units = carrier[:, :, 2:, :]
    weight = torch.tensor(
        [
            [1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0],
        ]
    )

    homogeneous = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="homogeneous",
    )
    anchored = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="anchored",
        anchored_gamma_initial=0.25,
    )
    free = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="free",
    )
    with torch.no_grad():
        homogeneous.projection.weight.copy_(weight)
        anchored.projection.weight.copy_(weight)
        free.projection.weight.copy_(weight)
        anchored.raw_axis_transform.copy_(torch.diag(torch.arange(1.0, 9.0)))

    expected_w = torch.nn.functional.linear(cells, weight)
    normalized_units = torch.nn.functional.layer_norm(row_units, (8,))
    expected_anchored = expected_w + 0.25 * torch.einsum(
        "bnkd,bnmd->bnmk",
        torch.nn.functional.linear(
            normalized_units,
            anchored.effective_axis_transform(),
        ),
        cells,
    )
    expected_free = torch.einsum("bnkd,bnmd->bnmk", row_units, cells)

    assert torch.equal(homogeneous(carrier), expected_w)
    assert torch.allclose(anchored(carrier), expected_anchored, atol=1.0e-6, rtol=1.0e-6)
    assert torch.equal(free(carrier), expected_free)


def test_query_row_anchored_gamma_zero_exactly_matches_homogeneous_readout() -> None:
    config = _config()
    carrier = torch.randn(1, 3, 4, 8)
    homogeneous = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="homogeneous",
    )
    anchored = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="anchored",
        anchored_gamma_initial=0.0,
    )
    with torch.no_grad():
        anchored.projection.weight.copy_(homogeneous.projection.weight)
    assert torch.equal(anchored(carrier), homogeneous(carrier))


def test_query_row_homogeneous_and_gamma_zero_match_query_base_whole_forward() -> None:
    models = []
    for contract_id, options in (
        ("tabu.query.base", {}),
        (
            "tabu.query.row",
            {"row_token_count": 2, "row_readout_mode": "homogeneous"},
        ),
        (
            "tabu.query.row",
            {
                "row_token_count": 2,
                "row_readout_mode": "anchored",
                "anchored_gamma_initial": 0.0,
            },
        ),
    ):
        torch.manual_seed(20260831)
        models.append(
            build_model(
                contract_id,
                config=_config(),
                profile="completion.artificial_mask.v1",
                **options,
            )
        )
    episode = _completion_episode()
    encoded = [model._encode_dense_queries(episode) for model in models]
    outputs = [model(episode) for model in models]
    base_states = encoded[0][3]
    for row_encoded, output in zip(encoded[1:], outputs[1:], strict=True):
        assert torch.equal(row_encoded[2][:, :, :-2], encoded[0][2])
        assert torch.equal(row_encoded[3][:, :, :-2], base_states)
        assert torch.equal(output.auxiliaries["coordinates"], outputs[0].auxiliaries["coordinates"])
        assert torch.equal(
            output.auxiliaries["completion_support_weights"],
            outputs[0].auxiliaries["completion_support_weights"],
        )
        assert torch.equal(
            output.auxiliaries["completion_support_ids"],
            outputs[0].auxiliaries["completion_support_ids"],
        )
        assert torch.equal(output["numeric"], outputs[0]["numeric"])


def test_query_row_readout_parameter_and_gradient_boundaries() -> None:
    config = _config()
    carrier = torch.randn(1, 2, 4, 8, requires_grad=True)
    anchored = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="anchored",
    )
    loss = anchored(carrier).square().mean()
    loss.backward()
    assert anchored.projection.weight.grad is not None
    assert anchored.raw_axis_transform.grad is not None
    assert anchored.gamma.grad is not None
    assert carrier.grad is not None
    assert all(
        torch.isfinite(value).all()
        for value in (
            anchored.projection.weight.grad,
            anchored.raw_axis_transform.grad,
            anchored.gamma.grad,
            carrier.grad,
        )
    )
    assert torch.allclose(
        torch.linalg.matrix_norm(anchored.effective_axis_transform(), ord=2),
        torch.tensor(1.0),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    homogeneous = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="homogeneous",
    )
    free = QueryRowGeometryAdapter(
        config,
        token_count=2,
        row_readout_mode="free",
    )
    assert set(dict(homogeneous.named_parameters())) == {"projection.weight"}
    assert not dict(free.named_parameters())["projection.weight"].requires_grad
    assert "raw_axis_transform" not in dict(free.named_parameters())
    assert "gamma" not in dict(free.named_parameters())


@pytest.mark.parametrize("mode", ("homogeneous", "free"))
def test_non_anchored_readout_rejects_irrelevant_gamma_variants(mode: str) -> None:
    with pytest.raises(ValueError, match="anchored_gamma_initial is fixed at 0.01"):
        build_model(
            "tabu.query.row",
            config=_config(),
            profile="supervised.label_broadcast.v1",
            row_token_count=2,
            row_readout_mode=mode,
            anchored_gamma_initial=9.0,
        )


def test_query_row_modes_keep_steps_one_to_three_identical() -> None:
    encoded = []
    models = []
    for mode in RowReadoutMode:
        torch.manual_seed(20260831)
        model = build_contract(
            "tabu.query.row",
            config=_config(),
            profile="completion.artificial_mask.v1",
            row_token_count=2,
            row_readout_mode=mode.value,
        ).model
        models.append(model)
        encoded.append(model._encode_dense_queries(_completion_episode()))
    for candidate in encoded[1:]:
        assert torch.equal(candidate[2], encoded[0][2])
        assert torch.equal(candidate[3], encoded[0][3])
    assert len({model.checkpoint_identity()["row_readout"]["mode"] for model in models}) == 3
    assert len({model.component_manifest.manifest_hash for model in models}) == 3
    assert len({model.component_composition.composition_hash for model in models}) == 3
    assert len({model.variant_ref.semantic_hash for model in models}) == 3


def test_row_tokens_are_receiver_only_and_change_only_nonhomogeneous_readout() -> None:
    for mode, expect_coordinate_change in (("anchored", True), ("homogeneous", False)):
        torch.manual_seed(20260831)
        model = build_model(
            "tabu.query.row",
            config=_config(),
            profile="completion.artificial_mask.v1",
            row_token_count=2,
            row_readout_mode=mode,
        )
        first = model._encode_dense_queries(_completion_episode())
        first_coordinates = model.geometry(first[3])
        with torch.no_grad():
            model.row_unit_markers.add_(
                torch.linspace(
                    -0.5,
                    0.5,
                    model.row_unit_markers.numel(),
                ).reshape_as(model.row_unit_markers)
            )
        second = model._encode_dense_queries(_completion_episode())
        second_coordinates = model.geometry(second[3])
        assert torch.equal(first[3][:, :, :-2], second[3][:, :, :-2])
        assert not torch.equal(first[3][:, :, -2:], second[3][:, :, -2:])
        assert (not torch.equal(first_coordinates, second_coordinates)) is expect_coordinate_change


def test_query_row_checkpoint_rejects_mode_and_legacy_contract_before_tensor_load() -> None:
    anchored = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
        row_readout_mode="anchored",
    ).model
    free = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
        row_readout_mode="free",
    ).model
    with pytest.raises(ValueError, match="mismatch"):
        free.load_checkpoint_state(anchored.state_dict(), anchored.checkpoint_identity())
    legacy_identity = dict(anchored.checkpoint_identity())
    legacy_identity["contract_version"] = "0.1.0"
    with pytest.raises(ValueError, match="contract_version"):
        anchored.load_checkpoint_state({}, legacy_identity)


def test_query_row_k_triad_rejects_token_count_change() -> None:
    with pytest.raises(ValueError, match="requires K"):
        build_model(
            "tabu.query.row",
            config=_config(),
            profile="completion.artificial_mask.v1",
            row_token_count=3,
        )


def test_query_row_component_correctness_is_local_unissued() -> None:
    model = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    ).model
    result = verify_tabu_query_row_component_correctness(model)
    assert result.run_status is QueryRunStatus.PASSED
    assert result.evidence_level is QueryEvidenceLevel.LOCAL_UNISSUED
    assert result.claim_status.value == "none"


def test_query_ladder_can_bind_row_contract_without_changing_future_stages() -> None:
    ladder = TabUQueryEvaluationLadder.initial(
        contract_id="tabu.query.row",
        contract_version="0.2.0",
    )
    assert ladder.contract_id == "tabu.query.row"
    assert ladder.stages[0].harness_status is QueryHarnessStatus.IMPLEMENTED
    assert ladder.stages[2].run_status is QueryRunStatus.NOT_RUN


def test_query_row_runtime_growth_holds_public_envelope_and_changes_identity() -> None:
    base = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    row = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    ).model
    assessment = assess_query_runtime_growth(base, row, _completion_episode())
    assert assessment.passed
    result = verify_tabu_query_row_component_evolvability((assessment,))
    assert result.run_status is QueryRunStatus.PASSED
    assert result.evidence_level is QueryEvidenceLevel.LOCAL_UNISSUED


def test_query_family_base_is_identity_free_abstract() -> None:
    assert inspect.isabstract(QueryFamilyModelBase)
    assert not hasattr(QueryFamilyModelBase, "model_id")
    assert not hasattr(QueryFamilyPlan, "model_id")


def test_query_forward_is_public_truth_free_and_role_bound() -> None:
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    output = model(_completion_episode())
    assert output.model_id == "tabu.query.base"
    assert output.contract_version == "0.1.0"
    assert output.trace is not None
    assert output.trace.metadata["cell_role"] == "query"
    assert output.trace.metadata["geometry"] == "global_W"
    assert output.trace.metadata["response_mechanism"] == "shared_W_fallback"
    assert output.trace.metadata.get("unit") is None
    assert all(event.metadata.get("unit") is None for event in output.trace.events)


def test_supervised_query_excludes_query_rows_from_column_context() -> None:
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="supervised.label_broadcast.v1",
    )
    resolved, _, _, _, _ = model._encode_dense_queries(_supervised_episode())
    context = resolved.metadata["context_mask"]
    assert not bool(context[0, 1].any())
    assert bool(context[0, 0].all())
    assert bool(resolved.visible_mask[0, 1, 0])


def test_query_checkpoint_rejects_axis_b_identity_before_tensor_load() -> None:
    query = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    old = build_model(
        "tabu.cell.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    with pytest.raises(ValueError, match="mismatch"):
        query.validate_checkpoint_identity(old.checkpoint_identity())
    with pytest.raises(ValueError, match="mismatch"):
        query.load_checkpoint_state(old.state_dict(), old.checkpoint_identity())


def test_query_profile_isolation() -> None:
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    with pytest.raises(ValueError, match="rejects query"):
        model(_supervised_episode())
    supervised = build_model(
        "tabu.query.base",
        config=_config(),
        profile="supervised.label_broadcast.v1",
    )
    with pytest.raises(ValueError, match="rejects artificial"):
        supervised(_completion_episode())


def test_query_truth_substitution_is_not_a_forward_input() -> None:
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    first = model(_completion_episode())
    second_episode = replace(_completion_episode(), episode_id="query-base-completion-other")
    # The episode id is part of trace identity, so compare the tensor-facing
    # events and prediction values rather than the trace id itself.
    second = model(second_episode)
    assert torch.equal(first["numeric"], second["numeric"])
    assert [event.output_hash for event in first.trace.events] == [
        event.output_hash for event in second.trace.events
    ]


def test_family_growth_probe_changes_only_declared_axes() -> None:
    base = query_family_probe("base")
    row = query_family_probe("r")
    column = query_family_probe("c")
    row_column = query_family_probe("rc")
    assert base.geometry == "global_W"
    assert assess_query_family_growth(base, row, expected_axes=("row",)).passed
    assert assess_query_family_growth(base, column, expected_axes=("column",)).passed
    assert assess_query_family_growth(base, row_column, expected_axes=("row", "column")).passed
    assert row.row_axis.mode is AxisMode.HETEROGENEOUS
    assert column.column_axis.mode is AxisMode.HETEROGENEOUS
    assert base.plan_hash != row.plan_hash != column.plan_hash != row_column.plan_hash


def test_query_ladder_keeps_future_stages_not_run() -> None:
    ladder = TabUQueryEvaluationLadder.initial()
    assert len(ladder.stages) == 6
    assert ladder.stages[0].harness_status is QueryHarnessStatus.IMPLEMENTED
    assert ladder.stages[0].evidence_level is QueryEvidenceLevel.LOCAL_UNISSUED
    assert ladder.stages[2].harness_status is QueryHarnessStatus.NOT_IMPLEMENTED
    assert ladder.stages[2].run_status is QueryRunStatus.NOT_RUN
    assert ladder.stages[2].evidence_level is QueryEvidenceLevel.NONE
    assert ladder.stages[2].stage is QueryEvaluationStage.SYNTHETIC_FIT


def test_query_ladder_can_bind_all_six_executed_stage_results() -> None:
    initial = TabUQueryEvaluationLadder.initial()
    stages = tuple(
        stage.model_copy(
            update={
                "harness_status": QueryHarnessStatus.IMPLEMENTED,
                "run_status": QueryRunStatus.PASSED,
            }
        )
        for stage in initial.stages
    )
    ladder = TabUQueryEvaluationLadder.with_stage_results(
        stages,
        contract_id="tabu.query.row",
        contract_version="0.2.0",
    )
    assert all(stage.run_status is QueryRunStatus.PASSED for stage in ladder.stages)
    assert ladder.contract_id == "tabu.query.row"


def test_query_component_correctness_status_is_local_only() -> None:
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    )
    result = verify_tabu_query_base_component_correctness(model)
    assert result.run_status is QueryRunStatus.PASSED
    assert result.evidence_level is QueryEvidenceLevel.LOCAL_UNISSUED


class ExperimentalQueryTerminal(QueryTerminalAdapter):
    """Test-only query-neutral terminal extension."""


def _experimental_query_terminal_factory(config, settings):
    return ExperimentalQueryTerminal(config, numeric_terminal=str(settings["numeric_terminal"]))


def test_query_registry_supports_protected_extension_without_axis_b_types() -> None:
    registry = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    ).component_registry.fork()
    impl_ref, impl_hash = implementation_source_identity(ExperimentalQueryTerminal)
    factory_ref, factory_hash = implementation_source_identity(_experimental_query_terminal_factory)
    spec = QueryComponentSpec(
        component_id="research.test.query-terminal",
        component_version="1.0.0",
        role=QueryComponentRole.TERMINAL,
        interface_id="tabu.query-terminal.v1",
        implementation_ref=impl_ref,
        implementation_sha256=impl_hash,
        factory_ref=factory_ref,
        factory_sha256=factory_hash,
        factory_dependency_sha256=factory_dependency_hash(_experimental_query_terminal_factory),
        maturity="experimental",
        configurable_fields=("numeric_terminal",),
    )
    registry.register(spec, _experimental_query_terminal_factory, ExperimentalQueryTerminal)
    manifest = canonical_query_base_manifest()
    manifest = replace(
        manifest,
        terminal=type(manifest.terminal)(
            spec.component_id,
            spec.component_version,
            spec.role,
            {"numeric_terminal": "local_linear"},
        ),
    )
    model = build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
        component_manifest=manifest,
        component_registry=registry,
    )
    assert isinstance(model.terminal, ExperimentalQueryTerminal)
    assert model.variant_ref.semantic_hash != build_model(
        "tabu.query.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    ).variant_ref.semantic_hash
