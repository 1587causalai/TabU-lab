from __future__ import annotations

import inspect
from dataclasses import replace

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
    QueryTerminalAdapter,
    TabUQueryBaseModel,
    TabUQueryRowModel,
    build_model,
    canonical_query_base_manifest,
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


def test_query_row_contract_builds_augmented_carrier_and_row_projection() -> None:
    result = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    )
    assert result.executable
    assert isinstance(result.model, TabUQueryRowModel)
    assert result.model.geometry.geometry == "row_unit_projection"
    assert result.model.family_plan.response_mechanism == "row_unit_projection"
    assert result.model.row_token_count == 2
    assert result.model.component_manifest.geometry.component_id == (
        "tabu.query.geometry.row_unit_projection"
    )
    assert result.model.component_manifest.dynamics.component_id == "tabu.query.row.dynamics"
    assert all(
        "tabu.cell" not in ref["component_id"]
        for ref in result.model.component_manifest.as_dict().values()
        if isinstance(ref, dict) and "component_id" in ref
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
    assert output.trace.metadata["geometry"] == "row_unit_projection"
    assert output.trace.metadata["response_mechanism"] == "row_unit_projection"
    assert output.trace.metadata["unit_semantics"] == "abstract_axis_roles_with_row_unit_tokens"
    dynamics_event = next(event for event in output.trace.events if event.name == "dynamics_plan")
    assert dynamics_event.metadata["shape"] == (1, 3, 4, 8)
    assert dynamics_event.metadata["source_count"] == 5


def test_query_row_checkpoint_identity_rejects_token_count_change() -> None:
    first = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=2,
    ).model
    second = build_contract(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=3,
    ).model
    with pytest.raises(ValueError, match="family_plan|row_token_count"):
        second.validate_checkpoint_identity(first.checkpoint_identity())


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
        contract_version="0.1.0",
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
        contract_version="0.1.0",
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
