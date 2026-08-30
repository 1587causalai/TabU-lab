"""Deterministic preregistration builders for the frozen F0 matrix."""

from __future__ import annotations

from tabu_lab.contracts import OriginState, canonical_hash, origin_mask
from tabu_lab.registry import get_model_spec

from .contracts import (
    AugmentedReadoutGeometry,
    BaselineRole,
    BaselineSpec,
    DatasetAdapterSpec,
    DatasetOrigin,
    DynamicsBlockKind,
    DynamicsSemanticConfig,
    FitDatasetSpec,
    FitDevice,
    FitEvidenceMode,
    FitExecutionConfig,
    FitExperimentSpec,
    FitPassGate,
    FitSeedConfig,
    FitStage,
    FitTrainingConfig,
    GraphUnitReceiverPlan,
    LabelAddressPlan,
    ModelSemanticConfig,
    NumericTerminal,
    RecommendationAddressPlan,
    RedistributionPolicy,
    ReferenceBackendConfig,
)
from .fixture_registry import (
    build_registered_f0_fixture,
    f0_generator_source_hash,
    f0_generator_source_uri,
)
from .fixtures import BUILDABLE_CONTRACTS
from .splits import (
    GraphElementId,
    GraphPartition,
    GraphSplitManifest,
    GraphSplitScope,
    InteractionId,
    InteractionPartition,
    InteractionSplitManifest,
    RowPartition,
    RowSplitManifest,
)

_EXPERIMENT_IDS = {
    "tabuf": "F0-001-tabuf-v1",
    "tabu.unit_row": "F0-002-tabu-unit-row-v1",
    "tabu.unit_pair": "F0-003-tabu-unit-pair-v1",
    "tabul": "F0-004-tabul-v1",
    "tabufl": "F0-005-tabufl-v1",
    "tabu4graph": "F0-006-tabu4graph-v1",
    "tabu4rec": "F0-007-tabu4rec-v1",
}

_COMPLETION_V2_EXPERIMENT_IDS = {
    "tabuf": "F0-008-tabuf-identifiable-v2",
    "tabu.unit_row": "F0-009-tabu-unit-row-identifiable-v2",
    "tabu.unit_pair": "F0-010-tabu-unit-pair-identifiable-v2",
}

_GRAPH_V2_EXPERIMENT_ID = "F0-011-tabu4graph-row-unit-v2"
_SUPERVISED_V2_EXPERIMENT_IDS = {
    "tabul": "F0-012-tabul-predictor-address-v2",
    "tabufl": "F0-013-tabufl-independent-ledgers-v2",
}
_REC_V2_EXPERIMENT_ID = "F0-014-tabu4rec-axis-address-v2"
_SUPERVISED_V3_EXPERIMENT_IDS = {
    "tabul": "F0-015-tabul-unit-linked-address-v3",
    "tabufl": "F0-016-tabufl-independent-dynamics-v3",
}
_TABUFL_V4_EXPERIMENT_ID = "F0-017-tabufl-balanced-joint-v4"
_TABUFL_V5_EXPERIMENT_ID = "F0-018-tabufl-balanced-16f-v5"
_BASE_EXPERIMENT_IDS = {
    "v1": "F0-023-tabu-cell-base-completion-v1",
    "completion": "F0-023-tabu-cell-base-completion-v1",
    "supervised_regression": "F0-024-tabu-cell-base-supervised-regression-v1",
    "supervised_classification": "F0-025-tabu-cell-base-supervised-classification-v1",
}

_SUPERSEDES_EXPERIMENT_IDS: dict[str, tuple[str, ...]] = {
    "F0-001-tabuf-v1": ("G000-tabuf-artificial-mask",),
    "F0-008-tabuf-identifiable-v2": ("F0-001-tabuf-v1",),
    "F0-009-tabu-unit-row-identifiable-v2": ("F0-002-tabu-unit-row-v1",),
    "F0-010-tabu-unit-pair-identifiable-v2": ("F0-003-tabu-unit-pair-v1",),
    "F0-011-tabu4graph-row-unit-v2": ("F0-006-tabu4graph-v1",),
    "F0-012-tabul-predictor-address-v2": ("F0-004-tabul-v1",),
    "F0-013-tabufl-independent-ledgers-v2": ("F0-005-tabufl-v1",),
    "F0-014-tabu4rec-axis-address-v2": ("F0-007-tabu4rec-v1",),
    "F0-015-tabul-unit-linked-address-v3": ("F0-012-tabul-predictor-address-v2",),
    "F0-016-tabufl-independent-dynamics-v3": ("F0-013-tabufl-independent-ledgers-v2",),
    "F0-017-tabufl-balanced-joint-v4": ("F0-016-tabufl-independent-dynamics-v3",),
    "F0-018-tabufl-balanced-16f-v5": ("F0-017-tabufl-balanced-joint-v4",),
}

_REVISION_RATIONALES = {
    "F0-001-tabuf-v1": (
        "Replace the historical untyped one-seed G000 draft with the decision-complete "
        "three-seed fit-first contract."
    ),
    "F0-008-tabuf-identifiable-v2": (
        "Replace the representation-degenerate v1 completion fixture with the "
        "representation-identifiable v2 fixture."
    ),
    "F0-009-tabu-unit-row-identifiable-v2": (
        "Replace the representation-degenerate v1 completion fixture with the "
        "representation-identifiable v2 fixture and repaired row-unit geometry."
    ),
    "F0-010-tabu-unit-pair-identifiable-v2": (
        "Replace the representation-degenerate v1 completion fixture with the "
        "representation-identifiable v2 fixture."
    ),
    "F0-011-tabu4graph-row-unit-v2": (
        "Replace the legacy graph-unit-only receiver plan with same-row visible-cell "
        "support for the node Unit receiver."
    ),
    "F0-012-tabul-predictor-address-v2": (
        "Replace the v1 matched-UF label address with a representation-identifiable "
        "fixture and predictor-only per-label address."
    ),
    "F0-013-tabufl-independent-ledgers-v2": (
        "Replace the v1 joint fixture with representation-identifiable feature and "
        "label ledgers plus predictor-only per-label address."
    ),
    "F0-014-tabu4rec-axis-address-v2": (
        "Replace the v1 matched-UF recommendation readout with an axis-identifiable "
        "fixture and explicit user/item axis address bootstrap."
    ),
    "F0-015-tabul-unit-linked-address-v3": (
        "Replace predictor-only label addressing with predictor-Unit-linked per-label "
        "addressing while keeping the identifiable v2 fixture fixed."
    ),
    "F0-016-tabufl-independent-dynamics-v3": (
        "Replace predictor-only label addressing with predictor-Unit-linked independent "
        "label dynamics while keeping the identifiable v2 fixture fixed."
    ),
    "F0-017-tabufl-balanced-joint-v4": (
        "Replace the asymmetric v3 joint target schedule with a balanced, "
        "support-realizable joint feature/label fixture."
    ),
    "F0-018-tabufl-balanced-16f-v5": (
        "Replace the 12-feature-target v4 diagnostic with the frozen 16-feature-target "
        "and 32-label-target contract under an independent generator source."
    ),
}


def _typed_split(contract_id: str, fixture, *, experiment_id: str):  # type: ignore[no-untyped-def]
    common = {
        "dataset_id": fixture.dataset.dataset_id,
        "dataset_hash": fixture.dataset.dataset_hash,
        "split_id": f"{experiment_id}-fit-split-v1",
        "fit_partition": "train",
        "strategy": "fixed_f0_episode",
        "seed": fixture.split_seed,
        "require_complete": True,
    }
    if contract_id == "tabu4rec":
        observed = origin_mask(fixture.dataset.origin_states, OriginState.OBSERVED)
        interactions = tuple(
            InteractionId(
                user_id=fixture.dataset.row_ids[row],
                item_id=fixture.dataset.feature_specs[column].name,
            )
            for row, column in observed.nonzero(as_tuple=False).tolist()
        )
        return InteractionSplitManifest(
            **common,
            partitions=(InteractionPartition(name="train", interactions=interactions),),
        )
    if contract_id == "tabu4graph":
        elements = tuple(
            GraphElementId(
                graph_id=fixture.dataset.dataset_id,
                node_id=row_id,
            )
            for row_id in fixture.dataset.row_ids
        )
        return GraphSplitManifest(
            **common,
            scope=GraphSplitScope.NODE,
            partitions=(GraphPartition(name="train", elements=elements),),
        )
    return RowSplitManifest(
        **common,
        partitions=(RowPartition(name="train", row_ids=fixture.dataset.row_ids),),
    )


def build_f0_preregistration(
    contract_id: str,
    *,
    device: FitDevice | str,
    device_index: int | None = None,
    reference: ReferenceBackendConfig | None = None,
    block_kind: DynamicsBlockKind | str | None = None,
    numeric_terminal: NumericTerminal | str | None = None,
    augmented_readout_geometry: AugmentedReadoutGeometry | str | None = None,
    supervised_label_address_plan: LabelAddressPlan | str | None = None,
    experiment_id: str | None = None,
    supersedes_experiment_ids: tuple[str, ...] | None = None,
    revision_rationale: str | None = None,
    fixture_version: str = "v1",
    rec_axis_summary_dim: int | None = None,
    rec_matched_residual_scale: float | None = None,
    recommendation_address_plan: RecommendationAddressPlan | str | None = None,
    deterministic_algorithms: bool = True,
    evidence_mode: FitEvidenceMode | str = FitEvidenceMode.GATE,
    exact_resume: bool = True,
) -> FitExperimentSpec:
    """Build one decision-complete F0 preregistration from canonical fixtures.

    ``block_kind`` is optional only for recreating historical preregistrations;
    new paired experiments should pass ``"omab"`` or ``"mab"`` explicitly so
    the selected variant receives a distinct semantic identity.
    ``numeric_terminal`` is likewise optional for historical compatibility;
    selecting ``"local_linear"`` requires a versioned experiment revision.
    """

    if contract_id not in BUILDABLE_CONTRACTS:
        raise ValueError(f"unsupported F0 contract: {contract_id!r}")
    resolved_device = FitDevice(device)
    fixture = build_registered_f0_fixture(contract_id, fixture_version=fixture_version)
    requested_label_plan = (
        None
        if supervised_label_address_plan is None
        else LabelAddressPlan(supervised_label_address_plan)
    )
    if requested_label_plan is not None and contract_id not in {"tabul", "tabufl"}:
        raise ValueError("supervised_label_address_plan is only valid for TabUL/TabUFL")
    if (
        requested_label_plan is LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
        and fixture_version not in {"v2", "v4", "v5"}
    ):
        raise ValueError("the unit-linked label repair requires the identifiable v2/v4/v5 fixture")
    if fixture_version in {"v4", "v5"} and requested_label_plan not in {
        None,
        LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2,
    }:
        raise ValueError("the TabUFL v4/v5 fixtures are frozen to the unit-linked label plan")
    if contract_id == "tabu.cell.base":
        default_experiment_id = _BASE_EXPERIMENT_IDS[fixture_version]
    elif fixture_version == "v5" and contract_id == "tabufl":
        default_experiment_id = _TABUFL_V5_EXPERIMENT_ID
    elif fixture_version == "v4" and contract_id == "tabufl":
        default_experiment_id = _TABUFL_V4_EXPERIMENT_ID
    elif (
        contract_id in {"tabul", "tabufl"}
        and requested_label_plan is LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
    ):
        default_experiment_id = _SUPERVISED_V3_EXPERIMENT_IDS[contract_id]
    elif fixture_version == "v1":
        default_experiment_id = _EXPERIMENT_IDS[contract_id]
    elif fixture_version == "v2" and contract_id in _COMPLETION_V2_EXPERIMENT_IDS:
        default_experiment_id = _COMPLETION_V2_EXPERIMENT_IDS[contract_id]
    elif fixture_version == "v2" and contract_id == "tabu4graph":
        default_experiment_id = _GRAPH_V2_EXPERIMENT_ID
    elif fixture_version == "v2" and contract_id in _SUPERVISED_V2_EXPERIMENT_IDS:
        default_experiment_id = _SUPERVISED_V2_EXPERIMENT_IDS[contract_id]
    elif fixture_version == "v2" and contract_id == "tabu4rec":
        default_experiment_id = _REC_V2_EXPERIMENT_ID
    else:  # fixture builder normally rejects this before reaching the branch
        raise ValueError(f"no preregistered F0 experiment id for {contract_id} {fixture_version}")
    resolved_experiment_id = experiment_id or default_experiment_id
    if (
        numeric_terminal is not None
        and NumericTerminal(numeric_terminal) is NumericTerminal.LOCAL_LINEAR
        and experiment_id is None
    ):
        raise ValueError(
            "local-linear numeric terminals require an explicit versioned experiment_id"
        )
    if supersedes_experiment_ids is None:
        resolved_supersedes = (
            _SUPERSEDES_EXPERIMENT_IDS.get(resolved_experiment_id, ())
            if experiment_id is None
            else ()
        )
    else:
        resolved_supersedes = supersedes_experiment_ids
    resolved_revision_rationale = revision_rationale
    if resolved_revision_rationale is None and experiment_id is None:
        resolved_revision_rationale = _REVISION_RATIONALES.get(resolved_experiment_id)
    model_spec = get_model_spec(contract_id)
    label_columns = tuple(fixture.builder_options.get("label_columns", ()))
    label_address_plan = requested_label_plan
    if contract_id in {"tabul", "tabufl"} and label_address_plan is None:
        label_address_plan = (
            LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
            if fixture_version in {"v4", "v5"}
            else (
                LabelAddressPlan.PREDICTOR_ONLY_PER_LABEL_V1
                if fixture_version == "v2"
                else LabelAddressPlan.MATCHED_UF
            )
        )
    target_feature = fixture.builder_options.get("target_feature")
    response_family = "ratings" if contract_id == "tabu4rec" else None
    resolved_recommendation_address_plan = None
    resolved_rec_axis_summary_dim = None
    resolved_rec_matched_residual_scale = None
    if contract_id == "tabu4rec":
        resolved_recommendation_address_plan = RecommendationAddressPlan(
            recommendation_address_plan
            or (
                RecommendationAddressPlan.AXIS_ADDRESS_BOOTSTRAP_V1
                if fixture_version == "v2"
                else RecommendationAddressPlan.MATCHED_UF
            )
        )
        if (
            resolved_recommendation_address_plan
            is RecommendationAddressPlan.AXIS_ADDRESS_BOOTSTRAP_V1
        ):
            resolved_rec_axis_summary_dim = int(
                fixture.builder_options["rec_axis_summary_dim"]
                if rec_axis_summary_dim is None
                else rec_axis_summary_dim
            )
            resolved_rec_matched_residual_scale = float(
                fixture.builder_options["rec_matched_residual_scale"]
                if rec_matched_residual_scale is None
                else rec_matched_residual_scale
            )
    augmented_contracts = {"tabuf", "tabul", "tabufl", "tabu4rec"}
    geometry = (
        AugmentedReadoutGeometry(augmented_readout_geometry or AugmentedReadoutGeometry.MATCHED_UF)
        if contract_id in augmented_contracts
        else None
    )
    if contract_id not in augmented_contracts and augmented_readout_geometry is not None:
        raise ValueError("augmented_readout_geometry is only valid for augmented F0 contracts")
    resolved_reference = reference
    if resolved_reference is None:
        resolved_reference = (
            ReferenceBackendConfig(
                geometry_normalization="rms_unit",
                routing_bandwidth=2.5,
            )
            if fixture_version == "v2" and contract_id == "tabu.unit_row"
            else ReferenceBackendConfig()
        )
    semantic_values: dict[str, object] = {
        "reference": resolved_reference,
        "augmented_readout_geometry": geometry,
        "label_columns": label_columns,
        "label_address_plan": label_address_plan,
        "target_feature": target_feature,
        "graph_unit_receiver_plan": (
            GraphUnitReceiverPlan.SAME_ROW_VISIBLE_CELLS
            if contract_id == "tabu4graph" and fixture_version == "v2"
            else (
                GraphUnitReceiverPlan.LEGACY_GRAPH_UNITS_ONLY
                if contract_id == "tabu4graph"
                else None
            )
        ),
        "response_family": response_family,
        "recommendation_address_plan": resolved_recommendation_address_plan,
        "rec_axis_summary_dim": resolved_rec_axis_summary_dim,
        "rec_matched_residual_scale": resolved_rec_matched_residual_scale,
        "profile_id": fixture.builder_options.get("profile"),
    }
    if block_kind is not None:
        semantic_values["dynamics"] = DynamicsSemanticConfig(block_kind=block_kind)
    if numeric_terminal is not None:
        semantic_values["numeric_terminal"] = NumericTerminal(numeric_terminal)

    return FitExperimentSpec(
        experiment_id=resolved_experiment_id,
        supersedes_experiment_ids=resolved_supersedes,
        revision_rationale=resolved_revision_rationale,
        stage=FitStage.F0,
        contract_id=contract_id,
        contract_version=model_spec.contract_version,
        model_spec=model_spec,
        model_spec_hash=canonical_hash(model_spec),
        dataset=FitDatasetSpec(
            dataset_id=fixture.dataset.dataset_id,
            origin=DatasetOrigin.GENERATED,
            source_uri=f0_generator_source_uri(fixture_version=fixture_version),
            source_sha256=f0_generator_source_hash(fixture_version=fixture_version),
            dataset_hash=fixture.dataset.dataset_hash,
            license_id="Apache-2.0",
            redistribution=RedistributionPolicy.ALLOWED,
            adapter=DatasetAdapterSpec(
                adapter_id="tabu-f0-fixtures",
                adapter_version={
                    "v1": "1.0.0",
                    "v2": "2.0.0",
                "v4": "4.0.0",
                "v5": "5.0.0",
                "completion": "2.0.0",
                "supervised_regression": "2.0.0",
                "supervised_classification": "2.0.0",
                }[fixture_version],
            ),
        ),
        split=_typed_split(
            contract_id,
            fixture,
            experiment_id=resolved_experiment_id,
        ),
        episode_schedule=fixture.episode_schedule,
        semantic=ModelSemanticConfig(**semantic_values),
        training=FitTrainingConfig(
            learning_rate=1.0e-2,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_updates=1200,
            wall_clock_budget_minutes=10,
            exact_resume=exact_resume,
        ),
        execution=FitExecutionConfig(
            device=resolved_device,
            device_index=device_index,
            deterministic_algorithms=deterministic_algorithms,
            evidence_mode=FitEvidenceMode(evidence_mode),
        ),
        seeds=FitSeedConfig(
            model_seeds=(1729, 2718, 31415),
            data_seed=104729,
            split_seed=130363,
            episode_order_seed=130363,
        ),
        target_families=fixture.episode_schedule.target_families,
        baselines=(
            BaselineSpec(
                baseline_id="exact-support-mean-mode",
                role=BaselineRole.TRIVIAL,
            ),
            BaselineSpec(
                baseline_id="frozen-initialization",
                role=BaselineRole.DIAGNOSTIC,
            ),
        ),
        pass_gate=FitPassGate(
            stage=FitStage.F0,
            max_loss_ratio=0.01,
            max_numeric_mse=1.0e-3,
            min_categorical_accuracy=1.0,
            max_categorical_nll=0.05,
        ),
    )


def f0_dataset_manifest(
    contract_id: str,
    *,
    fixture_version: str = "v1",
) -> dict[str, object]:
    """Return the deterministic prepared-dataset manifest for one F0 contract."""

    spec = build_f0_preregistration(
        contract_id,
        device=FitDevice.CPU,
        fixture_version=fixture_version,
    )
    fixture = build_registered_f0_fixture(contract_id, fixture_version=fixture_version)
    return {
        "schema": "tabu.prepared-dataset.v1",
        "dataset": spec.dataset,
        "fixture_id": fixture.fixture_id,
        "fixture_hash": fixture.fixture_hash,
        "shape": fixture.dataset.shape,
        "feature_specs": fixture.dataset.feature_specs,
        "row_ids": fixture.dataset.row_ids,
        "claim_boundary": "generated_f0_fixture_only",
    }


__all__ = [
    "build_f0_preregistration",
    "f0_dataset_manifest",
    "f0_generator_source_hash",
]
