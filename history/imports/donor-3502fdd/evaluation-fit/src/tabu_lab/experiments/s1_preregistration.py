"""Decision-complete preregistration builders for the frozen S1 matrix."""

from __future__ import annotations

from tabu_lab.contracts import canonical_hash
from tabu_lab.registry import get_model_spec

from .contracts import (
    AugmentedReadoutGeometry,
    BaselineRole,
    BaselineSpec,
    DatasetAdapterSpec,
    DatasetOrigin,
    FitDatasetSpec,
    FitDevice,
    FitExecutionConfig,
    FitExperimentSpec,
    FitPassGate,
    FitSeedConfig,
    FitStage,
    FitTrainingConfig,
    GraphUnitReceiverPlan,
    LabelAddressPlan,
    ModelSemanticConfig,
    RecommendationAddressPlan,
    RedistributionPolicy,
    ReferenceBackendConfig,
)
from .s1_registry import (
    S1ExperimentRegistration,
    S1Recipe,
    get_s1_registration,
    list_s1_registrations,
)


def _reference_config(registration: S1ExperimentRegistration) -> ReferenceBackendConfig:
    if registration.contract_id == "tabu.unit_row":
        # Preserve the repaired F0 row-Unit geometry in S1.
        return ReferenceBackendConfig(
            geometry_normalization="rms_unit",
            routing_bandwidth=2.5,
        )
    return ReferenceBackendConfig()


def _semantic_config(
    registration: S1ExperimentRegistration,
) -> ModelSemanticConfig:
    contract_id = registration.contract_id
    base = contract_id == "tabu.cell.base"
    base_supervised = base and registration.recipe in {
        S1Recipe.BASE_SUPERVISED_REGRESSION,
        S1Recipe.BASE_SUPERVISED_CLASSIFICATION,
    }
    return ModelSemanticConfig(
        reference=_reference_config(registration),
        augmented_readout_geometry=(
            AugmentedReadoutGeometry.MATCHED_UF
            if contract_id in {"tabuf", "tabul", "tabufl", "tabu4rec"}
            else None
        ),
        label_columns=(6,) if base_supervised else ((6, 7) if contract_id in {"tabul", "tabufl"} else ()),
        label_address_plan=(
            LabelAddressPlan.PREDICTOR_UNIT_LINKED_PER_LABEL_V2
            if contract_id in {"tabul", "tabufl"}
            else None
        ),
        target_feature=2 if contract_id == "tabu4graph" else None,
        graph_unit_receiver_plan=(
            GraphUnitReceiverPlan.SAME_ROW_VISIBLE_CELLS if contract_id == "tabu4graph" else None
        ),
        response_family=(registration.recipe.value if contract_id == "tabu4rec" else None),
        recommendation_address_plan=(
            RecommendationAddressPlan.AXIS_ADDRESS_BOOTSTRAP_V1
            if contract_id == "tabu4rec"
            else None
        ),
        rec_axis_summary_dim=2 if contract_id == "tabu4rec" else None,
        rec_matched_residual_scale=0.1 if contract_id == "tabu4rec" else None,
        profile_id=(
            "supervised.label_broadcast.v1" if base_supervised
            else "completion.artificial_mask.v1" if base else None
        ),
    )


def build_s1_preregistration(experiment_id: str) -> FitExperimentSpec:
    """Build one canonical CUDA:0 S1 preregistration from its registry entry."""

    registration = get_s1_registration(experiment_id)
    corpus = registration.build_corpus()
    model_spec = get_model_spec(registration.contract_id)
    return FitExperimentSpec(
        experiment_id=registration.experiment_id,
        stage=FitStage.S1,
        contract_id=registration.contract_id,
        contract_version=model_spec.contract_version,
        model_spec=model_spec,
        model_spec_hash=canonical_hash(model_spec),
        dataset=FitDatasetSpec(
            dataset_id=corpus.dataset.dataset_id,
            origin=DatasetOrigin.GENERATED,
            source_uri=registration.source_uri,
            source_sha256=registration.source_hash,
            dataset_hash=corpus.dataset.dataset_hash,
            license_id="Apache-2.0",
            redistribution=RedistributionPolicy.ALLOWED,
            adapter=DatasetAdapterSpec(
                adapter_id=registration.adapter_id,
                adapter_version=registration.adapter_version,
            ),
        ),
        split=corpus.typed_split,
        episode_schedule=corpus.schedule,
        semantic=_semantic_config(registration),
        training=FitTrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_updates=3000,
            wall_clock_budget_minutes=15,
            exact_resume=True,
        ),
        execution=FitExecutionConfig(
            device=FitDevice.CUDA,
            device_index=0,
            deterministic_algorithms=True,
        ),
        seeds=FitSeedConfig(
            model_seeds=(1729, 2718, 31415),
            data_seed=104729,
            split_seed=130363,
            episode_order_seed=130363,
        ),
        target_families=corpus.schedule.target_families,
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
            stage=FitStage.S1,
            max_loss_ratio=0.10,
            max_trivial_baseline_ratio=0.50,
            max_numeric_nrmse=0.05,
            min_categorical_accuracy=0.98,
            max_categorical_nll=0.10,
        ),
    )


def build_all_s1_preregistrations() -> tuple[FitExperimentSpec, ...]:
    return tuple(
        build_s1_preregistration(registration.experiment_id)
        for registration in list_s1_registrations()
    )


def validate_s1_binding(spec: FitExperimentSpec) -> None:
    """Fail closed when a loaded S1 spec drifts from its frozen registration."""

    if not isinstance(spec, FitExperimentSpec):
        raise TypeError("S1 binding validation requires FitExperimentSpec")
    if spec.stage is not FitStage.S1:
        raise ValueError("S1 binding validation requires stage S1")
    expected = build_s1_preregistration(spec.experiment_id)
    if spec != expected:
        raise ValueError(
            "S1 preregistration does not match its registered generator, corpus, or config"
        )


def s1_dataset_manifest(experiment_id: str) -> dict[str, object]:
    """Return the deterministic prepared-dataset binding for one S1 experiment."""

    registration = get_s1_registration(experiment_id)
    corpus = registration.build_corpus()
    spec = build_s1_preregistration(experiment_id)
    return {
        "schema": "tabu.prepared-dataset.v1",
        "dataset": spec.dataset,
        "typed_split_hash": corpus.typed_split.content_hash,
        "schedule_hash": corpus.schedule.content_hash,
        "schedule_realization_hash": corpus.schedule_realization.content_hash,
        "fit_value_mask_hash": corpus.fit_value_mask_hash,
        "source_ledger_hashes": dict(corpus.source_ledger_hashes),
        "corpus_hash": corpus.corpus_hash,
        "shape": corpus.dataset.shape,
        "feature_specs": corpus.dataset.feature_specs,
        "claim_boundary": "generated_s1_fit_only_not_generalization",
    }


__all__ = [
    "S1Recipe",
    "build_all_s1_preregistrations",
    "build_s1_preregistration",
    "s1_dataset_manifest",
    "validate_s1_binding",
]
