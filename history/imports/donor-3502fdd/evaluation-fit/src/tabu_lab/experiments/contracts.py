"""Strict fit-first experiment contracts.

This module is the typed boundary between a preregistration and an executable
fit run.  Runtime code must receive :class:`FitExperimentSpec`, not an
unvalidated mapping.  Every field that can change training semantics has a
named, immutable type and participates in ``spec_hash``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, field_validator, model_serializer, model_validator

from tabu_lab.contracts.canonical import canonical_hash, require_sha256
from tabu_lab.evidence.schemas import ArtifactRef, EvidenceSchema
from tabu_lab.models.types import DynamicsBlockKind
from tabu_lab.registry import ModelSpec

from .splits import (
    FitSplitManifest,
    GraphSplitManifest,
    InteractionSplitManifest,
    RowSplitManifest,
)


class FitStage(StrEnum):
    F0 = "F0"
    S1 = "S1"
    R1 = "R1"


class DatasetOrigin(StrEnum):
    GENERATED = "generated"
    CLASSIC = "classic"


class RedistributionPolicy(StrEnum):
    ALLOWED = "allowed"
    METADATA_ONLY = "metadata_only"
    FORBIDDEN = "forbidden"


class ScheduleSampling(StrEnum):
    FIXED = "fixed"
    DETERMINISTIC_SHUFFLE = "deterministic_shuffle"


class FitTargetFamily(StrEnum):
    COMPLETION = "completion"
    LABEL = "label"


class FitTargetOrigin(StrEnum):
    ARTIFICIAL_MASK = "artificial_mask"
    QUERY = "query"


class NumericTerminal(StrEnum):
    NADARAYA_WATSON = "nadaraya_watson"
    LOCAL_LINEAR = "local_linear"


class CategoricalTerminal(StrEnum):
    NADARAYA_WATSON = "nadaraya_watson"


class AugmentedReadoutGeometry(StrEnum):
    """Executable geometry choices frozen by the augmented model contracts."""

    MATCHED_UF = "matched_uf"
    MATCHED_UFC = "matched_ufc"


class GraphUnitReceiverPlan(StrEnum):
    """Versioned graph Unit-special row-axis realization."""

    LEGACY_GRAPH_UNITS_ONLY = "legacy_graph_units_only"
    SAME_ROW_VISIBLE_CELLS = "same_row_visible_cells"


class LabelAddressPlan(StrEnum):
    """Versioned supervised L-lane address realizations."""

    MATCHED_UF = "matched_uf"
    PREDICTOR_ONLY_PER_LABEL_V1 = "predictor_only_per_label_v1"
    PREDICTOR_UNIT_LINKED_PER_LABEL_V2 = "predictor_unit_linked_per_label_v2"


class RecommendationAddressPlan(StrEnum):
    """Versioned truth-free recommendation address realizations."""

    MATCHED_UF = "matched_uf"
    AXIS_ADDRESS_BOOTSTRAP_V1 = "axis_address_bootstrap_v1"
    CELL_GLOBAL_SUPPORT_V1 = "cell_global_support_v1"


class FitOptimizer(StrEnum):
    ADAMW = "adamw"


class FitDevice(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class FitEvidenceMode(StrEnum):
    """Whether an execution can satisfy a reproducible fit gate."""

    GATE = "gate"
    DIAGNOSTIC_NONDETERMINISTIC = "diagnostic_nondeterministic"


class BaselineRole(StrEnum):
    TRIVIAL = "trivial"
    DIAGNOSTIC = "diagnostic"


class RequiredArtifact(StrEnum):
    PREREGISTRATION = "preregistration.yaml"
    RESOLVED_CONFIGS = "resolved-configs"
    DATASET_MANIFEST = "dataset-manifest.json"
    SPLIT_MANIFEST = "split-manifest.json"
    COMPILER_MANIFEST = "compiler-manifest.json"
    FEASIBILITY = "feasibility.json"
    METRICS = "metrics.jsonl"
    EVALUATION = "evaluation.json"
    FORWARD_TRACES = "forward-traces.json"
    BASELINES = "baselines.json"
    CHECKPOINT = "checkpoint"
    ENVIRONMENT = "environment.json"
    CHECKSUMS = "artifacts.sha256"
    RECEIPT = "receipt.json"
    VERDICT = "verdict.md"


class RunAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INVALID = "invalid"


class DatasetAdapterSpec(EvidenceSchema):
    adapter_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    adapter_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")


class FitDatasetSpec(EvidenceSchema):
    dataset_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    origin: DatasetOrigin
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_id: str = Field(min_length=1)
    redistribution: RedistributionPolicy
    adapter: DatasetAdapterSpec
    raw_files_in_git: Literal[False] = False
    split_before_compile: Literal[True] = True

    @field_validator("source_uri", "license_id")
    @classmethod
    def _nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dataset source and license fields cannot be empty")
        return normalized

    @field_validator("source_sha256", "dataset_hash")
    @classmethod
    def _valid_hash(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))


class EpisodeSchedule(EvidenceSchema):
    schedule_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    sampling: ScheduleSampling
    episode_count: int = Field(gt=0)
    targets_per_episode: int = Field(gt=0)
    target_families: tuple[FitTargetFamily, ...] = Field(min_length=1)
    target_origins: tuple[FitTargetOrigin, ...] = Field(min_length=1)
    sampler_seed: int = Field(ge=0)
    order_seed: int = Field(ge=0)
    recipe_hashes: tuple[str, ...] = ()

    @field_validator("target_families", "target_origins")
    @classmethod
    def _unique_enums(cls, values: tuple[StrEnum, ...]) -> tuple[StrEnum, ...]:
        if len(values) != len(set(values)):
            raise ValueError("episode schedule declarations must be unique")
        return values

    @field_validator("recipe_hashes")
    @classmethod
    def _valid_recipe_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        hashes = tuple(require_sha256(value, field_name="recipe_hash") for value in values)
        if len(hashes) != len(set(hashes)):
            raise ValueError("recipe_hashes must be unique")
        return hashes

    @model_validator(mode="after")
    def _schedule_is_closed(self) -> EpisodeSchedule:
        if self.sampling is ScheduleSampling.FIXED:
            if len(self.recipe_hashes) != self.episode_count:
                raise ValueError("fixed schedules require exactly one recipe hash per episode")
        elif self.recipe_hashes:
            raise ValueError(
                "deterministic_shuffle schedules derive recipes and cannot embed recipe_hashes"
            )
        if (
            FitTargetFamily.COMPLETION in self.target_families
            and FitTargetOrigin.ARTIFICIAL_MASK not in self.target_origins
        ):
            raise ValueError("completion schedules require artificial_mask targets")
        if (
            FitTargetFamily.LABEL in self.target_families
            and FitTargetOrigin.QUERY not in self.target_origins
        ):
            raise ValueError("label schedules require query targets")
        return self

    @property
    def schedule_hash(self) -> str:
        return self.content_hash


class ReferenceBackendConfig(EvidenceSchema):
    backend: Literal["dense_reference_v0"] = "dense_reference_v0"
    d_model: int = Field(default=32, gt=0)
    n_heads: int = Field(default=4, gt=0)
    d_ff: int = Field(default=64, gt=0)
    n_blocks: int = Field(default=2, gt=0)
    inducing_slots: int = Field(default=4, gt=0)
    matched_slots: int = Field(default=4, gt=0)
    max_features: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    presence_tau: float = Field(default=1.0e-6, gt=0.0)
    denominator_epsilon: float = Field(default=1.0e-8, gt=0.0)
    routing_bandwidth: float = Field(default=1.0, gt=0.0)
    geometry_normalization: Literal["none", "rms_unit"] = "none"

    @model_validator(mode="after")
    def _heads_divide_width(self) -> ReferenceBackendConfig:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


class DynamicsSemanticConfig(EvidenceSchema):
    """Global dynamics block choice for one executable model variant."""

    block_kind: DynamicsBlockKind = DynamicsBlockKind.OMAB


class ModelSemanticConfig(EvidenceSchema):
    """Resolved model semantics used to build and identify one run."""

    reference: ReferenceBackendConfig
    dynamics: DynamicsSemanticConfig = Field(default_factory=DynamicsSemanticConfig)
    numeric_terminal: NumericTerminal = NumericTerminal.NADARAYA_WATSON
    categorical_terminal: CategoricalTerminal = CategoricalTerminal.NADARAYA_WATSON
    augmented_readout_geometry: AugmentedReadoutGeometry | None = None
    label_columns: tuple[int, ...] = ()
    label_address_plan: LabelAddressPlan | None = None
    target_feature: int | None = Field(default=None, ge=0)
    graph_unit_receiver_plan: GraphUnitReceiverPlan | None = None
    response_family: str | None = None
    recommendation_address_plan: RecommendationAddressPlan | None = None
    rec_axis_summary_dim: int | None = Field(default=None, gt=0)
    rec_matched_residual_scale: float | None = Field(default=None, gt=0.0)
    profile_id: str | None = None

    @model_serializer(mode="wrap")
    def _serialize_with_legacy_defaults(self, handler: Any) -> Any:
        """Keep omitted v1 fields omitted when re-materializing old evidence.

        Pydantic normally emits every default from ``model_dump``.  That would
        silently add ``dynamics.block_kind=omab`` to a legacy semantic
        preimage and change its identity.  An explicitly supplied dynamics
        section remains serialized (including the canonical ``omab`` value),
        while old instances preserve their original payload shape.
        """

        payload = handler(self)
        if "dynamics" not in self.model_fields_set and isinstance(payload, dict):
            payload.pop("dynamics", None)
        return payload

    @property
    def content_hash(self) -> str:
        """Hash explicit new fields while preserving legacy v1 identities.

        Historical preregistrations did not carry ``semantic.dynamics``.  A
        parsed legacy instance still resolves to OMAB at runtime, but its
        original serialized semantic hash must remain stable for old receipts
        and checkpoints.  New preregistrations that explicitly provide the
        field include it in the canonical hash.
        """

        payload = self.model_dump(mode="python", by_alias=False)
        if "dynamics" not in self.model_fields_set:
            payload.pop("dynamics", None)
        return canonical_hash(payload)

    @field_validator("label_columns")
    @classmethod
    def _unique_label_columns(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(value) is not int for value in values):
            raise ValueError("label_columns must contain integers")
        if len(values) != len(set(values)):
            raise ValueError("label_columns must be unique")
        return values

    @field_validator("response_family")
    @classmethod
    def _normalized_response_family(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("response_family must be non-empty or None")
        return normalized

    @field_validator("profile_id")
    @classmethod
    def _normalized_profile_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("profile_id must be non-empty or None")
        return normalized


class FitTrainingConfig(EvidenceSchema):
    optimizer: Literal[FitOptimizer.ADAMW] = FitOptimizer.ADAMW
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_clip_norm: float = Field(default=1.0, gt=0.0)
    max_updates: int = Field(gt=0)
    max_epochs: int | None = Field(default=None, gt=0)
    wall_clock_budget_minutes: int = Field(gt=0)
    episode_batch_size: Literal[1] = 1
    scheduler: Literal["none"] = "none"
    exact_resume: bool = True
    warm_start: Literal[False] = False
    pretraining: Literal[False] = False


class FitExecutionConfig(EvidenceSchema):
    device: FitDevice
    device_index: int | None = Field(default=None, ge=0)
    dtype: Literal["float32"] = "float32"
    deterministic_algorithms: bool = True
    evidence_mode: FitEvidenceMode = FitEvidenceMode.GATE
    compile_on_cpu: Literal[True] = True
    backend: Literal["dense_reference_v0"] = "dense_reference_v0"

    @model_validator(mode="after")
    def _device_index_is_typed(self) -> FitExecutionConfig:
        if self.device is FitDevice.CUDA and self.device_index is None:
            raise ValueError("CUDA execution requires device_index")
        if self.device is not FitDevice.CUDA and self.device_index is not None:
            raise ValueError("device_index is only valid for CUDA execution")
        if (
            self.deterministic_algorithms
            and self.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
        ):
            raise ValueError("nondeterministic diagnostic mode requires deterministic=false")
        if (
            not self.deterministic_algorithms
            and self.evidence_mode is not FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
        ):
            raise ValueError("deterministic=false requires explicit diagnostic evidence mode")
        return self


class FitSeedConfig(EvidenceSchema):
    model_seeds: tuple[int, ...] = Field(min_length=1)
    data_seed: int = Field(ge=0)
    split_seed: int = Field(ge=0)
    episode_order_seed: int = Field(ge=0)

    @field_validator("model_seeds")
    @classmethod
    def _valid_model_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("model_seeds must be non-negative integers")
        if len(values) != len(set(values)):
            raise ValueError("model_seeds must be unique")
        return values


class FitPassGate(EvidenceSchema):
    stage: FitStage
    required_coverage: Literal[1.0] = 1.0
    max_loss_ratio: float = Field(gt=0.0)
    max_trivial_baseline_ratio: float | None = Field(default=None, gt=0.0)
    max_numeric_mse: float | None = Field(default=None, ge=0.0)
    max_numeric_nrmse: float | None = Field(default=None, ge=0.0)
    min_categorical_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_categorical_nll: float | None = Field(default=None, ge=0.0)
    require_all_seeds: Literal[True] = True

    @model_validator(mode="after")
    def _stage_metrics_are_complete(self) -> FitPassGate:
        if self.stage is FitStage.F0:
            required = (
                self.max_numeric_mse,
                self.min_categorical_accuracy,
                self.max_categorical_nll,
            )
            if any(value is None for value in required):
                raise ValueError("F0 gate requires numeric MSE, categorical accuracy, and NLL")
            if (
                self.max_loss_ratio > 0.01
                or self.max_numeric_mse is None
                or self.max_numeric_mse > 1.0e-3
                or self.min_categorical_accuracy is None
                or self.min_categorical_accuracy < 1.0
                or self.max_categorical_nll is None
                or self.max_categorical_nll > 0.05
            ):
                raise ValueError("F0 gate cannot be weaker than the frozen fit-first thresholds")
        if self.stage is FitStage.S1:
            required = (
                self.max_trivial_baseline_ratio,
                self.max_numeric_nrmse,
                self.min_categorical_accuracy,
                self.max_categorical_nll,
            )
            if any(value is None for value in required):
                raise ValueError("S1 gate requires baseline, NRMSE, accuracy, and NLL")
            if (
                self.max_loss_ratio > 0.10
                or self.max_trivial_baseline_ratio is None
                or self.max_trivial_baseline_ratio > 0.50
                or self.max_numeric_nrmse is None
                or self.max_numeric_nrmse > 0.05
                or self.min_categorical_accuracy is None
                or self.min_categorical_accuracy < 0.98
                or self.max_categorical_nll is None
                or self.max_categorical_nll > 0.10
            ):
                raise ValueError("S1 gate cannot be weaker than the frozen fit-first thresholds")
        if self.stage is FitStage.R1 and self.max_trivial_baseline_ratio is None:
            raise ValueError("R1 gate requires a trivial-baseline ratio")
        if self.stage is FitStage.R1 and (
            self.max_loss_ratio > 0.25
            or self.max_trivial_baseline_ratio is None
            or self.max_trivial_baseline_ratio > 0.80
        ):
            raise ValueError("R1 gate cannot be weaker than the frozen fit-first thresholds")
        return self


class FitKillConditions(EvidenceSchema):
    fail_on_nonfinite: Literal[True] = True
    require_nonzero_gradient_by_step: int = Field(default=10, gt=0)
    require_nonzero_parameter_delta: Literal[True] = True
    stop_model_after_stage_failure: Literal[True] = True
    preserve_failed_receipt: Literal[True] = True


class BaselineSpec(EvidenceSchema):
    baseline_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    role: BaselineRole
    max_updates: int | None = Field(default=None, gt=0)


class FitExperimentSpec(EvidenceSchema):
    """Decision-complete input accepted by the future fit runner."""

    schema_version: Literal["tabu.fit-experiment.v1"] = "tabu.fit-experiment.v1"
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    supersedes_experiment_ids: tuple[str, ...] = ()
    revision_rationale: str | None = Field(default=None, min_length=1)
    stage: FitStage
    contract_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
    model_spec: ModelSpec
    model_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset: FitDatasetSpec
    split: FitSplitManifest
    episode_schedule: EpisodeSchedule
    semantic: ModelSemanticConfig
    training: FitTrainingConfig
    execution: FitExecutionConfig
    seeds: FitSeedConfig
    target_families: tuple[FitTargetFamily, ...] = Field(min_length=1)
    baselines: tuple[BaselineSpec, ...] = Field(min_length=1)
    pass_gate: FitPassGate
    kill_conditions: FitKillConditions = Field(default_factory=FitKillConditions)
    heldout_policy: Literal["diagnostic_only"] = "diagnostic_only"
    required_artifacts: tuple[RequiredArtifact, ...] = tuple(RequiredArtifact)

    @field_validator("model_spec_hash")
    @classmethod
    def _valid_model_spec_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="model_spec_hash")

    @field_validator("target_families", "required_artifacts")
    @classmethod
    def _unique_declarations(cls, values: tuple[StrEnum, ...]) -> tuple[StrEnum, ...]:
        if len(values) != len(set(values)):
            raise ValueError("fit experiment declarations must be unique")
        return values

    @field_validator("supersedes_experiment_ids")
    @classmethod
    def _unique_superseded_experiments(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("supersedes_experiment_ids must be unique")
        return values

    @field_validator("revision_rationale")
    @classmethod
    def _nonblank_revision_rationale(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("revision_rationale cannot be blank")
        return value

    @model_validator(mode="after")
    def _experiment_is_closed(self) -> FitExperimentSpec:
        if self.experiment_id in self.supersedes_experiment_ids:
            raise ValueError("an experiment cannot supersede itself")
        if bool(self.supersedes_experiment_ids) != (self.revision_rationale is not None):
            raise ValueError(
                "supersedes_experiment_ids and revision_rationale must be declared together"
            )
        if self.contract_id == "tabu4do":
            raise ValueError("tabu4do is design_open and cannot enter fit experiments")
        if self.model_spec.contract_id != self.contract_id:
            raise ValueError("model_spec.contract_id must match contract_id")
        if self.model_spec.contract_version != self.contract_version:
            raise ValueError("model_spec.contract_version must match contract_version")
        expected_spec_hash = canonical_hash(self.model_spec)
        if self.model_spec_hash != expected_spec_hash:
            raise ValueError("model_spec_hash does not match the complete ModelSpec")
        if self.dataset.dataset_id != self.split.dataset_id:
            raise ValueError("dataset and split dataset_id must match")
        if self.dataset.dataset_hash != self.split.dataset_hash:
            raise ValueError("dataset and split hashes must match")
        if self.pass_gate.stage is not self.stage:
            raise ValueError("pass_gate.stage must match experiment stage")
        if set(self.target_families) != set(self.episode_schedule.target_families):
            raise ValueError("experiment and episode schedule target families must match")
        if self.episode_schedule.sampler_seed != self.seeds.data_seed:
            raise ValueError("episode sampler_seed must match seeds.data_seed")
        if self.episode_schedule.order_seed != self.seeds.episode_order_seed:
            raise ValueError("episode order_seed must match seeds.episode_order_seed")
        if self.execution.evidence_mode is FitEvidenceMode.GATE:
            if not self.execution.deterministic_algorithms or not self.training.exact_resume:
                raise ValueError("gate evidence requires deterministic execution and exact resume")
        elif self.training.exact_resume:
            raise ValueError("nondeterministic diagnostic execution cannot claim exact resume")
        if self.split.seed is not None and self.split.seed != self.seeds.split_seed:
            raise ValueError("split manifest seed must match seeds.split_seed")
        if self.seeds.model_seeds != (1729, 2718, 31415):
            raise ValueError("fit-first v1 freezes model seeds to 1729, 2718, and 31415")
        if self.seeds.data_seed != 104729 or self.seeds.split_seed != 130363:
            raise ValueError("fit-first v1 freezes data_seed=104729 and split_seed=130363")
        expected_origin = (
            DatasetOrigin.CLASSIC if self.stage is FitStage.R1 else DatasetOrigin.GENERATED
        )
        if self.dataset.origin is not expected_origin:
            raise ValueError(f"{self.stage.value} requires {expected_origin.value} data")
        # LL/NW is an independent numeric readout axis.  A local-linear
        # selection changes executable semantics, so it must be introduced by
        # an explicitly versioned preregistration rather than silently
        # rewriting a historical experiment.
        if (
            self.semantic.numeric_terminal is NumericTerminal.LOCAL_LINEAR
            and self.revision_rationale is None
        ):
            raise ValueError(
                "local-linear numeric terminals require a versioned repair "
                "with revision_rationale and supersedes_experiment_ids"
            )
        expected_learning_rate = 1.0e-2 if self.stage is FitStage.F0 else 1.0e-3
        if not math.isclose(
            self.training.learning_rate,
            expected_learning_rate,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            raise ValueError(f"{self.stage.value} freezes learning_rate={expected_learning_rate:g}")
        budget_limits = {
            FitStage.F0: (1200, 10),
            FitStage.S1: (
                3000,
                15 if self.execution.device is FitDevice.CUDA else 30,
            ),
            FitStage.R1: (10000, 120),
        }
        max_updates, max_minutes = budget_limits[self.stage]
        if (
            self.training.max_updates > max_updates
            or self.training.wall_clock_budget_minutes > max_minutes
        ):
            raise ValueError(
                f"{self.stage.value} training budget exceeds the frozen fit-first ceiling"
            )
        if self.stage is FitStage.R1:
            if self.training.max_epochs is None or self.training.max_epochs > 100:
                raise ValueError("R1 requires an epoch ceiling no greater than 100")
        elif self.training.max_epochs is not None:
            raise ValueError("F0 and S1 budgets are update/time bounded, not epoch bounded")
        expected_sampling = (
            ScheduleSampling.FIXED
            if self.stage is FitStage.F0
            else ScheduleSampling.DETERMINISTIC_SHUFFLE
        )
        if self.episode_schedule.sampling is not expected_sampling:
            raise ValueError(
                f"{self.stage.value} requires {expected_sampling.value} episode sampling"
            )
        if self.stage is FitStage.F0 and self.episode_schedule.episode_count != 1:
            raise ValueError("F0 is exactly one fixed episode")
        if not any(baseline.role is BaselineRole.TRIVIAL for baseline in self.baselines):
            raise ValueError("fit experiments require at least one trivial baseline")
        if set(self.required_artifacts) != set(RequiredArtifact):
            raise ValueError("fit experiments must require the complete immutable artifact set")

        label_contracts = {"tabul", "tabufl", "tabu.cell.base"}
        if self.contract_id == "tabu.cell.base":
            if self.contract_version != "0.2.0":
                raise ValueError("TabUBase fit experiments require contract_version=0.2.0")
            if self.semantic.profile_id not in {
                "completion.artificial_mask.v1",
                "supervised.label_broadcast.v1",
            }:
                raise ValueError("TabUBase requires an explicit v0.2 profile_id")
            if self.semantic.profile_id == "supervised.label_broadcast.v1" and len(self.semantic.label_columns) != 1:
                raise ValueError("TabUBase supervised profile requires exactly one label column")
        elif self.contract_id in label_contracts - {"tabu.cell.base"}:
            if not self.semantic.label_columns:
                raise ValueError(f"{self.contract_id} requires label_columns")
            if self.semantic.label_address_plan is None:
                raise ValueError(f"{self.contract_id} requires an explicit label address plan")
        elif self.semantic.label_columns or self.semantic.label_address_plan is not None:
            raise ValueError("label semantic fields are only valid for tabul and tabufl")
        if self.contract_id == "tabu4graph":
            if self.semantic.target_feature is None or not isinstance(
                self.split, GraphSplitManifest
            ):
                raise ValueError("tabu4graph requires target_feature and GraphSplitManifest")
            if self.semantic.graph_unit_receiver_plan is None:
                raise ValueError("tabu4graph requires an explicit graph Unit receiver plan")
        elif (
            self.semantic.target_feature is not None
            or self.semantic.graph_unit_receiver_plan is not None
        ):
            raise ValueError("graph semantic fields are only valid for tabu4graph")
        if self.contract_id == "tabu4rec":
            if self.semantic.response_family is None or not isinstance(
                self.split, InteractionSplitManifest
            ):
                raise ValueError("tabu4rec requires response_family and InteractionSplitManifest")
            if self.semantic.recommendation_address_plan is None:
                raise ValueError("tabu4rec requires an explicit recommendation address plan")
            axis_plan = (
                self.semantic.recommendation_address_plan
                is RecommendationAddressPlan.AXIS_ADDRESS_BOOTSTRAP_V1
            )
            axis_fields = (
                self.semantic.rec_axis_summary_dim,
                self.semantic.rec_matched_residual_scale,
            )
            if axis_plan and any(value is None for value in axis_fields):
                raise ValueError("axis-address Rec requires dimension and residual scale")
            if not axis_plan and any(value is not None for value in axis_fields):
                raise ValueError("matched-UF Rec cannot declare axis-address fields")
        elif (
            self.semantic.response_family is not None
            or self.semantic.recommendation_address_plan is not None
            or self.semantic.rec_axis_summary_dim is not None
            or self.semantic.rec_matched_residual_scale is not None
        ):
            raise ValueError("recommendation semantic fields are only valid for tabu4rec")
        augmented_contracts = {"tabuf", "tabul", "tabufl", "tabu4rec"}
        if self.contract_id in augmented_contracts:
            if self.semantic.augmented_readout_geometry is None:
                raise ValueError(
                    f"{self.contract_id} requires an explicit augmented_readout_geometry"
                )
        elif self.semantic.augmented_readout_geometry is not None:
            raise ValueError(
                "augmented_readout_geometry is only valid for tabuf, tabul, tabufl, and tabu4rec"
            )
        if self.contract_id not in {"tabu4graph", "tabu4rec"} and not isinstance(
            self.split, RowSplitManifest
        ):
            raise ValueError("tabular fit contracts require RowSplitManifest")
        expected_families = (
            {FitTargetFamily.LABEL}
            if self.contract_id == "tabul"
            else {FitTargetFamily.COMPLETION, FitTargetFamily.LABEL}
            if self.contract_id == "tabufl"
            else {FitTargetFamily.COMPLETION}
        )
        if self.contract_id == "tabu.cell.base" and self.semantic.profile_id == "supervised.label_broadcast.v1":
            expected_families = {FitTargetFamily.LABEL}
        elif self.contract_id == "tabu.cell.base":
            expected_families = {FitTargetFamily.COMPLETION}
        if set(self.target_families) != expected_families:
            raise ValueError(
                f"{self.contract_id} requires target families "
                + ", ".join(sorted(family.value for family in expected_families))
            )
        expected_origins = (
            {FitTargetOrigin.QUERY}
            if self.contract_id == "tabul"
            else {FitTargetOrigin.ARTIFICIAL_MASK, FitTargetOrigin.QUERY}
            if self.contract_id == "tabufl"
            else {FitTargetOrigin.ARTIFICIAL_MASK}
        )
        if self.contract_id == "tabu.cell.base" and self.semantic.profile_id == "supervised.label_broadcast.v1":
            expected_origins = {FitTargetOrigin.QUERY}
        elif self.contract_id == "tabu.cell.base":
            expected_origins = {FitTargetOrigin.ARTIFICIAL_MASK}
        if set(self.episode_schedule.target_origins) != expected_origins:
            raise ValueError(
                f"{self.contract_id} requires target origins "
                + ", ".join(sorted(origin.value for origin in expected_origins))
            )
        return self

    @property
    def spec_hash(self) -> str:
        return self.content_hash


class FitMetricKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class FitFamilyMetrics(EvidenceSchema):
    family: FitTargetFamily
    kind: FitMetricKind
    targets: int = Field(ge=0)
    scored_targets: int = Field(ge=0)
    initial_loss: float = Field(ge=0.0)
    final_loss: float = Field(ge=0.0)
    trivial_baseline_loss: float | None = Field(default=None, ge=0.0)
    mse: float | None = Field(default=None, ge=0.0)
    nrmse: float | None = Field(default=None, ge=0.0)
    accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    nll: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _typed_metrics_are_consistent(self) -> FitFamilyMetrics:
        if self.scored_targets > self.targets:
            raise ValueError("scored_targets cannot exceed targets")
        if self.kind is FitMetricKind.NUMERIC:
            if self.accuracy is not None or self.nll is not None:
                raise ValueError("numeric family metrics cannot contain accuracy or NLL")
        elif self.mse is not None or self.nrmse is not None:
            raise ValueError("categorical family metrics cannot contain MSE or NRMSE")
        return self


class FitCountStatus(StrEnum):
    READY = "ready"
    INVALID = "invalid"


class FitCountValidation(EvidenceSchema):
    status: FitCountStatus
    targets: int = Field(ge=0)
    scored_targets: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    required_coverage: float = Field(ge=0.0, le=1.0)
    reasons: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.status is FitCountStatus.READY


def validate_positive_fit_counts(
    *,
    targets: int,
    scored_targets: int,
    coverage: float,
    required_coverage: float = 1.0,
) -> FitCountValidation:
    """Fail closed when a nominal fit has no truth or incomplete scoring."""

    if type(targets) is not int or targets < 0:
        raise ValueError("targets must be a non-negative integer")
    if type(scored_targets) is not int or scored_targets < 0:
        raise ValueError("scored_targets must be a non-negative integer")
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("coverage must be finite and in [0, 1]")
    if not math.isfinite(required_coverage) or not 0.0 <= required_coverage <= 1.0:
        raise ValueError("required_coverage must be finite and in [0, 1]")

    reasons: list[str] = []
    if targets == 0:
        reasons.append("zero_targets")
    if scored_targets > targets:
        reasons.append("scored_targets_exceed_targets")
    expected_coverage = scored_targets / targets if targets else 0.0
    if not math.isclose(coverage, expected_coverage, rel_tol=0.0, abs_tol=1.0e-12):
        reasons.append("coverage_count_mismatch")
    if scored_targets != targets:
        reasons.append("not_all_targets_scored")
    if coverage + 1.0e-12 < required_coverage:
        reasons.append("coverage_below_required")
    status = FitCountStatus.INVALID if reasons else FitCountStatus.READY
    return FitCountValidation(
        status=status,
        targets=targets,
        scored_targets=scored_targets,
        coverage=coverage,
        required_coverage=required_coverage,
        reasons=tuple(reasons),
    )


class FitEvaluationBundle(EvidenceSchema):
    schema_version: Literal["tabu.fit-evaluation.v1"] = "tabu.fit-evaluation.v1"
    evaluation_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    stage: FitStage
    model_seed: int = Field(ge=0)
    targets: int = Field(ge=0)
    scored_targets: int = Field(ge=0)
    coverage: float = Field(ge=0.0, le=1.0)
    families: tuple[FitFamilyMetrics, ...] = Field(min_length=1)
    gradient_nonzero_by_step: int | None = Field(default=None, gt=0)
    gradient_group_nonzero_by_step: dict[str, int] = Field(default_factory=dict)
    gradient_group_max_norms: dict[str, float] = Field(default_factory=dict)
    mechanism_source_counts: dict[str, int] = Field(default_factory=dict)
    mechanism_active_target_counts: dict[str, int] = Field(default_factory=dict)
    mechanism_scored_target_count: int = Field(default=0, ge=0)
    mechanism_gradient_norms: dict[str, float] = Field(default_factory=dict)
    parameter_delta_norm: float = Field(ge=0.0)
    nonfinite_seen: bool = False
    checkpoint_reloaded: bool

    @model_validator(mode="after")
    def _aggregate_counts_match(self) -> FitEvaluationBundle:
        targets = sum(family.targets for family in self.families)
        scored = sum(family.scored_targets for family in self.families)
        if self.targets != targets or self.scored_targets != scored:
            raise ValueError("fit evaluation aggregate counts must equal family counts")
        allowed_groups = {"carrier", "tokenizer", "dynamics", "readout", "other"}
        unknown_steps = set(self.gradient_group_nonzero_by_step) - allowed_groups
        unknown_norms = set(self.gradient_group_max_norms) - allowed_groups
        if unknown_steps or unknown_norms:
            raise ValueError("fit evaluation contains an unknown gradient group")
        if any(step <= 0 for step in self.gradient_group_nonzero_by_step.values()):
            raise ValueError("gradient group first-nonzero steps must be positive")
        if any(
            not math.isfinite(norm) or norm < 0.0 for norm in self.gradient_group_max_norms.values()
        ):
            raise ValueError("gradient group norms must be finite and nonnegative")
        if any(not name or count < 0 for name, count in self.mechanism_source_counts.items()):
            raise ValueError("mechanism source counts require names and nonnegative counts")
        if any(
            not name or count < 0
            for name, count in self.mechanism_active_target_counts.items()
        ):
            raise ValueError(
                "mechanism active-target counts require names and nonnegative counts"
            )
        if any(
            count > self.mechanism_scored_target_count
            for count in self.mechanism_active_target_counts.values()
        ):
            raise ValueError(
                "mechanism active-target counts cannot exceed the mechanism scored-target count"
            )
        if any(
            not name or not math.isfinite(norm) or norm < 0.0
            for name, norm in self.mechanism_gradient_norms.items()
        ):
            raise ValueError("mechanism gradients require names and finite nonnegative norms")
        return self

    @property
    def count_validation(self) -> FitCountValidation:
        return validate_positive_fit_counts(
            targets=self.targets,
            scored_targets=self.scored_targets,
            coverage=self.coverage,
        )

    def require_positive_fit_ready(self) -> None:
        validation = self.count_validation
        if not validation.ready:
            raise ValueError("positive fit gate is invalid: " + ", ".join(validation.reasons))

    @property
    def evaluation_hash(self) -> str:
        return self.content_hash


def derive_attempt_id(*, run_id: str, attempt_nonce: str) -> str:
    """Derive a stable attempt id without reusing the stable semantic run id."""

    normalized_run_id = run_id.strip()
    if not normalized_run_id.startswith("run-"):
        raise ValueError("run_id must start with 'run-'")
    require_sha256(normalized_run_id.removeprefix("run-"), field_name="run_id")
    normalized_nonce = attempt_nonce.strip()
    if not normalized_nonce:
        raise ValueError("attempt_nonce cannot be empty")
    digest = canonical_hash(
        {
            "schema": "tabu.run-attempt-identity.v1",
            "run_id": normalized_run_id,
            "attempt_nonce": normalized_nonce,
        }
    )
    return f"attempt-{digest}"


class RunAttempt(EvidenceSchema):
    """Immutable terminal record for one non-overwriting execution attempt.

    ``run_id`` identifies the frozen semantic experiment.  A caller-selected,
    recorded nonce derives a separate ``attempt_id`` so reruns never overwrite
    an earlier success, failure, or invalid receipt.
    """

    schema_version: Literal["tabu.run-attempt.v1"] = "tabu.run-attempt.v1"
    run_id: str = Field(pattern=r"^run-[0-9a-f]{64}$")
    attempt_id: str = Field(pattern=r"^attempt-[0-9a-f]{64}$")
    attempt_nonce: str = Field(min_length=1)
    status: RunAttemptStatus
    started_at: datetime
    completed_at: datetime
    output_manifest: ArtifactRef
    receipt: ArtifactRef

    @field_validator("attempt_nonce")
    @classmethod
    def _normalize_nonce(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("attempt_nonce cannot be empty")
        return normalized

    @model_validator(mode="after")
    def _attempt_is_content_bound(self) -> RunAttempt:
        expected = derive_attempt_id(
            run_id=self.run_id,
            attempt_nonce=self.attempt_nonce,
        )
        if self.attempt_id != expected:
            raise ValueError("attempt_id does not match run_id and attempt_nonce")
        started = _as_utc(self.started_at, field_name="started_at")
        completed = _as_utc(self.completed_at, field_name="completed_at")
        if completed < started:
            raise ValueError("completed_at cannot precede started_at")
        if self.output_manifest.kind != "run_output_manifest":
            raise ValueError("output_manifest.kind must be run_output_manifest")
        if self.receipt.kind != "receipt":
            raise ValueError("receipt.kind must be receipt")
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "completed_at", completed)
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        attempt_nonce: str,
        status: RunAttemptStatus,
        started_at: datetime,
        completed_at: datetime,
        output_manifest: ArtifactRef,
        receipt: ArtifactRef,
    ) -> RunAttempt:
        return cls(
            run_id=run_id,
            attempt_id=derive_attempt_id(
                run_id=run_id,
                attempt_nonce=attempt_nonce,
            ),
            attempt_nonce=attempt_nonce,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            output_manifest=output_manifest,
            receipt=receipt,
        )

    @property
    def relative_output_directory(self) -> str:
        return f"{self.run_id}/{self.attempt_id}"

    def require_new(self, existing_attempt_ids: Iterable[str]) -> None:
        """Fail before a writer can reuse an immutable attempt directory."""

        if self.attempt_id in set(existing_attempt_ids):
            raise FileExistsError(
                f"run attempt already exists and cannot be overwritten: {self.attempt_id}"
            )

    @property
    def attempt_hash(self) -> str:
        return self.content_hash


def _as_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


__all__ = [
    "BaselineRole",
    "BaselineSpec",
    "CategoricalTerminal",
    "DatasetAdapterSpec",
    "DatasetOrigin",
    "DynamicsBlockKind",
    "DynamicsSemanticConfig",
    "EpisodeSchedule",
    "FitCountStatus",
    "FitCountValidation",
    "FitDatasetSpec",
    "FitDevice",
    "FitEvaluationBundle",
    "FitExecutionConfig",
    "FitExperimentSpec",
    "FitFamilyMetrics",
    "FitKillConditions",
    "FitMetricKind",
    "FitOptimizer",
    "FitPassGate",
    "FitSeedConfig",
    "FitStage",
    "FitTargetFamily",
    "FitTargetOrigin",
    "FitTrainingConfig",
    "ModelSemanticConfig",
    "NumericTerminal",
    "RedistributionPolicy",
    "ReferenceBackendConfig",
    "RequiredArtifact",
    "RunAttempt",
    "RunAttemptStatus",
    "ScheduleSampling",
    "derive_attempt_id",
    "validate_positive_fit_counts",
]
