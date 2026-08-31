"""Typed, immutable contracts for evolvable TabU pretraining programs.

The evolution layer intentionally treats every mutable research choice as a
versioned node.  A :class:`ProgramSnapshot` is only a set of references; the
repository resolver binds those references to content hashes before a run may
start.  Human-facing descriptions are excluded from semantic identity so a
prose correction does not invalidate training artifacts.
"""

from __future__ import annotations

import math
import re
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import canonical_hash, require_sha256
from tabu_lab.evidence import RunIdentity

_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_PORT = re.compile(r"^[a-z][a-z0-9_.-]*$")


class StrictModel(BaseModel):
    """Frozen, extra-forbidden base used by source and resolved manifests."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProgramLane(StrEnum):
    GROW = "grow"
    EVIDENCE = "evidence"


class EvidenceStatus(StrEnum):
    LOCAL_UNISSUED = "local_unissued"
    FROZEN_NOT_RUN = "frozen_not_run"
    EVIDENCE_CANDIDATE_UNREVIEWED = "evidence_candidate_unreviewed"


class NodeMaturity(StrEnum):
    DRAFT = "draft"
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    CANONICAL = "canonical"
    EXERCISE_ONLY = "exercise_only"


class EvolutionNodeKind(StrEnum):
    MODEL_CONTRACT = "model_contract"
    COMPONENT = "component"
    COMPONENT_GRAPH = "component_graph"
    GENERATOR = "generator"
    WORLD_MIXTURE = "world_mixture"
    SAMPLING_POLICY = "sampling_policy"
    OBJECTIVE_BUNDLE = "objective_bundle"
    TRAINING_RECIPE = "training_recipe"
    EVALUATION_PROTOCOL = "evaluation_protocol"
    STATE_PROJECTION = "state_projection"


class SamplingPolicyKind(StrEnum):
    FIXED = "fixed"
    PIECEWISE = "piecewise"
    ADAPTIVE = "adaptive"


class CompatibilityDisposition(StrEnum):
    REUSE_EXACT = "reuse_exact"
    RESCORE = "rescore"
    RERUN_INFERENCE = "rerun_inference"
    WARM_START_AVAILABLE = "warm_start_available"


class ImpactDisposition(StrEnum):
    UNCHANGED = "unchanged"
    REUSE_EXACT = "reuse_exact"
    RESCORE = "rescore"
    RERUN_INFERENCE = "rerun_inference"
    RETRAIN = "retrain"
    WARM_START_AVAILABLE = "warm_start_available"
    BLOCKED = "blocked"


class NodeRef(StrictModel):
    node_id: str
    version: str

    @field_validator("node_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("node_id must be a namespaced lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("version must be semantic-versioned")
        return value

    @property
    def ref(self) -> str:
        return f"{self.node_id}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> NodeRef:
        node_id, separator, version = value.rpartition("@")
        if not separator:
            raise ValueError("node ref must use node_id@version")
        return cls(node_id=node_id, version=version)


class ProgramRef(StrictModel):
    program_id: str
    version: str

    @field_validator("program_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("program_id must be a namespaced lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("program version must be semantic-versioned")
        return value

    @property
    def ref(self) -> str:
        return f"{self.program_id}@{self.version}"


class SourceBinding(StrictModel):
    """Repository-relative source or Python-symbol identity."""

    source_path: str
    sha256: str
    symbol_ref: str | None = None
    hash_mode: Literal["file", "python_symbol"] = "file"

    @field_validator("source_path")
    @classmethod
    def _relative_source(cls, value: str) -> str:
        normalized = value.removeprefix("./")
        if (
            not normalized
            or normalized.startswith(("/", "\\", "../"))
            or "\\" in normalized
            or "/../" in normalized
            or normalized == ".."
        ):
            raise ValueError("source_path must be repository-relative")
        return normalized

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source binding sha256")

    @model_validator(mode="after")
    def _symbol_mode_is_complete(self) -> SourceBinding:
        if self.hash_mode == "python_symbol" and not self.symbol_ref:
            raise ValueError("python_symbol bindings require symbol_ref")
        if self.hash_mode == "file" and self.symbol_ref is not None:
            raise ValueError("file bindings cannot declare symbol_ref")
        if self.symbol_ref is not None and ":" not in self.symbol_ref:
            raise ValueError("symbol_ref must use module:qualname")
        return self


class PortSpec(StrictModel):
    name: str
    interface_id: str
    required: bool = True

    @field_validator("name")
    @classmethod
    def _valid_port(cls, value: str) -> str:
        if _PORT.fullmatch(value) is None:
            raise ValueError("port names must be lowercase identifiers")
        return value

    @field_validator("interface_id")
    @classmethod
    def _valid_interface(cls, value: str) -> str:
        if not value.strip() or "@" not in value:
            raise ValueError("interface_id must be a versioned interface reference")
        return value


class GraphComponentInstance(StrictModel):
    instance_id: str
    component: NodeRef
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("instance_id")
    @classmethod
    def _valid_instance(cls, value: str) -> str:
        if _PORT.fullmatch(value) is None:
            raise ValueError("component instance_id must be a lowercase identifier")
        return value


class PortConnection(StrictModel):
    """One typed edge; ``$input`` and ``$output`` name graph boundaries."""

    source_instance: str
    source_port: str
    target_instance: str
    target_port: str

    @field_validator("source_instance", "target_instance")
    @classmethod
    def _valid_endpoint(cls, value: str) -> str:
        if value not in {"$input", "$output"} and _PORT.fullmatch(value) is None:
            raise ValueError("connection endpoints must be component IDs or graph boundaries")
        return value

    @field_validator("source_port", "target_port")
    @classmethod
    def _valid_port(cls, value: str) -> str:
        if _PORT.fullmatch(value) is None:
            raise ValueError("connection ports must be lowercase identifiers")
        return value

    @model_validator(mode="after")
    def _direction_is_valid(self) -> PortConnection:
        if self.source_instance == "$output" or self.target_instance == "$input":
            raise ValueError("graph boundary connection direction is reversed")
        if self.source_instance == "$input" and self.target_instance == "$output":
            raise ValueError("component graph cannot bypass every component")
        return self


class EvolutionNodeBase(StrictModel):
    schema_version: Literal["tabu.evolution-node.v1"] = "tabu.evolution-node.v1"
    kind: EvolutionNodeKind
    node_id: str
    version: str
    maturity: NodeMaturity
    description: str = ""

    @field_validator("node_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("node_id must be a namespaced lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("node version must be semantic-versioned")
        return value

    @property
    def ref(self) -> str:
        return f"{self.node_id}@{self.version}"

    def identity_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="python", exclude={"description"})

    @property
    def node_hash(self) -> str:
        return canonical_hash(self.identity_payload())

    def dependency_refs(self) -> tuple[NodeRef, ...]:
        return ()


class ModelContractNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.MODEL_CONTRACT] = EvolutionNodeKind.MODEL_CONTRACT
    contract_ref: str
    contract_hash: str
    source: SourceBinding
    episode_interface: Literal["tabu.evidence-episode@3"] = "tabu.evidence-episode@3"
    prediction_interface: Literal["tabu.prediction-bundle@1"] = "tabu.prediction-bundle@1"
    truth_interface: Literal["tabu.truth-sidecar@1"] = "tabu.truth-sidecar@1"
    executable: bool = True

    @field_validator("contract_hash")
    @classmethod
    def _valid_contract_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="contract_hash")


class ComponentNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.COMPONENT] = EvolutionNodeKind.COMPONENT
    component_ref: str
    inputs: tuple[PortSpec, ...]
    outputs: tuple[PortSpec, ...]
    implementation: SourceBinding
    truth_visible: Literal[False] = False

    @model_validator(mode="after")
    def _unique_ports(self) -> ComponentNode:
        for label, ports in (("input", self.inputs), ("output", self.outputs)):
            names = tuple(port.name for port in ports)
            if len(names) != len(set(names)):
                raise ValueError(f"component {label} ports must be unique")
        if not self.outputs:
            raise ValueError("component requires at least one output port")
        return self


class ComponentGraphNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.COMPONENT_GRAPH] = EvolutionNodeKind.COMPONENT_GRAPH
    model_contract: NodeRef
    external_inputs: tuple[PortSpec, ...]
    external_outputs: tuple[PortSpec, ...]
    components: tuple[GraphComponentInstance, ...]
    connections: tuple[PortConnection, ...]
    builder_id: str
    builder_options: dict[str, JsonValue] = Field(default_factory=dict)
    executable: bool = True

    @model_validator(mode="after")
    def _local_uniqueness(self) -> ComponentGraphNode:
        instance_ids = tuple(component.instance_id for component in self.components)
        if not instance_ids or len(instance_ids) != len(set(instance_ids)):
            raise ValueError("component graph instance IDs must be non-empty and unique")
        for label, ports in (
            ("external input", self.external_inputs),
            ("external output", self.external_outputs),
        ):
            names = tuple(port.name for port in ports)
            if not names or len(names) != len(set(names)):
                raise ValueError(f"component graph {label} ports must be non-empty and unique")
        return self

    def dependency_refs(self) -> tuple[NodeRef, ...]:
        return (self.model_contract, *(component.component for component in self.components))


class GeneratorNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.GENERATOR] = EvolutionNodeKind.GENERATOR
    generator_ref: str
    output_interface: Literal["tabu.evidence-truth-pair@1"] = "tabu.evidence-truth-pair@1"
    implementation: SourceBinding
    runtime_ref: str
    immutable_config: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("runtime_ref")
    @classmethod
    def _runtime_is_importable(cls, value: str) -> str:
        if ":" not in value:
            raise ValueError("runtime_ref must use module:qualname")
        return value

    @model_validator(mode="after")
    def _runtime_is_source_bound(self) -> GeneratorNode:
        if (
            self.implementation.hash_mode != "python_symbol"
            or self.implementation.symbol_ref != self.runtime_ref
        ):
            raise ValueError("generator runtime_ref must equal its bound Python symbol")
        return self


class WorldMixtureEntry(StrictModel):
    generator: NodeRef
    weight: float = Field(gt=0.0)

    @field_validator("weight")
    @classmethod
    def _finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("mixture weights must be finite")
        return value


class WorldMixtureNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.WORLD_MIXTURE] = EvolutionNodeKind.WORLD_MIXTURE
    entries: tuple[WorldMixtureEntry, ...]

    @model_validator(mode="after")
    def _normalized_unique_entries(self) -> WorldMixtureNode:
        refs = tuple(entry.generator.ref for entry in self.entries)
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("world mixture generator refs must be non-empty and unique")
        if not math.isclose(sum(entry.weight for entry in self.entries), 1.0, abs_tol=1.0e-9):
            raise ValueError("world mixture weights must sum to one")
        return self

    def dependency_refs(self) -> tuple[NodeRef, ...]:
        return tuple(entry.generator for entry in self.entries)


class SamplingPolicySegment(StrictModel):
    start_step: int = Field(ge=0)
    weights: dict[str, float]

    @field_validator("weights")
    @classmethod
    def _valid_weights(cls, values: dict[str, float]) -> dict[str, float]:
        if not values or any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("policy weights must be finite and positive")
        return values


class SamplingPolicyNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.SAMPLING_POLICY] = EvolutionNodeKind.SAMPLING_POLICY
    policy_kind: SamplingPolicyKind
    implementation: SourceBinding
    deterministic: bool
    serializable_state: bool
    state_interface: Literal["tabu.sampling-policy-state@1"] = "tabu.sampling-policy-state@1"
    segments: tuple[SamplingPolicySegment, ...] = ()
    adaptive_ema: float = Field(default=0.9, ge=0.0, lt=1.0)
    adaptive_temperature: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _valid_strategy(self) -> SamplingPolicyNode:
        if self.policy_kind is SamplingPolicyKind.PIECEWISE:
            starts = tuple(segment.start_step for segment in self.segments)
            if not starts or starts[0] != 0 or starts != tuple(sorted(set(starts))):
                raise ValueError("piecewise policy segments must start at zero and be ordered")
        elif self.segments:
            raise ValueError("only piecewise policies declare segments")
        return self


class ObjectiveTerm(StrictModel):
    objective_id: str
    weight: float = Field(gt=0.0)
    target_origin: Literal["artificial_mask", "query", "natural_missing"]

    @field_validator("objective_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("objective_id must be namespaced and lowercase")
        return value


class ObjectiveBundleNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.OBJECTIVE_BUNDLE] = EvolutionNodeKind.OBJECTIVE_BUNDLE
    objectives: tuple[ObjectiveTerm, ...]
    truth_entry: Literal["loss_only"] = "loss_only"

    @model_validator(mode="after")
    def _unique_objectives(self) -> ObjectiveBundleNode:
        ids = tuple(item.objective_id for item in self.objectives)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("objective bundle IDs must be non-empty and unique")
        return self


class TrainingRecipeNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.TRAINING_RECIPE] = EvolutionNodeKind.TRAINING_RECIPE
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)
    checkpoint_interval: int = Field(gt=0)
    scheduler: Literal["none", "step"] = "none"
    scheduler_step_size: int | None = Field(default=None, gt=0)
    scheduler_gamma: float | None = Field(default=None, gt=0.0, le=1.0)
    episode_options: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _scheduler_is_complete(self) -> TrainingRecipeNode:
        if self.scheduler == "step" and (
            self.scheduler_step_size is None or self.scheduler_gamma is None
        ):
            raise ValueError("step scheduler requires step_size and gamma")
        if self.scheduler == "none" and (
            self.scheduler_step_size is not None or self.scheduler_gamma is not None
        ):
            raise ValueError("scheduler parameters require scheduler=step")
        return self


class EvaluationProtocolNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.EVALUATION_PROTOCOL] = (
        EvolutionNodeKind.EVALUATION_PROTOCOL
    )
    prediction_interface: Literal["tabu.prediction-bundle@1"] = "tabu.prediction-bundle@1"
    prediction_compatibility_key: str
    split_authority: str
    modes: tuple[str, ...]
    metrics: tuple[str, ...]
    can_rescore_from_predictions: bool = True

    @model_validator(mode="after")
    def _nonempty_protocol(self) -> EvaluationProtocolNode:
        if not self.prediction_compatibility_key.strip() or not self.split_authority.strip():
            raise ValueError("evaluation compatibility and split authority cannot be blank")
        if not self.modes or not self.metrics:
            raise ValueError("evaluation protocol requires modes and metrics")
        return self


class StateProjectionNode(EvolutionNodeBase):
    kind: Literal[EvolutionNodeKind.STATE_PROJECTION] = EvolutionNodeKind.STATE_PROJECTION
    source_model: NodeRef
    source_graph: NodeRef
    target_model: NodeRef
    target_graph: NodeRef
    implementation: SourceBinding
    validation_ref: str
    verified: bool
    transfers_optimizer_state: Literal[False] = False
    transfers_evidence_status: Literal[False] = False

    @model_validator(mode="after")
    def _projection_is_executable(self) -> StateProjectionNode:
        if (
            self.implementation.hash_mode != "python_symbol"
            or self.implementation.symbol_ref is None
        ):
            raise ValueError("StateProjection requires a bound Python symbol")
        if "::" not in self.validation_ref:
            raise ValueError("StateProjection validation_ref must use path::test_name")
        return self

    def dependency_refs(self) -> tuple[NodeRef, ...]:
        return (
            self.source_model,
            self.source_graph,
            self.target_model,
            self.target_graph,
        )


EvolutionNode: TypeAlias = Annotated[
    ModelContractNode
    | ComponentNode
    | ComponentGraphNode
    | GeneratorNode
    | WorldMixtureNode
    | SamplingPolicyNode
    | ObjectiveBundleNode
    | TrainingRecipeNode
    | EvaluationProtocolNode
    | StateProjectionNode,
    Field(discriminator="kind"),
]


class CompatibilityEdge(StrictModel):
    schema_version: Literal["tabu.compatibility-edge.v1"] = "tabu.compatibility-edge.v1"
    edge_id: str
    version: str
    source: NodeRef
    target: NodeRef
    disposition: CompatibilityDisposition
    verifier: SourceBinding
    verified: bool
    constraints: dict[str, JsonValue] = Field(default_factory=dict)
    description: str = ""

    @field_validator("edge_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("edge_id must be namespaced and lowercase")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("edge version must be semantic-versioned")
        return value

    @property
    def ref(self) -> str:
        return f"{self.edge_id}@{self.version}"

    @property
    def edge_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python", exclude={"description"}))


class ProgramSnapshot(StrictModel):
    schema_version: Literal["tabu.program-snapshot.v1"] = "tabu.program-snapshot.v1"
    program_id: str
    version: str
    research_question: str
    lane: ProgramLane = ProgramLane.GROW
    evidence_status: EvidenceStatus = EvidenceStatus.LOCAL_UNISSUED
    model_contract: NodeRef
    component_graph: NodeRef
    world_mixture: NodeRef
    sampling_policy: NodeRef
    objective_bundle: NodeRef
    training_recipe: NodeRef
    evaluation_protocol: NodeRef
    state_projection: NodeRef | None = None
    parent: ProgramRef | None = None
    description: str = ""

    @field_validator("program_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("program_id must be namespaced and lowercase")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("program version must be semantic-versioned")
        return value

    @model_validator(mode="after")
    def _lane_status_matches(self) -> ProgramSnapshot:
        if (
            self.lane is ProgramLane.GROW
            and self.evidence_status is not EvidenceStatus.LOCAL_UNISSUED
        ):
            raise ValueError("grow snapshots must remain local_unissued")
        if (
            self.lane is ProgramLane.EVIDENCE
            and self.evidence_status is EvidenceStatus.LOCAL_UNISSUED
        ):
            raise ValueError("evidence snapshots require an evidence-lane status")
        return self

    @property
    def ref(self) -> str:
        return f"{self.program_id}@{self.version}"

    @property
    def program_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python", exclude={"description"}))

    def slot_refs(self) -> dict[str, NodeRef]:
        values = {
            "model_contract": self.model_contract,
            "component_graph": self.component_graph,
            "world_mixture": self.world_mixture,
            "sampling_policy": self.sampling_policy,
            "objective_bundle": self.objective_bundle,
            "training_recipe": self.training_recipe,
            "evaluation_protocol": self.evaluation_protocol,
        }
        if self.state_projection is not None:
            values["state_projection"] = self.state_projection
        return values


class ResolvedNodeRef(StrictModel):
    node_id: str
    version: str
    kind: EvolutionNodeKind
    node_hash: str

    @field_validator("node_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="node_hash")

    @property
    def ref(self) -> str:
        return f"{self.node_id}@{self.version}"


class ResolvedProgramSnapshot(StrictModel):
    schema_version: Literal["tabu.resolved-program-snapshot.v1"] = (
        "tabu.resolved-program-snapshot.v1"
    )
    program_id: str
    version: str
    research_question: str
    lane: ProgramLane
    evidence_status: EvidenceStatus
    source_program_hash: str
    slots: dict[str, ResolvedNodeRef]
    dependency_closure: tuple[ResolvedNodeRef, ...]
    manifest_closure_hash: str
    snapshot_hash: str

    @field_validator("source_program_hash", "manifest_closure_hash", "snapshot_hash")
    @classmethod
    def _valid_hashes(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _snapshot_hash_matches(self) -> ResolvedProgramSnapshot:
        payload = self.model_dump(mode="python", exclude={"snapshot_hash"})
        if canonical_hash(payload) != self.snapshot_hash:
            raise ValueError("snapshot_hash does not match resolved program content")
        return self


class FrozenProgram(StrictModel):
    schema_version: Literal["tabu.frozen-program.v1"] = "tabu.frozen-program.v1"
    lane: Literal[ProgramLane.EVIDENCE] = ProgramLane.EVIDENCE
    evidence_status: Literal[EvidenceStatus.FROZEN_NOT_RUN] = EvidenceStatus.FROZEN_NOT_RUN
    resolved: ResolvedProgramSnapshot
    freeze_hash: str

    @field_validator("freeze_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="freeze_hash")

    @model_validator(mode="after")
    def _freeze_hash_matches(self) -> FrozenProgram:
        payload = self.model_dump(mode="python", exclude={"freeze_hash"})
        if canonical_hash(payload) != self.freeze_hash:
            raise ValueError("freeze_hash does not match frozen program")
        return self


class SnapshotChange(StrictModel):
    slot: str
    source: ResolvedNodeRef | None
    target: ResolvedNodeRef | None


class ImpactAction(StrictModel):
    object_kind: str
    disposition: ImpactDisposition
    reason: str


class ImpactReport(StrictModel):
    schema_version: Literal["tabu.impact-report.v1"] = "tabu.impact-report.v1"
    source_snapshot_hash: str
    target_snapshot_hash: str
    changes: tuple[SnapshotChange, ...]
    actions: tuple[ImpactAction, ...]
    report_hash: str

    @field_validator("source_snapshot_hash", "target_snapshot_hash", "report_hash")
    @classmethod
    def _valid_hashes(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _report_hash_matches(self) -> ImpactReport:
        payload = self.model_dump(mode="python", exclude={"report_hash"})
        if canonical_hash(payload) != self.report_hash:
            raise ValueError("report_hash does not match impact report")
        return self


class ManifestLock(StrictModel):
    schema_version: Literal["tabu.evolution-manifest-lock.v1"] = (
        "tabu.evolution-manifest-lock.v1"
    )
    nodes: dict[str, str]
    edges: dict[str, str]
    programs: dict[str, str]

    @field_validator("nodes", "edges", "programs")
    @classmethod
    def _valid_mapping(cls, values: dict[str, str]) -> dict[str, str]:
        return {
            key: require_sha256(value, field_name=f"manifest lock {key}")
            for key, value in sorted(values.items())
        }

    @property
    def lock_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class ProgramRunStatus(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


class ProgramInitializationMode(StrEnum):
    COLD = "cold"
    WARM_START = "warm_start"


class ProgramCheckpointKind(StrEnum):
    PROGRAM_FULL_STATE = "program_full_state"
    WEIGHTS_ONLY = "weights_only"


class ProgramInitialization(StrictModel):
    schema_version: Literal["tabu.program-initialization.v1"] = (
        "tabu.program-initialization.v1"
    )
    mode: ProgramInitializationMode = ProgramInitializationMode.COLD
    projection_ref: str | None = None
    source_checkpoint_kind: ProgramCheckpointKind | None = None
    source_snapshot_hash: str | None = None
    source_run_identity_hash: str | None = None
    source_checkpoint_sha256: str | None = None
    source_lane: ProgramLane | None = None
    source_evidence_status: EvidenceStatus | None = None

    @field_validator(
        "source_snapshot_hash",
        "source_run_identity_hash",
        "source_checkpoint_sha256",
    )
    @classmethod
    def _valid_optional_hash(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _mode_is_complete(self) -> ProgramInitialization:
        core_source_values = (
            self.projection_ref,
            self.source_checkpoint_kind,
            self.source_snapshot_hash,
            self.source_checkpoint_sha256,
        )
        source_provenance_values = (
            self.source_run_identity_hash,
            self.source_lane,
            self.source_evidence_status,
        )
        if self.mode is ProgramInitializationMode.COLD:
            if any(
                value is not None
                for value in (*core_source_values, *source_provenance_values)
            ):
                raise ValueError("cold initialization cannot bind a source checkpoint")
            return self
        if any(value is None for value in core_source_values):
            raise ValueError("warm-start initialization requires complete source identity")
        assert self.projection_ref is not None
        if "@" not in self.projection_ref:
            raise ValueError("warm-start projection_ref must be versioned")
        if self.source_checkpoint_kind is ProgramCheckpointKind.PROGRAM_FULL_STATE:
            if any(
                value is None
                for value in source_provenance_values
            ):
                raise ValueError("full-state warm start requires source run identity")
        elif any(
            value is not None
            for value in source_provenance_values
        ):
            raise ValueError("weights-only warm start cannot claim source run identity")
        return self


class ProgramArtifact(StrictModel):
    name: str
    sha256: str
    size_bytes: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="artifact sha256")


class ProgramRunReceipt(StrictModel):
    schema_version: Literal["tabu.program-run-receipt.v1"] = "tabu.program-run-receipt.v1"
    lane: ProgramLane
    evidence_status: EvidenceStatus
    status: ProgramRunStatus
    resolved_snapshot: ResolvedProgramSnapshot
    run_identity: RunIdentity
    initialization: ProgramInitialization
    training_config: dict[str, JsonValue]
    execution_config: dict[str, JsonValue]
    snapshot_hash: str
    run_identity_hash: str
    step: int = Field(ge=0)
    target_steps: int = Field(gt=0)
    policy_state_hash: str
    artifacts: tuple[ProgramArtifact, ...]
    receipt_hash: str

    @field_validator(
        "snapshot_hash", "run_identity_hash", "policy_state_hash", "receipt_hash"
    )
    @classmethod
    def _valid_hashes(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _receipt_is_consistent(self) -> ProgramRunReceipt:
        if (
            self.lane is ProgramLane.GROW
            and self.evidence_status is not EvidenceStatus.LOCAL_UNISSUED
        ):
            raise ValueError("grow run receipt cannot claim evidence status")
        if (
            self.lane is ProgramLane.EVIDENCE
            and self.evidence_status is not EvidenceStatus.EVIDENCE_CANDIDATE_UNREVIEWED
        ):
            raise ValueError("evidence run receipt must remain an unreviewed candidate")
        if self.status is ProgramRunStatus.COMPLETED and self.step != self.target_steps:
            raise ValueError("completed receipt must reach target_steps")
        if self.snapshot_hash != self.resolved_snapshot.snapshot_hash:
            raise ValueError("receipt snapshot wrapper does not match resolved snapshot")
        if self.run_identity_hash != self.run_identity.identity_hash:
            raise ValueError("receipt RunIdentity wrapper does not match RunIdentity")
        if self.run_identity.spec_hash != self.snapshot_hash:
            raise ValueError("RunIdentity spec hash does not match resolved snapshot")
        if canonical_hash(self.training_config) != self.run_identity.training_config_hash:
            raise ValueError("receipt training config does not match RunIdentity")
        if canonical_hash(self.training_config.get("initialization")) != canonical_hash(
            self.initialization.model_dump(mode="json")
        ):
            raise ValueError("receipt initialization does not match training config")
        if canonical_hash(self.execution_config) != self.run_identity.execution_config_hash:
            raise ValueError("receipt execution config does not match RunIdentity")
        payload = self.model_dump(mode="python", exclude={"receipt_hash"})
        if canonical_hash(payload) != self.receipt_hash:
            raise ValueError("receipt_hash does not match program receipt")
        return self


__all__ = [
    "CompatibilityDisposition",
    "CompatibilityEdge",
    "ComponentGraphNode",
    "ComponentNode",
    "EvaluationProtocolNode",
    "EvidenceStatus",
    "EvolutionNode",
    "EvolutionNodeBase",
    "EvolutionNodeKind",
    "FrozenProgram",
    "GeneratorNode",
    "GraphComponentInstance",
    "ImpactAction",
    "ImpactDisposition",
    "ImpactReport",
    "ManifestLock",
    "ModelContractNode",
    "NodeMaturity",
    "NodeRef",
    "ObjectiveBundleNode",
    "ObjectiveTerm",
    "PortConnection",
    "PortSpec",
    "ProgramArtifact",
    "ProgramCheckpointKind",
    "ProgramInitialization",
    "ProgramInitializationMode",
    "ProgramLane",
    "ProgramRef",
    "ProgramRunReceipt",
    "ProgramRunStatus",
    "ProgramSnapshot",
    "ResolvedNodeRef",
    "ResolvedProgramSnapshot",
    "SamplingPolicyKind",
    "SamplingPolicyNode",
    "SamplingPolicySegment",
    "SnapshotChange",
    "SourceBinding",
    "StateProjectionNode",
    "StrictModel",
    "TrainingRecipeNode",
    "WorldMixtureEntry",
    "WorldMixtureNode",
]
