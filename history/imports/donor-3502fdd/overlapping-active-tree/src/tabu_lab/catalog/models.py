"""Typed, public-safe records for the Git-native TabU research catalog.

The catalog is deliberately an index, not a second source of truth.  Canonical
manifests live in their domain directories; :class:`CatalogIndex` records their
stable identities, validated public payloads, and typed lineage.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts.canonical import canonical_hash, require_sha256
from tabu_lab.evaluation.foundry.contracts import (
    AdapterKind,
    ComparisonReport,
    EvalResult,
    EvalSuiteSpec,
    EvaluationStatus,
    ProducerProvenance,
    comparison_publication_eligible,
    derive_comparison_summary,
)
from tabu_lab.evidence.public_safety import (
    contains_absolute_local_path,
    contains_local_file_uri,
    is_sensitive_public_key,
)
from tabu_lab.evidence.schemas import EvidenceSchema, Receipt, ReceiptStatus

from .source_revision import CatalogSourceRevision

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$"
_SEMVER_PATTERN = r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$"
def _walk_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        strings: list[str] = []
        for key, item in value.items():
            strings.extend(_walk_strings(key))
            strings.extend(_walk_strings(item))
        return tuple(strings)
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_walk_strings(item))
        return tuple(strings)
    return ()


def require_public_string(value: str, *, field_name: str) -> str:
    """Reject host-private path disclosure while preserving public URIs."""

    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    if contains_local_file_uri(normalized):
        raise ValueError(f"{field_name} cannot expose a local file URI")
    if contains_absolute_local_path(normalized):
        raise ValueError(f"{field_name} cannot expose an absolute local path")
    return normalized


def require_relative_source_path(value: str) -> str:
    normalized = require_public_string(value, field_name="source_path").replace("\\", "/")
    if normalized.startswith("../") or "/../" in normalized or normalized == "..":
        raise ValueError("source_path must stay within the catalog repository")
    return normalized.removeprefix("./")


class PublicEvidenceSchema(EvidenceSchema):
    """Evidence schema with a fail-closed public path disclosure check."""

    @model_validator(mode="after")
    def _contains_no_private_paths(self) -> PublicEvidenceSchema:
        payload = self.model_dump(mode="python", by_alias=False)
        stack: list[Any] = [payload]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, item in value.items():
                    if is_sensitive_public_key(key):
                        raise ValueError(
                            f"public catalog objects cannot expose sensitive field {key!r}"
                        )
                    stack.append(item)
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
        for value in _walk_strings(payload):
            if contains_absolute_local_path(value):
                raise ValueError("public catalog objects cannot expose absolute local paths")
        return self


class CatalogObjectKind(StrEnum):
    MODEL_CONTRACT = "model_contract"
    EXPERIMENT = "experiment"
    RUN = "run"
    RUN_ATTEMPT = "run_attempt"
    RECEIPT = "receipt"
    MODEL_ARTIFACT = "model_artifact"
    DATASET_SNAPSHOT = "dataset_snapshot"
    EVAL_SUITE = "eval_suite"
    EVAL_RESULT = "eval_result"
    EVAL_COMPARISON = "eval_comparison"
    REVIEW = "review"
    CLAIM = "claim"
    VERIFICATION_SUITE = "verification_suite"
    VERIFICATION_RESULT = "verification_result"


class LineageRelation(StrEnum):
    IMPLEMENTS = "implements"
    USES_DATA = "uses_data"
    PRODUCED = "produced"
    EVALUATED_BY = "evaluated_by"
    SUPPORTS = "supports"
    RESUMES_FROM = "resumes_from"
    SUPERSEDES = "supersedes"
    VERIFIED_BY = "verified_by"


class ExperimentStatus(StrEnum):
    DRAFT = "draft"
    PREREGISTERED = "preregistered"
    RUNNABLE = "runnable"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"
    REVISED = "revised"
    REVIEWED = "reviewed"


_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.DRAFT: frozenset({ExperimentStatus.PREREGISTERED}),
    ExperimentStatus.PREREGISTERED: frozenset(
        {ExperimentStatus.RUNNABLE, ExperimentStatus.REVISED}
    ),
    ExperimentStatus.RUNNABLE: frozenset({ExperimentStatus.RUNNING, ExperimentStatus.REVISED}),
    ExperimentStatus.RUNNING: frozenset(
        {
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.KILLED,
            ExperimentStatus.REVISED,
        }
    ),
    ExperimentStatus.SUCCEEDED: frozenset(
        {ExperimentStatus.REVIEWED, ExperimentStatus.REVISED}
    ),
    ExperimentStatus.FAILED: frozenset({ExperimentStatus.REVIEWED, ExperimentStatus.REVISED}),
    ExperimentStatus.KILLED: frozenset({ExperimentStatus.REVIEWED, ExperimentStatus.REVISED}),
    ExperimentStatus.REVISED: frozenset(
        {ExperimentStatus.PREREGISTERED, ExperimentStatus.REVIEWED}
    ),
    ExperimentStatus.REVIEWED: frozenset({ExperimentStatus.REVISED}),
}


class RunStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"


_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PLANNED: frozenset({RunStatus.RUNNING}),
    RunStatus.RUNNING: frozenset(
        {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.KILLED}
    ),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.KILLED: frozenset(),
}


class RunAttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    KILLED = "killed"
    INVALID = "invalid"


class FailureCategory(StrEnum):
    MODEL = "model"
    DATA = "data"
    EVALUATOR = "evaluator"
    ARTIFACT = "artifact"
    INFRASTRUCTURE = "infrastructure"
    BUDGET = "budget"
    KILL_CONDITION = "kill-condition"


class ModelArtifactStatus(StrEnum):
    PRODUCED = "produced"
    VERIFIED = "verified"
    RELEASED = "released"
    RETRACTED = "retracted"


_ARTIFACT_TRANSITIONS: dict[ModelArtifactStatus, frozenset[ModelArtifactStatus]] = {
    ModelArtifactStatus.PRODUCED: frozenset(
        {ModelArtifactStatus.VERIFIED, ModelArtifactStatus.RETRACTED}
    ),
    ModelArtifactStatus.VERIFIED: frozenset(
        {ModelArtifactStatus.RELEASED, ModelArtifactStatus.RETRACTED}
    ),
    ModelArtifactStatus.RELEASED: frozenset({ModelArtifactStatus.RETRACTED}),
    ModelArtifactStatus.RETRACTED: frozenset(),
}


class ReviewDecision(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    REJECTED = "rejected"


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class ClaimMaturity(StrEnum):
    PROPOSAL = "proposal"
    F0_FIT = "f0_fit"
    MICROBENCHMARK = "microbenchmark"
    FORMAL_BENCHMARK = "formal_benchmark"
    SUPPORTED_MODEL = "supported_model"
    FOUNDATION_MODEL = "foundation_model"


class EvidencePointer(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-evidence-pointer.v1"] = (
        "tabu.catalog-evidence-pointer.v1"
    )
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str | None = None

    @field_validator("uri")
    @classmethod
    def _public_uri(cls, value: str) -> str:
        return require_public_string(value, field_name="uri")

    @field_validator("sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        return require_sha256(value)


class ObjectRef(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-object-ref.v1"] = "tabu.catalog-object-ref.v1"
    kind: CatalogObjectKind
    object_id: str = Field(pattern=_ID_PATTERN)
    evidence_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )

    @field_validator("evidence_sha256")
    @classmethod
    def _optional_evidence_hash(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="evidence_sha256")


class StatusEvent(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-status-event.v1"] = "tabu.catalog-status-event.v1"
    status: str = Field(min_length=1)
    evidence_hashes: tuple[str, ...] = ()
    note: str | None = None

    @field_validator("evidence_hashes")
    @classmethod
    def _hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(require_sha256(value, field_name="evidence_hash") for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("status event evidence hashes must be unique")
        return normalized


def _validate_history(
    history: tuple[StatusEvent, ...],
    *,
    declared: StrEnum,
    enum_type: type[StrEnum],
    initial: StrEnum,
    transitions: dict[StrEnum, frozenset[StrEnum]],
) -> None:
    if not history:
        raise ValueError("status_history cannot be empty")
    parsed = tuple(enum_type(event.status) for event in history)
    if parsed[0] is not initial:
        raise ValueError(f"status_history must start at {initial.value}")
    for before, after in pairwise(parsed):
        if after not in transitions[before]:
            raise ValueError(f"illegal status transition: {before.value} -> {after.value}")
    if parsed[-1] is not declared:
        raise ValueError("declared status must match the final status_history event")


class ExperimentRecord(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-experiment.v1"] = "tabu.catalog-experiment.v1"
    experiment_id: str = Field(pattern=_ID_PATTERN)
    contract_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    hypothesis: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    status: ExperimentStatus
    status_history: tuple[StatusEvent, ...] = Field(min_length=1)
    preregistration: EvidencePointer | None = None
    preregistration_review: EvidencePointer | None = None
    source_identity: EvidencePointer | None = None
    dataset_snapshot_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    supersedes_experiment_ids: tuple[str, ...] = ()
    revision_rationale: str | None = Field(default=None, min_length=1)

    @field_validator(
        "dataset_snapshot_ids", "run_ids", "review_ids", "supersedes_experiment_ids"
    )
    @classmethod
    def _unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("record references must be unique")
        return values

    @field_validator("revision_rationale")
    @classmethod
    def _nonblank_revision_rationale(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("revision_rationale cannot be blank")
        return value

    @model_validator(mode="after")
    def _state_is_evidence_gated(self) -> ExperimentRecord:
        if self.experiment_id in self.supersedes_experiment_ids:
            raise ValueError("an experiment cannot supersede itself")
        if bool(self.supersedes_experiment_ids) != (self.revision_rationale is not None):
            raise ValueError(
                "supersedes_experiment_ids and revision_rationale must be declared together"
            )
        if self.supersedes_experiment_ids and self.preregistration is None:
            raise ValueError("a superseding experiment must bind its preregistration")
        _validate_history(
            self.status_history,
            declared=self.status,
            enum_type=ExperimentStatus,
            initial=ExperimentStatus.DRAFT,
            transitions=_EXPERIMENT_TRANSITIONS,
        )
        if self.status is not ExperimentStatus.DRAFT and (
            self.preregistration is None or self.preregistration_review is None
        ):
            raise ValueError(
                "preregistered and later experiments require a preregistration and "
                "independent review evidence"
            )
        if self.status is not ExperimentStatus.DRAFT:
            assert self.preregistration is not None
            assert self.preregistration_review is not None
            preregistered_events = tuple(
                event
                for event in self.status_history
                if event.status == ExperimentStatus.PREREGISTERED.value
            )
            required_hashes = {
                self.preregistration.sha256,
                self.preregistration_review.sha256,
            }
            if not preregistered_events or not required_hashes.issubset(
                preregistered_events[-1].evidence_hashes
            ):
                raise ValueError(
                    "preregistered promotion event must bind preregistration and review hashes"
                )
            if not self.review_ids:
                raise ValueError(
                    "preregistered and later experiments require a cataloged review record"
                )
        if self.status in {
            ExperimentStatus.RUNNABLE,
            ExperimentStatus.RUNNING,
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.KILLED,
            ExperimentStatus.REVIEWED,
        } and self.source_identity is None:
            raise ValueError("runnable and later experiments require a reviewed source identity")
        if self.status in {
            ExperimentStatus.RUNNABLE,
            ExperimentStatus.RUNNING,
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.KILLED,
            ExperimentStatus.REVIEWED,
        }:
            assert self.source_identity is not None
            runnable_events = tuple(
                event
                for event in self.status_history
                if event.status == ExperimentStatus.RUNNABLE.value
            )
            if not runnable_events or self.source_identity.sha256 not in (
                runnable_events[-1].evidence_hashes
            ):
                raise ValueError("runnable promotion event must bind the source identity hash")
        if self.status in {
            ExperimentStatus.RUNNING,
            ExperimentStatus.SUCCEEDED,
            ExperimentStatus.FAILED,
            ExperimentStatus.KILLED,
            ExperimentStatus.REVIEWED,
        } and not self.run_ids:
            raise ValueError("running and terminal experiments require at least one run")
        if self.status is ExperimentStatus.REVIEWED and not self.review_ids:
            raise ValueError("reviewed experiments require an independent review record")
        return self


class RunRecord(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-run.v1"] = "tabu.catalog-run.v1"
    run_id: str = Field(pattern=_ID_PATTERN)
    experiment_id: str = Field(pattern=_ID_PATTERN)
    status: RunStatus
    status_history: tuple[StatusEvent, ...] = Field(min_length=1)
    attempt_ids: tuple[str, ...] = ()
    receipt: EvidencePointer | None = None
    failure_category: FailureCategory | None = None
    artifact_ids: tuple[str, ...] = ()
    resumes_from_run_id: str | None = None

    @field_validator("attempt_ids", "artifact_ids")
    @classmethod
    def _unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("run references must be unique")
        return values

    @model_validator(mode="after")
    def _terminal_run_has_receipt(self) -> RunRecord:
        _validate_history(
            self.status_history,
            declared=self.status,
            enum_type=RunStatus,
            initial=RunStatus.PLANNED,
            transitions=_RUN_TRANSITIONS,
        )
        terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.KILLED}
        if self.status in terminal and self.receipt is None:
            raise ValueError("terminal runs require an immutable receipt")
        if self.status not in terminal and self.receipt is not None:
            raise ValueError("nonterminal runs cannot attach a terminal receipt")
        if self.status in {RunStatus.FAILED, RunStatus.KILLED} and self.failure_category is None:
            raise ValueError("failed or killed runs require a failure category")
        if self.status not in {RunStatus.FAILED, RunStatus.KILLED} and (
            self.failure_category is not None
        ):
            raise ValueError("failure_category is only valid for failed or killed runs")
        return self


class RunAttemptRecord(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-run-attempt.v1"] = "tabu.catalog-run-attempt.v1"
    attempt_id: str = Field(pattern=_ID_PATTERN)
    run_id: str = Field(pattern=_ID_PATTERN)
    status: RunAttemptStatus
    receipt: EvidencePointer
    failure_category: FailureCategory | None = None

    @model_validator(mode="after")
    def _failure_is_classified(self) -> RunAttemptRecord:
        failed = self.status in {
            RunAttemptStatus.FAILED,
            RunAttemptStatus.KILLED,
            RunAttemptStatus.INVALID,
        }
        if failed and self.failure_category is None:
            raise ValueError("non-successful attempts require a failure category")
        if not failed and self.failure_category is not None:
            raise ValueError("successful attempts cannot declare a failure category")
        return self


class ArtifactStatusEvent(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-artifact-status-event.v1"] = (
        "tabu.catalog-artifact-status-event.v1"
    )
    status: ModelArtifactStatus
    evidence: EvidencePointer
    note: str | None = None


class ModelArtifact(PublicEvidenceSchema):
    schema_version: Literal["tabu.model-artifact.v2"] = "tabu.model-artifact.v2"
    artifact_id: str = Field(pattern=_ID_PATTERN)
    contract_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    contract_version: str = Field(pattern=_SEMVER_PATTERN)
    producer_run_id: str = Field(pattern=_ID_PATTERN)
    producer_receipt: EvidencePointer
    checkpoint: EvidencePointer
    checkpoint_format: str = Field(min_length=1)
    checkpoint_schema_version: str = Field(min_length=1)
    model_state_schema_version: str = Field(min_length=1)
    model_spec: EvidencePointer
    semantic_config: EvidencePointer
    compiler_manifest: EvidencePointer
    license_id: str = Field(min_length=1)
    parent_artifact_ids: tuple[str, ...] = ()
    status: ModelArtifactStatus = ModelArtifactStatus.PRODUCED
    status_history: tuple[ArtifactStatusEvent, ...] = Field(min_length=1)
    evaluation_result_ids: tuple[str, ...] = ()
    review_ids: tuple[str, ...] = ()
    retraction_reason: str | None = None

    @field_validator("parent_artifact_ids", "evaluation_result_ids", "review_ids")
    @classmethod
    def _unique_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("artifact references must be unique")
        return values

    @model_validator(mode="after")
    def _artifact_lifecycle(self) -> ModelArtifact:
        statuses = tuple(event.status for event in self.status_history)
        if statuses[0] is not ModelArtifactStatus.PRODUCED:
            raise ValueError("artifact status_history must start at produced")
        for before, after in pairwise(statuses):
            if after not in _ARTIFACT_TRANSITIONS[before]:
                raise ValueError(
                    f"illegal artifact status transition: {before.value} -> {after.value}"
                )
        if statuses[-1] is not self.status:
            raise ValueError("artifact status must match final status_history event")
        if self.status in {ModelArtifactStatus.VERIFIED, ModelArtifactStatus.RELEASED} and (
            len(self.status_history) < 2
        ):
            raise ValueError("verified or released artifacts require verification evidence")
        if self.status is ModelArtifactStatus.RELEASED and not self.review_ids:
            raise ValueError("released artifacts require independent review")
        if self.status is ModelArtifactStatus.RETRACTED and not self.retraction_reason:
            raise ValueError("retracted artifacts require a reason")
        if self.status is not ModelArtifactStatus.RETRACTED and self.retraction_reason is not None:
            raise ValueError("retraction_reason is only valid for retracted artifacts")
        return self


class DatasetAdapter(PublicEvidenceSchema):
    schema_version: Literal["tabu.dataset-adapter.v1"] = "tabu.dataset-adapter.v1"
    adapter_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    adapter_version: str = Field(pattern=_SEMVER_PATTERN)


class DatasetAuthorityStatus(StrEnum):
    """Review state of the authority that produced an evaluation snapshot."""

    SELF_CONSISTENT_UNREVIEWED = "self_consistent_unreviewed"
    REVIEWED = "reviewed"


class DatasetSnapshotSpec(PublicEvidenceSchema):
    schema_version: Literal[
        "tabu.dataset-snapshot.v2",
        "tabu.dataset-snapshot.v3",
    ] = "tabu.dataset-snapshot.v2"
    dataset_snapshot_id: str = Field(pattern=_ID_PATTERN)
    dataset_id: str = Field(pattern=_ID_PATTERN)
    source_uri: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license_id: str = Field(min_length=1)
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fit_partition: str = Field(min_length=1)
    adapter: DatasetAdapter
    split_before_compile: Literal[True] = True
    episode_recipe_hashes: tuple[str, ...] = ()
    evaluation_scenario_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    truth_sidecar_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    request_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    authority_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
        exclude_if=lambda value: value is None,
    )
    authority_status: DatasetAuthorityStatus | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    review_ids: tuple[str, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )
    mask_boundary: str = Field(min_length=1)
    contamination_boundary: str = Field(min_length=1)

    @field_validator("source_uri")
    @classmethod
    def _source_uri(cls, value: str) -> str:
        return require_public_string(value, field_name="source_uri")

    @field_validator(
        "source_sha256",
        "content_sha256",
        "split_manifest_sha256",
    )
    @classmethod
    def _hashes(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=info.field_name)

    @field_validator("truth_sidecar_sha256")
    @classmethod
    def _optional_truth_hash(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="truth_sidecar_sha256")

    @field_validator("request_sha256", "authority_sha256")
    @classmethod
    def _optional_authority_hash(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else require_sha256(value, field_name=info.field_name)

    @field_validator("episode_recipe_hashes")
    @classmethod
    def _recipe_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(require_sha256(value) for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("episode recipe hashes must be unique")
        return normalized

    @field_validator("review_ids")
    @classmethod
    def _unique_review_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("dataset authority review ids must be unique")
        return values

    @model_validator(mode="after")
    def _evaluation_binding_is_complete(self) -> DatasetSnapshotSpec:
        evaluation_fields = (
            self.evaluation_scenario_id,
            self.truth_sidecar_sha256,
        )
        if any(value is not None for value in evaluation_fields):
            if any(value is None for value in evaluation_fields):
                raise ValueError(
                    "evaluation snapshots require scenario and truth-sidecar bindings"
                )
            if len(self.episode_recipe_hashes) != 1:
                raise ValueError(
                    "evaluation snapshots require exactly one preparation recipe hash"
                )
            if self.fit_partition != "train":
                raise ValueError("evaluation snapshot preprocessing must be fitted on train")
        authority_fields = (
            self.request_sha256,
            self.authority_sha256,
            self.authority_status,
        )
        if self.schema_version == "tabu.dataset-snapshot.v2":
            if any(value is not None for value in authority_fields) or self.review_ids:
                raise ValueError("dataset authority lineage requires dataset-snapshot.v3")
            return self
        if self.evaluation_scenario_id is None:
            raise ValueError("dataset-snapshot.v3 is reserved for evaluation snapshots")
        if any(value is None for value in authority_fields):
            raise ValueError(
                "dataset-snapshot.v3 requires request, authority, and authority-status bindings"
            )
        if self.authority_status is DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED:
            if self.review_ids:
                raise ValueError("unreviewed dataset authority cannot cite promotion reviews")
        elif not self.review_ids:
            raise ValueError("reviewed dataset authority requires at least one review id")
        return self

    @property
    def publication_eligible(self) -> bool:
        """Whether this snapshot may support a public evaluation result."""

        return self.authority_status is DatasetAuthorityStatus.REVIEWED

    @property
    def authority_review_subject_sha256(self) -> str | None:
        """Hash the exact unreviewed manifest that an authority review approves.

        Promotion changes only ``authority_status`` and ``review_ids``.  By
        normalizing those two fields back to their unreviewed values, a review
        stays bound to the immutable data/request/authority manifest and cannot
        be replayed after any substantive snapshot field changes.
        """

        if self.schema_version != "tabu.dataset-snapshot.v3":
            return None
        review_subject = self.model_copy(
            update={
                "authority_status": DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED,
                "review_ids": (),
            }
        )
        return review_subject.content_hash


class ReviewRecord(PublicEvidenceSchema):
    schema_version: Literal["tabu.review.v1"] = "tabu.review.v1"
    review_id: str = Field(pattern=_ID_PATTERN)
    subjects: tuple[ObjectRef, ...] = Field(min_length=1)
    developer_identity: str = Field(min_length=1)
    reviewer_identity: str = Field(min_length=1)
    decision: ReviewDecision
    report: EvidencePointer
    gong_approval: EvidencePointer | None = None

    @model_validator(mode="after")
    def _review_is_independent(self) -> ReviewRecord:
        if self.developer_identity.strip().casefold() == self.reviewer_identity.strip().casefold():
            raise ValueError("reviewer must differ from developer")
        return self


class ClaimRecord(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-claim.v1"] = "tabu.catalog-claim.v1"
    claim_id: str = Field(pattern=_ID_PATTERN)
    statement: str = Field(min_length=1)
    boundary: str = Field(min_length=1)
    maturity: ClaimMaturity = ClaimMaturity.PROPOSAL
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence: tuple[ObjectRef, ...] = ()
    receipt: EvidencePointer | None = None
    review_ids: tuple[str, ...] = ()
    gong_approval: EvidencePointer | None = None

    @model_validator(mode="after")
    def _claim_promotion_is_evidence_gated(self) -> ClaimRecord:
        if self.maturity is not ClaimMaturity.PROPOSAL and not self.evidence:
            raise ValueError("non-proposal claim maturity requires linked evidence")
        if self.status in {ClaimStatus.REVIEWED, ClaimStatus.ACCEPTED} and not self.review_ids:
            raise ValueError("reviewed or accepted claims require review records")
        if self.status is ClaimStatus.ACCEPTED and self.gong_approval is None:
            raise ValueError("accepted claims require gong approval evidence")
        high_maturity = {
            ClaimMaturity.FORMAL_BENCHMARK,
            ClaimMaturity.SUPPORTED_MODEL,
            ClaimMaturity.FOUNDATION_MODEL,
        }
        if self.maturity in high_maturity and (
            self.receipt is None or not self.review_ids or self.gong_approval is None
        ):
            raise ValueError(
                "benchmark/supported/foundation maturity requires receipt, review, and approval"
            )
        return self


class LineageEdge(PublicEvidenceSchema):
    schema_version: Literal["tabu.lineage-edge.v1"] = "tabu.lineage-edge.v1"
    source: ObjectRef
    relation: LineageRelation
    target: ObjectRef
    evidence_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("evidence_hash")
    @classmethod
    def _evidence_hash(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value)

    @model_validator(mode="after")
    def _relation_has_legal_kinds(self) -> LineageEdge:
        allowed: dict[LineageRelation, set[tuple[CatalogObjectKind, CatalogObjectKind]]] = {
            LineageRelation.IMPLEMENTS: {
                (CatalogObjectKind.EXPERIMENT, CatalogObjectKind.MODEL_CONTRACT),
                (CatalogObjectKind.MODEL_ARTIFACT, CatalogObjectKind.MODEL_CONTRACT),
            },
            LineageRelation.USES_DATA: {
                (CatalogObjectKind.EXPERIMENT, CatalogObjectKind.DATASET_SNAPSHOT),
                (CatalogObjectKind.RUN, CatalogObjectKind.DATASET_SNAPSHOT),
                (CatalogObjectKind.EVAL_SUITE, CatalogObjectKind.DATASET_SNAPSHOT),
                (CatalogObjectKind.EVAL_RESULT, CatalogObjectKind.DATASET_SNAPSHOT),
            },
            LineageRelation.PRODUCED: {
                (CatalogObjectKind.EXPERIMENT, CatalogObjectKind.RUN),
                (CatalogObjectKind.RUN, CatalogObjectKind.MODEL_ARTIFACT),
                (CatalogObjectKind.RUN, CatalogObjectKind.RECEIPT),
                (CatalogObjectKind.RUN, CatalogObjectKind.RUN_ATTEMPT),
            },
            LineageRelation.EVALUATED_BY: {
                (CatalogObjectKind.MODEL_ARTIFACT, CatalogObjectKind.EVAL_RESULT),
                (CatalogObjectKind.RUN, CatalogObjectKind.EVAL_RESULT),
            },
            LineageRelation.SUPPORTS: {
                (CatalogObjectKind.EVAL_RESULT, CatalogObjectKind.CLAIM),
                (CatalogObjectKind.REVIEW, CatalogObjectKind.CLAIM),
                (CatalogObjectKind.RECEIPT, CatalogObjectKind.CLAIM),
            },
            LineageRelation.RESUMES_FROM: {
                (CatalogObjectKind.RUN, CatalogObjectKind.RUN),
            },
            LineageRelation.SUPERSEDES: {
                (kind, kind) for kind in CatalogObjectKind
            },
            LineageRelation.VERIFIED_BY: {
                (CatalogObjectKind.MODEL_CONTRACT, CatalogObjectKind.VERIFICATION_RESULT),
            },
        }
        pair = (self.source.kind, self.target.kind)
        if pair not in allowed[self.relation]:
            raise ValueError(
                f"illegal lineage kinds for {self.relation.value}: "
                f"{self.source.kind.value} -> {self.target.kind.value}"
            )
        if self.source == self.target:
            raise ValueError("lineage edges cannot self-reference")
        return self

    @property
    def edge_id(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class CatalogEntry(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-entry.v1"] = "tabu.catalog-entry.v1"
    kind: CatalogObjectKind
    object_id: str = Field(pattern=_ID_PATTERN)
    object_schema_version: str = Field(min_length=1)
    object_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_path: str = Field(min_length=1)
    status: str | None = None
    data: dict[str, JsonValue]

    @field_validator("source_path")
    @classmethod
    def _source_path(cls, value: str) -> str:
        return require_relative_source_path(value)

    @field_validator("object_hash", "source_hash")
    @classmethod
    def _hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _object_hash_matches(self) -> CatalogEntry:
        if canonical_hash(self.data) != self.object_hash:
            raise ValueError("catalog entry object_hash does not match data")
        return self

    @property
    def ref(self) -> ObjectRef:
        return ObjectRef(kind=self.kind, object_id=self.object_id)


class CatalogIndex(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-index.v1"] = "tabu.catalog-index.v1"
    source_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_revision: CatalogSourceRevision | None = None
    entries: tuple[CatalogEntry, ...] = ()
    lineage: tuple[LineageEdge, ...] = ()

    @field_validator("source_tree_hash")
    @classmethod
    def _source_tree_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source_tree_hash")

    @model_validator(mode="after")
    def _graph_is_closed(self) -> CatalogIndex:
        if (
            self.source_revision is not None
            and self.source_revision.catalog_source_tree_hash != self.source_tree_hash
        ):
            raise ValueError("catalog source revision does not bind source_tree_hash")
        ids = [entry.object_id for entry in self.entries]
        if len(ids) != len(set(ids)):
            duplicates = sorted({item for item in ids if ids.count(item) > 1})
            raise ValueError(f"duplicate catalog object ids: {', '.join(duplicates)}")

        refs = {(entry.kind, entry.object_id) for entry in self.entries}
        entries_by_id = {entry.object_id: entry for entry in self.entries}
        edge_ids: set[str] = set()
        for edge in self.lineage:
            if edge.edge_id in edge_ids:
                raise ValueError("duplicate lineage edge")
            edge_ids.add(edge.edge_id)
            for endpoint in (edge.source, edge.target):
                if (endpoint.kind, endpoint.object_id) not in refs:
                    raise ValueError(
                        f"dangling lineage reference: {endpoint.kind.value}/{endpoint.object_id}"
                    )
            if (
                edge.relation is LineageRelation.IMPLEMENTS
                and edge.target.kind is CatalogObjectKind.MODEL_CONTRACT
                and edge.evidence_hash is not None
            ):
                target = entries_by_id[edge.target.object_id]
                if edge.evidence_hash != target.object_hash:
                    raise ValueError(
                        "implements lineage evidence hash differs from its exact ModelSpec"
                    )

        self._reject_cycles()
        self._validate_cross_record_gates()
        expected_tree_hash = canonical_hash(
            {
                "schema": "tabu.catalog-source-tree.v1",
                "sources": [
                    {
                        "kind": entry.kind.value,
                        "object_id": entry.object_id,
                        "source_hash": entry.source_hash,
                        "source_path": entry.source_path,
                    }
                    for entry in self.entries
                ],
                "lineage": [edge.model_dump(mode="json") for edge in self.lineage],
            }
        )
        if self.source_tree_hash != expected_tree_hash:
            raise ValueError("catalog source_tree_hash does not match entries and lineage")
        return self

    def _reject_cycles(self) -> None:
        graph: dict[tuple[CatalogObjectKind, str], set[tuple[CatalogObjectKind, str]]] = {}
        for edge in self.lineage:
            source = (edge.source.kind, edge.source.object_id)
            target = (edge.target.kind, edge.target.object_id)
            graph.setdefault(source, set()).add(target)
            graph.setdefault(target, set())

        visiting: set[tuple[CatalogObjectKind, str]] = set()
        visited: set[tuple[CatalogObjectKind, str]] = set()

        def visit(node: tuple[CatalogObjectKind, str]) -> None:
            if node in visiting:
                raise ValueError("catalog lineage contains a cycle")
            if node in visited:
                return
            visiting.add(node)
            for target in graph.get(node, set()):
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in sorted(graph, key=lambda item: (item[0].value, item[1])):
            visit(node)

    def _validate_cross_record_gates(self) -> None:
        by_id = {entry.object_id: entry for entry in self.entries}
        receipts_by_hash: dict[str, CatalogEntry] = {}
        for entry in self.entries:
            if entry.kind is CatalogObjectKind.RECEIPT:
                receipt = Receipt.model_validate(entry.data)
                if receipt.receipt_hash in receipts_by_hash:
                    raise ValueError("duplicate receipt content hash")
                receipts_by_hash[receipt.receipt_hash] = entry

        experiment_superseders: dict[str, set[str]] = {}
        for edge in self.lineage:
            if (
                edge.relation is LineageRelation.SUPERSEDES
                and edge.source.kind is CatalogObjectKind.EXPERIMENT
                and edge.target.kind is CatalogObjectKind.EXPERIMENT
            ):
                experiment_superseders.setdefault(edge.target.object_id, set()).add(
                    edge.source.object_id
                )
        ambiguous = {
            target_id: tuple(sorted(source_ids))
            for target_id, source_ids in experiment_superseders.items()
            if len(source_ids) > 1
        }
        if ambiguous:
            raise ValueError(
                "an experiment revision can have only one direct superseding successor"
            )

        def require_entry(
            object_id: str,
            kind: CatalogObjectKind,
            *,
            owner: str,
        ) -> CatalogEntry:
            entry = by_id.get(object_id)
            if entry is None or entry.kind is not kind:
                raise ValueError(
                    f"{owner} references missing {kind.value} {object_id!r}"
                )
            return entry

        def receipt_for(
            pointer: EvidencePointer,
            *,
            run_id: str,
            owner: str,
        ) -> Receipt:
            entry = receipts_by_hash.get(pointer.sha256)
            if entry is None:
                raise ValueError(f"{owner} receipt pointer is not cataloged")
            if pointer.uri != entry.source_path:
                raise ValueError(f"{owner} receipt pointer does not bind its catalog source")
            receipt = Receipt.model_validate(entry.data)
            if pointer.sha256 != receipt.receipt_hash:
                raise ValueError(f"{owner} receipt pointer does not bind its typed receipt")
            if receipt.run_id != run_id:
                raise ValueError(f"{owner} receipt belongs to a different run")
            return receipt

        def approved_review(
            review_id: str,
            *,
            subject: ObjectRef,
            owner: str,
            require_gong: bool = False,
        ) -> ReviewRecord:
            entry = require_entry(review_id, CatalogObjectKind.REVIEW, owner=owner)
            review = ReviewRecord.model_validate(entry.data)
            if review.decision is not ReviewDecision.APPROVED:
                raise ValueError(f"{owner} requires an approved review")
            matching_subjects = tuple(
                candidate
                for candidate in review.subjects
                if candidate.kind is subject.kind and candidate.object_id == subject.object_id
            )
            if not matching_subjects:
                raise ValueError(f"{owner} review does not name the promoted subject")
            if subject.evidence_sha256 is not None and not any(
                candidate.evidence_sha256 == subject.evidence_sha256
                for candidate in matching_subjects
            ):
                raise ValueError(
                    f"{owner} review does not bind the promoted subject evidence hash"
                )
            if require_gong and review.gong_approval is None:
                raise ValueError(f"{owner} release review lacks gong approval")
            return review

        for entry in self.entries:
            if entry.kind is CatalogObjectKind.REVIEW:
                review = ReviewRecord.model_validate(entry.data)
                for subject in review.subjects:
                    require_entry(subject.object_id, subject.kind, owner=review.review_id)
                continue

            if entry.kind is CatalogObjectKind.DATASET_SNAPSHOT:
                snapshot = DatasetSnapshotSpec.model_validate(entry.data)
                if snapshot.authority_status is DatasetAuthorityStatus.REVIEWED:
                    subject_sha256 = snapshot.authority_review_subject_sha256
                    assert subject_sha256 is not None
                    subject = ObjectRef(
                        kind=CatalogObjectKind.DATASET_SNAPSHOT,
                        object_id=snapshot.dataset_snapshot_id,
                        evidence_sha256=subject_sha256,
                    )
                    reviews = tuple(
                        approved_review(
                            review_id,
                            subject=subject,
                            owner=snapshot.dataset_snapshot_id,
                        )
                        for review_id in snapshot.review_ids
                    )
                    if not reviews:
                        raise ValueError(
                            "reviewed dataset authority lacks an approved cataloged review"
                        )
                continue

            if entry.kind is CatalogObjectKind.EXPERIMENT:
                experiment = ExperimentRecord.model_validate(entry.data)
                require_entry(
                    experiment.contract_id,
                    CatalogObjectKind.MODEL_CONTRACT,
                    owner=experiment.experiment_id,
                )
                for dataset_id in experiment.dataset_snapshot_ids:
                    require_entry(
                        dataset_id,
                        CatalogObjectKind.DATASET_SNAPSHOT,
                        owner=experiment.experiment_id,
                    )
                referenced_runs: list[RunRecord] = []
                for run_id in experiment.run_ids:
                    run_entry = require_entry(
                        run_id,
                        CatalogObjectKind.RUN,
                        owner=experiment.experiment_id,
                    )
                    run = RunRecord.model_validate(run_entry.data)
                    if run.experiment_id != experiment.experiment_id:
                        raise ValueError("experiment references a run owned by another experiment")
                    referenced_runs.append(run)
                required_run_status = {
                    ExperimentStatus.RUNNING: RunStatus.RUNNING,
                    ExperimentStatus.SUCCEEDED: RunStatus.SUCCEEDED,
                    ExperimentStatus.FAILED: RunStatus.FAILED,
                    ExperimentStatus.KILLED: RunStatus.KILLED,
                }.get(experiment.status)
                if required_run_status is not None and not any(
                    run.status is required_run_status for run in referenced_runs
                ):
                    raise ValueError(
                        "experiment status is not supported by a matching cataloged run"
                    )
                if experiment.status is ExperimentStatus.REVIEWED and not any(
                    run.status in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.KILLED}
                    for run in referenced_runs
                ):
                    raise ValueError("reviewed experiment requires a terminal cataloged run")
                for previous_id in experiment.supersedes_experiment_ids:
                    previous_entry = require_entry(
                        previous_id,
                        CatalogObjectKind.EXPERIMENT,
                        owner=experiment.experiment_id,
                    )
                    previous = ExperimentRecord.model_validate(previous_entry.data)
                    if previous.contract_id != experiment.contract_id:
                        raise ValueError(
                            "an experiment can supersede only an experiment for the same contract"
                        )
                declared_supersedes = set(experiment.supersedes_experiment_ids)
                supersedes_edges = tuple(
                    edge
                    for edge in self.lineage
                    if edge.source.kind is CatalogObjectKind.EXPERIMENT
                    and edge.source.object_id == experiment.experiment_id
                    and edge.relation is LineageRelation.SUPERSEDES
                )
                edge_targets = {edge.target.object_id for edge in supersedes_edges}
                if edge_targets != declared_supersedes or len(supersedes_edges) != len(
                    declared_supersedes
                ):
                    raise ValueError(
                        "experiment supersedes declarations and lineage edges must match exactly"
                    )
                if declared_supersedes:
                    assert experiment.preregistration is not None
                    if any(
                        edge.evidence_hash != experiment.preregistration.sha256
                        for edge in supersedes_edges
                    ):
                        raise ValueError(
                            "supersedes lineage must bind the successor preregistration hash"
                        )
                if experiment.status is not ExperimentStatus.DRAFT:
                    subject = ObjectRef(
                        kind=CatalogObjectKind.EXPERIMENT,
                        object_id=experiment.experiment_id,
                    )
                    reviews = tuple(
                        approved_review(
                            review_id,
                            subject=subject,
                            owner=experiment.experiment_id,
                        )
                        for review_id in experiment.review_ids
                    )
                    assert experiment.preregistration_review is not None
                    if not any(
                        review.report == experiment.preregistration_review
                        for review in reviews
                    ):
                        raise ValueError(
                            "experiment preregistration review pointer does not match an "
                            "approved cataloged review"
                        )
                continue

            if entry.kind is CatalogObjectKind.RUN:
                run = RunRecord.model_validate(entry.data)
                experiment_entry = require_entry(
                    run.experiment_id,
                    CatalogObjectKind.EXPERIMENT,
                    owner=run.run_id,
                )
                experiment = ExperimentRecord.model_validate(experiment_entry.data)
                for attempt_id in run.attempt_ids:
                    attempt_entry = require_entry(
                        attempt_id,
                        CatalogObjectKind.RUN_ATTEMPT,
                        owner=run.run_id,
                    )
                    attempt = RunAttemptRecord.model_validate(attempt_entry.data)
                    if attempt.run_id != run.run_id:
                        raise ValueError("run references an attempt owned by another run")
                for artifact_id in run.artifact_ids:
                    artifact_entry = require_entry(
                        artifact_id,
                        CatalogObjectKind.MODEL_ARTIFACT,
                        owner=run.run_id,
                    )
                    artifact = ModelArtifact.model_validate(artifact_entry.data)
                    if artifact.producer_run_id != run.run_id:
                        raise ValueError("run references an artifact produced by another run")
                if run.resumes_from_run_id is not None:
                    require_entry(
                        run.resumes_from_run_id,
                        CatalogObjectKind.RUN,
                        owner=run.run_id,
                    )
                if run.receipt is not None:
                    receipt = receipt_for(run.receipt, run_id=run.run_id, owner=run.run_id)
                    expected = {
                        RunStatus.SUCCEEDED: ReceiptStatus.SUCCEEDED,
                        RunStatus.FAILED: ReceiptStatus.FAILED,
                        RunStatus.KILLED: ReceiptStatus.CANCELLED,
                    }.get(run.status)
                    if expected is not None and receipt.status is not expected:
                        raise ValueError("terminal run status conflicts with its receipt")
                    if run.status in {
                        RunStatus.SUCCEEDED,
                        RunStatus.FAILED,
                        RunStatus.KILLED,
                    } and run.receipt.sha256 not in run.status_history[-1].evidence_hashes:
                        raise ValueError("terminal run event must bind the receipt hash")
                    selected_attempts = tuple(
                        RunAttemptRecord.model_validate(candidate.data)
                        for candidate in self.entries
                        if candidate.kind is CatalogObjectKind.RUN_ATTEMPT
                        and candidate.object_id in run.attempt_ids
                        and RunAttemptRecord.model_validate(candidate.data).receipt == run.receipt
                    )
                    if run.status in {
                        RunStatus.SUCCEEDED,
                        RunStatus.FAILED,
                        RunStatus.KILLED,
                    } and len(selected_attempts) != 1:
                        raise ValueError(
                            "terminal run receipt must select exactly one cataloged attempt"
                        )
                    if receipt.metadata.get("issuance_status") == "formal" and (
                        experiment.status
                        not in {
                            ExperimentStatus.SUCCEEDED,
                            ExperimentStatus.FAILED,
                            ExperimentStatus.KILLED,
                            ExperimentStatus.REVIEWED,
                        }
                        or run.run_id not in experiment.run_ids
                    ):
                        raise ValueError(
                            "formal terminal runs require a matching terminal ExperimentRecord"
                        )
                continue

            if entry.kind is CatalogObjectKind.RUN_ATTEMPT:
                attempt = RunAttemptRecord.model_validate(entry.data)
                run_entry = require_entry(
                    attempt.run_id,
                    CatalogObjectKind.RUN,
                    owner=attempt.attempt_id,
                )
                run = RunRecord.model_validate(run_entry.data)
                if attempt.attempt_id not in run.attempt_ids:
                    raise ValueError("attempt is not listed by its owning run")
                receipt = receipt_for(
                    attempt.receipt,
                    run_id=attempt.run_id,
                    owner=attempt.attempt_id,
                )
                expected = {
                    RunAttemptStatus.SUCCEEDED: ReceiptStatus.SUCCEEDED,
                    RunAttemptStatus.FAILED: ReceiptStatus.FAILED,
                    RunAttemptStatus.KILLED: ReceiptStatus.CANCELLED,
                }.get(attempt.status)
                if expected is not None and receipt.status is not expected:
                    raise ValueError("attempt status conflicts with its receipt")
                continue

            if entry.kind is CatalogObjectKind.MODEL_ARTIFACT:
                artifact = ModelArtifact.model_validate(entry.data)
                contract_entry = require_entry(
                    artifact.contract_id,
                    CatalogObjectKind.MODEL_CONTRACT,
                    owner=artifact.artifact_id,
                )
                if (
                    artifact.model_spec.uri != contract_entry.source_path
                    or artifact.model_spec.sha256 != contract_entry.object_hash
                ):
                    raise ValueError(
                        "artifact ModelSpec pointer does not bind its cataloged contract"
                    )
                if contract_entry.data.get("contract_version") != artifact.contract_version:
                    raise ValueError("artifact contract version differs from its ModelSpec")
                producer = require_entry(
                    artifact.producer_run_id,
                    CatalogObjectKind.RUN,
                    owner=artifact.artifact_id,
                )
                run = RunRecord.model_validate(producer.data)
                if run.status is not RunStatus.SUCCEEDED or run.receipt is None:
                    raise ValueError(
                        "only a successful receipted run can produce a model artifact"
                    )
                receipt = receipt_for(
                    artifact.producer_receipt,
                    run_id=run.run_id,
                    owner=artifact.artifact_id,
                )
                if run.receipt != artifact.producer_receipt:
                    raise ValueError(
                        "artifact producer receipt does not match the successful run"
                    )
                receipt_root = str(PurePosixPath(artifact.producer_receipt.uri).parent)
                receipt_artifacts = {
                    str(PurePosixPath(receipt_root) / item.uri): item
                    for item in receipt.artifacts
                }
                checkpoint_source = receipt_artifacts.get(artifact.checkpoint.uri)
                if (
                    checkpoint_source is None
                    or checkpoint_source.sha256 != artifact.checkpoint.sha256
                    or checkpoint_source.kind != "checkpoint"
                ):
                    raise ValueError(
                        "artifact checkpoint pointer is absent from its producer receipt"
                    )
                for pointer, label in (
                    (artifact.semantic_config, "semantic config"),
                    (artifact.compiler_manifest, "compiler manifest"),
                ):
                    receipt_source = receipt_artifacts.get(pointer.uri)
                    if receipt_source is None:
                        raise ValueError(
                            f"artifact {label} pointer is absent from its producer receipt"
                        )
                    if receipt_source.sha256 != pointer.sha256:
                        raise ValueError(
                            f"artifact {label} digest differs from its producer receipt"
                        )
                if artifact.artifact_id not in run.artifact_ids:
                    raise ValueError("artifact is not listed by its producer run")
                if receipt.metadata.get("issuance_status") != "formal":
                    raise ValueError("local_unissued runs cannot register a model artifact")
                if artifact.status in {
                    ModelArtifactStatus.VERIFIED,
                    ModelArtifactStatus.RELEASED,
                }:
                    verification_events = tuple(
                        event
                        for event in artifact.status_history
                        if event.status is ModelArtifactStatus.VERIFIED
                    )
                    if len(verification_events) != 1:
                        raise ValueError(
                            "verified artifact requires exactly one verification event"
                        )
                    verification_pointer = verification_events[0].evidence
                    verification_entry = receipts_by_hash.get(verification_pointer.sha256)
                    if (
                        verification_entry is None
                        or verification_entry.source_path != verification_pointer.uri
                    ):
                        raise ValueError(
                            "artifact verification evidence is not a cataloged receipt"
                        )
                    verification_receipt = Receipt.model_validate(verification_entry.data)
                    if (
                        verification_receipt.status is not ReceiptStatus.SUCCEEDED
                        or verification_receipt.metadata.get("issuance_status") != "formal"
                    ):
                        raise ValueError(
                            "artifact verification requires a successful formal receipt"
                        )
                    verification_artifacts = {
                        (item.kind, item.sha256) for item in verification_receipt.artifacts
                    }
                    if ("checkpoint", artifact.checkpoint.sha256) not in verification_artifacts:
                        raise ValueError(
                            "artifact verification receipt does not bind the checkpoint digest"
                        )
                    for pointer, label in (
                        (artifact.semantic_config, "semantic config"),
                        (artifact.compiler_manifest, "compiler manifest"),
                    ):
                        if not any(
                            digest == pointer.sha256
                            for _kind, digest in verification_artifacts
                        ):
                            raise ValueError(
                                f"artifact verification receipt does not bind {label} digest"
                            )
                for parent_id in artifact.parent_artifact_ids:
                    require_entry(
                        parent_id,
                        CatalogObjectKind.MODEL_ARTIFACT,
                        owner=artifact.artifact_id,
                    )
                for result_id in artifact.evaluation_result_ids:
                    require_entry(
                        result_id,
                        CatalogObjectKind.EVAL_RESULT,
                        owner=artifact.artifact_id,
                    )
                if artifact.status is ModelArtifactStatus.RELEASED:
                    subject = ObjectRef(
                        kind=CatalogObjectKind.MODEL_ARTIFACT,
                        object_id=artifact.artifact_id,
                    )
                    reviews = tuple(
                        approved_review(
                            review_id,
                            subject=subject,
                            owner=artifact.artifact_id,
                            require_gong=True,
                        )
                        for review_id in artifact.review_ids
                    )
                    if not reviews:
                        raise ValueError("released artifact lacks an approved release review")
                continue

            if entry.kind is CatalogObjectKind.CLAIM:
                claim = ClaimRecord.model_validate(entry.data)
                evidence_refs = set(claim.evidence)
                for evidence in claim.evidence:
                    require_entry(evidence.object_id, evidence.kind, owner=claim.claim_id)
                high_maturity = claim.maturity in {
                    ClaimMaturity.FORMAL_BENCHMARK,
                    ClaimMaturity.SUPPORTED_MODEL,
                    ClaimMaturity.FOUNDATION_MODEL,
                }
                if claim.review_ids:
                    subject = ObjectRef(
                        kind=CatalogObjectKind.CLAIM,
                        object_id=claim.claim_id,
                    )
                    reviews = tuple(
                        approved_review(
                            review_id,
                            subject=subject,
                            owner=claim.claim_id,
                            require_gong=(
                                claim.status is ClaimStatus.ACCEPTED or high_maturity
                            ),
                        )
                        for review_id in claim.review_ids
                    )
                    if claim.status is ClaimStatus.ACCEPTED or high_maturity:
                        assert claim.gong_approval is not None
                        if not any(
                            review.gong_approval == claim.gong_approval
                            for review in reviews
                        ):
                            raise ValueError(
                                "accepted claim gong approval does not match its review"
                            )
                    for review_id in claim.review_ids:
                        review_ref = ObjectRef(
                            kind=CatalogObjectKind.REVIEW,
                            object_id=review_id,
                        )
                        if review_ref not in evidence_refs:
                            raise ValueError(
                                "promoted claim must explicitly include its review in evidence"
                            )
                if claim.receipt is not None:
                    receipt_entry = receipts_by_hash.get(claim.receipt.sha256)
                    if (
                        receipt_entry is None
                        or receipt_entry.source_path != claim.receipt.uri
                    ):
                        raise ValueError("claim receipt pointer is not cataloged")
                    receipt = Receipt.model_validate(receipt_entry.data)
                    receipt_ref = ObjectRef(
                        kind=CatalogObjectKind.RECEIPT,
                        object_id=receipt.receipt_id,
                    )
                    if receipt_ref not in evidence_refs:
                        raise ValueError(
                            "claim receipt must be explicitly included in claim evidence"
                        )
                    if high_maturity and (
                        receipt.status is not ReceiptStatus.SUCCEEDED
                        or receipt.metadata.get("issuance_status") != "formal"
                    ):
                        raise ValueError(
                            "high-maturity claims require a successful formal receipt"
                        )
                if high_maturity:
                    assert claim.receipt is not None
                    supporting_runs = tuple(
                        RunRecord.model_validate(candidate.data)
                        for candidate in self.entries
                        if candidate.kind is CatalogObjectKind.RUN
                        and RunRecord.model_validate(candidate.data).receipt == claim.receipt
                    )
                    if len(supporting_runs) != 1:
                        raise ValueError(
                            "high-maturity claim receipt must select exactly one cataloged run"
                        )
                    supporting_run = supporting_runs[0]
                    if supporting_run.status is not RunStatus.SUCCEEDED:
                        raise ValueError("high-maturity claim requires a successful run")
                    supporting_attempts = tuple(
                        RunAttemptRecord.model_validate(candidate.data)
                        for candidate in self.entries
                        if candidate.kind is CatalogObjectKind.RUN_ATTEMPT
                        and candidate.object_id in supporting_run.attempt_ids
                        and RunAttemptRecord.model_validate(candidate.data).receipt
                        == claim.receipt
                    )
                    if (
                        len(supporting_attempts) != 1
                        or supporting_attempts[0].status is not RunAttemptStatus.SUCCEEDED
                    ):
                        raise ValueError(
                            "high-maturity claim receipt must select one successful attempt"
                        )
                    supporting_experiment_entry = require_entry(
                        supporting_run.experiment_id,
                        CatalogObjectKind.EXPERIMENT,
                        owner=claim.claim_id,
                    )
                    supporting_experiment = ExperimentRecord.model_validate(
                        supporting_experiment_entry.data
                    )
                    if (
                        supporting_experiment.status
                        not in {ExperimentStatus.SUCCEEDED, ExperimentStatus.REVIEWED}
                        or supporting_run.run_id not in supporting_experiment.run_ids
                    ):
                        raise ValueError(
                            "high-maturity claim requires a successful experiment/run lineage"
                        )
                    required_refs = {
                        ObjectRef(
                            kind=CatalogObjectKind.EXPERIMENT,
                            object_id=supporting_experiment.experiment_id,
                        ),
                        ObjectRef(
                            kind=CatalogObjectKind.RUN,
                            object_id=supporting_run.run_id,
                        ),
                        ObjectRef(
                            kind=CatalogObjectKind.RUN_ATTEMPT,
                            object_id=supporting_attempts[0].attempt_id,
                        ),
                    }
                    if not required_refs.issubset(evidence_refs):
                        raise ValueError(
                            "high-maturity claim must expose its experiment, run, and attempt"
                        )

                evidence_artifacts = tuple(
                    ModelArtifact.model_validate(
                        require_entry(ref.object_id, ref.kind, owner=claim.claim_id).data
                    )
                    for ref in evidence_refs
                    if ref.kind is CatalogObjectKind.MODEL_ARTIFACT
                )
                evidence_results = tuple(
                    ref
                    for ref in evidence_refs
                    if ref.kind in {
                        CatalogObjectKind.EVAL_RESULT,
                        CatalogObjectKind.EVAL_COMPARISON,
                    }
                )
                if claim.maturity is ClaimMaturity.MICROBENCHMARK and not evidence_results:
                    raise ValueError("microbenchmark claims require evaluation evidence")
                if claim.maturity is ClaimMaturity.FORMAL_BENCHMARK and not any(
                    ref.kind is CatalogObjectKind.EVAL_COMPARISON for ref in evidence_results
                ):
                    raise ValueError("formal benchmark claims require a comparison report")
                if claim.maturity in {
                    ClaimMaturity.SUPPORTED_MODEL,
                    ClaimMaturity.FOUNDATION_MODEL,
                } and not any(
                    artifact.status is ModelArtifactStatus.RELEASED
                    for artifact in evidence_artifacts
                ):
                    raise ValueError("supported-model claims require a released model artifact")
                if claim.maturity is ClaimMaturity.FOUNDATION_MODEL and not any(
                    ref.kind is CatalogObjectKind.EVAL_COMPARISON for ref in evidence_results
                ):
                    raise ValueError("foundation-model claims require formal comparison evidence")
                continue

            if entry.kind is CatalogObjectKind.EVAL_RESULT:
                result = EvalResult.model_validate(entry.data)
                execution_receipt = result.execution_receipt
                if (
                    execution_receipt is None
                    or not execution_receipt.publication_eligible
                ):
                    raise ValueError(
                        "public evaluation results require their own formal evaluation receipt"
                    )
                suite_entry = require_entry(
                    result.suite_id,
                    CatalogObjectKind.EVAL_SUITE,
                    owner=result.result_id,
                )
                suite = EvalSuiteSpec.model_validate(suite_entry.data)
                if suite.suite_hash != result.suite_hash:
                    raise ValueError("evaluation result suite hash does not match the catalog")
                if suite.suite_version != result.suite_version:
                    raise ValueError("evaluation result suite version does not match the catalog")
                scenario_matches = tuple(
                    scenario
                    for scenario in suite.scenarios
                    if scenario.scenario_id == result.scenario_id
                )
                if len(scenario_matches) != 1:
                    raise ValueError("evaluation result scenario is not cataloged by its suite")
                scenario = scenario_matches[0]
                if result.task is not scenario.task:
                    raise ValueError("evaluation result task does not match its scenario")
                if result.budget_hash != suite.budget.content_hash:
                    raise ValueError("evaluation result budget does not match its suite")
                if result.seed not in suite.budget.model_seeds:
                    raise ValueError("evaluation result seed is outside the frozen suite budget")
                if result.adapter.kind is AdapterKind.MODEL:
                    if result.adapter.contract_id not in scenario.applicable_contracts:
                        raise ValueError(
                            "evaluation model contract is not applicable to its scenario"
                        )
                    if result.adapter.artifact_id is None:
                        raise ValueError(
                            "public model evaluation results require a model artifact"
                        )
                    if (
                        result.producer.provenance
                        is not ProducerProvenance.RECEIPTED_RUN
                        or not result.producer.publication_eligible
                    ):
                        raise ValueError(
                            "public model evaluation results require a receipted producer"
                        )
                    assert result.producer.run_id is not None
                    assert result.producer.receipt_pointer is not None
                    assert result.producer.receipt_sha256 is not None
                    run_entry = require_entry(
                        result.producer.run_id,
                        CatalogObjectKind.RUN,
                        owner=result.result_id,
                    )
                    run = RunRecord.model_validate(run_entry.data)
                    if run.receipt is None:
                        raise ValueError("evaluation producer run has no receipt")
                    if (
                        run.receipt.uri != result.producer.receipt_pointer
                        or run.receipt.sha256 != result.producer.receipt_sha256
                    ):
                        raise ValueError(
                            "evaluation producer pointer does not match the cataloged run receipt"
                        )
                    producer_receipt = receipt_for(
                        run.receipt,
                        run_id=run.run_id,
                        owner=result.result_id,
                    )
                    if (
                        producer_receipt.status is not ReceiptStatus.SUCCEEDED
                        or producer_receipt.metadata.get("issuance_status") != "formal"
                    ):
                        raise ValueError(
                            "public model evaluation results require a successful formal "
                            "producer receipt"
                        )
                else:
                    baselines = {
                        baseline.baseline_id: baseline for baseline in scenario.baselines
                    }
                    baseline = baselines.get(result.adapter.adapter_id)
                    if (
                        baseline is None
                        or baseline.family != result.adapter.baseline_family
                        or result.producer.provenance
                        is not ProducerProvenance.UNISSUED_BASELINE
                        or execution_receipt.baseline_spec_sha256
                        != baseline.content_hash
                        or execution_receipt.producer_sha256
                        != baseline.content_hash
                    ):
                        raise ValueError(
                            "public baseline evaluation does not match its frozen scenario"
                        )
                if result.adapter.artifact_id is not None:
                    artifact_entry = require_entry(
                        result.adapter.artifact_id,
                        CatalogObjectKind.MODEL_ARTIFACT,
                        owner=result.result_id,
                    )
                    artifact = ModelArtifact.model_validate(artifact_entry.data)
                    if artifact.contract_id != result.adapter.contract_id:
                        raise ValueError(
                            "evaluation adapter contract differs from its model artifact"
                        )
                    if (
                        result.producer.run_id is None
                        or result.producer.receipt_pointer is None
                        or result.producer.receipt_sha256 is None
                        or artifact.producer_run_id != result.producer.run_id
                        or artifact.producer_receipt.uri
                        != result.producer.receipt_pointer
                        or artifact.producer_receipt.sha256
                        != result.producer.receipt_sha256
                    ):
                        raise ValueError(
                            "evaluation model artifact differs from its exact producer "
                            "run receipt"
                        )
                    if result.result_id not in artifact.evaluation_result_ids:
                        raise ValueError(
                            "evaluation result is not listed by its model artifact"
                        )
                if (
                    result.status is EvaluationStatus.SUCCEEDED
                    or result.truth_sidecar_sha256 is not None
                ):
                    dataset_matches: list[DatasetSnapshotSpec] = []
                    for candidate in self.entries:
                        if candidate.kind is not CatalogObjectKind.DATASET_SNAPSHOT:
                            continue
                        snapshot = DatasetSnapshotSpec.model_validate(candidate.data)
                        if (
                            snapshot.authority_status is DatasetAuthorityStatus.REVIEWED
                            and snapshot.dataset_snapshot_id
                            == execution_receipt.dataset_snapshot_id
                            and snapshot.content_hash
                            == execution_receipt.dataset_snapshot_sha256
                            and snapshot.request_sha256
                            == execution_receipt.dataset_request_sha256
                            and snapshot.authority_sha256
                            == execution_receipt.dataset_authority_sha256
                            and snapshot.dataset_id == scenario.dataset.dataset_id
                            and snapshot.source_uri == scenario.dataset.source_uri
                            and snapshot.license_id == scenario.dataset.license_id
                            and snapshot.source_sha256 == result.source_sha256
                            and snapshot.split_manifest_sha256 == result.split_sha256
                            and result.recipe_sha256 in snapshot.episode_recipe_hashes
                            and snapshot.evaluation_scenario_id == result.scenario_id
                            and snapshot.truth_sidecar_sha256
                            == result.truth_sidecar_sha256
                        ):
                            dataset_matches.append(snapshot)
                    if len(dataset_matches) != 1:
                        raise ValueError(
                            "evaluation result does not bind exactly one canonical dataset "
                            "snapshot and truth sidecar"
                        )
                continue

            if entry.kind is CatalogObjectKind.EVAL_COMPARISON:
                comparison = ComparisonReport.model_validate(entry.data)
                suite_entry = require_entry(
                    comparison.suite_id,
                    CatalogObjectKind.EVAL_SUITE,
                    owner=comparison.comparison_id,
                )
                suite = EvalSuiteSpec.model_validate(suite_entry.data)
                if suite.suite_hash != comparison.suite_hash:
                    raise ValueError("comparison suite hash does not match the catalog")
                results_by_hash = {
                    EvalResult.model_validate(
                        candidate.data
                    ).content_hash: EvalResult.model_validate(candidate.data)
                    for candidate in self.entries
                    if candidate.kind is CatalogObjectKind.EVAL_RESULT
                }
                if any(digest not in results_by_hash for digest in comparison.result_hashes):
                    raise ValueError("comparison references a missing evaluation result hash")
                referenced_results = tuple(
                    results_by_hash[digest] for digest in comparison.result_hashes
                )
                expected_aggregates, expected_failure_counts = derive_comparison_summary(
                    suite,
                    referenced_results,
                )
                expected_hashes = tuple(
                    sorted(result.content_hash for result in referenced_results)
                )
                expected_identity = canonical_hash(
                    {
                        "schema": "tabu.eval-comparison-identity.v2",
                        "suite_hash": suite.suite_hash,
                        "result_hashes": expected_hashes,
                    }
                )
                if comparison.result_hashes != expected_hashes:
                    raise ValueError("comparison result hashes are not canonical")
                if comparison.comparison_id != f"compare-{expected_identity[:24]}":
                    raise ValueError("comparison id does not bind its exact result hashes")
                if comparison.suite_version != suite.suite_version:
                    raise ValueError("comparison suite version does not match the catalog")
                if comparison.variable_under_test != suite.variable_under_test:
                    raise ValueError("comparison variable under test does not match its suite")
                if comparison.claim_boundary != suite.claim_boundary:
                    raise ValueError("comparison claim boundary does not match its suite")
                if comparison.aggregates != expected_aggregates:
                    raise ValueError("comparison aggregates do not match cited evaluation results")
                if comparison.failure_counts != expected_failure_counts:
                    raise ValueError("comparison failure counts do not match cited results")
                if comparison.publication_eligible != comparison_publication_eligible(
                    suite,
                    referenced_results,
                ):
                    raise ValueError("comparison publication status does not match cited results")
                if not comparison.publication_eligible:
                    raise ValueError("unissued comparisons cannot enter the public catalog")

    @property
    def catalog_hash(self) -> str:
        return self.content_hash

    def collections(self) -> dict[str, tuple[CatalogEntry, ...]]:
        grouped: dict[str, list[CatalogEntry]] = {
            kind.value: [] for kind in CatalogObjectKind
        }
        for entry in self.entries:
            grouped[entry.kind.value].append(entry)
        return {name: tuple(items) for name, items in grouped.items()}

    def show(self, object_id: str) -> CatalogEntry:
        matches = [entry for entry in self.entries if entry.object_id == object_id]
        if not matches:
            raise KeyError(f"catalog object not found: {object_id}")
        return matches[0]


class CatalogCheckIssue(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-check-issue.v1"] = "tabu.catalog-check-issue.v1"
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_path: str | None = None

    @field_validator("source_path")
    @classmethod
    def _optional_source_path(cls, value: str | None) -> str | None:
        return None if value is None else require_relative_source_path(value)


class CatalogCheckReport(PublicEvidenceSchema):
    schema_version: Literal["tabu.catalog-check-report.v1"] = "tabu.catalog-check-report.v1"
    ok: bool
    catalog_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    issues: tuple[CatalogCheckIssue, ...] = ()

    @model_validator(mode="after")
    def _coherent(self) -> CatalogCheckReport:
        if self.ok == bool(self.issues):
            raise ValueError("catalog check ok flag must agree with issues")
        return self


__all__ = [
    "ArtifactStatusEvent",
    "CatalogCheckIssue",
    "CatalogCheckReport",
    "CatalogEntry",
    "CatalogIndex",
    "CatalogObjectKind",
    "ClaimMaturity",
    "ClaimRecord",
    "ClaimStatus",
    "DatasetAdapter",
    "DatasetAuthorityStatus",
    "DatasetSnapshotSpec",
    "EvidencePointer",
    "ExperimentRecord",
    "ExperimentStatus",
    "FailureCategory",
    "LineageEdge",
    "LineageRelation",
    "ModelArtifact",
    "ModelArtifactStatus",
    "ObjectRef",
    "ReviewDecision",
    "ReviewRecord",
    "RunAttemptRecord",
    "RunAttemptStatus",
    "RunRecord",
    "RunStatus",
    "StatusEvent",
    "require_public_string",
    "require_relative_source_path",
]
