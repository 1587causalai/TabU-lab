"""Versioned, strict schema models for run evidence and bounded claims."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from .canonical import canonical_hash as hash_canonical
from .canonical import require_sha256, to_canonical_data
from .public_safety import (
    contains_absolute_local_path,
    contains_private_identity_or_secret,
    require_public_evidence_safe,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class EvidenceSchema(BaseModel):
    """Immutable, extra-forbidden base with deterministic content identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @model_validator(mode="after")
    def _is_canonicalizable(self) -> EvidenceSchema:
        payload = self.model_dump(mode="python", by_alias=False)
        to_canonical_data(payload)
        require_public_evidence_safe(payload)
        return self

    @property
    def content_hash(self) -> str:
        return hash_canonical(self.model_dump(mode="python", by_alias=False))

    @property
    def schema_hash(self) -> str:
        return self.content_hash


class ArtifactRef(EvidenceSchema):
    artifact_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(ge=0)
    media_type: str | None = None

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return require_sha256(value)


class ContractReference(EvidenceSchema):
    contract_id: str = Field(min_length=1)
    contract_version: str = Field(min_length=1)
    maturity_required: str = Field(min_length=1)


class PreregistrationData(EvidenceSchema):
    kind: str = Field(min_length=1)
    rows: int = Field(gt=0)
    numeric_features: int = Field(ge=0)
    categorical_features: int = Field(ge=0)
    split_policy: Literal["split_before_compile"]
    fit_partition: str = Field(min_length=1)
    target_origin: str = Field(min_length=1)
    natural_missing_targets: str = Field(min_length=1)


class BaselinePlan(EvidenceSchema):
    name: str = Field(min_length=1)
    budget: str = Field(min_length=1)


class ModelDefaultPlan(EvidenceSchema):
    carrier: str = Field(min_length=1)
    dynamics: str = Field(min_length=1)
    numeric_terminal: str = Field(min_length=1)
    categorical_terminal: str = Field(min_length=1)
    ll_terminal: str = Field(min_length=1)


class TrainingPlan(EvidenceSchema):
    optimizer: str = Field(min_length=1)
    max_steps: int = Field(gt=0)
    seeds: tuple[NonNegativeInt, ...] = Field(min_length=1)
    device: str = Field(min_length=1)
    dtype: Literal["float32"] = "float32"
    wall_clock_budget_minutes: int = Field(gt=0)

    @field_validator("seeds")
    @classmethod
    def _valid_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("preregistration seeds must be non-negative integers")
        if len(values) != len(set(values)):
            raise ValueError("preregistration seeds must be unique")
        return values


class MetricPlan(EvidenceSchema):
    primary: str = Field(min_length=1)
    diagnostics: tuple[NonEmptyString, ...] = Field(min_length=1)


class ExitPlan(EvidenceSchema):
    passed: str = Field(min_length=1, alias="pass")
    kill: str = Field(min_length=1)


class ReviewPlan(EvidenceSchema):
    developer_and_reviewer_must_differ: Literal[True]
    gong_approval_required_for_release: Literal[True]


class Preregistration(EvidenceSchema):
    """Pre-committed experiment boundary; status cannot imply evidence."""

    schema_version: Literal["tabu-lab.preregistration.v1"]
    experiment_id: str = Field(min_length=1)
    status: Literal["proposed"]
    created_at: date
    contract: ContractReference
    hypothesis: str = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)
    data: PreregistrationData
    baseline: BaselinePlan
    model_default: ModelDefaultPlan
    training: TrainingPlan
    metrics: MetricPlan
    pass_conditions: tuple[NonEmptyString, ...] = Field(min_length=1)
    kill_conditions: tuple[NonEmptyString, ...] = Field(min_length=1)
    exit_conditions: ExitPlan
    required_artifacts: tuple[NonEmptyString, ...] = Field(min_length=1)
    review: ReviewPlan

    @field_validator("pass_conditions", "kill_conditions", "required_artifacts")
    @classmethod
    def _nonempty_unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("preregistration lists cannot contain empty values")
        if len(values) != len(set(values)):
            raise ValueError("preregistration lists cannot contain duplicates")
        return values


_RUN_HASH_FIELDS = (
    "spec_hash",
    "code_hash",
    "data_hash",
    "split_hash",
    "compiler_hash",
    "semantic_config_hash",
    "execution_config_hash",
    "training_config_hash",
)


def _normalize_named_seeds(seeds: Mapping[str, int]) -> dict[str, int]:
    if not seeds:
        raise ValueError("run seeds must be a non-empty named map")
    normalized: dict[str, int] = {}
    for name, seed in seeds.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("run seed names must be non-empty strings")
        seed_name = name.strip()
        if seed_name in normalized:
            raise ValueError(f"duplicate normalized run seed name: {seed_name!r}")
        if type(seed) is not int or seed < 0:
            raise ValueError("run seeds must be non-negative integers")
        normalized[seed_name] = seed
    return dict(sorted(normalized.items()))


def derive_seed_hash(seeds: Mapping[str, int]) -> str:
    """Hash a canonical named RNG seed map, independent of insertion order."""

    normalized = _normalize_named_seeds(seeds)
    return hash_canonical({"schema": "tabu.named-seeds.v1", "seeds": normalized})


def derive_run_id(
    *,
    spec_hash: str,
    code_hash: str,
    data_hash: str,
    split_hash: str,
    compiler_hash: str,
    semantic_config_hash: str,
    execution_config_hash: str,
    training_config_hash: str,
    seed_hash: str,
) -> str:
    components = {
        "spec_hash": spec_hash,
        "code_hash": code_hash,
        "data_hash": data_hash,
        "split_hash": split_hash,
        "compiler_hash": compiler_hash,
        "semantic_config_hash": semantic_config_hash,
        "execution_config_hash": execution_config_hash,
        "training_config_hash": training_config_hash,
        "seed_hash": seed_hash,
    }
    payload = {
        field_name: require_sha256(value, field_name=field_name)
        for field_name, value in components.items()
    }
    return f"run-{hash_canonical({'schema': 'tabu.run-identity.v2', **payload})}"


class RunIdentity(EvidenceSchema):
    run_id: str = Field(pattern=r"^run-[0-9a-f]{64}$")
    spec_hash: Sha256
    code_hash: Sha256
    data_hash: Sha256
    split_hash: Sha256
    compiler_hash: Sha256
    semantic_config_hash: Sha256
    execution_config_hash: Sha256
    training_config_hash: Sha256
    seeds: Mapping[str, NonNegativeInt] = Field(min_length=1)
    seed_hash: Sha256

    @field_validator(*_RUN_HASH_FIELDS)
    @classmethod
    def _valid_identity_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=info.field_name)

    @field_validator("seed_hash")
    @classmethod
    def _valid_seed_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="seed_hash")

    @field_validator("seeds")
    @classmethod
    def _valid_named_seeds(cls, values: Mapping[str, int]) -> Mapping[str, int]:
        return MappingProxyType(_normalize_named_seeds(values))

    @field_serializer("seeds")
    def _serialize_named_seeds(self, values: Mapping[str, int]) -> dict[str, int]:
        return dict(values)

    @model_validator(mode="after")
    def _derived_id_matches(self) -> RunIdentity:
        expected_seed_hash = derive_seed_hash(self.seeds)
        if self.seed_hash != expected_seed_hash:
            raise ValueError("seed_hash does not match the canonical named seed map")
        expected = derive_run_id(
            **{field_name: getattr(self, field_name) for field_name in _RUN_HASH_FIELDS},
            seed_hash=self.seed_hash,
        )
        if self.run_id != expected:
            raise ValueError("run_id does not match the canonical RunIdentity components")
        return self

    @classmethod
    def create(
        cls,
        *,
        seeds: Mapping[str, int] | None = None,
        seed: int | None = None,
        **components: Any,
    ) -> RunIdentity:
        """Create a derived identity; ``seed`` is a legacy single-seed adapter."""

        if seeds is not None and seed is not None:
            raise ValueError("pass named seeds or legacy seed, not both")
        if seeds is None:
            if seed is None:
                raise ValueError("RunIdentity.create requires a named seed map")
            seeds = {"global": seed}
        normalized_seeds = _normalize_named_seeds(seeds)
        seed_hash = derive_seed_hash(normalized_seeds)
        run_id = derive_run_id(**components, seed_hash=seed_hash)
        return cls(
            run_id=run_id,
            seeds=normalized_seeds,
            seed_hash=seed_hash,
            **components,
        )

    @property
    def seed(self) -> int | None:
        """Legacy read adapter, populated only for ``{"global": seed}`` identities."""

        if tuple(self.seeds) == ("global",):
            return self.seeds["global"]
        return None

    @property
    def identity_hash(self) -> str:
        return self.content_hash


class EnvironmentDisclosure(EvidenceSchema):
    environment_hash: Sha256
    host_class: str = Field(min_length=1)
    operating_system: str = Field(min_length=1)
    device: str = Field(min_length=1)
    architecture: str | None = None
    accelerator: str | None = None
    python_version: str | None = None

    @field_validator("environment_hash")
    @classmethod
    def _valid_environment_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="environment_hash")

    @field_validator(
        "host_class",
        "operating_system",
        "device",
        "architecture",
        "accelerator",
        "python_version",
    )
    @classmethod
    def _public_environment_strings(
        cls, value: str | None, info: object
    ) -> str | None:
        if value is None:
            return value
        field_name = getattr(info, "field_name", "environment")
        if contains_absolute_local_path(value):
            raise ValueError(f"{field_name} cannot expose an absolute local path")
        if contains_private_identity_or_secret(value):
            raise ValueError(f"{field_name} cannot expose a host identity or secret")
        return value

    @field_validator("host_class")
    @classmethod
    def _generalized_host_class(cls, value: str) -> str:
        # ``workstation`` is retained for the v1 RunBundle compatibility
        # fixtures.  Fit/eval capture emits one of the device-derived classes.
        if value not in {"cpu-host", "cuda-host", "mps-host", "workstation"}:
            raise ValueError("host_class must be a generalized public host class")
        return value

    @field_validator("operating_system")
    @classmethod
    def _operating_system_family(cls, value: str) -> str:
        allowed = {
            "AIX",
            "Android",
            "Darwin",
            "Emscripten",
            "FreeBSD",
            "Haiku",
            "iOS",
            "Java",
            "Linux",
            "NetBSD",
            "OpenBSD",
            "SunOS",
            "WASI",
            "Windows",
        }
        # Preserve historical lower-case receipt fixtures without accepting a
        # free-form machine description in an OS-family field.
        if value not in allowed and value.casefold() not in {
            item.casefold() for item in allowed
        }:
            raise ValueError("operating_system must disclose only an OS family")
        return value

    @field_validator("device")
    @classmethod
    def _public_device_class(cls, value: str) -> str:
        if re.fullmatch(
            r"(?:cpu|cuda|mps|xpu|hpu|ipu|mtia|meta|privateuseone)(?::[0-9]+)?",
            value,
            flags=re.IGNORECASE,
        ) is None:
            raise ValueError("device must be a public accelerator class or index")
        return value

    @field_validator("architecture")
    @classmethod
    def _public_architecture(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if re.fullmatch(
            r"(?:"
            r"aarch64|arm64|armv[4-9][a-z0-9]*|"
            r"amd64|x86_64|x86|i[3-6]86|"
            r"ppc(?:64(?:le)?)?|powerpc(?:64(?:le)?)?|"
            r"riscv(?:32|64)|s390x?|sparc(?:64)?|"
            r"mips(?:64)?(?:el)?|loongarch64|wasm(?:32|64)"
            r")",
            value,
            flags=re.IGNORECASE,
        ) is None:
            raise ValueError("architecture must disclose only a processor family")
        return value

    @field_validator("python_version")
    @classmethod
    def _public_python_version(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[A-Za-z0-9.+-]*)?", value) is None:
            raise ValueError("python_version must be a version, not host metadata")
        return value


class RunBundle(EvidenceSchema):
    """Portable run identity; artifacts are references, not embedded claims."""

    schema_version: Literal["tabu.run-bundle.v1"] = "tabu.run-bundle.v1"
    identity: RunIdentity
    created_at: datetime = Field(default_factory=_utc_now)
    model_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    fit_partition: str = Field(min_length=1)
    environment: EnvironmentDisclosure
    episode_recipe_hashes: tuple[Sha256, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("episode_recipe_hashes")
    @classmethod
    def _valid_recipe_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            require_sha256(value, field_name="episode_recipe_hash") for value in values
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("episode_recipe_hashes must be unique")
        return normalized

    @model_validator(mode="after")
    def _unique_artifacts(self) -> RunBundle:
        ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("RunBundle artifact ids must be unique")
        return self

    @property
    def run_bundle_hash(self) -> str:
        return self.content_hash

    @property
    def run_id(self) -> str:
        return self.identity.run_id


class ReceiptStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Receipt(EvidenceSchema):
    """One immutable observation about execution of a RunBundle."""

    schema_version: Literal["tabu.receipt.v1"] = "tabu.receipt.v1"
    receipt_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^run-[0-9a-f]{64}$")
    run_identity_hash: Sha256
    run_bundle_hash: Sha256
    status: ReceiptStatus
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    command: tuple[str, ...] = ()
    evaluation_hashes: tuple[Sha256, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()
    error: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("run_identity_hash", "run_bundle_hash")
    @classmethod
    def _valid_run_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=info.field_name)

    @classmethod
    def from_run_bundle(
        cls,
        bundle: RunBundle,
        *,
        receipt_id: str,
        status: ReceiptStatus,
        **values: Any,
    ) -> Receipt:
        return cls(
            receipt_id=receipt_id,
            run_id=bundle.identity.run_id,
            run_identity_hash=bundle.identity.identity_hash,
            run_bundle_hash=bundle.run_bundle_hash,
            status=status,
            **values,
        )

    @field_validator("evaluation_hashes")
    @classmethod
    def _valid_evaluation_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(require_sha256(value, field_name="evaluation_hash") for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("evaluation_hashes must be unique")
        return normalized

    @model_validator(mode="after")
    def _coherent_status(self) -> Receipt:
        if self.completed_at is not None and self.completed_at < self.created_at:
            raise ValueError("Receipt.completed_at cannot precede created_at")
        if self.status is ReceiptStatus.FAILED and not self.error:
            raise ValueError("failed Receipt requires an error boundary")
        if self.status is not ReceiptStatus.FAILED and self.error is not None:
            raise ValueError("Receipt.error is only valid for failed status")
        ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("Receipt artifact ids must be unique")
        return self

    @property
    def receipt_hash(self) -> str:
        return self.content_hash


class ClaimStatus(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class ClaimRecord(EvidenceSchema):
    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: ClaimStatus = ClaimStatus.PROPOSED
    evidence_hashes: tuple[Sha256, ...] = ()
    receipt_hash: Sha256 | None = None
    developer_identity: str | None = Field(default=None, min_length=1)
    reviewer_identity: str | None = Field(default=None, min_length=1)
    review_report_hash: Sha256 | None = None
    gong_approval_record: str | None = Field(default=None, min_length=1)
    gong_approval_hash: Sha256 | None = None
    boundary: str = Field(
        default="proposal only; no benchmark, release, or generalization claim",
        min_length=1,
    )
    notes: str | None = None

    @field_validator("evidence_hashes")
    @classmethod
    def _valid_evidence_hashes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(require_sha256(value, field_name="evidence_hash") for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("claim evidence_hashes must be unique")
        return normalized

    @field_validator("receipt_hash", "review_report_hash", "gong_approval_hash")
    @classmethod
    def _valid_claim_hash(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return require_sha256(value, field_name=info.field_name)

    @field_validator("developer_identity", "reviewer_identity", "gong_approval_record")
    @classmethod
    def _strip_optional_identity(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else None

    @model_validator(mode="after")
    def _accepted_needs_independent_approval_evidence(self) -> ClaimRecord:
        if self.status is not ClaimStatus.ACCEPTED:
            return self
        required = {
            "receipt_hash": self.receipt_hash,
            "developer_identity": self.developer_identity,
            "reviewer_identity": self.reviewer_identity,
            "review_report_hash": self.review_report_hash,
            "gong_approval_record": self.gong_approval_record,
            "gong_approval_hash": self.gong_approval_hash,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "accepted ClaimRecord requires receipt, independent review, and gong approval; "
                f"missing {missing}"
            )
        assert self.developer_identity is not None
        assert self.reviewer_identity is not None
        if self.developer_identity.casefold() == self.reviewer_identity.casefold():
            raise ValueError("accepted ClaimRecord developer and reviewer must differ")
        required_hashes = {
            self.receipt_hash,
            self.review_report_hash,
            self.gong_approval_hash,
        }
        if not required_hashes.issubset(self.evidence_hashes):
            raise ValueError(
                "accepted ClaimRecord evidence_hashes must include receipt, review, "
                "and gong approval hashes"
            )
        return self


class ClaimLedger(EvidenceSchema):
    """Explicit separation between proposed claims and linked evidence."""

    schema_version: Literal["tabu.claim-ledger.v1"] = "tabu.claim-ledger.v1"
    ledger_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=_utc_now)
    claims: tuple[ClaimRecord, ...] = Field(
        default=(),
        validation_alias=AliasChoices("claims", "entries"),
    )
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_claims(self) -> ClaimLedger:
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("ClaimLedger claim ids must be unique")
        return self

    @property
    def entries(self) -> tuple[ClaimRecord, ...]:
        return self.claims

    @property
    def ledger_hash(self) -> str:
        return self.content_hash


__all__ = [
    "ArtifactRef",
    "BaselinePlan",
    "ClaimLedger",
    "ClaimRecord",
    "ClaimStatus",
    "ContractReference",
    "EnvironmentDisclosure",
    "EvidenceSchema",
    "ExitPlan",
    "MetricPlan",
    "ModelDefaultPlan",
    "Preregistration",
    "PreregistrationData",
    "Receipt",
    "ReceiptStatus",
    "ReviewPlan",
    "RunBundle",
    "RunIdentity",
    "TrainingPlan",
    "derive_run_id",
    "derive_seed_hash",
]
