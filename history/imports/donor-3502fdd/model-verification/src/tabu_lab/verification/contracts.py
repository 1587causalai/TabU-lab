"""Typed contracts for Model Verification & Evaluation (MVE).

Verification is intentionally an evidence lane, not a maturity score.  A
result records the exact contract, suite, and individual checks that produced
it; the caller may later attach a formal receipt or an independent review.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import require_sha256
from tabu_lab.evidence.schemas import EvidenceSchema

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.@-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class VerificationAxis(StrEnum):
    COMPONENT_CORRECTNESS = "component_correctness"
    ARCHITECTURE_EVOLVABILITY = "architecture_evolvability"


class AssessmentOutcome(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"
    FAILED = "failed"
    PASSED = "passed"


class EvidenceLevel(StrEnum):
    NONE = "none"
    LOCAL_UNISSUED = "local_unissued"
    FORMAL = "formal"
    REVIEWED = "reviewed"


class VerificationCheck(EvidenceSchema):
    """An allow-listed, semantic check declaration."""

    schema_version: Literal["tabu.verification-check.v1"] = "tabu.verification-check.v1"
    check_id: Identifier
    axis: VerificationAxis
    description: str = Field(min_length=1)
    required: bool = True


class VerificationSuite(EvidenceSchema):
    """A closed suite whose checks are resolved through the check registry."""

    schema_version: Literal["tabu.verification-suite.v1"] = "tabu.verification-suite.v1"
    suite_id: Identifier
    suite_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    title: str = Field(min_length=1)
    axis: VerificationAxis
    checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    claim_boundary: str = Field(min_length=1)

    @field_validator("checks")
    @classmethod
    def _unique_checks(cls, values: tuple[VerificationCheck, ...]) -> tuple[VerificationCheck, ...]:
        ids = tuple(item.check_id for item in values)
        if len(ids) != len(set(ids)):
            raise ValueError("verification suite check_id values must be unique")
        return values

    @model_validator(mode="after")
    def _checks_match_axis(self) -> VerificationSuite:
        if any(item.axis is not self.axis for item in self.checks):
            raise ValueError("verification suite checks must use the suite axis")
        return self

    @property
    def suite_hash(self) -> str:
        return self.content_hash


class ModelCompositionDescriptor(EvidenceSchema):
    """The concrete composition used by one model build."""

    schema_version: Literal["tabu.model-composition.v1"] = "tabu.model-composition.v1"
    contract_id: Identifier
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    components: dict[str, str] = Field(min_length=1)
    configuration: dict[str, JsonValue] = Field(default_factory=dict)
    implementation_modules: dict[str, str] = Field(default_factory=dict)

    @field_validator("components", "implementation_modules")
    @classmethod
    def _nonempty_values(cls, values: dict[str, str]) -> dict[str, str]:
        if any(not key.strip() or not value.strip() for key, value in values.items()):
            raise ValueError("composition names and values cannot be blank")
        return values

    @property
    def composition_hash(self) -> str:
        return self.content_hash


class VerificationCheckResult(EvidenceSchema):
    """One deterministic probe outcome."""

    schema_version: Literal["tabu.verification-check-result.v1"] = (
        "tabu.verification-check-result.v1"
    )
    check_id: Identifier
    outcome: AssessmentOutcome
    detail: str = Field(min_length=1)
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    references: tuple[str, ...] = ()


class VerificationResult(EvidenceSchema):
    """Hash-bound result for one suite and one exact ModelSpec."""

    schema_version: Literal["tabu.verification-result.v1"] = "tabu.verification-result.v1"
    result_id: Identifier
    suite_id: Identifier
    suite_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    suite_hash: Sha256
    axis: VerificationAxis
    contract_id: Identifier
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    model_spec_hash: Sha256
    composition: ModelCompositionDescriptor | None = None
    outcome: AssessmentOutcome
    evidence_level: EvidenceLevel = EvidenceLevel.LOCAL_UNISSUED
    checks: tuple[VerificationCheckResult, ...] = Field(min_length=1)
    receipt_ref: str | None = None
    review_ref: str | None = None
    blockers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    claim_boundary: str = Field(min_length=1)

    @field_validator("suite_hash", "model_spec_hash")
    @classmethod
    def _hash(cls, value: str) -> str:
        return require_sha256(value)

    @field_validator("checks")
    @classmethod
    def _unique_check_results(
        cls, values: tuple[VerificationCheckResult, ...]
    ) -> tuple[VerificationCheckResult, ...]:
        ids = tuple(item.check_id for item in values)
        if len(ids) != len(set(ids)):
            raise ValueError("verification result check_id values must be unique")
        return values

    @model_validator(mode="after")
    def _evidence_gate(self) -> VerificationResult:
        if (
            self.evidence_level in {EvidenceLevel.FORMAL, EvidenceLevel.REVIEWED}
            and not self.receipt_ref
        ):
            raise ValueError("formal/reviewed verification requires a receipt_ref")
        if self.evidence_level is EvidenceLevel.REVIEWED and not self.review_ref:
            raise ValueError("reviewed verification requires a review_ref")
        return self

    @property
    def result_hash(self) -> str:
        return self.content_hash


__all__ = [
    "AssessmentOutcome",
    "EvidenceLevel",
    "ModelCompositionDescriptor",
    "VerificationAxis",
    "VerificationCheck",
    "VerificationCheckResult",
    "VerificationResult",
    "VerificationSuite",
]
