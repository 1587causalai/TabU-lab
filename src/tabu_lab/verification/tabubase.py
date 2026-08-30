"""Typed local evidence for the first two TabUBase verification stages."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from tabu_lab.contracts import PredictionBundle
from tabu_lab.evidence.schemas import EvidenceSchema
from tabu_lab.models.component_contract import inspect_tabu_base_composition
from tabu_lab.models.table_cell import TabUCellBaseModel

from .composability import SubstitutionStatus, assess_tabu_base_substitution


class TabUBaseVerificationStage(StrEnum):
    COMPONENT_CORRECTNESS = "component_correctness"
    COMPONENT_EVOLVABILITY = "component_evolvability"


class TabUBaseVerificationStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class VerificationCheck(EvidenceSchema):
    check_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    passed: bool


class TabUBaseLocalVerification(EvidenceSchema):
    """A content-addressed assertion, deliberately not a formal receipt."""

    schema_version: Literal["tabu.tabubase-local-verification.v1"]
    evidence_status: Literal["local_unissued"]
    stage: TabUBaseVerificationStage
    contract_id: Literal["tabu.cell.base"]
    contract_version: Literal["0.2.0"]
    model_spec_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_id: str = Field(min_length=1)
    reference_composition_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_composition_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    checks: tuple[VerificationCheck, ...] = Field(min_length=1)
    status: TabUBaseVerificationStatus
    claim_boundary: Literal[
        "local component verification only; not a training result, formal receipt, "
        "or accepted claim"
    ]

    @model_validator(mode="after")
    def status_matches_checks(self) -> TabUBaseLocalVerification:
        expected = (
            TabUBaseVerificationStatus.PASS
            if all(check.passed for check in self.checks)
            else TabUBaseVerificationStatus.FAIL
        )
        if self.status is not expected:
            raise ValueError("verification status must be derived from the checks")
        if self.stage is TabUBaseVerificationStage.COMPONENT_CORRECTNESS:
            if self.candidate_composition_hash is not None:
                raise ValueError("component correctness cannot carry a candidate composition")
        elif self.candidate_composition_hash is None:
            raise ValueError("component evolvability requires a candidate composition")
        return self


def verify_tabu_base_component_correctness(
    model: TabUCellBaseModel,
) -> TabUBaseLocalVerification:
    """Close Unit/ModelSpec/builder/component identity for one built model."""

    composition = inspect_tabu_base_composition(model)
    checks = (
        VerificationCheck(check_id="model_spec_identity", passed=True),
        VerificationCheck(
            check_id="cell_unit_semantics",
            passed=composition.unit_semantics == "table_cell_as_unit",
        ),
        VerificationCheck(
            check_id="runtime_components_bound",
            passed=composition.declaration_status == "model_spec_declared",
        ),
        VerificationCheck(
            check_id="truth_sidecar_boundary",
            passed=composition.truth_boundary == "loss_sidecar_step_5_only",
        ),
    )
    status = (
        TabUBaseVerificationStatus.PASS
        if all(check.passed for check in checks)
        else TabUBaseVerificationStatus.FAIL
    )
    return TabUBaseLocalVerification(
        schema_version="tabu.tabubase-local-verification.v1",
        evidence_status="local_unissued",
        stage=TabUBaseVerificationStage.COMPONENT_CORRECTNESS,
        contract_id="tabu.cell.base",
        contract_version="0.2.0",
        model_spec_hash=composition.model_spec_hash,
        profile_id=composition.profile_id,
        reference_composition_hash=composition.composition_hash,
        checks=checks,
        status=status,
        claim_boundary=(
            "local component verification only; not a training result, formal receipt, "
            "or accepted claim"
        ),
    )


def verify_tabu_base_component_evolvability(
    *,
    reference_model: TabUCellBaseModel,
    candidate_model: TabUCellBaseModel,
    reference_prediction: PredictionBundle,
    candidate_prediction: PredictionBundle,
    expected_axis: str,
) -> TabUBaseLocalVerification:
    """Record one bounded built-in component substitution as local evidence."""

    reference = inspect_tabu_base_composition(reference_model)
    candidate = inspect_tabu_base_composition(candidate_model)
    assessment = assess_tabu_base_substitution(
        reference_model=reference_model,
        candidate_model=candidate_model,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis=expected_axis,
    )
    checks = (
        VerificationCheck(
            check_id="one_declared_axis_changed",
            passed=assessment.changed_axes == (expected_axis,),
        ),
        VerificationCheck(check_id="forward_interface_stable", passed=assessment.interface_stable),
        VerificationCheck(check_id="predictions_bound", passed=assessment.predictions_bound),
        VerificationCheck(
            check_id="input_evidence_matched",
            passed=assessment.input_evidence_matched,
        ),
        VerificationCheck(
            check_id="components_declared",
            passed=assessment.components_declared,
        ),
        VerificationCheck(
            check_id="non_target_config_stable",
            passed=assessment.non_target_config_stable,
        ),
        VerificationCheck(
            check_id="variant_identity_changed",
            passed=assessment.variant_identity_changed,
        ),
    )
    status = (
        TabUBaseVerificationStatus.PASS
        if assessment.status is SubstitutionStatus.PASS
        else TabUBaseVerificationStatus.FAIL
    )
    return TabUBaseLocalVerification(
        schema_version="tabu.tabubase-local-verification.v1",
        evidence_status="local_unissued",
        stage=TabUBaseVerificationStage.COMPONENT_EVOLVABILITY,
        contract_id="tabu.cell.base",
        contract_version="0.2.0",
        model_spec_hash=reference.model_spec_hash,
        profile_id=reference.profile_id,
        reference_composition_hash=reference.composition_hash,
        candidate_composition_hash=candidate.composition_hash,
        checks=checks,
        status=status,
        claim_boundary=(
            "local component verification only; not a training result, formal receipt, "
            "or accepted claim"
        ),
    )


__all__ = [
    "TabUBaseLocalVerification",
    "TabUBaseVerificationStage",
    "TabUBaseVerificationStatus",
    "VerificationCheck",
    "verify_tabu_base_component_correctness",
    "verify_tabu_base_component_evolvability",
]
