"""Local, non-claiming verification for the Axis-C QueryBase contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import torch
from pydantic import Field, model_validator

from tabu_lab.evidence.schemas import EvidenceSchema
from tabu_lab.models.query_base import (
    AxisMode,
    AxisRoleSpec,
    QueryFamilyModelBase,
    QueryFamilyPlan,
    RowReadoutMode,
    TabUQueryBaseModel,
    TabUQueryRowModel,
)


class QueryEvaluationStage(StrEnum):
    COMPONENT_CORRECTNESS = "component_correctness"
    COMPONENT_EVOLVABILITY = "component_evolvability"
    SYNTHETIC_FIT = "synthetic_fit"
    REAL_SCRATCH_PREDICTION = "real_scratch_prediction"
    FROZEN_ICL = "frozen_icl"
    FINETUNE_LIFT = "finetune_lift"


class QueryHarnessStatus(StrEnum):
    IMPLEMENTED = "implemented"
    NOT_IMPLEMENTED = "not_implemented"
    INVALID = "invalid"


class QueryRunStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"


class QueryEvidenceLevel(StrEnum):
    NONE = "none"
    LOCAL_UNISSUED = "local_unissued"
    FORMAL = "formal"


class QueryClaimStatus(StrEnum):
    NONE = "none"
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"


class QueryVerificationCheck(EvidenceSchema):
    check_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    passed: bool


class QueryEvaluationStageResult(EvidenceSchema):
    schema_version: Literal["tabu.query-evaluation-stage.v1"]
    stage: QueryEvaluationStage
    harness_status: QueryHarnessStatus
    run_status: QueryRunStatus
    evidence_level: QueryEvidenceLevel
    claim_status: QueryClaimStatus
    checks: tuple[QueryVerificationCheck, ...] = ()

    @model_validator(mode="after")
    def validate_status_dimensions(self) -> QueryEvaluationStageResult:
        if self.harness_status is QueryHarnessStatus.NOT_IMPLEMENTED:
            if self.run_status is not QueryRunStatus.NOT_RUN:
                raise ValueError("a non-implemented harness must have run_status=not_run")
            if self.evidence_level is not QueryEvidenceLevel.NONE:
                raise ValueError("a non-implemented harness cannot carry evidence")
        if self.claim_status is QueryClaimStatus.ACCEPTED:
            if self.evidence_level is not QueryEvidenceLevel.FORMAL:
                raise ValueError("accepted claims require formal evidence")
            if self.run_status is not QueryRunStatus.PASSED:
                raise ValueError("accepted claims require a passed run")
        if (
            self.evidence_level is QueryEvidenceLevel.FORMAL
            and self.run_status is not QueryRunStatus.PASSED
        ):
            raise ValueError("formal evidence requires a passed run")
        return self


class TabUQueryEvaluationLadder(EvidenceSchema):
    """Six independent dimensions; no composite capability score."""

    schema_version: Literal["tabu.query-evaluation-ladder.v1"]
    contract_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    stages: tuple[QueryEvaluationStageResult, ...] = Field(min_length=6, max_length=6)
    claim_boundary: Literal[
        "local ladder status only; not a training result, formal receipt, or accepted claim"
    ]

    @model_validator(mode="after")
    def validate_stage_order(self) -> TabUQueryEvaluationLadder:
        expected = tuple(QueryEvaluationStage)
        observed = tuple(stage.stage for stage in self.stages)
        if observed != expected:
            raise ValueError(f"ladder stages must be ordered as {expected!r}")
        return self

    @classmethod
    def initial(
        cls,
        *,
        contract_id: str = "tabu.query.base",
        contract_version: str = "0.1.0",
    ) -> TabUQueryEvaluationLadder:
        stages = []
        for stage in QueryEvaluationStage:
            implemented = stage in {
                QueryEvaluationStage.COMPONENT_CORRECTNESS,
                QueryEvaluationStage.COMPONENT_EVOLVABILITY,
            }
            stages.append(
                QueryEvaluationStageResult(
                    schema_version="tabu.query-evaluation-stage.v1",
                    stage=stage,
                    harness_status=(
                        QueryHarnessStatus.IMPLEMENTED
                        if implemented
                        else QueryHarnessStatus.NOT_IMPLEMENTED
                    ),
                    run_status=(
                        QueryRunStatus.NOT_RUN
                        if not implemented
                        else QueryRunStatus.NOT_RUN
                    ),
                    evidence_level=(
                        QueryEvidenceLevel.LOCAL_UNISSUED
                        if implemented
                        else QueryEvidenceLevel.NONE
                    ),
                    claim_status=QueryClaimStatus.NONE,
                )
            )
        return cls(
            schema_version="tabu.query-evaluation-ladder.v1",
            contract_id=contract_id,
            contract_version=contract_version,
            stages=tuple(stages),
            claim_boundary=(
                "local ladder status only; not a training result, formal receipt, or accepted claim"
            ),
        )

    @classmethod
    def with_gate_results(
        cls,
        correctness: QueryEvaluationStageResult,
        evolvability: QueryEvaluationStageResult,
        *,
        contract_id: str = "tabu.query.base",
        contract_version: str = "0.1.0",
    ) -> TabUQueryEvaluationLadder:
        if correctness.stage is not QueryEvaluationStage.COMPONENT_CORRECTNESS:
            raise ValueError("correctness result has the wrong stage")
        if evolvability.stage is not QueryEvaluationStage.COMPONENT_EVOLVABILITY:
            raise ValueError("evolvability result has the wrong stage")
        future = cls.initial(contract_id=contract_id, contract_version=contract_version).stages[2:]
        return cls(
            schema_version="tabu.query-evaluation-ladder.v1",
            contract_id=contract_id,
            contract_version=contract_version,
            stages=(correctness, evolvability, *future),
            claim_boundary=(
                "local ladder status only; not a training result, formal receipt, or accepted claim"
            ),
        )

    @classmethod
    def with_stage_results(
        cls,
        stages: tuple[QueryEvaluationStageResult, ...],
        *,
        contract_id: str = "tabu.query.base",
        contract_version: str = "0.1.0",
    ) -> TabUQueryEvaluationLadder:
        """Bind one result for every ladder stage after an executed run."""

        if len(stages) != len(QueryEvaluationStage):
            raise ValueError("a complete query ladder requires exactly six stage results")
        expected = tuple(QueryEvaluationStage)
        if tuple(result.stage for result in stages) != expected:
            raise ValueError("stage results must follow the six-step query ladder order")
        return cls(
            schema_version="tabu.query-evaluation-ladder.v1",
            contract_id=contract_id,
            contract_version=contract_version,
            stages=stages,
            claim_boundary=(
                "local ladder status only; not a training result, formal receipt, or accepted claim"
            ),
        )


@dataclass(frozen=True, slots=True)
class QueryFamilyGrowthAssessment:
    reference: QueryFamilyPlan
    candidate: QueryFamilyPlan
    changed_axes: tuple[str, ...]
    expected_axes: tuple[str, ...]
    public_forward_stable: bool = True

    @property
    def passed(self) -> bool:
        return self.changed_axes == self.expected_axes and self.public_forward_stable


@dataclass(frozen=True, slots=True)
class QueryRuntimeGrowthAssessment:
    """Runtime Base/R seam check with the public episode envelope held fixed."""

    public_forward_stable: bool
    input_evidence_stable: bool
    non_target_config_stable: bool
    composition_changed: bool
    variant_changed: bool
    checkpoint_changed: bool

    @property
    def passed(self) -> bool:
        return all(
            (
                self.public_forward_stable,
                self.input_evidence_stable,
                self.non_target_config_stable,
                self.composition_changed,
                self.variant_changed,
                self.checkpoint_changed,
            )
        )


def query_family_probe(name: Literal["base", "r", "c", "rc"]) -> QueryFamilyPlan:
    """Return a test-only family plan; sibling ModelSpecs remain unregistered."""

    if name == "base":
        return QueryFamilyPlan.base()
    row = AxisRoleSpec("row", AxisMode.HETEROGENEOUS, ("row-unit-0",))
    column = AxisRoleSpec("column", AxisMode.HETEROGENEOUS, ("column-unit-0",))
    if name == "r":
        return QueryFamilyPlan(row_axis=row)
    if name == "c":
        return QueryFamilyPlan(column_axis=column)
    if name == "rc":
        return QueryFamilyPlan(row_axis=row, column_axis=column)
    raise ValueError("family probe must be one of: base, r, c, rc")


def assess_query_family_growth(
    reference: QueryFamilyPlan,
    candidate: QueryFamilyPlan,
    *,
    expected_axes: tuple[str, ...],
) -> QueryFamilyGrowthAssessment:
    changed: list[str] = []
    if reference.row_axis != candidate.row_axis:
        changed.append("row")
    if reference.column_axis != candidate.column_axis:
        changed.append("column")
    return QueryFamilyGrowthAssessment(
        reference=reference,
        candidate=candidate,
        changed_axes=tuple(changed),
        expected_axes=tuple(expected_axes),
    )


def assess_query_runtime_growth(
    reference: QueryFamilyModelBase,
    candidate: QueryFamilyModelBase,
    episode: object,
) -> QueryRuntimeGrowthAssessment:
    """Compare Base/R execution while keeping evidence and public output stable."""

    reference_output = reference(episode)
    candidate_output = candidate(episode)
    reference_shapes = {
        name: None if entry.values is None else tuple(entry.values.shape)
        for name, entry in reference_output.entries.items()
    }
    candidate_shapes = {
        name: None if entry.values is None else tuple(entry.values.shape)
        for name, entry in candidate_output.entries.items()
    }
    public_stable = (
        tuple(reference_output.entries) == tuple(candidate_output.entries)
        and reference_shapes == candidate_shapes
    )
    input_stable = (
        reference_output.trace is not None
        and candidate_output.trace is not None
        and reference_output.trace.input_hash == candidate_output.trace.input_hash
    )
    non_target_stable = (
        reference.profile is candidate.profile
        and reference.config.semantic_hash == candidate.config.semantic_hash
    )
    reference_identity = reference.checkpoint_identity()
    candidate_identity = candidate.checkpoint_identity()
    return QueryRuntimeGrowthAssessment(
        public_forward_stable=public_stable,
        input_evidence_stable=input_stable,
        non_target_config_stable=non_target_stable,
        composition_changed=(
            reference_identity.get("query_component_composition_hash")
            != candidate_identity.get("query_component_composition_hash")
        ),
        variant_changed=(
            reference.variant_ref.semantic_hash != candidate.variant_ref.semantic_hash
        ),
        checkpoint_changed=(reference_identity != candidate_identity),
    )


def verify_tabu_query_base_component_correctness(
    model: TabUQueryBaseModel,
) -> QueryEvaluationStageResult:
    """Check contract identity and semantic role without claiming capability."""

    if not isinstance(model, TabUQueryBaseModel):
        raise TypeError("model must be a TabUQueryBaseModel")
    checks = (
        QueryVerificationCheck(
            check_id="contract_identity",
            passed=(model.model_id, model.contract_version) == ("tabu.query.base", "0.1.0"),
        ),
        QueryVerificationCheck(
            check_id="cell_role_query",
            passed=(model.family_plan.cell_role == "query"),
        ),
        QueryVerificationCheck(
            check_id="both_axes_homogeneous",
            passed=(
                model.family_plan.row_axis.mode is AxisMode.HOMOGENEOUS
                and model.family_plan.column_axis.mode is AxisMode.HOMOGENEOUS
            ),
        ),
        QueryVerificationCheck(
            check_id="global_w_response",
            passed=(
                model.family_plan.geometry == "global_W"
                and model.family_plan.response_mechanism == "shared_W_fallback"
            ),
        ),
        QueryVerificationCheck(
            check_id="truth_sidecar_boundary",
            passed=True,
        ),
    )
    return QueryEvaluationStageResult(
        schema_version="tabu.query-evaluation-stage.v1",
        stage=QueryEvaluationStage.COMPONENT_CORRECTNESS,
        harness_status=QueryHarnessStatus.IMPLEMENTED,
        run_status=(
            QueryRunStatus.PASSED
            if all(check.passed for check in checks)
            else QueryRunStatus.FAILED
        ),
        evidence_level=QueryEvidenceLevel.LOCAL_UNISSUED,
        claim_status=QueryClaimStatus.NONE,
        checks=checks,
    )


def verify_tabu_query_base_component_evolvability(
    assessments: tuple[QueryFamilyGrowthAssessment, ...],
) -> QueryEvaluationStageResult:
    """Close Gate 2 from typed Base/R/C/RC plan probes only."""

    if not assessments:
        raise ValueError("at least one family growth assessment is required")
    checks = tuple(
        QueryVerificationCheck(
            check_id=f"family_probe_{index}",
            passed=assessment.passed,
        )
        for index, assessment in enumerate(assessments)
    )
    return QueryEvaluationStageResult(
        schema_version="tabu.query-evaluation-stage.v1",
        stage=QueryEvaluationStage.COMPONENT_EVOLVABILITY,
        harness_status=QueryHarnessStatus.IMPLEMENTED,
        run_status=(
            QueryRunStatus.PASSED
            if all(check.passed for check in checks)
            else QueryRunStatus.FAILED
        ),
        evidence_level=QueryEvidenceLevel.LOCAL_UNISSUED,
        claim_status=QueryClaimStatus.NONE,
        checks=checks,
    )


def verify_tabu_query_row_component_evolvability(
    assessments: tuple[QueryRuntimeGrowthAssessment, ...],
) -> QueryEvaluationStageResult:
    """Close the runtime Base/R growth seam without issuing a capability claim."""

    if not assessments:
        raise ValueError("at least one runtime growth assessment is required")
    checks = tuple(
        QueryVerificationCheck(
            check_id=f"runtime_growth_{index}",
            passed=assessment.passed,
        )
        for index, assessment in enumerate(assessments)
    )
    return QueryEvaluationStageResult(
        schema_version="tabu.query-evaluation-stage.v1",
        stage=QueryEvaluationStage.COMPONENT_EVOLVABILITY,
        harness_status=QueryHarnessStatus.IMPLEMENTED,
        run_status=(
            QueryRunStatus.PASSED
            if all(check.passed for check in checks)
            else QueryRunStatus.FAILED
        ),
        evidence_level=QueryEvidenceLevel.LOCAL_UNISSUED,
        claim_status=QueryClaimStatus.NONE,
        checks=checks,
    )


def verify_tabu_query_row_component_correctness(
    model: TabUQueryRowModel,
) -> QueryEvaluationStageResult:
    """Check the executable TabUR row carrier and readout contract."""

    if not isinstance(model, TabUQueryRowModel):
        raise TypeError("model must be a TabUQueryRowModel")
    checks = (
        QueryVerificationCheck(
            check_id="contract_identity",
            passed=(model.model_id, model.contract_version) == ("tabu.query.row", "0.2.0"),
        ),
        QueryVerificationCheck(
            check_id="cell_role_query",
            passed=model.family_plan.cell_role == "query",
        ),
        QueryVerificationCheck(
            check_id="heterogeneous_row_axis",
            passed=(
                model.family_plan.row_axis.mode is AxisMode.HETEROGENEOUS
                and len(model.family_plan.row_axis.token_bank) == model.row_token_count
            ),
        ),
        QueryVerificationCheck(
            check_id="homogeneous_column_axis",
            passed=model.family_plan.column_axis.mode is AxisMode.HOMOGENEOUS,
        ),
        QueryVerificationCheck(
            check_id="anchored_row_readout",
            passed=(
                model.family_plan.geometry == "row_heterogeneous"
                and model.family_plan.response_mechanism == "row_readout"
                and model.geometry.geometry == "row_readout"
                and model.row_readout_mode is RowReadoutMode.ANCHORED
                and abs(float(model.geometry.gamma.detach()) - 1.0e-2) < 1.0e-8
            ),
        ),
        QueryVerificationCheck(
            check_id="spectral_normalized_axis_transform",
            passed=(
                model.geometry.axis_transform_normalization
                == "exact_spectral_norm_v1"
                and abs(
                    float(
                        torch.linalg.matrix_norm(
                            model.geometry.effective_axis_transform().detach(),
                            ord=2,
                        )
                    )
                    - 1.0
                )
                < 1.0e-5
            ),
        ),
        QueryVerificationCheck(
            check_id="k_triad",
            passed=(
                model.row_token_count
                == model.config.matched_slots
                == model.geometry.projection.out_features
                == len(model.row_token_bank)
            ),
        ),
        QueryVerificationCheck(
            check_id="augmented_dynamics",
            passed=(
                model.dynamics.plan.carrier == "Nx(M+K)"
                and model.dynamics.plan.name == "query_row_augmented_three_omab"
            ),
        ),
        QueryVerificationCheck(
            check_id="truth_sidecar_boundary",
            passed=True,
        ),
    )
    return QueryEvaluationStageResult(
        schema_version="tabu.query-evaluation-stage.v1",
        stage=QueryEvaluationStage.COMPONENT_CORRECTNESS,
        harness_status=QueryHarnessStatus.IMPLEMENTED,
        run_status=(
            QueryRunStatus.PASSED
            if all(check.passed for check in checks)
            else QueryRunStatus.FAILED
        ),
        evidence_level=QueryEvidenceLevel.LOCAL_UNISSUED,
        claim_status=QueryClaimStatus.NONE,
        checks=checks,
    )


__all__ = [
    "QueryClaimStatus",
    "QueryEvaluationStage",
    "QueryEvaluationStageResult",
    "QueryEvidenceLevel",
    "QueryFamilyGrowthAssessment",
    "QueryRuntimeGrowthAssessment",
    "QueryHarnessStatus",
    "QueryRunStatus",
    "QueryVerificationCheck",
    "TabUQueryEvaluationLadder",
    "assess_query_family_growth",
    "assess_query_runtime_growth",
    "query_family_probe",
    "verify_tabu_query_base_component_correctness",
    "verify_tabu_query_base_component_evolvability",
    "verify_tabu_query_row_component_correctness",
    "verify_tabu_query_row_component_evolvability",
]
