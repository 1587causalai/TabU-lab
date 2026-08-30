"""Rebuildable four-axis MVE status projection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import Field

from tabu_lab.contracts import canonical_hash
from tabu_lab.evidence.schemas import EvidenceSchema
from tabu_lab.registry import get_model_spec, list_models

from .contracts import AssessmentOutcome, EvidenceLevel, VerificationResult
from .runner import load_suite, read_result


class MVEAxisCell(EvidenceSchema):
    schema_version: Literal["tabu.mve-axis-cell.v1"] = "tabu.mve-axis-cell.v1"
    outcome: AssessmentOutcome
    evidence_level: EvidenceLevel
    references: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class MVEStatusRow(EvidenceSchema):
    schema_version: Literal["tabu.mve-status-row.v1"] = "tabu.mve-status-row.v1"
    contract_id: str = Field(min_length=1)
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    model_spec_ref: str
    component_correctness: MVEAxisCell
    architecture_evolvability: MVEAxisCell
    synthetic_fit: MVEAxisCell
    real_data_prediction: MVEAxisCell


class MVEStatusReport(EvidenceSchema):
    schema_version: Literal["tabu.mve-status-report.v1"] = "tabu.mve-status-report.v1"
    rows: tuple[MVEStatusRow, ...] = ()
    claim_boundary: str = (
        "MVE is a four-axis evidence projection; it emits no total score, ranking, "
        "promotion, or claim."
    )


def _result_for(
    contract_id: str,
    suite_id: str,
    results_root: Path,
) -> tuple[VerificationResult | None, str | None]:
    for path in sorted(results_root.rglob("*.json")) if results_root.is_dir() else ():
        try:
            result = read_result(path)
        except ValueError:
            continue
        if result.contract_id == contract_id and result.suite_id == suite_id:
            return result, path.as_posix()
    return None, None


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec_references(spec: object, *, contract_id: str) -> list[str]:
    return [
        f"specs/models/{contract_id}.yaml@sha256:{canonical_hash(spec)}",
        (f"{spec.upstream.repository}:{spec.upstream.path}@sha256:{spec.upstream.sha256}"),
    ]


def _verification_cell(
    contract_id: str, suite_id: str, suite_path: str, results_root: Path
) -> MVEAxisCell:
    result, result_path = _result_for(contract_id, suite_id, results_root)
    spec = get_model_spec(contract_id)
    refs = _spec_references(spec, contract_id=contract_id)
    suite = load_suite(Path(__file__).resolve().parent / "suites" / Path(suite_path).name)
    refs.append(f"{suite_path}@sha256:{suite.suite_hash}")
    if result_path:
        refs.append(f"{result_path}@sha256:{result.result_hash}")
    if result is None:
        return MVEAxisCell(
            outcome=AssessmentOutcome.NOT_RUN,
            evidence_level=EvidenceLevel.NONE,
            references=tuple(refs),
            limitations=("No structured verification result has been issued for this contract.",),
        )
    return MVEAxisCell(
        outcome=result.outcome,
        evidence_level=result.evidence_level,
        references=tuple(refs),
        blockers=result.blockers,
        limitations=result.limitations,
    )


def _synthetic_cell(contract_id: str, root: Path) -> MVEAxisCell:
    spec = get_model_spec(contract_id)
    preregs = sorted(
        root.glob(
            f"experiments/fit-first/F0/*{contract_id.replace('.', '-')}*/preregistration.yaml"
        )
    )
    if not preregs:
        preregs = sorted(root.glob("experiments/fit-first/F0/*/preregistration.yaml"))
        preregs = [
            path
            for path in preregs
            if path.read_text(encoding="utf-8").find(f"contract_id: {contract_id}") >= 0
        ]
    refs = [
        *_spec_references(spec, contract_id=contract_id),
        *(f"{path.relative_to(root).as_posix()}@sha256:{_file_hash(path)}" for path in preregs[:1]),
    ]
    return MVEAxisCell(
        outcome=AssessmentOutcome.NOT_RUN,
        evidence_level=EvidenceLevel.NONE,
        references=tuple(refs),
        limitations=(
            "Synthetic fit status is projected from F0/S1 receipts; no separate "
            "MVE runner is used.",
        ),
    )


def _real_data_cell(
    contract_id: str,
    root: Path,
    results_root: Path,
) -> MVEAxisCell:
    spec = get_model_spec(contract_id)
    prereg_ref = (
        "experiments/fit-first/R1/R1-001-tabul-sklearn-diabetes-regression-v1/preregistration.yaml"
    )
    refs = _spec_references(spec, contract_id=contract_id)
    prereg_path = root / prereg_ref
    if prereg_path.is_file():
        refs.append(f"{prereg_ref}@sha256:{_file_hash(prereg_path)}")
    else:
        refs.append(prereg_ref)
    if contract_id != "tabul":
        return MVEAxisCell(
            outcome=AssessmentOutcome.NOT_RUN,
            evidence_level=EvidenceLevel.NONE,
            references=tuple(refs),
            limitations=("Only the TabUL Diabetes R1 vertical slice is in scope for v0.",),
        )
    expected_hash = canonical_hash(get_model_spec("tabul"))
    from tabu_lab.experiments.r1_runner import R1RunReceipt

    if results_root.is_dir():
        for path in sorted(results_root.rglob("*.json")):
            try:
                receipt = R1RunReceipt.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if receipt.model_spec_hash != expected_hash:
                continue
            receipt_ref = (
                path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
            )
            refs.append(f"{receipt_ref}@sha256:{receipt.content_hash}")
            return MVEAxisCell(
                outcome=AssessmentOutcome(receipt.outcome),
                evidence_level=EvidenceLevel.LOCAL_UNISSUED,
                references=tuple(refs),
                limitations=(
                    "R1 receipt is local_unissued; model-artifact scoring and EvalSuite "
                    "handoff remain downstream.",
                ),
            )
    return MVEAxisCell(
        outcome=AssessmentOutcome.NOT_RUN,
        evidence_level=EvidenceLevel.NONE,
        references=tuple(refs),
        limitations=("Only the TabUL Diabetes R1 vertical slice is in scope for v0.",),
    )


def build_status(
    *,
    contract_id: str | None = None,
    repository: str | Path = ".",
    results_root: str | Path = "verification/results",
) -> MVEStatusReport:
    root = Path(repository).resolve()
    result_root = Path(results_root)
    if not result_root.is_absolute():
        result_root = root / result_root
    specs = list_models() if contract_id is None else (get_model_spec(contract_id),)
    rows: list[MVEStatusRow] = []
    for spec in sorted(specs, key=lambda item: item.contract_id):
        model_ref = f"specs/models/{spec.contract_id}.yaml@sha256:{canonical_hash(spec)}"
        if spec.maturity.build_state.value == "design_open":
            na = MVEAxisCell(
                outcome=AssessmentOutcome.NOT_APPLICABLE,
                evidence_level=EvidenceLevel.NONE,
                references=(model_ref,),
                blockers=("ModelSpec is design_open; no executable assessment is declared.",),
            )
            rows.append(
                MVEStatusRow(
                    contract_id=spec.contract_id,
                    contract_version=spec.contract_version,
                    model_spec_ref=model_ref,
                    component_correctness=na,
                    architecture_evolvability=na,
                    synthetic_fit=na,
                    real_data_prediction=na,
                )
            )
            continue
        rows.append(
            MVEStatusRow(
                contract_id=spec.contract_id,
                contract_version=spec.contract_version,
                model_spec_ref=model_ref,
                component_correctness=_verification_cell(
                    contract_id=spec.contract_id,
                    suite_id="component-contract-v0",
                    suite_path="verification/suites/component-contract-v0.yaml",
                    results_root=result_root,
                ),
                architecture_evolvability=_verification_cell(
                    contract_id=spec.contract_id,
                    suite_id="architecture-evolvability-v0",
                    suite_path="verification/suites/architecture-evolvability-v0.yaml",
                    results_root=result_root,
                ),
                synthetic_fit=_synthetic_cell(spec.contract_id, root),
                real_data_prediction=_real_data_cell(spec.contract_id, root, result_root),
            )
        )
    return MVEStatusReport(rows=tuple(rows))


__all__ = ["MVEAxisCell", "MVEStatusReport", "MVEStatusRow", "build_status"]
