"""Structured MVE suite loading and execution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from tabu_lab.contracts import canonical_hash
from tabu_lab.registry import get_model_spec

from .composition import describe_model
from .contracts import (
    AssessmentOutcome,
    EvidenceLevel,
    VerificationResult,
    VerificationSuite,
)
from .registry import get_check


class VerificationRunnerError(ValueError):
    """A suite or result cannot be safely loaded or written."""


def suite_root() -> Path:
    return Path(__file__).resolve().parent / "suites"


def list_suites() -> tuple[VerificationSuite, ...]:
    suites: list[VerificationSuite] = []
    for path in sorted(suite_root().glob("*.yaml")):
        suites.append(load_suite(path))
    return tuple(suites)


def load_suite(path: str | Path) -> VerificationSuite:
    source = Path(path)
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        suite = VerificationSuite.model_validate(raw)
        for check in suite.checks:
            get_check(check.check_id)
        return suite
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise VerificationRunnerError(f"invalid verification suite {source}: {exc}") from exc


def validate_suites() -> tuple[VerificationSuite, ...]:
    return list_suites()


def _derive_outcome(outcomes: list[AssessmentOutcome]) -> AssessmentOutcome:
    if any(value is AssessmentOutcome.FAILED for value in outcomes):
        return AssessmentOutcome.FAILED
    if any(value is AssessmentOutcome.BLOCKED for value in outcomes):
        return AssessmentOutcome.BLOCKED
    applicable = [value for value in outcomes if value is not AssessmentOutcome.NOT_APPLICABLE]
    if not applicable:
        return AssessmentOutcome.NOT_APPLICABLE
    if any(value is AssessmentOutcome.NOT_RUN for value in applicable):
        return AssessmentOutcome.NOT_RUN
    return AssessmentOutcome.PASSED


def run_suite(
    contract_id: str,
    suite: VerificationSuite,
    *,
    evidence_level: EvidenceLevel = EvidenceLevel.LOCAL_UNISSUED,
    receipt_ref: str | None = None,
    review_ref: str | None = None,
    context: Mapping[str, Any] | None = None,
) -> VerificationResult:
    spec = get_model_spec(contract_id)
    ctx = dict(context or {})
    checks = tuple(get_check(item.check_id)(contract_id, ctx) for item in suite.checks)
    outcome = _derive_outcome([item.outcome for item in checks])
    blockers = tuple(item.detail for item in checks if item.outcome is AssessmentOutcome.BLOCKED)
    limitations = (
        ("local verification is not a formal receipt or independent review",)
        if evidence_level is EvidenceLevel.LOCAL_UNISSUED
        else ()
    )
    result_id = f"mve-{suite.suite_id}-{contract_id}-{suite.suite_hash[:12]}"
    composition = None
    if contract_id != "tabu4do":
        try:
            from .probes import _model

            model, _ = _model(contract_id)
            composition = describe_model(
                model,
                contract_id=contract_id,
                contract_version=spec.contract_version,
            )
        except Exception:
            composition = None
    return VerificationResult(
        result_id=result_id,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        axis=suite.axis,
        contract_id=contract_id,
        contract_version=spec.contract_version,
        model_spec_hash=canonical_hash(spec),
        composition=composition,
        outcome=outcome,
        evidence_level=evidence_level,
        checks=checks,
        receipt_ref=receipt_ref,
        review_ref=review_ref,
        blockers=blockers,
        limitations=limitations,
        claim_boundary=suite.claim_boundary,
    )


def write_result(result: VerificationResult, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def read_result(path: str | Path) -> VerificationResult:
    source = Path(path)
    try:
        return VerificationResult.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VerificationRunnerError(f"invalid verification result {source}: {exc}") from exc


__all__ = [
    "VerificationRunnerError",
    "list_suites",
    "load_suite",
    "read_result",
    "run_suite",
    "suite_root",
    "validate_suites",
    "write_result",
]
