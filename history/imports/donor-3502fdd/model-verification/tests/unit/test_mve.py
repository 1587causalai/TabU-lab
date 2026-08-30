from __future__ import annotations

import json

import pytest

from tabu_lab.models import BuilderRegistry
from tabu_lab.verification import (
    AssessmentOutcome,
    EvidenceLevel,
    VerificationResult,
    list_suites,
    run_suite,
)
from tabu_lab.verification.status import build_status


def test_mve_suites_are_allow_listed_and_round_trip() -> None:
    suites = list_suites()
    assert {suite.suite_id for suite in suites} == {
        "component-contract-v0",
        "architecture-evolvability-v0",
    }
    assert all(check.check_id for suite in suites for check in suite.checks)
    assert all(suite.suite_hash == suite.content_hash for suite in suites)


def test_unknown_check_id_fails_closed() -> None:
    from tabu_lab.verification.registry import VerificationRegistryError, get_check

    with pytest.raises(VerificationRegistryError):
        get_check("arbitrary.module.call")


def test_builder_registry_rejects_duplicate_and_supports_extension() -> None:
    registry = BuilderRegistry()
    registry.register("example", lambda **_: "ok")
    assert registry.build("example") == "ok"
    with pytest.raises(ValueError):
        registry.register("example", lambda **_: "again")


def test_tab4do_status_is_not_applicable() -> None:
    report = build_status(contract_id="tabu4do")
    row = report.rows[0]
    assert row.component_correctness.outcome is AssessmentOutcome.NOT_APPLICABLE
    assert row.component_correctness.evidence_level is EvidenceLevel.NONE


def test_component_suite_result_is_hash_bound(tmp_path) -> None:
    suite = next(item for item in list_suites() if item.suite_id == "component-contract-v0")
    result = run_suite("tabu.unit_pair", suite)
    assert result.model_spec_hash
    assert result.composition is not None
    assert result.evidence_level is EvidenceLevel.LOCAL_UNISSUED
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result.model_dump(mode="json")), encoding="utf-8")
    parsed = VerificationResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert parsed.result_hash == result.result_hash


def test_tabubase_component_suite_declares_both_public_profiles() -> None:
    suite = next(item for item in list_suites() if item.suite_id == "component-contract-v0")
    check_ids = {check.check_id for check in suite.checks}
    assert "component.profile_matrix" in check_ids
