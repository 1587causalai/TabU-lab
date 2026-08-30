from __future__ import annotations

from tabu_lab.verification import AssessmentOutcome, list_suites, run_suite
from tabu_lab.verification.probes import (
    architecture_builder_registry,
    architecture_substitution_nonregression,
)

_ORACLE_KEYS = {
    "build",
    "forward_finite",
    "gradient_reachability",
    "seed_dtype_checkpoint",
    "truth_sidecar_abstention",
}


def test_substitution_nonregression_passes_on_tabuf() -> None:
    result = architecture_substitution_nonregression("tabuf", {})
    assert result.outcome is AssessmentOutcome.PASSED
    seam_reports = result.metrics["seam_reports"]
    assert set(seam_reports) == {
        "numeric_terminal:nadaraya_watson",
        "numeric_terminal:local_linear",
        "geometry_normalization:rms_unit",
        "block_kind:mab",
    }
    for outcomes in seam_reports.values():
        assert set(outcomes) == _ORACLE_KEYS
        assert all(value.startswith("passed") for value in outcomes.values())


def test_substitution_nonregression_covers_label_plans_on_tabul() -> None:
    result = architecture_substitution_nonregression("tabul", {})
    assert result.outcome is AssessmentOutcome.PASSED
    assert set(result.metrics["seam_reports"]) == {
        "numeric_terminal:nadaraya_watson",
        "numeric_terminal:local_linear",
        "geometry_normalization:rms_unit",
        "block_kind:mab",
        "label_address_plan:predictor_only_per_label_v1",
        "label_address_plan:predictor_unit_linked_per_label_v2",
    }


def test_substitution_nonregression_boundaries() -> None:
    assert (
        architecture_substitution_nonregression("tabu4do", {}).outcome
        is AssessmentOutcome.NOT_APPLICABLE
    )
    assert (
        architecture_substitution_nonregression("not.a.contract", {}).outcome
        is AssessmentOutcome.BLOCKED
    )


def test_builder_registry_extension_probe_exercises_real_behavior() -> None:
    result = architecture_builder_registry("tabuf", {})
    assert result.outcome is AssessmentOutcome.PASSED
    assert result.metrics["baseline_builder_count"] >= 1
    assert result.metrics["probe_extension_id"] == "probe.extension"


def test_evolvability_suite_carries_new_check() -> None:
    suite = next(
        item for item in list_suites() if item.suite_id == "architecture-evolvability-v0"
    )
    assert suite.suite_version == "0.2.0"
    result = run_suite("tabuf", suite)
    check_ids = {check.check_id for check in result.checks}
    assert "architecture.substitution_nonregression" in check_ids
