from __future__ import annotations

from pathlib import Path

from tabu_lab.contracts import canonical_json
from tabu_lab.evolution import EvolutionRepository
from tabu_lab.evolution.drills import build_evolution_drill_reports

ROOT = Path(__file__).resolve().parents[2]


def test_checked_evolution_impact_projections_match_manifests() -> None:
    reports = build_evolution_drill_reports(EvolutionRepository.load(ROOT))

    assert set(reports) == {
        "component-replacement.json",
        "evaluation-vnext.json",
        "generator-vnext.json",
        "math-contract-vnext.json",
        "query-base-v3-mainline.json",
        "query-row-v3-mainline.json",
    }
    for filename, report in reports.items():
        expected = canonical_json(report.model_dump(mode="python")) + "\n"
        observed = (ROOT / "docs/reports/evolution-impact" / filename).read_text(
            encoding="utf-8"
        )
        assert observed == expected
