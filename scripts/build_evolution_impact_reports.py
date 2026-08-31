#!/usr/bin/env python3
"""Build or verify generated impact reports for the bounded evolution drills."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabu_lab.contracts import canonical_json
from tabu_lab.evolution import EvolutionRepository
from tabu_lab.evolution.drills import build_evolution_drill_reports

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "reports" / "evolution-impact"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    reports = build_evolution_drill_reports(EvolutionRepository.load(ROOT))
    stale: list[str] = []
    for filename, report in reports.items():
        path = OUTPUT / filename
        rendered = canonical_json(report.model_dump(mode="python")) + "\n"
        if arguments.check:
            actual = path.read_text(encoding="utf-8") if path.is_file() else ""
            if actual != rendered:
                stale.append(path.relative_to(ROOT).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"WROTE: {path.relative_to(ROOT)}")
    if stale:
        raise SystemExit("stale evolution impact projections: " + ", ".join(stale))
    if arguments.check:
        print("PASS: evolution impact projections are current")


if __name__ == "__main__":
    main()
