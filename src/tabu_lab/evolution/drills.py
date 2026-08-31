"""Canonical small evolution drills used as non-evidentiary projections."""

from __future__ import annotations

from .impact import impact_report
from .models import ImpactReport
from .repository import EvolutionRepository

BASE_PROGRAM = "tabu.pretraining.query-base@1.0.0"
EVOLUTION_DRILLS = {
    "component-replacement.json": (
        BASE_PROGRAM,
        "tabu.pretraining.query-base-component-adapter@1.1.0-exercise",
    ),
    "evaluation-vnext.json": (
        BASE_PROGRAM,
        "tabu.pretraining.query-base-eval-v2@1.1.0-exercise",
    ),
    "generator-vnext.json": (
        "tabu.pretraining.query-base-generator-v2-projectable@1.0.0-exercise",
        "tabu.pretraining.query-base-generator-v3@1.1.0-exercise",
    ),
    "math-contract-vnext.json": (
        BASE_PROGRAM,
        "tabu.pretraining.query-base-math-exercise@1.1.0-exercise",
    ),
    "query-base-v3-mainline.json": (
        "tabu.pretraining.query-base@1.1.0",
        "tabu.pretraining.query-base@1.2.0",
    ),
    "query-row-v3-mainline.json": (
        "tabu.pretraining.query-row@1.1.0",
        "tabu.pretraining.query-row@1.2.0",
    ),
}


def build_evolution_drill_reports(
    repository: EvolutionRepository,
) -> dict[str, ImpactReport]:
    return {
        filename: impact_report(
            repository,
            repository.resolve(source),
            repository.resolve(target),
        )
        for filename, (source, target) in EVOLUTION_DRILLS.items()
    }


__all__ = ["BASE_PROGRAM", "EVOLUTION_DRILLS", "build_evolution_drill_reports"]
