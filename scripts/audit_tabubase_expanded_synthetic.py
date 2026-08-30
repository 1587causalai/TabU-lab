#!/usr/bin/env python3
"""Audit the TabUBase expanded synthetic generator before any training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_expanded_synthetic import (
    CONTEXT_CANDIDATE_INITIAL_ROWS,
    FROZEN_CONTEXT_ROWS_SCHEDULE,
    GENERATOR_VERSION,
    LONG_CONTEXT_CANDIDATE_ROWS,
    LONG_CONTEXT_ROWS_SCHEDULE,
    audit_expanded_synthetic_generator,
    audit_expanded_training_episode_universe,
    evaluate_selected_world_ridge_reference_gate,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-seed", type=int, default=1729)
    parser.add_argument("--coverage-worlds", type=int, default=2048)
    parser.add_argument("--training-compile-worlds", type=int, default=20_000)
    parser.add_argument("--selected-reference-worlds", type=int, default=8)
    parser.add_argument("--small-context", type=int, default=8)
    parser.add_argument("--large-context", type=int, default=128)
    parser.add_argument("--query-rows", type=int, default=128)
    parser.add_argument("--long-context-curriculum", action="store_true")
    args = parser.parse_args()

    context_rows_schedule = (
        LONG_CONTEXT_ROWS_SCHEDULE
        if args.long_context_curriculum
        else FROZEN_CONTEXT_ROWS_SCHEDULE
    )
    context_candidate_rows = (
        LONG_CONTEXT_CANDIDATE_ROWS
        if args.long_context_curriculum
        else CONTEXT_CANDIDATE_INITIAL_ROWS
    )

    generator_audit = audit_expanded_synthetic_generator(
        root_seed=args.root_seed,
        coverage_worlds=args.coverage_worlds,
        context_rows_schedule=context_rows_schedule,
        context_candidate_rows=context_candidate_rows,
    )
    training_universe_audit = audit_expanded_training_episode_universe(
        root_seed=args.root_seed,
        world_count=args.training_compile_worlds,
        context_rows_schedule=context_rows_schedule,
        context_candidate_rows=context_candidate_rows,
    )
    reference_gate = evaluate_selected_world_ridge_reference_gate(
        root_seed=args.root_seed,
        selected_worlds=args.selected_reference_worlds,
        small_context=args.small_context,
        large_context=args.large_context,
        query_rows=args.query_rows,
    )
    result = {
        "schema_version": "tabu.expanded-synthetic-stage-a-audit.v4",
        "status": "local_unissued",
        "generator_version": GENERATOR_VERSION,
        "root_seed": args.root_seed,
        "coverage_worlds": args.coverage_worlds,
        "training_compile_worlds": args.training_compile_worlds,
        "context_rows_schedule": list(context_rows_schedule),
        "context_candidate_rows": context_candidate_rows,
        "generator_audit": generator_audit,
        "training_universe_audit": training_universe_audit,
        "selected_world_reference_gate": reference_gate,
        "gates": {
            "G-D0": generator_audit["gates"]["G-D0"]["passed"],
            "G-D1": generator_audit["gates"]["G-D1"]["passed"],
            "G-D2": generator_audit["gates"]["G-D2"]["passed"],
            "G-D2U": training_universe_audit["passed"],
            "G-D3": reference_gate["passed"],
        },
        "passed": (
            generator_audit["passed"]
            and training_universe_audit["passed"]
            and reference_gate["passed"]
        ),
        "not_evaluated_gates": ["G-D4", "G-D5"],
        "claim_boundary": (
            "generator correctness and selected-world reference recovery only; "
            "no model training, frozen-ICL result, formal receipt, or model claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printed = result | {"result_path": str(args.output), "result_sha256": _sha256(args.output)}
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
