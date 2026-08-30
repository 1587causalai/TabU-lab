#!/usr/bin/env python3
"""Build or verify the checked-in Evaluation Foundry JSON Schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.adapters.eval_data_freeze import EvalDataAuthorityFreezeManifest
from tabu_lab.adapters.eval_data_workflow import (
    EvalDataCheckReport,
    EvalDataPreparationRequest,
    PreparedEvalDataBundle,
)
from tabu_lab.evaluation.foundry import ComparisonReport, EvalResult, EvalSuiteSpec

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_TYPES = {
    "eval-data-authority-freeze": EvalDataAuthorityFreezeManifest,
    "eval-data-check-report": EvalDataCheckReport,
    "eval-data-preparation-request": EvalDataPreparationRequest,
    "eval-data-prepared-bundle": PreparedEvalDataBundle,
    "eval-comparison": ComparisonReport,
    "eval-result": EvalResult,
    "eval-suite": EvalSuiteSpec,
}


def _render(name: str) -> bytes:
    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://research.wehub.us/schemas/{name}.schema.json",
        **SCHEMA_TYPES[name].model_json_schema(),
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def check() -> tuple[str, ...]:
    issues: list[str] = []
    for name in sorted(SCHEMA_TYPES):
        path = ROOT / "schemas" / f"{name}.schema.json"
        if not path.is_file():
            issues.append(f"missing evaluation schema: {path.name}")
        elif path.read_bytes() != _render(name):
            issues.append(f"stale evaluation schema: {path.name}")
    return tuple(issues)


def build() -> None:
    for name in sorted(SCHEMA_TYPES):
        (ROOT / "schemas" / f"{name}.schema.json").write_bytes(_render(name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        issues = check()
        if issues:
            raise SystemExit("\n".join(issues))
        print("PASS: checked-in evaluation schemas match runtime contracts")
        return 0
    build()
    print(f"WROTE: {len(SCHEMA_TYPES)} evaluation schemas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
