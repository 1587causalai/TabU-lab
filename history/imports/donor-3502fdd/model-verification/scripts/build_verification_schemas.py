#!/usr/bin/env python3
"""Build or check the checked-in MVE JSON Schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.verification import VerificationResult, VerificationSuite

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = {
    "verification-suite": VerificationSuite,
    "verification-result": VerificationResult,
}


def render(name: str) -> bytes:
    payload = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://research.wehub.us/schemas/{name}.schema.json",
        **SCHEMAS[name].model_json_schema(),
    }
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    issues: list[str] = []
    for name in sorted(SCHEMAS):
        path = ROOT / "schemas" / f"{name}.schema.json"
        expected = render(name)
        if args.check:
            if not path.is_file() or path.read_bytes() != expected:
                issues.append(f"stale or missing verification schema: {path.name}")
        else:
            path.write_bytes(expected)
    if issues:
        raise SystemExit("\n".join(issues))
    print(
        "PASS: checked-in MVE schemas match runtime contracts"
        if args.check
        else f"WROTE: {len(SCHEMAS)} MVE schemas"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
