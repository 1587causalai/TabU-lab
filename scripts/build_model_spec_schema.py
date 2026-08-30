#!/usr/bin/env python3
"""Build or verify the checked-in public ModelSpec JSON Schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.registry import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "schemas" / "model-spec.schema.json"


def rendered_schema() -> str:
    return json.dumps(
        ModelSpec.model_json_schema(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = rendered_schema()
    if arguments.check:
        actual = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
        if actual != expected:
            raise SystemExit(
                "schemas/model-spec.schema.json is stale; run "
                "`uv run python scripts/build_model_spec_schema.py`"
            )
        print("PASS: public ModelSpec schema is current")
        return
    OUTPUT.write_text(expected, encoding="utf-8")
    print(f"WROTE: {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
