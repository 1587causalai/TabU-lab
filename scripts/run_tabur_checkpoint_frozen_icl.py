#!/usr/bin/env python3
"""Run Stage-5 frozen ICL from a profile-bound TabUR synthetic checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabu_lab.experiments import run_query_row_frozen_icl  # noqa: E402


def _ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("context-rows must contain positive integers")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--context-rows", type=_ints, default=(8, 16, 32))
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    result = run_query_row_frozen_icl(
        seed=args.seed,
        rows=args.rows,
        context_rows=args.context_rows,
        row_token_count=args.row_token_count,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    payload = {
        "schema_version": "tabu.query-row.checkpoint-frozen-icl-result.v1",
        **result.as_dict(),
    }
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
