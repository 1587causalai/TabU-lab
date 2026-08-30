#!/usr/bin/env python3
"""Run the R4 diverse supervised synthetic prior v2 exits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import validate_query_row_supervised_synthetic_v2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-seed", type=int, default=1729)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate_query_row_supervised_synthetic_v2(
        root_seed=args.root_seed,
        worlds=args.worlds,
        row_token_count=args.row_token_count,
        device=args.device,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
