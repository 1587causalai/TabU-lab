#!/usr/bin/env python3
"""Run R5 B0/B1 v2 pretraining with bounded optimizer screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import run_query_row_r5_bounded_pretraining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--pilot-worlds", type=int, default=64)
    parser.add_argument("--pilot-updates", type=int, default=100)
    parser.add_argument("--validation-worlds", type=int, default=48)
    args = parser.parse_args()
    result = run_query_row_r5_bounded_pretraining(
        output_root=args.output_root,
        device=args.device,
        row_token_count=args.row_token_count,
        pilot_worlds=args.pilot_worlds,
        pilot_updates=args.pilot_updates,
        validation_worlds=args.validation_worlds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
