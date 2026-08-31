#!/usr/bin/env python3
"""Run the corrected R3 real-regression adaptation diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import run_query_row_r3_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panel-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--label-budget",
        type=int,
        default=None,
        help="Optional bounded context override; default uses every train-partition row.",
    )
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--test-limit", type=int)
    args = parser.parse_args()
    result = run_query_row_r3_diagnosis(
        checkpoint=args.checkpoint,
        panel_manifest=args.panel_manifest,
        output=args.output,
        device=args.device,
        label_budget=args.label_budget,
        row_token_count=args.row_token_count,
        learning_rate=args.learning_rate,
        test_limit=args.test_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
