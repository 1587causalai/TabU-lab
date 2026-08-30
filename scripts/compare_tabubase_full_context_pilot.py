#!/usr/bin/env python3
"""Compare one frozen TabUBase checkpoint with aligned classical baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_full_context_comparison import (
    compare_full_context_pilot_receipts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-receipt", type=Path, required=True)
    parser.add_argument("--baseline-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_full_context_pilot_receipts(
        args.frozen_receipt,
        args.baseline_receipt,
        checkpoint_seed=args.checkpoint_seed,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
