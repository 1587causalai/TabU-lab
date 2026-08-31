#!/usr/bin/env python3
"""Run the TabUR Stage-4 scratch-only diagnostic on the full train/test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import run_query_row_real_scratch_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="iris,wine,diabetes")
    parser.add_argument(
        "--label-budget",
        type=int,
        default=None,
        help="Optional bounded context override; default uses every train-partition row.",
    )
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional bounded query override; default evaluates every held-out row.",
    )
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_query_row_real_scratch_benchmark(
        dataset_ids=tuple(item.strip() for item in args.datasets.split(",") if item.strip()),
        label_budget=args.label_budget,
        updates=args.updates,
        test_limit=args.test_limit,
        row_token_count=args.row_token_count,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
