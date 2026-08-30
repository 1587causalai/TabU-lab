#!/usr/bin/env python3
"""Run profile-bound TabUR synthetic pretraining and save a checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import run_query_row_synthetic_pretraining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="completion.artificial_mask.v1")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_query_row_synthetic_pretraining(
        profile=args.profile,
        seed=args.seed,
        rows=args.rows,
        worlds=args.worlds,
        steps=args.steps,
        learning_rate=args.learning_rate,
        row_token_count=args.row_token_count,
        device=args.device,
        output=args.output,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
