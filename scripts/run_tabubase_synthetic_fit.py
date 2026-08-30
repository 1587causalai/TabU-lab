#!/usr/bin/env python3
"""Run the bounded Stage-3 TabUBase synthetic fitting gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments import run_synthetic_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()
    result = run_synthetic_fit(seed=args.seed, steps=args.steps)
    payload = json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
