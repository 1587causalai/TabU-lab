#!/usr/bin/env python3
"""Aggregate common-world TabUBase frozen-ICL results across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_icl import aggregate_common_world_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    result = aggregate_common_world_results(
        args.result,
        output_path=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_clustered_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
