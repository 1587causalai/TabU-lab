#!/usr/bin/env python3
"""Compare old and expanded hardened frozen-ICL results on one paired panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_paired_frozen_icl import (
    compare_paired_frozen_icl_results,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-result", type=Path, required=True)
    parser.add_argument("--expanded-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "synthetic", "real"), default="auto")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    result = compare_paired_frozen_icl_results(
        args.old_result,
        args.expanded_result,
        kind=args.kind,
        output_path=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
