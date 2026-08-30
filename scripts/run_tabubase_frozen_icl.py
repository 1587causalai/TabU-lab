#!/usr/bin/env python3
"""Run TabUBase v2 held-out frozen ICL as local-unissued evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_icl import FrozenIclConfig, run_frozen_icl
from tabu_lab.experiments.tabubase_scale import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--world-seed", type=int, default=1729)
    parser.add_argument("--heldout-worlds", type=int, default=512)
    parser.add_argument("--query-rows", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--world-scope", choices=("heldout", "train_mixture"), default="heldout")
    args = parser.parse_args()
    result = run_frozen_icl(
        FrozenIclConfig(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            seed=args.seed,
            world_seed=args.world_seed,
            heldout_worlds=args.heldout_worlds,
            query_rows=args.query_rows,
            batch_size=args.batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
            world_scope=args.world_scope,
        ),
        device=resolve_device(args.device),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed_primary_frozen_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
