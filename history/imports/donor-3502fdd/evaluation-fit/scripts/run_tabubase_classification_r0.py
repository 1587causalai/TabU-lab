#!/usr/bin/env python3
"""Run or aggregate the TabUBase classification R0 validation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_classification_r0 import (
    run_classification_r0_validation,
    select_global_r0_schedule,
)
from tabu_lab.experiments.tabubase_scale import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="iris,wine,breast_cancer")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-budget", type=int, default=128)
    parser.add_argument("--aggregate", nargs="*", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.aggregate]
        result = select_global_r0_schedule(payloads)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        if args.checkpoint_root is None:
            parser.error("--checkpoint-root is required unless --aggregate is used")
        result = run_classification_r0_validation(
            dataset_ids=tuple(item.strip() for item in args.datasets.split(",") if item.strip()),
            checkpoint_root=args.checkpoint_root,
            output_path=args.output,
            device=resolve_device(args.device),
            budget=args.label_budget,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
