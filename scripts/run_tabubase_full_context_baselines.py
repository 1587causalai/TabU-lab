#!/usr/bin/env python3
"""Run exact-split MLP/XGBoost baselines for full-context TabUBase evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_full_context_baselines import (
    FullContextBaselineConfig,
    run_full_context_baselines,
)
from tabu_lab.experiments.tabubase_openml_new6 import OPENML_NEW6_SPECS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset IDs; defaults to old6, or all new6 with --panel-manifest.",
    )
    parser.add_argument("--panel-manifest", type=Path, default=None)
    parser.add_argument(
        "--openml-data-home",
        type=Path,
        default=None,
        help="Explicit sklearn OpenML cache root; required when the runtime default differs.",
    )
    parser.add_argument("--split-seeds", default="1729,2718,31415")
    args = parser.parse_args()

    default_dataset_ids = (
        tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
        if args.panel_manifest is not None
        else (
            "iris",
            "wine",
            "breast_cancer",
            "digits",
            "diabetes",
            "california_housing",
        )
    )
    dataset_ids = (
        tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        if args.datasets is not None
        else default_dataset_ids
    )
    result = run_full_context_baselines(
        FullContextBaselineConfig(
            output_path=args.output,
            dataset_ids=dataset_ids,
            split_seeds=tuple(
                int(item.strip()) for item in args.split_seeds.split(",") if item.strip()
            ),
            panel_manifest_path=args.panel_manifest,
            openml_data_home=args.openml_data_home,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
