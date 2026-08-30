#!/usr/bin/env python3
"""Run the optimizer-free TabUBase real-data ICL panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_openml_new6 import OPENML_NEW6_SPECS
from tabu_lab.experiments.tabubase_real_icl import (
    FULL_CONTEXT_POLICY,
    LOW_SHOT_CONTEXT_POLICY,
    RealIclConfig,
    run_real_frozen_icl,
)
from tabu_lab.experiments.tabubase_scale import resolve_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset IDs; defaults to old6, or all new6 with --panel-manifest.",
    )
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=None,
        help="Required checked-in preregistration manifest when evaluating OpenML new6.",
    )
    parser.add_argument(
        "--openml-data-home",
        type=Path,
        default=None,
        help="Explicit sklearn OpenML cache root; required when the runtime default differs.",
    )
    parser.add_argument("--checkpoint-seeds", default="1729,2718,31415")
    parser.add_argument("--split-seeds", default="1729,2718,31415")
    parser.add_argument(
        "--context-policy",
        choices=(FULL_CONTEXT_POLICY, LOW_SHOT_CONTEXT_POLICY),
        default=FULL_CONTEXT_POLICY,
        help=(
            "full_train is the primary downstream frozen-ICL estimand and exposes every "
            "train-partition row; low_shot_grid is the historical K<=32 diagnostic."
        ),
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Defaults to every held-out row; historical low-shot replay must pass 256.",
    )
    parser.add_argument(
        "--query-chunk-rows",
        type=int,
        default=64,
        help=(
            "For full_train, chunks only the query-response readout after one complete "
            "transductive evidence episode; it never chunks or truncates evidence."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--checkpoint-run-suffix", default="-icl-kcurriculum-v1")
    args = parser.parse_args()
    default_dataset_ids = (
        tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
        if args.panel_manifest is not None
        else ("iris", "wine", "breast_cancer", "digits", "diabetes", "california_housing")
    )
    dataset_ids = (
        tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        if args.datasets is not None
        else default_dataset_ids
    )
    result = run_real_frozen_icl(
        RealIclConfig(
            checkpoint_root=args.checkpoint_root,
            output_path=args.output,
            dataset_ids=dataset_ids,
            checkpoint_seeds=tuple(
                int(item.strip()) for item in args.checkpoint_seeds.split(",") if item.strip()
            ),
            split_seeds=tuple(
                int(item.strip()) for item in args.split_seeds.split(",") if item.strip()
            ),
            context_policy=args.context_policy,
            query_limit=args.query_limit,
            query_chunk_rows=args.query_chunk_rows,
            bootstrap_replicates=args.bootstrap_replicates,
            checkpoint_run_suffix=args.checkpoint_run_suffix,
            panel_manifest_path=args.panel_manifest,
            openml_data_home=args.openml_data_home,
        ),
        device=resolve_device(args.device),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_parameter_hashes_unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
