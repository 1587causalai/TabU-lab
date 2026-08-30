#!/usr/bin/env python3
"""Run the paired TabUBase/XGBoost/MLP real-data panel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for extra_path in os.environ.get("TABU_EXTRA_SITE_PACKAGES", "").split(os.pathsep):
    if extra_path:
        # Append so the CUDA image stays authoritative for torch/numpy/scipy;
        # only packages absent from that image fall through to this path.
        sys.path.append(extra_path)


def main() -> int:
    from tabu_lab.experiments.tabubase_real_benchmark import run_real_benchmark
    from tabu_lab.experiments.tabubase_scale import resolve_device

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="iris,wine,breast_cancer,diabetes,california_housing",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--panel-manifest",
        type=Path,
        default=None,
        help="Explicit checked-in cached OpenML panel manifest for real-data runs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--label-budget", type=int, default=128)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--test-limit",
        type=int,
        default=512,
        help="Maximum held-out rows; pass 0 to evaluate every held-out row.",
    )
    parser.add_argument("--checkpoint-phase", choices=("PT-S1", "PT-S2"), default="PT-S1")
    parser.add_argument("--seeds", default="1729,2718,31415")
    parser.add_argument(
        "--nominal-tokenizer",
        choices=("episode_random_sphere", "source_scoped_frozen_codebook.v2"),
        default="episode_random_sphere",
    )
    parser.add_argument("--nominal-codebook-size", type=int, default=100)
    parser.add_argument("--nominal-codebook-seed", type=int, default=1729)
    parser.add_argument("--temperature-calibration", action="store_true")
    parser.add_argument("--checkpoint-run-suffix", default="")
    args = parser.parse_args()
    result = run_real_benchmark(
        dataset_ids=tuple(item.strip() for item in args.datasets.split(",") if item.strip()),
        checkpoint_root=args.checkpoint_root,
        output_path=args.output,
        device=resolve_device(args.device),
        budget=args.label_budget,
        updates=args.updates,
        learning_rate=args.learning_rate,
        checkpoint_phase=args.checkpoint_phase,
        seeds=tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip()),
        nominal_tokenizer=args.nominal_tokenizer,
        nominal_codebook_size=args.nominal_codebook_size,
        nominal_codebook_seed=args.nominal_codebook_seed,
        temperature_calibration=args.temperature_calibration,
        checkpoint_run_suffix=args.checkpoint_run_suffix,
        panel_manifest=args.panel_manifest,
        test_limit=None if args.test_limit == 0 else args.test_limit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
