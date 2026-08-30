#!/usr/bin/env python3
"""Run the preregistered Axis-C TabUR OpenML frozen-transfer panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabu_lab.experiments import run_query_row_openml_frozen_transfer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Profile-bound tabu.query.row@0.1.0 checkpoint; repeat for each B1 seed.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated subset for a smoke run; full panel is the default.",
    )
    parser.add_argument("--openml-data-home", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in args.checkpoint:
        if not path.is_file():
            raise SystemExit(f"checkpoint does not exist: {path}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    dataset_ids = None
    if args.datasets:
        dataset_ids = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
        if not dataset_ids:
            raise SystemExit("--datasets must contain at least one dataset id")
    result = run_query_row_openml_frozen_transfer(
        panel_manifest=args.panel_manifest,
        checkpoint_paths=tuple(args.checkpoint),
        dataset_ids=dataset_ids,
        device=args.device,
        openml_data_home=args.openml_data_home,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {
        "status": result["status"],
        "evidence_status": result["evidence_status"],
        "datasets": result["datasets"],
        "checkpoint_controls": [
            {
                "root_seed": item["root_seed"],
                "status": item["status"],
                "truth_substitution_prediction_unchanged": item[
                    "truth_substitution_prediction_unchanged"
                ],
            }
            for item in result["frozen_controls"]
        ],
        "output": str(args.output),
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
