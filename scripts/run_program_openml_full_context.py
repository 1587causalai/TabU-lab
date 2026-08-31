#!/usr/bin/env python3
"""Run or aggregate the versioned query-family OpenML full-context panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabu_lab.evolution import EvolutionRepository  # noqa: E402
from tabu_lab.evolution.openml_full_context import (  # noqa: E402
    aggregate_program_openml_full_context,
    load_program_openml_full_context_request,
    run_program_openml_full_context_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset = subparsers.add_parser("run-dataset", help="emit one immutable dataset receipt")
    dataset.add_argument("--repository", type=Path, required=True)
    dataset.add_argument("--request", type=Path, required=True)
    dataset.add_argument("--dataset", required=True)
    dataset.add_argument("--base-checkpoint", type=Path, required=True)
    dataset.add_argument("--base-selection-receipt", type=Path, required=True)
    dataset.add_argument("--row-checkpoint", type=Path, required=True)
    dataset.add_argument("--row-selection-receipt", type=Path, required=True)
    dataset.add_argument("--openml-data-home", type=Path, required=True)
    dataset.add_argument("--device", default="cuda", choices=("cpu", "mps", "cuda"))
    dataset.add_argument("--source-revision", required=True)
    dataset.add_argument("--source-archive-sha256", required=True)
    dataset.add_argument("--output", type=Path, required=True)

    aggregate = subparsers.add_parser(
        "aggregate", help="bind six immutable dataset receipts into one panel receipt"
    )
    aggregate.add_argument("--request", type=Path, required=True)
    aggregate.add_argument(
        "--receipt",
        type=Path,
        action="append",
        required=True,
        help="Repeat in frozen panel order for all six dataset receipts.",
    )
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = load_program_openml_full_context_request(args.request)
    if args.command == "run-dataset":
        repository = EvolutionRepository.load(args.repository)
        receipt = run_program_openml_full_context_dataset(
            repository,
            request=request,
            dataset_id=args.dataset,
            checkpoints={
                "tabu_base": args.base_checkpoint,
                "tabu_row": args.row_checkpoint,
            },
            selection_receipts={
                "tabu_base": args.base_selection_receipt,
                "tabu_row": args.row_selection_receipt,
            },
            openml_data_home=args.openml_data_home,
            device=args.device,
            evaluation_source_revision=args.source_revision,
            evaluation_source_archive_sha256=args.source_archive_sha256,
            output=args.output,
        )
        summary = {
            "dataset_id": receipt.dataset_id,
            "evidence_status": receipt.evidence_status,
            "output": str(args.output.resolve()),
            "receipt_hash": receipt.receipt_hash,
            "status": "passed",
        }
    else:
        panel = aggregate_program_openml_full_context(
            request=request,
            receipt_paths=tuple(args.receipt),
            output=args.output,
        )
        summary = {
            "arm_panel_success": panel.arm_panel_success,
            "evidence_status": panel.evidence_status,
            "output": str(args.output.resolve()),
            "receipt_hash": panel.receipt_hash,
            "status": "passed",
            "task_macros": panel.task_macros,
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
