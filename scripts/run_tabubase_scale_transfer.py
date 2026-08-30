#!/usr/bin/env python3
"""Run the frozen TabUBase PT-S0/PT-S1 protocol as local-unissued evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabu_lab.experiments.tabubase_scale import (
    CONTEXT_CANDIDATE_INITIAL_ROWS,
    DENSE_TRAINING_FORWARD_MODE,
    EXPANDED_SYNTHETIC_GENERATOR_VERSION,
    LEGACY_SYNTHETIC_GENERATOR_VERSION,
    LONG_CONTEXT_CANDIDATE_ROWS,
    LONG_CONTEXT_PRETRAINING_PROTOCOL_ID,
    LONG_CONTEXT_ROWS_SCHEDULE,
    QUERY_RESPONSE_TRAINING_FORWARD_MODE,
    SYNTHETIC_GENERATOR_VERSIONS,
    PretrainRunConfig,
    resolve_device,
    run_pretraining,
    source_tree_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("PT-S0", "PT-S1", "PT-S2"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--prefetch-workers", type=int, default=0)
    parser.add_argument("--prefetch-queue-depth", type=int, default=8)
    parser.add_argument(
        "--nominal-tokenizer",
        choices=("episode_random_sphere", "source_scoped_frozen_codebook.v2"),
        default="episode_random_sphere",
    )
    parser.add_argument("--nominal-codebook-size", type=int, default=100)
    parser.add_argument("--nominal-codebook-seed", type=int, default=1729)
    curriculum = parser.add_mutually_exclusive_group()
    curriculum.add_argument("--icl-k-curriculum", action="store_true")
    curriculum.add_argument("--long-context-curriculum", action="store_true")
    parser.add_argument("--query-readout-chunk-rows", type=int, default=64)
    parser.add_argument(
        "--generator-version",
        choices=SYNTHETIC_GENERATOR_VERSIONS,
        default=LEGACY_SYNTHETIC_GENERATOR_VERSION,
    )
    parser.add_argument("--validation-worlds", type=int)
    return parser


def _context_schedule(args: argparse.Namespace) -> tuple[int, ...]:
    if args.long_context_curriculum:
        return LONG_CONTEXT_ROWS_SCHEDULE
    return (2, 4, 8, 16, 32, 64) if args.icl_k_curriculum else (64,)


def _context_candidate_rows(args: argparse.Namespace) -> int:
    return (
        LONG_CONTEXT_CANDIDATE_ROWS
        if args.long_context_curriculum
        else CONTEXT_CANDIDATE_INITIAL_ROWS
    )


def _training_forward_mode(args: argparse.Namespace) -> str:
    return (
        QUERY_RESPONSE_TRAINING_FORWARD_MODE
        if args.long_context_curriculum
        else DENSE_TRAINING_FORWARD_MODE
    )


def _pretraining_protocol_id(args: argparse.Namespace) -> str:
    return (
        LONG_CONTEXT_PRETRAINING_PROTOCOL_ID
        if args.long_context_curriculum
        else args.generator_version
    )


def _validation_worlds(args: argparse.Namespace) -> int:
    if args.validation_worlds is not None:
        return args.validation_worlds
    return 192 if args.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION else 12


def _require_matching_promotion_receipt(
    pilot: dict[str, object],
    *,
    args: argparse.Namespace,
    expected_phase: str,
) -> None:
    expected: dict[str, object] = {
        "phase": expected_phase,
        "passed": True,
        "generator_version": args.generator_version,
        "nominal_tokenizer": args.nominal_tokenizer,
        "nominal_codebook_size": args.nominal_codebook_size,
        "nominal_codebook_seed": args.nominal_codebook_seed,
        "context_rows_schedule": list(_context_schedule(args)),
        "context_candidate_rows": _context_candidate_rows(args),
        "training_forward_mode": _training_forward_mode(args),
        "query_readout_chunk_rows": args.query_readout_chunk_rows,
        "pretraining_protocol_id": _pretraining_protocol_id(args),
        "validation_worlds": _validation_worlds(args),
        "source_tree_sha256": source_tree_sha256(),
    }
    if args.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION:
        expected["validation_context_policy"] = (
            "every_world_at_every_support_realizable_K"
        )
    mismatches = [
        key for key, value in expected.items() if pilot.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            "promotion receipt does not match the requested protocol: "
            + ", ".join(mismatches)
        )


def main() -> int:
    args = _parser().parse_args()
    if args.phase == "PT-S0":
        worlds, updates, checkpoints = 2_048, 2_000, (0, 2_000)
    elif args.phase == "PT-S1":
        if args.pilot_result is None:
            raise SystemExit("PT-S1 requires --pilot-result")
        pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
        _require_matching_promotion_receipt(pilot, args=args, expected_phase="PT-S0")
        worlds, updates = 20_000, 20_000
        checkpoints = (0, 2_000, 5_000, 10_000, 20_000)
    else:
        if args.pilot_result is None:
            raise SystemExit("PT-S2 requires --pilot-result")
        pilot = json.loads(args.pilot_result.read_text(encoding="utf-8"))
        _require_matching_promotion_receipt(pilot, args=args, expected_phase="PT-S1")
        worlds, updates = 200_000, 200_000
        checkpoints = (0, 20_000, 50_000, 100_000, 150_000, 200_000)
    result = run_pretraining(
        PretrainRunConfig(
            phase=args.phase,
            worlds=worlds,
            updates=updates,
            seed=args.seed,
            checkpoint_updates=checkpoints,
            prefetch_workers=args.prefetch_workers,
            prefetch_queue_depth=args.prefetch_queue_depth,
            nominal_tokenizer=args.nominal_tokenizer,
            nominal_codebook_size=args.nominal_codebook_size,
            nominal_codebook_seed=args.nominal_codebook_seed,
            context_rows_schedule=_context_schedule(args),
            generator_version=args.generator_version,
            validation_worlds=_validation_worlds(args),
            context_candidate_rows=_context_candidate_rows(args),
            training_forward_mode=_training_forward_mode(args),
            query_readout_chunk_rows=args.query_readout_chunk_rows,
        ),
        output_root=args.output_root,
        device=resolve_device(args.device),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
