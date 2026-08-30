#!/usr/bin/env python3
"""Run the bounded K<=512 TabUBase forward/backward runtime gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch

import torch

from tabu_lab.experiments.tabubase_expanded_synthetic import (
    LONG_CONTEXT_CANDIDATE_ROWS,
    RESPONSE_MODALITIES,
    build_expanded_synthetic_episode,
    sample_expanded_world_manifest,
)
from tabu_lab.experiments.tabubase_response_readout import (
    query_response_objective_loss,
)
from tabu_lab.experiments.tabubase_scale import (
    LONG_CONTEXT_PRETRAINING_PROTOCOL_ID,
    QUERY_RESPONSE_TRAINING_FORWARD_MODE,
    _git_commit_or_none,
    _sha256_file,
    _state_hash,
    build_tabubase_scale_model,
    resolve_device,
    source_tree_sha256,
)
from tabu_lab.models.components import CellTokenizer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--root-seed", type=int, default=1729)
    parser.add_argument("--context-rows", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--query-rows", type=int, default=64)
    parser.add_argument("--query-readout-chunk-rows", type=int, default=64)
    parser.add_argument("--predictor-width", type=int, default=32)
    parser.add_argument("--peak-allocated-bytes-max", type=int, default=8 * 1024**3)
    parser.add_argument("--peak-reserved-bytes-max", type=int, default=12 * 1024**3)
    return parser


def _selected_worlds(*, root_seed: int, predictor_width: int) -> dict[str, int]:
    selected: dict[str, int] = {}
    for world_index in range(20_000):
        manifest = sample_expanded_world_manifest(
            root_seed=root_seed,
            world_index=world_index,
        )
        if (
            manifest.predictor_width == predictor_width
            and manifest.response_modality not in selected
        ):
            selected[manifest.response_modality] = world_index
        if set(selected) == set(RESPONSE_MODALITIES):
            return selected
    raise RuntimeError("could not find every response modality at the requested width")


def _run_case(
    *,
    root_seed: int,
    world_index: int,
    modality: str,
    context_rows: int,
    query_rows: int,
    query_readout_chunk_rows: int,
    device: torch.device,
    peak_allocated_bytes_max: int,
    peak_reserved_bytes_max: int,
) -> dict[str, object]:
    episode, truth, metadata = build_expanded_synthetic_episode(
        root_seed=root_seed,
        world_index=world_index,
        context_rows=context_rows,
        query_rows=query_rows,
        context_candidate_rows=LONG_CONTEXT_CANDIDATE_ROWS,
    )
    if metadata["response_modality"] != modality:
        raise RuntimeError("selected runtime world changed response modality")
    model = build_tabubase_scale_model(
        seed=root_seed,
        device=device,
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
        nominal_codebook_size=100,
        nominal_codebook_seed=1729,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    parameter_hash_before = _state_hash(model)
    dense_terminal_called = False

    def forbidden_dense_terminal(*_args: object, **_kwargs: object) -> None:
        nonlocal dense_terminal_called
        dense_terminal_called = True
        raise RuntimeError("bounded long-context path called the dense terminal")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.monotonic()
    with patch.object(type(model), "_forward_dense", forbidden_dense_terminal):
        loss = query_response_objective_loss(
            model,
            episode.to(device),
            truth.to(device),
            context_rows=context_rows,
            query_readout_chunk_rows=query_readout_chunk_rows,
        )
        loss.backward()
    torch.cuda.synchronize(device)
    elapsed_seconds = time.monotonic() - started
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    gradients_finite = bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    )
    nonzero_gradient = any(bool(torch.count_nonzero(gradient)) for gradient in gradients)
    parameter_hash_after = _state_hash(model)
    checks = {
        "finite_forward_loss": bool(torch.isfinite(loss)),
        "finite_backward_gradients": gradients_finite,
        "at_least_one_nonzero_parameter_gradient": nonzero_gradient,
        "dense_terminal_not_called": not dense_terminal_called,
        "parameter_hash_unchanged_without_optimizer_step": (
            parameter_hash_before == parameter_hash_after
        ),
        "peak_allocated_within_bound": (
            peak_allocated_bytes <= peak_allocated_bytes_max
        ),
        "peak_reserved_within_bound": peak_reserved_bytes <= peak_reserved_bytes_max,
    }
    return {
        "world_index": world_index,
        "world_id": metadata["world_id"],
        "world_manifest_hash": metadata["world_manifest_hash"],
        "response_modality": modality,
        "predictor_width": metadata["predictor_width"],
        "context_rows": context_rows,
        "query_rows": query_rows,
        "loss": float(loss.detach().cpu()),
        "gradient_parameter_count": len(gradients),
        "parameter_hash_before_backward": parameter_hash_before,
        "parameter_hash_after_backward": parameter_hash_after,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "elapsed_seconds": elapsed_seconds,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    args = _parser().parse_args()
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise SystemExit("G-D5-L512 runtime audit requires an available CUDA device")
    if tuple(args.context_rows) != (128, 256, 512):
        raise SystemExit("bounded runtime context ladder is frozen to 128 256 512")
    if args.query_rows != 64 or args.query_readout_chunk_rows != 64:
        raise SystemExit("bounded runtime audit is frozen to Q=64 and readout chunks=64")
    if args.predictor_width != 32:
        raise SystemExit("bounded runtime audit is frozen to predictor width 32")

    selected = _selected_worlds(
        root_seed=args.root_seed,
        predictor_width=args.predictor_width,
    )
    cases = [
        _run_case(
            root_seed=args.root_seed,
            world_index=selected[modality],
            modality=modality,
            context_rows=context_rows,
            query_rows=args.query_rows,
            query_readout_chunk_rows=args.query_readout_chunk_rows,
            device=device,
            peak_allocated_bytes_max=args.peak_allocated_bytes_max,
            peak_reserved_bytes_max=args.peak_reserved_bytes_max,
        )
        for modality in RESPONSE_MODALITIES
        for context_rows in args.context_rows
    ]
    gates = {
        "all_12_cases_pass": all(bool(case["passed"]) for case in cases),
        "all_modalities_present": {
            str(case["response_modality"]) for case in cases
        }
        == set(RESPONSE_MODALITIES),
        "all_context_rows_present": {
            int(case["context_rows"]) for case in cases
        }
        == {128, 256, 512},
        "no_optimizer_created": True,
    }
    result = {
        "schema_version": "tabu.expanded-synthetic-long-context-runtime-audit.v1",
        "status": "local_unissued",
        "gate": "G-D5-L512",
        "gate_scope": "bounded_partial_runtime_gate_only",
        "protocol_id": LONG_CONTEXT_PRETRAINING_PROTOCOL_ID,
        "training_forward_mode": QUERY_RESPONSE_TRAINING_FORWARD_MODE,
        "root_seed": args.root_seed,
        "context_candidate_rows": LONG_CONTEXT_CANDIDATE_ROWS,
        "context_rows": list(args.context_rows),
        "query_rows": args.query_rows,
        "query_readout_chunk_rows": args.query_readout_chunk_rows,
        "predictor_width": args.predictor_width,
        "memory_bounds": {
            "peak_allocated_bytes_max": args.peak_allocated_bytes_max,
            "peak_reserved_bytes_max": args.peak_reserved_bytes_max,
        },
        "selected_world_indices": selected,
        "cases": cases,
        "gates": gates,
        "passed": all(gates.values()),
        "source_tree_sha256": source_tree_sha256(),
        "git_commit": _git_commit_or_none(),
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "runtime_backend": os.environ.get("WEHUB_RUNTIME_BACKEND"),
            "runtime_image": os.environ.get("WEHUB_RUNTIME_IMAGE"),
        },
        "explicit_non_claim": (
            "K<=512 does not close architecture G-D5 at K=1024,2048,4096,8192; "
            "this is not a formal receipt or model claim"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printed = result | {
        "result_path": str(args.output),
        "result_sha256": _sha256_file(args.output),
    }
    print(json.dumps(printed, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
