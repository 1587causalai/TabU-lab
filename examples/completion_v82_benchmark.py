"""Bounded FP32 V8.1/V8.2 latency/memory qualification, with zero optimizer updates.

Synthetic rows isolate width scaling. Optional existing tasks measure complete
real-table forwards on identical inputs; no predictive-quality claim is made.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# The pinned DGX container launcher intentionally does not forward PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.models.completion_v8 import (
    configure_memory,
    from_v81,
    load_weights,
    prepare_completion,
)
from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def timed(fn, device, repeats):
    # First pass verifies finite outputs and warms the selected path.
    fn()
    sync(device)
    times, peak = [], 0
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated())
    return {
        "seconds": times,
        "median_seconds": statistics.median(times),
        "peak_allocated_bytes": peak if device.type == "cuda" else None,
    }


def task_episode(task, model, device, max_rows):
    values = torch.tensor(task["values"], dtype=torch.float32, device=device)
    visible = torch.tensor(task["visible"], dtype=torch.bool, device=device)
    ns = task["n_support"]
    if max_rows and len(values) > max_rows:
        nq = min(task["n_query"], max_rows // 4)
        indices = list(range(max_rows - nq)) + list(range(ns, ns + nq))
        values, visible = values[indices], visible[indices]
        ns = max_rows - nq
    schemas = tuple(
        ColumnSchema(f["key"], f["kind"], None if f["kind"] == "numeric" else len(f["domain"]))
        for f in task["features"]
    )
    columns = tuple(
        values[:, a] if s.kind == "numeric" else values[:, a].long() for a, s in enumerate(schemas)
    )
    query = torch.zeros_like(visible)
    query[ns:, -1] = True
    assert not visible[query].any() and not values[query].any()
    inputs = RestorationInput(schemas, columns, visible, query, task["codebook_seed"])
    return prepare_completion(inputs, model.config)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--widths", type=int, nargs="*", default=[32, 128, 512, 1024, 1755])
    p.add_argument("--batch-rows", type=int, default=16)
    p.add_argument("--tasks", type=Path)
    p.add_argument("--names", nargs="*", default=["Bioresponse", "hiva_agnostic", "QSAR-TID-11"])
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--versions", nargs="+", choices=("v8.1", "v8.2"), default=["v8.1", "v8.2"])
    p.add_argument("--backward", action="store_true", help="Also qualify full-model backward")
    p.add_argument("--backward-only", action="store_true", help="Only full-model backward trials")
    args = p.parse_args()
    if args.repeats < 1 or args.batch_rows < 1:
        p.error("repeats and batch-rows must be positive")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(812)
    torch.set_default_dtype(torch.float32)
    if device.type == "cuda":
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, 48 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
        )
    old = load_weights(args.checkpoint, device=device)
    assert old.config.model_version == "v8.1"
    new = from_v81(old)
    assert all(v.dtype == torch.float32 for v in new.parameters())
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "torch": str(torch.__version__),
        "device": str(device),
        "dtype": "float32",
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "optimizer_updates": 0,
        "versions": args.versions,
        "parent_config": old.config.as_dict(),
        "candidate_config": new.config.as_dict(),
        "precision": new.execution_precision,
        "weights_copied_exactly": True,
        "row": [],
        "full_model": [],
        "scope": "fixed-weight engineering benchmark; not quality, adaptation or official ranking",
    }

    def write():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for width in args.widths:
        x = (torch.randn(args.batch_rows, width, 128, device=device) * 0.1).requires_grad_()
        eligible = torch.ones(args.batch_rows, width, dtype=torch.bool, device=device)
        for label, model in [("v8.1", old), ("v8.2", new)]:
            if label not in args.versions:
                continue
            axial = getattr(model.backbone, "axial", model.backbone)
            block = axial.layers[0].row
            for backward in (False, True):

                def run(block=block, x=x, eligible=eligible, backward=backward):
                    block.zero_grad(set_to_none=True)
                    x.grad = None
                    with torch.set_grad_enabled(backward):
                        out = block.batched(x, x, eligible)
                        if backward:
                            out.square().mean().backward()
                            if not torch.isfinite(x.grad).all():
                                raise FloatingPointError("nonfinite row input gradient")

                row = {
                    "version": label,
                    "columns": width,
                    "rows": args.batch_rows,
                    "backward": backward,
                    **timed(run, device, args.repeats),
                }
                result["row"].append(row)
                print(json.dumps(row), flush=True)
                write()
            block.zero_grad(set_to_none=True)
        del x
    for model in (old, new):
        model.zero_grad(set_to_none=True)
        configure_memory(model)
    if args.tasks:
        tasks = json.loads(args.tasks.read_text())
        tasks = [(t["validation"] | {"name": t["name"]}) if "validation" in t else t for t in tasks]
        tasks = [t for t in tasks if t["name"] in args.names]
        if set(t["name"] for t in tasks) != set(args.names):
            raise ValueError("requested task names are missing")
        result["tasks_sha256"] = hashlib.sha256(args.tasks.read_bytes()).hexdigest()
        for task in tasks:
            reference = None
            for label, model in [("v8.1", old), ("v8.2", new)]:
                if label not in args.versions:
                    continue
                ep = task_episode(task, model, device, args.max_rows)
                modes = (
                    [True] if args.backward_only else ([False, True] if args.backward else [False])
                )
                for backward in modes:
                    row = {
                        "version": label,
                        "name": task["name"],
                        "backward": backward,
                        "rows": int(ep.inputs.visible.shape[0]),
                        "columns": int(ep.inputs.visible.shape[1]),
                        "query_cells": int(ep.inputs.query.sum()),
                    }
                    try:

                        def run(model=model, ep=ep, backward=backward):
                            model.zero_grad(set_to_none=True)
                            with torch.set_grad_enabled(backward):
                                out = model(ep, decode=False)
                                if backward:
                                    # A fixed engineering probe, not a task training objective.
                                    out.states[-1].square().mean().backward()
                                    grads = [
                                        v.grad for v in model.parameters() if v.grad is not None
                                    ]
                                    if not all(torch.isfinite(g).all() for g in grads):
                                        raise FloatingPointError("nonfinite full-model gradient")

                        row.update(timed(run, device, args.repeats))
                        with torch.no_grad():
                            predicted = model(ep, decode=False).states[-1].detach()
                            row["query_code_abs_max"] = float(predicted.abs().max())
                            if label == "v8.1":
                                reference = predicted.clone()
                            elif reference is not None:
                                row["migration_query_code_rmse"] = float(
                                    (predicted - reference).square().mean().sqrt()
                                )
                        row["finite"] = True
                    except (FloatingPointError, torch.OutOfMemoryError) as exc:
                        row.update(finite=False, error=str(exc))
                    result["full_model"].append(row)
                    print(json.dumps(row), flush=True)
                    write()
                model.zero_grad(set_to_none=True)
    write()


if __name__ == "__main__":
    main()
