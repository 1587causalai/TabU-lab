"""CUDA qualification for the largest synthetic and real curriculum episodes."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration import (
    RestorationModel,
    prepare_episode,
    score_prepared_episode,
)
from tabu_lab.restoration_curriculum_fit import _configure_runtime, _episode_for, prepare_plan
from tabu_lab.restoration_optimizers import adamw, switch_to_muon


def run_preflight(args):
    plan = prepare_plan(Path(args.preregistration), Path(args.corpus_root), args.device)
    if args.device == "cuda:0" and not torch.cuda.is_available():
        return {"outcome": "blocked_resources", "reason": "CUDA unavailable"}
    _configure_runtime(args.device)
    config = plan["_config"]
    torch.manual_seed(plan["seeds"]["model"])
    model = RestorationModel(config).to(device=args.device, dtype=torch.float64)
    optimizer = adamw(model, plan["_optimizer"])
    largest_synthetic = max((table for table in plan["_tables"] if table.kind == "synthetic"),
                            key=lambda table: table.train_rows * table.width)
    largest_real = max(
        (table for table in plan["_tables"] if table.kind == "real"),
        key=lambda table: (min(table.train_rows, 204) if table.windowed else table.train_rows)
        * table.width,
    )
    stage_synthetic = plan["_stages"][0]
    stage_real = plan["_stages"][2]
    seeds = plan["seeds"]
    probes = [("synthetic", largest_synthetic, stage_synthetic),
              ("real", largest_real, stage_real)]
    measurements = []
    if args.device == "cuda:0":
        torch.cuda.reset_peak_memory_stats()
    for label, table, stage in probes:
        started = time.monotonic()
        episode, _ = _episode_for(table, 0, stage, seeds, config, args.device)
        optimizer.zero_grad(set_to_none=True)
        score = score_prepared_episode(model, prepare_episode(model, *episode))
        score.loss.backward()
        finite_gradients = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                               for parameter in model.parameters())
        if not bool(torch.isfinite(score.loss)) or not finite_gradients:
            raise FloatingPointError(f"nonfinite {label} preflight")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.grad is not None],
            plan["_optimizer"].grad_clip,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise FloatingPointError(f"nonfinite {label} parameters")
        measurements.append({"label": label, "table": table.name,
                             "rows": (min(table.train_rows, 204) if table.windowed
                                      else table.train_rows), "width": table.width,
                             "seconds": time.monotonic() - started})
        if label == "synthetic":
            optimizer, _partition = switch_to_muon(optimizer, model, plan["_optimizer"])
    peak = torch.cuda.max_memory_allocated() if args.device == "cuda:0" else 0
    result = {"outcome": "passed", "device": args.device, "probes": measurements,
              "peak_allocated_bytes": peak, "optimizer_transition": "passed"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


__all__ = ["run_preflight"]
