"""One full-size forward/backward/update before a restoration joint-fit run."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration import RestorationModel, prepare_episode, score_prepared_episode
from tabu_lab.restoration_joint_fit import _episode, _plan_query_mask, _write_json, prepare_plan


def run_preflight(args):
    if args.output.exists():
        raise FileExistsError("preflight output already exists")
    plan = prepare_plan(args.preregistration, args.corpus, args.device)
    table = max(plan.tables, key=lambda item: (item.train_rows * len(item.schema), item.name))
    opt = plan.spec["optimizer"]
    cuda = args.device.startswith("cuda")
    started = time.monotonic()
    timings = {}

    def mark(stage):
        nonlocal started
        if cuda:
            torch.cuda.synchronize()
        now = time.monotonic()
        timings[stage] = now - started
        started = now

    result = {"schema": "tabu.restoration.joint-preflight.v1", "status": "local_unissued",
              "identity": plan.identity, "table": table.name, "rows": table.train_rows,
              "columns": len(table.schema), "outcome": "failed",
              "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(plan.spec["seeds"]["model"])
        model = RestorationModel(plan.config).to(device=args.device, dtype=torch.float64)
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"],
                                     betas=tuple(opt["betas"]), eps=opt["eps"],
                                     weight_decay=opt["weight_decay"])
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        query, mask_info = _plan_query_mask(plan, table, plan.spec["seeds"]["masks"])
        result["mask"] = mask_info
        prepared = prepare_episode(model, *_episode(table, query, plan.spec["seeds"]["codes"],
                                                    args.device))
        mark("prepare_seconds")
        scored = score_prepared_episode(model, prepared)
        if not torch.isfinite(scored.loss):
            raise FloatingPointError("nonfinite preflight loss")
        result["loss"] = float(scored.loss.detach())
        mark("forward_seconds")
        scored.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt["grad_clip"],
                                                   error_if_nonfinite=True)
        result["gradient_norm"] = float(grad_norm)
        mark("backward_seconds")
        optimizer.step()
        tensors = list(model.parameters()) + [v for state in optimizer.state.values()
                                              for v in state.values() if torch.is_tensor(v)]
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise FloatingPointError("nonfinite preflight model or optimizer")
        mark("optimizer_seconds")
        result["outcome"] = "passed"
    except Exception as error:
        result["error_type"] = type(error).__name__
    result["timings"] = timings
    result["step_seconds"] = sum(timings.values())
    result["peak_cuda_bytes"] = (
        torch.cuda.max_memory_allocated() if cuda and torch.cuda.is_initialized() else 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    return result
