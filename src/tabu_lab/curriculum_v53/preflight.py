"""Bounded device probes including checkpointed next-update parity."""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration_v53 import V53LossConfig, V53Model
from tabu_lab.restoration_optimizers import adamw, switch_to_muon

from .artifacts import atomic_json, load_checkpoint, restore_rng, save_checkpoint
from .runner import _new_state, _seed_model, configure_runtime, train_step


def preflight(plan, output, *, device="cpu", max_seconds=120.0):
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("preflight max_seconds must be positive and finite")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "schema": "tabu.curriculum.v53.preflight.v1",
        "status": "local_unissued",
        "outcome": "started",
        "identity": plan.identity,
        "probes": [],
    }
    atomic_json(output / "started.json", result)
    try:
        runtime = configure_runtime(device)
        result["runtime"] = runtime
        for stage in plan.spec["stages"]:
            cohorts = {item["cohort"] for item in stage["sampling"]}
            tables = [
                table for table in plan.tables if table.role == "train" and table.cohort in cohorts
            ]
            candidates = {}
            for kind in sorted({table.kind for table in tables}):
                typed = [table for table in tables if table.kind == kind]
                # Both maximum row count and cell footprint matter to quadratic readout.
                for label, key in (
                    ("rows", lambda t: min(t.train_rows, t.window_rows or t.train_rows)),
                    ("cells", lambda t: min(t.train_rows, t.window_rows or t.train_rows) * t.width),
                ):
                    table = max(typed, key=key)
                    candidates[(kind, table.name)] = (label, table)
            for (kind, _), (label, table) in candidates.items():
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("preflight budget exhausted before next probe")
                _seed_model(plan)
                model = V53Model(plan.config).to(device=device, dtype=torch.float64)
                optimizer = adamw(model, plan.optimizer)
                if stage["optimizer"] == "muon":
                    optimizer, _ = switch_to_muon(optimizer, model, plan.optimizer)
                loss = V53LossConfig(**stage.get("loss", {}))
                first = train_step(
                    model,
                    optimizer,
                    plan,
                    table,
                    stage["recipe"][kind],
                    0,
                    device,
                    loss,
                    namespace=stage["name"],
                )
                state = _new_state(plan.spec["stages"])
                state.update(update=1, optimizer_kind=stage["optimizer"])
                checkpoint = output / f"probe-{len(result['probes']):03d}.pt"
                save_checkpoint(
                    checkpoint,
                    plan=plan,
                    model=model,
                    optimizer=optimizer,
                    state=state,
                    runtime=runtime,
                    lineage=[{"mode": "preflight_only_not_training_resume"}],
                    purpose="preflight",
                )
                payload, _digest = load_checkpoint(checkpoint)
                clone = V53Model(plan.config).to(device=device, dtype=torch.float64)
                clone.load_state_dict(payload["model"])
                other = adamw(clone, plan.optimizer)
                if stage["optimizer"] == "muon":
                    other, _ = switch_to_muon(other, clone, plan.optimizer)
                other.load_state_dict(payload["optimizer"])
                restore_rng(payload["rng"])
                actual = train_step(
                    model,
                    optimizer,
                    plan,
                    table,
                    stage["recipe"][kind],
                    1,
                    device,
                    loss,
                    namespace=stage["name"],
                )
                restore_rng(payload["rng"])
                restored = train_step(
                    clone,
                    other,
                    plan,
                    table,
                    stage["recipe"][kind],
                    1,
                    device,
                    loss,
                    namespace=stage["name"],
                )
                equal = actual["loss"] == restored["loss"] and all(
                    torch.equal(value, clone.state_dict()[key])
                    for key, value in model.state_dict().items()
                )
                if not equal:
                    raise AssertionError("checkpoint next-update parity failed")
                result["probes"].append(
                    {
                        "stage": stage["name"],
                        "table": table.name,
                        "selection": label,
                        "optimizer": stage["optimizer"],
                        "first_step_seconds": first["seconds"],
                        "rows": len(first["episode"]["row_ids"]),
                        "width": table.width,
                        "next_update_exact": True,
                        "gradient_norm": first["gradient_norm"],
                    }
                )
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("preflight budget exhausted during probe")
        result["outcome"] = "passed"
    except (KeyboardInterrupt, Exception) as error:
        result.update(
            outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    result["seconds"] = time.monotonic() - started
    if device == "cuda:0" and torch.cuda.is_available():
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["claim_boundary"] = "finite local steps and same-backend next-update equivalence only"
    atomic_json(output / "terminal.json", result)
    return result
