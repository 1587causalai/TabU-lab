"""Strictly resume a squared-loss OpenML12 checkpoint for one 900s update segment."""
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from tabu_lab.curriculum_v53 import runner
from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan


def main(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema="tabu.openml12.strict30.half.v1", outcome="preparing",
                   host=args.host, half=args.half, parent=str(Path(args.parent).resolve()),
                   parent_sha256=args.parent_sha, manifest=str(Path(args.manifest).resolve()),
                   target_successful_update_seconds=args.seconds,
                   continuation="strict_resume_same_identity_optimizer_rng_cursor_exposure",
                   source_script_sha256=sha256(__file__))
    atomic_json(root / "campaign.json", receipt)
    native_step = runner.train_step
    try:
        plan = load_v55_plan(args.manifest)
        parent, digest = load_checkpoint(args.parent)
        assert digest == args.parent_sha and parent["identity"] == plan.identity
        assert parent["state"]["stage_index"] == 0
        stage = plan.spec["stages"][0]
        assert stage["objective"] == {"kind": "squared"} and stage["max_seconds"] == 11000
        assert [(x["cohort"], x["episodes"]) for x in stage["sampling"]] == [
            ("focus2", 6), ("other10", 10), ("history618", 4)]
        runtime = runner.configure_runtime(args.device)
        parent_update = parent["state"]["update"]
        parent_stage_seconds = parent["state"]["stage_seconds"][0]
        assert stage["max_seconds"] - parent_stage_seconds > args.seconds + 100
        receipt.update(outcome="running", identity_sha256=plan.identity["sha256"],
                       parent_update=parent_update, parent_stage_seconds=parent_stage_seconds,
                       runtime=runtime)
        atomic_json(root / "campaign.json", receipt)
        budget = dict(successful_seconds=0.0, updates=0, recent=[], stop_requested=False)

        def timed_step(*pos, **kw):
            row = native_step(*pos, **kw)
            assert row["objective_kind"] == "squared"
            budget["successful_seconds"] += row["seconds"]
            budget["updates"] += 1
            budget["recent"] = (budget["recent"] + [row["seconds"]])[-10:]
            if budget["successful_seconds"] >= args.seconds - max(budget["recent"]) * 1.1:
                budget["stop_requested"] = True
                signal.raise_signal(signal.SIGTERM)
            return row

        runner.train_step = timed_step
        terminal = runner.run(plan, root / "run", device=args.device, resume=args.parent,
                              max_updates_this_invocation=50000)
        runner.train_step = native_step
        checkpoint = root / "run/checkpoint-progress.pt"
        saved, checkpoint_sha = load_checkpoint(checkpoint)
        assert saved["identity"] == plan.identity
        assert checkpoint_sha == terminal["checkpoint_sha256"]
        assert saved["state"]["update"] == terminal["durable_update"] == parent_update + budget["updates"]
        assert terminal["outcome"] == "interrupted" and budget["stop_requested"], terminal
        assert abs(sum(json.loads(line)["seconds"] for line in
                       (root / "run/updates.jsonl").open()) - budget["successful_seconds"]) < 1e-6
        receipt.update(outcome="training_completed", checkpoint=str(checkpoint),
                       checkpoint_sha256=checkpoint_sha, checkpoint_update=terminal["durable_update"],
                       added_updates=budget["updates"],
                       added_successful_seconds=budget["successful_seconds"],
                       final_stage_seconds=terminal["stage_seconds"][0],
                       old618_tables_with_gradient=sum(
                           1 for table in plan.tables if table.cohort == "history618"
                           and terminal["exposure"].get(table.name, {}).get("updates", 0)
                           > parent["state"]["exposure"].get(table.name, {}).get("updates", 0)))
        atomic_json(root / "budget.json", budget)
        atomic_json(root / "campaign.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("outcome", "checkpoint_update",
                         "checkpoint_sha256", "added_successful_seconds",
                         "old618_tables_with_gradient")}), flush=True)
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(root / "campaign.json", receipt)
        raise
    finally:
        runner.train_step = native_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "manifest", "parent", "parent-sha", "host", "device"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--half", type=int, required=True, choices=(1, 2))
    p.add_argument("--seconds", type=float, default=900)
    main(p.parse_args())
