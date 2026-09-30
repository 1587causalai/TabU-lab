"""One shared OpenML12 checkpoint per host, with joint618 replay."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
from pathlib import Path

from tabu_lab.curriculum_v53 import runner
from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry


FOCUS = {"kin8nm", "pumadyn32nh"}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args, root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifests").mkdir()
    bank = read(Path(args.bank_root) / "bank.json")
    assert len(bank["tables"]) == 12
    inventory = read(args.inventory)
    assert inventory["table_count"] == len(inventory["tables"]) == 618
    parent, digest = load_checkpoint(args.parent)
    assert digest == args.parent_sha
    parent_plan = load_v55_plan(args.parent_manifest)
    assert parent["identity"] == parent_plan.identity
    parent_entries = {e["id"]: e for e in read(args.parent_manifest)["tables"]}
    entries = []
    adaptation = []
    for t in inventory["tables"]:
        original = parent_entries[t["id"]]
        path = Path(t["remote_paths"][args.host])
        assert sha256(path) == t["sha256"] == original["sha256"]
        entries.append(dict(id=t["id"], path=os.path.relpath(path, root / "manifests"),
                            sha256=t["sha256"], cohort="history618",
                            kind=t["kind"], role="train",
                            window_rows=original.get("window_rows", 204),
                            target_column=t["target_column"]))
    for t in bank["tables"]:
        name = t["name"]
        path = Path(args.adapted_root) / "data" / f"{name}.json"
        data = read(path)
        source = read(Path(args.bank_root) / t["path"])
        assert data["values"] == source["values"] and data["splits"] == source["splits"]
        assert set(data["splits"]["train"]).isdisjoint(data["splits"]["test"])
        window = t["train_n"] if t["cohort"] == "old3" else min(204, t["train_n"])
        entries.append(dict(id=name, path=os.path.relpath(path, root / "manifests"),
                            sha256=sha256(path),
                            cohort="focus2" if name in FOCUS else "other10",
                            kind="real", role="train", window_rows=window,
                            target_column=t["width"] - 1))
        adaptation.append(dict(name=name, data_sha256=sha256(path),
                               historical_data_sha256=t["sha256"],
                               values_and_splits_equal=True, window_rows=window))
    assert len(entries) == 630 and len({e["id"] for e in entries}) == 630
    old_stage = parent_plan.spec["stages"][0]
    spec = dict(schema="tabu.curriculum.v55.v1",
                experiment_id=f"openml12-joint618-replay-{args.host}-20260927",
                description=("Shared OpenML12 fit, historical 20-update block: "
                             "kin8nm/pumadyn32nh 3x, other real tables 1x, "
                             "four full-618 replay updates; fixed final test."),
                model=parent_plan.spec["model"], optimizer=parent_plan.spec["optimizer"],
                seeds=dict(model=20260908, order=20260909, masks=20260910,
                           codes=20260911, windows=20260912, evaluation=20260913),
                tables=entries, probes=[],
                stages=[dict(name="openml12_joint_replay", question="Fit the 12 real tables jointly",
                             max_updates=100000, max_seconds=max(9000, args.seconds * 6),
                             sampling=[dict(cohort="focus2", episodes=6),
                                       dict(cohort="other10", episodes=10),
                                       dict(cohort="history618", episodes=4)],
                             recipe=dict(real=dict(kind="supervised_row", fraction=1/3),
                                         synthetic=dict(kind="supervised_row", fraction=.25)),
                             optimizer="adamw", evaluate_every=100000,
                             checkpoint_every=250, probes=[], loss=old_stage["loss"],
                             objective=old_stage.get("objective", dict(kind="squared")))])
    manifest = root / "manifests" / "joint-openml12.json"
    atomic_json(manifest, spec)
    plan = load_v55_plan(manifest)
    assert len(plan.tables) == 630
    for cursor in range(20):
        table, _ = schedule_entry(plan, 0, cursor)
        assert table.cohort in ("focus2", "other10", "history618")
    atomic_json(root / "data-adaptation.json", adaptation)
    atomic_json(root / "campaign.json", dict(
        outcome="prepared", host=args.host, shared_checkpoint=True,
        parent=args.parent, parent_sha256=digest, parent_update=parent["state"]["update"],
        continuation="weights-only then strict resume", plan_identity=plan.identity,
        manifest=str(manifest), bank_sha256=sha256(Path(args.bank_root) / "bank.json"),
        inventory_sha256=sha256(args.inventory), script_sha256=sha256(__file__),
        successful_update_seconds_target=args.seconds,
        schedule="per 20 updates: focus2 3x each; other10 1x each; old618 replay 4x",
        test_exposure_audit="not performed by owner request"))
    return plan


def run(args):
    root = Path(args.root)
    receipt = None
    native_step = runner.train_step
    try:
        plan = prepare(args, root)
        receipt = read(root / "campaign.json")
        if args.prepare_only:
            print(json.dumps(dict(outcome="prepared", identity=plan.identity["sha256"])), flush=True)
            return
        runtime = runner.configure_runtime(args.device)
        receipt.update(outcome="admission", runtime=runtime)
        atomic_json(root / "campaign.json", receipt)
        admission = runner.run(plan, root / "runs" / "admission", device=args.device,
                               initialize_from=args.parent, max_updates_this_invocation=1)
        assert admission["outcome"] == "stopped" and admission["durable_update"] == 1, admission
        first = read(root / "runs" / "admission" / "terminal.json")
        first_row = json.loads((root / "runs" / "admission" / "updates.jsonl").read_text().splitlines()[0])
        budget = dict(successful_seconds=first_row["seconds"], updates=1,
                      recent=[first_row["seconds"]], stop_requested=False)

        def timed_step(*pos, **kw):
            row = native_step(*pos, **kw)
            budget["successful_seconds"] += row["seconds"]
            budget["updates"] += 1
            budget["recent"] = (budget["recent"] + [row["seconds"]])[-10:]
            if budget["successful_seconds"] >= args.seconds - max(budget["recent"]) * 1.1:
                budget["stop_requested"] = True
                signal.raise_signal(signal.SIGTERM)
            return row

        receipt.update(outcome="running", admission_checkpoint_sha256=first["checkpoint_sha256"])
        atomic_json(root / "campaign.json", receipt)
        runner.train_step = timed_step
        terminal = runner.run(plan, root / "runs" / "main", device=args.device,
                              resume=root / "runs" / "admission" / "checkpoint-progress.pt",
                              max_updates_this_invocation=50000)
        runner.train_step = native_step
        assert terminal["outcome"] == "interrupted" and budget["stop_requested"], terminal.get("error")
        assert terminal["durable_update"] == budget["updates"]
        assert budget["successful_seconds"] <= args.seconds + max(budget["recent"])
        checkpoint = root / "runs" / "main" / "checkpoint-progress.pt"
        saved, checkpoint_sha = load_checkpoint(checkpoint)
        assert saved["state"]["update"] == budget["updates"]
        assert saved["identity"] == plan.identity
        exposure = terminal["exposure"]
        by_cohort = {c: sum(v["updates"] for k, v in exposure.items()
                            if next(e["cohort"] for e in plan.spec["tables"] if e["id"] == k) == c)
                     for c in ("focus2", "other10", "history618")}
        assert sum(by_cohort.values()) == budget["updates"]
        assert by_cohort["history618"] > 0 and by_cohort["focus2"] > 0 and by_cohort["other10"] > 0
        atomic_json(root / "budget.json", budget)
        receipt.update(outcome="training_completed", checkpoint=str(checkpoint),
                       checkpoint_sha256=checkpoint_sha, checkpoint_update=budget["updates"],
                       successful_update_seconds=budget["successful_seconds"],
                       update_counts=by_cohort, old_tables_with_gradient=sum(
                           1 for e in plan.spec["tables"] if e["cohort"] == "history618"
                           and exposure.get(e["id"], {}).get("updates", 0) > 0))
        atomic_json(root / "campaign.json", receipt)
        print(json.dumps(dict(outcome=receipt["outcome"], update=budget["updates"],
                              seconds=budget["successful_seconds"], sha256=checkpoint_sha,
                              update_counts=by_cohort)), flush=True)
    except Exception as error:
        if receipt is not None:
            receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
            atomic_json(root / "campaign.json", receipt)
        raise
    finally:
        runner.train_step = native_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("root", "bank-root", "adapted-root", "inventory", "parent-manifest",
                 "parent", "parent-sha", "host", "device"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seconds", type=float, default=1440)
    p.add_argument("--prepare-only", action="store_true")
    run(p.parse_args())
