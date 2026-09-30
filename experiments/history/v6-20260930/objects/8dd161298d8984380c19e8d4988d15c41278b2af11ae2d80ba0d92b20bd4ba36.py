"""Warm start a new, squared-loss joint OpenML12/old618 training stage."""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from tabu_lab.curriculum_v53 import runner
from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan, schedule_entry


def read(path):
    return json.loads(Path(path).read_text())


def prepare(args, root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "manifests").mkdir()
    old_manifest = Path(args.parent_manifest).resolve()
    old_plan = load_v55_plan(old_manifest)
    donor, donor_sha = load_checkpoint(args.parent)
    assert donor_sha == args.parent_sha and donor["identity"] == old_plan.identity
    assert donor["state"]["stage_index"] == 0
    spec = read(old_manifest)
    assert len(spec["tables"]) == 630 and not spec["probes"]
    assert [(x["cohort"], x["episodes"]) for x in spec["stages"][0]["sampling"]] == [
        ("focus2", 6), ("other10", 10), ("history618", 4)]
    for table in spec["tables"]:
        data_path = (old_manifest.parent / table["path"]).resolve()
        assert sha256(data_path) == table["sha256"]
        table["path"] = os.path.relpath(data_path, root / "manifests")
    spec["experiment_id"] = f"openml12-joint2h-squared-{args.host}-20260927"
    spec["description"] = (
        "Joint OpenML12 plus old618 replay; squared outer loss on every host. "
        "Weights-only initialization from the immutable 24-minute joint checkpoint; "
        "fresh optimizer, RNG, cursor and exposure.")
    stage = spec["stages"][0]
    stage["objective"] = {"kind": "squared"}
    stage["max_seconds"] = 11000
    stage["max_updates"] = 100000
    manifest = root / "manifests" / "joint-openml12-squared.json"
    atomic_json(manifest, spec)
    plan = load_v55_plan(manifest)
    assert plan.spec["stages"][0]["objective"] == {"kind": "squared"}
    assert len(plan.tables) == 630
    for cursor in range(20):
        table, _ = schedule_entry(plan, 0, cursor)
        assert table.cohort in ("focus2", "other10", "history618")
    receipt = dict(schema="tabu.openml12.joint2h.squared.v1", outcome="prepared",
                   host=args.host, manifest=str(manifest), identity=plan.identity,
                   objective={"kind": "squared"}, parent=str(Path(args.parent).resolve()),
                   parent_sha256=donor_sha, parent_update=donor["state"]["update"],
                   continuation="weights_only_new_objective_fresh_optimizer_rng_cursor_exposure",
                   successful_update_seconds_target=args.seconds,
                   sampling="focus2:6,other10:10,history618:4",
                   test_exposure_audit="not performed by owner request",
                   source_script_sha256=sha256(__file__))
    atomic_json(root / "campaign.json", receipt)
    return plan, receipt


def run(args):
    root = Path(args.root)
    receipt = None
    native_step = runner.train_step
    try:
        plan, receipt = prepare(args, root)
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
        assert first_row["objective_kind"] == "squared"
        budget = dict(successful_seconds=first_row["seconds"], updates=1,
                      recent=[first_row["seconds"]], stop_requested=False)

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

        receipt.update(outcome="running", admission_checkpoint_sha256=first["checkpoint_sha256"])
        atomic_json(root / "campaign.json", receipt)
        runner.train_step = timed_step
        terminal = runner.run(plan, root / "runs" / "main", device=args.device,
                              resume=root / "runs" / "admission" / "checkpoint-progress.pt",
                              max_updates_this_invocation=50000)
        runner.train_step = native_step
        assert terminal["outcome"] == "interrupted" and budget["stop_requested"], terminal.get("error")
        assert terminal["durable_update"] == budget["updates"]
        checkpoint = root / "runs" / "main" / "checkpoint-progress.pt"
        saved, checkpoint_sha = load_checkpoint(checkpoint)
        assert saved["identity"] == plan.identity and saved["state"]["update"] == budget["updates"]
        exposure = terminal["exposure"]
        cohort_of = {e["id"]: e["cohort"] for e in plan.spec["tables"]}
        counts = {c: sum(v["updates"] for key, v in exposure.items() if cohort_of[key] == c)
                  for c in ("focus2", "other10", "history618")}
        assert sum(counts.values()) == budget["updates"] and all(counts.values())
        receipt.update(outcome="training_completed", checkpoint=str(checkpoint),
                       checkpoint_sha256=checkpoint_sha, checkpoint_update=budget["updates"],
                       successful_update_seconds=budget["successful_seconds"],
                       cohort_updates=counts,
                       old_tables_with_gradient=sum(1 for e in plan.spec["tables"]
                                                    if e["cohort"] == "history618"
                                                    and exposure.get(e["id"], {}).get("updates", 0) > 0))
        atomic_json(root / "budget.json", budget)
        atomic_json(root / "campaign.json", receipt)
        print(json.dumps({k: receipt[k] for k in ("outcome", "checkpoint_update",
                         "checkpoint_sha256", "successful_update_seconds", "cohort_updates")}), flush=True)
    except Exception as error:
        if receipt is not None:
            receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
            atomic_json(root / "campaign.json", receipt)
        raise
    finally:
        runner.train_step = native_step


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "parent-manifest", "parent", "parent-sha", "host", "device"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--seconds", type=float, default=7190)
    p.add_argument("--prepare-only", action="store_true")
    run(p.parse_args())
