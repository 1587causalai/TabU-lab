"""Independent online W&B mirror of verified local training receipts."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import wandb


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    queue = root / f"queue-{args.host}.jsonl"
    queue.touch(exist_ok=True)
    ready = root / f"wandb-{args.host}.json"
    state = dict(status="starting", host=args.host, run_id=args.run_id,
                 project=args.project, group=args.group, queue=str(queue), offset=0,
                 last_update=args.parent_update)
    atomic(ready, state)
    run = None
    try:
        settings = wandb.Settings(disable_git=True, disable_code=True,
                                  console="off", silent=True,
                                  x_disable_stats=True, x_disable_meta=True)
        run = wandb.init(entity=args.entity, project=args.project,
                         id=args.run_id, name=f"openml12-joint2h-squared-{args.host}",
                         group=args.group, resume="allow", mode="online",
                         settings=settings,
                         config=dict(host_role=args.host, model=args.model,
                                     objective="squared",
                                     continuation="weights_only_from_24min_checkpoint",
                                     parent_checkpoint_sha256=args.parent_sha,
                                     manifest_identity_sha256=args.identity,
                                     target_additional_update_seconds=args.target_seconds,
                                     sampling="focus2:6,other10:10,history618:4"))
        if run is None:
            raise RuntimeError("wandb init returned no run")
        run.define_metric("update")
        for pattern in ("train/*", "progress/*", "evaluation/*"):
            run.define_metric(pattern, step_metric="update")
        state.update(status="ready", url=run.url)
        atomic(ready, state)
        offset = 0
        while True:
            with queue.open("rb") as file:
                file.seek(offset)
                while True:
                    line = file.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    event = json.loads(line)
                    if event["host"] != args.host:
                        raise ValueError("queue host mismatch")
                    if event["kind"] == "train":
                        update = event["update"]
                        if update < state["last_update"]:
                            raise ValueError("W&B update regressed")
                        values = {"update": update,
                                  "train/loss": event["loss"],
                                  "train/objective_loss": event["objective_loss"],
                                  "train/gradient_norm": event["gradient_norm"],
                                  "progress/additional_successful_seconds": event["additional_successful_seconds"],
                                  "progress/old618_updates": event["cohort_updates"]["history618"],
                                  "progress/real12_updates": (
                                      event["cohort_updates"]["focus2"] +
                                      event["cohort_updates"]["other10"])}
                        run.log(values)
                        state["last_update"] = update
                    elif event["kind"] == "final":
                        values = {"update": event["update"],
                                  "progress/additional_successful_seconds": event["additional_successful_seconds"],
                                  "evaluation/r2_macro": event["r2_macro"],
                                  "evaluation/slog_macro": event["slog_macro"],
                                  "evaluation/complete": 1}
                        run.log(values)
                        run.summary.update(values)
                        state.update(status="completed", last_update=event["update"])
                    else:
                        raise ValueError("unknown queue event")
                    offset = file.tell()
                    state["offset"] = offset
                    atomic(ready, state)
                    if state["status"] == "completed":
                        run.finish()
                        return
            time.sleep(3)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__)
        atomic(ready, state)
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "host", "run-id", "entity", "project", "group", "model",
                "parent-sha", "identity"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--parent-update", type=int, required=True)
    p.add_argument("--target-seconds", type=float, default=7190)
    main(p.parse_args())
