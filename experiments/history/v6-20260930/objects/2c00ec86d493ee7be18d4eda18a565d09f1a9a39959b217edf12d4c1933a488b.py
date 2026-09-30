"""Independent online W&B observer for the DGX2 V6 update journal."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import wandb


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def fetch(remote_root, offset):
    script = "\n".join([
        "import json,pathlib",
        f"root=pathlib.Path({remote_root!r})",
        f"offset={offset}",
        "campaign=root/'campaign.json'",
        "journal=root/'updates.jsonl'",
        "out={'campaign':json.loads(campaign.read_text()) if campaign.exists() else None,"
        "'offset':offset,'rows':[]}",
        "if journal.exists():",
        " with journal.open('rb') as stream:",
        "  stream.seek(offset)",
        "  while True:",
        "   line=stream.readline()",
        "   if not line or not line.endswith(b'\\n'):break",
        "   out['rows'].append(json.loads(line))",
        "   out['offset']=stream.tell()",
        "print(json.dumps(out))",
    ])
    done = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx2", "python3", "-"],
        input=script, text=True, capture_output=True, check=True, timeout=30,
    )
    return json.loads(done.stdout)


def main(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    state = dict(status="starting", remote_root=args.remote_root, offset=0,
                 last_update=0, run_id=args.run_id)
    status = root / "status.json"
    atomic(status, state)
    run = None
    try:
        settings = wandb.Settings(
            disable_git=True, disable_code=True, console="off", silent=True,
            x_disable_stats=True, x_disable_meta=True, init_timeout=45,
        )
        run = wandb.init(
            entity="zj3712", project="restoration-v6-target-broadcast",
            id=args.run_id, name="dgx2-H8-puma-old618", resume="allow",
            mode="online", settings=settings,
            config=dict(continuation="V5.5_to_V6_weights_only", host="dgx2",
                        parent_sha256=args.parent_sha, target_update_seconds=900,
                        sampling="focus2:6,other10:10,history618:4"),
        )
        state.update(status="ready", url=run.url)
        atomic(status, state)
        while True:
            response = fetch(args.remote_root, state["offset"])
            for row in response["rows"]:
                update = row["update"]
                if update <= state["last_update"]:
                    raise ValueError("V6 update journal regressed")
                run.log({
                    "train/loss": row["loss"],
                    "train/gradient_norm": row["gradient_norm"],
                    "train/successful_seconds": row["successful_update_seconds"],
                    "train/query_rows": row["query_rows"],
                    "train/scored_cells": row["scored_cells"],
                    "cohort/" + row["cohort"]: update,
                }, step=update)
                state["last_update"] = update
            state["offset"] = response["offset"]
            campaign = response["campaign"] or {}
            state["campaign_outcome"] = campaign.get("outcome")
            atomic(status, state)
            if campaign.get("outcome") in ("training_completed", "failed"):
                if (campaign.get("checkpoint_update", 0) == state["last_update"]
                        or campaign["outcome"] == "failed"):
                    state["status"] = ("completed" if campaign["outcome"] ==
                                       "training_completed" else "training_failed")
                    state["checkpoint_sha256"] = campaign.get("checkpoint_sha256")
                    atomic(status, state)
                    run.finish(exit_code=0 if state["status"] == "completed" else 1)
                    return
            time.sleep(10)
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        atomic(status, state)
        if run:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("root", "remote-root", "run-id", "parent-sha"):
        parser.add_argument("--" + name, required=True)
    main(parser.parse_args())
