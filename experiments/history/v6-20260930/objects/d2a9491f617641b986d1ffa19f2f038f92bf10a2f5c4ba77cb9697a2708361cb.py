"""Evaluate the fixed historical OpenML12 bank after the joint fit is durable."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main(args):
    root = Path(args.root)
    while True:
        p = root / "campaign.json"
        if p.exists():
            campaign = json.loads(p.read_text())
            if campaign["outcome"] == "failed":
                raise RuntimeError(f"training failed: {campaign.get('error')}")
            if campaign["outcome"] == "training_completed":
                break
        time.sleep(12)
    checkpoint = Path(campaign["checkpoint"])
    assert checkpoint.is_file() and campaign["checkpoint_sha256"]
    output = "evaluation-openml12-joint-20260927"
    assert not (Path(args.bank_root) / output).exists()
    command = [str(Path.home() / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{args.evaluator}', run_name='__main__')",
               "--root", args.bank_root, "--manifest", campaign["manifest"],
               "--checkpoint", str(checkpoint), "--expected-sha", campaign["checkpoint_sha256"],
               "--device", args.device, "--output", output]
    print(json.dumps(dict(command=command, checkpoint_sha256=campaign["checkpoint_sha256"])), flush=True)
    env = os.environ.copy()
    if args.device == "mps":
        env.update(PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0")
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    done = subprocess.run(command, cwd=args.source, env=env, check=False)
    if done.returncode:
        raise SystemExit(done.returncode)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "bank-root", "evaluator", "source", "device"):
        p.add_argument("--" + key, required=True)
    main(p.parse_args())
