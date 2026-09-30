"""Evaluate a strict-resume 15/30-minute checkpoint on the frozen OpenML12 bank."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from pathlib import Path


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def run(args):
    root, bank_root = Path(args.root), Path(args.bank_root)
    while True:
        path = root / "campaign.json"
        if path.exists():
            campaign = json.loads(path.read_text())
            if campaign["outcome"] == "failed":
                raise RuntimeError(f"training failed: {campaign.get('error_type')}: {campaign.get('error')}")
            if campaign["outcome"] == "training_completed":
                break
        time.sleep(15)
    manifest = json.loads(Path(campaign["manifest"]).read_text())
    assert manifest["stages"][0]["objective"] == {"kind": "squared"}
    checkpoint = Path(campaign["checkpoint"])
    assert checkpoint.is_file() and campaign["checkpoint_sha256"]
    output_name = f"evaluation-joint30m-squared-{args.label}-full-20260927"
    output = bank_root / output_name
    assert not output.exists()
    command = [str(Path.home() / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{args.evaluator}',run_name='__main__')",
               "--host", args.host, "--bank-root", str(bank_root),
               "--manifest", campaign["manifest"], "--checkpoint", str(checkpoint),
               "--expected-sha", campaign["checkpoint_sha256"],
               "--device", args.device, "--output", output_name]
    env = os.environ.copy()
    if args.device == "mps":
        env.update(PYTORCH_ENABLE_MPS_FALLBACK="0", PYTORCH_MPS_FAST_MATH="0")
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    done = subprocess.run(command, cwd=args.source, env=env, check=False)
    if done.returncode:
        raise RuntimeError(f"full-context evaluator exited {done.returncode}")
    receipt = json.loads((output / "terminal.json").read_text())
    assert receipt["outcome"] == "completed" and receipt["checkpoint_sha256"] == campaign["checkpoint_sha256"]
    bank = json.loads((bank_root / "bank.json").read_text())
    table_scores = {}
    for entry in bank["tables"]:
        name = entry["name"]
        data = json.loads((bank_root / entry["path"]).read_text())
        values = data["values"]
        train = [values[i][-1] for i in data["splits"]["train"]]
        median = statistics.median(train)
        mad = statistics.median(abs(y - median) for y in train)
        assert mad > 1e-12
        test_ids = set(data["splits"]["test"])
        rows = json.loads((output / (name + "-predictions.json")).read_text())
        assert {r["row_id"] for r in rows} == test_ids and len(rows) == len(test_ids)
        assert all(r["target"] == values[r["row_id"]][-1] for r in rows)
        numerator = statistics.fmean(math.log1p(((r["target"] - r["prediction"]) / mad) ** 2)
                                     for r in rows)
        denominator = statistics.fmean(math.log1p(((r["target"] - median) / mad) ** 2)
                                       for r in rows)
        table_scores[name] = dict(r2=receipt["datasets"][name]["metrics"]["r2"],
                                  slog=1 - numerator / denominator,
                                  support_rows=len(train), query_rows=len(rows), train_mad=mad)
    assert len(table_scores) == 12
    metrics = dict(schema="openml12-joint30m-squared-r2-slog-v1", outcome="completed",
                   host=args.host, label=args.label,
                   checkpoint_sha256=campaign["checkpoint_sha256"],
                   checkpoint_update=campaign["checkpoint_update"],
                   added_successful_seconds=campaign["added_successful_seconds"],
                   slog_formula="1 - mean(log1p(((y-pred)/s)^2)) / mean(log1p(((y-median_train)/s)^2))",
                   scale="s=max(raw_train_MAD, 1e-12) in original target units",
                   tables=table_scores,
                   macro={key: statistics.fmean(t[key] for t in table_scores.values())
                          for key in ("r2", "slog")})
    assert abs(metrics["macro"]["r2"] - receipt["macro_r2"]) < 1e-10
    atomic(output / "metrics-r2-slog.json", metrics)
    print(json.dumps(dict(host=args.host, checkpoint_update=metrics["checkpoint_update"],
                          macro=metrics["macro"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("root", "bank-root", "evaluator", "source", "device", "host", "label"):
        p.add_argument("--" + key, required=True)
    run(p.parse_args())
