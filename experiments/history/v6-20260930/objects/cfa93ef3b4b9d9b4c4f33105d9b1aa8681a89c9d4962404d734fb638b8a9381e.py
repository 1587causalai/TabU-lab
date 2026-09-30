"""Retrospectively score saved OpenML12 full-context predictions with S_log."""
from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).parent
ROOT = HERE.parent.parent
BASE = json.loads((HERE.parent / "openml12-frozen-icl-20260927" / "results.json").read_text())
RECEIPTS = json.loads((HERE / "full-context-receipts.json").read_text())
HOSTS = ("dgx2", "dustinstudio", "gongqian-mini")
FLOOR = 1e-12  # Original target units; verified inactive for all 12 tables.

REMOTE = r'''
import json, pathlib, statistics
root = pathlib.Path.home() / "experiments/openml12-frozen-icl-20260927"
bank = json.loads((root / "bank.json").read_text())
out = {}
for entry in bank["tables"]:
    name = entry["name"]
    data = json.loads((root / entry["path"]).read_text())
    values = data["values"]
    train = [values[i][-1] for i in data["splits"]["train"]]
    median = statistics.median(train)
    mad = statistics.median([abs(y - median) for y in train])
    test_ids = set(data["splits"]["test"])
    runs = {}
    for key, folder in (("parent", "evaluation-parent-fulltrain-fulltest-20260927"),
                        ("tuned", "evaluation-fulltrain-fulltest-20260927")):
        rows = json.loads((root / folder / (name + "-predictions.json")).read_text())
        assert len(rows) == len(test_ids) and {r["row_id"] for r in rows} == test_ids
        assert all(r["target"] == values[r["row_id"]][-1] for r in rows)
        runs[key] = rows
    out[name] = dict(data_sha256=entry["sha256"], train_n=len(train), test_n=len(test_ids),
                     train_median=median, train_mad=mad, predictions=runs)
print(json.dumps(out, allow_nan=False))
'''


def r2(rows, key):
    ys = [r["target"] for r in rows]
    mean = statistics.fmean(ys)
    return 1 - math.fsum((r["target"] - r[key]) ** 2 for r in rows) / math.fsum(
        (y - mean) ** 2 for y in ys)


def slog(rows, key, median, scale):
    numerator = statistics.fmean(math.log1p(((r["target"] - r[key]) / scale) ** 2)
                                  for r in rows)
    denominator = statistics.fmean(math.log1p(((r["target"] - median) / scale) ** 2)
                                    for r in rows)
    return 1 - numerator / denominator


def main():
    names = list(BASE["historical_reference"]["datasets"])
    remote = {}
    for host in HOSTS:
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                                input=REMOTE, text=True, capture_output=True, timeout=120,
                                check=True)
        remote[host] = json.loads(result.stdout)
        assert set(remote[host]) == set(names)
    output = {"schema": "openml12-full-context-r2-slog-v1",
              "slog_formula": "1 - mean(log1p(((y-pred)/s)^2)) / mean(log1p(((y-median_train)/s)^2))",
              "scale": "s=max(raw_train_MAD, 1e-12) in original target units",
              "floor": FLOOR, "retrospective": True, "floor_activated_tables": [],
              "tables": {}}
    for name in names:
        ref = BASE["historical_reference"]["datasets"][name]
        first = remote[HOSTS[0]][name]
        median, mad = first["train_median"], first["train_mad"]
        if mad < FLOOR:
            output["floor_activated_tables"].append(name)
        scale = max(mad, FLOOR)
        table = {"support_rows": first["train_n"], "query_rows": first["test_n"],
                 "data_sha256": first["data_sha256"], "train_median": median,
                 "train_mad": mad, "models": {}}
        assert ref["data_sha256"] == first["data_sha256"]
        for host in HOSTS:
            item = remote[host][name]
            assert (item["data_sha256"], item["train_n"], item["test_n"],
                    item["train_median"], item["train_mad"]) == (
                        first["data_sha256"], first["train_n"], first["test_n"], median, mad)
            for stage in ("parent", "tuned"):
                rows = item["predictions"][stage]
                receipt = RECEIPTS[stage][host]
                expected = receipt["datasets"][name]["metrics"]["r2"]
                actual = r2(rows, "prediction")
                assert abs(actual - expected) < 1e-10, (host, stage, name, actual, expected)
                table["models"][f"{host}_{stage}"] = {
                    "r2": expected, "slog": slog(rows, "prediction", median, scale),
                    "checkpoint_sha256": receipt["checkpoint_sha256"]}
        archive = ROOT / "archive/tar-openml12-test-eval-kinpuma3x-20260910/output" / (name + "-predictions.json")
        rows = json.loads(archive.read_text())
        assert len(rows) == first["test_n"]
        assert {r["row_id"] for r in rows} == {r["row_id"] for r in first["predictions"]["tuned"]}
        expected = ref["xgboost"]["r2"]
        actual = r2(rows, "xgboost")
        assert abs(actual - expected) < 1e-10, (name, actual, expected)
        table["models"]["xgboost"] = {"r2": expected,
                                          "slog": slog(rows, "xgboost", median, scale)}
        output["tables"][name] = table
    output["macro"] = {model: {
        metric: statistics.fmean(output["tables"][name]["models"][model][metric] for name in names)
        for metric in ("r2", "slog")}
        for model in [*(f"{h}_{stage}" for h in HOSTS for stage in ("parent", "tuned")), "xgboost"]}
    (HERE / "metrics-r2-slog.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["macro"], ensure_ascii=False))


if __name__ == "__main__":
    main()
