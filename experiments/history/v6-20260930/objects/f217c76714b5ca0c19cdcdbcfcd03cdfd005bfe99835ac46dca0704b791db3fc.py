"""Summarize equal-table retention deltas and worst regressions from fixed old618 bank."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


HERE = Path(__file__).parent
HOSTS = ("dgx2", "dustinstudio", "gongqian-mini")
METRICS = {"numeric": ("query_numeric_r2",),
           "nominal": ("query_discrete_accuracy",),
           "ordinal": ("query_discrete_accuracy", "query_ordinal_rank_mae")}


def flatten(stage):
    return {(group, table): value for group, item in stage["groups"].items()
            for table, value in item["tables"].items()}


def main():
    out = {"schema": "openml12-joint30m-old618-tail-v1", "hosts": {},
           "delta": "end minus pre; positive improves R2/accuracy, negative improves rank MAE"}
    for host in HOSTS:
        stages = json.loads((HERE / f"retention-summary-{host}.json").read_text())
        pre, mid, end = (flatten(stages[label]) for label in ("pre", "mid", "end"))
        assert set(pre) == set(mid) == set(end) and len(pre) == 618
        result = {"tables": 618, "metrics": {}}
        for kind, metrics in METRICS.items():
            for metric in metrics:
                rows = []
                for (group, table), item in pre.items():
                    if item["kind"] != kind:
                        continue
                    before = item["metrics"].get(metric)
                    middle = mid[(group, table)]["metrics"].get(metric)
                    after = end[(group, table)]["metrics"].get(metric)
                    if before is None or middle is None or after is None:
                        continue
                    delta = after - before
                    improvement = -delta if metric == "query_ordinal_rank_mae" else delta
                    rows.append(dict(group=group, table=table, pre=before, mid=middle,
                                     end=after, delta=delta, signed_improvement=improvement))
                assert rows
                ordered = sorted(rows, key=lambda x: x["signed_improvement"])
                result["metrics"][f"{kind}/{metric}"] = dict(
                    tables=len(rows), improved=sum(x["signed_improvement"] > 0 for x in rows),
                    worse=sum(x["signed_improvement"] < 0 for x in rows),
                    unchanged=sum(x["signed_improvement"] == 0 for x in rows),
                    mean_delta=statistics.fmean(x["delta"] for x in rows),
                    median_delta=statistics.median(x["delta"] for x in rows),
                    worst_five=ordered[:5])
        out["hosts"][host] = result
    (HERE / "retention-tail-30m.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({h: {k: {"improved": v["improved"], "worse": v["worse"],
                               "mean_delta": v["mean_delta"]}
                          for k, v in out["hosts"][h]["metrics"].items()} for h in HOSTS}))


if __name__ == "__main__":
    main()
