"""Summarize checkpoint-bound ordinal100 start/interim/final receipts."""

from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left = math.floor(position)
    return ordered[left] + (
        ordered[min(left + 1, len(ordered) - 1)] - ordered[left]
    ) * (position - left)


def distribution(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p10": percentile(values, 0.1),
        "p90": percentile(values, 0.9),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "table_count": len(values),
    }


def main():
    output = ROOT / "evidence/window-result.json"
    if output.exists():
        raise FileExistsError(output)
    completion = json.loads(
        (ROOT / "evidence/execution/window-completion.json").read_text()
    )
    assert completion["status"] == "completed" and completion["budget_completed"] is True
    points = {
        "step0": json.loads((ROOT / "evaluations/step0/terminal.json").read_text()),
        "interim": json.loads(
            (ROOT / "evaluations/interim-ordinal100/terminal.json").read_text()
        ),
        "final": json.loads((ROOT / "evaluations/final-30min/terminal.json").read_text()),
    }
    manifest = json.loads((ROOT / "manifests/candidate.json").read_text())
    kinds = {}
    for table in manifest["tables"]:
        raw = (ROOT / "manifests" / table["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == table["sha256"]
        kinds[table["id"]] = json.loads(raw)["target_kind"]
    authority = json.loads(
        (ROOT / "evidence/validation/actual-evaluation-bank-authority.json").read_text()
    )
    groups = [
        ("ordinal100", "ordinal100_fit", "ordinal", "query_discrete_accuracy"),
        ("ordinal100_rank", "ordinal100_fit", "ordinal", "query_ordinal_rank_mae"),
        ("nominal100", "nominal100_fit", "nominal", "query_discrete_accuracy"),
        ("numeric100", "tanh100_fit", "numeric", "query_numeric_r2"),
        ("old120_numeric", "train_fit", "numeric", "query_numeric_r2"),
        ("old120_nominal", "train_fit", "nominal", "query_discrete_accuracy"),
        ("old120_ordinal", "train_fit", "ordinal", "query_discrete_accuracy"),
        ("old120_ordinal_rank", "train_fit", "ordinal", "query_ordinal_rank_mae"),
        ("real78", "real78_fit", "numeric", "query_numeric_r2"),
    ]
    result = {
        "schema": "tabu.ordinal100.window-result.v2",
        "status": "completed",
        "distribution_percentile_method": "linear Type 7 over equal-weight table metrics",
        "identity_sha256": completion["identity"]["sha256"],
        "checkpoint_update": completion["checkpoint_update"],
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "actual_training_seconds": completion["actual_training_seconds"],
        "cohort_updates": completion["cohort_updates"],
        "actual_query_bank_sha256": authority["actual_query_bank_sha256"],
        "external_baseline_alignment_established": False,
        "points": {
            name: {
                "checkpoint_update": receipt["checkpoint_update"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            }
            for name, receipt in points.items()
        },
        "groups": {},
    }
    for label, probe, kind, metric in groups:
        per_point = {}
        table_values = {}
        for point, receipt in points.items():
            if probe not in receipt["probes"]:
                continue
            rows = [
                row for row in receipt["probes"][probe]["by_table"]
                if kinds[row["table"]] == kind
            ]
            values = {
                row["table"]: row["metrics"][metric]
                for row in rows
                if isinstance(row["metrics"].get(metric), (int, float))
                and math.isfinite(row["metrics"][metric])
            }
            per_point[point] = distribution(list(values.values()))
            table_values[point] = values
        start = table_values["step0"]
        final = table_values["final"]
        assert start.keys() == final.keys()
        delta = {table: final[table] - start[table] for table in start}
        lower_is_better = metric in ("query_ordinal_rank_mae",)
        improved = sum(
            value < 0 if lower_is_better else value > 0
            for value in delta.values()
        )
        tied = sum(value == 0 for value in delta.values())
        declined = len(delta) - improved - tied
        result["groups"][label] = {
            "probe": probe,
            "kind": kind,
            "metric": metric,
            "points": per_point,
            "paired_final_minus_step0": distribution(list(delta.values())),
            "final_minus_step0_increased_tied_decreased": [
                sum(value > 0 for value in delta.values()),
                sum(value == 0 for value in delta.values()),
                sum(value < 0 for value in delta.values()),
            ],
            "improved_tied_declined": [improved, tied, declined],
            "lower_is_better": lower_is_better,
            "final_worst10": sorted(
                final.items(),
                key=lambda item: item[1],
                reverse=metric in ("query_ordinal_rank_mae",),
            )[:10],
            "per_table": {
                table: {
                    "step0": start[table],
                    "final": final[table],
                    "delta": delta[table],
                }
                for table in start
            },
        }
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))


if __name__ == "__main__":
    main()
