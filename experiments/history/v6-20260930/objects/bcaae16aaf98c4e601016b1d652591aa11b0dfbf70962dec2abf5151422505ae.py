"""Run one bounded 618-table Mini H4 continuation and its fixed-Query evaluations."""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PARENT = Path("/Users/gongqian/experiments/tabu-v55-triad90b-h4-mini-20260926/"
              "numeric/runs/attempt-002-main/checkpoint-progress.pt")
PARENT_EVALUATION = Path("/Users/gongqian/experiments/tabu-v55-triad90b-h4-mini-20260926/"
                         "numeric/evaluations/endpoint/terminal.json")
PARENT_SHA = "f7dc2f8919dd9703cb4f38b6997c46bc58f4561b75485b9a63733ecb41ce9053"
BASE = [str(Path.home() / ".local/bin/wehub-python"), "--profile", "train-20260920",
        "-u", "-m", "tabu_lab.cli", "curriculum-v55"]
ARMS = ("attempt-000-zero", "attempt-001-admission", "attempt-002-main")
PROBES = ("ordinal100_fit", "nominal100_fit", "tanh100_fit", "train_fit",
          "real78_fit", "new120_fit")
COHORT_SIZES = {"ordinal100": 100, "nominal100": 100, "tanh100": 100,
                "old120": 120, "new120": 120, "real78": 78}
STOP_SECONDS = 7196.0
BUDGET_SECONDS = 7200.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def journal() -> list[dict]:
    rows = []
    for arm in ARMS:
        path = ROOT / "runs" / arm / "updates.jsonl"
        if path.exists():
            for line in path.read_bytes().splitlines(keepends=True):
                if line.endswith(b"\n"):
                    rows.append(json.loads(line))
    return rows


def current(state: dict) -> None:
    for arm in reversed(ARMS):
        cp = ROOT / "runs" / arm / "checkpoint-progress.json"
        if cp.exists():
            durable = json.loads(cp.read_text())
            state["latest_saved_update"] = durable["update"]
            state["latest_saved_checkpoint_sha256"] = durable["sha256"]
            break
    evaluation = ROOT / "evaluations/endpoint/terminal.json"
    if evaluation.exists():
        result = json.loads(evaluation.read_text())
        if result.get("outcome") == "completed":
            state["latest_complete_evaluation_update"] = result["checkpoint_update"]
            state["latest_complete_evaluation_checkpoint_sha256"] = result["checkpoint_sha256"]
            state["latest_complete_evaluation_path"] = str(evaluation)
    elif (ROOT / "evaluations/step0/terminal.json").exists():
        result = json.loads((ROOT / "evaluations/step0/terminal.json").read_text())
        state["latest_complete_evaluation_update"] = result["checkpoint_update"]
        state["latest_complete_evaluation_checkpoint_sha256"] = result["checkpoint_sha256"]
        state["latest_complete_evaluation_path"] = str(ROOT / "evaluations/step0/terminal.json")
    atomic(ROOT / "joint.json", state)
    md = ["# Mini H4 joint618 当前状态", "",
          f"远端记录时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。",
          f"状态：`{state['status']}`；阶段：`{state.get('phase', 'preflight')}`。",
          f"父 checkpoint：`{PARENT_SHA}`。",
          f"实际成功更新秒数：`{state.get('actual_training_seconds', 0.0):.3f}` / `{BUDGET_SECONDS:.0f}`。",
          f"最新已保存点：update `{state.get('latest_saved_update', '尚无')}`，SHA `{state.get('latest_saved_checkpoint_sha256', '尚无')}`。",
          f"最新**已完成**固定评估：update `{state.get('latest_complete_evaluation_update', '尚无')}`，SHA `{state.get('latest_complete_evaluation_checkpoint_sha256', '尚无')}`。",
          f"评估回执：`{state.get('latest_complete_evaluation_path', '尚无')}`。",
          "", "这些指标限于已学表训练行 masked Query，不表示未见表泛化。", ""]
    temp = ROOT / "CURRENT.md.tmp"
    temp.write_text("\n".join(md))
    os.replace(temp, ROOT / "CURRENT.md")


def terminal(arm: str) -> dict:
    path = ROOT / "runs" / arm
    result = json.loads((path / "terminal.json").read_text())
    assert result["outcome"] in ("stopped", "interrupted") and not result.get("error")
    assert result["update"] == result["durable_update"]
    assert result["runtime"]["device"] == "mps" and result["runtime"]["dtype"] == "float32"
    assert sha(path / "checkpoint-progress.pt") == result["checkpoint_sha256"]
    assert json.loads((path / "checkpoint-progress.json").read_text())["sha256"] == result["checkpoint_sha256"]
    return result


def call(label: str, args: list[str], state: dict, *, arm: str | None = None,
         target_seconds: float | None = None) -> dict | None:
    evidence = ROOT / "evidence/execution"
    evidence.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "source/src"), PYTORCH_ENABLE_MPS_FALLBACK="0",
               PYTORCH_MPS_FAST_MATH="0", PYTHONDONTWRITEBYTECODE="1")
    record = {"label": label, "command": BASE + args,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running"}
    receipt = evidence / (label + "-launch.json")
    offsets = {}
    completed = 0
    seconds = 0.0
    if target_seconds is not None:
        prior = journal()
        completed = len(prior)
        seconds = sum(row["seconds"] for row in prior)
        offsets = {name: (ROOT / "runs" / name / "updates.jsonl").stat().st_size
                   if (ROOT / "runs" / name / "updates.jsonl").exists() else 0
                   for name in ARMS}
    with (evidence / (label + ".stdout.log")).open("x") as output:
        process = subprocess.Popen(BASE + args, cwd=ROOT / "source/src", env=env,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        record["pid"] = process.pid
        atomic(receipt, record)
        sent = False
        last_current = 0.0
        try:
            while process.poll() is None:
                if target_seconds is not None:
                    for name in ARMS:
                        path = ROOT / "runs" / name / "updates.jsonl"
                        if not path.exists():
                            continue
                        with path.open("rb") as source:
                            source.seek(offsets[name])
                            while True:
                                line = source.readline()
                                if not line or not line.endswith(b"\n"):
                                    break
                                row = json.loads(line)
                                assert row["update"] == completed + 1
                                completed += 1
                                seconds += row["seconds"]
                                offsets[name] = source.tell()
                    state["actual_training_seconds"] = seconds
                    state["actual_training_updates"] = completed
                    if seconds >= target_seconds and not sent:
                        process.send_signal(signal.SIGTERM)
                        sent = True
                        record.update(stop_reason="authorized_training_seconds",
                                      training_seconds_at_signal=seconds)
                        atomic(receipt, record)
                    if time.monotonic() - last_current > 30:
                        current(state)
                        last_current = time.monotonic()
                time.sleep(0.1 if target_seconds is not None else 1)
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=180)
            raise
        record["returncode"] = process.wait()
        record["status"] = "completed" if record["returncode"] == 0 or (
            sent and record["returncode"] == 3) else "failed"
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(receipt, record)
        assert record["status"] == "completed", record
    return terminal(arm) if arm else None


def run(label: str, arm: str, parent: Path, state: dict, *, initialize: bool = False,
        limit: int = 50000, target_seconds: float | None = None) -> dict:
    result = call(label, ["run", "--manifest", str(ROOT / "manifests/candidate.json"),
                          "--output-root", str(ROOT / "runs" / arm), "--device", "mps",
                          "--initialize-from" if initialize else "--resume-checkpoint", str(parent),
                          "--max-updates-this-invocation", str(limit)], state, arm=arm,
                  target_seconds=target_seconds)
    assert result is not None
    return result


def verify_old_probe_parity(initial: dict) -> None:
    parent = json.loads(PARENT_EVALUATION.read_text())
    assert parent["outcome"] == "completed" and parent["checkpoint_sha256"] == PARENT_SHA
    for name in PROBES[:5]:
        old, fresh = parent["probes"][name], initial["probes"][name]
        assert old["complete"] and fresh["complete"]
        left = {row["table"]: row for row in old["by_table"]}
        right = {row["table"]: row for row in fresh["by_table"]}
        assert left.keys() == right.keys()
        for table in left:
            a, b = left[table], right[table]
            assert a["query_rows"] == b["query_rows"]
            assert a["query_exposures"] == b["query_exposures"]
            assert a["unique_query_cells"] == b["unique_query_cells"]
            for metric, old_score in a["metrics"].items():
                new_score = b["metrics"][metric]
                if old_score is None:
                    assert new_score is None
                elif isinstance(old_score, (int, float)):
                    assert abs(old_score - new_score) <= 1e-8, (name, table, metric)


def evaluate(state: dict) -> dict:
    args = ["evaluate", "--manifest", str(ROOT / "manifests/candidate.json"),
            "--output-root", str(ROOT / "evaluations/endpoint"), "--device", "mps",
            "--checkpoint", str(ROOT / "runs/attempt-002-main/checkpoint-progress.pt")]
    for probe in PROBES:
        args += ["--probe", probe]
    call("endpoint", args, state)
    result = json.loads((ROOT / "evaluations/endpoint/terminal.json").read_text())
    checkpoint = terminal(ARMS[2])
    assert result["outcome"] == "completed"
    assert result["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    assert result["checkpoint_update"] == checkpoint["update"]
    assert set(result["probes"]) == set(PROBES)
    assert all(probe["complete"] for probe in result["probes"].values())
    return result


def main() -> None:
    (ROOT / "joint.lock").mkdir()
    assert sha(PARENT) == PARENT_SHA
    manifest = json.loads((ROOT / "manifests/candidate.json").read_text())
    plan = json.loads((ROOT / "plan/plan.json").read_text())
    bank = json.loads((ROOT / "evidence/actual-evaluation-bank-addresses.json").read_text())
    assert plan["outcome"] == "planned_not_run" and len(manifest["tables"]) == 618
    assert bank["manifest_file_sha256"] == sha(ROOT / "manifests/candidate.json")
    assert bank["mask_count"] == 1238
    assert manifest["model"]["backbone"]["heads"] == 4
    assert "objective" not in manifest["stages"][0]
    assert {row["cohort"]: row["episodes"] for row in manifest["stages"][0]["sampling"]} == COHORT_SIZES
    assert set(PROBES) == {probe["name"] for probe in manifest["probes"]}
    state = {"schema": "tabu.mini.joint618.120min-controller.v1", "status": "running",
             "phase": "preflight", "wrapper_pid": os.getpid(),
             "parent_checkpoint": str(PARENT), "parent_checkpoint_sha256": PARENT_SHA,
             "initialization": "weights_only_fresh_optimizer_rng_cursor_exposure",
             "continuation_within_stage": "strict_resume",
             "device": "mps", "dtype": "float32", "objective_kind": "squared",
             "budget_seconds": BUDGET_SECONDS, "stop_target_seconds": STOP_SECONDS,
             "cycle_cohort_updates": COHORT_SIZES,
             "actual_training_seconds": 0.0, "actual_training_updates": 0,
             "actual_query_bank_sha256": bank["query_address_bank_sha256"]}
    current(state)
    try:
        call("preflight", ["preflight", "--manifest", str(ROOT / "manifests/candidate.json"),
                           "--output-root", str(ROOT / "preflight"), "--device", "mps",
                           "--max-seconds", "120"], state)
        assert json.loads((ROOT / "preflight/terminal.json").read_text())["outcome"] == "passed"
        state["phase"] = "zero_step"; current(state)
        zero = run("zero", ARMS[0], PARENT, state, initialize=True, limit=0)
        assert zero["update"] == 0
        state["phase"] = "admission_initial_evaluation"; current(state)
        admitted = run("admission", ARMS[1], ROOT / "runs" / ARMS[0] / "checkpoint-progress.pt",
                       state, limit=1)
        assert admitted["update"] == 1
        first = json.loads((ROOT / "runs" / ARMS[1] / "evaluation-000-000000000-initial.json").read_text())
        assert first["update"] == 0 and set(first["probes"]) == set(PROBES)
        assert all(probe["complete"] for probe in first["probes"].values())
        verify_old_probe_parity(first)
        atomic(ROOT / "evaluations/step0/terminal.json", {
            "schema": "tabu.curriculum.v55.evaluation.v1", "outcome": "completed",
            "identity": zero["identity"], "runtime": zero["runtime"],
            "checkpoint_update": 0, "checkpoint_sha256": zero["checkpoint_sha256"],
            "checkpoint_source": str(ROOT / "runs" / ARMS[0] / "checkpoint-progress.pt"),
            "probes": first["probes"],
            "provenance": "Runner initial evaluation before first optimizer step after strict resume of zero checkpoint."})
        assert journal()[0]["objective_kind"] == "squared"
        state["phase"] = "training"; current(state)
        final = run("main", ARMS[2], ROOT / "runs" / ARMS[1] / "checkpoint-progress.pt",
                    state, target_seconds=STOP_SECONDS)
        rows = journal()
        seconds = sum(row["seconds"] for row in rows)
        assert len(rows) == final["update"] and STOP_SECONDS <= seconds <= BUDGET_SECONDS
        counts = collections.Counter(row["table"] for row in rows)
        assert len(counts) == 618 and min(counts.values()) >= 1
        cohort_counts = collections.Counter(row["cohort"] for row in rows)
        assert set(cohort_counts) == set(COHORT_SIZES)
        assert all(row["objective_kind"] == "squared" for row in rows)
        state.update(phase="endpoint_evaluation", actual_training_seconds=seconds,
                     actual_training_updates=len(rows), cohort_updates=dict(cohort_counts),
                     min_table_updates=min(counts.values()), max_table_updates=max(counts.values()))
        current(state)
        evaluate(state)
        state.update(status="completed", phase="completed", checkpoint_update=final["update"],
                     checkpoint_sha256=final["checkpoint_sha256"], budget_completed=True)
    except BaseException as error:
        state.update(status="failed", phase="failed", error_type=type(error).__name__,
                     error=str(error)[-1000:])
        raise
    finally:
        state["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        current(state)


if __name__ == "__main__":
    main()
