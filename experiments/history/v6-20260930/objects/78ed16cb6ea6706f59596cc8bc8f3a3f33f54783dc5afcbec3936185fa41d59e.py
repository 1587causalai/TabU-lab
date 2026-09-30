"""Finish the 7200-second joint run if invocation one reaches its 24k cap."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import signal
import time

import execute_joint as first

ROOT = Path(__file__).resolve().parent
STAGE = "01-joint618"
INITIAL = ROOT / "runs" / STAGE
RESUMED = ROOT / "runs" / "02-strict-resume"
MANIFEST = ROOT / "manifests/joint618-60min-v55.json"
IDENTITY = "2a7d192be89b8963a6eab90915df11a6f0727898daed4587df32a1d1d7aa8a2c"
SUPERVISOR = ROOT / "receipts/strict-resume-supervisor.json"
EXPECTED = ("ordinal100", "nominal100", "tanh100", "old120", "new120", "real78")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def report(status, **extra):
    first.atomic(SUPERVISOR, {"status": status, "utc": utc(), **extra})


def current(status, attempt, update, seconds, cohorts):
    side = attempt / "checkpoint-progress.json"
    saved = read(side) if side.exists() else {}
    endpoint = ROOT / "evaluations" / STAGE / "endpoint/terminal.json"
    step0 = ROOT / "evaluations" / STAGE / "step0/terminal.json"
    latest_eval = "endpoint" if endpoint.exists() else "step0" if step0.exists() else "none"
    lines = [
        "# Dustin joint618 120-minute run", "",
        f"Status: {status}",
        f"Stage: {STAGE}",
        f"Latest observed update: {update}",
        f"Successful training seconds: {seconds:.3f} / <7200",
        f"Latest durable checkpoint: {attempt}/checkpoint-progress.pt, update {saved.get('update', 'none')}, SHA {saved.get('sha256', 'none')}",
        f"Latest completed full fixed-Query evaluation: {latest_eval}",
        f"Cohort updates: {json.dumps(cohorts, sort_keys=True)}",
        f"Parent: {first.PARENT} (SHA {first.PARENT_SHA})",
        "Continuation: strict resume of model, optimizer, RNG, cursor and exposure",
        "W&B: https://wandb.ai/zj3712/restoration-v55-single-dgp/runs/v55-dustin-joint618-2a7d192b",
        "",
    ]
    target = ROOT / "CURRENT.md"
    temp = ROOT / "CURRENT.md.tmp"
    temp.write_text("\n".join(lines))
    os.replace(temp, target)


def wait_first():
    report("armed", expected_first_invocation_limit=24000,
           strict_resume_update_limit=100000, target_seconds=first.TARGET_SECONDS,
           hard_limit_seconds=7200.0)
    while True:
        controller = ROOT / "receipts/controller-terminal.json"
        if controller.exists() and read(controller).get("status") == "completed":
            report("not_needed_first_invocation_completed")
            return False
        terminal = ROOT / "receipts/01-joint618-terminal.json"
        if terminal.exists():
            status = read(terminal).get("status")
            if status == "completed":
                report("not_needed_first_invocation_completed")
                return False
            if status == "failed":
                return True
        time.sleep(10)


def validate_first():
    owner = read(ROOT / "receipts/01-joint618-terminal.json")
    terminal = read(INITIAL / "terminal.json")
    side = read(INITIAL / "checkpoint-progress.json")
    digest = first.sha(INITIAL / "checkpoint-progress.pt")
    if owner.get("error") != "training update seconds outside authorized stage window":
        raise ValueError("first invocation did not stop at the expected update cap")
    if not (terminal["outcome"] == "stopped" and not terminal.get("error")
            and terminal["update"] == terminal["durable_update"] == side["update"] == 24000
            and terminal["identity"]["sha256"] == side["identity"]["sha256"] == IDENTITY
            and digest == terminal["checkpoint_sha256"] == side["sha256"]
            and first.sha(INITIAL / "checkpoints" / (digest + ".pt")) == digest):
        raise ValueError("first invocation stopped without a complete strict-resume checkpoint")
    first.verify_parent_baseline(STAGE)
    offset, update, seconds, cohorts = first.consume(
        INITIAL / "updates.jsonl", 0, 0, 0.0, {name: 0 for name in EXPECTED})
    if update != 24000 or seconds >= 7200.0:
        raise ValueError("first invocation clock/update mismatch or hard cap exceeded")
    first.atomic(ROOT / "receipts/01-joint618-invocation-limit.json", owner)
    return digest, update, seconds, cohorts


def finish(attempt, update, seconds, cohorts, term, first_seconds):
    if not first.TARGET_SECONDS <= seconds < 7200.0:
        raise ValueError("combined successful-update clock outside two-hour window")
    final = ROOT / "receipts/01-joint618-terminal.json"
    first.atomic(final, {
        "status": "training_window_completed_evaluation_pending", "utc": utc(),
        "stage": STAGE, "update": update, "checkpoint_sha256": term["checkpoint_sha256"],
        "actual_training_seconds": seconds, "first_invocation_seconds": first_seconds,
        "cohort_updates": cohorts, "strict_resume": attempt == RESUMED,
    })
    current("endpoint_evaluating", attempt, update, seconds, cohorts)
    first.fixed_evaluate(STAGE, MANIFEST, attempt, IDENTITY, term)
    first.atomic(final, {
        "status": "completed", "utc": utc(), "stage": STAGE,
        "update": update, "checkpoint_sha256": term["checkpoint_sha256"],
        "actual_training_seconds": seconds, "first_invocation_seconds": first_seconds,
        "cohort_updates": cohorts, "strict_resume": attempt == RESUMED,
        "step0_evaluation": str(ROOT / "evaluations" / STAGE / "step0/terminal.json"),
        "endpoint_evaluation": str(ROOT / "evaluations" / STAGE / "endpoint/terminal.json"),
    })
    first.atomic(ROOT / "receipts/controller-terminal.json", {
        "status": "completed", "utc": utc(), "actual_training_seconds_total": seconds,
        "latest_checkpoint": str(attempt / "checkpoint-progress.pt"),
        "latest_checkpoint_sha256": term["checkpoint_sha256"],
        "stages": [STAGE], "strict_resume": attempt == RESUMED,
    })
    current("completed", attempt, update, seconds, cohorts)
    report("completed", update=update, actual_training_seconds=seconds,
           checkpoint_sha256=term["checkpoint_sha256"])


def resume(digest, update, seconds, cohorts):
    if RESUMED.exists():
        raise FileExistsError("strict-resume attempt already exists")
    first_seconds = seconds
    command = first.BASE + [
        "run", "--manifest", str(MANIFEST), "--output-root", str(RESUMED),
        "--device", "mps",
        "--resume-checkpoint", str(INITIAL / "checkpoint-progress.pt"),
        "--max-updates-this-invocation", "100000",
    ]
    process = first.start(command, ROOT / "receipts/02-strict-resume-training.stdout")
    first.atomic(ROOT / "receipts/02-strict-resume-launch.json", {
        "utc": utc(), "pid": process.pid, "command": command, "identity_sha256": IDENTITY,
        "resume_checkpoint_sha256": digest, "resume_checkpoint_update": update,
        "resume_mode": "strict_model_optimizer_rng_cursor_exposure",
        "cumulative_training_seconds_before_resume": seconds,
        "target_seconds": first.TARGET_SECONDS, "hard_limit_seconds": 7200.0,
    })
    report("resumed", pid=process.pid, parent_checkpoint_sha256=digest,
           parent_update=update, first_invocation_seconds=seconds)
    offset = 0
    stopped = False
    last_report = 0.0
    try:
        while process.poll() is None:
            offset, update, seconds, cohorts = first.consume(
                RESUMED / "updates.jsonl", offset, update, seconds, cohorts)
            if seconds >= first.TARGET_SECONDS and not stopped:
                os.kill(process.pid, signal.SIGTERM)
                stopped = True
                first.atomic(ROOT / "receipts/02-strict-resume-stop-signal.json", {
                    "utc": utc(), "pid": process.pid, "update": update,
                    "combined_successful_training_seconds": seconds,
                })
            if time.monotonic() - last_report >= 15:
                first.atomic(ROOT / "receipts/controller-progress.json", {
                    "utc": utc(), "stage": STAGE, "attempt": "02-strict-resume",
                    "update": update, "actual_training_seconds": seconds,
                    "cohort_updates": cohorts, "stop_signal_sent": stopped, "pid": process.pid,
                })
                current("strict_resume_training", RESUMED, update, seconds, cohorts)
                last_report = time.monotonic()
            time.sleep(0.5)
        process.wait()
        offset, update, seconds, cohorts = first.consume(
            RESUMED / "updates.jsonl", offset, update, seconds, cohorts)
        if not stopped:
            raise RuntimeError("strict resume ended before two-hour target")
        term = first.checkpoint(RESUMED, IDENTITY, update)
        finish(RESUMED, update, seconds, cohorts, term, first_seconds)
    except BaseException:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGTERM)
            process.wait(timeout=180)
        raise


def main():
    if not wait_first():
        return
    try:
        digest, update, seconds, cohorts = validate_first()
        if seconds >= first.TARGET_SECONDS:
            term = read(INITIAL / "terminal.json")
            finish(INITIAL, update, seconds, cohorts, term, seconds)
        else:
            resume(digest, update, seconds, cohorts)
    except BaseException as error:
        report("failed", error_type=type(error).__name__, error=str(error)[:1000])
        raise


if __name__ == "__main__":
    main()
