"""Run the authorized ordinal100 window with checkpoint-bound evaluations.

Training time is the sum of successful ``train_step.seconds`` rows.  Initial,
interim, and final fixed-bank evaluations do not consume the 1800 second
training budget.  The first attempt is paused near 900 seconds for the
ordinal-only interim evaluation, then resumed strictly from its checkpoint.
"""

from pathlib import Path
import collections
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PARENT = Path(
    "/home/cms/experiments/tabu-v55-nominal100-retention-dgx2-20260926/"
    "runs/attempt-003-second30min/checkpoint-progress.pt"
)
PARENT_SHA256 = "cb0ed2b5e80612de00a1888dc8a345132f36dc92a3fa097ea3a473065135a776"
BASE = [
    str(Path.home() / ".local/bin/wehub-python"),
    "--profile", "train-20260920", "-u", "-m", "tabu_lab.cli", "curriculum-v55",
]
MANIFEST = ROOT / "manifests/candidate.json"
GATE = ROOT / "evidence/execution"
FIRST = ROOT / "runs/attempt-001-main"
SECOND = ROOT / "runs/attempt-002-after-interim"
RECEIPT = GATE / "ordinal-window-controller.json"
CURRENT = ROOT / "CURRENT.json"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(run):
    journal = run / "updates.jsonl"
    if not journal.exists():
        return []
    return [
        json.loads(line)
        for line in journal.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def training_seconds(run):
    return sum(row["seconds"] for row in rows(run))


def validate_terminal(run, identity):
    terminal = json.loads((run / "terminal.json").read_text())
    checkpoint = run / "checkpoint-progress.pt"
    sidecar = json.loads(checkpoint.with_suffix(".json").read_text())
    digest = sha256(checkpoint)
    assert terminal["outcome"] in ("stopped", "interrupted") and not terminal.get("error")
    assert terminal["identity"] == identity
    assert terminal["update"] == terminal["durable_update"] == sidecar["update"]
    assert digest == terminal["checkpoint_sha256"] == sidecar["sha256"]
    return terminal


def fixed_summary(probe):
    by_table = probe["by_table"]
    accuracy = [row["metrics"]["query_discrete_accuracy"] for row in by_table]
    rank_mae = [row["metrics"]["query_ordinal_rank_mae"] for row in by_table]
    return {
        "tables": len(by_table),
        "accuracy_mean": statistics.mean(accuracy),
        "accuracy_median": statistics.median(accuracy),
        "accuracy_p10": sorted(accuracy)[max(0, int(len(accuracy) * 0.1) - 1)],
        "accuracy_min": min(accuracy),
        "rank_mae_mean": statistics.mean(rank_mae),
        "rank_mae_median": statistics.median(rank_mae),
        "rank_mae_p90": sorted(rank_mae)[min(len(rank_mae) - 1, int(len(rank_mae) * 0.9))],
        "rank_mae_max": max(rank_mae),
    }


def main():
    assert not RECEIPT.exists() and not FIRST.exists() and not SECOND.exists()
    assert sha256(PARENT) == PARENT_SHA256
    plan_receipt = json.loads((ROOT / "evidence/validation/plan.json").read_text())
    identity = plan_receipt["identity"]
    smoke = json.loads((ROOT / "evidence/qualification/ordinal-cuda-smoke.json").read_text())
    assert smoke["outcome"] == "passed" and smoke["table_count"] == 100
    assert smoke["plan_identity"]["sha256"] == identity["sha256"]
    result = {
        "schema": "tabu.ordinal100.bounded-window-controller.v1",
        "status": "starting",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "identity": identity,
        "parent": {"path": str(PARENT), "sha256": PARENT_SHA256, "update": 13292},
        "initialization_mode": "weights_only_initialization",
        "optimizer_state": "fresh AdamW with unchanged configuration",
        "training_budget_seconds": 1800.0,
        "interim_target_seconds": 900.0,
        "evaluation_time_charged_to_training_budget": False,
        "commands": [],
    }

    def save_status(status, **fields):
        result.update(status=status, **fields)
        atomic(RECEIPT, result)
        atomic(CURRENT, {
            "schema": "tabu.ordinal100.current.v1",
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": status,
            "identity_sha256": identity["sha256"],
            "training_budget_seconds": 1800.0,
            **{key: result[key] for key in (
                "checkpoint_update", "checkpoint_sha256", "actual_training_seconds",
                "interim_update", "interim_checkpoint_sha256", "budget_completed",
            ) if key in result},
        })

    def call(command, label, run=None, stop_at=None, prior_seconds=0.0, bind_step0=False):
        started = time.monotonic()
        log = GATE / f"{label}.stdout.log"
        with log.open("x") as output:
            process = subprocess.Popen(
                command, cwd=ROOT / "source/src", stdout=output, stderr=subprocess.STDOUT
            )
            command_receipt = {
                "label": label,
                "command": command,
                "pid": process.pid,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            result["commands"].append(command_receipt)
            save_status(result["status"])
            sent = False
            baseline_bound = not bind_step0
            while process.poll() is None:
                if bind_step0 and not baseline_bound:
                    evaluation = run / "evaluation-000-000000000-initial.json"
                    sidecar_path = run / "checkpoint-progress.json"
                    checkpoint = run / "checkpoint-progress.pt"
                    if evaluation.exists() and sidecar_path.exists() and (
                        sidecar_path.stat().st_mtime_ns > evaluation.stat().st_mtime_ns
                    ):
                        sidecar = json.loads(sidecar_path.read_text())
                        event = json.loads(evaluation.read_text())
                        if sidecar["update"] == 0 and event["update"] == 0:
                            assert sidecar["identity"] == identity
                            assert all(probe["complete"] for probe in event["probes"].values())
                            checkpoint_dir = ROOT / "checkpoints"
                            checkpoint_dir.mkdir(parents=True, exist_ok=True)
                            frozen = checkpoint_dir / "step0.pt"
                            shutil.copy2(checkpoint, frozen)
                            frozen_digest = sha256(frozen)
                            if frozen_digest != sidecar["sha256"]:
                                frozen.unlink(missing_ok=True)
                                time.sleep(0.25)
                                continue
                            atomic(checkpoint_dir / "step0.json", sidecar)
                            baseline_receipt = {
                                "schema": "tabu.ordinal100.fixed-evaluation-point.v1",
                                "outcome": "completed",
                                "label": "step0",
                                "identity": identity,
                                "checkpoint_update": 0,
                                "checkpoint_sha256": frozen_digest,
                                "checkpoint_source": str(frozen),
                                "probes": event["probes"],
                                "origin": "runner initial fixed-bank evaluation",
                                "ordinal100_summary": fixed_summary(event["probes"]["ordinal100_fit"]),
                            }
                            atomic(ROOT / "evaluations/step0/terminal.json", baseline_receipt)
                            atomic(GATE / "zero-checkpoint-binding.json", sidecar)
                            baseline_bound = True
                            result["step0_checkpoint_sha256"] = frozen_digest
                            result["step0_ordinal100_summary"] = baseline_receipt["ordinal100_summary"]
                            save_status("training_to_interim")
                successful_seconds = prior_seconds + (training_seconds(run) if run else 0.0)
                if stop_at is not None and successful_seconds >= stop_at and not sent:
                    process.send_signal(signal.SIGTERM)
                    sent = True
                    command_receipt.update(
                        stop_reason="authorized_successful_training_seconds",
                        training_seconds_at_signal=successful_seconds,
                    )
                    save_status(result["status"], actual_training_seconds=successful_seconds)
                time.sleep(0.25)
            command_receipt.update(
                returncode=process.returncode,
                wall_seconds=time.monotonic() - started,
            )
            save_status(result["status"])
            if stop_at is None:
                assert process.returncode == 0, command_receipt
            else:
                assert sent and process.returncode == 3, command_receipt
            assert baseline_bound

    try:
        save_status("initial_fixed_evaluation")
        call(
            BASE + [
                "run", "--manifest", str(MANIFEST), "--output-root", str(FIRST),
                "--device", "cuda:0", "--initialize-from", str(PARENT),
                "--max-updates-this-invocation", "20000",
            ],
            "attempt-001-main", run=FIRST, stop_at=899.5, bind_step0=True,
        )
        first_terminal = validate_terminal(FIRST, identity)
        first_seconds = training_seconds(FIRST)
        result.update(
            interim_update=first_terminal["update"],
            interim_checkpoint_sha256=first_terminal["checkpoint_sha256"],
            actual_training_seconds=first_seconds,
        )
        save_status("interim_ordinal100_evaluation")
        interim = ROOT / "evaluations/interim-ordinal100"
        call(
            BASE + [
                "evaluate", "--manifest", str(MANIFEST), "--output-root", str(interim),
                "--device", "cuda:0", "--checkpoint", str(FIRST / "checkpoint-progress.pt"),
                "--probe", "ordinal100_fit",
            ],
            "interim-ordinal100-evaluation",
        )
        interim_terminal = json.loads((interim / "terminal.json").read_text())
        assert interim_terminal["outcome"] == "completed"
        assert interim_terminal["checkpoint_sha256"] == first_terminal["checkpoint_sha256"]
        assert interim_terminal["checkpoint_update"] == first_terminal["update"]
        interim_summary = fixed_summary(interim_terminal["probes"]["ordinal100_fit"])
        atomic(GATE / "interim-ordinal100-result.json", {
            "checkpoint_update": first_terminal["update"],
            "checkpoint_sha256": first_terminal["checkpoint_sha256"],
            "actual_training_seconds": first_seconds,
            **interim_summary,
        })
        result["interim_ordinal100_summary"] = interim_summary
        save_status("resumed_to_final")
        call(
            BASE + [
                "run", "--manifest", str(MANIFEST), "--output-root", str(SECOND),
                "--device", "cuda:0", "--resume-checkpoint",
                str(FIRST / "checkpoint-progress.pt"),
                "--max-updates-this-invocation", str(20000 - first_terminal["update"]),
            ],
            "attempt-002-after-interim", run=SECOND, stop_at=1798.0,
            prior_seconds=first_seconds,
        )
        final_terminal = validate_terminal(SECOND, identity)
        all_rows = rows(FIRST) + rows(SECOND)
        actual_seconds = sum(row["seconds"] for row in all_rows)
        assert len(all_rows) == final_terminal["update"]
        assert 1798.0 <= actual_seconds <= 1802.0
        result.update(
            checkpoint_update=final_terminal["update"],
            checkpoint_sha256=final_terminal["checkpoint_sha256"],
            actual_training_seconds=actual_seconds,
            cohort_updates=dict(collections.Counter(row["cohort"] for row in all_rows)),
        )
        save_status("final_fixed_evaluation")
        final_evaluation = ROOT / "evaluations/final-30min"
        call(
            BASE + [
                "evaluate", "--manifest", str(MANIFEST),
                "--output-root", str(final_evaluation), "--device", "cuda:0",
                "--checkpoint", str(SECOND / "checkpoint-progress.pt"),
            ],
            "final-fixed-evaluation",
        )
        final_receipt = json.loads((final_evaluation / "terminal.json").read_text())
        assert final_receipt["outcome"] == "completed"
        assert final_receipt["checkpoint_sha256"] == final_terminal["checkpoint_sha256"]
        assert final_receipt["checkpoint_update"] == final_terminal["update"]
        assert all(probe["complete"] for probe in final_receipt["probes"].values())
        result["final_ordinal100_summary"] = fixed_summary(
            final_receipt["probes"]["ordinal100_fit"]
        )
        result["budget_completed"] = True
        save_status("completed")
        atomic(GATE / "window-completion.json", result)
    except BaseException as error:
        result.update(
            status="failed", error_type=type(error).__name__, error=str(error)[-1200:],
        )
        atomic(RECEIPT, result)
        atomic(CURRENT, {
            "schema": "tabu.ordinal100.current.v1",
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "failed",
            "identity_sha256": identity["sha256"],
            "error_type": type(error).__name__,
            "error": str(error)[-1200:],
        })
        raise
    finally:
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic(RECEIPT, result)


if __name__ == "__main__":
    main()
