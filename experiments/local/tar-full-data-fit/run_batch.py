"""Finite full-data diagnostic with one shared wall-clock budget."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=["tabu-lab"])
    args = parser.parse_args()
    spec_path = Path(__file__).resolve().with_name("preregistration.yaml")
    spec = json.loads(spec_path.read_text())
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + min(1800, spec["batch_wall_seconds"])
    outcomes = []

    def write(status):
        payload = dict(
            status=status,
            elapsed_seconds=time.monotonic() - started,
            batch_wall_seconds=min(1800, spec["batch_wall_seconds"]),
            lanes=outcomes,
        )
        temp = root / "batch-status.tmp"
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(root / "batch-status.json")

    for seed in spec["seeds"]:
        for dataset in ("iris", "diabetes"):
            remaining = deadline - time.monotonic() - 15
            if remaining <= spec["finalization_reserve_seconds"]:
                write("budget_exhausted")
                return 3
            lane_seconds = min(spec["lane_wall_seconds"], remaining)
            env = os.environ.copy()
            env["TABU_TAR_FIT_DEADLINE_UNIX"] = str(time.time() + lane_seconds)
            attempt = root / f"attempt-{dataset}-{seed}"
            # Shared file also reaches isolated Docker runtimes that filter env vars.
            (root / f"{attempt.name}.deadline.json").write_text(
                json.dumps(dict(deadline_unix=float(env["TABU_TAR_FIT_DEADLINE_UNIX"]))) + "\n"
            )
            command = [
                *args.command,
                "tar",
                "fit",
                "--preregistration",
                str(spec_path),
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
                "--output-root",
                str(attempt),
            ]
            write("running")
            with (root / f"{dataset}-{seed}.log").open("x") as log:
                proc = subprocess.Popen(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    code = proc.wait(timeout=lane_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        code = proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        code = proc.wait(timeout=3)
            result = attempt / "result.json"
            outcome = (
                json.loads(result.read_text()).get("outcome")
                if result.exists()
                else "missing_result"
            )
            outcomes.append(
                dict(
                    dataset=dataset, seed=seed, exit_code=code, timed_out=timed_out, outcome=outcome
                )
            )
            if (
                code
                or timed_out
                or outcome in ("failed", "interrupted", "blocked_resources", "missing_result")
            ):
                write("blocked")
                return 3
    write("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
