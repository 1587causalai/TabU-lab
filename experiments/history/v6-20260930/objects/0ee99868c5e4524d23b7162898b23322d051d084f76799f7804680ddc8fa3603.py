"""Arm full-context R²/S_log evaluation for each terminal checkpoint."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent


def start(host, config):
    home = Path(config["home"])
    source = home / "experiments" / config["source"] / "source/src"
    adapted = home / "experiments/openml12-finetune-20260927"
    script = adapted / "evaluate_squared_when_ready.py"
    subprocess.run(["scp", "-q", str(HERE / "evaluate_squared_when_ready.py"),
                    f"{host}:{script}"], check=True, timeout=30)
    command = ["python3", "-u", str(script),
               "--host", host, "--root", str(home / "experiments/openml12-joint2h-squared-20260927"),
               "--bank-root", str(home / "experiments/openml12-frozen-icl-20260927"),
               "--evaluator", str(adapted / "evaluate_full_context.py"),
               "--source", str(source), "--device", config["device"]]
    payload = dict(command=command, log=str(adapted / "joint2h-squared-evaluation.log"))
    remote = '''import json,subprocess
p=json.loads(%r)
f=open(p["log"],"w",buffering=1)
proc=subprocess.Popen(p["command"],stdout=f,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({"pid":proc.pid,"log":p["log"]}))
''' % json.dumps(payload)
    done = subprocess.run(["ssh", host, "python3", "-"], input=remote,
                          text=True, capture_output=True, check=True, timeout=30)
    result = json.loads(done.stdout)
    (HERE / f"eval-squared-launch-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(host=host, **result)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(start, host, config): host for host, config in HOSTS.items()}
        for job in concurrent.futures.as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
