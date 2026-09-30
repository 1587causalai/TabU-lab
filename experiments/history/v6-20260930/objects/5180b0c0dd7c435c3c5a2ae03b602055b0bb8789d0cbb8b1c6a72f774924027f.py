"""Arm frozen OpenML12 R² and S_log scoring for a 15-minute segment."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent


def start(host, config, half):
    home = Path(config["home"])
    source = home / "experiments" / config["source"] / "source/src"
    adapted = home / "experiments/openml12-finetune-20260927"
    script = adapted / "evaluate_30m_checkpoint.py"
    subprocess.run(["scp", "-q", str(HERE / "evaluate_30m_checkpoint.py"),
                    f"{host}:{script}"], check=True, timeout=30)
    label = f"half{half}"
    command = ["python3", "-u", str(script),
               "--host", host, "--label", label,
               "--root", str(home / "experiments/openml12-joint30m-squared-20260927" / label),
               "--bank-root", str(home / "experiments/openml12-frozen-icl-20260927"),
               "--evaluator", str(adapted / "evaluate_full_context.py"),
               "--source", str(source), "--device", config["device"]]
    payload = dict(command=command, log=str(adapted / f"joint30m-squared-{label}-evaluation.log"))
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
    (HERE / f"eval30m-{label}-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(host=host, half=half, **result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--half", type=int, choices=(1, 2), required=True)
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(start, host, config, args.half): host for host, config in HOSTS.items()}
        for job in concurrent.futures.as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
