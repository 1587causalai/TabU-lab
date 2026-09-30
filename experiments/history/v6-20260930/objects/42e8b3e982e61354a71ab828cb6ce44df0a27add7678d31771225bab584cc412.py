"""Deploy and run fixed-bank old618 retention for pre/mid/end checkpoints."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent
BASE = json.loads((HERE / "result-squared.json").read_text())
OLD_MANIFESTS = {
    "dgx2": "joint618-60min-20260926/manifests/joint618.json",
    "dustinstudio": "joint618-60min-dustin-20260926/manifests/01-joint618.json",
    "gongqian-mini": "joint618-60min-20260926/manifests/candidate.json",
}
BANK = HERE.parent / "joint618-60min-20260926/evidence/fixed-query-bank.json"


def start(host, config, label, smoke, deploy):
    home = Path(config["home"])
    source = home / "experiments" / config["source"] / "source/src"
    adapted = home / "experiments/openml12-finetune-20260927"
    root = home / "experiments/openml12-joint30m-squared-20260927"
    current = home / "experiments/openml12-joint2h-squared-20260927"
    script = adapted / "evaluate_old618.py"
    bank = root / "fixed-query-bank.json"
    if deploy:
        subprocess.run(["ssh", host, "mkdir", "-p", str(root)], check=True, timeout=30)
        subprocess.run(["scp", "-q", str(HERE / "evaluate_old618.py"), f"{host}:{script}"],
                       check=True, timeout=30)
        subprocess.run(["scp", "-q", str(BANK), f"{host}:{bank}"], check=True, timeout=60)
    if label == "pre":
        checkpoint = current / "runs/main/checkpoint-progress.pt"
        sha = BASE["hosts"][host]["checkpoint_sha256"]
    else:
        half = 1 if label == "mid" else 2
        code = '''import json,pathlib
p=pathlib.Path.home()/"experiments/openml12-joint30m-squared-20260927/half%d/campaign.json"
c=json.loads(p.read_text())
assert c["outcome"]=="training_completed"
print(json.dumps({"checkpoint":c["checkpoint"],"sha":c["checkpoint_sha256"]}))
''' % half
        done = subprocess.run(["ssh", host, "python3", "-"], input=code,
                              text=True, capture_output=True, timeout=30, check=True)
        info = json.loads(done.stdout)
        checkpoint, sha = Path(info["checkpoint"]), info["sha"]
    output = root / (f"retention-smoke-v2-{label}" if smoke else f"retention-{label}")
    command = [str(home / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{script}',run_name='__main__')",
               "--host", host, "--label", label,
               "--current-manifest", str(current / "manifests/joint-openml12-squared.json"),
               "--old-manifest", str(home / "experiments" / OLD_MANIFESTS[host]),
               "--checkpoint", str(checkpoint), "--expected-sha", sha,
               "--bank", str(bank), "--device", config["device"], "--output", str(output)]
    if smoke:
        command.append("--smoke")
    payload = dict(command=command, cwd=str(source), device=config["device"],
                   log=str(adapted / f"joint30m-old618-{'smoke-v2-' if smoke else ''}{label}.log"))
    remote = '''import json,os,subprocess
p=json.loads(%r)
env=os.environ.copy()
if p["device"]=="mps":
 env.update(PYTORCH_ENABLE_MPS_FALLBACK="0",PYTORCH_MPS_FAST_MATH="0")
else:
 env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
f=open(p["log"],"w",buffering=1)
proc=subprocess.Popen(p["command"],cwd=p["cwd"],stdout=f,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True,env=env)
print(json.dumps({"pid":proc.pid,"log":p["log"]}))
''' % json.dumps(payload)
    done = subprocess.run(["ssh", host, "python3", "-"], input=remote,
                          text=True, capture_output=True, timeout=30, check=True)
    result = json.loads(done.stdout)
    (HERE / f"old618-{'smoke-' if smoke else ''}{label}-{host}.json").write_text(
        json.dumps(result, indent=2) + "\n")
    return dict(host=host, label=label, smoke=smoke, **result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", choices=("pre", "mid", "end"), required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--deploy", action="store_true")
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(start, host, config, args.label, args.smoke, args.deploy): host
                for host, config in HOSTS.items()}
        for job in concurrent.futures.as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
