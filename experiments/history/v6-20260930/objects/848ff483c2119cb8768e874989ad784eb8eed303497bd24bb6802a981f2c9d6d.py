"""Launch one strict-resume 15-minute segment on the three hosts."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent
BASE = json.loads((HERE / "result-squared.json").read_text())


def start(host, config, half):
    home = Path(config["home"])
    adapted = home / "experiments/openml12-finetune-20260927"
    source = home / "experiments" / config["source"] / "source/src"
    first = home / "experiments/openml12-joint2h-squared-20260927"
    root = home / "experiments/openml12-joint30m-squared-20260927"
    segment = root / f"half{half}"
    script = adapted / "run_strict_half.py"
    subprocess.run(["scp", "-q", str(HERE / "run_strict_half.py"), f"{host}:{script}"],
                   timeout=30, check=True)
    if half == 1:
        parent = first / "runs/main/checkpoint-progress.pt"
        parent_sha = BASE["hosts"][host]["checkpoint_sha256"]
    else:
        code = '''import json,pathlib
p=pathlib.Path.home()/"experiments/openml12-joint30m-squared-20260927/half1/campaign.json"
c=json.loads(p.read_text())
assert c["outcome"]=="training_completed"
print(json.dumps({"path":c["checkpoint"],"sha":c["checkpoint_sha256"]}))
'''
        done = subprocess.run(["ssh", host, "python3", "-"], input=code,
                              text=True, capture_output=True, timeout=30, check=True)
        info = json.loads(done.stdout)
        parent, parent_sha = Path(info["path"]), info["sha"]
    command = [str(home / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{script}',run_name='__main__')",
               "--root", str(segment), "--manifest", str(first / "manifests/joint-openml12-squared.json"),
               "--parent", str(parent), "--parent-sha", parent_sha, "--host", host,
               "--device", config["device"], "--half", str(half), "--seconds", "900"]
    payload = dict(command=command, cwd=str(source), root=str(root), segment=str(segment),
                   log=str(adapted / f"joint30m-squared-half{half}.log"), device=config["device"])
    remote = '''import json,os,subprocess
p=json.loads(%r)
os.makedirs(p["root"],exist_ok=True)
assert not os.path.exists(p["segment"])
env=os.environ.copy()
if p["device"]=="mps":
 env.update(PYTORCH_ENABLE_MPS_FALLBACK="0",PYTORCH_MPS_FAST_MATH="0")
else:
 env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
f=open(p["log"],"w",buffering=1)
proc=subprocess.Popen(p["command"],cwd=p["cwd"],stdout=f,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True,env=env)
print(json.dumps({"pid":proc.pid,"log":p["log"],"parent":p["command"][p["command"].index("--parent")+1]}))
''' % json.dumps(payload)
    done = subprocess.run(["ssh", host, "python3", "-"], input=remote,
                          text=True, capture_output=True, timeout=30, check=True)
    result = json.loads(done.stdout)
    result.update(host=host, half=half, parent_sha256=parent_sha,
                  launched_utc=datetime.now(timezone.utc).isoformat())
    (HERE / f"launch-half{half}-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(host=host, pid=result["pid"], half=half)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--half", type=int, choices=(1, 2), required=True)
    args = p.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(start, host, config, args.half): host for host, config in HOSTS.items()}
        for job in concurrent.futures.as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
