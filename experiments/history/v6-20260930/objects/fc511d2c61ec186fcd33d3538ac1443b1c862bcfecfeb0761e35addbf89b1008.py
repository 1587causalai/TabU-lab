"""Launch the corrected squared-loss stage on the three established hosts."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).parent
HOSTS = {
    "dgx2": dict(home="/home/cms", source="joint618-60min-20260926",
                 sha="c7a6030f3833e10ba73602536788da29e074f5ef30b308f939a23f873c8723f8",
                 device="cuda:0"),
    "dustinstudio": dict(home="/Users/dustinstudio", source="joint618-60min-dustin-20260926",
                         sha="70f76ae65c24f4481c46279390678bc57866fbdd6f5efd46c51662035082887b",
                         device="mps"),
    "gongqian-mini": dict(home="/Users/gongqian", source="joint618-60min-20260926",
                           sha="e3da2393c28f6f05cab7e9d143710e69db9618015205b47f1fd3e9a041543863",
                           device="mps"),
}


def start(host, config):
    home = Path(config["home"])
    source = home / "experiments" / config["source"] / "source/src"
    adapted = home / "experiments/openml12-finetune-20260927"
    old = home / "experiments/openml12-joint-fit-20260927"
    root = home / "experiments/openml12-joint2h-squared-20260927"
    script = adapted / "run_squared.py"
    subprocess.run(["scp", "-q", str(HERE / "run_squared.py"), f"{host}:{script}"],
                   timeout=30, check=True)
    command = [str(home / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{script}',run_name='__main__')",
               "--root", str(root), "--parent-manifest", str(old / "manifests/joint-openml12.json"),
               "--parent", str(old / "runs/main/checkpoints" / (config["sha"] + ".pt")),
               "--parent-sha", config["sha"], "--host", host,
               "--device", config["device"], "--seconds", "7190"]
    payload = dict(command=command, cwd=str(source), log=str(adapted / "joint2h-squared-launch.log"),
                   device=config["device"], root=str(root))
    remote = '''import json,os,subprocess
p=json.loads(%r)
assert os.path.isdir(p["cwd"]) and not os.path.exists(p["root"])
env=os.environ.copy()
if p["device"]=="mps":
 env.update(PYTORCH_ENABLE_MPS_FALLBACK="0",PYTORCH_MPS_FAST_MATH="0")
else:
 env["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
f=open(p["log"],"w",buffering=1)
proc=subprocess.Popen(p["command"],cwd=p["cwd"],stdout=f,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True,env=env)
print(json.dumps({"pid":proc.pid,"command":p["command"],"cwd":p["cwd"],"log":p["log"]}))
''' % json.dumps(payload)
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=remote, text=True, capture_output=True, timeout=30, check=True)
    result = json.loads(done.stdout)
    result.update(host=host, launched_utc=datetime.now(timezone.utc).isoformat())
    (HERE / f"launch-squared-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(host=host, pid=result["pid"])


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(start, host, config): host for host, config in HOSTS.items()}
        for job in concurrent.futures.as_completed(jobs):
            print(json.dumps(job.result()), flush=True)
