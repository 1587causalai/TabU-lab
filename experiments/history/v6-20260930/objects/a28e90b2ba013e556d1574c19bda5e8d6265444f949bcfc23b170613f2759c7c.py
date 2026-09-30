"""Launch one bounded shared-12 fit on each established training host."""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


HOSTS = {
    "dgx2": dict(home="/home/cms", source="joint618-60min-20260926",
                 manifest="joint618.json",
                 parent="runs/attempt-002-main/checkpoints/1366ac9d552864f80df2c52247476215b87b5d74915d0697676ba1a9d8eeb1a6.pt",
                 device="cuda:0"),
    "dustinstudio": dict(home="/Users/dustinstudio", source="joint618-60min-dustin-20260926",
                         manifest="01-joint618.json",
                         parent="runs/02-strict-resume/checkpoints/fdb28693e5894d93dd0ccbdb90cb34d18ab06acbbfb08022085a213afda316ab.pt",
                         device="mps"),
    "gongqian-mini": dict(home="/Users/gongqian", source="joint618-60min-20260926",
                           manifest="candidate.json",
                           parent="runs/attempt-002-main/checkpoints/c579d50ce4caf00d6a6c47408f20f259782c40def1eba9133092eeb81f12aa34.pt",
                           device="mps"),
}


def start(host, c):
    home = Path(c["home"])
    source = home / "experiments" / c["source"]
    adapted = home / "experiments" / "openml12-finetune-20260927"
    root = home / "experiments" / "openml12-joint-fit-20260927"
    command = [str(home / ".local/bin/wehub-python"), "--profile", "train-20260920",
               "-u", "-c", f"import runpy; runpy.run_path('{adapted / 'run_joint.py'}', run_name='__main__')",
               "--root", str(root), "--bank-root", str(home / "experiments/openml12-frozen-icl-20260927"),
               "--adapted-root", str(adapted), "--inventory", str(adapted / "used-data-inventory.json"),
               "--parent-manifest", str(source / "manifests" / c["manifest"]),
               "--parent", str(source / c["parent"]),
               "--parent-sha", Path(c["parent"]).stem,
               "--host", host, "--device", c["device"], "--seconds", "1440"]
    payload = dict(command=command, cwd=str(source / "source/src"),
                   log=str(adapted / "joint-fit-launch.log"), device=c["device"])
    remote = '''import json,os,subprocess
p=json.loads(%r)
assert not os.path.exists(p["command"][p["command"].index("--root")+1])
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
    completed = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                               input=remote, text=True, capture_output=True, timeout=30, check=True)
    info = json.loads(completed.stdout)
    info.update(host=host, launched_utc=datetime.now(timezone.utc).isoformat())
    (Path(__file__).parent / f"launch-{host}.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(start, host, c): host for host, c in HOSTS.items()}
        for future in concurrent.futures.as_completed(futures):
            info = future.result()
            print(json.dumps({"host": info["host"], "pid": info["pid"]}), flush=True)
