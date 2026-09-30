"""Start isolated W&B observers for the corrected squared-loss campaign."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent
MONITOR = Path("/Users/dustinstudio/experiments/openml12-joint30m-squared-monitor-20260927")
BASE = json.loads((HERE / "result-squared.json").read_text())
IDENTITIES = {
    "dgx2": "d1db41341ad1d933a9105b312b878f1320d711021a916bedf264c41641fcae8a",
    "dustinstudio": "37c739f8a25a1b47952a1600f01fdd3b09ca7743a4259449495a6861d3be1905",
    "gongqian-mini": "d6f33c33a2a4bc6731a69c7fe580855d8cece3ce31774f1202c5e58b7a4bfb8e",
}


def start(host):
    model = "H4" if host == "gongqian-mini" else "H8"
    run_id = "openml12sq30m" + {"dgx2": "d", "dustinstudio": "s", "gongqian-mini": "m"}[host] + "20260927"
    python = "/Users/dustinstudio/.wehub/envs/train-torch213-py312-mps-20260920/bin/python"
    command = ["uv", "run", "--no-project", "--offline", "--python", python,
               "--with", "wandb==0.23.0", "python", "-u", str(MONITOR / "wandb_observer_30m.py"),
               "--root", str(MONITOR), "--host", host, "--run-id", run_id,
               "--entity", "zj3712", "--project", "restoration-v55-single-dgp",
               "--group", "openml12-joint30m-squared-20260927", "--model", model,
               "--parent-sha", BASE["hosts"][host]["checkpoint_sha256"],
               "--identity", IDENTITIES[host],
               "--parent-update", str(BASE["hosts"][host]["checkpoint_update"]),
               "--target-seconds", "1800"]
    payload = dict(command=command, log=str(MONITOR / f"observer-{host}.log"))
    remote = '''import json,subprocess
p=json.loads(%r)
f=open(p["log"],"w",buffering=1)
proc=subprocess.Popen(p["command"],stdout=f,stderr=subprocess.STDOUT,
                      stdin=subprocess.DEVNULL,start_new_session=True)
print(json.dumps({"pid":proc.pid,"log":p["log"]}))
''' % json.dumps(payload)
    done = subprocess.run(["ssh", "dustinstudio", "python3", "-"], input=remote,
                          text=True, capture_output=True, check=True, timeout=30)
    result = json.loads(done.stdout)
    (HERE / f"wandb-30m-launch-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
    return dict(host=host, **result)


if __name__ == "__main__":
    subprocess.run(["ssh", "dustinstudio", "mkdir", "-p", str(MONITOR)], check=True, timeout=30)
    subprocess.run(["scp", "-q", str(HERE / "wandb_observer_30m.py"),
                    f"dustinstudio:{MONITOR}/wandb_observer_30m.py"], check=True, timeout=30)
    for host in HOSTS:
        print(json.dumps(start(host)), flush=True)
