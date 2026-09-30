"""After paired midpoint tests complete, launch the second strict-resume segment."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from launch_squared import HOSTS


HERE = Path(__file__).parent
BASE = json.loads((HERE / "result-squared.json").read_text())
REMOTE = '''import json,pathlib
h=pathlib.Path.home()/"experiments"
r=h/"openml12-joint30m-squared-20260927/half1/campaign.json"
e=h/"openml12-frozen-icl-20260927/evaluation-joint30m-squared-half1-full-20260927/metrics-r2-slog.json"
out={"campaign":None,"metrics":None}
if r.exists():
 c=json.loads(r.read_text())
 out["campaign"]={k:c.get(k) for k in ("outcome","parent_sha256","parent_update","checkpoint_sha256","checkpoint_update","added_successful_seconds","error")}
if e.exists(): out["metrics"]=json.loads(e.read_text())
print(json.dumps(out))
'''


def fetch(host):
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=REMOTE, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


def main():
    while True:
        state = {host: fetch(host) for host in HOSTS}
        if any(v["campaign"] and v["campaign"]["outcome"] == "failed"
               for v in state.values()):
            raise RuntimeError("half1 training failed")
        if all(v["metrics"] for v in state.values()):
            break
        time.sleep(30)
    report = {"schema": "openml12-joint30m-midpoint-v1", "hosts": {}}
    for host, item in state.items():
        c, m = item["campaign"], item["metrics"]
        assert c["outcome"] == "training_completed" and m["outcome"] == "completed"
        assert c["checkpoint_sha256"] == m["checkpoint_sha256"]
        assert c["parent_sha256"] == BASE["hosts"][host]["checkpoint_sha256"]
        report["hosts"][host] = dict(checkpoint_sha256=c["checkpoint_sha256"],
                                     checkpoint_update=c["checkpoint_update"],
                                     added_successful_seconds=c["added_successful_seconds"],
                                     start_macro=BASE["hosts"][host]["terminal_macro"],
                                     midpoint_macro=m["macro"])
    (HERE / "mid30m.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({h: v["midpoint_macro"] for h, v in report["hosts"].items()}), flush=True)
    subprocess.run([sys.executable, str(HERE / "launch_strict_half.py"), "--half", "2"],
                   cwd=HERE, check=True)
    subprocess.run([sys.executable, str(HERE / "launch_eval_30m.py"), "--half", "2"],
                   cwd=HERE, check=True)
    print("half2_launched", flush=True)


if __name__ == "__main__":
    main()
