"""Forward three hosts' local update journals to isolated W&B queue files."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path


HERE = Path(__file__).parent
STATE = HERE / "collector-squared-state.json"
MONITOR = "/Users/dustinstudio/experiments/openml12-joint2h-squared-monitor-20260927"
HOSTS = ("dgx2", "dustinstudio", "gongqian-mini")
PARENT_UPDATES = {"dgx2": 1, "dustinstudio": 1, "gongqian-mini": 1}
ROOT_NAME = "openml12-joint2h-squared-20260927"


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def fetch(host, offset):
    code = '''import json,os
root=os.path.expanduser("~/experiments/openml12-joint2h-squared-20260927")
journal=root+"/runs/main/updates.jsonl"
campaign=root+"/campaign.json"
result={"offset":%d,"seconds_delta":0.0,"cohorts":{},"last":None,"campaign":None}
if os.path.isfile(campaign):
 result["campaign"]=json.load(open(campaign))["outcome"]
if os.path.isfile(journal):
 with open(journal,"rb") as f:
  f.seek(%d)
  while True:
   line=f.readline()
   if not line or not line.endswith(bytes([10])): break
   row=json.loads(line)
   result["offset"]=f.tell()
   result["seconds_delta"]+=row["seconds"]
   key=row["cohort"]
   result["cohorts"][key]=result["cohorts"].get(key,0)+1
   result["last"]={k:row[k] for k in ("update","loss","objective_loss","gradient_norm")}
print(json.dumps(result))
''' % (offset, offset)
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=code, text=True, capture_output=True, timeout=30, check=True)
    return json.loads(done.stdout)


def push(host, event):
    path = f"{MONITOR}/queue-{host}.jsonl"
    command = f"cat >> {shlex.quote(path)}"
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", "dustinstudio", command],
                          input=json.dumps(event, allow_nan=False) + "\n",
                          text=True, capture_output=True, timeout=30, check=True)
    if done.stdout.strip() or done.stderr.strip():
        raise RuntimeError("W&B queue append produced unexpected output")


def admission(host):
    code = '''import json,pathlib
p=pathlib.Path.home()/"experiments/openml12-joint2h-squared-20260927/runs/admission/updates.jsonl"
row=json.loads(p.read_text().splitlines()[0])
assert row["update"]==1 and row["objective_kind"]=="squared"
print(json.dumps({"seconds":row["seconds"],"cohort":row["cohort"]}))
'''
    done = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "python3", "-"],
                          input=code, text=True, capture_output=True, timeout=30, check=True)
    return json.loads(done.stdout)


def main():
    if STATE.exists():
        state = json.loads(STATE.read_text())
    else:
        state = {}
        for host in HOSTS:
            first = admission(host)
            cohorts = {"focus2": 0, "other10": 0, "history618": 0}
            cohorts[first["cohort"]] = 1
            state[host] = dict(offset=0, additional_successful_seconds=first["seconds"],
                               cohort_updates=cohorts, last_update=1, campaign="pending")
        atomic(STATE, state)
    while True:
        for host in HOSTS:
            previous = state[host]
            try:
                result = fetch(host, previous["offset"])
                previous["campaign"] = result["campaign"] or previous["campaign"]
                if result["last"] is None:
                    continue
                assert result["offset"] > previous["offset"]
                assert result["last"]["update"] > previous["last_update"]
                seconds = previous["additional_successful_seconds"] + result["seconds_delta"]
                cohorts = dict(previous["cohort_updates"])
                for name, count in result["cohorts"].items():
                    cohorts[name] += count
                event = dict(kind="train", host=host,
                             additional_successful_seconds=seconds,
                             cohort_updates=cohorts, **result["last"])
                push(host, event)
                previous.update(offset=result["offset"], last_update=event["update"],
                                additional_successful_seconds=seconds,
                                cohort_updates=cohorts)
                atomic(STATE, state)
                print(json.dumps(dict(host=host, update=event["update"],
                                      seconds=round(seconds, 2))), flush=True)
            except Exception as error:
                print(json.dumps(dict(host=host, error_type=type(error).__name__)), flush=True)
        atomic(STATE, state)
        if all(state[h]["campaign"] in ("training_completed", "budget_limited", "failed")
               for h in HOSTS):
            print("collector_terminal", flush=True)
            return
        time.sleep(45)


if __name__ == "__main__":
    main()
