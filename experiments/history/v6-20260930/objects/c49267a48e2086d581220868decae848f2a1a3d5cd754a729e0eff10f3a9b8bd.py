"""Launch the single bounded joint618 run and keep the Mac awake until it exits."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
assert not (ROOT / "joint.lock").exists(), "joint618 already started"
with (ROOT / "wrapper.log").open("x") as output:
    process = subprocess.Popen(["/usr/bin/python3", str(ROOT / "execute_joint.py")],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                               stderr=subprocess.STDOUT, start_new_session=True)
subprocess.Popen(["caffeinate", "-dimsu", "-w", str(process.pid)],
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
print(json.dumps({"wrapper_pid": process.pid, "root": str(ROOT)}))
