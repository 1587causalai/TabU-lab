"""Focused equal-compute-time readout of the 50-update V6 full32 checkpoint."""

import json
import runpy
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model

root = Path(__file__).resolve().parent
probe = runpy.run_path(str(root / "probe.py"))
run = root / "run-h8-150"
configure_runtime("cuda:0")
plan = load_v55_plan(root / "manifests/full32.json")
checkpoint = run / "v6-full32/checkpoint-u0050.pt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
model = V6Model(plan.config).to("cuda:0", dtype=execution_dtype("cuda:0"))
model.load_state_dict(payload["model"], strict=True)
model.eval().requires_grad_(False)
data = json.loads((root / "data/full32.json").read_text())
bank = json.loads((run / "evaluation-bank.json").read_text())
metrics, _ = probe["evaluate"](model, data, bank, "cuda:0")
result = {"model": "v6", "arm": "full32", "update": payload["state"]["update"],
          "train_step_seconds": payload["state"]["train_step_seconds"],
          "checkpoint_sha256": probe["digest"](checkpoint), "metrics": metrics}
probe["dump"](run / "v6-full32/evaluation-u0050.json", result)
print(json.dumps(result), flush=True)
