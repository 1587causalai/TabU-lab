"""Evaluate a saved paired-fit checkpoint on the frozen 8-window bank."""

import argparse
import json
import runpy
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model


def main(args):
    root = args.root.resolve()
    run = args.run.resolve()
    helper = runpy.run_path(str(root / "probe.py"))
    configure_runtime(args.device)
    plan = load_v55_plan(root / "manifests" / f"{args.arm}.json")
    checkpoint = run / f"{args.model}-{args.arm}" / f"checkpoint-u{args.update:04d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = (V6Model(plan.config) if args.model == "v6" else make_model(plan))
    model = model.to(args.device, dtype=execution_dtype(args.device))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    data = json.loads((root / "data" / f"{args.arm}.json").read_text())
    bank = json.loads((run / "evaluation-bank.json").read_text())
    metrics, _ = helper["evaluate"](model, data, bank, args.device)
    result = {"model": args.model, "arm": args.arm, "update": payload["state"]["update"],
              "train_step_seconds": payload["state"]["train_step_seconds"],
              "checkpoint_sha256": helper["digest"](checkpoint), "metrics": metrics}
    target = run / f"{args.model}-{args.arm}" / f"evaluation-u{args.update:04d}.json"
    helper["dump"](target, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--model", choices=("v55", "v6"), required=True)
    parser.add_argument("--arm", choices=("signal6", "full32"), required=True)
    parser.add_argument("--update", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
