"""Frozen OpenML12 evaluation with all train rows as support and all test rows as Query."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch

from tabu_lab.curriculum_v53.artifacts import atomic_json, load_checkpoint, sha256
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype

sys.path.insert(0, str(Path(__file__).parent))
from frozen_evaluate import make_input, metrics, state_hash


def run(args):
    bank_root = Path(args.bank_root)
    output = bank_root / args.output
    output.mkdir(exist_ok=False)
    receipt = dict(schema="tabu.openml12.full-train-full-test.v1", outcome="running",
                   protocol="one forward per table: every train row is support, every test row is Query",
                   host=args.host, datasets={}, bank_sha256=sha256(bank_root / "bank.json"),
                   evaluator_sha256=sha256(__file__), optimizer_updates=0)
    atomic_json(output / "started.json", receipt)
    try:
        runtime = configure_runtime(args.device)
        plan = load_v55_plan(args.manifest)
        payload, digest = load_checkpoint(args.checkpoint)
        assert digest == args.expected_sha and payload["identity"] == plan.identity
        model = make_model(plan).to(device=args.device, dtype=execution_dtype(args.device))
        assert model.config.as_dict() == payload["model_config"]
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        before = state_hash(model)
        receipt.update(runtime=runtime, checkpoint=str(Path(args.checkpoint).resolve()),
                       checkpoint_sha256=digest, checkpoint_update=payload["state"]["update"],
                       identity=plan.identity, model_state_sha256=before)
        atomic_json(output / "resolved.json", receipt)
        del payload
        bank = json.loads((bank_root / "bank.json").read_text())
        selected = [e for e in bank["tables"] if not args.only or e["name"] in args.only]
        assert len(selected) == (len(args.only) if args.only else 12)
        for entry in selected:
            start = time.monotonic()
            name = entry["name"]
            path = bank_root / entry["path"]
            assert sha256(path) == entry["sha256"]
            data = json.loads(path.read_text())
            support = data["splits"]["train"]
            query = data["splits"]["test"]
            assert len(support) == entry["train_n"] and len(query) == entry["test_n"]
            assert not set(support).intersection(query)
            trace = dict(context_row_ids=support, query_row_ids=query,
                         codebook_seed=entry["traces"][0]["codebook_seed"])
            inputs, request = make_input(data, trace, name, args.device, execution_dtype(args.device))
            if args.device == "mps":
                torch.mps.synchronize()
            elif args.device == "cuda:0":
                torch.cuda.synchronize()
            with torch.inference_mode():
                answer = model(inputs, request)
            prediction = answer.columns[0]
            assert len(answer.columns) == 1 and prediction.result.status == "ok"
            assert prediction.column == entry["width"] - 1
            values = prediction.decoded.detach().cpu().tolist()
            positions = prediction.target_indices.detach().cpu().tolist()
            targets = request.targets.detach().cpu().tolist()
            assert len(values) == len(query)
            rows = []
            for value, position in zip(values, positions):
                local, column = targets[position]
                assert column == entry["width"] - 1 and math.isfinite(value)
                row_id = query[local - len(support)]
                rows.append(dict(row_id=row_id, target=float(data["values"][row_id][-1]),
                                 prediction=value))
            assert {r["row_id"] for r in rows} == set(query)
            rows.sort(key=lambda r: r["row_id"])
            atomic_json(output / f"{name}-predictions.json", rows)
            receipt["datasets"][name] = dict(metrics=metrics(rows),
                                             support_rows=len(support), query_rows=len(query),
                                             forward_rows=len(support) + len(query),
                                             forwards=1, seconds=time.monotonic() - start)
            atomic_json(output / "progress.json", receipt)
            print(json.dumps(dict(table=name, **receipt["datasets"][name])), flush=True)
            del answer, prediction, inputs, request, rows, data
            gc.collect()
            if args.device == "mps":
                torch.mps.empty_cache()
        assert state_hash(model) == before and sha256(args.checkpoint) == digest
        receipt.update(outcome="completed", model_state_unchanged=True,
                       checkpoint_unchanged=True,
                       total_predictions=sum(d["metrics"]["n"] for d in receipt["datasets"].values()))
        if not args.only:
            assert receipt["total_predictions"] == 7894
            receipt["macro_r2"] = math.fsum(
                d["metrics"]["r2"] for d in receipt["datasets"].values()) / 12
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
        atomic_json(output / "terminal.json", receipt)
        raise
    atomic_json(output / "terminal.json", receipt)
    print(json.dumps(dict(outcome=receipt["outcome"], macro_r2=receipt.get("macro_r2"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("host", "bank-root", "manifest", "checkpoint", "expected-sha", "device", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--only", nargs="*")
    run(parser.parse_args())
