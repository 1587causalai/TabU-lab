"""Broadcast/old-loss target readout on the frozen full OpenML12 bank."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_evaluate import make_input, metrics, state_hash
from tabu_lab.curriculum_v53.artifacts import atomic_json, sha256
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v6 import V6Model


def load_v6_checkpoint(path, expected_sha):
    if sha256(path) != expected_sha:
        raise ValueError("V6 checkpoint SHA mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["schema"] != "tabu.v6.weights-only-checkpoint.v1":
        raise ValueError("not a V6 training checkpoint")
    return payload


def score_log(rows, train_targets):
    median = statistics.median(train_targets)
    mad = statistics.median(abs(y - median) for y in train_targets)
    if mad <= 1e-12:
        raise ValueError("S_log requires a nonconstant training target")
    numerator = statistics.fmean(
        math.log1p(((r["target"] - r["prediction"]) / mad) ** 2) for r in rows
    )
    denominator = statistics.fmean(
        math.log1p(((r["target"] - median) / mad) ** 2) for r in rows
    )
    return 1 - numerator / denominator


def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(schema="tabu.v6.broadcast-explicit-supervision.full-context.v1", outcome="running",
                  evaluation="all train support + all test Query in one forward per table",
                  bank_sha256=sha256(Path(args.bank_root)/"bank.json"),
                  checkpoint_sha256=args.expected_sha, optimizer_updates=0)
    atomic_json(out/"started.json", report)
    try:
        runtime = configure_runtime(args.device)
        if args.device=='mps': torch.mps.set_per_process_memory_fraction(args.memory_fraction)
        else: torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        plan = load_v55_plan(args.manifest)
        payload = load_v6_checkpoint(args.checkpoint, args.expected_sha)
        if payload["model_config"] != plan.config.as_dict():
            raise ValueError("V6 checkpoint model config drift")
        if payload["identity"].get("supervision") not in ("joint_all", "target_only"):
            raise ValueError("checkpoint is not the broadcast/joint-all trial")
        model = V6Model(plan.config, supervision="target_only").to(
            args.device, dtype=execution_dtype(args.device)
        )
        model.load_state_dict(payload["model"], strict=True)
        model.eval().requires_grad_(False)
        before = state_hash(model)
        report.update(runtime=runtime, identity=payload["identity"], evaluation_supervision="target_only",
                      checkpoint_update=payload["state"]["update"],
                      model_state_sha256=before)
        atomic_json(out/"resolved.json", report)
        bank_root=Path(args.bank_root)
        bank=json.loads((bank_root/"bank.json").read_text())
        selected=[entry for entry in bank["tables"] if not args.only or entry["name"] in args.only]
        if len(selected) != (len(args.only) if args.only else 12):
            raise ValueError("requested table missing from frozen bank")
        for entry in selected:
            tick=time.monotonic()
            name=entry["name"]
            path=bank_root/entry["path"]
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"dataset SHA mismatch: {name}")
            data=json.loads(path.read_text())
            support=data["splits"]["train"]
            query=data["splits"]["test"]
            if set(support)&set(query):
                raise ValueError("support/test overlap in bank")
            trace=dict(context_row_ids=support, query_row_ids=query,
                       codebook_seed=entry["traces"][0]["codebook_seed"])
            inputs, request=make_input(data,trace,name,args.device,execution_dtype(args.device))
            with torch.inference_mode():
                answer=model(inputs,request)
            if len(answer.columns)!=1 or answer.columns[0].result.status!="ok":
                raise ValueError(f"target readout failed: {name}")
            column=answer.columns[0]
            if column.column!=entry["width"]-1 or column.decoded is None:
                raise ValueError("V6 target column/decoder mismatch")
            predictions=column.decoded.detach().cpu().tolist()
            if len(predictions)!=len(query):
                raise ValueError("Query prediction count mismatch")
            positions=column.target_indices.detach().cpu().tolist()
            addresses=request.targets.detach().cpu().tolist()
            rows=[]
            seen=set()
            for prediction, position in zip(predictions,positions,strict=True):
                row, col=addresses[position]
                if col!=entry["width"]-1 or row<len(support) or row>=len(support)+len(query):
                    raise ValueError("target readout address outside frozen Query")
                row_id=query[row-len(support)]
                if row_id in seen:
                    raise ValueError("duplicate target readout address")
                seen.add(row_id)
                rows.append(dict(row_id=row_id,target=float(data["values"][row_id][-1]),
                                 prediction=float(prediction)))
            if seen!=set(query):
                raise ValueError("target readout omitted frozen Query rows")
            if not all(math.isfinite(r["prediction"]) for r in rows):
                raise FloatingPointError("nonfinite full-context prediction")
            rows.sort(key=lambda r:r["row_id"])
            raw=[float(data["values"][i][-1]) for i in support]
            result=metrics(rows)
            result["slog"]=score_log(rows,raw)
            report.setdefault("tables",{})[name]=dict(**result,support_rows=len(support),
                        query_rows=len(query),forward_rows=len(support)+len(query),
                        seconds=time.monotonic()-tick)
            atomic_json(out/f"{name}-predictions.json",rows)
            atomic_json(out/"progress.json",report)
            print(json.dumps(dict(table=name,r2=result["r2"],slog=result["slog"],
                                  seconds=report["tables"][name]["seconds"])),flush=True)
            del answer,inputs,request,column
            torch.mps.empty_cache() if args.device == "mps" else torch.cuda.empty_cache()
        if state_hash(model)!=before or sha256(args.checkpoint)!=args.expected_sha:
            raise ValueError("V6 evaluation mutated model/checkpoint")
        report.update(outcome="completed",total_predictions=sum(t["n"] for t in report["tables"].values()))
        if not args.only:
            if report["total_predictions"]!=7894:
                raise ValueError("OpenML12 full bank prediction count differs")
            report["macro"]={metric:statistics.fmean(t[metric] for t in report["tables"].values())
                             for metric in ("r2","slog")}
    except Exception as error:
        report.update(outcome="failed",error_type=type(error).__name__,error=str(error))
        atomic_json(out/"terminal.json",report)
        raise
    atomic_json(out/"terminal.json",report)
    print(json.dumps(dict(outcome=report["outcome"],total_predictions=report["total_predictions"],
                          macro=report.get("macro"))),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("manifest","checkpoint","expected-sha","device","bank-root","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--only",nargs="*")
    parser.add_argument("--memory-fraction",type=float,default=0.5)
    main(parser.parse_args())
