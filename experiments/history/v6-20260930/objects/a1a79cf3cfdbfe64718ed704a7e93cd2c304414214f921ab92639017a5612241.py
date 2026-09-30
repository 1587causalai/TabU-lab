"""One real Puma episode: broadcast/old-loss CUDA forward and backward."""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_broadcast_oldloss import episode_seeds, make_model
from tabu_lab.curriculum_v53.artifacts import finite_state, load_checkpoint
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v6 import score_training_episode


def main(args):
    runtime=configure_runtime(args.device)
    plan=load_v55_plan(args.manifest)
    payload,sha=load_checkpoint(args.parent)
    assert sha==args.parent_sha
    model=make_model(args,plan,payload)
    table=next(t for t in plan.tables if t.name=="pumadyn32nh")
    recipe=plan.spec["stages"][0]["recipe"][table.kind]
    inputs,request,truth,info=build_episode(table,recipe,0,episode_seeds(plan),args.device,
         epsilon=plan.config.epsilon,codec_version=plan.config.codec_version)
    start=time.monotonic()
    loss_config=V53LossConfig(**plan.spec["stages"][0]["loss"])
    score=score_training_episode(model,inputs,truth,request=request,loss_config=loss_config)
    score.loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert finite_state(gradients) and len(gradients)>0
    query_rows=int(inputs.query[:,table.target_column].sum())
    assert score.query_rows==query_rows
    assert score.scored_cells==query_rows
    assert len(score.output.columns)==1
    assert score.output.columns[0].column==table.target_column
    assert all(c.result.status=="ok" for c in score.output.columns)
    assert bool((inputs.values[table.target_column][inputs.query[:,table.target_column]]==0).all())
    result=dict(outcome="passed",parent_sha256=sha,runtime=runtime,
                table=table.name,forward_rows=len(inputs.visible),support_rows=int(inputs.visible[:,table.target_column].sum()),
                query_rows=query_rows,supervised_cells=score.scored_cells,columns=table.width,
                loss=float(score.loss.detach()),gradient_tensors=len(gradients),
                seconds=time.monotonic()-start,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("manifest","parent","parent-sha","device"):
        p.add_argument("--"+name,required=True)
    main(p.parse_args())
