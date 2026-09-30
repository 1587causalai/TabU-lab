"""Inference-only oracle column-mask probe on the already fitted checkpoints."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from diagnosis import ROOT, BASE, RELEVANT, SEED, addresses, dump, load, metrics
from tabu_lab.curriculum_v53.artifacts import load_checkpoint
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53.readout import normalized_weights
from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest


def probe(checkpoint, weights_name, keep, episodes):
    configure_runtime("cuda:0")
    plan = load_v55_plan(BASE/"manifests"/"full32.json")
    payload, sha = load_checkpoint(checkpoint)
    assert payload["model_config"] == plan.config.as_dict()
    model = make_model(plan).to(device="cuda:0",dtype=execution_dtype("cuda:0"))
    model.load_state_dict(payload["model"],strict=True)
    model.eval().requires_grad_(False)
    data = load(BASE/"data"/"full32.json")
    schema = tuple(ColumnSchema(f["key"],"numeric") for f in data["features"])
    keep_columns = set(range(32)) if keep == "all" else (
        set(RELEVANT) if keep == "relevant6" else set(range(6)))
    train_y = [data["values"][i][-1] for i in data["splits"]["train"]]
    med = statistics.median(train_y)
    mad = statistics.median(abs(v-med) for v in train_y)
    records = {"train_query":[],"test_query":[]}
    geometry = {"train_query":[],"test_query":[]}
    for support,tq,vq in addresses(data["splits"],episodes):
        for label,query in (("train_query",tq),("test_query",vq)):
            ids = support+query
            values = [list(data["values"][i]) for i in ids]
            visible = torch.ones((len(ids),33),dtype=torch.bool,device="cuda:0")
            target = torch.zeros_like(visible)
            target[136:,-1] = True
            visible[136:,-1] = False
            for j in range(32):
                if j not in keep_columns:
                    visible[:,j] = False
                    for row in values: row[j] = 0.0
            for row in values[136:]: row[-1] = 0.0
            tensor = torch.tensor(values,dtype=execution_dtype("cuda:0"),device="cuda:0")
            inp = RestorationInput(schema,tuple(tensor[:,j] for j in range(33)),visible,target,SEED)
            req = RestorationRequest(target.nonzero())
            with torch.inference_mode():
                output = model(inp,req)
                pred = output.columns[0].decoded.detach().cpu().tolist()
                w = normalized_weights(output.units[136:],output.units[:136],1.0)
                geometry[label].append((1.0/w.square().sum(-1)).mean().item())
            assert output.columns[0].result.status == "ok" and len(pred)==68
            records[label].extend(dict(target=data["values"][i][-1],prediction=float(p))
                                  for i,p in zip(query,pred))
    result = dict(weights=weights_name,keep=keep,checkpoint_sha256=sha,
                  checkpoint_update=payload["state"]["update"],episodes=episodes,
                  train_query=metrics(records["train_query"],med,max(mad,1e-12)),
                  test_query=metrics(records["test_query"],med,max(mad,1e-12)),
                  ess={k:statistics.fmean(v) for k,v in geometry.items()})
    dump(ROOT/"mask"/f"{weights_name}-{keep}.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--weights-name",required=True)
    p.add_argument("--keep",choices=("all","relevant6","first6"),required=True)
    p.add_argument("--episodes",type=int,default=8)
    a = p.parse_args()
    probe(a.checkpoint,a.weights_name,a.keep,a.episodes)
