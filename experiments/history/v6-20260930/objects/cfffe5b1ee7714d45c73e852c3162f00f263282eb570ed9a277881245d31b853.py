"""Read-only layer-by-layer Unit geometry for the Puma synthetic probes."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from diagnosis import ROOT, BASE, SEED, addresses, dump, load, metrics
from tabu_lab.curriculum_v53.artifacts import load_checkpoint
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53.readout import normalized_weights
from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest


def probe(arm, checkpoint, label, episodes):
    configure_runtime("cuda:0")
    source = BASE if arm in ("signal6", "full32") else ROOT
    plan = load_v55_plan(source/"manifests"/f"{arm}.json")
    payload, sha = load_checkpoint(checkpoint)
    assert payload["model_config"] == plan.config.as_dict()
    model = make_model(plan).to(device="cuda:0", dtype=execution_dtype("cuda:0"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    data = load(source/"data"/f"{arm}.json")
    schema = tuple(ColumnSchema(f["key"],"numeric") for f in data["features"])
    train_y = [data["values"][i][-1] for i in data["splits"]["train"]]
    med = statistics.median(train_y)
    mad = statistics.median(abs(v-med) for v in train_y)
    stages = {}
    for support,tq,vq in addresses(data["splits"],episodes):
        ids = support+vq
        values = [list(data["values"][i]) for i in ids]
        for row in values[136:]: row[-1] = 0.0
        tensor = torch.tensor(values,dtype=execution_dtype("cuda:0"),device="cuda:0")
        columns = tuple(tensor[:,j] for j in range(tensor.shape[1]))
        visible = torch.ones_like(tensor,dtype=torch.bool)
        visible[136:,-1] = False
        target = ~visible
        inp = RestorationInput(schema,columns,visible,target,SEED)
        req = RestorationRequest(target.nonzero())
        with torch.inference_mode():
            prepared = model.prepare(inp,req)
            h = model.encoder.forward_prepared(prepared.inputs,prepared.features)
            n,m = visible.shape
            sources = torch.zeros(n+1,m+1,dtype=torch.bool,device=h.device)
            sources[:n,:m] = visible
            nulls = torch.zeros_like(sources)
            nulls[:n,:m] = ~(visible | target)
            nulls[n,m] = True
            h = h.masked_fill(nulls[...,None],0)
            def record(name,units):
                weights = normalized_weights(units[136:],units[:136],1.0)
                local_encoded = weights @ prepared.facts[-1].answers.encoded
                local = prepared.facts[-1].answers.decode(local_encoded).detach().cpu().tolist()
                distances = torch.cdist(units[136:],units[:136])
                x = stages.setdefault(name,dict(spread=[],distance=[],ess=[],records=[]))
                x["spread"].append(units[:136].std(0).norm().item())
                x["distance"].append(distances.mean().item())
                x["ess"].append((1.0/weights.square().sum(-1)).mean().item())
                x["records"].extend(dict(target=data["values"][i][-1],prediction=float(p))
                                    for i,p in zip(vq,local))
            record("encoder",h[:n,m])
            for j,layer in enumerate(model.backbone.layers,1):
                h = layer(h,sources,nulls)
                record(f"axial{j}",h[:n,m])
            units = h[:n,m]
            eligible = visible.any(-1)
            for j,block in enumerate(model.unit_blocks,1):
                units = block(units,units,eligible)
                record(f"unit{j}",units)
    result = dict(arm=arm,label=label,checkpoint_sha256=sha,
                  checkpoint_update=payload["state"]["update"],episodes=episodes,
                  stages={name:dict(spread=statistics.fmean(x["spread"]),
                                    distance=statistics.fmean(x["distance"]),
                                    ess=statistics.fmean(x["ess"]),
                                    local_r2=metrics(x["records"],med,max(mad,1e-12))["r2"])
                          for name,x in stages.items()})
    dump(ROOT/"stages"/f"{arm}-{label}.json",result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm",required=True)
    p.add_argument("--checkpoint",required=True,type=Path)
    p.add_argument("--label",required=True)
    p.add_argument("--episodes",type=int,default=8)
    a = p.parse_args()
    probe(a.arm,a.checkpoint,a.label,a.episodes)
