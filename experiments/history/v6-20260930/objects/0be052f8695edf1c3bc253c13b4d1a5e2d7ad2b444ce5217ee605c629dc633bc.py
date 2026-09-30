"""Frozen-weight bandwidth intervention; no parameter updates."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import replace
from pathlib import Path

import torch

from diagnosis import ROOT, BASE, SEED, addresses, dump, load, metrics
from tabu_lab.curriculum_v53.artifacts import load_checkpoint
from tabu_lab.curriculum_v53.protocol import load_v55_plan
from tabu_lab.curriculum_v53.runner import configure_runtime
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest, V55Model


def probe(arm, checkpoint, bandwidths, episodes):
    configure_runtime("cuda:0")
    source = BASE if arm in ("signal6","full32") else ROOT
    plan = load_v55_plan(source/"manifests"/f"{arm}.json")
    payload, sha = load_checkpoint(checkpoint)
    assert payload["model_config"] == plan.config.as_dict()
    data = load(source/"data"/f"{arm}.json")
    schema = tuple(ColumnSchema(f["key"],"numeric") for f in data["features"])
    train_y = [data["values"][i][-1] for i in data["splits"]["train"]]
    med = statistics.median(train_y)
    mad = statistics.median(abs(v-med) for v in train_y)
    result = dict(arm=arm,checkpoint_sha256=sha,checkpoint_update=payload["state"]["update"],
                  episodes=episodes,by_bandwidth={})
    for bandwidth in bandwidths:
        model = V55Model(replace(plan.config,bandwidth=bandwidth)).to(
            device="cuda:0",dtype=execution_dtype("cuda:0"))
        model.load_state_dict(payload["model"],strict=True)
        model.eval().requires_grad_(False)
        records = {"train_query":[],"test_query":[]}
        for support,tq,vq in addresses(data["splits"],episodes):
            for label,query in (("train_query",tq),("test_query",vq)):
                ids = support+query
                values = [list(data["values"][i]) for i in ids]
                for row in values[136:]: row[-1] = 0.0
                tensor = torch.tensor(values,dtype=execution_dtype("cuda:0"),device="cuda:0")
                visible = torch.ones_like(tensor,dtype=torch.bool)
                visible[136:,-1] = False
                target = ~visible
                inp = RestorationInput(schema,tuple(tensor[:,j] for j in range(tensor.shape[1])),
                                       visible,target,SEED)
                req = RestorationRequest(target.nonzero())
                with torch.inference_mode():
                    output = model(inp,req)
                    pred = output.columns[0].decoded.detach().cpu().tolist()
                records[label].extend(dict(target=data["values"][i][-1],prediction=float(p))
                                      for i,p in zip(query,pred))
        result["by_bandwidth"][str(bandwidth)] = {
            k:metrics(v,med,max(mad,1e-12)) for k,v in records.items()}
        print(json.dumps(dict(arm=arm,bandwidth=bandwidth,
                              **result["by_bandwidth"][str(bandwidth)])),flush=True)
    dump(ROOT/"bandwidth"/f"{arm}-u{result['checkpoint_update']}.json",result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm",required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--bandwidths",nargs="+",type=float,default=[0.5,1.0,2.0])
    p.add_argument("--episodes",type=int,default=8)
    a = p.parse_args()
    probe(a.arm,a.checkpoint,a.bandwidths,a.episodes)
