"""Controlled Puma synthetic ablations using the 2026-09-27 paired data.

All arms use the same train/test row identities, split, supervised-row episodes,
H4/Unit3 parent, squared objective and fixed evaluation addresses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "puma-synthetic-probe-20260927"
RELEVANT = (0, 1, 6, 12, 13, 17)
ARMS = ("noise12", "noise20", "constant32", "dense32", "linear32")
SEED = 20260927


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, median, scale):
    ys = [r["target"] for r in rows]
    preds = [r["prediction"] for r in rows]
    ym, pm = statistics.fmean(ys), statistics.fmean(preds)
    vy = statistics.fmean((y - ym) ** 2 for y in ys)
    vp = statistics.fmean((p - pm) ** 2 for p in preds)
    cov = statistics.fmean((y - ym) * (p - pm) for y, p in zip(ys, preds))
    nloss = statistics.fmean(math.log1p(((y-p)/scale)**2) for y,p in zip(ys,preds))
    dloss = statistics.fmean(math.log1p(((y-median)/scale)**2) for y in ys)
    return dict(n=len(rows), r2=1-statistics.fmean((y-p)**2 for y,p in zip(ys,preds))/vy,
                slog=1-nloss/dloss, target_sd=math.sqrt(vy), prediction_sd=math.sqrt(vp),
                correlation=cov/math.sqrt(vy*vp) if vp else 0.0)


def addresses(split, episodes=8):
    train, test = split["train"], split["test"]
    for k in range(episodes):
        rng = random.Random(SEED+1000+k)
        support = rng.sample(train, 136)
        train_query = rng.sample(list(set(train)-set(support)), 68)
        test_query = rng.sample(test, 68)
        yield support, train_query, test_query


def prepare():
    full = load(BASE / "data" / "full32.json")
    parent = load(BASE / "inputs" / "parent-manifest.json")
    values = full["values"]
    train = full["splits"]["train"]
    noise = tuple(j for j in range(32) if j not in RELEVANT)
    def unit(row, j):
        x = row[j]
        return x/2.356 if j < 12 else (x/75 if j < 17 else (x-1.375)/1.125)
    def nonlinear(row):
        u = lambda j: unit(row, j)
        return (1.2*(u(12)>0)*math.sin(math.pi*u(0))
                + 0.8*(u(17)>0)*u(6) + 0.6*u(1)*u(13))
    g = [nonlinear(row) for row in values]
    gm, gs = statistics.fmean(g[i] for i in train), statistics.pstdev(g[i] for i in train)
    old_noise = [(row[-1]-0.030*(v-gm)/gs)/0.020 for row,v in zip(values,g)]
    linear = [sum(unit(row,j) for j in RELEVANT)/math.sqrt(len(RELEVANT)) for row in values]
    lm, ls = statistics.fmean(linear[i] for i in train), statistics.pstdev(linear[i] for i in train)
    arm_data = {}
    for arm in ARMS:
        if arm == "noise12":
            columns = RELEVANT + noise[:6]
        elif arm == "noise20":
            columns = RELEVANT + noise[:14]
        else:
            columns = tuple(range(32))
        data = {k:v for k,v in full.items() if k not in ("values", "features")}
        data["schema"] = "puma-synthetic-diagnosis-v1"
        data["arm"] = arm
        data["features"] = [full["features"][j] for j in columns] + [full["features"][-1]]
        new_rows = []
        for i,row in enumerate(values):
            features = [row[j] for j in columns]
            if arm == "constant32":
                features = [row[j] if j in RELEVANT else 0.0 for j in columns]
            elif arm == "dense32":
                features = [row[j] if j in RELEVANT else row[RELEVANT[k % 6]]
                            for k,j in enumerate(columns)]
            target = (0.030*(linear[i]-lm)/ls + 0.020*old_noise[i]
                      if arm == "linear32" else row[-1])
            new_rows.append(features+[target])
        data["values"] = new_rows
        data_path = ROOT / "data" / f"{arm}.json"
        if data_path.exists():
            raise FileExistsError(data_path)
        dump(data_path, data)
        arm_data[arm] = dict(data_sha256=digest(data_path), columns=len(columns),
                             target_train_sd=statistics.pstdev(r[-1] for r in new_rows[:len(train)]))
        manifest = dict(schema="tabu.curriculum.v55.v1", experiment_id=f"puma-diagnosis-{arm}-deepthought-20260927",
                        description="Controlled signal density and nuisance width diagnosis",
                        model=parent["model"], optimizer=parent["optimizer"],
                        seeds=dict(model=20260908, order=20260909, masks=20260910,
                                   codes=20260911, windows=20260912, evaluation=20260913),
                        tables=[dict(id=arm, path=f"../data/{arm}.json", sha256=digest(data_path),
                                     cohort="stress", kind="synthetic", role="train",
                                     window_rows=204, target_column=len(columns))],
                        probes=[], stages=[dict(name="synthetic_fit", question="Which input factor blocks fitting?",
                                                max_updates=1000, max_seconds=1800,
                                                sampling=[dict(cohort="stress", episodes=1)],
                                                recipe=dict(synthetic=dict(kind="supervised_row", fraction=1/3)),
                                                optimizer="adamw", evaluate_every=1000,
                                                checkpoint_every=100, probes=[],
                                                loss=dict(discrete_weight=1.0,
                                                          state_weights=[0.0,1.0,0.0,0.0]),
                                                objective=dict(kind="squared"))])
        dump(ROOT / "manifests" / f"{arm}.json", manifest)
    dump(ROOT / "prepared.json", dict(base_data_sha256=digest(BASE/"data"/"full32.json"),
                                      parent_checkpoint_sha256=digest(BASE/"inputs"/
                                          "e3da2393c28f6f05cab7e9d143710e69db9618015205b47f1fd3e9a041543863.pt"),
                                      arms=arm_data, row_count=len(values), train_count=len(train)))
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    for arm in ARMS:
        plan = load_v55_plan(ROOT/"manifests"/f"{arm}.json")
        print(json.dumps(dict(arm=arm, identity=plan.identity["sha256"], **arm_data[arm])), flush=True)


def fit(arm, updates=600):
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53 import runner
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    parent = BASE/"inputs"/"e3da2393c28f6f05cab7e9d143710e69db9618015205b47f1fd3e9a041543863.pt"
    plan = load_v55_plan(ROOT/"manifests"/f"{arm}.json")
    payload, parent_sha = load_checkpoint(parent)
    assert payload["model_config"] == plan.config.as_dict()
    result = runner.run(plan, ROOT/"runs"/arm/f"updates-{updates}",
                        device="cuda:0", initialize_from=parent,
                        max_updates_this_invocation=updates)
    assert result["outcome"] == "stopped" and result["durable_update"] == updates
    print(json.dumps(dict(arm=arm, parent_sha256=parent_sha, outcome=result["outcome"],
                          checkpoint_sha256=result["checkpoint_sha256"],
                          durable_update=result["durable_update"],
                          peak_allocated_bytes=result.get("peak_allocated_bytes"))), flush=True)


def resume_full32(additional=400):
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53 import runner
    ckpt = (BASE/"runs"/"full32"/"resume-150-to-600"/"checkpoints"/
            "faa9de14d271709039acd6aa772c9f837c81b84580ea8db0b86a537c02943d9e.pt")
    plan = load_v55_plan(BASE/"manifests"/"full32.json")
    result = runner.run(plan, BASE/"runs"/"full32"/"resume-600-to-1000",
                        device="cuda:0", resume=ckpt,
                        max_updates_this_invocation=additional)
    assert result["outcome"] == "stopped" and result["durable_update"] == 600+additional
    print(json.dumps(dict(arm="full32", outcome=result["outcome"],
                          checkpoint_sha256=result["checkpoint_sha256"],
                          durable_update=result["durable_update"])), flush=True)


def evaluate(arm, checkpoint, episodes=8, readout=False):
    import torch
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53.runner import configure_runtime
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest
    from tabu_lab.models.restoration_v53.readout import normalized_weights
    configure_runtime("cuda:0")
    source = BASE if arm in ("signal6", "full32") else ROOT
    plan = load_v55_plan(source/"manifests"/f"{arm}.json")
    payload, sha = load_checkpoint(checkpoint)
    assert payload["identity"] == plan.identity
    model = make_model(plan).to(device="cuda:0", dtype=execution_dtype("cuda:0"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    data = load(source/"data"/f"{arm}.json")
    schema = tuple(ColumnSchema(f["key"], "numeric") for f in data["features"])
    train_y = [data["values"][i][-1] for i in data["splits"]["train"]]
    med = statistics.median(train_y)
    mad = statistics.median(abs(v-med) for v in train_y)
    records = {"train_query": [], "test_query": []}
    bandwidths = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    local_records = {b:{"train_query": [], "test_query": []} for b in bandwidths}
    bandwidth_ess = {b:{"train_query": [], "test_query": []} for b in bandwidths}
    geometry = {"train_query": [], "test_query": []}
    for support,tq,vq in addresses(data["splits"], episodes):
        for label,query in (("train_query",tq),("test_query",vq)):
            ids = support+query
            values = [list(data["values"][i]) for i in ids]
            for row in values[136:]: row[-1] = 0.0
            tensor = torch.tensor(values, dtype=execution_dtype("cuda:0"), device="cuda:0")
            columns = tuple(tensor[:,j] for j in range(tensor.shape[1]))
            visible = torch.ones_like(tensor, dtype=torch.bool)
            visible[136:,-1] = False
            target = ~visible
            inp = RestorationInput(schema, columns, visible, target, SEED)
            req = RestorationRequest(target.nonzero())
            with torch.inference_mode():
                output = model(inp, req)
                pred = output.columns[0].decoded.detach().cpu().tolist()
                if readout:
                    distances = torch.cdist(output.units[136:],output.units[:136])
                    for bandwidth in bandwidths:
                        weights = normalized_weights(output.units[136:],output.units[:136],bandwidth)
                        local_encoded = weights @ output.facts[-1].answers.encoded
                        local = output.facts[-1].answers.decode(local_encoded).detach().cpu().tolist()
                        bandwidth_ess[bandwidth][label].append((1.0/weights.square().sum(-1)).mean().item())
                        local_records[bandwidth][label].extend(
                            dict(row_id=i,target=data["values"][i][-1],prediction=float(p))
                            for i,p in zip(query,local))
                        if bandwidth == 1.0:
                            ess = bandwidth_ess[bandwidth][label][-1]
                            max_weight = weights.max(-1).values.mean().item()
                    geometry[label].append(dict(ess=ess, max_weight=max_weight,
                                                distance_mean=distances.mean().item(),
                                                distance_within_row_sd=distances.std(-1).mean().item(),
                                                distance_min=distances.min(-1).values.mean().item(),
                                                support_unit_spread=output.units[:136].std(0).norm().item()))
            assert output.columns[0].result.status == "ok" and len(pred)==68
            records[label].extend(dict(row_id=i,target=data["values"][i][-1],prediction=float(p))
                                  for i,p in zip(query,pred))
    result = dict(arm=arm, checkpoint_sha256=sha, checkpoint_update=payload["state"]["update"],
                  episodes=episodes, context_rows=136, query_rows_per_episode=68,
                  train_query=metrics(records["train_query"],med,max(mad,1e-12)),
                  test_query=metrics(records["test_query"],med,max(mad,1e-12)))
    if readout:
        result["local_only"] = {str(b):{k:metrics(v,med,max(mad,1e-12)) for k,v in local_records[b].items()}
                                for b in bandwidths}
        result["bandwidth_ess"] = {str(b):{k:statistics.fmean(v) for k,v in bandwidth_ess[b].items()}
                                   for b in bandwidths}
        result["geometry"] = {k:{field:statistics.fmean(vv[field] for vv in v)
                                  for field in ("ess","max_weight","distance_mean",
                                                "distance_within_row_sd","distance_min",
                                                "support_unit_spread")}
                              for k,v in geometry.items()}
    dump(ROOT/"evaluation"/f"{arm}-u{result['checkpoint_update']}.json",result)
    print(json.dumps(result),flush=True)


def baseline():
    import numpy as np
    import xgboost as xgb
    result = {}
    for arm in ("signal6","full32")+ARMS:
        source = BASE if arm in ("signal6","full32") else ROOT
        data = load(source/"data"/f"{arm}.json")
        X = np.asarray([r[:-1] for r in data["values"]])
        y = np.asarray([r[-1] for r in data["values"]])
        train,test = data["splits"]["train"],data["splits"]["test"]
        med = statistics.median(y[i] for i in train)
        mad = statistics.median(abs(y[i]-med) for i in train)
        result[arm] = {}
        for label,depth,trees in (("support136_d3t100",3,100),("support136_d6t600",6,600)):
            test_records = []
            for support,tq,vq in addresses(data["splits"]):
                booster = xgb.train(dict(objective="reg:squarederror",max_depth=depth,eta=.05,
                                         nthread=4,seed=SEED,tree_method="hist"),
                                    xgb.DMatrix(X[support],label=y[support]),num_boost_round=trees)
                pred = booster.predict(xgb.DMatrix(X[vq]))
                test_records.extend(dict(target=float(y[i]),prediction=float(p)) for i,p in zip(vq,pred))
            result[arm][label] = metrics(test_records,med,max(mad,1e-12))
        booster = xgb.train(dict(objective="reg:squarederror",max_depth=6,eta=.05,
                                 nthread=4,seed=SEED,tree_method="hist"),
                            xgb.DMatrix(X[train],label=y[train]),num_boost_round=600)
        pred = booster.predict(xgb.DMatrix(X[test]))
        result[arm]["full6553_d6t600"] = metrics(
            [dict(target=float(y[i]),prediction=float(p)) for i,p in zip(test,pred)],
            med,max(mad,1e-12))
        print(json.dumps(dict(arm=arm,**result[arm])),flush=True)
    dump(ROOT/"xgboost.json",result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","fit","resume-full32","evaluate","baseline"))
    p.add_argument("--arm",choices=("signal6","full32")+ARMS)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--updates",type=int,default=600)
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--readout",action="store_true")
    a = p.parse_args()
    if a.mode == "prepare": prepare()
    elif a.mode == "fit": fit(a.arm,a.updates)
    elif a.mode == "resume-full32": resume_full32(a.updates)
    elif a.mode == "evaluate": evaluate(a.arm,a.checkpoint,a.episodes,a.readout)
    else: baseline()
