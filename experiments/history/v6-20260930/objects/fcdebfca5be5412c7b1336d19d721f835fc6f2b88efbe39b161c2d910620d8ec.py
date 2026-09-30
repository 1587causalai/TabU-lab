"""Paired Puma-like synthetic fit probe for TabU V5.5 on deepthought.

Both arms share row identities, target values, split, model start and loss.
Only the 26 independent nuisance columns differ.
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
SEED = 20260927
N = 8192
TRAIN = 6553
RELEVANT = (0, 1, 6, 12, 13, 17)
ARMS = ("signal6", "full32")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows, median, scale):
    ys = [r["target"] for r in rows]
    preds = [r["prediction"] for r in rows]
    ym, pm = statistics.fmean(ys), statistics.fmean(preds)
    vy = statistics.fmean((y - ym) ** 2 for y in ys)
    vp = statistics.fmean((p - pm) ** 2 for p in preds)
    cov = statistics.fmean((y - ym) * (p - pm) for y, p in zip(ys, preds))
    nloss = statistics.fmean(math.log1p(((y - p) / scale) ** 2) for y, p in zip(ys, preds))
    dloss = statistics.fmean(math.log1p(((y - median) / scale) ** 2) for y in ys)
    return dict(n=len(rows), r2=1 - statistics.fmean((y - p) ** 2 for y, p in zip(ys, preds)) / vy,
                slog=1 - nloss / dloss, target_sd=math.sqrt(vy), prediction_sd=math.sqrt(vp),
                correlation=cov / math.sqrt(vy * vp) if vp else 0.0,
                target_mean=ym, prediction_mean=pm)


def prepare(parent_manifest):
    if ROOT.exists() and any(ROOT.glob("data/*.json")):
        raise FileExistsError("experiment data already exist; preparation is immutable")
    rng = random.Random(SEED)
    unit = [[rng.uniform(-1, 1) for _ in range(32)] for _ in range(N)]
    noise = [rng.gauss(0, 1) for _ in range(N)]
    def teacher(u):
        return (1.2 * (u[12] > 0) * math.sin(math.pi * u[0])
                + 0.8 * (u[17] > 0) * u[6] + 0.6 * u[1] * u[13])
    g = [teacher(u) for u in unit]
    gmean, gsd = statistics.fmean(g[:TRAIN]), statistics.pstdev(g[:TRAIN])
    target = [0.030 * (v - gmean) / gsd + 0.020 * e for v, e in zip(g, noise)]
    names = ([f"theta{i+1}" for i in range(6)]
             + [f"thetad{i+1}" for i in range(6)]
             + [f"tau{i+1}" for i in range(5)]
             + [f"mass_or_damping{i+1}" for i in range(15)])
    def raw(j, u):
        return 2.356 * u if j < 12 else (75 * u if j < 17 else 1.375 + 1.125 * u)
    split = dict(train=list(range(TRAIN)), test=list(range(TRAIN, N)))
    parent = json.loads(Path(parent_manifest).read_text())
    for arm in ARMS:
        columns = RELEVANT if arm == "signal6" else tuple(range(32))
        data = dict(schema="puma-synthetic-probe-v1", generator_seed=SEED,
                    rows=N, target_formula="0.030*z(g_train)+0.020*normal(0,1)",
                    values=[[*(raw(j, unit[i][j]) for j in columns), target[i]] for i in range(N)],
                    features=[dict(key=names[j], kind="numeric", domain=[]) for j in columns]
                    + [dict(key="acceleration_target", kind="numeric", domain=[])],
                    splits=split)
        data_path = ROOT / "data" / f"{arm}.json"
        dump(data_path, data)
        model = parent["model"]
        assert model["numeric_scaling"] == "zscore" and model["backbone"]["heads"] == 4
        manifest = dict(schema="tabu.curriculum.v55.v1",
                        experiment_id=f"puma-synthetic-{arm}-deepthought-20260927",
                        description="Paired 6-signal versus 32-column nuisance stress probe",
                        model=model, optimizer=parent["optimizer"],
                        seeds=dict(model=20260908, order=20260909, masks=20260910,
                                   codes=20260911, windows=20260912, evaluation=20260913),
                        tables=[dict(id=arm, path=f"../data/{arm}.json", sha256=digest(data_path),
                                     cohort="stress", kind="synthetic", role="train",
                                     window_rows=204, target_column=len(columns))],
                        probes=[],
                        stages=[dict(name="synthetic_fit", question="Can the same model fit this teacher?",
                                     max_updates=1000, max_seconds=1800,
                                     sampling=[dict(cohort="stress", episodes=1)],
                                     recipe=dict(synthetic=dict(kind="supervised_row", fraction=1/3)),
                                     optimizer="adamw", evaluate_every=1000,
                                     checkpoint_every=50, probes=[],
                                     loss=dict(discrete_weight=1.0,
                                               state_weights=[0.0, 1.0, 0.0, 0.0]),
                                     objective=dict(kind="squared"))])
        dump(ROOT / "manifests" / f"{arm}.json", manifest)
    dump(ROOT / "prepared.json", dict(seed=SEED, rows=N, train_n=TRAIN,
                                       test_n=N-TRAIN, relevant_columns=list(RELEVANT),
                                       parent_manifest_sha256=digest(Path(parent_manifest)),
                                       data_sha256={a: digest(ROOT / "data" / f"{a}.json") for a in ARMS},
                                       target_train_sd=statistics.pstdev(target[:TRAIN]),
                                       signal_train_sd=0.030,
                                       noise_sd=0.020))
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    for arm in ARMS:
        plan = load_v55_plan(ROOT / "manifests" / f"{arm}.json")
        print(json.dumps(dict(arm=arm, identity=plan.identity["sha256"],
                              table_rows=plan.tables[0].train_rows)), flush=True)


def fit(arm, parent_checkpoint, updates):
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53 import runner
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    plan = load_v55_plan(ROOT / "manifests" / f"{arm}.json")
    payload, parent_sha = load_checkpoint(parent_checkpoint)
    assert payload["model_config"] == plan.config.as_dict()
    assert plan.spec["stages"][0]["objective"] == {"kind": "squared"}
    run_dir = ROOT / "runs" / arm / f"updates-{updates}"
    result = runner.run(plan, run_dir, device="cuda:0", initialize_from=parent_checkpoint,
                        max_updates_this_invocation=updates)
    print(json.dumps(dict(arm=arm, parent_sha256=parent_sha, outcome=result["outcome"],
                          durable_update=result["durable_update"],
                          checkpoint_sha256=result["checkpoint_sha256"],
                          peak_allocated_bytes=result.get("peak_allocated_bytes"))), flush=True)


def resume(arm, checkpoint, additional_updates):
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53 import runner
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    plan = load_v55_plan(ROOT / "manifests" / f"{arm}.json")
    payload, previous_sha = load_checkpoint(checkpoint)
    assert payload["identity"] == plan.identity
    start = payload["state"]["update"]
    result = runner.run(plan, ROOT / "runs" / arm / f"resume-{start}-to-{start+additional_updates}",
                        device="cuda:0", resume=checkpoint,
                        max_updates_this_invocation=additional_updates)
    assert result["outcome"] == "stopped" and result["durable_update"] == start+additional_updates
    print(json.dumps(dict(arm=arm, resume_from_sha256=previous_sha,
                          outcome=result["outcome"], durable_update=result["durable_update"],
                          checkpoint_sha256=result["checkpoint_sha256"],
                          peak_allocated_bytes=result.get("peak_allocated_bytes"))), flush=True)


def evaluate(arm, checkpoint, episodes):
    import torch
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53.runner import configure_runtime
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v55 import ColumnSchema, RestorationInput, RestorationRequest
    configure_runtime("cuda:0")
    plan = load_v55_plan(ROOT / "manifests" / f"{arm}.json")
    payload, sha = load_checkpoint(checkpoint)
    assert payload["identity"] == plan.identity
    model = make_model(plan).to(device="cuda:0", dtype=execution_dtype("cuda:0"))
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    data = json.loads((ROOT / "data" / f"{arm}.json").read_text())
    schema = tuple(ColumnSchema(f["key"], "numeric") for f in data["features"])
    train, test = data["splits"]["train"], data["splits"]["test"]
    train_y = [data["values"][i][-1] for i in train]
    med = statistics.median(train_y)
    mad = statistics.median(abs(v-med) for v in train_y)
    records = {"train_query": [], "test_query": []}
    for k in range(episodes):
        rng = random.Random(SEED+1000+k)
        support = rng.sample(train, 136)
        available = list(set(train)-set(support))
        tq = rng.sample(available, 68)
        vq = rng.sample(test, 68)
        for label, query in (("train_query", tq), ("test_query", vq)):
            ids = support+query
            values = [list(data["values"][i]) for i in ids]
            for row in values[136:]: row[-1] = 0.0
            tensor = torch.tensor(values, dtype=execution_dtype("cuda:0"), device="cuda:0")
            columns = tuple(tensor[:, j] for j in range(tensor.shape[1]))
            visible = torch.ones_like(tensor, dtype=torch.bool)
            visible[136:, -1] = False
            target = ~visible
            inp = RestorationInput(schema, columns, visible, target, SEED)
            req = RestorationRequest(target.nonzero())
            with torch.inference_mode():
                answer = model(inp, req)
            pred = answer.columns[0].decoded.detach().cpu().tolist()
            assert answer.columns[0].result.status == "ok" and len(pred) == 68
            records[label].extend(dict(row_id=i, target=data["values"][i][-1],
                                       prediction=float(p)) for i,p in zip(query,pred))
            del inp, req, answer, tensor
    result = dict(arm=arm, checkpoint_sha256=sha, checkpoint_update=payload["state"]["update"],
                  episodes=episodes, context_rows=136, query_rows_per_episode=68,
                  train_query=metrics(records["train_query"], med, max(mad, 1e-12)),
                  test_query=metrics(records["test_query"], med, max(mad, 1e-12)))
    dump(ROOT / "evaluation" / f"{arm}.json", result)
    print(json.dumps(result), flush=True)


def xgboost_baseline():
    import xgboost as xgb
    import numpy as np
    result = {}
    for arm in ARMS:
        data = json.loads((ROOT / "data" / f"{arm}.json").read_text())
        tr, te = data["splits"]["train"], data["splits"]["test"]
        X = [r[:-1] for r in data["values"]]
        y = [r[-1] for r in data["values"]]
        train = xgb.DMatrix(np.asarray([X[i] for i in tr]), label=np.asarray([y[i] for i in tr]))
        test = xgb.DMatrix(np.asarray([X[i] for i in te]))
        model = xgb.train(dict(objective="reg:squarederror", max_depth=6, eta=0.05,
                               nthread=4, seed=SEED, tree_method="hist"), train,
                          num_boost_round=600)
        pred = model.predict(test)
        med = statistics.median(y[i] for i in tr)
        mad = statistics.median(abs(y[i]-med) for i in tr)
        rows = [dict(target=y[i], prediction=float(p)) for i,p in zip(te,pred)]
        result[arm] = metrics(rows, med, max(mad, 1e-12))
    dump(ROOT / "xgboost.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "fit", "resume", "evaluate", "xgboost"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--parent-manifest")
    parser.add_argument("--parent-checkpoint")
    parser.add_argument("--checkpoint")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.parent_manifest)
    elif args.mode == "fit":
        fit(args.arm, args.parent_checkpoint, args.updates)
    elif args.mode == "resume":
        resume(args.arm, args.checkpoint, args.updates)
    elif args.mode == "evaluate":
        evaluate(args.arm, args.checkpoint, args.episodes)
    else:
        xgboost_baseline()
