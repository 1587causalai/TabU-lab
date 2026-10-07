"""V7 old120 pilot: row-masked training, fixed train-fit and held-out-row evaluation.

Training episodes use train rows only: one table drawn uniformly per update,
25% of its train rows become target Query cells, the rest are supports. Test
rows never enter a training episode.

Evaluation banks are fixed before training:
  train_fit  - per table, ``--fit-masks`` random 25% train-row masks (V5.5 probe style)
  heldout    - per table, all train rows are supports and the test rows are Query
Baselines on the same banks: donor, support mean/mode, support-only XGBoost,
and the untrained V7 (update 0).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    checkpoint_state,
    evaluate_task,
    load_typed_table,
    make_optimizer,
    manifest_digest,
    save_checkpoint,
    table_task,
    train_step,
)

XGBOOST = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=8,
    tree_method="hist",
)


def load_tables(manifest_path: Path, data_root: Path, limit: int | None):
    manifest = json.loads(manifest_path.read_text())
    records = manifest["tables"][:limit] if limit else manifest["tables"]
    return [
        load_typed_table(
            data_root / Path(r["path"]).name,
            target=r["target_column"],
            expected_sha256=r["sha256"],
            name=r["id"],
        )
        for r in records
    ]


def admissible(table, supports, query) -> bool:
    """Nominal targets need every hidden Query category to have a support code."""
    if table.schema[table.target].kind != "nominal":
        return True
    values = table.values[table.target]
    return set(values[list(query)].tolist()) <= set(values[list(supports)].tolist())


def masked_task(table, rng, device, dtype, *, fraction=0.25, admission=True):
    rows = list(table.train_rows)
    count = max(1, round(fraction * len(rows)))
    for attempt in range(1000):
        query = rng.sample(rows, count)
        supports = sorted(set(rows) - set(query))
        if not admission or admissible(table, supports, query):
            task = table_task(
                table,
                rows,
                query,
                code_seed=rng.getrandbits(31),
                donor_seed=rng.getrandbits(31),
                device=device,
                dtype=dtype,
            )
            return task, attempt
    raise RuntimeError(f"{table.name}: no admissible training mask in 1000 draws")


def build_banks(tables, seed, fit_masks, device, dtype):
    fit, heldout = [], []
    for index, table in enumerate(tables):
        rng = random.Random(seed * 1000 + index)
        for _ in range(fit_masks):
            fit.append((index, masked_task(table, rng, device, dtype, admission=False)[0]))
        rows = list(table.train_rows) + list(table.test_rows)
        heldout.append(
            (
                index,
                table_task(
                    table,
                    rows,
                    table.test_rows,
                    code_seed=rng.getrandbits(31),
                    donor_seed=rng.getrandbits(31),
                    device=device,
                    dtype=dtype,
                ),
            )
        )
    return {"train_fit": fit, "heldout": heldout}


def task_parts(task):
    inputs = task.inputs
    target = int(inputs.query.any(0).nonzero()[0])
    supports = inputs.visible[:, target].nonzero(as_tuple=True)[0].cpu()
    query = inputs.query[:, target].nonzero(as_tuple=True)[0].cpu()
    truth = task.truth.values[target].cpu()[query]
    return target, supports, query, truth


def value_score(kind, prediction, truth) -> float:
    prediction, truth = prediction.detach().cpu(), truth.detach().cpu()
    if kind == "numeric":
        p, t = prediction.to(torch.float64), truth.to(torch.float64)
        total = float((t - t.mean()).square().sum())
        residual = float((p - t).square().sum())
        if not (math.isfinite(total) and math.isfinite(residual)):
            if not bool(torch.isfinite(p).all() & torch.isfinite(t).all()):
                return math.nan
            # R² is unchanged by a common scale. Avoid overflowing the raw
            # squared sums when both original predictions and truth are finite.
            scale = torch.maximum(p.abs().amax(), t.abs().amax())
            p, t = p / scale, t / scale
            total = float((t - t.mean()).square().sum())
            residual = float((p - t).square().sum())
        return 1.0 - residual / total if total > 0 else math.nan
    return float((prediction == truth).to(torch.float64).mean())


def xgboost_prediction(task, kind):
    # macOS wheels of torch and xgboost ship conflicting OpenMP runtimes; use --no-xgboost there.
    import numpy as np
    import xgboost as xgb

    target, supports, query, _ = task_parts(task)
    columns = [a for a in range(len(task.inputs.schema)) if a != target]
    features = np.stack([task.inputs.values[a].cpu().double().numpy() for a in columns], axis=1)
    labels = task.inputs.values[target].cpu().numpy()[supports.numpy()]
    x_fit, x_query = features[supports.numpy()], features[query.numpy()]
    if kind == "numeric":
        model = xgb.XGBRegressor(**XGBOOST, random_state=0)
        model.fit(x_fit, labels)
        return torch.from_numpy(model.predict(x_query).astype("float64"))
    classes = np.unique(labels)
    if len(classes) == 1:
        return torch.full((len(query),), int(classes[0]), dtype=torch.long)
    model = xgb.XGBClassifier(**XGBOOST, random_state=0, eval_metric="logloss")
    model.fit(x_fit, np.searchsorted(classes, labels))
    return torch.from_numpy(classes[model.predict(x_query).astype(int)]).to(torch.long)


def marginal_prediction(task, kind):
    target, supports, query, _ = task_parts(task)
    labels = task.inputs.values[target].cpu()[supports]
    if kind == "numeric":
        labels = labels.to(torch.float64)
        mean = labels.mean()
        if not bool(torch.isfinite(mean)) and bool(torch.isfinite(labels).all()):
            scale = labels.abs().amax()
            mean = (labels / scale).mean() * scale
        return mean.expand(len(query))
    return torch.bincount(labels).argmax().expand(len(query))


def aggregate(tables, per_table):
    """Macro means by target kind over tables; per_table maps method -> [score per table]."""
    result = {}
    for method, scores in per_table.items():
        summary = {}
        for kind in ("numeric", "nominal", "ordinal"):
            values = [
                s
                for t, s in zip(tables, scores, strict=True)
                if t.schema[t.target].kind == kind and math.isfinite(s)
            ]
            summary[kind] = sum(values) / len(values) if values else None
        result[method] = summary
    return result


def versus(model_scores, other_scores, tol=1e-9):
    """Compare finite score pairs; missing/undefined scores are not ties."""
    counts = {"win": 0, "tie": 0, "loss": 0, "not_compared": 0}
    for model_score, other_score in zip(model_scores, other_scores, strict=True):
        if not (math.isfinite(model_score) and math.isfinite(other_score)):
            counts["not_compared"] += 1
        elif model_score > other_score + tol:
            counts["win"] += 1
        elif model_score < other_score - tol:
            counts["loss"] += 1
        else:
            counts["tie"] += 1
    return counts


def baseline_scores(tables, banks, *, use_xgboost=True):
    scores = {}
    for bank, items in banks.items():
        per = {"xgboost": [[] for _ in tables], "support_marginal": [[] for _ in tables]}
        for index, task in items:
            kind = tables[index].schema[tables[index].target].kind
            truth = task_parts(task)[3]
            xgb_score = (
                value_score(kind, xgboost_prediction(task, kind), truth)
                if use_xgboost
                else math.nan
            )
            per["xgboost"][index].append(xgb_score)
            per["support_marginal"][index].append(
                value_score(kind, marginal_prediction(task, kind), truth)
            )
        scores[bank] = {m: [sum(v) / len(v) for v in rows] for m, rows in per.items()}
    return scores


def evaluate(model, tables, banks, baselines, rounds):
    report = {}
    for bank, items in banks.items():
        methods = ["donor"] + [f"round_{t}" for t in range(1, rounds + 1)]
        per = {m: [[] for _ in tables] for m in methods}
        statuses, code_losses = {}, [[] for _ in range(rounds + 1)]
        for index, task in items:
            kind = tables[index].schema[tables[index].target].kind
            truth = task_parts(task)[3]
            trajectory = evaluate_task(model, task)
            statuses[trajectory.status] = statuses.get(trajectory.status, 0) + 1
            per["donor"][index].append(value_score(kind, trajectory.donor, truth))
            for t, prediction in enumerate(trajectory.predictions, start=1):
                per[f"round_{t}"][index].append(value_score(kind, prediction, truth))
            if trajectory.code_losses is not None:
                for t, loss in enumerate(trajectory.code_losses):
                    code_losses[t].append(loss)
        scores = {m: [sum(v) / len(v) for v in rows] for m, rows in per.items()}
        scores |= baselines[bank]
        final = scores[f"round_{rounds}"]
        report[bank] = {
            "macro": aggregate(tables, scores),
            "final_vs_xgboost": versus(final, scores["xgboost"]),
            "final_vs_support_marginal": versus(final, scores["support_marginal"]),
            "mean_code_loss_by_round": [sum(v) / len(v) if v else None for v in code_losses],
            "statuses": statuses,
            "per_table": {t.name: {m: scores[m][i] for m in scores} for i, t in enumerate(tables)},
        }
    return report


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--max-steps", type=int, default=10**9)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--fit-masks", type=int, default=2)
    parser.add_argument("--tables", type=int, default=None, help="first N tables (smoke only)")
    parser.add_argument("--no-xgboost", action="store_true")
    parser.add_argument("--seed", type=int, default=730001)
    parser.add_argument("--config", type=json.loads, default={}, help="V7Config overrides")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "checkpoints").mkdir()
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    tables = load_tables(args.manifest, args.data_root, args.tables)
    config = V7Config(**args.config)
    manifest = {
        "experiment": "v7-old120-pilot-20260930",
        "tables": [[t.name, t.sha256, t.target] for t in tables],
        "protocol": {
            "train": "uniform table, 25% train-row target Query, supports = other train rows",
            "banks": {"train_fit_masks": args.fit_masks, "heldout": "train supports, test Query"},
            "admission": "nominal Query categories must appear in supports (training only)",
        },
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
    }
    write_json(
        args.out / "run.json",
        manifest | {"config": config.as_dict(), "args": {k: str(v) for k, v in vars(args).items()}},
    )

    torch.manual_seed(args.seed)
    model = V7Model(config).to(device, dtype)
    optimizer = make_optimizer(model)
    banks = build_banks(tables, args.seed + 1, args.fit_masks, device, dtype)
    started = time.time()
    baselines = baseline_scores(tables, banks, use_xgboost=not args.no_xgboost)
    print(f"baselines {time.time() - started:.0f}s", flush=True)

    def checkpoint_and_evaluate(step, elapsed):
        began = time.time()
        report = evaluate(model, tables, banks, baselines, config.rounds)
        report |= {"step": step, "train_seconds": elapsed, "eval_seconds": time.time() - began}
        write_json(args.out / f"eval-{step:08d}.json", report)
        if step:
            state = checkpoint_state(model, optimizer, step=step, manifest=manifest)
            save_checkpoint(args.out / "checkpoints" / f"step-{step:08d}.pt", state)
        brief = {b: report[b]["macro"][f"round_{config.rounds}"] for b in banks}
        print(json.dumps({"eval": step, **brief}), flush=True)

    checkpoint_and_evaluate(0, 0.0)
    print(f"manifest {manifest_digest(manifest)[:12]}", flush=True)
    rng = random.Random(args.seed)
    log = (args.out / "train.jsonl").open("a")
    window, rejected, step, train_seconds = [], 0, 0, 0.0
    while step < args.max_steps and train_seconds < args.max_seconds:
        tick = time.time()
        table = tables[rng.randrange(len(tables))]
        task, attempts = masked_task(table, rng, device, dtype)
        rejected += attempts
        record = train_step(model, optimizer, [task])
        step += 1
        train_seconds += time.time() - tick
        window.append(record)
        if step % args.log_every == 0:
            entry = {
                "step": step,
                "loss": sum(r.loss for r in window) / len(window),
                "initial_loss": sum(r.initial_losses[0] for r in window) / len(window),
                "final_round_loss": sum(r.round_losses[0][-1] for r in window) / len(window),
                "rejected_masks": rejected,
                "seconds_per_step": train_seconds / step,
            }
            log.write(json.dumps(entry) + "\n")
            log.flush()
            window = []
        if step % args.eval_every == 0:
            checkpoint_and_evaluate(step, train_seconds)
    if step % args.eval_every:
        checkpoint_and_evaluate(step, train_seconds)
    print(json.dumps({"done": step, "train_seconds": train_seconds}), flush=True)


if __name__ == "__main__":
    main()
