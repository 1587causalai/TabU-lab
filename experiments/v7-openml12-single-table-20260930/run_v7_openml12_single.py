"""V7 OpenML12 per-table scratch fit, using the frozen V6 row splits.

Each table is an independent run: no parent checkpoint, no replay, no mixing.
Training sees only the V6 fit rows. Monitor rows select the checkpoint.
Test rows are scored once, after that selection is frozen.

The published MLP/XGBoost scores are copied from the V6 single-table RESULT
and are not recomputed here.
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
    load_checkpoint,
    load_typed_table,
    make_optimizer,
    save_checkpoint,
    table_task,
    train_step,
)

TABLES = (
    "airfoil_self_noise",
    "bodyfat",
    "cars",
    "concrete_compressive_strength",
    "cpu_activity",
    "energy_efficiency",
    "kin8nm",
    "pumadyn32nh",
    "qsar_fish_toxicity",
    "red_wine",
    "space_ga",
    "white_wine",
)
MONITOR_SECONDS = (0, 180, 360, 540, 720, 900)
PUBLISHED = {
    "airfoil_self_noise": {
        "mlp": 0.94843,
        "xgb": 0.92841,
        "v6_dustin": 0.95927,
        "v55_dgx2": 0.96343,
    },
    "bodyfat": {"mlp": 0.98575, "xgb": 0.99639, "v6_dustin": 0.99456, "v55_dgx2": 0.98759},
    "cars": {"mlp": 0.96739, "xgb": 0.94824, "v6_dustin": 0.96501, "v55_dgx2": 0.95574},
    "concrete_compressive_strength": {
        "mlp": 0.89951,
        "xgb": 0.92685,
        "v6_dustin": 0.92543,
        "v55_dgx2": 0.92538,
    },
    "cpu_activity": {"mlp": 0.98070, "xgb": 0.98504, "v6_dustin": 0.93644, "v55_dgx2": 0.97312},
    "energy_efficiency": {
        "mlp": 0.99701,
        "xgb": 0.99856,
        "v6_dustin": 0.99771,
        "v55_dgx2": 0.99841,
    },
    "kin8nm": {"mlp": 0.92319, "xgb": 0.80195, "v6_dustin": 0.92005, "v55_dgx2": 0.88177},
    "pumadyn32nh": {"mlp": 0.50461, "xgb": 0.63689, "v6_dustin": 0.64792, "v55_dgx2": 0.62173},
    "qsar_fish_toxicity": {
        "mlp": 0.50542,
        "xgb": 0.53686,
        "v6_dustin": 0.50244,
        "v55_dgx2": 0.52439,
    },
    "red_wine": {"mlp": 0.37102, "xgb": 0.46506, "v6_dustin": 0.41949, "v55_dgx2": 0.32225},
    "space_ga": {"mlp": 0.77233, "xgb": 0.65110, "v6_dustin": 0.74531, "v55_dgx2": 0.73856},
    "white_wine": {"mlp": 0.41855, "xgb": 0.48376, "v6_dustin": 0.37489, "v55_dgx2": 0.39011},
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=True) + "\n")


def r2_mse(prediction, truth) -> dict[str, float]:
    prediction = prediction.detach().cpu().to(torch.float64)
    truth = truth.detach().cpu().to(torch.float64)
    error = prediction - truth
    mse = float(error.square().mean())
    total = float((truth - truth.mean()).square().sum())
    return {
        "mse": mse,
        "mae": float(error.abs().mean()),
        "r2": 1.0 - float(error.square().sum()) / total if total > 0 else math.nan,
        "n": int(truth.numel()),
    }


def load_split(split_root: Path, name: str, table) -> dict:
    split = json.loads((split_root / name / "split.json").read_text())
    fit, valid, train, test = (
        tuple(split["fit_row_ids"]),
        tuple(split["validation_row_ids"]),
        tuple(split["train_row_ids"]),
        tuple(split["test_row_ids"]),
    )
    if set(train) != set(table.train_rows) or set(test) != set(table.test_rows):
        raise ValueError(f"{name}: V6 split does not match the OpenML12 table rows")
    if set(fit) | set(valid) != set(train) or set(fit) & set(valid):
        raise ValueError(f"{name}: fit/validation must partition the original train rows")
    return {"fit": fit, "validation": valid, "train": train, "test": test, "seed": split["seed"]}


def window_task(table, pool, rng, device, dtype, window, n_query):
    rows = rng.sample(list(pool), window)
    query = rng.sample(rows, n_query)
    return table_task(
        table,
        rows,
        query,
        code_seed=rng.getrandbits(31),
        donor_seed=rng.getrandbits(31),
        device=device,
        dtype=dtype,
    )


def full_task(table, support, query, seed, device, dtype):
    """Hide Query labels even when Query rows come from the supplied support pool."""
    rng = random.Random(seed)
    query = list(query)
    query_set = set(query)
    rows = [row for row in support if row not in query_set] + query
    return table_task(
        table,
        rows,
        query,
        code_seed=rng.getrandbits(31),
        donor_seed=rng.getrandbits(31),
        device=device,
        dtype=dtype,
    )


def score_task(model, task) -> dict:
    trajectory = evaluate_task(model, task)
    truth = task.truth.values[task.inputs.query.any(0).nonzero()[0].item()]
    query = task.inputs.query.any(1).nonzero(as_tuple=True)[0]
    metrics = r2_mse(trajectory.predictions[-1], truth[query])
    metrics.update(
        status=trajectory.status,
        donor=r2_mse(trajectory.donor, truth[query]),
        support_marginal=trajectory.baselines["support_marginal"],
        code_losses=trajectory.code_losses,
    )
    return metrics


def run_table(name, args, device, dtype, config):
    out = args.out / "tables" / name
    if (out / "terminal.json").exists():
        return json.loads((out / "terminal.json").read_text())
    table = load_typed_table(args.data / f"{name}.json", name=name)
    split = load_split(args.splits, name, table)
    window = min(args.window, len(split["fit"]))
    n_query = max(1, round(window * args.query_fraction))
    identity = {
        "experiment": "v7-openml12-single-table-20260930",
        "table": name,
        "sha256": table.sha256,
        "target": table.target,
        "split_seed": split["seed"],
        "fit_rows": len(split["fit"]),
        "monitor_rows": len(split["validation"]),
        "test_rows": len(split["test"]),
        "window_rows": window,
        "query_rows": n_query,
        "config": config.as_dict(),
        "dtype": args.dtype,
        "parent": None,
    }
    write_json(out / "identity.json", identity)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = V7Model(config).to(device, dtype)
    optimizer = make_optimizer(model)
    rng = random.Random(args.seed + TABLES.index(name) * 1_000_003)
    fit_query = tuple(random.Random(args.seed + 17).sample(list(split["fit"]), n_query))
    banks = {
        "monitor": full_task(
            table, split["fit"], split["validation"], args.seed + 11, device, dtype
        ),
        "test": full_task(table, split["fit"], split["test"], args.seed + 13, device, dtype),
        "train_fit": full_task(table, split["fit"], fit_query, args.seed + 17, device, dtype),
    }
    nodes, elapsed, updates = {}, 0.0, 0
    log = (out / "updates.jsonl").open("a")

    def evaluate_node(seconds: int, current_model, current_optimizer) -> dict:
        report = {
            bank: score_task(current_model, task) for bank, task in banks.items() if bank != "test"
        }
        report.update(seconds=elapsed, updates=updates)
        write_json(out / "monitor" / f"sec{seconds:04d}.json", report)
        save_checkpoint(
            out / "checkpoints" / f"sec{seconds:04d}.pt",
            checkpoint_state(current_model, current_optimizer, step=updates, manifest=identity),
        )
        print(
            json.dumps({"table": name, "node": seconds, "monitor_r2": report["monitor"]["r2"]}),
            flush=True,
        )
        return report

    nodes["0"] = evaluate_node(0, model, optimizer)
    for target in MONITOR_SECONDS[1:]:
        while elapsed < min(target, args.seconds):
            tick = time.time()
            record = train_step(
                model,
                optimizer,
                [window_task(table, split["fit"], rng, device, dtype, window, n_query)],
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.time() - tick
            updates += 1
            log.write(
                json.dumps({"update": updates, "seconds": elapsed, "loss": record.loss}) + "\n"
            )
            if updates % 25 == 0:
                log.flush()
        nodes[str(target if elapsed >= target else int(elapsed))] = evaluate_node(
            target if elapsed >= target else int(elapsed), model, optimizer
        )
        if elapsed >= args.seconds:
            break
    log.close()
    chosen = min(nodes, key=lambda k: (nodes[k]["monitor"]["mse"], int(k)))
    selected_seconds = int(chosen)
    load_checkpoint(
        out / "checkpoints" / f"sec{selected_seconds:04d}.pt",
        model,
        optimizer,
        manifest=identity,
    )
    test = score_task(model, banks["test"])
    train_fit = score_task(model, banks["train_fit"])
    terminal = {
        "table": name,
        "selected_node": chosen,
        "updates": updates,
        "successful_seconds": elapsed,
        "train_r2": train_fit["r2"],
        "monitor_r2": nodes[chosen]["monitor"]["r2"],
        "test_r2": test["r2"],
        "test_mse": test["mse"],
        "donor_test_r2": test["donor"]["r2"],
        "published": PUBLISHED[name],
        "vs_mlp": test["r2"] - PUBLISHED[name]["mlp"],
        "vs_xgb": test["r2"] - PUBLISHED[name]["xgb"],
        "vs_v55_dgx2": test["r2"] - PUBLISHED[name]["v55_dgx2"],
        "nodes": {
            k: {"monitor_r2": v["monitor"]["r2"], "monitor_mse": v["monitor"]["mse"]}
            for k, v in nodes.items()
        },
        "identity": identity,
    }
    write_json(out / "terminal.json", terminal)
    print(json.dumps({"done": name, "test_r2": test["r2"], "selected": chosen}), flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    parser.add_argument("--seconds", type=float, default=900.0)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--query-fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--tables", nargs="*", default=list(TABLES))
    parser.add_argument("--config", type=json.loads, default={})
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5)
    config = V7Config(**args.config)
    write_json(
        args.out / "experiment.json",
        {
            "experiment": "v7-openml12-single-table-20260930",
            "tables": args.tables,
            "seconds": args.seconds,
            "window": args.window,
            "query_fraction": args.query_fraction,
            "config": config.as_dict(),
            "device": str(device),
            "dtype": args.dtype,
            "parent": None,
            "data": str(args.data),
            "splits": str(args.splits),
            "protocol": (
                "V6 fit/monitor/test splits; scratch V7; monitor-MSE selection; test after freeze"
            ),
        },
    )
    summary = [run_table(name, args, device, dtype, config) for name in args.tables]
    write_json(args.out / "summary.json", summary)
    print(json.dumps({"completed": [row["table"] for row in summary]}), flush=True)


if __name__ == "__main__":
    main()
