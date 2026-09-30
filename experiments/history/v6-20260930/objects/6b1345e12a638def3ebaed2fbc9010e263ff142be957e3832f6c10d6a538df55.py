"""Four matched short fits: V5.5/V6 x signal6/full32, with a shared H8 parent.

This is an isolated runner. It does not modify the live V6 training source.
Manifests/data must already exist under --root. All outputs use a new --output.
The common episode table name intentionally aligns row/mask/code randomness.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path


SEED = 20260927
ARMS = ("signal6", "full32")
KINDS = ("v55", "v6")
PAIR_NAME = "puma-paired"
PAIR_NAMESPACE = "v6-sparse-relevance-paired-20260927"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                allow_nan=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paired_seeds(plan):
    """Exactly mirror runner.train_step's seed namespace transformation."""
    seeds = dict(plan.spec["seeds"])
    for name in ("masks", "codes", "windows"):
        seeds[name] = int.from_bytes(hashlib.sha256(
            f"{seeds[name]}/{PAIR_NAMESPACE}".encode()).digest()[:8], "little")
    return seeds


def episode_identity(info):
    # Drop the target column index, which is legitimately 6 versus 32.
    return {key: info[key] for key in (
        "row_ids", "query_row_ids", "mask_seed", "code_seed", "window_seed")}


def metrics(records, median, scale):
    ys = [r["target"] for r in records]
    ps = [r["prediction"] for r in records]
    ym, pm = statistics.fmean(ys), statistics.fmean(ps)
    vy = statistics.fmean((v - ym) ** 2 for v in ys)
    vp = statistics.fmean((v - pm) ** 2 for v in ps)
    mse = statistics.fmean((y - p) ** 2 for y, p in zip(ys, ps))
    log_error = statistics.fmean(math.log1p(((y - p) / scale) ** 2)
                                for y, p in zip(ys, ps))
    log_ref = statistics.fmean(math.log1p(((y - median) / scale) ** 2) for y in ys)
    cov = statistics.fmean((y - ym) * (p - pm) for y, p in zip(ys, ps))
    return dict(n=len(ys), r2=1 - mse / vy, slog=1 - log_error / log_ref,
                prediction_sd=math.sqrt(vp), target_sd=math.sqrt(vy),
                prediction_sd_ratio=math.sqrt(vp / vy),
                correlation=cov / math.sqrt(vy * vp) if vp else 0.0,
                prediction_mean=pm, target_mean=ym)


def make_evaluation_bank(data, episodes):
    train, test = data["splits"]["train"], data["splits"]["test"]
    bank = []
    for index in range(episodes):
        rng = random.Random(SEED + 1000 + index)
        support = rng.sample(train, 136)
        available = sorted(set(train) - set(support))
        bank.append(dict(episode=index, support=support,
                         train_query=rng.sample(available, 68),
                         test_query=rng.sample(test, 68), code_seed=SEED))
    return bank


def evaluate(model, data, bank, device):
    import torch
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v55 import (
        ColumnSchema, RestorationInput, RestorationRequest,
    )
    schema = tuple(ColumnSchema(feature["key"], "numeric")
                   for feature in data["features"])
    target = len(schema) - 1
    train_y = [data["values"][i][target] for i in data["splits"]["train"]]
    median = statistics.median(train_y)
    scale = max(statistics.median(abs(y - median) for y in train_y), 1e-12)
    records = {"train_query": [], "test_query": []}
    model.eval()
    for episode in bank:
        for kind in records:
            query = episode[kind]
            rows = episode["support"] + query
            values = [list(data["values"][i]) for i in rows]
            for row in values[136:]:
                row[target] = 0.0
            tensor = torch.tensor(values, dtype=execution_dtype(device), device=device)
            visible = torch.ones_like(tensor, dtype=torch.bool)
            visible[136:, target] = False
            inputs = RestorationInput(schema, tuple(tensor[:, a] for a in range(len(schema))),
                                      visible, ~visible, episode["code_seed"])
            request = RestorationRequest((~visible).nonzero())
            with torch.inference_mode():
                output = model(inputs, request)
            column = output.columns[0]
            if column.column != target or column.result.status != "ok":
                raise ValueError("invalid target readout")
            # V6Model already divides its target encoding by two before decode.
            predictions = column.decoded.detach().cpu().tolist()
            if len(predictions) != 68:
                raise ValueError("evaluation Query count drift")
            records[kind].extend(dict(episode=episode["episode"], row_id=row,
                                      target=data["values"][row][target], prediction=float(pred))
                                 for row, pred in zip(query, predictions))
            del output, inputs, request, tensor
    return {kind: metrics(rows, median, scale) for kind, rows in records.items()}, records


def checkpoint(folder, model, optimizer, state, identity):
    import torch
    from tabu_lab.curriculum_v53.artifacts import rng_state, finite_state
    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
        raise FloatingPointError("checkpoint rejected: nonfinite state")
    path = folder / f"checkpoint-u{state['update']:04d}.pt"
    temporary = path.with_suffix(".tmp")
    torch.save(dict(schema="tabu.sparse-relevance-paired-checkpoint.v1", purpose="training",
                    identity=identity, model_config=model.config.as_dict(),
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    optimizer=optimizer.state_dict(), state=dict(state), rng=rng_state()), temporary)
    temporary.replace(path)
    receipt = dict(path=str(path.resolve()), sha256=digest(path), update=state["update"])
    dump(folder / "checkpoint-latest.json", receipt)
    return receipt


def main(args):
    import torch
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint, finite_state
    from tabu_lab.curriculum_v53.data import build_episode
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53.runner import configure_runtime, train_step
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v53 import V53LossConfig
    from tabu_lab.models.restoration_v6 import V6Model, score_training_episode
    from tabu_lab.restoration_optimizers import adamw

    root, output = args.root.resolve(), args.output.resolve()
    plans = {arm: load_v55_plan(root / "manifests" / f"{arm}.json") for arm in ARMS}
    payload, parent_sha = load_checkpoint(args.parent)
    if parent_sha != args.parent_sha or payload.get("purpose") != "training":
        raise ValueError("parent checkpoint SHA/purpose mismatch")
    first = plans[ARMS[0]]
    if first.config.backbone.heads != 8 or first.config.numeric_scaling != "zscore":
        raise ValueError("this probe requires the declared DGX2 H8 zscore configuration")
    for arm, plan in plans.items():
        if plan.config.as_dict() != payload["model_config"] or len(plan.tables) != 1:
            raise ValueError(f"{arm}: parent configuration or single-table contract differs")
        if plan.config.as_dict() != first.config.as_dict() or plan.optimizer != first.optimizer:
            raise ValueError("paired arms require identical model and optimizer settings")
        if plan.spec["seeds"] != first.spec["seeds"]:
            raise ValueError("paired arms require identical seed streams")
        table = plan.tables[0]
        if table.window_rows != 204 or table.target_column != table.width - 1:
            raise ValueError("expected 204 rows and final target column")
        if any(column.kind != "numeric" for column in table.schema):
            raise ValueError("this frozen Puma probe is numeric-only")
        if table.width != (7 if arm == "signal6" else 33):
            raise ValueError("signal6/full32 width mismatch")
        stage = plan.spec["stages"][0]
        if (stage["recipe"][table.kind]["kind"] != "supervised_row"
                or stage["recipe"][table.kind]["fraction"] != 1 / 3):
            raise ValueError("expected the original supervised-row 136+68 recipe")
        if stage.get("objective") != {"kind": "squared"}:
            raise ValueError("probe requires squared loss")
    datasets = {arm: json.loads(plans[arm].tables[0].path.read_text()) for arm in ARMS}
    d0, d1 = (datasets[arm] for arm in ARMS)
    if d0["splits"] != d1["splits"] or len(d0["values"]) != len(d1["values"]):
        raise ValueError("paired row identities/splits differ")
    keys = [item["key"] for item in d1["features"]]
    mapping = [keys.index(item["key"]) for item in d0["features"]]
    if any(a != [b[j] for j in mapping] for a, b in zip(d0["values"], d1["values"])):
        raise ValueError("signal6 values/targets are not an exact projection of full32")
    bank = make_evaluation_bank(d0, args.episodes)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "evaluation-bank.json", bank)
    runtime = configure_runtime(args.device)
    identity = dict(schema="tabu.sparse-relevance-paired-probe.v1", parent_sha256=parent_sha,
                    parent_update=payload["state"]["update"], initialization="weights_only",
                    script_sha256=digest(Path(__file__)), updates_per_run=args.updates,
                    arms=list(ARMS), models=list(KINDS), paired_table_name=PAIR_NAME,
                    namespace=PAIR_NAMESPACE, runtime=runtime,
                    training_context=136, training_query=68, evaluation_episodes=args.episodes,
                    evaluation_bank_sha256=digest(output / "evaluation-bank.json"),
                    manifest_sha256={a: digest(root / "manifests" / f"{a}.json") for a in ARMS},
                    source_identity={a: plans[a].identity for a in ARMS},
                    note="Equal updates, not equal compute; losses have different definitions.")
    import inspect
    identity["v6_code_sha256"] = {Path(inspect.getfile(obj)).name: digest(inspect.getfile(obj))
                                  for obj in (V6Model, score_training_episode)}
    dump(output / "campaign.json", identity)
    expected_episodes = {}
    summary = {}
    for arm in ARMS:
        plan = plans[arm]
        table = dataclasses.replace(plan.tables[0], name=PAIR_NAME)
        recipe = plan.spec["stages"][0]["recipe"][table.kind]
        loss_config = V53LossConfig(**plan.spec["stages"][0]["loss"])
        for kind in KINDS:
            folder = output / f"{kind}-{arm}"
            folder.mkdir()
            run_id = {**identity, "model_kind": kind, "arm": arm}
            model = (V6Model(plan.config) if kind == "v6" else make_model(plan))
            model = model.to(device=args.device, dtype=execution_dtype(args.device))
            model.load_state_dict(payload["model"], strict=True)
            for name, value in model.state_dict().items():
                if not torch.equal(value.cpu(), payload["model"][name].to(value.dtype)):
                    raise ValueError(f"initial weights differ: {name}")
            seed = plan.spec["seeds"]["model"]
            random.seed(seed)
            torch.manual_seed(seed)
            if args.device == "cuda:0":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.reset_peak_memory_stats()
            optimizer = adamw(model, plan.optimizer)
            state = dict(update=0, cursor=0, train_step_seconds=0.0)
            initial_score, records = evaluate(model, datasets[arm], bank, args.device)
            dump(folder / "evaluation-u0000.json", dict(parent_sha256=parent_sha,
                                                        metrics=initial_score, predictions=records))
            print(json.dumps(dict(event="initial-evaluation", model=kind, arm=arm,
                                  metrics=initial_score)), flush=True)
            saved = None
            for index in range(args.updates):
                if args.device == "cuda:0":
                    torch.cuda.synchronize()
                started = time.monotonic()
                if kind == "v55":
                    row = train_step(model, optimizer, plan, table, recipe, index, args.device,
                                     loss_config, namespace=PAIR_NAMESPACE,
                                     objective={"kind": "squared"})
                    info, loss, grad_norm = row["episode"], row["loss"], row["gradient_norm"]
                    scored = row["readout_targets"]
                else:
                    inputs, _, truth, info = build_episode(
                        table, recipe, index, paired_seeds(plan), args.device,
                        epsilon=plan.config.epsilon, codec_version=plan.config.codec_version)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    score = score_training_episode(model, inputs, truth)
                    score.loss.backward()
                    params = [p for p in model.parameters() if p.grad is not None]
                    if not params or not finite_state([p.grad for p in params]):
                        raise FloatingPointError("missing/nonfinite V6 gradients")
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(
                        params, plan.optimizer.grad_clip, error_if_nonfinite=True))
                    optimizer.step()
                    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
                        raise FloatingPointError("nonfinite V6 update")
                    loss, scored = float(score.loss.detach()), score.scored_cells
                    del inputs, truth, score, params
                if args.device == "cuda:0":
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - started
                paired = episode_identity(info)
                if info["selected_rows"] != 204 or len(info["query_row_ids"]) != 68:
                    raise ValueError("training support/Query counts differ from 136+68")
                if index not in expected_episodes:
                    expected_episodes[index] = paired
                    append(output / "training-episode-bank.jsonl", dict(index=index, **paired))
                elif paired != expected_episodes[index]:
                    raise ValueError(f"paired training episode differs at {kind}/{arm}/{index}")
                state.update(update=index + 1, cursor=index + 1,
                             train_step_seconds=state["train_step_seconds"] + elapsed)
                log = dict(update=index + 1, model=kind, arm=arm, loss=loss,
                           gradient_norm=grad_norm, scored_cells=scored,
                           seconds=elapsed, train_step_seconds=state["train_step_seconds"])
                append(folder / "updates.jsonl", log)
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.updates:
                    print(json.dumps(log), flush=True)
                if (index + 1) % 50 == 0 or index + 1 == args.updates:
                    saved = checkpoint(folder, model, optimizer, state, run_id)
            score, records = evaluate(model, datasets[arm], bank, args.device)
            result = dict(model=kind, arm=arm, checkpoint=saved, state=state,
                          initial_metrics=initial_score, metrics=score, context_rows=136, query_rows=68,
                          peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                if args.device == "cuda:0" else None))
            dump(folder / "evaluation.json", {**result, "predictions": records})
            dump(folder / "terminal.json", {**result, "outcome": "completed"})
            summary[f"{kind}-{arm}"] = result
            dump(output / "summary.json", summary)
            print(json.dumps(dict(event="completed", **result)), flush=True)
            del model, optimizer
            gc.collect()
            if args.device == "cuda:0":
                torch.cuda.empty_cache()
    dump(output / "terminal.json", dict(outcome="completed", runs=len(summary),
                                       paired_training_episodes_verified=len(expected_episodes),
                                       summary=summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha", required=True)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if args.updates < 1 or args.episodes < 1:
        parser.error("updates and episodes must be positive")
    main(args)
