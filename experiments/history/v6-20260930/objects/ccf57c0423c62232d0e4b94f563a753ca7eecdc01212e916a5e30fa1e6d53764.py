"""Full32 mechanism ablations; 150 updates with the existing paired episode bank.

No source patching, automatic host changes, or training start on import.
Run only after the experiment host/GPU has been allocated by the coordinator.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib.util
import inspect
import json
import random
import sys
import time
from pathlib import Path


VARIANTS = ("broadcast-only", "all-column-loss-only")


def load_helpers(path):
    spec = importlib.util.spec_from_file_location("frozen_sparse_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(torch, device):
    if device == "cuda:0":
        torch.cuda.synchronize()


def main(args):
    import torch
    # The verified host launcher may invoke this file through runpy from source/src.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ablation import MechanismAblation
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint, finite_state
    from tabu_lab.curriculum_v53.data import build_episode
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53.runner import configure_runtime
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v53 import V53LossConfig, score_prepared_episode
    from tabu_lab.models.restoration_v53.training import prepare_training_episode
    from tabu_lab.models.restoration_v6 import score_training_episode
    from tabu_lab.restoration_optimizers import adamw

    probe_root = args.probe_root.resolve()
    helpers = load_helpers(probe_root / "probe.py")
    plan = load_v55_plan(probe_root / "manifests" / "full32.json")
    payload, parent_sha = load_checkpoint(args.parent)
    if parent_sha != args.parent_sha or payload.get("purpose") != "training":
        raise ValueError("parent SHA/purpose mismatch")
    if plan.config.as_dict() != payload["model_config"]:
        raise ValueError("parent model configuration mismatch")
    if plan.config.backbone.heads != 8 or plan.config.numeric_scaling != "zscore":
        raise ValueError("expected the existing H8/zscore configuration")
    if len(plan.tables) != 1:
        raise ValueError("exactly one full32 table is required")
    original_table = plan.tables[0]
    if (original_table.width != 33 or original_table.target_column != 32
            or original_table.window_rows != 204
            or any(c.kind != "numeric" for c in original_table.schema)):
        raise ValueError("full32/204-row numeric contract mismatch")
    table = dataclasses.replace(original_table, name=helpers.PAIR_NAME)
    stage = plan.spec["stages"][0]
    recipe = stage["recipe"][table.kind]
    if (recipe.get("kind") != "supervised_row" or recipe.get("fraction") != 1 / 3
            or recipe.get("numeric_query_guard", {"kind": "none"}) != {"kind": "none"}
            or set(recipe) - {"kind", "fraction", "numeric_query_guard"}):
        raise ValueError("expected the original 136+68 recipe")
    if stage.get("objective") != {"kind": "squared"}:
        raise ValueError("only squared losses are used in these ablations")
    loss_config = V53LossConfig(**stage["loss"])
    if tuple(loss_config.state_weights) != (0.0, 1.0, 0.0, 0.0):
        raise ValueError("broadcast-only must retain target-only Query loss")

    reference = args.reference_run.resolve()
    campaign = json.loads((reference / "campaign.json").read_text())
    if (campaign["parent_sha256"] != parent_sha or campaign["namespace"] != helpers.PAIR_NAMESPACE
            or campaign["paired_table_name"] != helpers.PAIR_NAME):
        raise ValueError("the reference run has a different parent or episode namespace")
    if campaign["manifest_sha256"]["full32"] != helpers.digest(plan.path):
        raise ValueError("full32 manifest differs from the completed reference run")
    if campaign["script_sha256"] != helpers.digest(probe_root / "probe.py"):
        raise ValueError("the original probe helpers changed after the reference run")
    bank = json.loads((reference / "evaluation-bank.json").read_text())
    data = json.loads(original_table.path.read_text())
    if len(bank) != 8 or bank != helpers.make_evaluation_bank(data, 8):
        raise ValueError("reference evaluation bank is not the fixed 8 x 136+68 bank")
    training_bank = {}
    for line in (reference / "training-episode-bank.jsonl").read_text().splitlines():
        record = json.loads(line)
        index = record.pop("index")
        if index in training_bank:
            raise ValueError("duplicate reference training episode")
        training_bank[index] = record
    if any(index not in training_bank for index in range(args.updates)):
        raise ValueError("reference run lacks the requested training episode addresses")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    args.output_created = True
    runtime = configure_runtime(args.device)
    identity = dict(schema="tabu.full32-mechanism-ablation.v1", parent_sha256=parent_sha,
                    parent_update=payload["state"]["update"], initialization="weights_only",
                    model_config=plan.config.as_dict(), optimizer=dataclasses.asdict(plan.optimizer),
                    runtime=runtime, updates_per_variant=args.updates,
                    variants=list(VARIANTS), reference_run=str(reference),
                    reference_campaign_sha256=helpers.digest(reference / "campaign.json"),
                    reference_training_bank_sha256=helpers.digest(reference / "training-episode-bank.jsonl"),
                    evaluation_bank_sha256=helpers.digest(reference / "evaluation-bank.json"),
                    manifest_sha256=helpers.digest(plan.path), source_identity=plan.identity,
                    namespace=helpers.PAIR_NAMESPACE, paired_table_name=helpers.PAIR_NAME,
                    code_sha256={"run.py": helpers.digest(Path(__file__)),
                                 "ablation.py": helpers.digest(Path(inspect.getfile(MechanismAblation))),
                                 "probe.py": helpers.digest(probe_root / "probe.py"),
                                 "v6_training.py": helpers.digest(inspect.getfile(score_training_episode))},
                    timing="parameter_update_seconds includes successful forward/loss/backward/optimizer; excludes episode build, checkpoint and evaluation")
    helpers.dump(output / "campaign.json", identity)
    helpers.dump(output / "evaluation-bank.json", bank)
    results = {}
    for variant in VARIANTS:
        folder = output / variant
        folder.mkdir()
        model = MechanismAblation(plan.config, variant).to(
            device=args.device, dtype=execution_dtype(args.device))
        model.load_state_dict(payload["model"], strict=True)
        for name, tensor in model.state_dict().items():
            if not torch.equal(tensor.cpu(), payload["model"][name].to(tensor.dtype)):
                raise ValueError(f"initial parameter differs: {name}")
        seed = plan.spec["seeds"]["model"]
        random.seed(seed)
        torch.manual_seed(seed)
        if args.device == "cuda:0":
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        optimizer = adamw(model, plan.optimizer)
        state = dict(update=0, cursor=0, parameter_update_seconds=0.0,
                     episode_build_seconds=0.0)
        synchronize(torch, args.device)
        started = time.monotonic()
        initial, predictions = helpers.evaluate(model, data, bank, args.device)
        synchronize(torch, args.device)
        initial_seconds = time.monotonic() - started
        helpers.dump(folder / "evaluation-u0000.json", dict(update=0, parent_sha256=parent_sha,
                                                           metrics=initial, predictions=predictions,
                                                           evaluation_seconds=initial_seconds))
        print(json.dumps(dict(event="initial-evaluation", variant=variant, metrics=initial)), flush=True)
        last_checkpoint = None
        for index in range(args.updates):
            synchronize(torch, args.device)
            started = time.monotonic()
            inputs, request, truth, info = build_episode(
                table, recipe, index, helpers.paired_seeds(plan), args.device,
                epsilon=plan.config.epsilon, codec_version=plan.config.codec_version)
            synchronize(torch, args.device)
            build_seconds = time.monotonic() - started
            # Refuse a mismatch before any gradient or parameter update.
            if helpers.episode_identity(info) != training_bank[index]:
                raise ValueError(f"reference training episode differs at index {index}")
            if info["selected_rows"] != 204 or len(info["query_row_ids"]) != 68:
                raise ValueError("136+68 training row count drift")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            synchronize(torch, args.device)
            started = time.monotonic()
            if variant == "broadcast-only":
                prepared = prepare_training_episode(model, inputs, request, truth, loss_config)
                score = score_prepared_episode(model, prepared, loss_config=loss_config)
                scored_cells = len(prepared.visible.request.targets)
            else:
                score = score_training_episode(model, inputs, truth)
                scored_cells = score.scored_cells
            score.loss.backward()
            params = [p for p in model.parameters() if p.grad is not None]
            if not params or not finite_state([p.grad for p in params]):
                raise FloatingPointError("missing/nonfinite gradients")
            grad_norm = torch.nn.utils.clip_grad_norm_(params, plan.optimizer.grad_clip,
                                                       error_if_nonfinite=True)
            optimizer.step()
            synchronize(torch, args.device)
            update_seconds = time.monotonic() - started
            if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
                raise FloatingPointError("nonfinite parameter/optimizer state")
            expected_cells = 68 if variant == "broadcast-only" else 68 * 33
            if scored_cells != expected_cells:
                raise ValueError("scored Cell coverage differs from ablation definition")
            state.update(update=index + 1, cursor=index + 1,
                         parameter_update_seconds=state["parameter_update_seconds"] + update_seconds,
                         episode_build_seconds=state["episode_build_seconds"] + build_seconds)
            row = dict(update=index + 1, variant=variant, loss=float(score.loss.detach()),
                       gradient_norm=float(grad_norm), scored_cells=scored_cells,
                       parameter_update_seconds=update_seconds, episode_build_seconds=build_seconds,
                       cumulative_parameter_update_seconds=state["parameter_update_seconds"])
            helpers.append(folder / "updates.jsonl", row)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == args.updates:
                print(json.dumps(row), flush=True)
            del score, inputs, request, truth, params
            if variant == "broadcast-only":
                del prepared
            if (index + 1) % 50 == 0 or index + 1 == args.updates:
                last_checkpoint = helpers.checkpoint(folder, model, optimizer, state,
                                                     {**identity, "variant": variant})
        synchronize(torch, args.device)
        started = time.monotonic()
        final, predictions = helpers.evaluate(model, data, bank, args.device)
        synchronize(torch, args.device)
        final_seconds = time.monotonic() - started
        delta = {split: {metric: final[split][metric] - initial[split][metric]
                         for metric in ("r2", "slog", "prediction_sd")}
                 for split in ("train_query", "test_query")}
        result = dict(variant=variant, outcome="completed", checkpoint=last_checkpoint, state=state,
                      initial_metrics=initial, metrics=final, delta=delta,
                      initial_evaluation_seconds=initial_seconds, final_evaluation_seconds=final_seconds,
                      peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                            if args.device == "cuda:0" else None))
        helpers.dump(folder / f"evaluation-u{args.updates:04d}.json", {**result, "predictions": predictions})
        helpers.dump(folder / "terminal.json", result)
        results[variant] = result
        helpers.dump(output / "summary.json", results)
        print(json.dumps(dict(event="completed", **result)), flush=True)
        del model, optimizer
        gc.collect()
        if args.device == "cuda:0":
            torch.cuda.empty_cache()
    helpers.dump(output / "terminal.json", dict(outcome="completed", results=results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-root", type=Path,
                        default=Path(__file__).resolve().parent.parent / "v6-sparse-relevance-probe-20260927")
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=150)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("updates must be positive")
    try:
        main(args)
    except BaseException as error:
        if getattr(args, "output_created", False):
            receipt = dict(outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error_type=type(error).__name__, error=str(error),
                           note="Only already-written checkpoints are durable; inspect per-variant receipts.")
            (args.output / "failure.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
