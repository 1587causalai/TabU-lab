"""Training-only joint fitting over immutable synthetic tables, one shared model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import signal
import time
from collections import Counter
from pathlib import Path

import torch

from tabu_lab.models.tar import TabUTARModel, TARTrainer, TARTrainingConfig
from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint, source_digest
from tabu_lab.models.tar.episodes import covering_fit_episodes, sample_supervised_episode
from tabu_lab.models.tar.training import score
from tabu_lab.tar_data import dataset_features, validate_full_dataset
from tabu_lab.tar_fit import fit_model_config, git_source_state, gpu_preflight, sha


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def runner_source_hash():
    root = Path(__file__).parent
    files = ("tar_joint_fit.py", "tar_data.py", "tar_fit.py", "tar_sizes.py")
    return hashlib.sha256(
        json.dumps({name: sha(root / name) for name in files}, sort_keys=True).encode()
    ).hexdigest()


def round_order(names, seed, round_index):
    names = list(names)
    random.Random(f"{seed}/joint-order/{round_index}").shuffle(names)
    return names


def load_tables(spec, parent):
    if spec.get("corpus_manifest_sha256") and (
        sha(parent / "manifest.json") != spec["corpus_manifest_sha256"]
    ):
        raise ValueError("corpus manifest digest mismatch")
    tables = {}
    for name, digest in spec["datasets"].items():
        path = parent / "data" / f"{name}.json"
        if sha(path) != digest:
            raise ValueError(f"dataset digest mismatch: {name}")
        data = json.loads(path.read_text())
        coverage = validate_full_dataset(data, spec["expected_rows"][name])
        pool_ids = data["splits"]["train"]
        pool = torch.tensor(data["values"], dtype=torch.float64)[pool_ids]
        features = dataset_features(data)
        context = len(pool) - math.ceil(len(pool) * spec["mask_fraction"])
        bank = covering_fit_episodes(
            pool,
            features,
            row_ids=pool_ids,
            query_size=len(pool) - context,
            seed=spec["episode_seed"],
            namespace=f"{name}/covering-fit",
            count=spec["fit_eval_episodes"],
        )
        tables[name] = dict(
            pool=pool,
            row_ids=pool_ids,
            features=features,
            context=context,
            bank=bank,
            coverage=coverage,
            kind=data["target_kind"],
            target_variance=float(pool[:, -1].var(correction=0)),
        )
    return tables


def training_episode(name, table, spec, index):
    return sample_supervised_episode(
        table["pool"],
        table["features"],
        row_ids=table["row_ids"],
        context_size=table["context"],
        seed=spec["episode_seed"],
        namespace=f"{name}/training",
        episode_id=index,
    )


def fit_bank_diagnostics(table, smoothing):
    """Objective-only limits: hidden labels never enter a model forward here."""
    unsupported, total, majority_hits, oracle_hits, sse = 0, 0, 0, 0, 0.0
    for episode, truth, _ in table["bank"]:
        visible_labels = episode.values[episode.visible[:, -1], -1].tolist()
        counts = Counter(visible_labels)
        majority = counts.most_common(1)[0][0]
        groups = {}
        for (row, _), label in truth.items():
            total += 1
            unsupported += label not in counts
            majority_hits += label == majority
            groups.setdefault(tuple(episode.values[row, :-1].tolist()), []).append(label)
        for labels in groups.values():
            if table["kind"] == "numeric":
                mean = sum(labels) / len(labels)
                sse += sum((y - mean) ** 2 for y in labels)
            else:
                query_counts = Counter(labels)
                oracle_hits += max((n for y, n in query_counts.items() if y in counts), default=0)
    if table["kind"] == "numeric":
        return dict(
            query_count=total,
            duplicate_query_min_mse=sse / total,
            duplicate_query_min_nmse=sse / total / (table["target_variance"] + 1e-12),
        )
    classes = len(table["features"][-1].domain)
    floor_probability = smoothing / (1 + classes * smoothing)
    return dict(
        query_count=total,
        unsupported_queries=unsupported,
        query_label_seen_rate=1 - unsupported / total,
        unsupported_nll_floor=unsupported / total * -math.log(floor_probability),
        duplicate_and_support_accuracy_ceiling=oracle_hits / total,
        context_majority_accuracy=majority_hits / total,
    )


def save_joint_checkpoint(model, trainer, path, spec_hash, round_index):
    save_checkpoint(model, path, trainer=trainer)
    write_json(
        Path(path) / "joint-progress.json",
        dict(
            preregistration_sha256=spec_hash,
            runner_sha256=sha(__file__),
        git_source=git_source_state(),
            runner_source_sha256=runner_source_hash(),
            rounds=round_index,
            updates=trainer.step,
            model_state_sha256=state_hash(model),
            checkpoint_manifest_sha256=sha(Path(path) / "manifest.json"),
        ),
    )


@torch.no_grad()
def evaluate_all(model, tables):
    before = state_hash(model)
    metrics = {}
    for name, table in tables.items():
        losses, predictions, truth = [], [], []
        for ep, labels, _ in table["bank"]:
            out = model(ep)
            losses.append(float(score(out, labels)))
            predictions.extend(out.predictions)
            truth.extend(labels[p.address] for p in out.predictions)
        result = dict(loss=sum(losses) / len(losses))
        if table["kind"] == "numeric":
            mse = sum(
                (float(p.value) - y) ** 2 for p, y in zip(predictions, truth, strict=True)
            ) / len(truth)
            result.update(
                mse=mse,
                rmse=math.sqrt(mse),
                normalized_mse=mse / (table["target_variance"] + 1e-12),
            )
        else:
            result.update(
                nll=result["loss"],
                accuracy=sum(
                    int(p.probabilities.argmax()) == int(y)
                    for p, y in zip(predictions, truth, strict=True)
                )
                / len(truth),
            )
        metrics[name] = result
    if state_hash(model) != before:
        raise RuntimeError("evaluation mutated the shared model")
    return dict(model_state_sha256=before, datasets=metrics)


def run_joint_fit(args):
    prereg = Path(args.preregistration)
    spec = json.loads(prereg.read_text())
    if spec.get("schema") != "tabu.tar.joint-training-fit.1":
        raise ValueError("invalid joint training fit schema")
    mps_allowed = args.device == "mps" and spec.get("numerical_backend") == "experimental_fp32"
    if args.device != "cuda:0" and not mps_allowed and not (args.smoke and args.device == "cpu"):
        raise ValueError(
            "CUDA or explicit experimental_fp32 MPS required; CPU only supports smoke checks"
        )
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="local_unissued",
        outcome="started",
        preregistration_sha256=sha(prereg),
        runner_sha256=sha(__file__),
        git_source=git_source_state(),
        runner_source_sha256=runner_source_hash(),
        rounds=0,
        updates=0,
        evaluation_scope="training_masks_only",
    )
    write_json(root / "resolved.json", spec)
    start = time.monotonic()
    trainer = None
    periodic = []

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old_handler = signal.signal(signal.SIGTERM, interrupt)
    try:
        if args.device == "cuda:0":
            result["preflight"] = gpu_preflight()
            if not result["preflight"]["ready"]:
                result["outcome"] = "blocked_resources"
                return result
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
        if args.device == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS unavailable")
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
                raise RuntimeError("MPS requires explicit fallback=0")
            if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
                raise RuntimeError("MPS fast math must be disabled")
            result["mps_runtime"] = dict(
                macos=platform.mac_ver()[0],
                fallback="0",
                fast_math="0",
                numerical_backend="experimental_fp32",
            )
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        tables = load_tables(spec, prereg.parent)
        cfg, label = fit_model_config(spec, spec["initialization_seed"], smoke=args.smoke)
        model = TabUTARModel(cfg, device=args.device)
        initial_weights = spec.get("initial_weights")
        if initial_weights and not args.smoke:
            from safetensors.torch import load_file

            weights = prereg.parent / initial_weights["file"]
            if sha(weights) != initial_weights["file_sha256"]:
                raise ValueError("initial weights digest mismatch")
            model.load_state_dict(load_file(str(weights)), strict=True)
            if state_hash(model) != initial_weights["state_sha256"]:
                raise ValueError("initial model state differs from preregistration")
        rounds = 2 if args.smoke else spec["max_rounds"]
        training_config = TARTrainingConfig(
            learning_rate=spec["learning_rate"],
            final_learning_rate=spec["learning_rate"],
            warmup_steps=0,
            effective_episode_batch=1,
            optimizer_steps=rounds * len(tables),
        )
        start_round = 0
        resume = getattr(args, "resume_checkpoint", None)
        if resume:
            progress = json.loads((Path(resume) / "joint-progress.json").read_text())
            if (
                progress["preregistration_sha256"] != sha(prereg)
                or progress["runner_sha256"] != sha(__file__)
                or progress["runner_source_sha256"] != runner_source_hash()
                or progress["checkpoint_manifest_sha256"] != sha(Path(resume) / "manifest.json")
            ):
                raise ValueError("joint checkpoint recipe or source mismatch")
            model, trainer = load_checkpoint(
                resume, device=args.device, expected_config=cfg, restore_trainer=True
            )
            start_round = progress["rounds"]
            if (
                trainer.config != training_config
                or trainer.step != progress["updates"]
                or trainer.step != start_round * len(tables)
                or state_hash(model) != progress["model_state_sha256"]
            ):
                raise ValueError("joint checkpoint progress or optimizer mismatch")
            result["resumed_checkpoint_sha256"] = sha(Path(resume) / "manifest.json")
        else:
            trainer = TARTrainer(model, training_config)
        stop_round = getattr(args, "stop_after_round", None) or rounds
        if not start_round < stop_round <= rounds:
            raise ValueError("stop round must advance the checkpoint within the declared budget")
        result.update(start_round=start_round, rounds=start_round, updates=trainer.step)
        result.update(
            model_config=cfg.as_dict(),
            model_size=label,
            parameter_count=sum(p.numel() for p in model.parameters()),
            model_source_sha256=source_digest(),
            environment=dict(
                torch=torch.__version__, cuda=torch.version.cuda, architecture=platform.machine()
            ),
            initial=evaluate_all(model, tables),
            data_coverage={n: t["coverage"] for n, t in tables.items()},
            fit_diagnostics={
                n: fit_bank_diagnostics(t, cfg.category_smoothing_per_class)
                for n, t in tables.items()
            },
        )
        write_json(
            root / "evaluation-episodes.json",
            {n: [b[2] for b in t["bank"]] for n, t in tables.items()},
        )
        write_json(root / "initial.json", result["initial"])
        with (root / "curve.jsonl").open("x") as stream:
            for index in range(start_round, stop_round):
                if time.monotonic() - start >= spec["max_seconds"]:
                    result["outcome"] = "time_budget"
                    break
                for name in round_order(tables, spec["order_seed"], index):
                    episode, truth, trace = training_episode(name, tables[name], spec, index)
                    book = hashlib.sha256()
                    for col, value in sorted(model.compile_episode(episode)[4].items()):
                        book.update(str(col).encode())
                        book.update(value.detach().cpu().contiguous().numpy().tobytes())
                    digest = hashlib.sha256()
                    for tensor in (episode.values, episode.visible, episode.queries):
                        digest.update(tensor.cpu().contiguous().numpy().tobytes())
                    if args.device == "mps":
                        torch.mps.synchronize()
                    step_start = time.monotonic()
                    update = trainer.train_step([(episode, truth)])
                    if args.device == "mps":
                        torch.mps.synchronize()
                    update["train_seconds"] = time.monotonic() - step_start
                    if update["gradient_norm"] > spec["gradient_alarm_max"]:
                        raise FloatingPointError("gradient alarm")
                    update.update(
                        dataset=name,
                        round=index + 1,
                        episode=trace,
                        codebook_sha256=book.hexdigest(),
                        forward_input_sha256=digest.hexdigest(),
                    )
                    stream.write(json.dumps(update, allow_nan=False) + "\n")
                    stream.flush()
                    result["updates"] = trainer.step
                result["rounds"] = index + 1
                if (index + 1) % spec["evaluate_every"] == 0 or index + 1 == stop_round:
                    point = dict(
                        round=index + 1, updates=trainer.step, **evaluate_all(model, tables)
                    )
                    periodic.append(point)
                    with (root / "periodic-fit.jsonl").open("a") as out:
                        out.write(json.dumps(point, allow_nan=False) + "\n")
                    print(json.dumps(point), flush=True)
                    if spec.get("save_periodic_checkpoints", False):
                        save_joint_checkpoint(
                            model,
                            trainer,
                            root / f"checkpoint-round-{index + 1:06d}",
                            sha(prereg),
                            index + 1,
                        )
        result.update(final=evaluate_all(model, tables), periodic=periodic)
        result["outcome"] = (
            "smoke_completed"
            if args.smoke
            else "update_budget_completed"
            if result["rounds"] == rounds
            else "segment_completed"
            if result["rounds"] == stop_round
            else result["outcome"]
        )
        save_joint_checkpoint(model, trainer, root / "checkpoint", sha(prereg), result["rounds"])
        return result
    except KeyboardInterrupt:
        result["outcome"] = "interrupted"
        raise
    except Exception as exc:
        result.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        if trainer is not None:
            result["updates"] = trainer.step
            if args.device == "cuda:0":
                result["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if args.device == "mps" and torch.backends.mps.is_available():
            result["mps_current_allocated_gib"] = torch.mps.current_allocated_memory() / 2**30
            result["mps_driver_allocated_gib"] = torch.mps.driver_allocated_memory() / 2**30
        result["seconds"] = time.monotonic() - start
        write_json(root / "result.json", result)
        write_json(
            root / "checksums.json",
            {str(p.relative_to(root)): sha(p) for p in root.rglob("*") if p.is_file()},
        )
