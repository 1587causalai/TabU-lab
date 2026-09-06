"""Preregistered local TAR fitting with optional explicitly enabled observers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import time
import uuid
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_source_state(repository=None):
    """Record the local source commit without paths, remote URLs or formal issuance."""
    root = Path(repository) if repository is not None else Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return {"state": "unavailable", "commit": None}
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root,
            stderr=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"state": "unavailable", "commit": None}
    return {"state": "dirty" if dirty else "clean", "commit": head}


def gpu_preflight():
    """Check shared GB10 resources before importing/allocating model weights."""
    samples = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    if len(samples) != 1:
        raise RuntimeError("this bounded launcher expects one visible GPU")
    utilization, temperature = (int(v.strip()) for v in samples[0].split(","))
    processes = (
        subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        )
        .strip()
        .splitlines()
    )
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(memory["MemAvailable"].strip().split()[0]) / 1024**2
    result = dict(
        gpu_utilization_percent=utilization,
        temperature_c=temperature,
        compute_process_count=len(processes),
        available_memory_gib=available,
    )
    result["ready"] = not processes and utilization < 20 and temperature < 80 and available >= 16
    return result


def fit_model_config(spec, seed, *, smoke=False):
    """Resolve a named size with an explicit configuration identity."""
    from tabu_lab.models.tar import TARConfig
    from tabu_lab.tar_sizes import config_for_size

    if spec.get("model_overrides"):
        raise ValueError("choose a named model_size instead of unlabeled shape overrides")
    size = spec.get("model_size", "standard")
    cfg = config_for_size(size, initialization_seed=seed)
    if spec.get("expected_parameter_count", cfg.parameter_count) != cfg.parameter_count:
        raise ValueError("model size differs from preregistered parameter count")
    if smoke:
        return TARConfig(
            width=16,
            heads=4,
            ff_width=32,
            blocks=2,
            semantic_slots=3,
            inducing_slots=8,
            initialization_seed=seed,
        ), "cpu-smoke"
    return cfg, size


def run_fit(args):
    prereg_path = Path(args.preregistration)
    spec = json.loads(prereg_path.read_text())
    if spec.get("schema") != "tabu.tar.full-data-fit.1" or spec.get("model_id") != "tabu.tar":
        raise ValueError(
            "unsupported TAR fitting preregistration; "
            "older evaluation protocols require the archived source"
        )
    if args.seed not in spec["seeds"] or args.dataset not in spec["datasets"]:
        raise ValueError("dataset/seed not in preregistration")
    if not 0 < spec.get("mask_fraction", 0) < 1:
        raise ValueError("an explicit training label mask fraction is required")
    if spec.get("evaluation_mode") != "all_train_context_joint_test":
        raise ValueError("evaluation must use all train rows as context and joint test queries")
    if spec.get("protocol") != "resample_rows_and_nominal_codebooks_per_update":
        raise ValueError("explicit resampled episode protocol required")
    if spec.get("effective_episode_batch") != 1:
        raise ValueError("this bounded runner requires effective_episode_batch=1")
    if type(spec.get("fit_eval_episodes")) is not int or spec["fit_eval_episodes"] < 1:
        raise ValueError("fit evaluation bank must be nonempty")
    data_path = prereg_path.parent / "data" / f"{args.dataset}.json"
    if sha(data_path) != spec["datasets"][args.dataset]:
        raise ValueError("dataset snapshot digest mismatch")
    if args.device != "cuda:0" and not (args.smoke and args.device == "cpu"):
        raise ValueError("fit requires cuda:0; cpu is reserved for --smoke plumbing checks")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="local_unissued",
        outcome="started",
        dataset=args.dataset,
        seed=args.seed,
        preregistration_sha256=sha(prereg_path),
        dataset_sha256=sha(data_path),
        runner_sha256=sha(__file__),
        git_source=git_source_state(),
        device=args.device,
        smoke=bool(args.smoke),
        protocol=spec["protocol"],
        evaluation_mode=spec["evaluation_mode"],
    )
    (root / "resolved.json").write_text(
        json.dumps(dict(preregistration=spec, request=receipt), indent=2) + "\n"
    )
    from tabu_lab.observers import NullObserver

    observer = NullObserver()
    started = time.monotonic()
    deadline = float(os.environ.get("TABU_TAR_FIT_DEADLINE_UNIX", "inf"))
    deadline_file = root.parent / f"{root.name}.deadline.json"
    if deadline_file.exists():
        deadline = float(json.loads(deadline_file.read_text())["deadline_unix"])
    reserve = spec.get("finalization_reserve_seconds", 90)
    if math.isfinite(deadline):
        receipt["wall_deadline_unix"] = deadline
        receipt["finalization_reserve_seconds"] = reserve

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        from tabu_lab.tar_data import validate_full_dataset

        dataset = json.loads(data_path.read_text())
        coverage = validate_full_dataset(dataset, spec["expected_rows"][args.dataset])
        context_size = coverage["train_rows"] - math.ceil(
            coverage["train_rows"] * spec["mask_fraction"]
        )
        if context_size < 2:
            raise ValueError("mask fraction leaves insufficient context")
        coverage.update(
            training_context_rows=context_size,
            training_masked_labels=coverage["train_rows"] - context_size,
        )
        receipt["data_coverage"] = coverage
        (root / "data-coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
        if args.device == "cuda:0":
            receipt["preflight"] = gpu_preflight()
            if not receipt["preflight"]["ready"]:
                receipt["outcome"] = "blocked_resources"
                return receipt
        import numpy as np
        import torch

        from tabu_lab.models.tar import (
            TabUTARModel,
            TARConfig,
            TARFeature,
            TARTrainer,
            TARTrainingConfig,
            save_checkpoint,
            score,
        )
        from tabu_lab.models.tar.checkpoint import source_digest
        from tabu_lab.models.tar.episodes import (
            episode_seed,
            sample_supervised_episode,
            supervised_episode,
        )

        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no silent CPU fallback")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if args.device == "cuda:0":
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        cfg, profile = fit_model_config(spec, args.seed, smoke=args.smoke)
        receipt["model_profile"] = profile
        receipt["model_size"] = profile
        receipt["requested_model_size"] = spec.get("model_size", "standard")
        receipt["is_design_default_shape"] = cfg == TARConfig(initialization_seed=args.seed)
        model = TabUTARModel(cfg, device=args.device)
        values = torch.tensor(dataset["values"], dtype=torch.float64)
        parts = dataset["splits"]
        pool_ids = parts["train"]
        pool = values[pool_ids]
        holdout = values[parts["test"]]
        features = (
            *(TARFeature(column_id=a) for a in range(values.shape[1] - 1)),
            TARFeature(dataset["target_kind"], tuple(dataset["domain"]), values.shape[1] - 1),
        )

        # Training pool stays fixed; only episode roles and codebooks change.
        def sample(index, namespace):
            return sample_supervised_episode(
                pool,
                features,
                row_ids=pool_ids,
                context_size=context_size,
                seed=spec["episode_seed"],
                namespace=f"{args.dataset}/{namespace}",
                episode_id=index,
            )

        @torch.no_grad()
        def audit(item):
            episode, _, record = item
            # Hash actual compiler-produced vectors, not only RNG seed labels.
            books = model.compile_episode(episode)[4]
            digest = hashlib.sha256()
            for column, book in sorted(books.items()):
                digest.update(str(column).encode())
                digest.update(book.detach().cpu().contiguous().numpy().tobytes())
            record = dict(record, nominal_columns=sorted(books), codebook_sha256=digest.hexdigest())
            forward_digest = hashlib.sha256()
            for tensor in (episode.values, episode.visible, episode.queries):
                forward_digest.update(tensor.cpu().contiguous().numpy().tobytes())
            record["forward_input_sha256"] = forward_digest.hexdigest()
            return record

        fit_bank = [
            sample(i, "fit-evaluation")
            for i in range(2 if args.smoke else spec["fit_eval_episodes"])
        ]
        # The test context is the complete train split, not one sampled context.
        namespace = f"{args.dataset}/joint-test-evaluation"
        test_seed = episode_seed(spec["episode_seed"], namespace, 0)
        episode, truth = supervised_episode(pool, holdout, features, codebook_seed=test_seed)
        held_bank = [
            (
                episode,
                truth,
                dict(
                    namespace=namespace,
                    episode_id=0,
                    codebook_seed=test_seed,
                    context_row_ids=pool_ids,
                    query_row_ids=parts["test"],
                    evaluation_mode=spec["evaluation_mode"],
                ),
            )
        ]
        (root / "evaluation-episodes.json").write_text(
            json.dumps(
                {
                    "fit": [audit(item) for item in fit_bank],
                    "holdout": [audit(item) for item in held_bank],
                },
                indent=2,
            )
            + "\n"
        )
        # Reporting normalization only; model preprocessing remains episode-visible-only.
        scale = float(pool[:, -1].var(correction=0).sqrt())

        def metrics(predictions, targets):
            if dataset["target_kind"] == "numeric":
                pred = np.array([float(p.value.detach().cpu()) for p in predictions])
                mse = float(np.mean((pred - targets) ** 2))
                return dict(mse=mse, rmse=math.sqrt(mse), normalized_mse=mse / (scale**2 + 1e-12))
            probs = np.stack([p.probabilities.detach().cpu().numpy() for p in predictions])
            y = targets.astype(int)
            return dict(
                nll=float(-np.log(probs[np.arange(len(y)), y]).mean()),
                accuracy=float(np.mean(probs.argmax(-1) == y)),
            )

        @torch.no_grad()
        def evaluate(bank):
            predictions, targets, losses = [], [], []
            was_training = model.training
            model.eval()
            try:
                for episode, truth, _ in bank:
                    out = model(episode)
                    losses.append(float(score(out, truth)))
                    predictions.extend(out.predictions)
                    targets.extend(truth[p.address] for p in out.predictions)
                result = dict(
                    loss=float(np.mean(losses)), **metrics(predictions, np.asarray(targets))
                )
                if spec.get("monitor_terminal_weights"):
                    weights = torch.stack([p.weights for p in predictions])
                    result["support_max_weight_mean"] = float(weights.max(-1).values.mean())
                    result["support_entropy_mean"] = float(
                        -(weights * weights.clamp_min(1e-300).log()).sum(-1).mean()
                    )
                return result
            finally:
                model.train(was_training)

        def baseline(bank):
            errors, correct = [], []
            for episode, truth, _ in bank:
                y = episode.values[episode.visible[:, -1], -1]
                targets = torch.tensor(list(truth.values()), dtype=torch.float64)
                if dataset["target_kind"] == "numeric":
                    errors.extend(((targets - y.mean()) ** 2).tolist())
                else:
                    freq = torch.bincount(y.long(), minlength=len(dataset["domain"])).double()
                    freq = (freq + 1e-6) / (freq.sum() + len(freq) * 1e-6)
                    errors.extend((-freq[targets.long()].log()).tolist())
                    correct.extend((targets.long() == freq.argmax()).double().tolist())
            if dataset["target_kind"] == "numeric":
                return dict(mse=float(np.mean(errors)), rmse=math.sqrt(float(np.mean(errors))))
            return dict(nll=float(np.mean(errors)), accuracy=float(np.mean(correct)))

        receipt.update(
            config=cfg.as_dict(),
            model_source_sha256=source_digest(),
            model_config_sha256=hashlib.sha256(
                json.dumps(cfg.as_dict(), sort_keys=True).encode()
            ).hexdigest(),
            parameter_count=sum(p.numel() for p in model.parameters()),
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                architecture=platform.machine(),
            ),
            splits=parts,
            training_pool_row_ids=pool_ids,
            outer_split=dict(train_row_ids=pool_ids, test_row_ids=parts["test"]),
            test_context_rows=len(pool),
            test_query_rows=len(holdout),
            evaluation_bank_size=dict(fit=len(fit_bank), holdout=len(held_bank)),
            reporting_target_scale=scale,
            initial_fit=evaluate(fit_bank),
            initial_holdout=evaluate(held_bank),
        )
        receipt["constant_baseline"] = dict(fit=baseline(fit_bank), holdout=baseline(held_bank))
        from tabu_lab.observers import get_observer

        observer = get_observer(
            run_id=receipt["model_source_sha256"][:12],
            attempt_id=f"tar-{profile}-{args.dataset}-s{args.seed}-{uuid.uuid4().hex[:8]}",
            experiment_id=spec.get("observer_group", "tar-full-data-80-20"),
            contract_id="tabu.tar",
            seed=args.seed,
            stage="real-data-diagnostic",
            environment_payload=dict(
                host_class="gpu-experiment" if args.device == "cuda:0" else "cpu",
                architecture=platform.machine(),
                python_version=platform.python_version(),
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda,
                device=args.device,
                deterministic_algorithms=True,
            ),
        )
        # The observer link is a separate projection, not evidence identity or model metadata.
        if observer.run_url:
            mirror = root.parent / "observations"
            mirror.mkdir(exist_ok=True)
            (mirror / f"{root.name}.json").write_text(
                json.dumps(dict(run_url=observer.run_url), indent=2) + "\n"
            )
        observer.log_summary(
            dict(
                initial_fit=receipt["initial_fit"],
                initial_holdout=receipt["initial_holdout"],
                initial_objective=receipt["initial_fit"]["loss"],
                steps=0,
                verdict="started",
            )
        )
        steps = 2 if args.smoke else spec["max_updates"]
        trainer = TARTrainer(
            model,
            TARTrainingConfig(
                effective_episode_batch=1,
                learning_rate=spec["learning_rate"],
                final_learning_rate=spec["final_learning_rate"],
                warmup_steps=spec["warmup_steps"],
                optimizer_steps=steps,
            ),
        )
        train_start = time.monotonic()
        stop = "update_budget"
        seen_codebooks = set()
        periodic = []
        zero_gradient_streak = 0
        eval_every = spec.get("periodic_fit_every", 0)
        with (root / "curve.jsonl").open("x") as stream:
            for _ in range(steps):
                if (
                    time.monotonic() - train_start >= spec["max_train_seconds"]
                    or time.time() >= deadline - reserve
                ):
                    stop = "time_budget"
                    break
                item = sample(trainer.step, "training")
                record = audit(item)
                update = trainer.train_step([item[:2]])
                update["episode"] = record
                update["train_seconds"] = time.monotonic() - train_start
                stream.write(json.dumps(update, allow_nan=False) + "\n")
                stream.flush()
                seen_codebooks.add(record["codebook_seed"])
                fit_metrics = {}
                if eval_every and trainer.step % eval_every == 0:
                    current_fit = evaluate(fit_bank)
                    periodic.append(dict(step=trainer.step, fit=current_fit))
                    with (root / "periodic-fit.jsonl").open("a") as eval_stream:
                        eval_stream.write(json.dumps(periodic[-1], allow_nan=False) + "\n")
                    fit_metrics = {f"fit_{key}": value for key, value in current_fit.items()}
                observer.log_step(
                    dict(
                        step=trainer.step,
                        loss=update["loss"],
                        gradient_norm=update["gradient_norm"],
                        learning_rate=update["learning_rate"],
                        train_seconds=update["train_seconds"],
                        elapsed_seconds=time.monotonic() - started,
                        episode_index=record["episode_id"],
                        context_rows=len(record["context_row_ids"]),
                        query_rows=len(record["query_row_ids"]),
                        unique_codebooks=len(seen_codebooks),
                        updates_per_second=trainer.step / max(update["train_seconds"], 1e-9),
                        **fit_metrics,
                    )
                )
                if trainer.step == 1 or trainer.step % spec["log_every"] == 0:
                    print(
                        json.dumps(dict(dataset=args.dataset, seed=args.seed, **update)), flush=True
                    )
                zero_gradient_streak = (
                    zero_gradient_streak + 1
                    if update["gradient_norm"] == 0
                    and update["loss"] > 2 * receipt["initial_fit"]["loss"]
                    else 0
                )
                if update["gradient_norm"] > spec.get("gradient_alarm_max", math.inf):
                    stop = "gradient_alarm"
                    break
                if zero_gradient_streak >= spec.get("zero_gradient_patience", math.inf):
                    stop = "gradient_collapse_alarm"
                    break
        receipt.update(
            periodic_evaluations=periodic,
            updates=trainer.step,
            stop_reason=stop,
            train_seconds=time.monotonic() - train_start,
            final_fit=evaluate(fit_bank),
            final_holdout=evaluate(held_bank),
        )
        receipt["fit_loss_ratio"] = receipt["final_fit"]["loss"] / max(
            receipt["initial_fit"]["loss"], 1e-30
        )
        gates = spec["pass_criteria"]
        absolute = (
            receipt["final_fit"]["normalized_mse"] <= gates["diabetes_normalized_mse_max"]
            if dataset["target_kind"] == "numeric"
            else receipt["final_fit"]["accuracy"] >= gates["iris_accuracy_min"]
        )
        passed = absolute and receipt["fit_loss_ratio"] <= gates["fit_loss_ratio_max"]
        receipt["outcome"] = (
            "smoke_completed"
            if args.smoke
            else "stability_alarm"
            if stop in ("gradient_alarm", "gradient_collapse_alarm")
            else "resampled_fit_pass"
            if passed
            else "budget_limited"
            if stop == "time_budget"
            else "fit_gate_not_met"
        )
        if args.device == "cuda:0":
            receipt["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        save_checkpoint(model, root / "checkpoint", trainer=trainer)
        return receipt
    except KeyboardInterrupt as exc:
        receipt.update(outcome="interrupted", error_type=type(exc).__name__, error=str(exc))
        raise
    except Exception as exc:
        receipt.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        observer.log_summary(
            dict(
                initial_objective=receipt.get("initial_fit", {}).get("loss"),
                final_objective=receipt.get("final_fit", {}).get("loss"),
                loss_ratio=receipt.get("fit_loss_ratio"),
                steps=receipt.get("updates"),
                verdict=receipt["outcome"],
                **{
                    k: receipt[k]
                    for k in ("initial_fit", "final_fit", "initial_holdout", "final_holdout")
                    if k in receipt
                },
            )
        )
        observer.close()
        receipt["total_seconds"] = time.monotonic() - started
        (root / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        manifest = {
            str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()
        }
        (root / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n")
