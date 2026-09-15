"""Frozen whole-pipeline equivalence and paired timing for parallelization changes."""

from __future__ import annotations

import copy
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

from tabu_lab.models.restoration import RestorationModel, score_episode
from tabu_lab.restoration_fit import _configure_backend, prepare_plan

REFERENCE = "4e2c04e"
RTOL, ATOL = 1e-7, 1e-8


def load_reference(root):
    """Load complete frozen encoder/backbone/readout/scorer, never current imports."""
    package_name = "tabu_lab.models._parallel_v2_reference"
    if package_name in sys.modules:
        return sys.modules[package_name]
    package = types.ModuleType(package_name)
    package.__package__ = package_name
    package.__path__ = []
    package.source_hashes = {}
    sys.modules[package_name] = package
    names = ("_validation", "answers", "contracts", "readout", "backbone", "encoding",
             "model", "losses", "training", "__init__")
    for name in names:
        path = f"src/tabu_lab/models/restoration/{name}.py"
        code = subprocess.check_output(["git", "show", f"{REFERENCE}:{path}"], cwd=root)
        package.source_hashes[path] = hashlib.sha256(code).hexdigest()
        qualified = package_name if name == "__init__" else f"{package_name}.{name}"
        module = package if name == "__init__" else types.ModuleType(qualified)
        module.__package__ = package_name
        sys.modules[qualified] = module
        exec(compile(code, f"{REFERENCE}/{path}", "exec"), module.__dict__)
    return package


def make_pair(config, device, root):
    reference = load_reference(root)
    torch.manual_seed(1729)
    current = RestorationModel(config).double().to(device)
    previous = reference.RestorationModel(
        reference.RestorationConfig.from_dict(config.as_dict())
    ).double().to(device)
    previous.load_state_dict(current.state_dict(), strict=True)
    return {"previous_batched": (previous, reference.score_episode),
            "parallel_v2": (current, score_episode)}


def compare_tensors(left, right):
    if left.keys() != right.keys():
        raise AssertionError("parameter/output keys differ")
    maximum = 0.0
    for key, a in left.items():
        b = right[key]
        if a is None or b is None:
            if a is not None or b is not None:
                raise AssertionError("gradient presence differs")
            continue
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL, equal_nan=False)
        maximum = max(maximum, float((a.detach() - b.detach()).abs().max()))
    return maximum


def check_pair(pair, episode, loss_config=None):
    results = []
    for model, scorer in pair.values():
        model.zero_grad(set_to_none=True)
        score = scorer(model, *episode, loss_config=loss_config)
        score.loss.backward()
        values = {"loss": score.loss.detach(), "per_target": score.per_target.detach()}
        for c in score.output.columns:
            values[f"encoding_{c.column}"] = c.result.encoding.detach()
            values[f"decoded_{c.column}"] = c.decoded.detach()
        values.update({f"gradient_{k}": None if p.grad is None else p.grad.detach().clone()
                       for k, p in model.named_parameters()})
        for state, metrics in score.by_state.items():
            for key, value in metrics.items():
                values[f"report_{state}_{key}"] = torch.as_tensor(value, dtype=torch.float64)
        results.append(values)
        model.zero_grad(set_to_none=True)
    return compare_tensors(*results)


def check_updates(pair, episodes, optimizer_config):
    initial = {name: copy.deepcopy(model.state_dict()) for name, (model, _) in pair.items()}
    optimizers = {name: torch.optim.AdamW(
        model.parameters(), lr=optimizer_config["learning_rate"],
        betas=tuple(optimizer_config["betas"]), eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
    ) for name, (model, _) in pair.items()}
    differences = []
    for step in range(3):
        results = []
        for name, (model, scorer) in pair.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            score = scorer(model, *episodes[step % len(episodes)])
            score.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimizer_config["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            values = {f"model_{k}": v for k, v in model.state_dict().items()}
            values.update({f"optimizer_{i}_{k}": v
                           for i, state in optimizer.state_dict()["state"].items()
                           for k, v in state.items()})
            results.append(values)
        differences.append(compare_tensors(*results))
    for name, (model, _) in pair.items():
        model.load_state_dict(initial[name], strict=True)
        model.zero_grad(set_to_none=True)
    return differences


def run_benchmark(args):
    if args.output.exists():
        raise FileExistsError("benchmark output already exists")
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    result = {"outcome": "failed", "status": "local_unissued", "reference": REFERENCE,
              "scope": "full pipeline parity and compute timing; no training campaign",
              "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "rtol": RTOL, "atol": ATOL, "device": args.device, "dtype": "float64",
              "rounds": 4, "calls_per_round": 8, "warmup_calls": 4,
              "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "measurements": []}
    stage = "setup"
    try:
        _configure_backend()
        if args.device == "cuda:0":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            result["device_name"] = torch.cuda.get_device_name(0)
        result["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        plan = prepare_plan(args.preregistration, args.dataset, args.device)
        result["identity"] = plan.identity
        result["config"] = plan.config.as_dict()
        pair = make_pair(plan.config, args.device, root)
        result["reference_source_hashes"] = load_reference(root).source_hashes
        episodes = [plan.episode(i, args.device) for i in range(len(plan.queries))]
        stage = "forward_backward_parity"
        result["parity_max_absolute_differences"] = [check_pair(pair, e) for e in episodes]
        stage = "optimizer_parity"
        result["three_update_differences"] = check_updates(pair, episodes, plan.spec["optimizer"])

        def synchronize():
            if args.device == "cuda:0":
                torch.cuda.synchronize()

        def step(model, scorer, episode, backward):
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(backward):
                score = scorer(model, *episode)
                if backward:
                    score.loss.backward()

        stage = "timing"
        for phase in ("forward_no_grad", "forward_backward"):
            backward = phase == "forward_backward"
            for model, scorer in pair.values():
                for episode in episodes:
                    step(model, scorer, episode, backward)
            for round_index in range(4):
                names = list(pair) if round_index % 2 == 0 else list(reversed(pair))
                for name in names:
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("benchmark wall limit")
                    model, scorer = pair[name]
                    synchronize()
                    tick = time.perf_counter()
                    for i in range(8):
                        step(model, scorer, episodes[i % len(episodes)], backward)
                    synchronize()
                    result["measurements"].append({
                        "variant": name, "phase": phase, "round": round_index,
                        "seconds_per_call": (time.perf_counter() - tick) / 8,
                    })
        result["summary"] = {}
        for phase in ("forward_no_grad", "forward_backward"):
            summary = {}
            for name in pair:
                values = [x["seconds_per_call"] for x in result["measurements"]
                          if x["variant"] == name and x["phase"] == phase]
                summary[name] = {"median_seconds": statistics.median(values),
                                 "min_seconds": min(values), "max_seconds": max(values)}
            result["summary"][phase] = summary
        result["outcome"] = "completed"
    except Exception as error:
        result["error"] = {"stage": stage, "type": type(error).__name__}
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "identity"}, indent=2))
    return 0 if result["outcome"] == "completed" else 1
