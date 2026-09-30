"""Paired execution-only comparison with the pre-vectorization implementation."""

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

VARIANTS = ("serial", "batched_backbone", "batched_readout", "batched_both")


def load_baseline(root):
    modules, hashes = {}, {}
    for name in ("backbone", "model"):
        path = f"src/tabu_lab/models/restoration/{name}.py"
        code = subprocess.check_output(["git", "show", f"4806471:{path}"], cwd=root)
        hashes[path] = hashlib.sha256(code).hexdigest()
        qualified = f"tabu_lab.models.restoration._benchmark_serial_{name}"
        module = types.ModuleType(qualified)
        module.__package__ = "tabu_lab.models.restoration"
        sys.modules[qualified] = module
        exec(compile(code, f"4806471/{path}", "exec"), module.__dict__)
        modules[name] = module
    return modules, hashes


def variants(config, root, device):
    old, hashes = load_baseline(root)
    torch.manual_seed(1729)
    reference = RestorationModel(config).double().to(device)
    result = {}
    for name in VARIANTS:
        model = copy.deepcopy(reference)
        if name in ("serial", "batched_readout"):
            backbone = old["backbone"].AxialBackbone(config.backbone).double().to(device)
            backbone.load_state_dict(reference.backbone.state_dict(), strict=True)
            model.backbone = backbone
        if name in ("serial", "batched_backbone"):
            model._forward_prepared = types.MethodType(
                old["model"].RestorationModel._forward_prepared, model
            )
        result[name] = model
    return result, hashes


def parity(models, episode):
    reference, checks = None, []
    for name, model in models.items():
        model.zero_grad(set_to_none=True)
        score = score_episode(model, *episode)
        score.loss.backward()
        tensors = {"loss": score.loss.detach().cpu()}
        tensors.update({f"encoding_{c.column}": c.result.encoding.detach().cpu()
                        for c in score.output.columns})
        tensors.update({f"gradient_{k}": p.grad.detach().cpu()
                        for k, p in model.named_parameters() if p.grad is not None})
        if reference is None:
            reference = tensors
        if tensors.keys() != reference.keys():
            raise AssertionError("gradient/output keys differ")
        differences = {}
        for key, value in tensors.items():
            torch.testing.assert_close(value, reference[key], rtol=1e-7, atol=1e-8)
            differences[key] = float((value - reference[key]).abs().max())
        checks.append({"variant": name, "max_absolute_difference": max(differences.values())})
        model.zero_grad(set_to_none=True)
    return checks


def run_benchmark(args):
    if args.output.exists():
        raise FileExistsError("benchmark output already exists")
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    result = {"status": "local_unissued", "outcome": "failed",
              "scope": "same-input same-weight execution timing; no optimizer or checkpoint I/O",
              "device": args.device, "dtype": "float64", "rounds": 3,
              "iterations_per_round": 8, "warmup_per_variant_per_phase": 2,
              "baseline_ref": "4806471", "torch": str(torch.__version__),
              "benchmark_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cuda_runtime": torch.version.cuda, "measurements": [],
              "timing": "synchronized wall time; includes encoding, validation and scorer"}
    try:
        _configure_backend()
        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if args.device == "cuda:0":
            result["device_name"] = torch.cuda.get_device_name(0)
        result["source_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        plan = prepare_plan(args.preregistration, args.dataset, args.device)
        result["identity"] = plan.identity
        result["config"] = plan.config.as_dict()
        models, result["baseline_source_hashes"] = variants(plan.config, root, args.device)
        episodes = [plan.episode(i, args.device) for i in range(len(plan.queries))]
        result["parity"] = [parity(models, episode) for episode in episodes]
        result["parity_tolerance"] = {"rtol": 1e-7, "atol": 1e-8}

        def synchronize():
            if args.device == "cuda:0":
                torch.cuda.synchronize()

        def step(model, episode, backward):
            model.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(backward):
                score = score_episode(model, *episode)
                if backward:
                    score.loss.backward()

        for phase in ("forward_no_grad", "forward_backward"):
            backward = phase == "forward_backward"
            for model in models.values():
                for i in range(2):
                    step(model, episodes[i % len(episodes)], backward)
            synchronize()
            for round_index in range(3):
                # Rotate order to distribute thermal/clock and ordering effects.
                order = VARIANTS[round_index:] + VARIANTS[:round_index]
                for name in order:
                    if time.perf_counter() - started > 180:
                        raise TimeoutError("bounded benchmark wall limit")
                    model = models[name]
                    synchronize()
                    tick = time.perf_counter()
                    for i in range(8):
                        step(model, episodes[i % len(episodes)], backward)
                    synchronize()
                    result["measurements"].append({
                        "phase": phase, "round": round_index, "variant": name,
                        "seconds_per_call": (time.perf_counter() - tick) / 8,
                    })
        result["summary"] = {}
        for phase in ("forward_no_grad", "forward_backward"):
            summary = {}
            for name in VARIANTS:
                values = [x["seconds_per_call"] for x in result["measurements"]
                          if x["phase"] == phase and x["variant"] == name]
                summary[name] = {"median_seconds": statistics.median(values),
                                 "min_seconds": min(values), "max_seconds": max(values)}
            for value in summary.values():
                value["speedup_vs_serial"] = (
                    summary["serial"]["median_seconds"] / value["median_seconds"]
                )
            result["summary"][phase] = summary
        result["outcome"] = "completed"
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(
        {k: v for k, v in result.items() if k not in ("identity", "parity")}, indent=2
    ))
    return 0 if result["outcome"] == "completed" else 1
