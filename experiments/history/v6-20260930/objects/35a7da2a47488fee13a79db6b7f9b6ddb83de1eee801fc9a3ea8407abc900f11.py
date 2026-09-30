"""Bounded prepared-execution parity and timing, with no fitting campaign."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration import (
    RestorationModel,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration.end_to_end_checks import example_episode, small_config
from tabu_lab.restoration_pipeline_benchmark import compare_tensors


def _values(model, score):
    values = {"loss": score.loss.detach(), "per_target": score.per_target.detach()}
    values.update({f"encoding_{col.column}": col.result.encoding.detach()
                   for col in score.output.columns})
    values.update({f"gradient_{name}": None if p.grad is None else p.grad.detach().clone()
                   for name, p in model.named_parameters()})
    return values


def run_benchmark(args):
    """Compare fresh/full, prepared/full and prepared/training on one mixed episode."""
    if args.output.exists():
        raise FileExistsError("benchmark output already exists")
    result = {
        "schema": "tabu.restoration.prepared-benchmark.v1",
        "status": "local_unissued", "outcome": "failed", "device": args.device,
        "scope": (
            "6-row mixed damage episode; FP64; "
            "no optimizer or checkpoint I/O"
        ),
        "torch_version": torch.__version__, "threads": 1,
        "warmup_calls": 2, "rounds": 4, "calls_per_round": 6,
    }
    root = Path(__file__).parent
    paths = [Path(__file__), root / "restoration_pipeline_benchmark.py"]
    paths += sorted((root / "models/restoration").glob("*.py"))
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    old_threads = torch.get_num_threads()
    try:
        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device is unavailable")
        torch.set_num_threads(1)
        devices = [0] if args.device == "cuda:0" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(918)
            model = RestorationModel(small_config()).to(device=args.device, dtype=torch.float64)
            inputs, request, truth = example_episode(damage=True)
            # Use the typed constructor once; never pass clean truth to the model.
            from tabu_lab.models.restoration import (
                RestorationInput,
                RestorationRequest,
                TruthSidecar,
            )
            episode = (
                RestorationInput(inputs.schema, tuple(v.to(args.device) for v in inputs.values),
                                 inputs.visible.to(args.device), inputs.query.to(args.device),
                                 inputs.code_seed),
                RestorationRequest(request.targets.to(args.device)),
                TruthSidecar(tuple(v.to(args.device) for v in truth.values),
                             truth.states.to(args.device)),
            )

            def sync():
                if devices:
                    torch.cuda.synchronize()

            sync()
            tick = time.perf_counter()
            prepared = prepare_episode(model, *episode)
            sync()
            result["prepare_seconds"] = time.perf_counter() - tick
            scorers = {
                "fresh_full": lambda: score_episode(model, *episode),
                "prepared_full": lambda: score_prepared_episode(
                    model, prepared, decode=True, report=True,
                ),
                "prepared_training": lambda: score_prepared_episode(model, prepared),
            }
            expected = None
            differences = {}
            for name, scorer in scorers.items():
                model.zero_grad(set_to_none=True)
                score = scorer()
                score.loss.backward()
                actual = _values(model, score)
                if expected is None:
                    expected = actual
                differences[name] = compare_tensors(expected, actual)
            result["max_abs_differences"] = differences
            result["config"] = model.config.as_dict()
            result["measurements"] = []

            def step(scorer):
                model.zero_grad(set_to_none=True)
                scorer().loss.backward()

            for scorer in scorers.values():
                for _ in range(result["warmup_calls"]):
                    step(scorer)
            for round_index in range(result["rounds"]):
                names = list(scorers) if round_index % 2 == 0 else list(reversed(scorers))
                for name in names:
                    sync()
                    tick = time.perf_counter()
                    for _ in range(result["calls_per_round"]):
                        step(scorers[name])
                    sync()
                    result["measurements"].append({
                        "variant": name, "round": round_index,
                        "seconds_per_call": (
                            (time.perf_counter() - tick) / result["calls_per_round"]
                        ),
                    })
            result["median_seconds"] = {
                name: statistics.median(m["seconds_per_call"] for m in result["measurements"]
                                        if m["variant"] == name)
                for name in scorers
            }
            result["outcome"] = "passed"
    except (AssertionError, RuntimeError, ValueError, FloatingPointError) as error:
        result["error_type"] = type(error).__name__
    finally:
        torch.set_num_threads(old_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["outcome"] == "passed" else 1
