"""Bounded FP64 CUDA parity and checkpoint probes, without fitting or fallback."""

from __future__ import annotations

import copy
import hashlib
import io
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from .backbone import BackboneConfig
from .end_to_end_checks import example_episode
from .model import RestorationConfig, RestorationModel
from .training import score_episode

RTOL, ATOL = 1e-7, 1e-8


def _episode_to(episode, device):
    inputs, request, truth = episode
    return (
        replace(inputs, values=tuple(v.to(device) for v in inputs.values),
                visible=inputs.visible.to(device), query=inputs.query.to(device)),
        replace(request, targets=request.targets.to(device)),
        replace(truth, values=tuple(v.to(device) for v in truth.values),
                states=truth.states.to(device)),
    )


def _compare(name, pairs, *, exact=False):
    differences, mismatches = {}, []
    for key, left, right in pairs:
        if left is None or right is None:
            if left is not None or right is not None:
                mismatches.append(key)
            continue
        left, right = left.detach().cpu(), right.detach().cpu()
        finite = bool(torch.isfinite(left).all() & torch.isfinite(right).all())
        same_shape = left.shape == right.shape
        differences[key] = (
            float((left.double() - right.double()).abs().max()) if finite and same_shape else None
        )
        if not finite or not same_shape or not torch.allclose(
            left, right, rtol=0 if exact else RTOL, atol=0 if exact else ATOL,
        ):
            mismatches.append(key)
    return {"name": name, "outcome": "failed" if mismatches else "passed",
            "comparison": "exact" if exact else "allclose",
            "max_absolute_differences": differences, "mismatches": mismatches}


def _score_pairs(left, right):
    yield "loss", left.loss, right.loss
    yield "per_target_loss", left.per_target, right.per_target
    for a, b in zip(left.output.columns, right.output.columns, strict=True):
        yield f"column_{a.column}_encoding", a.result.encoding, b.result.encoding
        yield f"column_{a.column}_decoded", a.decoded, b.decoded


def _update(model, optimizer, episode):
    optimizer.zero_grad(set_to_none=True)
    score = score_episode(model, *episode)
    score.loss.backward()
    optimizer.step()
    return score


def _probes(device, checks):
    config = RestorationConfig(backbone=BackboneConfig(layers=1, heads=4, ff_width=32, slots=2))
    torch.random.default_generator.manual_seed(101)
    torch.cuda.default_generators[device.index].manual_seed(101)
    cpu = RestorationModel(config).double()
    gpu = copy.deepcopy(cpu).to(device)
    episode = example_episode(damage=True)
    cuda_episode = _episode_to(episode, device)
    left, right = score_episode(cpu, *episode), score_episode(gpu, *cuda_episode)
    left.loss.backward()
    right.loss.backward()
    pairs = list(_score_pairs(left, right))
    pairs.extend((f"gradient_{name}", p.grad, dict(gpu.named_parameters())[name].grad)
                 for name, p in cpu.named_parameters())
    checks.append(_compare("cpu_cuda_forward_backward", pairs))

    gpu = copy.deepcopy(cpu).to(device)
    optimizer = torch.optim.AdamW(gpu.parameters(), lr=1e-4, foreach=False)
    _update(gpu, optimizer, cuda_episode)
    buffer = io.BytesIO()
    torch.save({"config": config.as_dict(), "model": gpu.state_dict(),
                "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device)}, buffer)
    left = _update(gpu, optimizer, cuda_episode)
    left_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed = RestorationModel(RestorationConfig.from_dict(saved["config"])).double().to(device)
    resumed.load_state_dict(saved["model"], strict=True)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-4, foreach=False)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"].cpu())
    torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
    right = _update(resumed, resumed_optimizer, cuda_episode)
    pairs = list(_score_pairs(left, right))
    pairs.extend((f"model_{k}", v, resumed.state_dict()[k]) for k, v in gpu.state_dict().items())
    resumed_state = resumed_optimizer.state_dict()["state"]
    for key, state in optimizer.state_dict()["state"].items():
        pairs.extend((f"optimizer_{key}_{k}", v, resumed_state[key][k]) for k, v in state.items())
    pairs.extend((("cpu_rng", left_rng[0], torch.get_rng_state()),
                  ("cuda_rng", left_rng[1], torch.cuda.get_rng_state(device))))
    check = _compare("two_update_checkpoint_continuation", pairs, exact=True)
    check.update(updates=2, checkpoint="BytesIO", restored=("model", "optimizer", "CPU/CUDA RNG"))
    checks.append(check)
    return config.as_dict()


def verify_device(device="cuda:0"):
    """Return a sanitized failure record when CUDA or any required check fails."""
    started = time.perf_counter()
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    result = {"schema_version": "tabu.restoration.device-check.v1", "status": "local_unissued",
              "scope": "bounded device correctness; not fit, benchmark, or issued evidence",
              "outcome": "failed", "device": device, "dtype": "float64",
              "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
              "component_source_sha256": digest.hexdigest(), "checks": [],
              "tolerance": {"rtol": RTOL, "atol": ATOL}, "peak_allocated_bytes": None}
    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    stage, ready = "cuda_availability", False
    try:
        if device != "cuda:0":
            raise ValueError("unsupported device")
        if torch.cuda.is_initialized():
            stage = "cuda_already_initialized"
            raise RuntimeError("CUDA must initialize inside the verification entry point")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if not torch.cuda.is_available():
            result["error"] = {"category": "cuda_unavailable", "stage": stage}
            return result
        stage = "cuda_setup"
        target = torch.device(device)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(target)
        torch.cuda.reset_peak_memory_stats(target)
        ready = True
        result["cuda_name"] = torch.cuda.get_device_name(target)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(target))
        result["determinism"] = {"algorithms": True, "cpu_threads": 1,
                                 "cublas_workspace_config": ":4096:8", "allow_tf32": False}
        stage = "model_probes"
        with torch.random.fork_rng(devices=[target.index]):
            result["config"] = _probes(target, result["checks"])
        torch.cuda.synchronize(target)
        result["outcome"] = (
            "passed" if all(c["outcome"] == "passed" for c in result["checks"]) else "failed"
        )
        if result["outcome"] == "failed":
            result["error"] = {"category": "comparison_mismatch", "stage": stage}
    except Exception as error:
        result["error"] = {"category": type(error).__name__, "stage": stage}
    finally:
        if ready:
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        result["elapsed_seconds"] = time.perf_counter() - started
    return result
