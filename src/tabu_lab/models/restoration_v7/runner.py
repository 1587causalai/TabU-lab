"""Data-agnostic V7 update step and identity-bound checkpoints.

The table set, split, episode order, budget and checkpoint selection belong
to a RunSpec supplied by the caller; this module only fixes how one optimizer
step reduces its episodes and what a checkpoint must bind.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..restoration.contracts import RestorationInput, TruthSidecar
from .config import V7Config
from .model import V7Model, prepare_episode
from .training import reference_values, score_rounds

CHECKPOINT_SCHEMA = "tabu.restoration.v7-checkpoint.v1"


@dataclass(frozen=True)
class V7Task:
    """One admitted episode: visible inputs, isolated truth, recorded donor seed."""

    inputs: RestorationInput
    truth: TruthSidecar
    donor_seed: int


@dataclass(frozen=True)
class OptimizerSpec:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    def __post_init__(self):
        for name in ("lr", "eps", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"optimizer.{name} must be finite and nonnegative")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("optimizer.betas must contain two coefficients")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value < 1
            for value in self.betas
        ):
            raise ValueError("optimizer.betas coefficients must be finite and lie in [0, 1)")
        object.__setattr__(self, "betas", tuple(self.betas))


@dataclass(frozen=True)
class StepRecord:
    loss: float  # task-equal mean of the per-episode weighted round sums
    round_losses: tuple[tuple[float, ...], ...]  # [B_ep][K]
    initial_losses: tuple[float, ...]  # L_0 per episode; diagnostic only


def make_optimizer(model: V7Model, spec: OptimizerSpec | None = None) -> torch.optim.AdamW:
    spec = spec or OptimizerSpec()
    return torch.optim.AdamW(model.parameters(), **asdict(spec))


def train_step(
    model: V7Model,
    optimizer: torch.optim.Optimizer,
    tasks: Sequence[V7Task],
    *,
    grad_clip_norm: float | None = None,
) -> StepRecord:
    """One backward and update across single-column and/or joint-column tasks.

    Cyclic configurations backpropagate through the complete unroll.

    Each episode is first averaged over its own Query rows, then episodes are
    averaged with equal weight; a nonfinite loss or gradient aborts before the
    parameters change.
    """
    if not tasks:
        raise ValueError("a V7 optimizer step needs at least one episode")
    if grad_clip_norm is not None and (not math.isfinite(grad_clip_norm) or grad_clip_norm <= 0):
        raise ValueError("grad_clip_norm must be positive and finite")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scores = []
    for task in tasks:
        if int(task.inputs.query.any(0).sum()) > 1:
            from .joint import prepare_joint_episode, score_joint

            episode = prepare_joint_episode(
                task.inputs, donor_seed=task.donor_seed, config=model.config
            )
            scores.append(
                score_joint(model(episode, decode=False), episode, task.truth, model.config)
            )
            continue
        episode = prepare_episode(
            task.inputs,
            donor_seed=task.donor_seed,
            code_dim=model.config.code_dim,
            epsilon=model.config.epsilon,
            codec=model.config.codec,
        )
        output = model(episode, decode=False)
        scores.append(
            score_rounds(output, episode, reference_values(episode, task.truth), model.config)
        )
    loss = torch.stack([score.loss for score in scores]).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads and not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):
        raise FloatingPointError("nonfinite: V7 parameter gradient")
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, error_if_nonfinite=True)
    optimizer.step()
    return StepRecord(
        float(loss.detach()),
        tuple(tuple(score.round_losses.detach().tolist()) for score in scores),
        tuple(float(score.initial_loss) for score in scores),
    )


def manifest_digest(manifest: dict) -> str:
    text = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _cpu_copy(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return value


def _finite(value) -> bool:
    if isinstance(value, Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _execution(model: V7Model) -> dict:
    weight = model.rounds[0].lift.weight
    return {
        "device": weight.device.type,
        "dtype": str(weight.dtype),
        "torch": str(torch.__version__),
    }


def checkpoint_state(
    model: V7Model, optimizer: torch.optim.Optimizer, *, step: int, manifest: dict
) -> dict:
    """Snapshot weights, optimizer and run identity independently of subsequent updates.

    ``manifest`` is the caller's data/split/sampling/evaluation-bank record.
    """
    if type(step) is not int or step < 0:
        raise ValueError("step must be a non-negative integer")
    device = next(model.parameters()).device
    accelerator_rng = None
    if device.type == "cuda":
        accelerator_rng = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        accelerator_rng = torch.mps.get_rng_state()
    return _cpu_copy(
        {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": "restoration_v7",
            "share_rounds": model.config.share_rounds,
            "config": model.config.as_dict(),
            "manifest": manifest,
            "manifest_sha256": manifest_digest(manifest),
            "execution": _execution(model),
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_cpu_rng": torch.get_rng_state(),
            "accelerator_rng": accelerator_rng,
        }
    )


def save_checkpoint(path: str | Path, state: dict) -> None:
    """Write once; an existing file is never overwritten."""
    with Path(path).open("xb") as handle:
        torch.save(_cpu_copy(state), handle)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(
    path: str | Path,
    model: V7Model,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    manifest: dict,
) -> int:
    """Strict resume: schema, ModelSpec, manifest and execution must all match."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("architecture") != "restoration_v7":
        raise ValueError("unsupported V7 checkpoint schema")
    if V7Config.from_dict(state["config"]) != model.config:
        raise ValueError("checkpoint ModelSpec or sharing mode differs from the model")
    if not isinstance(state.get("manifest"), dict) or (
        state.get("manifest_sha256") != manifest_digest(state["manifest"])
    ):
        raise ValueError("checkpoint stored manifest differs from its digest")
    if state.get("manifest_sha256") != manifest_digest(manifest):
        raise ValueError("checkpoint run manifest differs from the declared manifest")
    if state.get("execution") != _execution(model):
        raise ValueError("checkpoint device/dtype/torch identity differs from the model")
    if not _finite(state["model"]) or not _finite(state.get("optimizer", {})):
        raise ValueError("checkpoint model or optimizer contains nonfinite state")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    accelerator_rng = state.get("accelerator_rng")
    if accelerator_rng is not None:
        device = next(model.parameters()).device
        if device.type == "cuda":
            torch.cuda.set_rng_state(accelerator_rng, device)
        elif device.type == "mps":
            torch.mps.set_rng_state(accelerator_rng)
    return int(state["step"])


__all__ = [
    "CHECKPOINT_SCHEMA",
    "OptimizerSpec",
    "StepRecord",
    "V7Task",
    "checkpoint_state",
    "load_checkpoint",
    "make_optimizer",
    "manifest_digest",
    "save_checkpoint",
    "train_step",
]
