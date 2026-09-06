"""Masked-cell objective and optimizer steps, separate from truth-free forward."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def score(output, truth):
    """Truth is an address -> scalar mapping, read only after predictions exist."""
    numeric, discrete = [], []
    if not output.predictions:
        raise ValueError("no-valid-episode: empty query set")
    for pred in output.predictions:
        if pred.status != "ok":
            raise ValueError("no-valid-episode: cannot silently discard an unsupported target")
        if pred.address not in truth:
            raise ValueError("truth sidecar is missing a query")
        target = torch.as_tensor(truth[pred.address], device=pred.value.device, dtype=torch.float64)
        if target.numel() != 1 or not bool(torch.isfinite(target)):
            raise ValueError("invalid truth scalar")
        if pred.probabilities is None:
            numeric.append(0.5 * ((pred.value - target) / pred.scale).square())
        else:
            if float(target) != int(target) or not 0 <= int(target) < len(pred.probabilities):
                raise ValueError("out-of-domain: query truth is outside declared classes")
            discrete.append(-pred.probabilities[int(target)].log())
    branches = [torch.stack(xs).mean() for xs in (numeric, discrete) if xs]
    return torch.stack(branches).sum()


@dataclass(frozen=True)
class TARTrainingConfig:
    learning_rate: float = 1e-4
    final_learning_rate: float = 1e-5
    adam_betas: tuple[float, float] = (0.9, 0.95)
    adam_epsilon: float = 1e-8
    matrix_weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    effective_episode_batch: int = 16
    warmup_steps: int = 2000
    optimizer_steps: int = 100000

    def __post_init__(self):
        for name in ("effective_episode_batch", "warmup_steps", "optimizer_steps"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.warmup_steps < self.optimizer_steps:
            raise ValueError("warmup must be nonnegative and below total steps")
        if self.effective_episode_batch < 1:
            raise ValueError("effective_episode_batch must be positive")
        for v in (
            self.learning_rate,
            self.final_learning_rate,
            self.adam_epsilon,
            self.gradient_clip_norm,
        ):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("training scales must be finite and positive")
        if self.matrix_weight_decay < 0 or not math.isfinite(self.matrix_weight_decay):
            raise ValueError("invalid weight decay")
        if len(self.adam_betas) != 2 or not all(0 <= v < 1 for v in self.adam_betas):
            raise ValueError("invalid Adam betas")


class TARTrainer:
    """One explicit optimizer update. Experiment authorization belongs to the runner."""

    def __init__(self, model, config=None):
        self.model = model
        self.config = cfg = config or TARTrainingConfig()
        decay = []
        other = []
        for name, p in model.named_parameters():
            matrix = name in ("continuous", "response_cell") or name.endswith(".weight")
            (decay if matrix else other).append(p)
        self.optimizer = torch.optim.AdamW(
            [
                dict(params=decay, weight_decay=cfg.matrix_weight_decay),
                dict(params=other, weight_decay=0.0),
            ],
            lr=cfg.learning_rate,
            betas=cfg.adam_betas,
            eps=cfg.adam_epsilon,
        )
        self.step = 0

    def learning_rate(self, t):
        c = self.config
        if t <= c.warmup_steps:
            return c.learning_rate * t / c.warmup_steps
        fraction = (t - c.warmup_steps) / (c.optimizer_steps - c.warmup_steps)
        return c.final_learning_rate + 0.5 * (c.learning_rate - c.final_learning_rate) * (
            1 + math.cos(math.pi * fraction)
        )

    def train_step(self, episodes_and_truth):
        items = list(episodes_and_truth)
        cfg = self.config
        if len(items) != cfg.effective_episode_batch:
            raise ValueError("batch size must equal explicit effective_episode_batch")
        if self.step >= cfg.optimizer_steps:
            raise ValueError("declared optimizer budget exhausted")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total = 0.0
        try:
            for episode, truth in items:
                loss = score(self.model(episode), truth) / len(items)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("numerical-failure: loss is not finite")
                loss.backward()
                total += float(loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.gradient_clip_norm, error_if_nonfinite=True
            )
        except Exception:
            # No optimizer update or silent successful-step increment on failure.
            raise
        lr = self.learning_rate(self.step + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.step += 1
        return dict(loss=total, gradient_norm=float(norm), learning_rate=lr, step=self.step)
