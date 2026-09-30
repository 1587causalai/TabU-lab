"""Optimizer partitions for the restoration curriculum.

The first curriculum stage uses AdamW.  Later stages move only the two
dimensional backbone Linear weights to native Muon while retaining AdamW state
for every other parameter.  The partition is recorded in checkpoints so a
resume cannot silently change which tensors are owned by either optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float
    grad_clip: float
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = "match_rms_adamw"


class MixedOptimizer:
    """A small checkpointable wrapper around native Muon and AdamW."""

    def __init__(self, muon, adamw, model):
        self.muon = muon
        self.adamw = adamw
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        self.partition = {
            key: [
                [names[id(parameter)] for parameter in group["params"]]
                for group in optimizer.param_groups
            ]
            for key, optimizer in (("muon", muon), ("adamw", adamw))
        }
        assigned = [parameter for group in self.param_groups for parameter in group["params"]]
        if len({id(parameter) for parameter in assigned}) != len(assigned) or {
            id(parameter) for parameter in assigned
        } != set(names):
            raise ValueError("optimizer partition must cover each parameter exactly once")

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    @property
    def state(self):
        return {**self.muon.state, **self.adamw.state}

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return {
            "kind": "restoration.native_muon_plus_adamw",
            "format_version": 1,
            "partition": self.partition,
            "muon": self.muon.state_dict(),
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state):
        if (
            state.get("kind") != "restoration.native_muon_plus_adamw"
            or state.get("format_version") != 1
            or state.get("partition") != self.partition
        ):
            raise ValueError("mixed optimizer state partition or format mismatch")
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def _adamw_groups(model, cfg: OptimizerConfig, excluded: set[int] | None = None):
    excluded = excluded or set()
    decay, other = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) in excluded:
            continue
        matrix = name.endswith(".weight")
        (decay if matrix else other).append(parameter)
    return [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": other, "weight_decay": 0.0},
    ]


def hidden_linear_weights(model):
    selected = {
        f"{name}.weight": module.weight
        for name, module in model.named_modules()
        if name.startswith("backbone.layers.") and isinstance(module, torch.nn.Linear)
    }
    if not selected or any(parameter.ndim != 2 for parameter in selected.values()):
        raise ValueError("Muon requires two-dimensional backbone Linear weights")
    if len({id(parameter) for parameter in selected.values()}) != len(selected):
        raise ValueError("aliased backbone matrices need an explicit optimizer partition")
    return selected


def adamw(model, cfg: OptimizerConfig):
    return torch.optim.AdamW(
        _adamw_groups(model, cfg),
        lr=cfg.learning_rate,
        betas=cfg.betas,
        eps=cfg.eps,
    )


def switch_to_muon(optimizer, model, cfg: OptimizerConfig):
    """Move hidden Linear weights out of AdamW and create their Muon state."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Muon transition requires an AdamW parent")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("this PyTorch runtime has no torch.optim.Muon")
    selected = hidden_linear_weights(model)
    ids = {id(parameter) for parameter in selected.values()}
    for group in optimizer.param_groups:
        group["params"] = [parameter for parameter in group["params"] if id(parameter) not in ids]
    for parameter in list(optimizer.state):
        if id(parameter) in ids:
            del optimizer.state[parameter]
    muon = torch.optim.Muon(
        list(selected.values()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov,
        ns_steps=cfg.muon_ns_steps,
        adjust_lr_fn=cfg.muon_adjust_lr_fn,
    )
    return MixedOptimizer(muon, optimizer, model), {
        "muon": list(selected),
        "adamw": [name for name, _ in model.named_parameters() if name not in selected],
    }


__all__ = ["MixedOptimizer", "OptimizerConfig", "adamw", "hidden_linear_weights", "switch_to_muon"]
