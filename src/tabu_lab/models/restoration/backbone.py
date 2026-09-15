"""Reference axial OMABs with fixed visible sources and exact Null closure.

Q/K/V read raw carriers. Only the local FFN uses learned-scale RMS normalization.
This implements the table-restoration realization, not a replacement for P02.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ._validation import finite, positive


@dataclass(frozen=True)
class BackboneConfig:
    width: int = 128
    layers: int = 2
    heads: int = 4
    ff_width: int = 256
    kind: str = "inducing"
    slots: int = 256
    tau_presence: float = 1.0
    reference_mass: float = 1.0
    norm_eps: float = 1e-6

    def __post_init__(self):
        for name in ("width", "layers", "heads", "ff_width", "slots"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.width % self.heads:
            raise ValueError("carrier width must be divisible by head count")
        if self.kind not in ("direct", "inducing"):
            raise ValueError("backbone kind must be direct or inducing")
        positive(self.tau_presence, "tau_presence")
        positive(self.reference_mass, "reference_mass")
        positive(self.norm_eps, "norm_eps")


class OMAB(nn.Module):
    """One receiver set and one source set; empty sources retain local FFN work."""

    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        d = config.width
        self.attention_presence = nn.Linear(d, d, bias=False)
        self.ff_presence = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.attention_presence.weight)
        nn.init.eye_(self.ff_presence.weight)
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm_scale = nn.Parameter(torch.ones(d))
        self.ff = nn.Sequential(
            nn.Linear(d, config.ff_width, bias=False),
            nn.GELU(),
            nn.Linear(config.ff_width, d, bias=False),
        )

    def log_presence(self, carriers: Tensor, *, local: bool = False) -> Tensor:
        """Stable log rho; only an EXACT zero projection has log presence -inf.

        Scale before squaring. Forming rho in the carrier dtype and then taking
        log would delete tiny positive sources before a large content score can
        restore their mass. FP64 accumulation is part of this reference realization.
        """
        projection = self.ff_presence if local else self.attention_presence
        projected = torch.nn.functional.linear(carriers.double(), projection.weight.double())
        finite(projected, "projected presence")
        scale = projected.abs().amax(-1)
        active = scale > 0
        safe_scale = torch.where(active, scale, torch.ones_like(scale))
        squared = (projected / safe_scale[:, None]).square().sum(-1)
        safe_squared = torch.where(active, squared, torch.ones_like(squared))
        log_mass = 2 * safe_scale.log() + safe_squared.log()
        log_rho = log_mass - torch.logaddexp(
            log_mass, log_mass.new_tensor(math.log(self.config.tau_presence))
        )
        return torch.where(active, log_rho, torch.full_like(log_rho, -torch.inf))

    def presence(self, carriers: Tensor, *, local: bool = False) -> Tensor:
        return self.log_presence(carriers, local=local).exp()

    def forward(self, receivers: Tensor, sources: Tensor, eligible: Tensor) -> Tensor:
        # Delete ineligible sources before any projection; zero-mass real sources
        # are also removed before log/softmax, including learned nullspaces.
        finite(receivers, "OMAB receivers")
        sources = sources[eligible]
        log_p_source = self.log_presence(sources)
        active = torch.isfinite(log_p_source)
        sources, log_p_source = sources[active], log_p_source[active]
        if len(sources):
            heads, dim = self.config.heads, self.config.width // self.config.heads
            q = self.q(receivers).reshape(-1, heads, dim).transpose(0, 1)
            k = self.k(sources).reshape(-1, heads, dim).transpose(0, 1)
            v = self.v(sources).reshape(-1, heads, dim).transpose(0, 1)
            logits = q.double() @ k.double().transpose(-1, -2) / math.sqrt(dim) + log_p_source
            finite(logits, "OMAB logits")
            reference = logits.new_full(
                (*logits.shape[:-1], 1), math.log(self.config.reference_mass)
            )
            weights = torch.cat((logits, reference), -1).softmax(-1)[..., :-1]
            mixed = (weights @ v.double()).transpose(0, 1).reshape(len(receivers), -1)
            update = torch.nn.functional.linear(
                self.presence(receivers)[:, None] * mixed, self.out.weight.double()
            ).to(receivers)
        else:
            update = torch.zeros_like(receivers)
        residual = receivers + update
        normalized = (
            self.norm_scale
            * residual
            * torch.rsqrt(residual.square().mean(-1, keepdim=True) + self.config.norm_eps)
        )
        local_update = self.presence(residual, local=True)[:, None] * self.ff(normalized).double()
        result = residual + local_update.to(residual)
        finite(result, "OMAB output")
        return result


class AxialLayer(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.column = OMAB(config)  # direct or inducing read; never collect's parameters
        self.row = OMAB(config)
        if config.kind == "inducing":
            self.collect = OMAB(config)
            self.slot_seed = nn.Parameter(
                torch.randn(config.slots, config.width) / math.sqrt(config.width)
            )
        else:
            self.collect = None
            self.register_parameter("slot_seed", None)

    def forward(self, h: Tensor, source_mask: Tensor, null_mask: Tensor) -> Tensor:
        n, m = h.shape[0] - 1, h.shape[1] - 1
        columns = []
        for a in range(m + 1):
            source = h[:, a]
            eligible = source_mask[:, a]
            if self.collect is not None and a < m:
                # The seed residual is not evidence. Check projected collect mass,
                # not carrier norm, and keep the slot count independent of N.
                has_evidence = bool(
                    torch.isfinite(self.collect.log_presence(source[eligible])).any()
                )
                if has_evidence:
                    source = self.collect(self.slot_seed, source, eligible)
                    eligible = torch.ones(len(source), dtype=torch.bool, device=h.device)
                else:
                    source = h.new_empty(0, h.shape[-1])
                    eligible = source_mask.new_empty(0)
            columns.append(self.column(h[:, a], source, eligible))
        h = torch.stack(columns, 1).masked_fill(null_mask[..., None], 0)
        rows = [self.row(h[r], h[r], source_mask[r]) for r in range(n + 1)]
        return torch.stack(rows).masked_fill(null_mask[..., None], 0)


class AxialBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(AxialLayer(config) for _ in range(config.layers))

    def forward(self, h: Tensor, visible: Tensor, query: Tensor) -> Tensor:
        n, m = visible.shape
        sources = torch.zeros(n + 1, m + 1, dtype=torch.bool, device=h.device)
        sources[:n, :m] = visible
        nulls = torch.zeros_like(sources)
        nulls[:n, :m] = ~(visible | query)
        nulls[n, m] = True
        h = h.masked_fill(nulls[..., None], 0)
        for layer in self.layers:
            h = layer(h, sources, nulls)
        return h
