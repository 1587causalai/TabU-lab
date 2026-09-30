"""Reference axial OMABs with fixed visible sources and exact Null closure.

Q/K/V read raw carriers. Only the local FFN uses learned-scale RMS normalization.
This implements the table-restoration realization, not a replacement for P02.

Execution is batched: one column sublayer updates all augmented columns in a
single masked attention call, and one row sublayer updates all augmented rows
in one call. Source deletion still happens before any projection — ineligible
entries are replaced by exact zero carriers, whose bias-free projections are
zero and whose log presence is -inf, so they occupy no softmax mass. This is
value-equivalent to the former per-column filtered lists, including NaN in
ineligible payloads never entering a projection. The empty-evidence gate for
inducing collect is a multiplicative tensor switch, not a host-side branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ._validation import finite, positive
from ._dtype import solve_dtype


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

        Nonfinite projections intentionally propagate to the OMAB output stage.
        They must not be classified as exact-zero evidence: an overflowing
        projection from finite inputs is a numerical failure, not an empty
        source. The output check below reports it without adding a host sync here.
        """
        projection = self.ff_presence if local else self.attention_presence
        dtype = solve_dtype(carriers)
        projected = torch.nn.functional.linear(carriers.to(dtype), projection.weight.to(dtype))
        finite_projection = torch.isfinite(projected).all(-1)
        scale = projected.abs().amax(-1)
        active = scale > 0
        safe_scale = torch.where(active, scale, torch.ones_like(scale))
        squared = (projected / safe_scale[..., None]).square().sum(-1)
        safe_squared = torch.where(active, squared, torch.ones_like(squared))
        log_mass = 2 * safe_scale.log() + safe_squared.log()
        log_rho = log_mass - torch.logaddexp(
            log_mass, log_mass.new_tensor(math.log(self.config.tau_presence))
        )
        zero_log_presence = torch.full_like(log_rho, -torch.inf)
        invalid_log_presence = torch.full_like(log_rho, torch.nan)
        # Exact-zero is the only inactive state. Nonfinite projections must
        # remain nonfinite so the OMAB output guard reports numerical failure.
        return torch.where(
            finite_projection,
            torch.where(active, log_rho, zero_log_presence),
            invalid_log_presence,
        )

    def presence(self, carriers: Tensor, *, local: bool = False) -> Tensor:
        return self.log_presence(carriers, local=local).exp()

    def forward(self, receivers: Tensor, sources: Tensor, eligible: Tensor) -> Tensor:
        """Single receiver/source sets; a thin wrapper over the batched operator."""
        if receivers.ndim != 2 or sources.ndim != 2 or eligible.ndim != 1:
            raise ValueError("single-set OMAB expects [R,d] receivers, [S,d] sources, [S] mask")
        if len(sources) != len(eligible) or receivers.shape[1] != sources.shape[1]:
            raise ValueError("sources, mask, and receiver width must align")
        return self.batched(receivers[None], sources[None], eligible[None])[0]

    def batched(self, receivers: Tensor, sources: Tensor, eligible: Tensor) -> Tensor:
        return self._batched_with_presence(receivers, sources, eligible)[0]

    def _batched_with_presence(self, receivers, sources, eligible):
        """Batched OMAB over independent receiver/source pairs.

        ``receivers`` is [B, R, d], ``sources`` is [B, S, d], and ``eligible``
        is the [B, S] boolean source-eligibility mask. Batches share this
        module's parameters but never exchange information. Ineligible sources
        are zeroed before any projection; their zero K/V and -inf log presence
        remove them from both numerator and denominator, exactly like deleting
        them from a per-set list. The fixed reference mass keeps every softmax
        row well defined, including fully masked rows.

        Receivers, sources, and intermediate logits are not separately
        finite-checked: every model-internal call passes stage-checked tensors
        (encoder output or a previous OMAB output), nonfinite content propagates
        through the softmax into the result, and the output check below is the
        explicit stage boundary. This keeps one host sync per OMAB instead of
        one per intermediate tensor.
        """
        if receivers.ndim != 3 or sources.ndim != 3 or eligible.ndim != 2:
            raise ValueError("batched OMAB expects [B,R,d], [B,S,d], [B,S] tensors")
        if (
            receivers.shape[0] != sources.shape[0]
            or sources.shape[:2] != eligible.shape
            or receivers.shape[2] != sources.shape[2]
        ):
            raise ValueError("batched receivers, sources, and mask must align")
        # Delete ineligible sources before any projection: exact zero carriers,
        # never NaN payloads, enter K/V. Zero projection gives -inf log presence.
        sources = torch.where(eligible[..., None], sources, torch.zeros_like(sources))
        log_p_source = self.log_presence(sources)  # [B, S], -inf deletes the entry
        # Delete only exact-zero presence before K/V. A nonfinite presence is
        # retained so it propagates through the stage and is reported by the
        # explicit OMAB output check instead of being silently treated as empty.
        sources = torch.where(
            ~torch.isneginf(log_p_source)[..., None], sources, torch.zeros_like(sources)
        )
        heads, dim = self.config.heads, self.config.width // self.config.heads
        batch, n_receivers = receivers.shape[0], receivers.shape[1]
        q = self.q(receivers).reshape(batch, n_receivers, heads, dim).transpose(1, 2)
        k = self.k(sources).reshape(batch, -1, heads, dim).transpose(1, 2)
        v = self.v(sources).reshape(batch, -1, heads, dim).transpose(1, 2)
        dtype = solve_dtype(q)
        content = q.to(dtype) @ k.to(dtype).transpose(-1, -2) / math.sqrt(dim)
        logits = content + log_p_source[:, None, None, :]
        reference = logits.new_full(
            (*logits.shape[:-1], 1), math.log(self.config.reference_mass)
        )
        weights = torch.cat((logits, reference), -1).softmax(-1)[..., :-1]
        mixed = (weights @ v.to(dtype)).transpose(1, 2).reshape(batch, n_receivers, -1)
        update = torch.nn.functional.linear(
            self.presence(receivers)[..., None] * mixed, self.out.weight.to(dtype)
        ).to(receivers)
        residual = receivers + update
        normalized = (
            self.norm_scale
            * residual
            * torch.rsqrt(residual.square().mean(-1, keepdim=True) + self.config.norm_eps)
        )
        local_update = self.presence(residual, local=True)[..., None] * self.ff(
            normalized
        ).to(residual)
        result = residual + local_update.to(residual)
        finite(result, "OMAB output")
        return result, log_p_source


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
        m = h.shape[1] - 1
        columns = h.transpose(0, 1)  # [M+1, N+1, d]: batch element a is column a
        if self.collect is not None:
            visible_sources = columns[:m]
            eligible = source_mask.transpose(0, 1)[:m]
            # Empty-evidence gate as a tensor switch: projected collect mass on
            # visible carriers only. No host sync; the seed residual is not
            # evidence, so a gated column's summaries become exact zeros and
            # the read sublayer masks them through -inf presence.
            summaries, source_presence = self.collect._batched_with_presence(
                self.slot_seed.unsqueeze(0).expand(m, -1, -1), visible_sources, eligible
            )
            has_evidence = torch.isfinite(source_presence).any(-1)
            summaries = summaries * has_evidence.to(summaries.dtype)[:, None, None]
            # The Unit extension column reads zero sources: local remainder only.
            read_sources = torch.cat((summaries, h.new_zeros(1, *summaries.shape[1:])), dim=0)
            read_eligible = source_mask.new_ones(m + 1, summaries.shape[1])
            h = self.column.batched(columns, read_sources, read_eligible).transpose(0, 1)
        else:
            h = self.column.batched(columns, columns, source_mask.transpose(0, 1)).transpose(0, 1)
        h = h.masked_fill(null_mask[..., None], 0)
        h = self.row.batched(h, h, source_mask)
        return h.masked_fill(null_mask[..., None], 0)


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
