"""Numerically safe O-closed attention primitives.

The mathematical closure lives on the carrier, not in the projection layers:
zero receivers receive no update and zero sources receive exactly zero mass,
even when Q/K/V projections have biases.  Empty support is represented by an
explicit boolean status instead of a uniform or zero pseudo-prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE


def presence_gate(value: Tensor, tau: float | Tensor = 1.0e-6) -> Tensor:
    """Return ``||value||^2 / (tau + ||value||^2)`` in a safe work dtype."""

    if value.ndim == 0:
        raise ValueError("presence_gate expects a token axis")
    work = value.float() if value.dtype in {torch.float16, torch.bfloat16} else value
    norm_squared = work.square().sum(dim=-1)
    resolved_tau = torch.as_tensor(tau, dtype=work.dtype, device=value.device)
    if resolved_tau.numel() != 1 or not bool(torch.isfinite(resolved_tau)):
        raise ValueError("tau must be one finite scalar")
    if float(resolved_tau.detach().cpu()) <= 0.0:
        raise ValueError("tau must be positive")
    # ``n / (tau + n)`` becomes inf/inf for large finite inputs.  The
    # algebraically equivalent form below saturates safely at one.
    gate = 1.0 - resolved_tau / (resolved_tau + norm_squared)
    gate = torch.where(norm_squared == 0, torch.zeros_like(gate), gate)
    gate = torch.nan_to_num(gate, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
    return gate.to(dtype=value.dtype)


def o_inject(receiver: Tensor, source: Tensor, *, tau: float | Tensor = 1.0e-6) -> Tensor:
    """Zero-preserving simultaneous injection ``h + p(h)e``."""

    if receiver.shape != source.shape:
        raise ValueError("receiver and source must have identical shapes")
    return receiver + presence_gate(receiver, tau).unsqueeze(-1) * source


@dataclass(frozen=True)
class OAttentionOutput:
    """Attention output with explicit support status.

    ``weights`` is ``[B,H,R,S]``.  Its sum is slightly below one because the
    mathematical denominator contains ``denominator_epsilon``.  With an empty
    or entirely zero source axis it is exactly zero and ``support_available``
    is false for every receiver.
    """

    output: Tensor
    weights: Tensor
    support_available: Tensor
    support_count: Tensor


class OAttention(nn.Module):
    """Dense multi-head reference implementation of OAttention.

    Inputs use batch-first shapes ``[B,R,D]`` and ``[B,S,D]``.  ``source_mask``
    selects source positions and ``pair_mask`` optionally selects receiver-
    source edges (``True`` means allowed).  The implementation deliberately
    supports ``S == 0``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        dropout: float = 0.0,
        presence_tau: float = 1.0e-6,
        denominator_epsilon: float = 1.0e-8,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if presence_tau <= 0.0 or denominator_epsilon <= 0.0:
            raise ValueError("presence_tau and denominator_epsilon must be positive")
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = d_model // n_heads
        self.dropout = float(dropout)
        self.presence_tau = float(presence_tau)
        self.denominator_epsilon = float(denominator_epsilon)
        self.q_proj = nn.Linear(
            d_model, d_model, bias=bias, dtype=DEFAULT_FLOAT_DTYPE
        )
        self.k_proj = nn.Linear(
            d_model, d_model, bias=bias, dtype=DEFAULT_FLOAT_DTYPE
        )
        self.v_proj = nn.Linear(
            d_model, d_model, bias=bias, dtype=DEFAULT_FLOAT_DTYPE
        )
        self.out_proj = nn.Linear(
            d_model, d_model, bias=bias, dtype=DEFAULT_FLOAT_DTYPE
        )

    def _validate(
        self,
        receivers: Tensor,
        sources: Tensor,
        source_mask: Tensor | None,
        pair_mask: Tensor | None,
        receiver_presence: Tensor | None,
        source_presence: Tensor | None,
    ) -> tuple[int, int, int]:
        if receivers.ndim != 3 or sources.ndim != 3:
            raise ValueError("receivers and sources must be [batch, tokens, d_model]")
        batch, n_receivers, d_model = receivers.shape
        if sources.shape[0] != batch or sources.shape[-1] != d_model:
            raise ValueError("receiver/source batch and d_model axes must agree")
        if d_model != self.d_model:
            raise ValueError("input d_model does not match the module")
        n_sources = sources.shape[1]
        if source_mask is not None and (
            source_mask.shape != (batch, n_sources) or source_mask.dtype is not torch.bool
        ):
            raise ValueError("source_mask must be bool [batch, sources]")
        if pair_mask is not None:
            if pair_mask.shape != (batch, n_receivers, n_sources):
                raise ValueError("pair_mask must be [batch, receivers, sources]")
            if pair_mask.dtype is not torch.bool:
                raise ValueError("pair_mask must be bool (True means allowed)")
        if receiver_presence is not None and receiver_presence.shape != (batch, n_receivers):
            raise ValueError("receiver_presence must be [batch, receivers]")
        if source_presence is not None and source_presence.shape != (batch, n_sources):
            raise ValueError("source_presence must be [batch, sources]")
        return batch, n_receivers, n_sources

    def forward(
        self,
        receivers: Tensor,
        sources: Tensor,
        *,
        source_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
        receiver_presence: Tensor | None = None,
        source_presence: Tensor | None = None,
    ) -> OAttentionOutput:
        batch, n_receivers, n_sources = self._validate(
            receivers,
            sources,
            source_mask,
            pair_mask,
            receiver_presence,
            source_presence,
        )
        if n_sources == 0:
            weights = receivers.new_zeros(batch, self.n_heads, n_receivers, 0)
            return OAttentionOutput(
                output=torch.zeros_like(receivers),
                weights=weights,
                support_available=torch.zeros(
                    batch, n_receivers, dtype=torch.bool, device=receivers.device
                ),
                support_count=torch.zeros(
                    batch, n_receivers, dtype=torch.long, device=receivers.device
                ),
            )

        q = self.q_proj(receivers).view(
            batch, n_receivers, self.n_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(sources).view(
            batch, n_sources, self.n_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(sources).view(
            batch, n_sources, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Reductions and exponentials are kept in FP32 under low-precision
        # autocast.  Casting back remains differentiable.
        work_q = q.float() if q.dtype in {torch.float16, torch.bfloat16} else q
        work_k = k.float() if k.dtype in {torch.float16, torch.bfloat16} else k
        scores = torch.matmul(work_q, work_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        limit = torch.finfo(scores.dtype).max
        scores = torch.nan_to_num(scores, nan=0.0, posinf=limit, neginf=-limit)

        resolved_source_presence = (
            presence_gate(sources, self.presence_tau)
            if source_presence is None
            else source_presence.to(device=sources.device, dtype=scores.dtype)
        )
        resolved_receiver_presence = (
            presence_gate(receivers, self.presence_tau)
            if receiver_presence is None
            else receiver_presence.to(device=receivers.device, dtype=scores.dtype)
        )
        if not bool(torch.isfinite(resolved_source_presence).all()) or bool(
            (resolved_source_presence < 0).any()
        ):
            raise ValueError("source_presence must be finite and non-negative")
        if not bool(torch.isfinite(resolved_receiver_presence).all()) or bool(
            (resolved_receiver_presence < 0).any()
        ):
            raise ValueError("receiver_presence must be finite and non-negative")
        allowed = resolved_source_presence[:, None, None, :] > 0
        if source_mask is not None:
            allowed = allowed & source_mask[:, None, None, :]
        if pair_mask is not None:
            allowed = allowed & pair_mask[:, None, :, :]
        allowed = allowed.expand(batch, self.n_heads, n_receivers, n_sources)

        masked_scores = scores.masked_fill(~allowed, -torch.inf)
        offset = masked_scores.amax(dim=-1, keepdim=True)
        offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
        # Multiplying the factory numerator and denominator by exp(-offset)
        # is exact only when epsilon is scaled as well.  Choosing a nonnegative
        # offset keeps that correction finite even when every allowed score is
        # very negative.
        offset = offset.clamp_min(0.0)
        exp_scores = torch.where(
            allowed,
            torch.exp(masked_scores - offset),
            torch.zeros_like(masked_scores),
        )
        mass = exp_scores * resolved_source_presence[:, None, None, :]
        denominator = mass.sum(dim=-1, keepdim=True)
        scaled_epsilon = self.denominator_epsilon * torch.exp(-offset)
        weights = mass / (denominator + scaled_epsilon)
        value_weights = F.dropout(weights, self.dropout, self.training)

        work_v = v.float() if v.dtype in {torch.float16, torch.bfloat16} else v
        attended = torch.matmul(value_weights.to(work_v.dtype), work_v)
        attended = attended.to(receivers.dtype).transpose(1, 2).reshape(
            batch, n_receivers, self.d_model
        )
        projected = self.out_proj(attended)
        receiver_gate = resolved_receiver_presence.unsqueeze(-1)
        edge_available = allowed.any(dim=1)
        support_available = edge_available.any(dim=-1)
        # The normalized value path already contains the epsilon-bearing
        # denominator.  A boolean support gate closes the output-projection
        # bias for empty support without attenuating non-empty content twice.
        output = (
            projected
            * receiver_gate.to(projected.dtype)
            * support_available.unsqueeze(-1).to(projected.dtype)
        )
        support_count = edge_available.sum(dim=-1)
        return OAttentionOutput(
            output=output,
            weights=weights.to(receivers.dtype),
            support_available=support_available,
            support_count=support_count,
        )


__all__ = ["OAttention", "OAttentionOutput", "o_inject", "presence_gate"]
