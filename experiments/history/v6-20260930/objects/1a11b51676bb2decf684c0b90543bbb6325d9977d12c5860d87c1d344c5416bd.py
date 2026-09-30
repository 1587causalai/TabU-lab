"""Standard MAB counterpart for the O-closed :mod:`omab` block.

The implementation intentionally mirrors :class:`tabu_lab.primitives.OMAB`
parameter-for-parameter.  ``MAB`` is a control/ablation block: it keeps the
same pre-normalization, projections, residual layout, masks, and FFN, while
omitting the O-closed presence gates and exact-zero restoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .oattention import OAttention, OAttentionOutput


class MABAttention(OAttention):
    """Masked scaled dot-product attention without O-presence weighting.

    Subclassing :class:`OAttention` is deliberate: the projection parameter
    names and shapes remain identical, so a same-seed MAB/OMAB pair can share
    a strict ``state_dict``.  The forward path only differs in how masks and
    attention weights are computed.
    """

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
        del receiver_presence, source_presence
        batch, n_receivers, n_sources = self._validate(
            receivers,
            sources,
            source_mask,
            pair_mask,
            None,
            None,
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

        work_q = q.float() if q.dtype in {torch.float16, torch.bfloat16} else q
        work_k = k.float() if k.dtype in {torch.float16, torch.bfloat16} else k
        scores = torch.matmul(work_q, work_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        limit = torch.finfo(scores.dtype).max
        scores = torch.nan_to_num(scores, nan=0.0, posinf=limit, neginf=-limit)

        allowed = torch.ones(
            batch,
            self.n_heads,
            n_receivers,
            n_sources,
            dtype=torch.bool,
            device=receivers.device,
        )
        if source_mask is not None:
            allowed = allowed & source_mask[:, None, None, :]
        if pair_mask is not None:
            allowed = allowed & pair_mask[:, None, :, :]

        masked_scores = scores.masked_fill(~allowed, -torch.inf)
        offset = masked_scores.amax(dim=-1, keepdim=True)
        offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
        offset = offset.clamp_min(0.0)
        exp_scores = torch.where(
            allowed,
            torch.exp(masked_scores - offset),
            torch.zeros_like(masked_scores),
        )
        denominator = exp_scores.sum(dim=-1, keepdim=True)
        safe_denominator = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
        weights = torch.where(
            denominator > 0,
            exp_scores / safe_denominator,
            torch.zeros_like(exp_scores),
        )
        value_weights = F.dropout(weights, self.dropout, self.training)

        work_v = v.float() if v.dtype in {torch.float16, torch.bfloat16} else v
        attended = torch.matmul(value_weights.to(work_v.dtype), work_v)
        attended = attended.to(receivers.dtype).transpose(1, 2).reshape(
            batch, n_receivers, self.d_model
        )
        projected = self.out_proj(attended)
        edge_available = allowed.any(dim=1)
        support_available = edge_available.any(dim=-1)
        support_count = edge_available.sum(dim=-1)
        return OAttentionOutput(
            output=projected,
            weights=weights.to(receivers.dtype),
            support_available=support_available,
            support_count=support_count,
        )


@dataclass(frozen=True)
class MABOutput:
    state: Tensor
    attention: OAttentionOutput


class MAB(nn.Module):
    """Pre-norm standard attention plus an ungated FFN residual.

    The public call signature matches :class:`OMAB`.  ``zero_when_no_support``
    remains a shared typed support policy, while O-specific presence and
    exact-zero gates are intentionally absent.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        *,
        dropout: float = 0.0,
        presence_tau: float = 1.0e-6,
        denominator_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.receiver_norm = nn.LayerNorm(d_model, dtype=DEFAULT_FLOAT_DTYPE)
        self.source_norm = nn.LayerNorm(d_model, dtype=DEFAULT_FLOAT_DTYPE)
        self.attention = MABAttention(
            d_model,
            n_heads,
            dropout=dropout,
            presence_tau=presence_tau,
            denominator_epsilon=denominator_epsilon,
        )
        self.ff_norm = nn.LayerNorm(d_model, dtype=DEFAULT_FLOAT_DTYPE)
        self.ff1 = nn.Linear(d_model, d_ff, dtype=DEFAULT_FLOAT_DTYPE)
        self.ff2 = nn.Linear(d_ff, d_model, dtype=DEFAULT_FLOAT_DTYPE)
        self.dropout = nn.Dropout(dropout)
        # Retain the fields for configuration/state compatibility.  MAB does
        # not consult either value in its forward path.
        self.presence_tau = float(presence_tau)

    def forward(
        self,
        receivers: Tensor,
        sources: Tensor,
        *,
        source_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
        zero_when_no_support: bool = False,
    ) -> MABOutput:
        attention = self.attention(
            self.receiver_norm(receivers),
            self.source_norm(sources),
            source_mask=source_mask,
            pair_mask=pair_mask,
        )
        residual = receivers + self.dropout(attention.output)
        ff = self.ff2(F.gelu(self.ff1(self.ff_norm(residual))))
        state = residual + self.dropout(ff)
        if zero_when_no_support:
            state = torch.where(
                attention.support_available.unsqueeze(-1),
                state,
                torch.zeros_like(state),
            )
        return MABOutput(state=state, attention=attention)


__all__ = ["MAB", "MABAttention", "MABOutput"]
