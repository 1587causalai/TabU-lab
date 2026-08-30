"""O-closed attention/FFN residual block."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .oattention import OAttention, OAttentionOutput, presence_gate


@dataclass(frozen=True)
class OMABOutput:
    state: Tensor
    attention: OAttentionOutput


class OMAB(nn.Module):
    """Pre-norm OAttention plus receiver-gated FFN residual."""

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
        self.attention = OAttention(
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
        self.presence_tau = float(presence_tau)

    def forward(
        self,
        receivers: Tensor,
        sources: Tensor,
        *,
        source_mask: Tensor | None = None,
        pair_mask: Tensor | None = None,
        zero_when_no_support: bool = False,
    ) -> OMABOutput:
        raw_receiver_presence = presence_gate(receivers, self.presence_tau)
        raw_source_presence = presence_gate(sources, self.presence_tau)
        attention = self.attention(
            self.receiver_norm(receivers),
            self.source_norm(sources),
            source_mask=source_mask,
            pair_mask=pair_mask,
            receiver_presence=raw_receiver_presence,
            source_presence=raw_source_presence,
        )
        residual = receivers + self.dropout(attention.output)
        ff = self.ff2(F.gelu(self.ff1(self.ff_norm(residual))))
        ff = presence_gate(residual, self.presence_tau).unsqueeze(-1) * ff
        state = residual + self.dropout(ff)
        state = torch.where(
            raw_receiver_presence.unsqueeze(-1) > 0,
            state,
            torch.zeros_like(state),
        )
        if zero_when_no_support:
            state = torch.where(
                attention.support_available.unsqueeze(-1),
                state,
                torch.zeros_like(state),
            )
        return OMABOutput(state=state, attention=attention)


__all__ = ["OMAB", "OMABOutput"]
