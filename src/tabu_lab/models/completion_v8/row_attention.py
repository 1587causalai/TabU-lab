"""V8.2 row attention: signed linear aggregation with source/receiver presence.

Only the attention update changes. OMAB's residual, local FFN, parameter names,
eligibility and scoped precision remain intact. No pairwise row matrix is built.
"""

import math

import torch
from torch.nn import functional as F

from ..restoration._validation import finite
from ..restoration.backbone import OMAB
from .encoder import stable_normalize

RAW_LINEAR = "presence_linear_v1"
MEAN_LINEAR = "presence_linear_mean_v1"
LINEAR_MODES = (RAW_LINEAR, MEAN_LINEAR)


class PresenceLinearOMAB(OMAB):
    """S = K^T (rho_source V); update = rho_receiver (Q S / sqrt(d)) W_O.

    Q/K/V project raw carriers; the mean variant then normalizes Q/K per head
    and averages over role-eligible sources (independent of presence values).
    Source presence occurs exactly once. No softmax,
    positive feature map, weight-sum denominator or output RMS is introduced.
    The inherited reference_mass applies to the unchanged column OMABs only.
    """

    def __init__(self, config, *, strict_finite_content=False, mode=RAW_LINEAR):
        super().__init__(config, strict_finite_content=strict_finite_content)
        if mode not in LINEAR_MODES:
            raise ValueError("unknown presence linear attention mode")
        self.mode = mode

    def attention_update(self, receivers, sources, eligible):
        if receivers.ndim != 3 or sources.ndim != 3 or eligible.ndim != 2:
            raise ValueError("batched linear OMAB expects [B,R,D], [B,S,D], [B,S]")
        if (
            receivers.shape[0] != sources.shape[0]
            or sources.shape[:2] != eligible.shape
            or receivers.shape[-1] != self.config.width
            or sources.shape[-1] != self.config.width
            or eligible.dtype != torch.bool
        ):
            raise ValueError("linear OMAB receivers, sources and boolean mask must align")
        # Delete before projection, including NaN payloads outside eligibility.
        sources = torch.where(eligible[..., None], sources, torch.zeros_like(sources))
        log_p_source = self.log_presence(sources)
        has_presence = ~torch.isneginf(log_p_source)
        sources = torch.where(has_presence[..., None], sources, torch.zeros_like(sources))
        has_sources = has_presence.any(-1)
        # Empty sets have no QK edges, even if an otherwise finite Q would overflow.
        q_carriers = torch.where(has_sources[:, None, None], receivers, torch.zeros_like(receivers))
        batch, nr = receivers.shape[:2]
        ns = sources.shape[1]
        heads, dim = self.config.heads, self.config.width // self.config.heads
        dtype = self.accumulation_dtype(receivers)
        q = F.linear(q_carriers.to(dtype), self.q.weight.to(dtype))
        k = F.linear(sources.to(dtype), self.k.weight.to(dtype))
        v = F.linear(sources.to(dtype), self.v.weight.to(dtype))
        q = q.reshape(batch, nr, heads, dim).transpose(1, 2)
        k = k.reshape(batch, ns, heads, dim).transpose(1, 2)
        v = v.reshape(batch, ns, heads, dim).transpose(1, 2)
        source_weight = log_p_source.exp()
        if self.mode == MEAN_LINEAR:
            q = stable_normalize(q, epsilon=self.config.norm_eps)
            k = stable_normalize(k, epsilon=self.config.norm_eps)
            # Count structural eligibility, never learned presence. This keeps
            # the gate smooth at zero and preserves common gate attenuation.
            count = eligible.sum(-1, keepdim=True).clamp_min(1).to(dtype)
            source_weight = source_weight / count
        summary = k.transpose(-1, -2) @ (source_weight[:, None, :, None] * v)
        finite(summary, "V8.2 row attention summary")
        mixed = ((q / math.sqrt(dim)) @ summary).transpose(1, 2)
        mixed = mixed.reshape(batch, nr, self.config.width)
        finite(mixed, "V8.2 row attention aggregate")
        update = F.linear(
            self.presence(q_carriers)[..., None] * mixed, self.out.weight.to(dtype)
        ).to(receivers)
        finite(update, "V8.2 row attention update")
        return update, log_p_source


def install_presence_linear_rows(backbone, *, mode):
    """Replace row OMABs only, retaining every tensor and the initialization RNG."""
    axial = getattr(backbone, "axial", backbone)
    for layer in axial.layers:
        if layer.row_collect is not None:
            raise ValueError("V8.2 requires direct rows (row_slots=0)")
        old = layer.row
        # Constructors draw CPU parameters; restoring RNG avoids changing later
        # initialization/sampling simply because the attention operator changed.
        with torch.random.fork_rng(devices=[]):
            row = PresenceLinearOMAB(
                old.config, strict_finite_content=old.strict_finite_content, mode=mode
            ).to(next(old.parameters()))
        row.load_state_dict(old.state_dict(), strict=True)
        row.train(old.training)
        layer.row = row
