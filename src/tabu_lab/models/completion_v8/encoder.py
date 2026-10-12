"""Whole-table bijection for the experimental V8 completion branch.

This encoder has its own linear attention. The inherited Token Dynamics keep
their original attention implementation and checkpoint semantics.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..restoration._validation import finite
from ..restoration.backbone import OFFN, BackboneConfig


def stable_normalize(x, *, mean_square=False, epsilon=1e-6):
    """x/sqrt(sum-or-mean(x**2)+epsilon), without squaring large values.

    Rescaling also rescales epsilon: this preserves the declared operator,
    including its small-signal behavior. Detaching a numerical scale leaves
    the exact derivative unchanged because the scale cancels algebraically.
    vector_norm and hypot avoid both overflow and a 0/0 at constant zero.
    """
    finite(x, "V8 normalization input")
    scale = x.detach().abs().amax(-1, keepdim=True).clamp_min(1)
    scaled = x / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    if mean_square:
        norm = norm / math.sqrt(x.shape[-1])
    denominator = torch.hypot(norm, math.sqrt(epsilon) / scale)
    return scaled / denominator


class SetAttention(nn.Module):
    """One-head sum attention with an independent Q/K width (historically 32)."""

    def __init__(self, width=32, head_dim=32):
        super().__init__()
        self.width, self.head_dim = width, head_dim
        self.q = nn.Linear(width, head_dim, bias=False)
        self.k = nn.Linear(width, head_dim, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.presence_projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.presence_projection.weight)
        self.rms_gain = nn.Parameter(torch.ones(width))

    def forward(self, x, valid=None, *, support_rows=None):
        if valid is not None and support_rows is not None:
            raise ValueError("use row identity or the legacy Cell mask, not both")
        # Delete Null payload before projections, including nonfinite sentinels.
        if valid is not None:
            x = torch.where(valid[..., None], x, 0)
        finite(x, "V8 attention input")
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        raw = x.to(dtype)
        b = F.layer_norm(raw, (self.width,), eps=1e-5)
        # Only row identity restricts the new path. Null is already exact
        # zero: bias-free LN/projections/SiLU preserve its zero contribution.
        source = b if support_rows is None else b[:, support_rows]
        q = F.silu(F.linear(b, self.q.weight.to(dtype)))
        k, v = [F.silu(F.linear(source, p.weight.to(dtype))) for p in (self.k, self.v)]
        q, k = stable_normalize(q), stable_normalize(k)
        if valid is not None:
            k, v = [torch.where(valid[..., None], z, 0) for z in (k, v)]
        # Accumulate over source chunks: no quadratic cell-to-cell matrix.
        summary = raw.new_zeros((*raw.shape[:-2], self.head_dim, self.width))
        for kc, vc in zip(k.split(256, -2), v.split(256, -2), strict=True):
            summary = summary + kc.transpose(-1, -2) @ vc
        finite(summary, "V8 K transpose V")
        h = (q @ summary) / math.sqrt(self.head_dim)
        finite(h, "V8 attention aggregate")
        h = self.rms_gain.to(dtype) * stable_normalize(h, mean_square=True)
        projected = F.linear(raw, self.presence_projection.weight.to(dtype))
        finite(projected, "V8 receiver presence projection")
        scale = projected.abs().amax(-1)
        active = scale > 0
        safe_scale = torch.where(active, scale, 1)
        mass = (projected / safe_scale[..., None]).square().sum(-1)
        log_mass = 2 * safe_scale.log() + torch.where(active, mass, 1).log()
        presence = torch.where(active, log_mass.sigmoid(), 0)
        result = F.linear(presence[..., None] * h, self.out.weight.to(dtype))
        finite(result, "V8 attention output")
        return result.to(x) if valid is None else torch.where(valid[..., None], result.to(x), 0)


class CouplingBlock(nn.Module):
    def __init__(self, axis, width=32):
        super().__init__()
        self.axis = axis
        self.attention = SetAttention(width=width)
        # Reuse the established RMS/GELU/presence OFFN arithmetic; the FFN hidden width remains 64.
        self.offn = OFFN(
            BackboneConfig(
                width=width, layers=1, heads=1, ff_width=64, kind="direct", slots=1, norm_eps=1e-6
            )
        )

    def attention_update(self, x, valid, support_rows=None):
        # Axis names denote positions that interact: columns within one row,
        # then rows within one column (the manuscript's col/row convention).
        if self.axis == "row":
            if support_rows is not None:
                return self.attention(x.transpose(0, 1), support_rows=support_rows).transpose(0, 1)
            return self.attention(x.transpose(0, 1), valid.T).transpose(0, 1)
        return self.attention(x, None if support_rows is not None else valid)

    def local_update(self, x, valid):
        if valid is not None:
            x = torch.where(valid[..., None], x, 0)
        # Same inherited OFFN formula and parameters, but avoid silently
        # collapsing its RMS-normalized content if x.square() overflows.
        normalized = self.offn.norm_scale * stable_normalize(
            x, mean_square=True, epsilon=self.offn.config.norm_eps
        )
        update = self.offn.presence(x, local=True)[..., None] * self.offn.ff(normalized).to(x)
        finite(update, "V8 OFFN update")
        return update.to(x) if valid is None else torch.where(valid[..., None], update.to(x), 0)

    def forward(self, u, v, valid, support_rows=None):
        u = u + self.attention_update(v, valid, support_rows)
        v = v + self.local_update(u, None if support_rows is not None else valid)
        return u, v

    def inverse(self, u, v, valid, support_rows=None):
        v = v - self.local_update(u, None if support_rows is not None else valid)
        u = u - self.attention_update(v, valid, support_rows)
        return u, v


class TableEncoder(nn.Module):
    """Per-cell phi followed by column/row/column/row additive coupling.

    Inversion requires the WHOLE table and the same role mask. Null is zero
    outside the active-coordinate bijection and never passes through phi.
    """

    def __init__(self, phi, *, code_dim=64):
        super().__init__()
        if code_dim not in (32, 64):
            raise ValueError("V8 table code_dim must be 32 or 64")
        self.code_dim = code_dim
        self.phi = phi
        self.blocks = nn.ModuleList(
            CouplingBlock(axis, width=code_dim // 2) for axis in ("column", "row", "column", "row")
        )

    def validate(self, table, valid):
        if table.ndim != 3 or table.shape[-1] != self.code_dim or table.shape[:2] != valid.shape:
            raise ValueError(f"V8 encoder requires [rows, columns, {self.code_dim}] and mask")
        if valid.dtype != torch.bool or valid.device != table.device:
            raise ValueError("V8 encoder mask must be boolean on the table device")

    def forward(self, codes, valid, *, support_rows=None):
        self.validate(codes, valid)
        self.validate_rows(codes, support_rows)
        encoded = torch.zeros_like(codes)
        if bool(valid.any()):
            encoded[valid] = self.phi(codes[valid])
        u, v = encoded.chunk(2, -1)
        # The new dynamics receive only row identity. The active-coordinate
        # mask is confined to the phi boundary, never used inside these blocks.
        dynamics_valid = valid if support_rows is None else None
        for block in self.blocks:
            u, v = block(u, v, dynamics_valid, support_rows)
        return torch.cat((u, v), -1)

    def inverse(self, embeddings, valid, *, support_rows=None):
        self.validate(embeddings, valid)
        self.validate_rows(embeddings, support_rows)
        u, v = torch.where(valid[..., None], embeddings, 0).chunk(2, -1)
        dynamics_valid = valid if support_rows is None else None
        for block in reversed(self.blocks):
            u, v = block.inverse(u, v, dynamics_valid, support_rows)
        joined = torch.cat((u, v), -1)
        codes = torch.zeros_like(joined)
        if bool(valid.any()):
            codes[valid] = self.phi.inverse(joined[valid])
        finite(codes, "V8 whole-table inverse")
        return codes

    @staticmethod
    def validate_rows(table, support_rows):
        if support_rows is not None and (
            support_rows.dtype != torch.bool or support_rows.shape != table.shape[:1]
            or support_rows.device != table.device
        ):
            raise ValueError("support_rows must be a boolean row vector on the table device")
