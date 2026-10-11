"""Optional positive linear attention in the existing 64D response coordinates."""

import math

from torch import nn
from torch.nn import functional as F

from ..restoration._validation import finite


class PositiveLinearReadout(nn.Module):
    """Cell-only Q/K, visible-only E values; no pairwise matrix or ridge solve."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 32, bias=False)
        self.k = nn.Linear(128, 32, bias=False)

    def forward(self, queries, supports, values):
        if queries.ndim != 2 or queries.shape[1] != 128:
            raise ValueError("readout queries must be [q,128]")
        if supports.ndim != 2 or supports.shape[1] != 128 or supports.shape[0] == 0:
            raise ValueError("readout requires nonempty [n,128] visible supports")
        if values.shape != (supports.shape[0], 64):
            raise ValueError("readout values must be [n,64]")
        q, k = F.softplus(self.q(queries)) + 1e-4, F.softplus(self.k(supports)) + 1e-4
        finite(q, "positive readout Q")
        finite(k, "positive readout K")
        finite(values, "positive readout values")
        # A per-query scale and ONE common source scale cancel in the ratio.
        # Detaching these auxiliary scales preserves the quotient's derivative.
        q = q / q.detach().amax(-1, keepdim=True)
        k = k / k.detach().amax()
        # Divide both sufficient statistics by n before multiplying values.
        k = k / supports.shape[0]
        denominator = q @ k.sum(0)
        finite(denominator, "positive readout denominator")
        if bool((denominator <= 0).any()):
            raise ValueError("positive readout denominator underflow")
        result = (q @ (k.T @ values)) / denominator[:, None]
        finite(result, "positive readout prediction")
        return result


class SignedLinearReadout(nn.Module):
    """Signed sum readout: projection → SiLU, with no LayerNorm or L2.

    Values stay in the encoder's 64D coordinates. The fixed 1/sqrt(32) scale
    does not normalize by source count or preserve constant responses.
    """

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 32, bias=False)
        self.k = nn.Linear(128, 32, bias=False)

    def forward(self, queries, supports, values):
        if queries.ndim != 2 or queries.shape[1] != 128:
            raise ValueError("readout queries must be [q,128]")
        if supports.ndim != 2 or supports.shape[1] != 128 or supports.shape[0] == 0:
            raise ValueError("readout requires nonempty [n,128] visible supports")
        if values.shape != (supports.shape[0], 64):
            raise ValueError("readout values must be [n,64]")
        finite(values, "signed readout values")
        q = F.silu(self.q(queries))
        k = F.silu(self.k(supports))
        finite(q, "signed readout Q")
        finite(k, "signed readout K")
        # Only a 32x64 sufficient statistic is accumulated. Never form Q Kᵀ.
        summary = values.new_zeros(32, 64)
        for kc, vc in zip(k.split(256), values.split(256), strict=True):
            summary = summary + kc.T @ vc
        finite(summary, "signed readout summary")
        result = (q @ summary) / math.sqrt(32)
        finite(result, "signed readout prediction")
        return result
