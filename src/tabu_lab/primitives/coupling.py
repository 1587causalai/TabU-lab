"""Clamped affine coupling bijections for value-geometry maps.

Each block keeps one coordinate group fixed and rescales and shifts the
other group with parameters computed from the fixed group, so the inverse is
closed form and never inverts the parameter network.  The log-scale passes
through ``alpha * tanh``: every multiplier lies in ``[exp(-alpha), exp(alpha)]``.
That bounds the multiplicative branch only; the shift network and the
composition still enter the inverse Jacobian.

The maps are deterministic per vector: no dropout, no batch statistics, and
no normalisation on the main path, so ``forward`` and ``inverse`` are always
the two directions of one function.  No base distribution or log-determinant
is provided; the bijection is used as a coordinate change, not as a density
model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE


def _parameter_network(in_features: int, hidden: Sequence[int], out_features: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layer = nn.Linear(width, size, dtype=DEFAULT_FLOAT_DTYPE)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.extend((layer, nn.GELU()))
        width = size
    last = nn.Linear(width, out_features, dtype=DEFAULT_FLOAT_DTYPE)
    # A zero final layer gives s = t = 0, so every block starts as the identity.
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class AffineCoupling(nn.Module):
    """One-sided coupling ``y_B = x_B * exp(alpha * tanh s(x_A)) + t(x_A)``.

    ``update_index`` selects the coordinates in ``B``; every other coordinate
    is ``A`` and passes through unchanged.  With ``scale=False`` the block is
    the additive coupling ``y_B = x_B + t(x_A)``.  Inputs are ``[..., dim]``.
    """

    def __init__(
        self,
        dim: int,
        update_index: Sequence[int],
        *,
        hidden: Sequence[int] = (128, 128),
        alpha: float = 0.5,
        scale: bool = True,
    ) -> None:
        super().__init__()
        if dim < 2:
            raise ValueError("dim must be at least 2")
        update = sorted({int(i) for i in update_index})
        if len(update) != len(update_index) or not update:
            raise ValueError("update_index must be non-empty and free of duplicates")
        if update[0] < 0 or update[-1] >= dim or len(update) == dim:
            raise ValueError("update_index must be a proper subset of range(dim)")
        if any(int(size) <= 0 for size in hidden):
            raise ValueError("hidden widths must be positive")
        if scale and not (alpha > 0.0 and math.isfinite(alpha)):
            raise ValueError("alpha must be positive and finite")
        updated = set(update)
        keep = [i for i in range(dim) if i not in updated]
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.scale = bool(scale)
        self.register_buffer("keep_index", torch.tensor(keep, dtype=torch.long), persistent=False)
        self.register_buffer(
            "update_index", torch.tensor(update, dtype=torch.long), persistent=False
        )
        n_update = len(update)
        self.net = _parameter_network(
            len(keep), tuple(int(size) for size in hidden), (2 if scale else 1) * n_update
        )

    def _shift_and_log_scale(self, kept: Tensor) -> tuple[Tensor, Tensor | None]:
        out = self.net(kept)
        if not self.scale:
            return out, None
        raw_log_scale, shift = out.chunk(2, dim=-1)
        return shift, self.alpha * torch.tanh(raw_log_scale)

    def _check(self, value: Tensor) -> None:
        if value.shape[-1] != self.dim:
            raise ValueError(f"expected trailing dimension {self.dim}, got {value.shape[-1]}")

    def forward(self, x: Tensor) -> Tensor:
        self._check(x)
        kept = x.index_select(-1, self.keep_index)
        shift, log_scale = self._shift_and_log_scale(kept)
        moved = x.index_select(-1, self.update_index)
        if log_scale is not None:
            moved = moved * torch.exp(log_scale)
        return x.index_copy(-1, self.update_index, moved + shift)

    def inverse(self, y: Tensor) -> Tensor:
        self._check(y)
        kept = y.index_select(-1, self.keep_index)
        shift, log_scale = self._shift_and_log_scale(kept)
        moved = y.index_select(-1, self.update_index) - shift
        if log_scale is not None:
            moved = moved * torch.exp(-log_scale)
        return y.index_copy(-1, self.update_index, moved)


class CouplingValueMap(nn.Module):
    """Stack of one-sided couplings on ``R^dim`` that starts as the identity.

    Block ``k`` updates the second half of the coordinates when ``k`` is even
    and the first half when ``k`` is odd, so no permutation is left over and
    the zero-initialised stack is exactly ``Id``.  The V7 default is
    ``dim=64``, four blocks, ``hidden=(128, 128)`` and ``alpha=0.5``.
    """

    def __init__(
        self,
        dim: int = 64,
        *,
        n_blocks: int = 4,
        hidden: Sequence[int] = (128, 128),
        alpha: float = 0.5,
        scale: bool = True,
    ) -> None:
        super().__init__()
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")
        half = dim // 2
        first, second = range(half), range(half, dim)
        self.dim = int(dim)
        self.blocks = nn.ModuleList(
            AffineCoupling(
                dim,
                second if k % 2 == 0 else first,
                hidden=hidden,
                alpha=alpha,
                scale=scale,
            )
            for k in range(n_blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: Tensor) -> Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


@torch.no_grad()
def inverse_perturbation_gain(
    value_map: CouplingValueMap, embedding: Tensor, delta: Tensor
) -> Tensor:
    """Return ``||inv(e + delta) - inv(e)|| / ||delta||`` per vector.

    This is a diagnostic for how much the inverse amplifies an embedding-space
    error near actual readout outputs; it is not a training target.
    """

    if embedding.shape != delta.shape:
        raise ValueError("embedding and delta must have identical shapes")
    step = delta.norm(dim=-1)
    if bool((step == 0).any()):
        raise ValueError("delta must be non-zero for every vector")
    moved = value_map.inverse(embedding + delta) - value_map.inverse(embedding)
    return moved.norm(dim=-1) / step


__all__ = ["AffineCoupling", "CouplingValueMap", "inverse_perturbation_gain"]
