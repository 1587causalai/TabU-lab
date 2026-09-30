"""Step-4 readout for the TabUBase cell carrier."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import (
    NumericReadoutOutput,
    SameColumnNumericLocalLinear,
    SameColumnNumericNW,
)

from .types import ReferenceConfig


def _numeric_terminal(
    config: ReferenceConfig,
    *,
    numeric_terminal: str,
) -> tuple[nn.Module, str]:
    """Resolve the declared same-column terminal without changing geometry."""

    value = getattr(numeric_terminal, "value", numeric_terminal)
    if value == "nadaraya_watson":
        terminal = SameColumnNumericNW(config.routing_bandwidth)
    elif value == "local_linear":
        terminal = SameColumnNumericLocalLinear(config.routing_bandwidth)
    else:
        raise ValueError(f"unknown numeric terminal: {numeric_terminal!r}")
    return terminal, value


class PairUnitReadout(nn.Module):
    """Global learned projection $z_{ra}=Wc_{ra}$ plus a typed terminal.

    The historical class name is retained for compatibility with the reviewed
    implementation substrate. The public model contract names the semantic
    object correctly: each table cell is one Unit.
    """

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        numeric_terminal: str = "local_linear",
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(
            config.d_model,
            config.matched_slots,
            bias=False,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    @property
    def ll_ridge(self) -> float | None:
        """Expose the local-linear ridge in checkpoint/trace identity."""

        value: Any = getattr(self.terminal, "ridge", None)
        return None if value is None else float(value)

    def forward(
        self,
        cells: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        work_cells = cells.to(dtype=DEFAULT_FLOAT_DTYPE)
        coordinates = F.linear(
            work_cells,
            self.projection.weight.to(dtype=DEFAULT_FLOAT_DTYPE),
        )
        return coordinates, self.terminal(coordinates, support_values, visible_mask)


__all__ = ["PairUnitReadout"]
