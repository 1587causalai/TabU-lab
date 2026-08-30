"""Explicit dense dynamics plans for TabU-family carriers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import MAB, OMAB

from .types import DynamicsBlockKind, ReferenceConfig


@dataclass(frozen=True)
class DynamicsPlan:
    name: str
    stages: tuple[str, ...]
    carrier: str

    def resolved_name(self, block_kind: DynamicsBlockKind) -> str:
        """Return a trace name that reflects the selected block variant."""

        return self.name.replace("omab", block_kind.value)


def _block(config: ReferenceConfig) -> MAB | OMAB:
    block_type = MAB if config.block_kind is DynamicsBlockKind.MAB else OMAB
    return block_type(
        config.d_model,
        config.n_heads,
        config.d_ff,
        dropout=config.dropout,
        presence_tau=config.presence_tau,
        denominator_epsilon=config.denominator_epsilon,
    )


class InducedCarrierBlock(nn.Module):
    """Column inducing slots, token-from-slot, then row-axis OMAB."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        exclude_row_self: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.exclude_row_self = bool(exclude_row_self)
        self.slot_seeds = nn.Parameter(
            torch.empty(
                config.inducing_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.slot_mab = _block(config)
        self.token_mab = _block(config)
        self.row_mab = _block(config)
        nn.init.normal_(self.slot_seeds, std=0.02)

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
    ) -> Tensor:
        if carrier.ndim != 4:
            raise ValueError("carrier must be [B,R,C,D]")
        batch, n_rows, n_columns, d_model = carrier.shape
        expected = (batch, n_rows, n_columns)
        if column_source_mask.shape != expected or row_source_mask.shape != expected:
            raise ValueError("source masks must match carrier axes")
        if column_source_mask.dtype is not torch.bool or row_source_mask.dtype is not torch.bool:
            raise ValueError("source masks must be bool")

        columns = carrier.permute(0, 2, 1, 3).reshape(batch * n_columns, n_rows, d_model)
        column_mask = column_source_mask.permute(0, 2, 1).reshape(batch * n_columns, n_rows)
        slot_queries = (
            self.slot_seeds.to(carrier.dtype).unsqueeze(0).expand(batch * n_columns, -1, -1)
        )
        slots = self.slot_mab(
            slot_queries,
            columns,
            source_mask=column_mask,
            zero_when_no_support=True,
        ).state
        column_states = self.token_mab(columns, slots).state
        token_carrier = (
            column_states.reshape(batch, n_columns, n_rows, d_model)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        rows = token_carrier.reshape(batch * n_rows, n_columns, d_model)
        row_mask = row_source_mask.reshape(batch * n_rows, n_columns)
        row_pair_mask = None
        if self.exclude_row_self:
            row_pair_mask = ~torch.eye(
                n_columns,
                dtype=torch.bool,
                device=carrier.device,
            ).unsqueeze(0).expand(batch * n_rows, -1, -1)
        output = self.row_mab(
            rows,
            rows,
            source_mask=row_mask,
            pair_mask=row_pair_mask,
        ).state
        return output.reshape(batch, n_rows, n_columns, d_model)


class AugmentedDynamics(nn.Module):
    plan = DynamicsPlan(
        name="augmented_three_omab",
        stages=("slot_from_visible", "token_from_slot", "row_axis"),
        carrier="(N+K)x(M+K)",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(InducedCarrierBlock(config) for _ in range(config.n_blocks))

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
    ) -> Tensor:
        for block in self.blocks:
            carrier = block(
                carrier,
                column_source_mask=column_source_mask,
                row_source_mask=row_source_mask,
            )
        return carrier


class CellUnitDynamics(AugmentedDynamics):
    plan = DynamicsPlan(
        name="cell_unit_three_omab",
        stages=("slot_from_col_mates", "unit_from_slot", "row_mates"),
        carrier="NxM",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList(
            InducedCarrierBlock(config, exclude_row_self=False) for _ in range(config.n_blocks)
        )
