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

        columns = carrier.permute(0, 2, 1, 3).reshape(
            batch * n_columns, n_rows, d_model
        )
        column_mask = column_source_mask.permute(0, 2, 1).reshape(
            batch * n_columns, n_rows
        )
        slot_queries = self.slot_seeds.to(carrier.dtype).unsqueeze(0).expand(
            batch * n_columns, -1, -1
        )
        slots = self.slot_mab(
            slot_queries,
            columns,
            source_mask=column_mask,
            zero_when_no_support=True,
        ).state
        column_states = self.token_mab(columns, slots).state
        token_carrier = column_states.reshape(
            batch, n_columns, n_rows, d_model
        ).permute(0, 2, 1, 3).contiguous()

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


class DualAxisCarrierBlock(nn.Module):
    """TabU4Rec four-OMAB user/item induced-axis block."""

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.config = config
        self.user_slots = nn.Parameter(
            torch.empty(
                config.inducing_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.item_slots = nn.Parameter(
            torch.empty(
                config.inducing_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.user_slot_mab = _block(config)
        self.user_token_mab = _block(config)
        self.item_slot_mab = _block(config)
        self.item_token_mab = _block(config)
        nn.init.normal_(self.user_slots, std=0.02)
        nn.init.normal_(self.item_slots, std=0.02)

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
    ) -> Tensor:
        batch, n_rows, n_columns, d_model = carrier.shape
        columns = carrier.permute(0, 2, 1, 3).reshape(
            batch * n_columns, n_rows, d_model
        )
        column_mask = column_source_mask.permute(0, 2, 1).reshape(
            batch * n_columns, n_rows
        )
        user_slots = self.user_slots.to(carrier.dtype).unsqueeze(0).expand(
            batch * n_columns, -1, -1
        )
        user_slots = self.user_slot_mab(
            user_slots,
            columns,
            source_mask=column_mask,
            zero_when_no_support=True,
        ).state
        columns = self.user_token_mab(columns, user_slots).state
        state = columns.reshape(batch, n_columns, n_rows, d_model).permute(
            0, 2, 1, 3
        ).contiguous()

        rows = state.reshape(batch * n_rows, n_columns, d_model)
        row_mask = row_source_mask.reshape(batch * n_rows, n_columns)
        item_slots = self.item_slots.to(carrier.dtype).unsqueeze(0).expand(
            batch * n_rows, -1, -1
        )
        item_slots = self.item_slot_mab(
            item_slots,
            rows,
            source_mask=row_mask,
            zero_when_no_support=True,
        ).state
        rows = self.item_token_mab(rows, item_slots).state
        return rows.reshape(batch, n_rows, n_columns, d_model)


class AugmentedDynamics(nn.Module):
    plan = DynamicsPlan(
        name="augmented_three_omab",
        stages=("slot_from_visible", "token_from_slot", "row_axis"),
        carrier="(N+K)x(M+K)",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            InducedCarrierBlock(config) for _ in range(config.n_blocks)
        )

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


class SupervisedDynamics(AugmentedDynamics):
    plan = DynamicsPlan(
        name="oinject_augmented_three_omab",
        stages=("multi_label_oinject", "slot_from_context", "token_from_slot", "row_axis"),
        carrier="(N+K)x(M+K)",
    )


class PredictorUnitAddressDynamics(nn.Module):
    """Build one shared query Unit from same-row visible predictors only."""

    plan = DynamicsPlan(
        name="predictor_unit_address_omab_v2",
        stages=("shared_unit_query", "visible_predictor_aggregation"),
        carrier="row predictor cells -> shared Unit slots",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.unit_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.aggregate = _block(config)
        nn.init.normal_(self.unit_seeds, std=0.02)

    def forward(
        self,
        typed_cells: Tensor,
        *,
        visible_predictor_mask: Tensor,
    ) -> Tensor:
        if typed_cells.ndim != 4:
            raise ValueError("typed_cells must be [B,N,M,D]")
        if (
            visible_predictor_mask.shape != typed_cells.shape[:3]
            or visible_predictor_mask.dtype is not torch.bool
        ):
            raise ValueError("visible predictor mask must be bool [B,N,M]")
        batch, n_rows, n_features, d_model = typed_cells.shape
        sources = torch.where(
            visible_predictor_mask.unsqueeze(-1),
            typed_cells,
            torch.zeros_like(typed_cells),
        ).reshape(batch * n_rows, n_features, d_model)
        source_mask = visible_predictor_mask.reshape(batch * n_rows, n_features)
        queries = self.unit_seeds.to(typed_cells.dtype).unsqueeze(0).expand(
            batch * n_rows, -1, -1
        )
        units = self.aggregate(
            queries,
            sources,
            source_mask=source_mask,
            zero_when_no_support=True,
        ).state
        return units.reshape(batch, n_rows, units.shape[1], d_model)


class RowUnitDynamics(AugmentedDynamics):
    plan = DynamicsPlan(
        name="row_unit_three_omab",
        stages=("slot_from_visible", "token_from_slot", "row_axis"),
        carrier="Nx(M+K)",
    )


class PairUnitDynamics(AugmentedDynamics):
    plan = DynamicsPlan(
        name="pair_unit_three_omab",
        stages=("slot_from_col_mates", "unit_from_slot", "row_mates"),
        carrier="NxM",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList(
            InducedCarrierBlock(config, exclude_row_self=True)
            for _ in range(config.n_blocks)
        )
