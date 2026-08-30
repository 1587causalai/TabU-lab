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


class RoleAwareInducedCarrierBlock(InducedCarrierBlock):
    """Induced carrier block with explicit source/receiver role masks.

    The table-cell family has structural positions which may receive an axis
    update without becoming sources for the corresponding inducing stage.  A
    plain :class:`InducedCarrierBlock` cannot express that distinction because
    it updates every receiver.  This block keeps the proven slot/token/row
    operators but merges each stage through a typed receiver mask.
    """

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
        column_receiver_mask: Tensor,
        row_receiver_mask: Tensor,
    ) -> Tensor:
        if carrier.ndim != 4:
            raise ValueError("carrier must be [B,R,C,D]")
        batch, n_rows, n_columns, d_model = carrier.shape
        expected = (batch, n_rows, n_columns)
        masks = (
            column_source_mask,
            row_source_mask,
            column_receiver_mask,
            row_receiver_mask,
        )
        if any(mask.shape != expected for mask in masks):
            raise ValueError("cell-family masks must match carrier axes")
        if any(mask.dtype is not torch.bool for mask in masks):
            raise ValueError("cell-family masks must be bool")

        columns = carrier.permute(0, 2, 1, 3).reshape(batch * n_columns, n_rows, d_model)
        column_mask = column_source_mask.permute(0, 2, 1).reshape(batch * n_columns, n_rows)
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
        column_receivers = column_receiver_mask.permute(0, 2, 1).reshape(
            batch * n_columns, n_rows
        )
        column_states = torch.where(
            column_receivers.unsqueeze(-1), column_states, columns
        )
        token_carrier = column_states.reshape(batch, n_columns, n_rows, d_model).permute(
            0, 2, 1, 3
        ).contiguous()

        rows = token_carrier.reshape(batch * n_rows, n_columns, d_model)
        row_mask = row_source_mask.reshape(batch * n_rows, n_columns)
        row_pair_mask = None
        if self.exclude_row_self:
            row_pair_mask = ~torch.eye(
                n_columns,
                dtype=torch.bool,
                device=carrier.device,
            ).unsqueeze(0).expand(batch * n_rows, -1, -1)
        row_states = self.row_mab(
            rows,
            rows,
            source_mask=row_mask,
            pair_mask=row_pair_mask,
        ).state
        row_receivers = row_receiver_mask.reshape(batch * n_rows, n_columns)
        row_states = torch.where(row_receivers.unsqueeze(-1), row_states, rows)
        return row_states.reshape(batch, n_rows, n_columns, d_model)


class CellFamilyDynamics(nn.Module):
    """Three-stage OMAB dynamics for the axis-B cell-special carriers."""

    plan = DynamicsPlan(
        name="cell_family_three_omab",
        stages=("slot_from_visible", "token_from_slot", "row_axis"),
        carrier="role-aware cell carrier",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            RoleAwareInducedCarrierBlock(config, exclude_row_self=True)
            for _ in range(config.n_blocks)
        )

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
        column_receiver_mask: Tensor,
        row_receiver_mask: Tensor,
    ) -> Tensor:
        for block in self.blocks:
            carrier = block(
                carrier,
                column_source_mask=column_source_mask,
                row_source_mask=row_source_mask,
                column_receiver_mask=column_receiver_mask,
                row_receiver_mask=row_receiver_mask,
            )
        return carrier


class RecommendationDynamics(nn.Module):
    plan = DynamicsPlan(
        name="dual_axis_four_omab",
        stages=("user_slot", "user_token", "item_slot", "item_token"),
        carrier="(N+K)x(M+K)",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            DualAxisCarrierBlock(config) for _ in range(config.n_blocks)
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


class GraphLocalBlock(nn.Module):
    """Graph-local Unit-axis update plus row-local Feature mixing."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        unit_receiver_plan: str = "same_row_visible_cells",
    ) -> None:
        super().__init__()
        if unit_receiver_plan not in {
            "legacy_graph_units_only",
            "same_row_visible_cells",
        }:
            raise ValueError("unknown graph Unit receiver plan")
        self.unit_receiver_plan = unit_receiver_plan
        self.graph_mab = _block(config)
        self.row_mab = _block(config)
        self.unit_mab = _block(config)
        self.feature_mab = _block(config)

    def forward(
        self,
        cells: Tensor,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        *,
        visible_mask: Tensor,
        graph: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, n_rows, n_features, d_model = cells.shape
        if graph.ndim == 2:
            graph = graph.unsqueeze(0).expand(batch, -1, -1)
        graph = graph.to(torch.bool)
        graph = graph | graph.transpose(-1, -2)
        graph = graph | torch.eye(n_rows, dtype=torch.bool, device=graph.device).unsqueeze(0)

        feature_major = cells.permute(0, 2, 1, 3).reshape(
            batch * n_features, n_rows, d_model
        )
        feature_visible = visible_mask.permute(0, 2, 1).reshape(
            batch * n_features, n_rows
        )
        pair_mask = graph.unsqueeze(1).expand(batch, n_features, n_rows, n_rows).reshape(
            batch * n_features, n_rows, n_rows
        )
        feature_major = self.graph_mab(
            feature_major,
            feature_major,
            source_mask=feature_visible,
            pair_mask=pair_mask,
        ).state
        cells = feature_major.reshape(batch, n_features, n_rows, d_model).permute(
            0, 2, 1, 3
        ).contiguous()

        # Structural states remain explicit and contextual: Unit slots read
        # their graph neighborhood, Feature slots read their visible column.
        k = unit_tokens.shape[2]
        units = unit_tokens.permute(0, 2, 1, 3).reshape(batch * k, n_rows, d_model)
        unit_pair = graph.unsqueeze(1).expand(batch, k, n_rows, n_rows).reshape(
            batch * k, n_rows, n_rows
        )
        unit_tokens = self.unit_mab(units, units, pair_mask=unit_pair).state.reshape(
            batch, k, n_rows, d_model
        ).permute(0, 2, 1, 3)

        row_visible = visible_mask.reshape(batch * n_rows, n_features)
        if self.unit_receiver_plan == "legacy_graph_units_only":
            rows = cells.reshape(batch * n_rows, n_features, d_model)
            cells = self.row_mab(rows, rows, source_mask=row_visible).state.reshape(
                batch, n_rows, n_features, d_model
            )
        else:
            # The mathematical carrier applies row-axis OMAB to ordinary
            # cells and Unit-specials as receivers.  K/V remains strictly the
            # same row's visible ordinary values: the appended Unit-special
            # positions are receiver-only and therefore cannot create an
            # undeclared structural source path.
            row_receivers = torch.cat((cells, unit_tokens), dim=2).reshape(
                batch * n_rows, n_features + k, d_model
            )
            receiver_only_units = torch.zeros(
                batch * n_rows,
                k,
                dtype=torch.bool,
                device=visible_mask.device,
            )
            row_sources = torch.cat((row_visible, receiver_only_units), dim=1)
            row_states = self.row_mab(
                row_receivers,
                row_receivers,
                source_mask=row_sources,
            ).state.reshape(batch, n_rows, n_features + k, d_model)
            cells = row_states[:, :, :n_features]
            unit_tokens = row_states[:, :, n_features:]

        feature_queries = feature_tokens.permute(0, 1, 2, 3).reshape(
            batch * n_features, k, d_model
        )
        visible_sources = cells.permute(0, 2, 1, 3).reshape(
            batch * n_features, n_rows, d_model
        )
        feature_tokens = self.feature_mab(
            feature_queries,
            visible_sources,
            source_mask=feature_visible,
            zero_when_no_support=True,
        ).state.reshape(batch, n_features, k, d_model)
        return cells, unit_tokens, feature_tokens


class GraphDynamics(nn.Module):
    plan = DynamicsPlan(
        name="graph_four_stage",
        stages=(
            "target_feature_broadcast",
            "graph_local_unit_feature_evidence",
            "row_axis_feature_mix",
            "global_feature_prototype_for_readout",
        ),
        carrier="(N+K)x(M+K) views",
    )

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        unit_receiver_plan: str = "same_row_visible_cells",
    ) -> None:
        super().__init__()
        self.unit_receiver_plan = unit_receiver_plan
        self.blocks = nn.ModuleList(
            GraphLocalBlock(config, unit_receiver_plan=unit_receiver_plan)
            for _ in range(config.n_blocks)
        )

    def forward(
        self,
        cells: Tensor,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        *,
        visible_mask: Tensor,
        graph: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        for block in self.blocks:
            cells, unit_tokens, feature_tokens = block(
                cells,
                unit_tokens,
                feature_tokens,
                visible_mask=visible_mask,
                graph=graph,
            )
        return cells, unit_tokens, feature_tokens


__all__ = [
    "AugmentedDynamics",
    "CellFamilyDynamics",
    "DynamicsPlan",
    "GraphDynamics",
    "PairUnitDynamics",
    "PredictorUnitAddressDynamics",
    "RecommendationDynamics",
    "RoleAwareInducedCarrierBlock",
    "RowUnitDynamics",
    "SupervisedDynamics",
]
