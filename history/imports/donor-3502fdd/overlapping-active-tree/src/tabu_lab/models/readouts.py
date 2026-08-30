"""Explicit Step 4 geometry/readout components."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import (
    BilinearNumericLocalLinear,
    BilinearNumericNW,
    GlobalUserItemNumericLocalLinear,
    GlobalUserItemNumericNW,
    NumericReadoutOutput,
    RoutingOutput,
    SameColumnNumericLocalLinear,
    SameColumnNumericNW,
)

from .types import ReferenceConfig


def _numeric_terminal(
    config: ReferenceConfig,
    *,
    bilinear_support: bool,
    numeric_terminal: str,
) -> tuple[nn.Module, str]:
    """Instantiate the selected numeric kernel without changing geometry.

    LL and NW share the same routing support contract.  The only difference
    is the value estimator applied after support weights have been formed.
    ``numeric_terminal`` is deliberately passed through every readout rather
    than hidden in a model-specific branch, so one semantic config axis can
    produce either paired variant.
    """

    value = getattr(numeric_terminal, "value", numeric_terminal)
    if value == "nadaraya_watson":
        terminal = (
            BilinearNumericNW(config.routing_bandwidth)
            if bilinear_support
            else SameColumnNumericNW(config.routing_bandwidth)
        )
    elif value == "local_linear":
        terminal = (
            BilinearNumericLocalLinear(config.routing_bandwidth)
            if bilinear_support
            else SameColumnNumericLocalLinear(config.routing_bandwidth)
        )
    else:
        raise ValueError(f"unknown numeric terminal: {numeric_terminal!r}")
    return terminal, value


def _gauge_stable_bilinear(
    left: Tensor,
    right: Tensor,
    *,
    center_dimensions: tuple[int, ...],
) -> Tensor:
    """Evaluate a bilinear term after removing an exact routing gauge.

    In exact arithmetic this is ``<left, right> - <left_anchor, right_anchor>``.
    The removed term is constant along every centered routing axis, so the
    subsequent centering is semantically identical while avoiding a large
    float32 dot-product followed by catastrophic cancellation.
    """

    left_anchor = left.mean(dim=center_dimensions, keepdim=True)
    right_anchor = right.mean(dim=center_dimensions, keepdim=True)
    return ((left - left_anchor) * right).sum(dim=-1) + (left_anchor * (right - right_anchor)).sum(
        dim=-1
    )


class MatchedUFReadout(nn.Module):
    """Matched ``diag(U F^T)`` coordinates with same-column numeric NW."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        bilinear_support: bool = False,
        geometry: str = "matched_uf",
        recommendation_address_plan: str = "matched_uf",
        rec_axis_summary_dim: int = 2,
        rec_matched_residual_scale: float = 0.1,
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__()
        if geometry not in {"matched_uf", "matched_ufc"}:
            raise ValueError("geometry must be matched_uf or matched_ufc")
        self.geometry = geometry
        self.geometry_normalization = config.geometry_normalization
        self.d_model = config.d_model
        self.bilinear_support = bool(bilinear_support)
        self.uses_empirical_support = bool(bilinear_support)
        if recommendation_address_plan not in {
            "matched_uf",
            "axis_address_bootstrap_v1",
        }:
            raise ValueError("unknown recommendation address plan")
        if recommendation_address_plan != "matched_uf" and not bilinear_support:
            raise ValueError("recommendation address plans require bilinear support")
        self.recommendation_address_plan = recommendation_address_plan
        self.axis_bootstrap = (
            AxisAddressBootstrap(
                config.d_model,
                summary_dim=rec_axis_summary_dim,
                matched_residual_scale=rec_matched_residual_scale,
            )
            if recommendation_address_plan == "axis_address_bootstrap_v1"
            else None
        )
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=bilinear_support,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def coordinates(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        cells: Tensor | None = None,
    ) -> Tensor:
        if unit_tokens.ndim != 4 or feature_tokens.ndim != 4:
            raise ValueError("unit/feature tokens must be [B,N,K,D] and [B,M,K,D]")
        if unit_tokens.shape[0] != feature_tokens.shape[0]:
            raise ValueError("unit/feature batches must match")
        if unit_tokens.shape[2:] != feature_tokens.shape[2:]:
            raise ValueError("matched slot and token axes must match")
        # Device-resident geometry follows one float32 path on CPU, CUDA, and
        # MPS.  Normalization and centering control the address gauge without
        # introducing a backend-specific precision branch.
        work_dtype = DEFAULT_FLOAT_DTYPE
        work_units = unit_tokens.to(dtype=work_dtype)
        work_features = feature_tokens.to(dtype=work_dtype)
        if self.geometry_normalization == "rms_unit":
            # Normalize each matched token, rather than the final coordinates
            # or its hidden dimensions.  This removes the scale gauge while
            # retaining angular information and is deliberately local to the
            # address-forming boundary.
            scale = math.sqrt(self.d_model)
            work_units = F.normalize(work_units, p=2.0, dim=-1, eps=1.0e-6) * scale
            work_features = F.normalize(work_features, p=2.0, dim=-1, eps=1.0e-6) * scale
        # Preserve the v0 ``none`` trajectory exactly.  The RMS plan carries
        # its own conventional ``1/sqrt(d)`` matched-coordinate scale and is
        # therefore a separately hashed semantic configuration.
        coordinate_scale = (
            math.sqrt(self.d_model) if self.geometry_normalization == "rms_unit" else 1.0
        )
        center_dimensions = (1, 2) if self.bilinear_support else (1,)
        matched = (
            _gauge_stable_bilinear(
                work_units.unsqueeze(2),
                work_features.unsqueeze(1),
                center_dimensions=center_dimensions,
            )
            / coordinate_scale
        )
        if self.geometry == "matched_uf":
            raw = matched
        else:
            if cells is None or cells.shape != (
                unit_tokens.shape[0],
                unit_tokens.shape[1],
                feature_tokens.shape[1],
                unit_tokens.shape[-1],
            ):
                raise ValueError("matched_ufc geometry requires evolved [B,N,M,D] cells")
            work_cells = cells.to(dtype=work_dtype)
            unit_cell = (
                _gauge_stable_bilinear(
                    work_units.unsqueeze(2),
                    work_cells.unsqueeze(3),
                    center_dimensions=center_dimensions,
                )
                / coordinate_scale
            )
            feature_cell = (
                _gauge_stable_bilinear(
                    work_features.unsqueeze(1),
                    work_cells.unsqueeze(3),
                    center_dimensions=center_dimensions,
                )
                / coordinate_scale
            )
            raw = matched + unit_cell + feature_cell
        # Translation is an exact gauge freedom of RBF routing.  Same-column
        # models may center each Feature independently.  Rec compares both
        # within-item and within-user addresses, so it uses one response-family
        # center shared by the complete matrix.
        return raw - raw.mean(dim=center_dimensions, keepdim=True)

    def forward(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates, output, _ = self.forward_with_auxiliaries(
            unit_tokens,
            feature_tokens,
            support_values,
            visible_mask,
            cells,
        )
        return coordinates, output

    def forward_with_auxiliaries(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput, dict[str, Tensor]]:
        coordinates = self.coordinates(unit_tokens, feature_tokens, cells)
        auxiliaries: dict[str, Tensor] = {}
        if self.axis_bootstrap is not None:
            matched_coordinates = coordinates
            coordinates, user_summary, item_summary = self.axis_bootstrap(
                matched_coordinates,
                support_values,
                visible_mask,
            )
            auxiliaries = {
                "rec_matched_coordinates": matched_coordinates,
                "rec_user_axis_summary": user_summary,
                "rec_item_axis_summary": item_summary,
            }
        return (
            coordinates,
            self.terminal(coordinates, support_values, visible_mask),
            auxiliaries,
        )


class MatchedScoreReadout(nn.Module):
    """The TabU4Rec mainline parameterized matched score.

    The current model-factory contract defines Step 4 as
    ``z[r,a,k] = <u[r,k], f[k,a]>`` followed by ``sum_k z[r,a,k]``.  It is
    deliberately independent of empirical response values and therefore has
    no support arm or learned readout terminal.  The dense public API still
    returns an empty routing ledger so PredictionBundle remains typed and
    auditable without manufacturing support mass.
    """

    recommendation_address_plan = "matched_uf"
    geometry = "matched_uf"
    bilinear_support = False
    uses_empirical_support = False
    numeric_terminal = "parameterized_matching"
    numeric_terminal_trace = "matching_score"

    def __init__(self, config: ReferenceConfig, *, numeric_terminal: str | None = None) -> None:
        super().__init__()
        # ``numeric_terminal`` is accepted for builder compatibility, but the
        # mainline matching score is not an empirical numeric kernel.
        del numeric_terminal
        self.geometry_normalization = config.geometry_normalization
        self.d_model = config.d_model

    def coordinates(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        cells: Tensor | None = None,
    ) -> Tensor:
        del cells
        if unit_tokens.ndim != 4 or feature_tokens.ndim != 4:
            raise ValueError("unit/feature tokens must be [B,N,K,D] and [B,M,K,D]")
        if unit_tokens.shape[0] != feature_tokens.shape[0]:
            raise ValueError("unit/feature batches must match")
        if unit_tokens.shape[2:] != feature_tokens.shape[2:]:
            raise ValueError("matched slot and token axes must match")

        work_units = unit_tokens.to(dtype=DEFAULT_FLOAT_DTYPE)
        work_features = feature_tokens.to(dtype=DEFAULT_FLOAT_DTYPE)
        coordinate_scale = 1.0
        if self.geometry_normalization == "rms_unit":
            scale = math.sqrt(self.d_model)
            work_units = F.normalize(work_units, p=2.0, dim=-1, eps=1.0e-6) * scale
            work_features = F.normalize(work_features, p=2.0, dim=-1, eps=1.0e-6) * scale
            coordinate_scale = scale

        # Do not center this geometry.  The source contract defines the score
        # as the literal paired inner product, not a routing gauge.
        return (
            torch.einsum(
                "bnkd,bmkd->bnmk",
                work_units,
                work_features,
            ).to(dtype=unit_tokens.dtype)
            / coordinate_scale
        )

    def forward_with_auxiliaries(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput, dict[str, Tensor]]:
        del support_values, visible_mask
        coordinates = self.coordinates(unit_tokens, feature_tokens, cells)
        batch, n_rows, n_features, _ = coordinates.shape
        empty_shape = (batch, n_rows, n_features, 0)
        empty_weights = coordinates.new_zeros(empty_shape)
        empty_support = torch.zeros(
            empty_shape,
            dtype=torch.bool,
            device=coordinates.device,
        )
        routing = RoutingOutput(
            weights=empty_weights,
            log_weights=empty_weights,
            support_mask=empty_support,
            support_available=torch.zeros(
                batch,
                n_rows,
                n_features,
                dtype=torch.bool,
                device=coordinates.device,
            ),
            support_count=torch.zeros(
                batch,
                n_rows,
                n_features,
                dtype=torch.long,
                device=coordinates.device,
            ),
        )
        return (
            coordinates,
            NumericReadoutOutput(
                values=coordinates.sum(dim=-1),
                support_available=torch.ones(
                    batch,
                    n_rows,
                    n_features,
                    dtype=torch.bool,
                    device=coordinates.device,
                ),
                routing=routing,
            ),
            {},
        )

    def forward(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates, output, _ = self.forward_with_auxiliaries(
            unit_tokens,
            feature_tokens,
            support_values,
            visible_mask,
            cells,
        )
        return coordinates, output


class CellTokenGlobalSupportReadout(nn.Module):
    """Unit-as-cell geometry with one jointly normalized user/item pool."""

    recommendation_address_plan = "cell_global_support_v1"
    geometry = "cell_token"
    bilinear_support = True
    uses_empirical_support = True

    def __init__(self, config: ReferenceConfig, *, numeric_terminal: str = "local_linear") -> None:
        super().__init__()
        self.projection = nn.Linear(
            config.d_model, config.matched_slots, bias=False, dtype=DEFAULT_FLOAT_DTYPE
        )
        value = getattr(numeric_terminal, "value", numeric_terminal)
        if value == "local_linear":
            self.terminal = GlobalUserItemNumericLocalLinear(config.routing_bandwidth)
        elif value == "nadaraya_watson":
            self.terminal = GlobalUserItemNumericNW(config.routing_bandwidth)
        else:
            raise ValueError(f"unknown numeric terminal: {numeric_terminal!r}")
        self.numeric_terminal = value
        self.numeric_terminal_trace = "local_linear" if value == "local_linear" else "nw"

    def forward_with_auxiliaries(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput, dict[str, Tensor]]:
        del unit_tokens, feature_tokens
        if cells is None or cells.ndim != 4:
            raise ValueError("cell-token readout requires evolved cells [B,N,M,D]")
        coordinates = F.linear(
            cells.to(dtype=DEFAULT_FLOAT_DTYPE),
            self.projection.weight.to(dtype=DEFAULT_FLOAT_DTYPE),
        )
        coordinates = coordinates.to(cells.dtype)
        output = self.terminal(coordinates, support_values, visible_mask)
        batch, n_rows, n_features, _ = coordinates.shape
        weights = output.routing.weights.reshape(batch, n_rows, n_features, n_rows, n_features)
        row_ids = torch.arange(n_rows, device=coordinates.device).view(1, 1, 1, n_rows, 1)
        col_ids = torch.arange(n_features, device=coordinates.device).view(1, 1, 1, 1, n_features)
        query_rows = torch.arange(n_rows, device=coordinates.device).view(1, n_rows, 1, 1, 1)
        query_cols = torch.arange(n_features, device=coordinates.device).view(
            1, 1, n_features, 1, 1
        )
        user_weights = torch.where(col_ids == query_cols, weights, torch.zeros_like(weights)).sum(
            dim=-1
        )
        item_weights = torch.where(row_ids == query_rows, weights, torch.zeros_like(weights)).sum(
            dim=-2
        )
        user_mass = user_weights.sum(dim=-1)
        item_mass = item_weights.sum(dim=-1)
        user_available, item_available = user_mass > 0, item_mass > 0
        user_values = (
            user_weights.to(support_values.dtype) * support_values.permute(0, 2, 1).unsqueeze(1)
        ).sum(dim=-1) / user_mass.clamp_min(torch.finfo(user_mass.dtype).tiny).to(
            support_values.dtype
        )
        item_values = (item_weights.to(support_values.dtype) * support_values.unsqueeze(2)).sum(
            dim=-1
        ) / item_mass.clamp_min(torch.finfo(item_mass.dtype).tiny).to(support_values.dtype)
        return (
            coordinates,
            output,
            {
                "rec_global_support_weights": output.routing.weights,
                "rec_user_arm_support_weights": user_weights,
                "rec_item_arm_support_weights": item_weights,
                "rec_user_arm_support_available": user_available,
                "rec_item_arm_support_available": item_available,
                "rec_arm_weights": torch.stack((user_mass, item_mass), dim=-1),
                "rec_user_arm_values": torch.where(
                    user_available, user_values, torch.zeros_like(user_values)
                ),
                "rec_item_arm_values": torch.where(
                    item_available, item_values, torch.zeros_like(item_values)
                ),
            },
        )

    def forward(
        self,
        unit_tokens: Tensor,
        feature_tokens: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        cells: Tensor | None = None,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates, output, _ = self.forward_with_auxiliaries(
            unit_tokens, feature_tokens, support_values, visible_mask, cells
        )
        return coordinates, output


class AxisAddressBootstrap(nn.Module):
    """Truth-free, identifier-free user/item summaries for Rec routing."""

    def __init__(
        self,
        d_model: int,
        *,
        summary_dim: int = 2,
        matched_residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if summary_dim <= 0:
            raise ValueError("axis summary_dim must be positive")
        if matched_residual_scale <= 0.0:
            raise ValueError("matched residual scale must be positive")
        self.summary_dim = int(summary_dim)
        self.matched_residual_scale = float(matched_residual_scale)
        self.user_encoder = nn.Sequential(
            nn.Linear(1, d_model, dtype=DEFAULT_FLOAT_DTYPE),
            nn.GELU(),
            nn.Linear(d_model, summary_dim, dtype=DEFAULT_FLOAT_DTYPE),
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(1, d_model, dtype=DEFAULT_FLOAT_DTYPE),
            nn.GELU(),
            nn.Linear(d_model, summary_dim, dtype=DEFAULT_FLOAT_DTYPE),
        )

    def forward(
        self,
        matched_coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if matched_coordinates.ndim != 4:
            raise ValueError("matched coordinates must be [B,N,M,K]")
        if support_values.shape != matched_coordinates.shape[:3]:
            raise ValueError("axis bootstrap support_values must be [B,N,M]")
        if visible_mask.shape != support_values.shape or visible_mask.dtype is not torch.bool:
            raise ValueError("axis bootstrap visible_mask must be bool [B,N,M]")
        encoded_user = self.user_encoder(support_values.unsqueeze(-1))
        encoded_item = self.item_encoder(support_values.unsqueeze(-1))
        source = visible_mask.unsqueeze(-1)
        encoded_user = torch.where(source, encoded_user, torch.zeros_like(encoded_user))
        encoded_item = torch.where(source, encoded_item, torch.zeros_like(encoded_item))
        user_count = visible_mask.sum(dim=2, keepdim=True).clamp_min(1)
        item_count = visible_mask.sum(dim=1).clamp_min(1)
        user_summary = encoded_user.sum(dim=2) / user_count.to(encoded_user.dtype)
        item_summary = encoded_item.sum(dim=1) / item_count.unsqueeze(-1).to(encoded_item.dtype)
        batch, n_users, n_items = support_values.shape
        user_axis = user_summary.unsqueeze(2).expand(batch, n_users, n_items, -1)
        item_axis = item_summary.unsqueeze(1).expand(batch, n_users, n_items, -1)
        bounded_matched = self.matched_residual_scale * torch.tanh(matched_coordinates)
        coordinates = torch.cat((bounded_matched, user_axis, item_axis), dim=-1)
        return coordinates, user_summary, item_summary


class PredictorOnlyLabelReadout(nn.Module):
    """Per-label Unit address from same-row visible predictor typed tokens."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        n_labels: int,
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__()
        if n_labels <= 0:
            raise ValueError("predictor-only label readout requires at least one label")
        self.n_labels = int(n_labels)
        self.max_features = config.max_features
        self.projection = nn.Parameter(
            torch.empty(
                n_labels,
                config.max_features,
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.projection, std=0.05)
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=False,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def coordinates(
        self,
        typed_cells: Tensor,
        visible_predictor_mask: Tensor,
        *,
        label_columns: tuple[int, ...],
    ) -> Tensor:
        if typed_cells.ndim != 4:
            raise ValueError("typed_cells must be [B,N,M,D]")
        if (
            visible_predictor_mask.shape != typed_cells.shape[:3]
            or visible_predictor_mask.dtype is not torch.bool
        ):
            raise ValueError("visible predictor mask must be bool [B,N,M]")
        _, _, n_features, _ = typed_cells.shape
        if n_features > self.max_features:
            raise ValueError("label address exceeds max_features")
        if len(label_columns) != self.n_labels or len(set(label_columns)) != self.n_labels:
            raise ValueError("label columns must match the per-label projection count")
        if any(index < 0 or index >= n_features for index in label_columns):
            raise ValueError("label column is outside the typed feature axis")
        predictor_cells = torch.where(
            visible_predictor_mask.unsqueeze(-1),
            typed_cells,
            torch.zeros_like(typed_cells),
        )
        projection = self.projection[:, :n_features]
        addresses = torch.einsum("bnmd,lmkd->bnlk", predictor_cells, projection)
        counts = visible_predictor_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        addresses = addresses / counts.unsqueeze(-1).to(addresses.dtype).sqrt()
        addresses = addresses - addresses.mean(dim=1, keepdim=True)
        coordinates = typed_cells.new_zeros(
            typed_cells.shape[0],
            typed_cells.shape[1],
            n_features,
            addresses.shape[-1],
        )
        coordinates[:, :, list(label_columns)] = addresses
        return coordinates

    def forward(
        self,
        typed_cells: Tensor,
        visible_predictor_mask: Tensor,
        support_values: Tensor,
        label_source_mask: Tensor,
        *,
        label_columns: tuple[int, ...],
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates = self.coordinates(
            typed_cells,
            visible_predictor_mask,
            label_columns=label_columns,
        )
        return coordinates, self.terminal(
            coordinates,
            support_values,
            label_source_mask,
        )


class PredictorUnitLinkedLabelReadout(nn.Module):
    """Per-label address with a mandatory predictor-derived Unit residual."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        n_labels: int,
        unit_residual_scale: float = 0.1,
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__()
        if n_labels <= 0:
            raise ValueError("unit-linked label readout requires at least one label")
        if unit_residual_scale <= 0.0:
            raise ValueError("unit residual scale must be positive")
        self.n_labels = int(n_labels)
        self.max_features = config.max_features
        self.matched_slots = config.matched_slots
        self.unit_residual_scale = float(unit_residual_scale)
        self.predictor_projection = nn.Parameter(
            torch.empty(
                n_labels,
                config.max_features,
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.unit_projection = nn.Parameter(
            torch.empty(
                n_labels,
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.predictor_projection, std=0.05)
        nn.init.normal_(self.unit_projection, std=0.05)
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=False,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def coordinates(
        self,
        typed_cells: Tensor,
        visible_predictor_mask: Tensor,
        predictor_units: Tensor,
        *,
        label_columns: tuple[int, ...],
    ) -> Tensor:
        if typed_cells.ndim != 4:
            raise ValueError("typed_cells must be [B,N,M,D]")
        if (
            visible_predictor_mask.shape != typed_cells.shape[:3]
            or visible_predictor_mask.dtype is not torch.bool
        ):
            raise ValueError("visible predictor mask must be bool [B,N,M]")
        batch, n_rows, n_features, d_model = typed_cells.shape
        if predictor_units.shape != (
            batch,
            n_rows,
            self.matched_slots,
            d_model,
        ):
            raise ValueError("predictor Units must be [B,N,K,D]")
        if n_features > self.max_features:
            raise ValueError("unit-linked label address exceeds max_features")
        if len(label_columns) != self.n_labels or len(set(label_columns)) != self.n_labels:
            raise ValueError("label columns must match the per-label projection count")
        if any(index < 0 or index >= n_features for index in label_columns):
            raise ValueError("label column is outside the typed feature axis")

        predictor_cells = torch.where(
            visible_predictor_mask.unsqueeze(-1),
            typed_cells,
            torch.zeros_like(typed_cells),
        )
        direct = torch.einsum(
            "bnmd,lmkd->bnlk",
            predictor_cells,
            self.predictor_projection[:, :n_features],
        )
        counts = visible_predictor_mask.sum(dim=-1, keepdim=True).clamp_min(1)
        direct = direct / counts.unsqueeze(-1).to(direct.dtype).sqrt()
        unit_address = torch.einsum(
            "bnkd,lkd->bnlk",
            predictor_units,
            self.unit_projection,
        )
        addresses = direct + self.unit_residual_scale * torch.tanh(unit_address)
        addresses = addresses - addresses.mean(dim=1, keepdim=True)
        coordinates = typed_cells.new_zeros(
            batch,
            n_rows,
            n_features,
            addresses.shape[-1],
        )
        coordinates[:, :, list(label_columns)] = addresses
        return coordinates

    def forward(
        self,
        typed_cells: Tensor,
        visible_predictor_mask: Tensor,
        predictor_units: Tensor,
        support_values: Tensor,
        label_source_mask: Tensor,
        *,
        label_columns: tuple[int, ...],
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates = self.coordinates(
            typed_cells,
            visible_predictor_mask,
            predictor_units,
            label_columns=label_columns,
        )
        return coordinates, self.terminal(
            coordinates,
            support_values,
            label_source_mask,
        )


class RowUnitReadout(nn.Module):
    """``<u_k, cell>`` coordinates for Unit-as-row TabU."""

    def __init__(
        self, config: ReferenceConfig, *, numeric_terminal: str = "nadaraya_watson"
    ) -> None:
        super().__init__()
        self.geometry_normalization = config.geometry_normalization
        self.d_model = config.d_model
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=False,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def forward(
        self,
        unit_tokens: Tensor,
        cells: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        work_dtype = DEFAULT_FLOAT_DTYPE
        work_units = unit_tokens.to(dtype=work_dtype)
        work_cells = cells.to(dtype=work_dtype)
        coordinate_scale = 1.0
        if self.geometry_normalization == "rms_unit":
            scale = math.sqrt(self.d_model)
            work_units = F.normalize(work_units, p=2.0, dim=-1, eps=1.0e-6) * scale
            work_cells = F.normalize(work_cells, p=2.0, dim=-1, eps=1.0e-6) * scale
            coordinate_scale = scale
        raw = (
            _gauge_stable_bilinear(
                work_units.unsqueeze(2),
                work_cells.unsqueeze(3),
                center_dimensions=(1,),
            )
            / coordinate_scale
        )
        coordinates = raw - raw.mean(dim=1, keepdim=True)
        return coordinates, self.terminal(coordinates, support_values, visible_mask)


class PairUnitReadout(nn.Module):
    """Learned ``W c`` coordinates for the Unit-as-cell TabU contract.

    The class name remains ``PairUnitReadout`` for runtime/API continuity with the
    frozen ``tabu.unit_pair`` contract ID; the semantic object is a cell, not a
    second model family.
    """

    def __init__(self, config: ReferenceConfig, *, numeric_terminal: str = "local_linear") -> None:
        super().__init__()
        self.projection = nn.Linear(
            config.d_model,
            config.matched_slots,
            bias=False,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=False,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def forward(
        self,
        cells: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        work_dtype = DEFAULT_FLOAT_DTYPE
        work_cells = cells.to(dtype=work_dtype)
        raw = F.linear(
            work_cells,
            self.projection.weight.to(dtype=work_dtype),
        )
        # The Unit-as-cell contract is z_ra = W c_ra.  Any centering or gauge
        # transform is a separate experimental variant, not part of default.
        return raw, self.terminal(raw, support_values, visible_mask)


class CellSpecialReadout(nn.Module):
    """Axis-B row/column special projection followed by a typed terminal.

    ``row`` emits $U_r c_{ra}$, ``column`` emits $F_a c_{ra}$, and ``row_column``
    concatenates the two direct-sum coordinates.  This is deliberately not a
    matched-UF readout: the cell state participates in both inner products.
    """

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        mode: str,
        numeric_terminal: str = "local_linear",
    ) -> None:
        super().__init__()
        if mode not in {"row", "column", "row_column"}:
            raise ValueError("mode must be row, column, or row_column")
        self.mode = mode
        self.special_width = config.matched_slots
        self.width = self.special_width * (2 if mode == "row_column" else 1)
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            bilinear_support=False,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    def coordinates(self, carrier: Tensor, *, n_rows: int, n_features: int) -> Tensor:
        if carrier.ndim != 4:
            raise ValueError("carrier must be [B,R,C,D]")
        batch, carrier_rows, carrier_columns, _ = carrier.shape
        base_k = self.special_width
        expected_rows = n_rows + (base_k if self.mode in {"column", "row_column"} else 0)
        expected_columns = n_features + (base_k if self.mode in {"row", "row_column"} else 0)
        if (carrier_rows, carrier_columns) != (expected_rows, expected_columns):
            raise ValueError("carrier shape does not match the selected cell-special mode")
        cells = carrier[:, :n_rows, :n_features]
        pieces: list[Tensor] = []
        if self.mode in {"row", "row_column"}:
            row_specials = carrier[:, :n_rows, n_features : n_features + base_k]
            pieces.append(torch.einsum("bnkd,bnmd->bnmk", row_specials, cells))
        if self.mode in {"column", "row_column"}:
            column_specials = carrier[:, n_rows : n_rows + base_k, :n_features]
            pieces.append(torch.einsum("bkmd,bnmd->bnmk", column_specials, cells))
        return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]

    def forward(
        self,
        carrier: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        *,
        n_rows: int,
        n_features: int,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        coordinates = self.coordinates(carrier, n_rows=n_rows, n_features=n_features)
        return coordinates, self.terminal(coordinates, support_values, visible_mask)


class CellMatchingReadout(nn.Module):
    """Axis-B Rec ``M`` profile: matched user/item specials, no support arm."""

    numeric_terminal = "parameterized_matching"
    numeric_terminal_trace = "matching"
    uses_empirical_support = False

    def __init__(self) -> None:
        super().__init__()

    def coordinates(self, carrier: Tensor, *, n_rows: int, n_features: int) -> Tensor:
        if carrier.ndim != 4:
            raise ValueError("carrier must be [B,R,C,D]")
        batch, carrier_rows, carrier_columns, _ = carrier.shape
        k = carrier_rows - n_rows
        if k <= 0 or carrier_columns != n_features + k:
            raise ValueError("matching profile requires a square-special carrier")
        users = carrier[:, n_rows:, :n_features]
        items = carrier[:, :n_rows, n_features:]
        # Pair the same slot on the two sides; the cell state is intentionally
        # excluded from this profile's Z constructor.
        return torch.einsum("bkmd,bnkd->bnmk", users, items)

    def forward(
        self,
        carrier: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        *,
        n_rows: int,
        n_features: int,
    ) -> tuple[Tensor, NumericReadoutOutput]:
        del support_values, visible_mask
        coordinates = self.coordinates(carrier, n_rows=n_rows, n_features=n_features)
        batch, rows, features, _ = coordinates.shape
        empty_shape = (batch, rows, features, 0)
        empty = coordinates.new_zeros(empty_shape)
        empty_mask = torch.zeros(empty_shape, dtype=torch.bool, device=coordinates.device)
        routing = RoutingOutput(
            weights=empty,
            log_weights=empty,
            support_mask=empty_mask,
            support_available=torch.ones(
                batch, rows, features, dtype=torch.bool, device=coordinates.device
            ),
            support_count=torch.zeros(
                batch, rows, features, dtype=torch.long, device=coordinates.device
            ),
        )
        return coordinates, NumericReadoutOutput(
            values=coordinates.sum(dim=-1),
            support_available=routing.support_available,
            routing=routing,
        )


__all__ = [
    "AxisAddressBootstrap",
    "CellTokenGlobalSupportReadout",
    "CellSpecialReadout",
    "CellMatchingReadout",
    "MatchedScoreReadout",
    "MatchedUFReadout",
    "PairUnitReadout",
    "PredictorOnlyLabelReadout",
    "PredictorUnitLinkedLabelReadout",
    "RowUnitReadout",
]
