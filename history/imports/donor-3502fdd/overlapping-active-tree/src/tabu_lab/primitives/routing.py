"""Typed, numerically safe routing and numeric kernel terminals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE


@dataclass(frozen=True)
class RoutingOutput:
    weights: Tensor
    log_weights: Tensor
    support_mask: Tensor
    support_available: Tensor
    support_count: Tensor


@dataclass(frozen=True)
class NumericReadoutOutput:
    values: Tensor
    support_available: Tensor
    routing: RoutingOutput


@dataclass(frozen=True)
class CategoricalReadoutOutput:
    values: Tensor
    probabilities: Tensor
    log_probabilities: Tensor
    class_support_available: Tensor
    domain_values: Tensor
    domain_mask: Tensor
    support_available: Tensor
    routing: RoutingOutput


def masked_rbf_weights(
    squared_distance: Tensor,
    allowed: Tensor,
    *,
    bandwidth: Tensor | float,
) -> RoutingOutput:
    """Normalize RBF weights without manufacturing mass for empty support."""

    if squared_distance.shape != allowed.shape or allowed.dtype is not torch.bool:
        raise ValueError("squared_distance and bool allowed masks must have identical shapes")
    tau = torch.as_tensor(bandwidth, dtype=squared_distance.dtype, device=squared_distance.device)
    if tau.numel() != 1 or float(tau.detach().cpu()) <= 0.0:
        raise ValueError("bandwidth must be one positive scalar")
    scores = -squared_distance / (2.0 * tau.square())
    limit = torch.finfo(scores.dtype).max
    scores = torch.nan_to_num(scores, nan=-limit, posinf=limit, neginf=-limit)
    masked = scores.masked_fill(~allowed, -torch.inf)
    offset = masked.amax(dim=-1, keepdim=True)
    offset = torch.where(torch.isfinite(offset), offset, torch.zeros_like(offset))
    shifted = masked - offset
    mass = torch.where(allowed, torch.exp(shifted), torch.zeros_like(masked))
    denominator = mass.sum(dim=-1, keepdim=True)
    available = allowed.any(dim=-1)
    weights = torch.where(
        available.unsqueeze(-1),
        mass / denominator.clamp_min(torch.finfo(mass.dtype).tiny),
        torch.zeros_like(mass),
    )
    finite_floor = -torch.finfo(scores.dtype).max
    log_weights = torch.where(
        allowed & available.unsqueeze(-1),
        shifted - denominator.clamp_min(torch.finfo(mass.dtype).tiny).log(),
        torch.full_like(scores, finite_floor),
    )
    return RoutingOutput(
        weights=weights,
        log_weights=log_weights,
        support_mask=allowed,
        support_available=available,
        support_count=allowed.sum(dim=-1),
    )


class SameColumnNumericNW(nn.Module):
    """Nadaraya--Watson readout over other visible cells in the same column."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        self.log_bandwidth = nn.Parameter(
            torch.tensor(float(bandwidth), dtype=DEFAULT_FLOAT_DTYPE).log()
        )

    @property
    def bandwidth(self) -> Tensor:
        return self.log_bandwidth.clamp(min=-13.8, max=13.8).exp()

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        if coordinates.ndim != 4:
            raise ValueError("coordinates must be [B,N,M,K]")
        if support_values.shape != coordinates.shape[:3]:
            raise ValueError("support_values must be [B,N,M]")
        if visible_mask.shape != support_values.shape or visible_mask.dtype is not torch.bool:
            raise ValueError("visible_mask must be bool [B,N,M]")
        batch, n_rows, n_features, _ = coordinates.shape
        supports = coordinates.permute(0, 2, 1, 3)
        # Keep routing on the repository-wide float32 execution path.  The
        # readout boundary centers/normalizes coordinates before this step.
        work_dtype = DEFAULT_FLOAT_DTYPE
        work_coordinates = coordinates.to(dtype=work_dtype)
        work_supports = supports.to(dtype=work_dtype)
        work_difference = work_coordinates.unsqueeze(3) - work_supports.unsqueeze(1)
        squared_distance = work_difference.square().sum(dim=-1)
        allowed = (
            visible_mask.permute(0, 2, 1).unsqueeze(1).expand(batch, n_rows, n_features, n_rows)
        )
        diagonal = torch.eye(n_rows, dtype=torch.bool, device=coordinates.device)
        allowed = allowed & ~diagonal.view(1, n_rows, 1, n_rows)
        routing = masked_rbf_weights(
            squared_distance,
            allowed,
            bandwidth=self.bandwidth.to(dtype=work_dtype),
        )
        routing = RoutingOutput(
            weights=routing.weights.to(coordinates.dtype),
            log_weights=routing.log_weights,
            support_mask=routing.support_mask,
            support_available=routing.support_available,
            support_count=routing.support_count,
        )
        source_values = support_values.permute(0, 2, 1).unsqueeze(1)
        values = (routing.weights.to(source_values.dtype) * source_values).sum(dim=-1)
        values = torch.where(routing.support_available, values, torch.zeros_like(values))
        return NumericReadoutOutput(
            values=values,
            support_available=routing.support_available,
            routing=routing,
        )


def _same_column_routing(
    coordinates: Tensor,
    visible_mask: Tensor,
    bandwidth: Tensor,
) -> RoutingOutput:
    """Build the same-column support ledger shared by local-linear readout."""

    batch, n_rows, n_features, _ = coordinates.shape
    work = coordinates.to(dtype=DEFAULT_FLOAT_DTYPE)
    supports = work.permute(0, 2, 1, 3)
    difference = work.unsqueeze(3) - supports.unsqueeze(1)
    squared_distance = difference.square().sum(dim=-1)
    allowed = visible_mask.permute(0, 2, 1).unsqueeze(1).expand(batch, n_rows, n_features, n_rows)
    diagonal = torch.eye(n_rows, dtype=torch.bool, device=coordinates.device)
    allowed = allowed & ~diagonal.view(1, n_rows, 1, n_rows)
    routing = masked_rbf_weights(squared_distance, allowed, bandwidth=bandwidth)
    return RoutingOutput(
        weights=routing.weights.to(coordinates.dtype),
        log_weights=routing.log_weights,
        support_mask=routing.support_mask,
        support_available=routing.support_available,
        support_count=routing.support_count,
    )


def _global_user_item_routing(
    coordinates: Tensor,
    visible_mask: Tensor,
    bandwidth: Tensor,
) -> RoutingOutput:
    """Route over the union of same-user and same-item visible cells.

    Unlike the dual-arm terminals below, this creates one support ledger and one
    normalization denominator per query.  The source axis remains flattened only
    internally; the public support mask records provenance through the row/column
    coordinates used to construct it.
    """

    if coordinates.ndim != 4:
        raise ValueError("coordinates must be [B,N,M,K]")
    if visible_mask.shape != coordinates.shape[:3] or visible_mask.dtype is not torch.bool:
        raise ValueError("visible_mask must be bool [B,N,M]")
    batch, n_rows, n_features, d_address = coordinates.shape
    n_cells = n_rows * n_features
    work = coordinates.to(dtype=DEFAULT_FLOAT_DTYPE)
    flat = work.reshape(batch, n_cells, d_address)
    difference = work.reshape(batch, n_cells, 1, d_address) - flat.unsqueeze(1)
    squared_distance = difference.square().sum(dim=-1)
    row_ids = torch.arange(n_rows, device=coordinates.device).repeat_interleave(n_features)
    feature_ids = torch.arange(n_features, device=coordinates.device).repeat(n_rows)
    same_axis = (row_ids.view(n_cells, 1) == row_ids.view(1, n_cells)) | (
        feature_ids.view(n_cells, 1) == feature_ids.view(1, n_cells)
    )
    allowed = visible_mask.reshape(batch, 1, n_cells).expand(batch, n_cells, n_cells)
    allowed = allowed & same_axis.view(1, n_cells, n_cells)
    allowed = allowed & ~torch.eye(n_cells, dtype=torch.bool, device=coordinates.device).view(
        1, n_cells, n_cells
    )
    routing = masked_rbf_weights(squared_distance, allowed, bandwidth=bandwidth)
    return RoutingOutput(
        weights=routing.weights.reshape(batch, n_rows, n_features, n_cells).to(coordinates.dtype),
        log_weights=routing.log_weights.reshape(batch, n_rows, n_features, n_cells),
        support_mask=routing.support_mask.reshape(batch, n_rows, n_features, n_cells),
        support_available=routing.support_available.reshape(batch, n_rows, n_features),
        support_count=routing.support_count.reshape(batch, n_rows, n_features),
    )


class GlobalUserItemNumericNW(nn.Module):
    """Nadaraya--Watson over one jointly normalized user/item support pool."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        self.log_bandwidth = nn.Parameter(
            torch.tensor(float(bandwidth), dtype=DEFAULT_FLOAT_DTYPE).log()
        )

    @property
    def bandwidth(self) -> Tensor:
        return self.log_bandwidth.clamp(min=-13.8, max=13.8).exp()

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        if support_values.shape != coordinates.shape[:3]:
            raise ValueError("support_values must be [B,N,M]")
        routing = _global_user_item_routing(
            coordinates,
            visible_mask,
            self.bandwidth.to(dtype=DEFAULT_FLOAT_DTYPE),
        )
        source_values = support_values.reshape(support_values.shape[0], -1).unsqueeze(1)
        values = (routing.weights.to(source_values.dtype) * source_values).sum(dim=-1)
        values = torch.where(routing.support_available, values, torch.zeros_like(values))
        return NumericReadoutOutput(
            values=values,
            support_available=routing.support_available,
            routing=routing,
        )


class GlobalUserItemNumericLocalLinear(nn.Module):
    """Local-linear terminal over one jointly normalized user/item support pool."""

    def __init__(self, bandwidth: float = 1.0, *, ridge: float = 1.0e-3) -> None:
        super().__init__()
        if bandwidth <= 0.0 or ridge <= 0.0:
            raise ValueError("bandwidth and ridge must be positive")
        self.log_bandwidth = nn.Parameter(
            torch.tensor(float(bandwidth), dtype=DEFAULT_FLOAT_DTYPE).log()
        )
        self.ridge = float(ridge)

    @property
    def bandwidth(self) -> Tensor:
        return self.log_bandwidth.clamp(min=-13.8, max=13.8).exp()

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        if support_values.shape != coordinates.shape[:3]:
            raise ValueError("support_values must be [B,N,M]")
        batch, n_rows, n_features, d_address = coordinates.shape
        work = coordinates.to(dtype=DEFAULT_FLOAT_DTYPE)
        flat = work.reshape(batch, n_rows * n_features, d_address)
        difference = (
            work.reshape(batch, n_rows * n_features, 1, d_address) - flat.unsqueeze(1)
        ).reshape(batch, n_rows, n_features, n_rows * n_features, d_address)
        routing = _global_user_item_routing(
            coordinates,
            visible_mask,
            self.bandwidth.to(dtype=DEFAULT_FLOAT_DTYPE),
        )
        source_values = support_values.reshape(batch, 1, 1, n_rows * n_features).expand(
            batch, n_rows, n_features, n_rows * n_features
        )
        values = _local_linear_values(
            routing,
            difference,
            source_values,
            ridge=self.ridge,
        )
        return NumericReadoutOutput(
            values=values,
            support_available=routing.support_available,
            routing=routing,
        )


def _local_linear_values(
    routing: RoutingOutput,
    difference: Tensor,
    source_values: Tensor,
    *,
    ridge: float,
) -> Tensor:
    """Evaluate a query-centered weighted local-linear ridge estimate."""

    if ridge <= 0.0:
        raise ValueError("local-linear ridge must be positive")
    work_dtype = DEFAULT_FLOAT_DTYPE
    weights = routing.weights.to(dtype=work_dtype)
    values = source_values.to(dtype=work_dtype)
    delta = difference.to(dtype=work_dtype)
    available = routing.support_available
    safe_weights = torch.where(available.unsqueeze(-1), weights, torch.zeros_like(weights))
    mean_x = (safe_weights.unsqueeze(-1) * delta).sum(dim=-2)
    mean_y = (safe_weights * values).sum(dim=-1)
    centered_x = delta - mean_x.unsqueeze(-2)
    centered_y = values - mean_y.unsqueeze(-1)
    covariance = torch.einsum(
        "bnmj,bnmjk,bnmjl->bnmkl",
        safe_weights,
        centered_x,
        centered_x,
    )
    k = delta.shape[-1]
    ridge_diag = torch.eye(k, dtype=work_dtype, device=delta.device) * float(ridge)
    covariance = covariance + ridge_diag.view(1, 1, 1, k, k)
    cross = torch.einsum(
        "bnmj,bnmjk,bnmj->bnmk",
        safe_weights,
        centered_x,
        centered_y,
    )
    slope = torch.linalg.solve(covariance, cross.unsqueeze(-1)).squeeze(-1)
    prediction = mean_y - (slope * mean_x).sum(dim=-1)
    return torch.where(available, prediction, torch.zeros_like(prediction)).to(source_values.dtype)


class SameColumnNumericLocalLinear(nn.Module):
    """Local-linear kernel prediction over other visible cells in one column."""

    def __init__(self, bandwidth: float = 1.0, *, ridge: float = 1.0e-3) -> None:
        super().__init__()
        if bandwidth <= 0.0 or ridge <= 0.0:
            raise ValueError("bandwidth and ridge must be positive")
        self.log_bandwidth = nn.Parameter(
            torch.tensor(float(bandwidth), dtype=DEFAULT_FLOAT_DTYPE).log()
        )
        self.ridge = float(ridge)

    @property
    def bandwidth(self) -> Tensor:
        return self.log_bandwidth.clamp(min=-13.8, max=13.8).exp()

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        if coordinates.ndim != 4:
            raise ValueError("coordinates must be [B,N,M,K]")
        if support_values.shape != coordinates.shape[:3]:
            raise ValueError("support_values must be [B,N,M]")
        if visible_mask.shape != support_values.shape or visible_mask.dtype is not torch.bool:
            raise ValueError("visible_mask must be bool [B,N,M]")
        routing = _same_column_routing(
            coordinates,
            visible_mask,
            self.bandwidth.to(dtype=DEFAULT_FLOAT_DTYPE),
        )
        work = coordinates.to(dtype=DEFAULT_FLOAT_DTYPE)
        difference = work.unsqueeze(3) - work.permute(0, 2, 1, 3).unsqueeze(1)
        source_values = support_values.permute(0, 2, 1).unsqueeze(1)
        values = _local_linear_values(
            routing,
            difference,
            source_values,
            ridge=self.ridge,
        )
        return NumericReadoutOutput(
            values=values,
            support_available=routing.support_available,
            routing=routing,
        )


class BilinearNumericNW(nn.Module):
    """Two-arm NW support for TabU4Rec.

    Column/user and row/item arms are normalized independently.  Active arms
    receive equal mass; an empty arm receives zero mass and the remaining arm
    is renormalized to one.
    """

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        self.column = SameColumnNumericNW(bandwidth)

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        column = self.column(coordinates, support_values, visible_mask)
        batch, n_rows, n_items, _ = coordinates.shape
        row_supports = coordinates.unsqueeze(2).expand(
            batch, n_rows, n_items, n_items, coordinates.shape[-1]
        )
        row_queries = coordinates.unsqueeze(3)
        work_dtype = DEFAULT_FLOAT_DTYPE
        row_difference = row_queries.to(dtype=work_dtype) - row_supports.to(dtype=work_dtype)
        row_distance = row_difference.square().sum(dim=-1)
        row_allowed = visible_mask.unsqueeze(2).expand(batch, n_rows, n_items, n_items)
        item_diagonal = torch.eye(n_items, dtype=torch.bool, device=coordinates.device)
        row_allowed = row_allowed & ~item_diagonal.view(1, 1, n_items, n_items)
        row_routing = masked_rbf_weights(
            row_distance,
            row_allowed,
            bandwidth=self.column.bandwidth.to(dtype=work_dtype),
        )
        row_routing = RoutingOutput(
            weights=row_routing.weights.to(coordinates.dtype),
            log_weights=row_routing.log_weights,
            support_mask=row_routing.support_mask,
            support_available=row_routing.support_available,
            support_count=row_routing.support_count,
        )
        row_values = (
            row_routing.weights.to(support_values.dtype) * support_values.unsqueeze(2)
        ).sum(dim=-1)
        n_active = column.support_available.to(torch.int64) + row_routing.support_available.to(
            torch.int64
        )
        column_mix = torch.where(
            column.support_available,
            n_active.clamp_min(1).reciprocal().to(column.values.dtype),
            torch.zeros_like(column.values),
        )
        row_mix = torch.where(
            row_routing.support_available,
            n_active.clamp_min(1).reciprocal().to(row_values.dtype),
            torch.zeros_like(row_values),
        )
        values = column_mix * column.values + row_mix * row_values
        available = n_active > 0
        # Preserve both arms in one last axis: [column supports | row supports].
        combined_weights = torch.cat(
            [
                column.routing.weights * column_mix.unsqueeze(-1),
                row_routing.weights * row_mix.unsqueeze(-1),
            ],
            dim=-1,
        )
        log_dtype = column.routing.log_weights.dtype
        finite_floor = -torch.finfo(log_dtype).max
        column_log_mix = column_mix.to(log_dtype).clamp_min(torch.finfo(log_dtype).tiny).log()
        row_log_mix = row_mix.to(log_dtype).clamp_min(torch.finfo(log_dtype).tiny).log()
        column_support_mask = column.routing.support_mask & column.support_available.unsqueeze(-1)
        row_support_mask = row_routing.support_mask & row_routing.support_available.unsqueeze(-1)
        combined_support_mask = torch.cat((column_support_mask, row_support_mask), dim=-1)
        combined_log_weights = torch.cat(
            (
                column.routing.log_weights + column_log_mix.unsqueeze(-1),
                row_routing.log_weights + row_log_mix.unsqueeze(-1),
            ),
            dim=-1,
        )
        combined_log_weights = torch.where(
            combined_support_mask,
            combined_log_weights,
            torch.full_like(combined_log_weights, finite_floor),
        )
        routing = RoutingOutput(
            weights=combined_weights,
            log_weights=combined_log_weights,
            support_mask=combined_support_mask,
            support_available=available,
            support_count=column.routing.support_count + row_routing.support_count,
        )
        return NumericReadoutOutput(values=values, support_available=available, routing=routing)


class BilinearNumericLocalLinear(nn.Module):
    """Two-arm local-linear kernel prediction for TabU4Rec.

    The user/item arms reuse the exact support ledger and bandwidth parameter
    of :class:`BilinearNumericNW`; only the value estimate on each active arm
    changes from a Nadaraya--Watson mean to a query-centred local-linear ridge
    fit.  This keeps support shape, empty-arm handling, and public routing
    identities invariant across the LL/NW pair.
    """

    def __init__(self, bandwidth: float = 1.0, *, ridge: float = 1.0e-3) -> None:
        super().__init__()
        self.column = SameColumnNumericLocalLinear(bandwidth, ridge=ridge)

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        column = self.column(coordinates, support_values, visible_mask)
        batch, n_rows, n_items, d_address = coordinates.shape
        row_supports = coordinates.unsqueeze(2).expand(batch, n_rows, n_items, n_items, d_address)
        row_queries = coordinates.unsqueeze(3)
        work_dtype = DEFAULT_FLOAT_DTYPE
        # The local fit is expressed in source-minus-query coordinates.  The
        # query-centred intercept in ``_local_linear_values`` is invariant to
        # flipping this sign together with its fitted slope.
        row_difference = row_supports.to(dtype=work_dtype) - row_queries.to(dtype=work_dtype)
        row_distance = row_difference.square().sum(dim=-1)
        row_allowed = visible_mask.unsqueeze(2).expand(batch, n_rows, n_items, n_items)
        item_diagonal = torch.eye(n_items, dtype=torch.bool, device=coordinates.device)
        row_allowed = row_allowed & ~item_diagonal.view(1, 1, n_items, n_items)
        row_routing = masked_rbf_weights(
            row_distance,
            row_allowed,
            bandwidth=self.column.bandwidth.to(dtype=work_dtype),
        )
        row_routing = RoutingOutput(
            weights=row_routing.weights.to(coordinates.dtype),
            log_weights=row_routing.log_weights,
            support_mask=row_routing.support_mask,
            support_available=row_routing.support_available,
            support_count=row_routing.support_count,
        )
        row_values = _local_linear_values(
            row_routing,
            row_difference,
            support_values.unsqueeze(2),
            ridge=self.column.ridge,
        )

        n_active = column.support_available.to(torch.int64) + row_routing.support_available.to(
            torch.int64
        )
        column_mix = torch.where(
            column.support_available,
            n_active.clamp_min(1).reciprocal().to(column.values.dtype),
            torch.zeros_like(column.values),
        )
        row_mix = torch.where(
            row_routing.support_available,
            n_active.clamp_min(1).reciprocal().to(row_values.dtype),
            torch.zeros_like(row_values),
        )
        values = column_mix * column.values + row_mix * row_values
        available = n_active > 0

        combined_weights = torch.cat(
            [
                column.routing.weights * column_mix.unsqueeze(-1),
                row_routing.weights * row_mix.unsqueeze(-1),
            ],
            dim=-1,
        )
        log_dtype = column.routing.log_weights.dtype
        finite_floor = -torch.finfo(log_dtype).max
        column_log_mix = column_mix.to(log_dtype).clamp_min(torch.finfo(log_dtype).tiny).log()
        row_log_mix = row_mix.to(log_dtype).clamp_min(torch.finfo(log_dtype).tiny).log()
        column_support_mask = column.routing.support_mask & column.support_available.unsqueeze(-1)
        row_support_mask = row_routing.support_mask & row_routing.support_available.unsqueeze(-1)
        combined_support_mask = torch.cat((column_support_mask, row_support_mask), dim=-1)
        combined_log_weights = torch.cat(
            (
                column.routing.log_weights + column_log_mix.unsqueeze(-1),
                row_routing.log_weights + row_log_mix.unsqueeze(-1),
            ),
            dim=-1,
        )
        combined_log_weights = torch.where(
            combined_support_mask,
            combined_log_weights,
            torch.full_like(combined_log_weights, finite_floor),
        )
        routing = RoutingOutput(
            weights=combined_weights,
            log_weights=combined_log_weights,
            support_mask=combined_support_mask,
            support_available=available,
            support_count=column.routing.support_count + row_routing.support_count,
        )
        return NumericReadoutOutput(values=values, support_available=available, routing=routing)


def categorical_from_routing(
    routing: RoutingOutput,
    support_values: Tensor,
    visible_mask: Tensor,
    domain_values: Tensor,
    domain_mask: Tensor,
) -> CategoricalReadoutOutput:
    """Aggregate same-column routing mass over a declared categorical domain."""

    if support_values.ndim != 3 or visible_mask.shape != support_values.shape:
        raise ValueError("support_values and visible_mask must be [B,N,M]")
    if visible_mask.dtype is not torch.bool:
        raise ValueError("visible_mask must be bool")
    batch, n_rows, n_features = support_values.shape
    same_column_shape = (batch, n_rows, n_features, n_rows)
    bilinear_shape = (batch, n_rows, n_features, n_rows + n_features)
    if routing.weights.shape not in {same_column_shape, bilinear_shape}:
        raise ValueError("routing weights must be [B,N,M,N] or bilinear [B,N,M,N+M]")
    domains = torch.as_tensor(
        domain_values, device=support_values.device, dtype=support_values.dtype
    )
    declared = torch.as_tensor(domain_mask, device=support_values.device, dtype=torch.bool)
    if domains.ndim != 2 or domains.shape[0] != n_features or domains.shape != declared.shape:
        raise ValueError("domain_values/domain_mask must be matching [M,C] tensors")
    categorical_features = declared.any(dim=-1)
    for feature in categorical_features.nonzero(as_tuple=False).flatten().tolist():
        feature_domain = domains[feature][declared[feature]]
        if torch.unique(feature_domain).numel() != feature_domain.numel():
            raise ValueError("categorical domains must not contain duplicates")
        visible_values = support_values[:, :, feature][visible_mask[:, :, feature]]
        if visible_values.numel() and not bool(
            (visible_values.unsqueeze(-1) == feature_domain).any(dim=-1).all()
        ):
            raise ValueError("visible categorical value is outside its declared domain")

    column_source_values = (
        support_values.permute(0, 2, 1).unsqueeze(1).expand(batch, n_rows, n_features, n_rows)
    )
    if routing.weights.shape == bilinear_shape:
        # TabU4Rec's item arm appends same-row, other-item supports after the
        # user arm's same-item, other-row supports.  Keep the same ordering as
        # ``BilinearNumericNW`` so typed categorical mass and public support
        # IDs describe the exact same interaction family.
        row_source_values = support_values.unsqueeze(2).expand(
            batch, n_rows, n_features, n_features
        )
        routed_source_values = torch.cat((column_source_values, row_source_values), dim=-1)
    else:
        routed_source_values = column_source_values
    source_values = routed_source_values.unsqueeze(-1)
    domain_grid = domains.view(1, 1, n_features, 1, -1)
    domain_grid_mask = declared.view(1, 1, n_features, 1, -1)
    membership = (source_values == domain_grid) & domain_grid_mask
    probabilities = (routing.weights.unsqueeze(-1) * membership.to(routing.weights.dtype)).sum(
        dim=-2
    )
    probabilities = probabilities * declared.view(1, 1, n_features, -1)
    available = routing.support_available & categorical_features.view(1, 1, n_features)
    probabilities = torch.where(
        available.unsqueeze(-1), probabilities, torch.zeros_like(probabilities)
    )
    log_dtype = routing.log_weights.dtype
    finite_floor = -torch.finfo(log_dtype).max
    class_support = routing.support_mask.unsqueeze(-1) & membership
    class_log_mass = torch.logsumexp(
        torch.where(
            class_support,
            routing.log_weights.unsqueeze(-1),
            torch.full_like(routing.log_weights.unsqueeze(-1), finite_floor),
        ),
        dim=-2,
    )
    class_available = class_support.any(dim=-2)
    declared_view = declared.view(1, 1, n_features, -1)
    log_probabilities = torch.where(
        available.unsqueeze(-1) & declared_view & class_available,
        class_log_mass,
        torch.full_like(class_log_mass, finite_floor),
    )
    selected = probabilities.argmax(dim=-1)
    expanded_domains = domains.view(1, 1, n_features, -1).expand(batch, n_rows, -1, -1)
    values = expanded_domains.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    values = torch.where(available, values, torch.zeros_like(values))
    feature_weights = routing.weights * categorical_features.view(1, 1, n_features, 1)
    feature_support_mask = routing.support_mask & categorical_features.view(1, 1, n_features, 1)
    feature_log_weights = torch.where(
        feature_support_mask,
        routing.log_weights,
        torch.full_like(routing.log_weights, finite_floor),
    )
    categorical_routing = RoutingOutput(
        weights=feature_weights,
        log_weights=feature_log_weights,
        support_mask=feature_support_mask,
        support_available=available,
        support_count=torch.where(
            categorical_features.view(1, 1, n_features),
            routing.support_count,
            torch.zeros_like(routing.support_count),
        ),
    )
    return CategoricalReadoutOutput(
        values=values,
        probabilities=probabilities,
        log_probabilities=log_probabilities,
        class_support_available=class_available & declared_view,
        domain_values=domains,
        domain_mask=declared,
        support_available=available,
        routing=categorical_routing,
    )


class SameColumnCategoricalNW(nn.Module):
    """Same-column categorical NW over an explicit, schema-declared domain."""

    def __init__(self, bandwidth: float = 1.0) -> None:
        super().__init__()
        self.router = SameColumnNumericNW(bandwidth)

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
        domain_values: Tensor,
        domain_mask: Tensor,
    ) -> CategoricalReadoutOutput:
        routing = self.router(coordinates, support_values, visible_mask).routing
        return categorical_from_routing(
            routing,
            support_values,
            visible_mask,
            domain_values,
            domain_mask,
        )


__all__ = [
    "BilinearNumericLocalLinear",
    "BilinearNumericNW",
    "CategoricalReadoutOutput",
    "GlobalUserItemNumericLocalLinear",
    "GlobalUserItemNumericNW",
    "NumericReadoutOutput",
    "RoutingOutput",
    "SameColumnCategoricalNW",
    "SameColumnNumericNW",
    "categorical_from_routing",
    "masked_rbf_weights",
]
