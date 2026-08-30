"""Axis-B table-cell-as-Unit reference models.

This module deliberately gives the new family its own public model identity.
The low-level PairUnit dynamics/readout are reused as implementation
primitives, but the model contract and trace remain ``tabu.cell.base`` rather
than aliasing the legacy ``tabu.unit_pair`` contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from tabu_lab.primitives import NumericReadoutOutput, RoutingOutput

from .component_registry import TABU_CELL_BASE_COMPONENTS
from .components import CellTokenizer
from .dynamics import CellFamilyDynamics, PairUnitDynamics
from .readouts import CellMatchingReadout, CellSpecialReadout, PairUnitReadout
from .reference import DenseReferenceModel, _shape_event
from .types import ModelVariantRef, ReferenceConfig, TabUCellBaseProfile

_CELL_TOKENIZER_METADATA = {
    "tokenizer_version": "cell-tokenizer.v1",
    "feature_identity": "forbidden",
    "continuous_tokenizer": "context_only_standardization_then_shared_learnable_fourier",
    "nominal_tokenizer": "episode_random_sphere",
    "scale_epsilon": CellTokenizer._scale_epsilon,
}
_LABEL_BROADCAST_TAU = 1.0e-6


def _reference_config_payload(config: ReferenceConfig) -> dict[str, Any]:
    return {key: getattr(value, "value", value) for key, value in config.__dict__.items()}


def _validate_label_broadcast_tau(value: float) -> float:
    tau = float(value)
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError("label_broadcast_tau must be a finite positive float")
    return tau


class LabelColumnBroadcast(nn.Module):
    """Step-3 supervised route with raw semantic source eligibility."""

    def __init__(self, *, tau: float = _LABEL_BROADCAST_TAU) -> None:
        super().__init__()
        self.tau = _validate_label_broadcast_tau(tau)

    def forward(self, cells: Tensor, inputs: Any) -> Tensor:
        if cells.ndim != 4:
            raise ValueError("label broadcast requires [B,N,M,D] cell tokens")
        n_features = cells.shape[2]
        response_flags = [
            getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None)) == "response"
            for spec in getattr(inputs, "feature_specs", ())
        ]
        response_mask = (
            torch.tensor(
                response_flags,
                dtype=torch.bool,
                device=cells.device,
            )
            if response_flags
            else torch.zeros(n_features, dtype=torch.bool, device=cells.device)
        )
        if response_mask.numel() not in {0, n_features}:
            raise ValueError("response roles must match the feature axis")
        if response_flags and int(response_mask.sum().item()) != 1:
            raise ValueError("supervised.label_broadcast.v1 requires exactly one response column")
        query_columns = inputs.query_target_mask.any(dim=(0, 1)).to(device=cells.device)
        if response_flags and bool((query_columns & ~response_mask).any()):
            raise ValueError("query target columns must be declared response features")
        if not bool(response_mask.any()):
            # A dense implementation probe may omit FeatureSpec roles; query
            # columns still identify a label source without consulting truth.
            response_mask = query_columns
        if not bool(response_mask.any()):
            return cells

        visible = inputs.visible_mask.to(device=cells.device)
        query = inputs.query_target_mask.to(device=cells.device)
        # Raw roles + visibility/query-marker are the only source eligibility
        # signals.  Hidden-state nonzero status is deliberately not consulted.
        source_mask = (visible | query) & response_mask.view(1, 1, n_features)
        label_mix = (cells * source_mask.unsqueeze(-1).to(cells.dtype)).sum(dim=2)
        receiver_mask = (~response_mask).view(1, 1, n_features)
        receiver_mask = receiver_mask & (visible | inputs.target_mask.to(device=cells.device))
        receiver_mask = receiver_mask & ~inputs.natural_missing_mask.to(device=cells.device)
        norm_sq = cells.square().sum(dim=-1, keepdim=True)
        receiver_gate = norm_sq / (self.tau + norm_sq)
        return torch.where(
            receiver_mask.unsqueeze(-1),
            cells + receiver_gate * label_mix.unsqueeze(2),
            cells,
        )


def _label_broadcast(
    cells: Tensor,
    inputs: Any,
    *,
    enabled: bool,
    tau: float = _LABEL_BROADCAST_TAU,
) -> Tensor:
    if not enabled:
        return cells
    return LabelColumnBroadcast(tau=tau)(cells, inputs)


def _apply_cell_null_contract(
    readout: NumericReadoutOutput,
    coordinates: Tensor,
    cell_states: Tensor,
    *,
    null_mask: Tensor | None = None,
) -> NumericReadoutOutput:
    """Make ``c_ra = 0`` an explicit typed no-support outcome."""

    if coordinates.ndim != 4 or cell_states.ndim != 4:
        raise ValueError("cell null gating requires [B,N,M,*] tensors")
    if coordinates.shape[:3] != cell_states.shape[:3]:
        raise ValueError("cell null gating axes must match")
    present = cell_states.abs().sum(dim=-1) > 0
    if null_mask is not None:
        if null_mask.shape != cell_states.shape[:3] or null_mask.dtype is not torch.bool:
            raise ValueError("cell null mask must be bool and match cell axes")
        # MAB is retained as a non-O ablation and can produce a residual for
        # a zero receiver.  The source contract still treats natural missing
        # as exact null, so apply that semantic boundary explicitly at
        # readout (nulls remain excluded from every source mask upstream).
        present = present & ~null_mask
    routing = readout.routing
    query_mask = present.unsqueeze(-1)
    support_mask = routing.support_mask & query_mask
    finite_floor = -torch.finfo(routing.log_weights.dtype).max
    log_weights = torch.where(
        support_mask,
        routing.log_weights,
        torch.full_like(routing.log_weights, finite_floor),
    )
    weights = torch.where(query_mask, routing.weights, torch.zeros_like(routing.weights))
    return NumericReadoutOutput(
        values=torch.where(present, readout.values, torch.zeros_like(readout.values)),
        support_available=readout.support_available & present,
        routing=RoutingOutput(
            weights=weights,
            log_weights=log_weights,
            support_mask=support_mask,
            support_available=routing.support_available & present,
            support_count=support_mask.sum(dim=-1),
        ),
    )


class TabUCellBaseModel(DenseReferenceModel):
    """TabUBase: cell Units with the global learned projection ``z = Wc``.

    The first vertical slice defaults to artificial-mask completion only.
    The optional ``label_broadcast`` route enables the source-contract
    supervised extension without changing the cell Unit or global ``W``
    readout; row/column special carriers belong to the sibling profiles.
    """

    model_id = "tabu.cell.base"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        numeric_terminal: str = "local_linear",
        profile: TabUCellBaseProfile | str | None = None,
        label_broadcast: bool | None = None,
        label_broadcast_tau: float = _LABEL_BROADCAST_TAU,
        nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
        variant_ref: ModelVariantRef | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config, marker="mask", feature_identity=False)
        self.tokenizer = CellTokenizer(
            config,
            marker="mask",
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
        )
        self.nominal_tokenizer = self.tokenizer.nominal_tokenizer
        self.nominal_codebook_size = self.tokenizer.nominal_codebook_size
        self.nominal_codebook_seed = self.tokenizer.nominal_codebook_seed
        tokenizer_version = (
            "cell-tokenizer.v2"
            if self.nominal_tokenizer == CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2
            else "cell-tokenizer.v1"
        )
        self.tokenizer_metadata = dict(_CELL_TOKENIZER_METADATA)
        if tokenizer_version == "cell-tokenizer.v2":
            self.tokenizer_metadata.update(
                {
                    "tokenizer_version": tokenizer_version,
                    "nominal_tokenizer": self.nominal_tokenizer,
                    "nominal_codebook_size": self.nominal_codebook_size,
                    "nominal_codebook_seed": self.nominal_codebook_seed,
                    "nominal_codebook_hash": self.tokenizer.nominal_codebook_hash,
                    "nominal_codebook_scope": "source_codebook_id_and_domain_label",
                }
            )
        if profile is None:
            profile = (
                TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
                if bool(label_broadcast)
                else TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1
            )
        self.profile = TabUCellBaseProfile(profile)
        expected_broadcast = self.profile.uses_label_broadcast
        if label_broadcast is not None and bool(label_broadcast) != expected_broadcast:
            raise ValueError("label_broadcast is derived from the explicit TabUCellBaseProfile")
        self.label_broadcast = expected_broadcast
        self.label_broadcast_tau = _validate_label_broadcast_tau(label_broadcast_tau)
        if tokenizer_version == "cell-tokenizer.v1":
            semantic_config_hash = config.semantic_hash
        else:
            semantic_payload = {
                "reference_config": _reference_config_payload(config),
                "tokenizer": self.tokenizer_metadata,
            }
            semantic_config_hash = hashlib.sha256(
                json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        self.variant_ref = variant_ref or ModelVariantRef(
            contract_id=self.model_id,
            contract_version="0.2.0",
            profile_id=self.profile.value,
            # DenseReferenceModel resolves the canonical packaged ModelSpec
            # before this identity envelope is created.  Reuse that exact
            # hash instead of hashing the display string, so even an
            # unbound local build carries a real contract hash.
            model_spec_hash=self.model_spec_hash,
            source_identity="unbound-local-source",
            semantic_config_hash=semantic_config_hash,
        )
        self.dynamics = PairUnitDynamics(config)
        self.readout = PairUnitReadout(config, numeric_terminal=numeric_terminal)

    def _encode_dense_cells(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, Tensor, Tensor]:
        """Resolve one dense episode through Step 3 without materializing readout routing.

        Full-context evaluation can contain tens of thousands of rows.  The
        standard same-column terminal materializes an ``N x N`` support ledger,
        even when only query-response predictions are required.  This helper
        exposes the exact shared symbolizer/tokenizer/dynamics path so an
        evaluator can apply a query-target-only terminal without changing the
        model state or the episode evidence.
        """

        resolved = self._resolve_inputs(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        context_mask = resolved.visible_mask
        if self.profile.uses_label_broadcast:
            query_rows = resolved.query_target_mask.any(dim=2, keepdim=True)
            context_mask = resolved.visible_mask & ~query_rows
        resolved = replace(
            resolved,
            metadata={
                **dict(resolved.metadata),
                "context_mask": context_mask,
                "profile_id": self.profile.value,
            },
        )
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        dynamics_input = _label_broadcast(
            tokens.cells,
            resolved,
            enabled=self.label_broadcast,
            tau=self.label_broadcast_tau,
        )
        cells = self.dynamics(
            dynamics_input,
            column_source_mask=resolved.visible_mask,
            row_source_mask=resolved.visible_mask,
        )
        return resolved, symbols, dynamics_input, cells

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        emit_trace = bool(kwargs.get("emit_trace", True))
        resolved, symbols, dynamics_input, cells = self._encode_dense_cells(inputs, **kwargs)
        coordinates, readout = self.readout(cells, resolved.values, resolved.visible_mask)
        readout = _apply_cell_null_contract(
            readout,
            coordinates,
            cells,
            null_mask=resolved.natural_missing_mask,
        )
        events = (
            (
                _shape_event(
                    "symbolizer",
                    symbols.values,
                    input_tensor=resolved.values,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    unit="cell",
                ),
                _shape_event(
                    "tokenizer",
                    dynamics_input,
                    input_tensor=symbols.values,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    **self.tokenizer_metadata,
                    label_broadcast=self.label_broadcast,
                    label_broadcast_tau=self.label_broadcast_tau,
                    reference_config=_reference_config_payload(self.config),
                    terminal=self.readout.numeric_terminal,
                    ll_ridge=getattr(self.readout, "ll_ridge", None),
                    bandwidth=self.config.routing_bandwidth,
                ),
                _shape_event(
                    "dynamics_plan",
                    cells,
                    input_tensor=dynamics_input,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    operation_trace=self.dynamics.plan.stages,
                    plan=self._dynamics_plan_name(self.dynamics),
                ),
                _shape_event(
                    "readout",
                    readout.values,
                    input_tensor=coordinates,
                    source_mask=resolved.visible_mask,
                    operation_trace=(
                        "global_projection",
                        f"numeric_{self.readout.numeric_terminal_trace}",
                    ),
                    terminal=f"numeric_{self.readout.numeric_terminal_trace}",
                    geometry="cell_projection",
                ),
                _shape_event(
                    "prediction_boundary",
                    resolved.target_mask,
                    input_tensor=readout.values,
                    source_mask=resolved.visible_mask,
                    operation_trace=("model_forward_complete",),
                    supervision_boundary="sidecar_only",
                    truth_not_available=True,
                    model_forward_complete=True,
                ),
            )
            if emit_trace
            else ()
        )
        return self._bundle(
            inputs=resolved,
            values=readout.values,
            support_available=readout.support_available,
            coordinates=coordinates,
            routing_weights=readout.routing.weights,
            routing_log_weights=readout.routing.log_weights,
            routing_support_mask=readout.routing.support_mask,
            events=events,
            metadata={
                "dynamics_plan": self._dynamics_plan_name(self.dynamics),
                "unit": "cell",
                "family_id": "tabu.table_cell_as_unit",
                "numeric_terminal": self.readout.numeric_terminal,
                "profile_id": self.profile.value,
                "contract_version": self.variant_ref.contract_version,
                "variant_ref": self.variant_ref.as_dict(),
                "variant_hash": self.variant_ref.semantic_hash,
                **self.tokenizer_metadata,
                "label_broadcast": self.label_broadcast,
                "label_broadcast_tau": self.label_broadcast_tau,
                "reference_config": _reference_config_payload(self.config),
                "terminal": self.readout.numeric_terminal,
                "ll_ridge": getattr(self.readout, "ll_ridge", None),
                "bandwidth": self.config.routing_bandwidth,
                "query_marker": "unified" if bool(resolved.query_target_mask.any()) else "absent",
            },
            emit_trace=emit_trace,
        )

    def checkpoint_identity(self) -> dict[str, Any]:
        """Return the identity envelope required before loading a checkpoint."""
        identity = {
            "model_id": self.model_id,
            "contract_version": self.variant_ref.contract_version,
            "profile_id": self.profile.value,
            "tokenizer_version": self.tokenizer_metadata["tokenizer_version"],
            "label_broadcast": self.label_broadcast,
            "label_broadcast_tau": self.label_broadcast_tau,
            "variant_hash": self.variant_ref.semantic_hash,
            "reference_config": _reference_config_payload(self.config),
            "terminal": self.readout.numeric_terminal,
            "ll_ridge": getattr(self.readout, "ll_ridge", None),
            "bandwidth": self.config.routing_bandwidth,
            "variant_ref": self.variant_ref.as_dict(),
        }
        if self.tokenizer_metadata["tokenizer_version"] == "cell-tokenizer.v2":
            identity.update(
                {
                    "nominal_tokenizer": self.nominal_tokenizer,
                    "nominal_codebook_size": self.nominal_codebook_size,
                    "nominal_codebook_seed": self.nominal_codebook_seed,
                    "nominal_codebook_hash": self.tokenizer.nominal_codebook_hash,
                }
            )
        return identity

    def validate_checkpoint_identity(self, identity: Mapping[str, Any]) -> None:
        expected = self.checkpoint_identity()
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"checkpoint identity mismatch at {key}: expected {value!r}")


class _TabUCellSpecialModel(DenseReferenceModel):
    """Shared implementation for the row/column/direct-sum cell variants."""

    axis: str
    model_id: str

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        numeric_terminal: str = "local_linear",
        label_broadcast: bool = False,
        label_broadcast_tau: float = _LABEL_BROADCAST_TAU,
    ):
        config = config or ReferenceConfig()
        super().__init__(config, marker="mask", feature_identity=False)
        self.tokenizer = CellTokenizer(config, marker="mask")
        self.label_broadcast = bool(label_broadcast)
        self.label_broadcast_tau = _validate_label_broadcast_tau(label_broadcast_tau)
        self.dynamics = CellFamilyDynamics(config)
        self.readout = CellSpecialReadout(
            config,
            mode=self.axis,
            numeric_terminal=numeric_terminal,
        )
        self.row_special_seeds = (
            nn.Parameter(torch.empty(config.matched_slots, config.d_model))
            if self.axis in {"row", "row_column"}
            else None
        )
        self.column_special_seeds = (
            nn.Parameter(torch.empty(config.matched_slots, config.d_model))
            if self.axis in {"column", "row_column"}
            else None
        )
        for seed in (self.row_special_seeds, self.column_special_seeds):
            if seed is not None:
                nn.init.normal_(seed, std=0.02)

    def _initial_carrier(
        self, cells: Tensor, visible_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, n_rows, n_features, d_model = cells.shape
        k = self.config.matched_slots
        extra_rows = k if self.axis in {"column", "row_column"} else 0
        extra_columns = k if self.axis in {"row", "row_column"} else 0
        carrier = cells.new_zeros(batch, n_rows + extra_rows, n_features + extra_columns, d_model)
        carrier[:, :n_rows, :n_features] = cells
        if self.row_special_seeds is not None:
            carrier[:, :n_rows, n_features:] = self.row_special_seeds.to(cells.dtype).view(
                1, 1, k, d_model
            )
        if self.column_special_seeds is not None:
            carrier[:, n_rows:, :n_features] = self.column_special_seeds.to(cells.dtype).view(
                1, k, 1, d_model
            )

        shape = carrier.shape[:3]
        column_sources = torch.zeros(shape, dtype=torch.bool, device=cells.device)
        row_sources = torch.zeros_like(column_sources)
        column_receivers = torch.zeros_like(column_sources)
        row_receivers = torch.zeros_like(column_sources)
        column_sources[:, :n_rows, :n_features] = visible_mask
        row_sources[:, :n_rows, :n_features] = visible_mask
        column_receivers[:, :n_rows, :n_features] = True
        row_receivers[:, :n_rows, :n_features] = True
        if self.column_special_seeds is not None:
            # Column specials read visible cells in their column during the
            # induced-column stage, but never source the row-axis K/V pool.
            column_receivers[:, n_rows:, :n_features] = True
        if self.row_special_seeds is not None:
            # Row specials are receiver-only on the row axis.  They deliberately
            # do not write ordinary feature inducing slots.
            row_receivers[:, :n_rows, n_features:] = True
        return carrier, column_sources, row_sources, column_receivers, row_receivers

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        resolved = self._resolve_inputs(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        token_cells = _label_broadcast(
            tokens.cells,
            resolved,
            enabled=self.label_broadcast,
            tau=self.label_broadcast_tau,
        )
        (
            carrier,
            column_sources,
            row_sources,
            column_receivers,
            row_receivers,
        ) = self._initial_carrier(token_cells, resolved.visible_mask)
        carrier_input = carrier
        carrier = self.dynamics(
            carrier,
            column_source_mask=column_sources,
            row_source_mask=row_sources,
            column_receiver_mask=column_receivers,
            row_receiver_mask=row_receivers,
        )
        n_rows, n_features = resolved.values.shape[1:]
        coordinates, readout = self.readout(
            carrier,
            resolved.values,
            resolved.visible_mask,
            n_rows=n_rows,
            n_features=n_features,
        )
        cell_states = carrier[:, :n_rows, :n_features]
        readout = _apply_cell_null_contract(
            readout,
            coordinates,
            cell_states,
            null_mask=resolved.natural_missing_mask,
        )
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                unit="cell",
            ),
            _shape_event(
                "tokenizer",
                token_cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                **_CELL_TOKENIZER_METADATA,
                label_broadcast=self.label_broadcast,
                label_broadcast_tau=self.label_broadcast_tau,
            ),
            _shape_event(
                "dynamics_plan",
                carrier,
                input_tensor=carrier_input,
                source_mask=column_sources | row_sources,
                null_mask=resolved.natural_missing_mask,
                operation_trace=self.dynamics.plan.stages,
                plan=self._dynamics_plan_name(self.dynamics),
                carrier_role=self.axis,
                column_receiver_count=int(column_receivers.sum().item()),
                row_receiver_count=int(row_receivers.sum().item()),
            ),
            _shape_event(
                "readout",
                readout.values,
                input_tensor=coordinates,
                source_mask=resolved.visible_mask,
                operation_trace=(
                    f"{self.axis}_special_projection",
                    f"numeric_{self.readout.numeric_terminal_trace}",
                ),
                terminal=f"numeric_{self.readout.numeric_terminal_trace}",
                geometry=f"cell_{self.axis}_projection",
            ),
            _shape_event(
                "prediction_boundary",
                resolved.target_mask,
                input_tensor=readout.values,
                source_mask=resolved.visible_mask,
                operation_trace=("model_forward_complete",),
                supervision_boundary="sidecar_only",
                truth_not_available=True,
                model_forward_complete=True,
            ),
        )
        return self._bundle(
            inputs=resolved,
            values=readout.values,
            support_available=readout.support_available,
            coordinates=coordinates,
            routing_weights=readout.routing.weights,
            routing_log_weights=readout.routing.log_weights,
            routing_support_mask=readout.routing.support_mask,
            events=events,
            metadata={
                "dynamics_plan": self._dynamics_plan_name(self.dynamics),
                "unit": "cell",
                "family_id": "tabu.table_cell_as_unit",
                "numeric_terminal": self.readout.numeric_terminal,
                "profile_id": "completion.artificial_mask",
                "carrier_role": self.axis,
                **_CELL_TOKENIZER_METADATA,
                "label_broadcast": self.label_broadcast,
                "label_broadcast_tau": self.label_broadcast_tau,
                "carrier_shape": tuple(carrier.shape),
                "query_marker": "unified" if bool(resolved.query_target_mask.any()) else "absent",
            },
        )


class TabUCellRowModel(_TabUCellSpecialModel):
    """TabUR: row-special direct projection for cell Units."""

    model_id = "tabu.cell.row"
    axis = "row"


class TabUCellColumnModel(_TabUCellSpecialModel):
    """TabUC: column-special direct projection for cell Units."""

    model_id = "tabu.cell.column"
    axis = "column"


class TabUCellRowColumnModel(_TabUCellSpecialModel):
    """TabURC: concatenated row and column direct projections."""

    model_id = "tabu.cell.row_column"
    axis = "row_column"


class TabUCellRecModel(DenseReferenceModel):
    """Axis-B recommendation family with explicit ``m``, ``w``, and ``rc`` profiles.

    The registry keeps this contract design-open until profile lineage is
    reviewed.  The direct model builder is nevertheless useful for component
    conformance: each profile has a stable identity in the forward metadata.
    """

    model_id = "tabu.cell.rec"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        profile: str,
        numeric_terminal: str | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        profile = profile.lower()
        if profile not in {"m", "w", "rc"}:
            raise ValueError("tabu.cell.rec profile must be one of m, w, rc")
        if profile == "m":
            if numeric_terminal not in {None, "parameterized_matching"}:
                raise ValueError("the m profile has a fixed parameterized-matching terminal")
            resolved_terminal = "parameterized_matching"
        else:
            resolved_terminal = numeric_terminal or "local_linear"
        super().__init__(config, marker="mask", feature_identity=False)
        self.tokenizer = CellTokenizer(config, marker="mask")
        self.profile = profile
        self.numeric_terminal = resolved_terminal
        self.dynamics = PairUnitDynamics(config) if profile == "w" else CellFamilyDynamics(config)
        self.readout = (
            PairUnitReadout(config, numeric_terminal=resolved_terminal)
            if profile == "w"
            else (
                CellMatchingReadout()
                if profile == "m"
                else CellSpecialReadout(
                    config,
                    mode="row_column",
                    numeric_terminal=resolved_terminal,
                )
            )
        )
        self.row_special_seeds = (
            nn.Parameter(torch.empty(config.matched_slots, config.d_model))
            if profile in {"m", "rc"}
            else None
        )
        self.column_special_seeds = (
            nn.Parameter(torch.empty(config.matched_slots, config.d_model))
            if profile in {"m", "rc"}
            else None
        )
        for seed in (self.row_special_seeds, self.column_special_seeds):
            if seed is not None:
                nn.init.normal_(seed, std=0.02)

    def _initial_special_carrier(
        self, cells: Tensor, visible_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, n_rows, n_features, d_model = cells.shape
        k = self.config.matched_slots
        carrier = cells.new_zeros(batch, n_rows + k, n_features + k, d_model)
        carrier[:, :n_rows, :n_features] = cells
        carrier[:, :n_rows, n_features:] = self.row_special_seeds.to(cells.dtype).view(
            1, 1, k, d_model
        )
        carrier[:, n_rows:, :n_features] = self.column_special_seeds.to(cells.dtype).view(
            1, k, 1, d_model
        )
        shape = carrier.shape[:3]
        column_sources = torch.zeros(shape, dtype=torch.bool, device=cells.device)
        row_sources = torch.zeros_like(column_sources)
        column_receivers = torch.zeros_like(column_sources)
        row_receivers = torch.zeros_like(column_sources)
        column_sources[:, :n_rows, :n_features] = visible_mask
        row_sources[:, :n_rows, :n_features] = visible_mask
        column_receivers[:, :n_rows, :n_features] = True
        column_receivers[:, n_rows:, :n_features] = True
        row_receivers[:, :n_rows, :n_features] = True
        row_receivers[:, :n_rows, n_features:] = True
        return carrier, column_sources, row_sources, column_receivers, row_receivers

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        resolved = self._resolve_inputs(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        if bool(resolved.query_target_mask.any()):
            raise ValueError("tabu.cell.rec accepts artificial-mask targets, not QUERY origins")
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        if self.profile == "w":
            carrier_input = tokens.cells
            carrier = self.dynamics(
                tokens.cells,
                column_source_mask=resolved.visible_mask,
                row_source_mask=resolved.visible_mask,
            )
            coordinates, readout = self.readout(
                carrier,
                resolved.values,
                resolved.visible_mask,
            )
            source_mask = resolved.visible_mask
            carrier_shape = tuple(carrier.shape)
        else:
            (
                carrier,
                column_sources,
                row_sources,
                column_receivers,
                row_receivers,
            ) = self._initial_special_carrier(tokens.cells, resolved.visible_mask)
            carrier_input = carrier
            carrier = self.dynamics(
                carrier,
                column_source_mask=column_sources,
                row_source_mask=row_sources,
                column_receiver_mask=column_receivers,
                row_receiver_mask=row_receivers,
            )
            n_rows, n_features = resolved.values.shape[1:]
            coordinates, readout = self.readout(
                carrier,
                resolved.values,
                resolved.visible_mask,
                n_rows=n_rows,
                n_features=n_features,
            )
            readout = _apply_cell_null_contract(
                readout,
                coordinates,
                carrier[:, :n_rows, :n_features],
                null_mask=resolved.natural_missing_mask,
            )
            source_mask = column_sources | row_sources
            carrier_shape = tuple(carrier.shape)
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                unit="cell",
            ),
            _shape_event(
                "tokenizer",
                tokens.cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                **_CELL_TOKENIZER_METADATA,
                label_broadcast=False,
            ),
            _shape_event(
                "dynamics_plan",
                carrier,
                input_tensor=carrier_input,
                source_mask=source_mask,
                null_mask=resolved.natural_missing_mask,
                operation_trace=self.dynamics.plan.stages,
                plan=self._dynamics_plan_name(self.dynamics),
                profile=self.profile,
            ),
            _shape_event(
                "readout",
                readout.values,
                input_tensor=coordinates,
                source_mask=resolved.visible_mask,
                operation_trace=(
                    "matched_special_inner_products"
                    if self.profile == "m"
                    else (
                        "global_projection"
                        if self.profile == "w"
                        else "row_column_special_projection"
                    ),
                    f"numeric_{getattr(self.readout, 'numeric_terminal_trace', 'matching')}",
                ),
                terminal=f"numeric_{getattr(self.readout, 'numeric_terminal_trace', 'matching')}",
                geometry=f"cell_rec_{self.profile}",
            ),
            _shape_event(
                "prediction_boundary",
                resolved.target_mask,
                input_tensor=readout.values,
                source_mask=resolved.visible_mask,
                operation_trace=("model_forward_complete",),
                supervision_boundary="sidecar_only",
                truth_not_available=True,
                model_forward_complete=True,
            ),
        )
        return self._bundle(
            inputs=resolved,
            values=readout.values,
            support_available=readout.support_available,
            coordinates=coordinates,
            routing_weights=readout.routing.weights,
            routing_log_weights=readout.routing.log_weights,
            routing_support_mask=readout.routing.support_mask,
            events=events,
            metadata={
                "dynamics_plan": self._dynamics_plan_name(self.dynamics),
                "unit": "cell",
                "family_id": "tabu.table_cell_as_unit",
                "profile_id": f"recommendation.{self.profile}",
                "numeric_terminal": self.numeric_terminal,
                "carrier_role": self.profile,
                **_CELL_TOKENIZER_METADATA,
                "carrier_shape": carrier_shape,
            },
        )


__all__ = [
    "LabelColumnBroadcast",
    "TabUCellBaseModel",
    "TabUCellColumnModel",
    "TabUCellRecModel",
    "TabUCellRowColumnModel",
    "TabUCellRowModel",
]

# Public factory entries are deliberately named/versioned so a substitution
# changes the composition identity and cannot silently shadow the Base line.
TABU_CELL_BASE_COMPONENTS.register("tokenizer", "cell-tokenizer.v1", CellTokenizer)
TABU_CELL_BASE_COMPONENTS.register(
    "tokenizer",
    "source-scoped-frozen-codebook.v2",
    lambda config, **kwargs: CellTokenizer(
        config,
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
        **kwargs,
    ),
)
TABU_CELL_BASE_COMPONENTS.register("broadcast", "label-broadcast.v1", LabelColumnBroadcast)
TABU_CELL_BASE_COMPONENTS.register("dynamics", "pair-unit-omab.v1", PairUnitDynamics)
TABU_CELL_BASE_COMPONENTS.register("readout", "global-Wc.v1", PairUnitReadout)
TABU_CELL_BASE_COMPONENTS.register("objective", "typed-sidecar.v1", lambda **_: None)
