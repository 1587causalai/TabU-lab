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

from tabu_lab.primitives import NumericReadoutOutput, RoutingOutput, presence_gate

from .components import CellTokenizer, NumericScaleState
from .dynamics import CellUnitDynamics
from .readouts import PairUnitReadout
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
        if len(response_flags) != n_features or sum(response_flags) != 1:
            raise ValueError(
                "supervised.label_broadcast.v1 requires exactly one declared response column"
            )
        response_mask = torch.tensor(
            response_flags,
            dtype=torch.bool,
            device=cells.device,
        )
        query_columns = inputs.query_target_mask.any(dim=(0, 1)).to(device=cells.device)
        if not torch.equal(query_columns, response_mask):
            raise ValueError("query targets must match the single declared response column")

        visible = inputs.visible_mask.to(device=cells.device)
        query = inputs.query_target_mask.to(device=cells.device)
        # Raw roles + visibility/query-marker are the only source eligibility
        # signals.  Hidden-state nonzero status is deliberately not consulted.
        source_mask = (visible | query) & response_mask.view(1, 1, n_features)
        label_mix = (cells * source_mask.unsqueeze(-1).to(cells.dtype)).sum(dim=2)
        receiver_mask = (~response_mask).view(1, 1, n_features)
        receiver_mask = receiver_mask & (visible | inputs.target_mask.to(device=cells.device))
        receiver_mask = receiver_mask & ~inputs.natural_missing_mask.to(device=cells.device)
        receiver_gate = presence_gate(cells, self.tau).unsqueeze(-1)
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
        profile: TabUCellBaseProfile | str,
        label_broadcast: bool | None = None,
        label_broadcast_tau: float = _LABEL_BROADCAST_TAU,
        nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
        _component_tokenizer: nn.Module | None = None,
        _component_dynamics: nn.Module | None = None,
        _component_readout: nn.Module | None = None,
        _component_composition: Any | None = None,
        _component_registry: Any | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config)
        supplied_components = (
            _component_tokenizer,
            _component_dynamics,
            _component_readout,
            _component_composition,
            _component_registry,
        )
        if any(value is not None for value in supplied_components) and not all(
            value is not None for value in supplied_components
        ):
            raise ValueError("resolved component injection requires one complete composition")
        if _component_tokenizer is None:
            self.tokenizer = CellTokenizer(
                config,
                marker="mask",
                nominal_tokenizer=nominal_tokenizer,
                nominal_codebook_size=nominal_codebook_size,
                nominal_codebook_seed=nominal_codebook_seed,
            )
            self.dynamics = CellUnitDynamics(config)
            self.readout = PairUnitReadout(config, numeric_terminal=numeric_terminal)
            self.component_composition = None
            self.component_registry = None
        else:
            if not isinstance(_component_tokenizer, CellTokenizer):
                raise TypeError("resolved tokenizer violates the TabUBase interface")
            if not isinstance(_component_dynamics, CellUnitDynamics):
                raise TypeError("resolved dynamics violates the TabUBase interface")
            if not isinstance(_component_readout, PairUnitReadout):
                raise TypeError("resolved readout violates the TabUBase interface")
            self.tokenizer = _component_tokenizer
            self.dynamics = _component_dynamics
            self.readout = _component_readout
            self.component_composition = _component_composition
            self.component_registry = _component_registry
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
        self.profile = TabUCellBaseProfile(profile)
        expected_broadcast = self.profile.uses_label_broadcast
        if label_broadcast is not None and bool(label_broadcast) != expected_broadcast:
            raise ValueError("label_broadcast is derived from the explicit TabUCellBaseProfile")
        self.label_broadcast = expected_broadcast
        self.label_broadcast_tau = _validate_label_broadcast_tau(label_broadcast_tau)
        if not self.label_broadcast and self.label_broadcast_tau != _LABEL_BROADCAST_TAU:
            raise ValueError(
                "completion.artificial_mask.v1 requires the canonical label_broadcast_tau"
            )
        semantic_payload = {
            "reference_config": _reference_config_payload(config),
            "profile_id": self.profile.value,
            "tokenizer": self.tokenizer_metadata,
            "label_broadcast": self.label_broadcast,
            "label_broadcast_tau": self.label_broadcast_tau,
            "numeric_terminal": self.readout.numeric_terminal,
            "ll_ridge": self.readout.ll_ridge,
        }
        if self.component_composition is not None:
            semantic_payload["component_composition_hash"] = (
                self.component_composition.composition_hash
            )
        semantic_config_hash = hashlib.sha256(
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        expected_variant = ModelVariantRef(
            contract_id=self.model_id,
            contract_version=self.contract_version,
            profile_id=self.profile.value,
            # DenseReferenceModel resolves the canonical packaged ModelSpec
            # before this identity envelope is created.  Reuse that exact
            # hash instead of hashing the display string, so even an
            # unbound local build carries a real contract hash.
            model_spec_hash=self.model_spec_hash,
            source_identity="unbound-local-source",
            semantic_config_hash=semantic_config_hash,
        )
        # Source provenance is bound by a later trusted receipt boundary, not
        # by arbitrary model-construction input.  Local builds therefore carry
        # the explicit unbound sentinel and callers cannot forge an approved
        # source identity through this constructor.
        self.variant_ref = expected_variant

    def _component_manifest_identity(self) -> dict[str, Any]:
        composition = self.component_composition
        if composition is None:
            return {}
        payload = composition.as_dict()
        return {
            "component_manifest_hash": payload["manifest_hash"],
            "component_composition_hash": composition.composition_hash,
            "component_spec_hashes": payload["component_spec_hashes"],
            "experimental_component_axes": composition.experimental_axes,
        }

    def _validate_profile_input(self, inputs: Any) -> None:
        """Reject target-origin/profile mixtures before tokenization."""

        natural_targets = inputs.natural_missing_mask & inputs.target_mask
        if bool((natural_targets & ~inputs.unsupported_target_mask).any()):
            raise ValueError("natural-missing targets must use the unsupported origin")

        if self.profile is TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1:
            if bool(inputs.query_target_mask.any()):
                raise ValueError("completion.artificial_mask.v1 rejects query target origins")
            response_count = sum(
                getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None))
                == "response"
                for spec in inputs.feature_specs
            )
            if response_count:
                raise ValueError("completion.artificial_mask.v1 requires zero response columns")
            return

        if bool(inputs.artificial_target_mask.any()):
            raise ValueError("supervised.label_broadcast.v1 rejects artificial-mask target origins")
        if not bool(inputs.query_target_mask.any()):
            raise ValueError("supervised.label_broadcast.v1 requires query targets")

        n_features = inputs.values.shape[2]
        response_flags = tuple(
            getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None)) == "response"
            for spec in inputs.feature_specs
        )
        if len(response_flags) != n_features or sum(response_flags) != 1:
            raise ValueError(
                "supervised.label_broadcast.v1 requires exactly one declared response column"
            )
        response_mask = torch.tensor(
            response_flags,
            dtype=torch.bool,
            device=inputs.values.device,
        )
        query_columns = inputs.query_target_mask.any(dim=(0, 1))
        if not torch.equal(query_columns, response_mask):
            raise ValueError("query targets must match the single declared response column")

    def _encode_dense_cells(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, Tensor, Tensor, NumericScaleState]:
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
        self._validate_profile_input(resolved)
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
        if tokens.numeric_scale_state is None:
            raise RuntimeError("TabUBase tokenizer did not expose numeric scale state")
        dynamics_input = _label_broadcast(
            tokens.cells,
            resolved,
            enabled=self.label_broadcast,
            tau=self.label_broadcast_tau,
        )
        cells = self.dynamics(
            dynamics_input,
            column_source_mask=context_mask,
            row_source_mask=resolved.visible_mask,
        )
        return resolved, symbols, dynamics_input, cells, tokens.numeric_scale_state

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        emit_trace = bool(kwargs.get("emit_trace", True))
        (
            resolved,
            symbols,
            dynamics_input,
            cells,
            numeric_scale_state,
        ) = self._encode_dense_cells(inputs, **kwargs)
        coordinates, readout = self.readout(
            cells,
            numeric_scale_state.standardized_values,
            resolved.visible_mask,
        )
        readout = _apply_cell_null_contract(
            readout,
            coordinates,
            cells,
            null_mask=resolved.natural_missing_mask,
        )
        numeric_features = torch.tensor(
            tuple(kind == "numeric" for kind in symbols.feature_kinds),
            dtype=torch.bool,
            device=readout.values.device,
        ).view(1, 1, -1)
        numeric_raw_prediction = torch.where(
            readout.support_available & numeric_features & ~resolved.unsupported_target_mask,
            readout.values * numeric_scale_state.scale + numeric_scale_state.mean,
            torch.zeros_like(readout.values),
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
                    numeric_prediction_scale="context_standardized",
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
            extra_auxiliaries={
                "numeric_raw_prediction": numeric_raw_prediction,
                "numeric_context_mean": numeric_scale_state.mean,
                "numeric_context_std": numeric_scale_state.std,
                "numeric_context_scale": numeric_scale_state.scale,
                "numeric_context_count": numeric_scale_state.context_count,
            },
            metadata={
                "dynamics_plan": self._dynamics_plan_name(self.dynamics),
                "unit": "cell",
                "family_id": "tabu.table_cell_as_unit",
                "numeric_terminal": self.readout.numeric_terminal,
                "numeric_prediction_scale": "context_standardized",
                "profile_id": self.profile.value,
                "contract_version": self.variant_ref.contract_version,
                "variant_ref": self.variant_ref.as_dict(),
                "variant_hash": self.variant_ref.semantic_hash,
                **self._component_manifest_identity(),
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
            **self._component_manifest_identity(),
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
        unexpected = sorted(set(identity) - set(expected))
        if unexpected:
            raise ValueError(f"checkpoint identity has unexpected fields: {unexpected}")
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"checkpoint identity mismatch at {key}: expected {value!r}")
