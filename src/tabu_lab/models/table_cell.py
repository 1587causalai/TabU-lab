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

from .components import CellTokenizer
from .dynamics import PairUnitDynamics
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
        self.dynamics = PairUnitDynamics(config)
        self.readout = PairUnitReadout(config, numeric_terminal=numeric_terminal)
        semantic_payload = {
            "reference_config": _reference_config_payload(config),
            "profile_id": self.profile.value,
            "tokenizer": self.tokenizer_metadata,
            "label_broadcast": self.label_broadcast,
            "label_broadcast_tau": self.label_broadcast_tau,
            "numeric_terminal": self.readout.numeric_terminal,
            "ll_ridge": self.readout.ll_ridge,
        }
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
        if variant_ref is not None:
            required = {
                "contract_id": expected_variant.contract_id,
                "contract_version": expected_variant.contract_version,
                "profile_id": expected_variant.profile_id,
                "model_spec_hash": expected_variant.model_spec_hash,
                "semantic_config_hash": expected_variant.semantic_config_hash,
            }
            for key, expected in required.items():
                if getattr(variant_ref, key) != expected:
                    raise ValueError(
                        f"variant_ref mismatch at {key}: expected {expected!r}"
                    )
        self.variant_ref = variant_ref or expected_variant

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
        unexpected = sorted(set(identity) - set(expected))
        if unexpected:
            raise ValueError(f"checkpoint identity has unexpected fields: {unexpected}")
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"checkpoint identity mismatch at {key}: expected {value!r}")
