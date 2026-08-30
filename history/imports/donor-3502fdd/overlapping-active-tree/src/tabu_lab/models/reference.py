"""Dense, contract-first reference implementations for the TabU model family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import (
    NumericReadoutOutput,
    RoutingOutput,
    categorical_from_routing,
    o_inject,
)

from .components import Symbolizer, Tokenizer, TokenTable
from .dynamics import (
    AugmentedDynamics,
    GraphDynamics,
    PairUnitDynamics,
    PredictorUnitAddressDynamics,
    RecommendationDynamics,
    RowUnitDynamics,
    SupervisedDynamics,
)
from .readouts import (
    CellTokenGlobalSupportReadout,
    MatchedScoreReadout,
    MatchedUFReadout,
    PairUnitReadout,
    PredictorOnlyLabelReadout,
    PredictorUnitLinkedLabelReadout,
    RowUnitReadout,
)
from .types import (
    DenseModelInput,
    DenseTraceEvent,
    ReferenceConfig,
    feature_layout,
    hash_dense_input,
)

if TYPE_CHECKING:
    from tabu_lab.contracts import EvidenceEpisode, PredictionBundle


def _tensor_hash(tensor: Tensor) -> str:
    detached = tensor.detach().cpu().contiguous()
    if detached.dtype is torch.bfloat16:
        detached = detached.float()
    digest = hashlib.sha256()
    digest.update(str(tuple(detached.shape)).encode())
    digest.update(str(detached.dtype).encode())
    digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _shape_event(
    stage: str,
    tensor: Tensor,
    *,
    input_tensor: Tensor | None = None,
    source_mask: Tensor | None = None,
    null_mask: Tensor | None = None,
    operation_trace: Sequence[str] | None = None,
    **metadata: Any,
) -> DenseTraceEvent:
    input_value = tensor if input_tensor is None else input_tensor
    if source_mask is None:
        source_mask = torch.zeros(0, dtype=torch.bool, device=tensor.device)
    if null_mask is None or not bool(null_mask.any()):
        null_norm = 0.0
    else:
        resolved_null = null_mask.to(device=tensor.device, dtype=torch.bool)
        while resolved_null.ndim < tensor.ndim:
            resolved_null = resolved_null.unsqueeze(-1)
        resolved_null = resolved_null.expand_as(tensor)
        null_norm = float(tensor.detach()[resolved_null].float().norm().item())
    event_metadata = {
        "shape": tuple(tensor.shape),
        "source_mask_hash": _tensor_hash(source_mask),
        "source_count": int(source_mask.to(torch.bool).sum().item()),
        "null_norm": null_norm,
        "operation_trace": tuple(operation_trace or (stage,)),
        **metadata,
    }
    return DenseTraceEvent(
        stage=stage,
        shape=tuple(tensor.shape),
        input_hash=_tensor_hash(input_value),
        output_hash=_tensor_hash(tensor),
        metadata=event_metadata,
    )


def _routing_diagnostics(
    *,
    coordinates: Tensor,
    weights: Tensor,
    log_weights: Tensor,
    support_mask: Tensor,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Return truth-free router diagnostics for receipts and fit debugging.

    The statistics are computed from the exact routing tensors consumed by the
    terminal.  They intentionally do not inspect targets or a TruthSidecar, so
    exposing them in ``ForwardTrace`` cannot create a supervision side channel.
    Per-query tensors remain available as auxiliaries while the trace stores
    only compact scalar summaries.
    """

    if weights.shape != log_weights.shape or weights.shape != support_mask.shape:
        raise ValueError("router diagnostic tensors must have identical shapes")
    if support_mask.dtype is not torch.bool:
        raise ValueError("router diagnostic support_mask must be bool")
    if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(log_weights).all()):
        raise ValueError("router diagnostic weights must be finite")

    # Receipt diagnostics are host-side evidence.  Moving the detached values
    # to CPU before FP64 aggregation avoids asking MPS to materialize an
    # unsupported float64 tensor while preserving stable public summaries.
    detached_weights = weights.detach().cpu().to(dtype=torch.float64)
    detached_logs = log_weights.detach().cpu().to(dtype=torch.float64)
    detached_support = support_mask.detach().cpu()
    available = detached_support.any(dim=-1)
    source_count = detached_support.sum(dim=-1)
    supported_weights = torch.where(
        detached_support,
        detached_weights,
        torch.zeros_like(detached_weights),
    )
    supported_logs = torch.where(
        detached_support,
        detached_logs,
        torch.zeros_like(detached_logs),
    )
    entropy = (-(supported_weights * supported_logs).sum(dim=-1)).clamp_min(0.0)
    squared_mass = supported_weights.square().sum(dim=-1)
    effective_support = torch.where(
        available,
        squared_mass.clamp_min(torch.finfo(squared_mass.dtype).tiny).reciprocal(),
        torch.zeros_like(squared_mass),
    )
    if detached_support.shape[-1] == 0:
        # Parameterized matching readouts have no empirical support ledger.
        # Keep the diagnostics typed and finite rather than reducing an empty
        # support axis (which would raise on every forward).
        max_weight = torch.zeros_like(available, dtype=detached_weights.dtype)
        minimum_log_weight = torch.zeros_like(max_weight)
        maximum_log_weight = torch.zeros_like(max_weight)
        log_weight_span = torch.zeros_like(max_weight)
    else:
        max_weight = supported_weights.max(dim=-1).values
        positive_infinity = torch.full_like(detached_logs, torch.inf)
        negative_infinity = torch.full_like(detached_logs, -torch.inf)
        minimum_log_weight = (
            torch.where(detached_support, detached_logs, positive_infinity).min(dim=-1).values
        )
        maximum_log_weight = (
            torch.where(detached_support, detached_logs, negative_infinity).max(dim=-1).values
        )
        log_weight_span = torch.where(
            available,
            maximum_log_weight - minimum_log_weight,
            torch.zeros_like(maximum_log_weight),
        )

    def summarize(name: str, values: Tensor) -> dict[str, float]:
        selected = values[available]
        if selected.numel() == 0:
            return {
                f"{name}_minimum": 0.0,
                f"{name}_mean": 0.0,
                f"{name}_maximum": 0.0,
            }
        selected = selected.double()
        return {
            f"{name}_minimum": float(selected.min().item()),
            f"{name}_mean": float(selected.mean().item()),
            f"{name}_maximum": float(selected.max().item()),
        }

    coordinate_values = coordinates.detach().cpu().to(dtype=torch.float64)
    coordinate_rms = (
        float(coordinate_values.square().mean().sqrt().item()) if coordinate_values.numel() else 0.0
    )
    summary: dict[str, Any] = {
        "coordinate_rms": coordinate_rms,
        "query_count": int(available.numel()),
        "available_query_count": int(available.sum().item()),
        **summarize("source_count", source_count),
        **summarize("entropy", entropy),
        **summarize("effective_support_size", effective_support),
        **summarize("max_weight", max_weight),
        **summarize("log_weight_span", log_weight_span),
    }
    # The scalar trace summary above is an explicit CPU/FP64 receipt
    # calculation.  Live per-query auxiliaries return to the model execution
    # device and the repository-wide float32 tensor policy.
    execution_device = coordinates.device
    auxiliaries = {
        "routing_source_count": source_count.to(device=execution_device),
        "routing_entropy": entropy.to(
            device=execution_device,
            dtype=DEFAULT_FLOAT_DTYPE,
        ),
        "routing_effective_support_size": effective_support.to(
            device=execution_device,
            dtype=DEFAULT_FLOAT_DTYPE,
        ),
        "routing_max_weight": max_weight.to(
            device=execution_device,
            dtype=DEFAULT_FLOAT_DTYPE,
        ),
        "routing_log_weight_span": log_weight_span.to(
            device=execution_device,
            dtype=DEFAULT_FLOAT_DTYPE,
        ),
    }
    return summary, auxiliaries


def _make_forward_trace(
    *,
    model_id: str,
    contract_version: str,
    config: ReferenceConfig,
    inputs: DenseModelInput,
    events: Sequence[DenseTraceEvent],
    metadata: Mapping[str, Any],
) -> Any:
    input_hash = hash_dense_input(inputs)
    # A model id names the public contract, not necessarily one executable
    # architecture.  In particular, the cell Rec contract deliberately
    # exposes M/W/RC profiles through one design-open id.  Keep the trace
    # identity tied to the selected semantic variant so receipts from those
    # profiles cannot be mistaken for one another.  Restrict this to stable
    # architecture/contract fields; per-episode diagnostics (routing stats,
    # query markers, shapes) belong to the input/events and must not change
    # model identity.
    variant_keys = (
        "family_id",
        "unit",
        "profile_id",
        "carrier_role",
        "numeric_terminal",
        "dynamics_plan",
        "feature_identity",
        "continuous_tokenizer",
        "nominal_tokenizer",
        "scale_epsilon",
        "label_broadcast",
        "label_broadcast_tau",
        "contract_version",
        "variant_hash",
        "tokenizer_version",
        "ll_ridge",
        "bandwidth",
    )
    variant_payload = {
        key: metadata[key] for key in variant_keys if key in metadata
    }
    model_variant = json.dumps(
        variant_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    model_hash = hashlib.sha256(
        f"{model_id}:{contract_version}:{config.semantic_hash}:{model_variant}".encode()
    ).hexdigest()
    trace_id = hashlib.sha256(
        f"{model_id}:{inputs.episode_id}:{model_hash}:{input_hash}".encode()
    ).hexdigest()
    from tabu_lab.contracts import ForwardTrace, TraceEvent

    contract_events = tuple(
        TraceEvent(
            name=event.stage,
            component=event.stage,
            input_hash=event.input_hash,
            output_hash=event.output_hash,
            metadata=dict(event.metadata),
        )
        for event in events
    )
    trace_metadata = {
        **dict(metadata),
        "model_variant": model_variant,
        # The selected semantic variant is authoritative and cannot be
        # shadowed by a caller-provided diagnostic key.
        "block_kind": config.block_kind.value,
        "variant_role": ("canonical" if config.block_kind.value == "omab" else "non_o_ablation"),
    }
    return ForwardTrace(
        trace_id=trace_id,
        episode_id=inputs.episode_id,
        model_id=model_id,
        model_hash=model_hash,
        input_hash=input_hash,
        events=contract_events,
        metadata=trace_metadata,
    )


class DenseReferenceModel(nn.Module):
    """Base class enforcing the truth-free four-step forward boundary."""

    model_id: str

    def __init__(self, config: ReferenceConfig, *, marker: str, feature_identity: bool = False):
        super().__init__()
        self.config = config
        from tabu_lab.contracts import canonical_hash
        from tabu_lab.registry import get_model_spec

        model_spec = get_model_spec(self.model_id)
        self.contract_version = model_spec.contract_version
        self.model_spec_hash = canonical_hash(model_spec.model_dump(mode="json"))
        self.symbolizer = Symbolizer()
        self.tokenizer = Tokenizer(config, marker=marker, add_feature_identity=feature_identity)

    def forward(self, episode: EvidenceEpisode) -> PredictionBundle:
        """Fail-closed public boundary: only compiled evidence may reach a model."""

        from tabu_lab.contracts import EvidenceEpisode, PredictionBundle

        if not isinstance(episode, EvidenceEpisode):
            raise TypeError(
                "public model forward accepts EvidenceEpisode only; "
                "use the private _forward_dense helper only in implementation tests"
            )
        dense_input = DenseModelInput.from_any(episode)
        # EvidenceEpisode keeps detached CPU storage for stable hashing and
        # auditability.  Materialize a device-local dense carrier only at the
        # executable model boundary, so ``model.to('cuda')(episode)`` works
        # without mutating the truth-free episode or changing its hash.
        model_device = next(self.parameters(), dense_input.values).device
        dense_input = dense_input.to(model_device)
        prediction = self._forward_dense(dense_input)
        if not isinstance(prediction, PredictionBundle):
            raise TypeError("reference model implementation must return PredictionBundle")
        return prediction

    def _dynamics_plan_name(self, dynamics: nn.Module) -> str:
        """Resolve the trace plan name for this model's block variant."""

        return dynamics.plan.resolved_name(self.config.block_kind)

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> PredictionBundle:
        raise NotImplementedError

    def _resolve_inputs(
        self,
        inputs: Any,
        *,
        visible_mask: Tensor | None,
        target_mask: Tensor | None,
        natural_missing_mask: Tensor | None,
        graph: Tensor | None,
        target_feature: int | None,
        episode_id: str | None,
    ) -> DenseModelInput:
        return DenseModelInput.from_any(
            inputs,
            visible_mask=visible_mask,
            target_mask=target_mask,
            natural_missing_mask=natural_missing_mask,
            graph=graph,
            target_feature=target_feature,
            episode_id=episode_id,
        )

    def _bundle(
        self,
        *,
        inputs: DenseModelInput,
        values: Tensor,
        support_available: Tensor,
        coordinates: Tensor,
        routing_weights: Tensor,
        routing_log_weights: Tensor,
        routing_support_mask: Tensor,
        events: Sequence[DenseTraceEvent],
        metadata: Mapping[str, Any],
        support_visible_mask: Tensor | None = None,
        extra_auxiliaries: Mapping[str, Tensor] | None = None,
        emit_trace: bool = True,
    ) -> Any:
        layout = feature_layout(inputs)
        _, n_rows, n_features = inputs.values.shape
        if support_visible_mask is None:
            support_visible_mask = inputs.visible_mask
        if (
            support_visible_mask.shape != inputs.visible_mask.shape
            or support_visible_mask.dtype is not torch.bool
        ):
            raise ValueError("support_visible_mask must be bool and match input values")
        if bool((support_visible_mask & ~inputs.visible_mask).any()):
            raise ValueError("support_visible_mask must be a subset of visible evidence")
        numeric_features = layout.numeric_mask.view(1, 1, n_features)
        categorical_features = layout.categorical_mask.view(1, 1, n_features)
        if metadata.get("numeric_terminal") == "parameterized_matching" and bool(
            layout.categorical_mask.any()
        ):
            raise ValueError(
                "parameterized matching is numeric-only in the current TabU4Rec "
                "mainline; select an explicit categorical appendix terminal"
            )

        # Numeric terminals may use richer support axes (for example Rec's
        # row-plus-column arms).  Categorical probabilities must consume the
        # same declared support geometry as numeric values; otherwise a Rec
        # item arm would silently disappear for categorical interactions.
        numeric_values = values.masked_fill(~numeric_features, 0.0)
        numeric_available = support_available & numeric_features
        numeric_weights = routing_weights * numeric_features.unsqueeze(-1)
        categorical_weights = routing_weights
        if (
            routing_log_weights.shape != routing_weights.shape
            or routing_support_mask.shape != routing_weights.shape
            or routing_support_mask.dtype is not torch.bool
            or not bool(torch.isfinite(routing_log_weights).all())
        ):
            raise ValueError("routing weights, finite log weights, and support mask must align")
        categorical_support_mask = routing_support_mask & categorical_features.unsqueeze(-1)
        finite_floor = -torch.finfo(routing_log_weights.dtype).max
        categorical_log_weights = torch.where(
            categorical_support_mask,
            routing_log_weights,
            torch.full_like(routing_log_weights, finite_floor),
        )
        categorical_routing = RoutingOutput(
            weights=categorical_weights,
            log_weights=categorical_log_weights,
            support_mask=categorical_support_mask,
            support_available=categorical_support_mask.any(dim=-1),
            support_count=categorical_support_mask.sum(dim=-1),
        )
        categorical_readout = None
        if bool(layout.categorical_mask.any()):
            categorical_readout = categorical_from_routing(
                categorical_routing,
                inputs.values,
                support_visible_mask & categorical_features,
                layout.domain_values,
                layout.domain_mask,
            )

        if emit_trace:
            routing_diagnostics, routing_auxiliaries = _routing_diagnostics(
                coordinates=coordinates,
                weights=routing_weights,
                log_weights=routing_log_weights,
                support_mask=routing_support_mask,
            )
        else:
            routing_diagnostics = {}
            routing_auxiliaries = {}
        merged_auxiliaries = dict(extra_auxiliaries or {})
        duplicate_diagnostics = set(merged_auxiliaries).intersection(routing_auxiliaries)
        if duplicate_diagnostics:
            raise ValueError(
                "extra auxiliaries cannot replace router diagnostics: "
                + ", ".join(sorted(duplicate_diagnostics))
            )
        merged_auxiliaries.update(routing_auxiliaries)
        extra_auxiliaries = merged_auxiliaries

        trace = (
            _make_forward_trace(
                model_id=self.model_id,
                contract_version=self.contract_version,
                config=self.config,
                inputs=inputs,
                events=events,
                metadata={
                    "backend": "dense_reference",
                    "truth_boundary": "objective_sidecar_only",
                    "routing_bandwidth": float(self.config.routing_bandwidth),
                    "routing_diagnostics": routing_diagnostics,
                    **dict(metadata),
                },
            )
            if emit_trace
            else None
        )
        target_mask = inputs.target_mask
        unsupported_target_mask = inputs.unsupported_target_mask
        artificial_target_mask = inputs.artificial_target_mask
        query_target_mask = inputs.query_target_mask
        categorical_values = None if categorical_readout is None else categorical_readout.values
        categorical_probabilities = (
            None if categorical_readout is None else categorical_readout.probabilities
        )
        categorical_log_probabilities = (
            None if categorical_readout is None else categorical_readout.log_probabilities
        )
        categorical_class_support_available = (
            None if categorical_readout is None else categorical_readout.class_support_available
        )
        categorical_available = (
            torch.zeros_like(support_available)
            if categorical_readout is None
            else categorical_readout.support_available
        )
        if inputs.squeezed_batch:
            numeric_values = numeric_values[0]
            numeric_available = numeric_available[0]
            numeric_weights = numeric_weights[0]
            categorical_values = None if categorical_values is None else categorical_values[0]
            categorical_probabilities = (
                None if categorical_probabilities is None else categorical_probabilities[0]
            )
            categorical_log_probabilities = (
                None if categorical_log_probabilities is None else categorical_log_probabilities[0]
            )
            categorical_class_support_available = (
                None
                if categorical_class_support_available is None
                else categorical_class_support_available[0]
            )
            categorical_available = categorical_available[0]
            categorical_weights = categorical_weights[0]
            coordinates = coordinates[0]
            target_mask = target_mask[0]
            unsupported_target_mask = unsupported_target_mask[0]
            artificial_target_mask = artificial_target_mask[0]
            query_target_mask = query_target_mask[0]
            if extra_auxiliaries is not None:
                extra_auxiliaries = {name: tensor[0] for name, tensor in extra_auxiliaries.items()}
        from tabu_lab.contracts import (
            PredictionBundle,
            PredictionEntry,
            PredictionKind,
            PredictionStatus,
        )

        numeric_feature_mask = layout.numeric_mask.view(
            *((1,) * (target_mask.ndim - 1)), n_features
        )
        categorical_feature_mask = layout.categorical_mask.view(
            *((1,) * (target_mask.ndim - 1)), n_features
        )
        numeric_target_mask = target_mask & numeric_feature_mask
        categorical_target_mask = target_mask & categorical_feature_mask
        numeric_unsupported = unsupported_target_mask & numeric_feature_mask
        categorical_unsupported = unsupported_target_mask & categorical_feature_mask
        combined_available = torch.where(
            categorical_feature_mask,
            categorical_available,
            numeric_available,
        )
        combined_available = combined_available & ~unsupported_target_mask
        status = "ok"
        eligible_target_mask = target_mask & ~unsupported_target_mask
        if bool(eligible_target_mask.any()):
            active_support = combined_available[eligible_target_mask]
            if not bool(active_support.any()):
                status = "no_support"
            elif not bool(active_support.all()):
                status = "partial_abstention"
        elif bool(unsupported_target_mask.any()):
            status = "unsupported"

        def support_ids_for(weights: Tensor) -> tuple[Tensor, str]:
            n_support = weights.shape[-1]
            if n_support == n_rows:
                support_kind = "row"
            elif n_support == n_rows + n_features:
                support_kind = "row_then_feature"
            else:
                support_kind = "local_axis"
            support_axis = torch.arange(n_support, device=weights.device)
            support_ids = support_axis.view(*((1,) * (weights.ndim - 1)), n_support).expand_as(
                weights
            )
            return support_ids, support_kind

        family_weights = numeric_weights
        if numeric_weights.shape[-1] == categorical_weights.shape[-1]:
            family_weights = torch.where(
                categorical_feature_mask.unsqueeze(-1),
                categorical_weights,
                numeric_weights,
            )
        family_ids, _ = support_ids_for(family_weights)
        completion_support_weights = family_weights.masked_fill(
            ~artificial_target_mask.unsqueeze(-1), 0.0
        )
        label_support_weights = family_weights.masked_fill(~query_target_mask.unsqueeze(-1), 0.0)
        raw_numeric_terminal = metadata.get("numeric_terminal", "nadaraya_watson")
        numeric_terminal = getattr(raw_numeric_terminal, "value", raw_numeric_terminal)
        if numeric_terminal not in {
            "nadaraya_watson",
            "local_linear",
            "parameterized_matching",
        }:
            raise ValueError(
                "numeric_terminal metadata must be nadaraya_watson, local_linear, "
                "or parameterized_matching"
            )
        numeric_terminal_label = (
            "empirical_nadaraya_watson"
            if numeric_terminal == "nadaraya_watson"
            else (
                "empirical_local_linear"
                if numeric_terminal == "local_linear"
                else "parameterized_matching"
            )
        )

        def make_entry(
            *,
            kind: Any,
            entry_values: Tensor,
            family_targets: Tensor,
            family_unsupported: Tensor,
            family_available: Tensor,
            family_weights: Tensor,
            terminal: str,
            entry_metadata: Mapping[str, Any] | None = None,
        ) -> Any:
            eligible = family_targets & ~family_unsupported
            if bool(eligible.any()) and not bool(family_available[eligible].any()):
                entry_status = PredictionStatus.NO_SUPPORT
            elif (
                bool(family_targets.any())
                and not bool(eligible.any())
                and bool(family_unsupported.any())
            ):
                entry_status = PredictionStatus.UNSUPPORTED
            else:
                entry_status = PredictionStatus.OK
            if entry_status in {PredictionStatus.NO_SUPPORT, PredictionStatus.UNSUPPORTED}:
                public_values = None
                public_ids = torch.empty(0, dtype=torch.int64, device=entry_values.device)
                public_weights = torch.empty(
                    0, dtype=family_weights.dtype, device=entry_values.device
                )
                support_kind = "row" if family_weights.shape[-1] == n_rows else "local_axis"
            else:
                public_values = entry_values.masked_fill(
                    family_unsupported.unsqueeze(-1)
                    if entry_values.ndim == family_unsupported.ndim + 1
                    else family_unsupported,
                    0.0,
                )
                public_weights = family_weights.masked_fill(family_unsupported.unsqueeze(-1), 0.0)
                public_ids, support_kind = support_ids_for(public_weights)
            return PredictionEntry(
                kind=kind,
                status=entry_status,
                values=public_values,
                support_ids=public_ids,
                support_weights=public_weights,
                metadata={
                    "terminal": terminal,
                    "support_id_kind": support_kind,
                    **dict(entry_metadata or {}),
                },
            )

        entries: dict[str, Any] = {}
        if bool(layout.numeric_mask.any()):
            entries["numeric"] = make_entry(
                kind=PredictionKind.NUMERIC,
                entry_values=numeric_values,
                family_targets=numeric_target_mask,
                family_unsupported=numeric_unsupported,
                family_available=numeric_available & ~numeric_unsupported,
                family_weights=numeric_weights,
                terminal=numeric_terminal_label,
            )
        if categorical_readout is not None:
            schema_metadata = tuple(
                {
                    "feature_index": index,
                    "kind": layout.kinds[index],
                    "domain": layout.domains[index],
                    "codebook_id": layout.codebook_ids[index],
                }
                for index in range(n_features)
                if bool(layout.categorical_mask[index])
            )
            categorical_kwargs = {
                "family_targets": categorical_target_mask,
                "family_unsupported": categorical_unsupported,
                "family_available": categorical_available & ~categorical_unsupported,
                "family_weights": categorical_weights,
                "entry_metadata": {"feature_schema": schema_metadata},
            }
            entries["categorical"] = make_entry(
                kind=PredictionKind.CATEGORICAL,
                entry_values=categorical_values,
                terminal="same_column_categorical_nadaraya_watson",
                **categorical_kwargs,
            )
            entries["distribution"] = make_entry(
                kind=PredictionKind.DISTRIBUTION,
                entry_values=categorical_probabilities,
                terminal="same_column_categorical_empirical_distribution",
                **categorical_kwargs,
            )
        if not entries:
            raise ValueError("at least one declared feature family is required")
        return PredictionBundle(
            contract_version=self.contract_version,
            episode_id=inputs.episode_id,
            model_id=self.model_id,
            entries=entries,
            auxiliaries={
                "abstention": target_mask & ~combined_available,
                "target_mask": target_mask,
                "support_available": combined_available,
                "numeric_target_mask": numeric_target_mask,
                "numeric_support_available": numeric_available & ~numeric_unsupported,
                "categorical_target_mask": categorical_target_mask,
                "categorical_support_available": categorical_available & ~categorical_unsupported,
                "categorical_domain_values": layout.domain_values,
                "categorical_domain_mask": layout.domain_mask,
                **(
                    {}
                    if categorical_log_probabilities is None
                    else {"categorical_log_probabilities": categorical_log_probabilities}
                ),
                **(
                    {}
                    if categorical_class_support_available is None
                    else {
                        "categorical_class_support_available": (categorical_class_support_available)
                    }
                ),
                "coordinates": coordinates,
                "artificial_target_mask": artificial_target_mask,
                "query_target_mask": query_target_mask,
                "completion_target_mask": artificial_target_mask,
                "completion_support_available": combined_available & artificial_target_mask,
                "completion_support_ids": family_ids,
                "completion_support_weights": completion_support_weights,
                "label_target_mask": query_target_mask,
                "label_support_available": combined_available & query_target_mask,
                "label_support_ids": family_ids,
                "label_support_weights": label_support_weights,
                "unsupported_target_mask": unsupported_target_mask,
                **dict(extra_auxiliaries or {}),
            },
            trace=trace,
            metadata={
                **dict(metadata),
                "contract_version": self.contract_version,
                "prediction_schema_version": "tabu.prediction-bundle.v1",
                "backend": "dense_reference",
                "block_kind": self.config.block_kind.value,
                "variant_role": (
                    "canonical" if self.config.block_kind.value == "omab" else "non_o_ablation"
                ),
                "output_type": "mixed_typed" if categorical_readout is not None else "numeric",
                "distribution": numeric_terminal_label,
                "routing_bandwidth": float(self.config.routing_bandwidth),
                "trace_mode": "full" if emit_trace else "disabled_for_training",
                "status": status,
                "abstention": "empty_support",
                "categorical": (
                    "declared_schema_same_column_nw"
                    if categorical_readout is not None
                    else "not_declared"
                ),
                "supported_output_types": (
                    "numeric",
                    "categorical",
                    "empirical_distribution",
                    "abstention",
                ),
            },
        )


class AugmentedCompletionModel(DenseReferenceModel):
    """Shared explicit carrier for TabUF, TabUFL, TabUL, Rec, and Graph views."""

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        marker: str = "mask",
        supervised: bool = False,
        recommendation: bool = False,
        label_columns: Sequence[int] = (-1,),
        readout_geometry: str = "matched_uf",
        recommendation_address_plan: str = "matched_uf",
        rec_axis_summary_dim: int = 2,
        rec_matched_residual_scale: float = 0.1,
        label_address_plan: str = "matched_uf",
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config, marker=marker)
        self.label_columns = tuple(int(index) for index in label_columns)
        if label_address_plan not in {
            "matched_uf",
            "predictor_only_per_label_v1",
            "predictor_unit_linked_per_label_v2",
        }:
            raise ValueError("unknown supervised label address plan")
        if not supervised and label_address_plan != "matched_uf":
            raise ValueError("label address plans require a supervised model")
        self.unit_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.feature_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.unit_seeds, std=0.02)
        nn.init.normal_(self.feature_seeds, std=0.02)
        if recommendation:
            self.dynamics = RecommendationDynamics(config)
        elif supervised:
            # The v2 L-lane is deliberately independent: completion keeps the
            # TabUF dynamics while labels use a separate predictor-only Unit
            # dynamics below.  This prevents one ledger from borrowing the
            # other's target carrier while retaining both trainable paths.
            self.dynamics = (
                AugmentedDynamics(config)
                if label_address_plan == "predictor_unit_linked_per_label_v2"
                else SupervisedDynamics(config)
            )
        else:
            self.dynamics = AugmentedDynamics(config)
        # ``matched_uf`` is the current model-factory mainline: a literal
        # diagonal User-special/Item-special score.  Historical empirical
        # support remains reachable only through an explicit alternate address
        # plan such as ``axis_address_bootstrap_v1``.
        use_parameterized_matching = (
            recommendation
            and readout_geometry == "matched_uf"
            and recommendation_address_plan == "matched_uf"
        )
        if use_parameterized_matching:
            # The model-factory mainline is a parameterized matched score: it
            # reads the diagonal User-special/Item-special inner products and
            # does not route through empirical response supports.  Keep the
            # older empirical bilinear readout available only when an
            # explicit appendix address plan is requested.
            self.readout = MatchedScoreReadout(config, numeric_terminal=numeric_terminal)
        elif recommendation and recommendation_address_plan == "cell_global_support_v1":
            self.readout = CellTokenGlobalSupportReadout(
                config,
                numeric_terminal=numeric_terminal,
            )
        else:
            self.readout = MatchedUFReadout(
                config,
                bilinear_support=recommendation,
                geometry=readout_geometry,
                recommendation_address_plan=recommendation_address_plan,
                rec_axis_summary_dim=rec_axis_summary_dim,
                rec_matched_residual_scale=rec_matched_residual_scale,
                numeric_terminal=numeric_terminal,
            )
        self.label_address_plan = label_address_plan
        self.label_dynamics = (
            PredictorUnitAddressDynamics(config)
            if supervised and label_address_plan == "predictor_unit_linked_per_label_v2"
            else None
        )
        if supervised and label_address_plan == "predictor_only_per_label_v1":
            self.label_readout = PredictorOnlyLabelReadout(
                config,
                n_labels=len(self.label_columns),
                numeric_terminal=numeric_terminal,
            )
        elif supervised and label_address_plan == "predictor_unit_linked_per_label_v2":
            self.label_readout = PredictorUnitLinkedLabelReadout(
                config,
                n_labels=len(self.label_columns),
                numeric_terminal=numeric_terminal,
            )
        else:
            self.label_readout = None
        self.supervised = bool(supervised)
        self.recommendation = bool(recommendation)

    def _resolved_label_columns(self, n_features: int) -> tuple[int, ...]:
        resolved = tuple(
            index if index >= 0 else n_features + index for index in self.label_columns
        )
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("label_columns must be non-empty and unique")
        if any(index < 0 or index >= n_features for index in resolved):
            raise ValueError("label column is outside the feature axis")
        return resolved

    def _declared_response_columns(self, inputs: DenseModelInput) -> tuple[int, ...]:
        from tabu_lab.contracts import FeatureRole

        declared = tuple(
            index
            for index, spec in enumerate(inputs.feature_specs)
            if spec.role is FeatureRole.RESPONSE
        )
        if self.recommendation:
            if not declared:
                raise ValueError(
                    "TabU4Rec requires at least one schema-declared RESPONSE interaction feature"
                )
            # One episode represents one interaction family.  Cross-item
            # support is meaningful only when response columns share a value
            # family and (for categorical values) one stable codebook.
            response_specs = tuple(inputs.feature_specs[index] for index in declared)
            family_signatures = {
                (spec.kind, spec.domain, spec.codebook_id) for spec in response_specs
            }
            if len(family_signatures) != 1:
                raise ValueError(
                    "TabU4Rec RESPONSE interaction columns must share one typed "
                    "value family and categorical codebook"
                )
            return declared
        if self.supervised:
            configured = self._resolved_label_columns(inputs.values.shape[2])
            if not declared:
                raise ValueError("supervised TabU requires schema-declared response features")
            if declared != configured:
                raise ValueError("declared response features must exactly match label_columns")
            return declared
        return declared

    def _tokenize(self, table: DenseModelInput, token_table: TokenTable) -> Tensor:
        cells = token_table.cells
        if not self.supervised:
            return cells
        labels = self._declared_response_columns(table)
        # TabUFL has both completion masks and query-label markers.  Targets in
        # declared label columns receive the independent query token.
        query_positions = torch.zeros_like(token_table.target_mask)
        query_positions[:, :, list(labels)] = table.query_target_mask[:, :, list(labels)]
        return torch.where(
            query_positions.unsqueeze(-1),
            self.tokenizer.query_token.view(1, 1, 1, -1).expand_as(cells),
            cells,
        )

    def _initial_carrier(
        self, cells: Tensor, visible_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, n_rows, n_features, d_model = cells.shape
        k = self.config.matched_slots
        carrier = cells.new_zeros(batch, n_rows + k, n_features + k, d_model)
        carrier[:, :n_rows, :n_features] = cells
        carrier[:, :n_rows, n_features:] = (
            self.unit_seeds.to(cells.dtype).view(1, 1, k, d_model).expand(batch, n_rows, -1, -1)
        )
        carrier[:, n_rows:, :n_features] = (
            self.feature_seeds.to(cells.dtype)
            .view(1, k, 1, d_model)
            .expand(batch, -1, n_features, -1)
        )

        column_sources = torch.zeros(
            batch, n_rows + k, n_features + k, dtype=torch.bool, device=cells.device
        )
        row_sources = torch.zeros_like(column_sources)
        column_sources[:, :n_rows, :n_features] = visible_mask
        column_sources[:, :n_rows, n_features:] = True
        row_sources[:, :n_rows, :n_features] = visible_mask
        # Added User rows carry the Item-special states.  The Rec contract
        # permits every Item-special state to source the item-axis OMAB, while
        # ordinary cells and the corner remain governed by the visible
        # interaction mask / exact-zero closure.
        row_sources[:, n_rows:, :n_features] = True
        # The factory row-axis contract is value-evidence only: Unit and
        # Feature specials remain receivers, but are not K/V sources unless a
        # model declares a separate structural-support path.  Their required
        # contextualization already occurs through the induced-column stages
        # above.  Allowing specials to source this OMAB silently creates a
        # second, undeclared margin-to-margin communication channel.
        return carrier, column_sources, row_sources

    def _forward_dense(
        self,
        inputs: Any,
        *,
        visible_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        natural_missing_mask: Tensor | None = None,
        graph: Tensor | None = None,
        target_feature: int | None = None,
        episode_id: str | None = None,
    ) -> Any:
        resolved = self._resolve_inputs(
            inputs,
            visible_mask=visible_mask,
            target_mask=target_mask,
            natural_missing_mask=natural_missing_mask,
            graph=graph,
            target_feature=target_feature,
            episode_id=episode_id,
        )
        if not self.supervised and bool(resolved.query_target_mask.any()):
            raise ValueError(
                f"{self.model_id} accepts artificial-mask completion targets, not QUERY origins"
            )
        response_columns = (
            self._declared_response_columns(resolved)
            if self.supervised or self.recommendation
            else ()
        )
        if self.recommendation:
            response_mask = torch.zeros_like(resolved.target_mask)
            response_mask[:, :, list(response_columns)] = True
            if bool((resolved.target_mask & ~response_mask).any()):
                raise ValueError(
                    "TabU4Rec targets must use the declared RESPONSE interaction family"
                )
            if bool((resolved.target_mask & resolved.natural_missing_mask).any()):
                raise ValueError(
                    "TabU4Rec targets must be artificial masks of observed "
                    "interactions; natural-missing cells cannot become targets"
                )
            if bool((resolved.target_mask & ~resolved.artificial_target_mask).any()):
                raise ValueError("TabU4Rec accepts artificial-mask completion targets only")
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        typed_cells = tokens.cells
        cells = self._tokenize(resolved, tokens)
        events = [
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                supervision_boundary="absent",
            ),
            _shape_event(
                "tokenizer",
                cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                null="exact_zero",
            ),
        ]
        if self.supervised:
            before_injection = cells
            labels = response_columns
            # Advanced list indexing has an ``IndexBackward`` path that lowers
            # to nondeterministic ``index_put_with_accumulate`` on Apple MPS.
            # A frozen dense selector expresses the same multi-label sum as a
            # deterministic elementwise contraction on every backend.
            label_set = frozenset(labels)
            label_selector = cells.new_tensor(
                [
                    1.0 if feature_index in label_set else 0.0
                    for feature_index in range(cells.shape[2])
                ]
            ).view(1, 1, -1, 1)
            label_source = (cells * label_selector).sum(dim=2, keepdim=True)
            cells = o_inject(
                cells,
                label_source.expand_as(cells),
                tau=self.config.presence_tau,
            )
            events.append(
                _shape_event(
                    "oinject",
                    cells,
                    input_tensor=before_injection,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    operation_trace=("multi_label_oinject",),
                    simultaneous=True,
                )
            )

        carrier, column_sources, row_sources = self._initial_carrier(cells, resolved.visible_mask)
        query_rows = resolved.query_target_mask.any(dim=-1)
        if self.supervised:
            # Query rows may read context but never write into cross-row slots.
            n_rows, n_features = resolved.values.shape[1:]
            column_sources[:, :n_rows, :n_features] &= ~query_rows.unsqueeze(-1)
        carrier_input = carrier
        carrier = self.dynamics(
            carrier,
            column_source_mask=column_sources,
            row_source_mask=row_sources,
        )
        n_rows, n_features = resolved.values.shape[1:]
        k = self.config.matched_slots
        carrier_null = torch.zeros(carrier.shape[:-1], dtype=torch.bool, device=carrier.device)
        carrier_null[:, :n_rows, :n_features] = resolved.natural_missing_mask
        carrier_null[:, n_rows:, n_features:] = True
        unit_tokens = carrier[:, :n_rows, n_features : n_features + k]
        feature_tokens = carrier[:, n_rows : n_rows + k, :n_features].permute(0, 2, 1, 3)
        evolved_cells = carrier[:, :n_rows, :n_features]
        events.append(
            _shape_event(
                "dynamics_plan",
                carrier,
                input_tensor=carrier_input,
                source_mask=column_sources | row_sources,
                null_mask=carrier_null,
                operation_trace=self.dynamics.plan.stages,
                plan=self._dynamics_plan_name(self.dynamics),
                stages=self.dynamics.plan.stages,
            )
        )
        readout_visible_mask = resolved.visible_mask
        if self.supervised:
            # Neither F completion nor L prediction may borrow any value from
            # a query row. L then narrows further to declared response columns
            # through its target-family ledger below.
            readout_visible_mask = readout_visible_mask & ~query_rows.unsqueeze(-1)
        if self.recommendation:
            response_sources = torch.zeros_like(readout_visible_mask)
            response_sources[:, :, list(response_columns)] = readout_visible_mask[
                :, :, list(response_columns)
            ]
            readout_visible_mask = response_sources
        coordinates, readout, readout_auxiliaries = self.readout.forward_with_auxiliaries(
            unit_tokens,
            feature_tokens,
            resolved.values,
            readout_visible_mask,
            evolved_cells,
        )
        rec_auxiliaries: dict[str, Tensor] = dict(readout_auxiliaries)
        if self.supervised and self.label_readout is not None:
            label_column_mask = torch.zeros_like(resolved.visible_mask)
            label_column_mask[:, :, list(response_columns)] = True
            visible_predictors = resolved.visible_mask & ~label_column_mask
            label_sources = readout_visible_mask & label_column_mask
            predictor_units = None
            if self.label_dynamics is not None:
                predictor_units = self.label_dynamics(
                    typed_cells,
                    visible_predictor_mask=visible_predictors,
                )
                label_coordinates, label_output = self.label_readout(
                    typed_cells,
                    visible_predictors,
                    predictor_units,
                    resolved.values,
                    label_sources,
                    label_columns=response_columns,
                )
                events.append(
                    _shape_event(
                        "predictor_unit_address_dynamics",
                        predictor_units,
                        input_tensor=typed_cells,
                        source_mask=visible_predictors,
                        operation_trace=self.label_dynamics.plan.stages,
                        plan=self._dynamics_plan_name(self.label_dynamics),
                        response_tokens_excluded=True,
                        shared_query_unit=True,
                        truth_not_available=True,
                    )
                )
            else:
                label_coordinates, label_output = self.label_readout(
                    typed_cells,
                    visible_predictors,
                    resolved.values,
                    label_sources,
                    label_columns=response_columns,
                )
            routing_select = label_column_mask.unsqueeze(-1)
            coordinates = torch.where(routing_select, label_coordinates, coordinates)
            readout = NumericReadoutOutput(
                values=torch.where(label_column_mask, label_output.values, readout.values),
                support_available=torch.where(
                    label_column_mask,
                    label_output.support_available,
                    readout.support_available,
                ),
                routing=RoutingOutput(
                    weights=torch.where(
                        routing_select,
                        label_output.routing.weights,
                        readout.routing.weights,
                    ),
                    log_weights=torch.where(
                        routing_select,
                        label_output.routing.log_weights,
                        readout.routing.log_weights,
                    ),
                    support_mask=torch.where(
                        routing_select,
                        label_output.routing.support_mask,
                        readout.routing.support_mask,
                    ),
                    support_available=torch.where(
                        label_column_mask,
                        label_output.routing.support_available,
                        readout.routing.support_available,
                    ),
                    support_count=torch.where(
                        label_column_mask,
                        label_output.routing.support_count,
                        readout.routing.support_count,
                    ),
                ),
            )
            rec_auxiliaries.update(
                {
                    "label_address_coordinates": label_coordinates,
                    "label_address_visible_predictor_mask": visible_predictors,
                    "label_address_visible_predictor_count": visible_predictors.sum(dim=-1),
                    **(
                        {}
                        if predictor_units is None
                        else {"label_address_predictor_units": predictor_units}
                    ),
                }
            )
            events.append(
                _shape_event(
                    (
                        "predictor_unit_linked_label_address"
                        if predictor_units is not None
                        else "predictor_only_label_address"
                    ),
                    label_coordinates,
                    input_tensor=typed_cells,
                    source_mask=visible_predictors,
                    operation_trace=(
                        "same_row_visible_predictor_typed_tokens",
                        "per_label_projection",
                        *(
                            ("shared_predictor_unit_residual",)
                            if predictor_units is not None
                            else ()
                        ),
                        "response_values_terminal_only",
                    ),
                    response_columns=response_columns,
                    response_tokens_excluded=True,
                    query_response_cells_excluded=True,
                    truth_not_available=True,
                )
            )
        if self.recommendation:
            recommendation_operation_trace = (
                (
                    "matched_special_inner_products",
                    "sum_matched_coordinates",
                )
                if not getattr(self.readout, "uses_empirical_support", False)
                else (
                    "same_item_other_users",
                    "same_user_other_response_columns",
                    "equal_active_arm_mix",
                    "single_active_arm_renormalizes_to_one",
                )
            )
            if not getattr(self.readout, "uses_empirical_support", False):
                events.append(
                    _shape_event(
                        "recommendation_matched_readout",
                        coordinates,
                        input_tensor=coordinates,
                        source_mask=readout_visible_mask,
                        operation_trace=recommendation_operation_trace,
                        uses_empirical_support=False,
                        truth_not_available=True,
                    )
                )
            elif (
                getattr(self.readout, "recommendation_address_plan", None)
                == "cell_global_support_v1"
            ):
                events.append(
                    _shape_event(
                        "recommendation_support_ledger",
                        readout.routing.weights,
                        input_tensor=coordinates,
                        source_mask=readout_visible_mask,
                        operation_trace=(
                            "cell_token_projection",
                            "same_item_and_same_user_support_union",
                            "single_joint_routing_normalization",
                        ),
                        arm_order=("user", "item"),
                        response_columns=response_columns,
                        joint_support_pool=True,
                        truth_not_available=True,
                    )
                )
            else:
                user_arm_weights = readout.routing.weights[..., :n_rows]
                item_arm_weights = readout.routing.weights[..., n_rows:]
                user_arm_available = user_arm_weights.sum(dim=-1) > 0
                item_arm_available = item_arm_weights.sum(dim=-1) > 0
                arm_weights = torch.stack(
                    (
                        user_arm_weights.sum(dim=-1),
                        item_arm_weights.sum(dim=-1),
                    ),
                    dim=-1,
                )
                user_mass = arm_weights[..., 0]
                item_mass = arm_weights[..., 1]
                user_source_values = resolved.values.permute(0, 2, 1).unsqueeze(1)
                item_source_values = resolved.values.unsqueeze(2)
                user_arm_values = (
                    user_arm_weights.to(user_source_values.dtype) * user_source_values
                ).sum(dim=-1) / user_mass.clamp_min(torch.finfo(user_mass.dtype).tiny).to(
                    user_source_values.dtype
                )
                item_arm_values = (
                    item_arm_weights.to(item_source_values.dtype) * item_source_values
                ).sum(dim=-1) / item_mass.clamp_min(torch.finfo(item_mass.dtype).tiny).to(
                    item_source_values.dtype
                )
                user_arm_values = torch.where(
                    user_arm_available,
                    user_arm_values,
                    torch.zeros_like(user_arm_values),
                )
                item_arm_values = torch.where(
                    item_arm_available,
                    item_arm_values,
                    torch.zeros_like(item_arm_values),
                )
                rec_auxiliaries = {
                    **rec_auxiliaries,
                    "rec_user_arm_support_weights": user_arm_weights,
                    "rec_item_arm_support_weights": item_arm_weights,
                    "rec_user_arm_support_available": user_arm_available,
                    "rec_item_arm_support_available": item_arm_available,
                    "rec_arm_weights": arm_weights,
                    "rec_user_arm_values": user_arm_values,
                    "rec_item_arm_values": item_arm_values,
                }
                if self.readout.axis_bootstrap is not None:
                    events.append(
                        _shape_event(
                            "recommendation_axis_address",
                            coordinates,
                            input_tensor=rec_auxiliaries["rec_matched_coordinates"],
                            source_mask=readout_visible_mask,
                            operation_trace=(
                                "visible_response_scalar_user_summary",
                                "visible_response_scalar_item_summary",
                                "bounded_matched_uf_residual",
                                "concatenate_axis_address",
                            ),
                            uses_identifiers=False,
                            truth_not_available=True,
                            summary_dim=self.readout.axis_bootstrap.summary_dim,
                            matched_residual_scale=(
                                self.readout.axis_bootstrap.matched_residual_scale
                            ),
                        )
                    )
                events.append(
                    _shape_event(
                        "recommendation_support_ledger",
                        readout.routing.weights,
                        input_tensor=coordinates,
                        source_mask=readout_visible_mask,
                        operation_trace=recommendation_operation_trace,
                        arm_order=("user", "item"),
                        response_columns=response_columns,
                    )
                )
        if self.supervised:
            labels = response_columns
            response_sources = torch.zeros_like(readout_visible_mask)
            response_sources[:, :, list(labels)] = readout_visible_mask[:, :, list(labels)]
            events.extend(
                (
                    _shape_event(
                        "completion_support_ledger",
                        readout.routing.weights * resolved.artificial_target_mask.unsqueeze(-1),
                        input_tensor=coordinates,
                        source_mask=readout_visible_mask,
                        operation_trace=(
                            "exclude_query_rows",
                            "same_column_completion_support",
                        ),
                        family="F",
                    ),
                    _shape_event(
                        "label_support_ledger",
                        readout.routing.weights * resolved.query_target_mask.unsqueeze(-1),
                        input_tensor=coordinates,
                        source_mask=response_sources,
                        operation_trace=(
                            "context_rows_only",
                            "declared_response_columns_only",
                        ),
                        family="L",
                    ),
                )
            )
        events.append(
            _shape_event(
                "readout",
                readout.values,
                input_tensor=coordinates,
                source_mask=readout_visible_mask,
                operation_trace=(
                    self.readout.recommendation_address_plan,
                    self.readout.numeric_terminal_trace,
                ),
                terminal=self.readout.numeric_terminal_trace,
            )
        )
        events.append(
            _shape_event(
                "prediction_boundary",
                resolved.target_mask,
                input_tensor=readout.values,
                source_mask=resolved.visible_mask,
                operation_trace=("model_forward_complete",),
                supervision_boundary="sidecar_only",
                truth_not_available=True,
                model_forward_complete=True,
            )
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
                "support": (
                    "parameterized_matching"
                    if self.recommendation
                    and not getattr(self.readout, "uses_empirical_support", False)
                    else (
                        "declared_response_bilinear_arms" if self.recommendation else "same_column"
                    )
                ),
                "geometry": self.readout.geometry,
                "numeric_terminal": self.readout.numeric_terminal,
                "support_ledgers": (
                    "completion_artificial_mask",
                    "label_context_response",
                )
                if self.supervised
                else ("completion_artificial_mask",),
                "response_columns": response_columns,
                "label_address_plan": self.label_address_plan,
                **(
                    {
                        "response_family": "all_schema_declared_RESPONSE_columns",
                        "recommendation_address_plan": (self.readout.recommendation_address_plan),
                        **(
                            {
                                "arm_order": ("user_same_item", "item_same_user"),
                                "arm_gating": "equal_across_active_arms",
                            }
                            if getattr(self.readout, "uses_empirical_support", False)
                            else {
                                "arm_gating": "not_applicable_to_mainline_matching",
                            }
                        ),
                    }
                    if self.recommendation
                    else {}
                ),
            },
            support_visible_mask=readout_visible_mask,
            extra_auxiliaries=rec_auxiliaries,
        )


class TabUFModel(AugmentedCompletionModel):
    model_id = "tabuf"


class TabUFLModel(AugmentedCompletionModel):
    model_id = "tabufl"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        label_columns: Sequence[int] = (-1,),
        readout_geometry: str = "matched_uf",
        label_address_plan: str = "matched_uf",
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__(
            config,
            marker="mask",
            supervised=True,
            label_columns=label_columns,
            readout_geometry=readout_geometry,
            label_address_plan=label_address_plan,
            numeric_terminal=numeric_terminal,
        )

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        resolved = DenseModelInput.from_any(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        labels = self._declared_response_columns(resolved)
        label_mask = torch.zeros_like(resolved.target_mask)
        label_mask[:, :, list(labels)] = True
        if bool((resolved.artificial_target_mask & label_mask).any()):
            raise ValueError(
                "TabUFL reserves label columns for QUERY targets; "
                "artificial label completion is not in the frozen contract"
            )
        if bool((resolved.query_target_mask & ~label_mask).any()):
            raise ValueError("TabUFL QUERY targets must use declared label columns")
        return super()._forward_dense(resolved)


class TabULModel(AugmentedCompletionModel):
    model_id = "tabul"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        label_columns: Sequence[int] = (-1,),
        readout_geometry: str = "matched_uf",
        label_address_plan: str = "matched_uf",
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__(
            config,
            marker="query",
            supervised=True,
            label_columns=label_columns,
            readout_geometry=readout_geometry,
            label_address_plan=label_address_plan,
            numeric_terminal=numeric_terminal,
        )

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        resolved = DenseModelInput.from_any(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        labels = self._declared_response_columns(resolved)
        outside_labels = resolved.target_mask.clone()
        outside_labels[:, :, list(labels)] = False
        if bool(outside_labels.any()):
            raise ValueError("TabUL has query-label targets only; feature masks belong to TabUFL")
        if bool(resolved.artificial_target_mask.any()):
            raise ValueError("TabUL excludes artificial-mask targets; use TabUFL for mixed F/L")
        return super()._forward_dense(resolved)


class TabU4RecModel(AugmentedCompletionModel):
    model_id = "tabu4rec"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        readout_geometry: str = "matched_uf",
        recommendation_address_plan: str = "matched_uf",
        rec_axis_summary_dim: int = 2,
        rec_matched_residual_scale: float = 0.1,
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        super().__init__(
            config,
            recommendation=True,
            readout_geometry=readout_geometry,
            recommendation_address_plan=recommendation_address_plan,
            rec_axis_summary_dim=rec_axis_summary_dim,
            rec_matched_residual_scale=rec_matched_residual_scale,
            numeric_terminal=numeric_terminal,
        )


class TabU4GraphModel(DenseReferenceModel):
    model_id = "tabu4graph"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        target_feature: int = 0,
        unit_receiver_plan: str = "same_row_visible_cells",
        numeric_terminal: str = "nadaraya_watson",
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config, marker="mask")
        self.default_target_feature = int(target_feature)
        self.unit_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        self.feature_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.unit_seeds, std=0.02)
        nn.init.normal_(self.feature_seeds, std=0.02)
        self.dynamics = GraphDynamics(
            config,
            unit_receiver_plan=unit_receiver_plan,
        )
        self.readout = MatchedUFReadout(config, numeric_terminal=numeric_terminal)

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
            raise ValueError("TabU4Graph accepts artificial-mask targets, not QUERY origins")
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        cells = tokens.cells
        batch, n_rows, n_features, d_model = cells.shape
        target_feature = (
            self.default_target_feature
            if resolved.target_feature is None
            else resolved.target_feature
        )
        if not 0 <= target_feature < n_features:
            raise ValueError("target_feature is outside the feature axis")
        outside_tau = resolved.target_mask.clone()
        outside_tau[:, :, target_feature] = False
        if bool(outside_tau.any()):
            raise ValueError(
                "TabU4Graph frozen contract allows targets only in the single tau column"
            )
        broadcast_source = cells[:, :, target_feature]
        broadcast_source = broadcast_source * resolved.visible_mask[:, :, target_feature].unsqueeze(
            -1
        )
        source = broadcast_source.unsqueeze(2).expand_as(cells)
        broadcast = o_inject(cells, source, tau=self.config.presence_tau)
        keep_target = torch.arange(n_features, device=cells.device) == target_feature
        cells = torch.where(keep_target.view(1, 1, -1, 1), cells, broadcast)
        broadcast_cells = cells
        unit_tokens = (
            self.unit_seeds.to(cells.dtype).view(1, 1, -1, d_model).expand(batch, n_rows, -1, -1)
        )
        feature_tokens = (
            self.feature_seeds.to(cells.dtype)
            .view(1, 1, -1, d_model)
            .expand(batch, n_features, -1, -1)
        )
        graph = resolved.graph
        if graph is None or resolved.graph_topology_hash is None:
            raise ValueError("TabU4Graph public contract requires a typed GraphTopology")
        identity = torch.eye(n_rows, dtype=torch.bool, device=cells.device)
        if graph.ndim == 2:
            closed_graph = graph | graph.transpose(0, 1) | identity
        else:
            closed_graph = graph | graph.transpose(-1, -2) | identity.unsqueeze(0)
        dynamics_input = cells
        cells, unit_tokens, feature_tokens = self.dynamics(
            cells,
            unit_tokens,
            feature_tokens,
            visible_mask=resolved.visible_mask,
            graph=graph,
        )
        coordinates, readout = self.readout(
            unit_tokens,
            feature_tokens,
            resolved.values,
            resolved.visible_mask,
        )
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                supervision_boundary="absent",
            ),
            _shape_event(
                "tokenizer",
                tokens.cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                null="exact_zero",
            ),
            _shape_event(
                "topology_contract",
                closed_graph,
                input_tensor=graph,
                source_mask=graph,
                operation_trace=(
                    f"declared_{resolved.graph_direction}",
                    "symmetrize_G_or_G_transpose",
                    "add_identity",
                ),
                raw_topology_hash=resolved.graph_topology_hash,
            ),
            _shape_event(
                "target_feature_broadcast",
                broadcast_cells,
                input_tensor=tokens.cells,
                source_mask=resolved.visible_mask[:, :, target_feature],
                null_mask=resolved.natural_missing_mask,
                operation_trace=("target_feature_broadcast",),
                target_feature=target_feature,
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
                    "global_feature_prototype_for_readout",
                    f"typed_{self.readout.numeric_terminal_trace}",
                ),
                terminal=f"typed_{self.readout.numeric_terminal_trace}",
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
                "graph_direction": resolved.graph_direction,
                "raw_topology_hash": resolved.graph_topology_hash,
                "closed_neighborhood_hash": _tensor_hash(closed_graph),
                "target_feature": target_feature,
                "graph_operations": self.dynamics.plan.stages,
                "graph_unit_receiver_plan": self.dynamics.unit_receiver_plan,
                "graph_neighborhood": "G_or_G_transpose_or_identity",
                "readout_path": "global_same_column_visible_support",
                "numeric_terminal": self.readout.numeric_terminal,
            },
        )


class TabUUnitRowModel(DenseReferenceModel):
    model_id = "tabu.unit_row"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        numeric_terminal: str = "local_linear",
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config, marker="mask")
        self.unit_seeds = nn.Parameter(
            torch.empty(
                config.matched_slots,
                config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.unit_seeds, std=0.02)
        self.dynamics = RowUnitDynamics(config)
        self.readout = RowUnitReadout(config, numeric_terminal=numeric_terminal)

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
            raise ValueError("Unit-row TabU accepts artificial-mask targets, not QUERY origins")
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        cells = tokens.cells
        batch, n_rows, n_features, d_model = cells.shape
        k = self.config.matched_slots
        carrier = cells.new_zeros(batch, n_rows, n_features + k, d_model)
        carrier[:, :, :n_features] = cells
        carrier[:, :, n_features:] = self.unit_seeds.to(cells.dtype).view(1, 1, k, d_model)
        column_sources = torch.zeros(
            batch, n_rows, n_features + k, dtype=torch.bool, device=cells.device
        )
        column_sources[:, :, :n_features] = resolved.visible_mask
        column_sources[:, :, n_features:] = True
        row_sources = torch.zeros_like(column_sources)
        row_sources[:, :, :n_features] = resolved.visible_mask
        carrier_input = carrier
        carrier = self.dynamics(
            carrier,
            column_source_mask=column_sources,
            row_source_mask=row_sources,
        )
        cells = carrier[:, :, :n_features]
        units = carrier[:, :, n_features:]
        coordinates, readout = self.readout(units, cells, resolved.values, resolved.visible_mask)
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                supervision_boundary="absent",
            ),
            _shape_event(
                "tokenizer",
                tokens.cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                null="exact_zero",
            ),
            _shape_event(
                "dynamics_plan",
                carrier,
                input_tensor=carrier_input,
                source_mask=column_sources | row_sources,
                operation_trace=self.dynamics.plan.stages,
                plan=self._dynamics_plan_name(self.dynamics),
            ),
            _shape_event(
                "readout",
                readout.values,
                input_tensor=coordinates,
                source_mask=resolved.visible_mask,
                operation_trace=(
                    "unit_cell_coordinates",
                    f"numeric_{self.readout.numeric_terminal_trace}",
                ),
                terminal=f"numeric_{self.readout.numeric_terminal_trace}",
                geometry="unit_cell",
                geometry_normalization=self.readout.geometry_normalization,
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
                "unit": "row",
                "numeric_terminal": self.readout.numeric_terminal,
                "geometry_normalization": self.readout.geometry_normalization,
            },
        )


class TabUUnitPairModel(DenseReferenceModel):
    model_id = "tabu.unit_pair"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        numeric_terminal: str = "local_linear",
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config, marker="mask", feature_identity=True)
        self.dynamics = PairUnitDynamics(config)
        self.readout = PairUnitReadout(config, numeric_terminal=numeric_terminal)

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
            raise ValueError("Unit-as-cell TabU accepts artificial-mask targets, not QUERY origins")
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        dynamics_input = tokens.cells
        cells = self.dynamics(
            tokens.cells,
            column_source_mask=resolved.visible_mask,
            row_source_mask=resolved.visible_mask,
        )
        coordinates, readout = self.readout(cells, resolved.values, resolved.visible_mask)
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                unit="cell_pair",
            ),
            _shape_event(
                "tokenizer",
                tokens.cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                feature_identity="compile_time_nonnull",
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
                geometry="projected_cell",
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
                "unit": "pair",
                "numeric_terminal": self.readout.numeric_terminal,
            },
        )


def model_config_dict(model: DenseReferenceModel) -> dict[str, Any]:
    return asdict(model.config)


__all__ = [
    "DenseReferenceModel",
    "TabU4GraphModel",
    "TabU4RecModel",
    "TabUFLModel",
    "TabUFModel",
    "TabULModel",
    "TabUUnitPairModel",
    "TabUUnitRowModel",
    "model_config_dict",
]
