"""Dense, contract-first reference implementations for the TabU model family."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import (
    RoutingOutput,
    categorical_from_routing,
)

from .components import Symbolizer
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
    variant_payload = {key: metadata[key] for key in variant_keys if key in metadata}
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

    def __init__(self, config: ReferenceConfig):
        super().__init__()
        self.config = config
        from tabu_lab.contracts import canonical_hash
        from tabu_lab.registry import get_model_spec, model_spec_identity_payload

        model_spec = get_model_spec(self.model_id)
        self.contract_version = model_spec.contract_version
        self.model_spec_hash = canonical_hash(model_spec_identity_payload(model_spec))
        self.symbolizer = Symbolizer()

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
            support_kind = "row" if n_support == n_rows else "local_axis"
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
        }:
            raise ValueError("numeric_terminal metadata must be nadaraya_watson or local_linear")
        numeric_terminal_label = (
            "empirical_nadaraya_watson"
            if numeric_terminal == "nadaraya_watson"
            else "empirical_local_linear"
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
            numeric_entry_metadata = {}
            if "numeric_prediction_scale" in metadata:
                numeric_entry_metadata = {
                    "value_space": metadata["numeric_prediction_scale"],
                    "raw_prediction_key": "numeric_raw_prediction",
                    "context_mean_key": "numeric_context_mean",
                    "context_scale_key": "numeric_context_scale",
                }
            entries["numeric"] = make_entry(
                kind=PredictionKind.NUMERIC,
                entry_values=numeric_values,
                family_targets=numeric_target_mask,
                family_unsupported=numeric_unsupported,
                family_available=numeric_available & ~numeric_unsupported,
                family_weights=numeric_weights,
                terminal=numeric_terminal_label,
                entry_metadata=numeric_entry_metadata,
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
