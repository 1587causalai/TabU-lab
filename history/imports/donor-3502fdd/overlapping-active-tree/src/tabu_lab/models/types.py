"""Shared, truth-free types for dense TabU reference models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE


class DynamicsBlockKind(StrEnum):
    """The global dynamics block variant selected for one model run."""

    OMAB = "omab"
    MAB = "mab"


class TabUCellBaseProfile(StrEnum):
    """Public evidence profiles for the frozen TabUBase 0.2 contract."""

    COMPLETION_ARTIFICIAL_MASK_V1 = "completion.artificial_mask.v1"
    SUPERVISED_LABEL_BROADCAST_V1 = "supervised.label_broadcast.v1"

    @property
    def uses_label_broadcast(self) -> bool:
        return self is TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1


@dataclass(frozen=True)
class ModelVariantRef:
    """Content-addressed identity carried by traces, checkpoints and receipts."""

    contract_id: str
    contract_version: str
    profile_id: str
    model_spec_hash: str
    source_identity: str
    semantic_config_hash: str

    def __post_init__(self) -> None:
        for name in ("contract_id", "contract_version", "profile_id", "source_identity"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("model_spec_hash", "semantic_config_hash"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def as_dict(self) -> dict[str, str]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ReferenceConfig:
    """Small deterministic defaults for the dense reference backend."""

    d_model: int = 32
    n_heads: int = 4
    d_ff: int = 64
    n_blocks: int = 2
    inducing_slots: int = 4
    matched_slots: int = 4
    max_features: int = 256
    dropout: float = 0.0
    presence_tau: float = 1.0e-6
    denominator_epsilon: float = 1.0e-8
    routing_bandwidth: float = 1.0
    # Router-local address gauge. ``rms_unit`` normalizes every evolved
    # Unit/Feature token independently before matched inner products.  It is
    # an explicit experimental plan, not a silent change to the ``none``
    # reference geometry.
    geometry_normalization: str = "none"
    # ``omab`` is the canonical O-closed implementation.  ``mab`` is a
    # parameter-isomorphic non-O control used for paired ablations.  Keep
    # this field last to preserve the positional layout of legacy configs.
    block_kind: DynamicsBlockKind = DynamicsBlockKind.OMAB

    def __post_init__(self) -> None:
        try:
            block_kind = DynamicsBlockKind(self.block_kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("block_kind must be 'omab' or 'mab'") from exc
        object.__setattr__(self, "block_kind", block_kind)
        integer_fields = (
            "d_model",
            "n_heads",
            "d_ff",
            "n_blocks",
            "inducing_slots",
            "matched_slots",
            "max_features",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("presence_tau", "denominator_epsilon", "routing_bandwidth"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.geometry_normalization not in {"none", "rms_unit"}:
            raise ValueError("geometry_normalization must be none or rms_unit")

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _as_bool_mask(value: Any, *, like: Tensor, name: str) -> Tensor:
    mask = torch.as_tensor(value, device=like.device)
    if mask.shape != like.shape:
        raise ValueError(f"{name} must match values shape")
    return mask.to(torch.bool)


@dataclass(frozen=True)
class DenseModelInput:
    """Truth-free dense carrier input.

    ``values`` contains physical zeros outside ``visible_mask``.  Target truth
    is intentionally absent and belongs to the training/evaluation sidecar.
    """

    values: Tensor
    visible_mask: Tensor
    target_mask: Tensor
    natural_missing_mask: Tensor
    artificial_target_mask: Tensor | None = None
    query_target_mask: Tensor | None = None
    unsupported_target_mask: Tensor | None = None
    feature_specs: tuple[Any, ...] = ()
    graph: Tensor | None = None
    graph_topology_hash: str | None = None
    graph_direction: str | None = None
    row_ids: tuple[str, ...] = ()
    target_feature: int | None = None
    episode_id: str = "dense-episode"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    squeezed_batch: bool = False

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values)
        if values.is_complex():
            raise ValueError("values must be real-valued")
        values = values.to(dtype=DEFAULT_FLOAT_DTYPE)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("DenseModelInput.values must be finite float32 values")
        object.__setattr__(self, "values", values)
        if values.ndim != 3:
            raise ValueError("values must be [B,N,M]")
        for name in ("visible_mask", "target_mask", "natural_missing_mask"):
            mask = getattr(self, name)
            if mask.shape != self.values.shape or mask.dtype is not torch.bool:
                raise ValueError(f"{name} must be bool and match values")
        artificial = self.artificial_target_mask
        query = self.query_target_mask
        unsupported = self.unsupported_target_mask
        if artificial is None and query is None and unsupported is None:
            artificial = self.target_mask.clone()
            query = torch.zeros_like(self.target_mask)
            unsupported = torch.zeros_like(self.target_mask)
        if artificial is None:
            artificial = torch.zeros_like(self.target_mask)
        if query is None:
            query = torch.zeros_like(self.target_mask)
        if unsupported is None:
            unsupported = torch.zeros_like(self.target_mask)
        object.__setattr__(self, "artificial_target_mask", artificial)
        object.__setattr__(self, "query_target_mask", query)
        object.__setattr__(self, "unsupported_target_mask", unsupported)
        for name, mask in (
            ("artificial_target_mask", artificial),
            ("query_target_mask", query),
            ("unsupported_target_mask", unsupported),
        ):
            if mask.shape != self.values.shape or mask.dtype is not torch.bool:
                raise ValueError(f"{name} must be bool and match values")
            if bool((mask & ~self.target_mask).any()):
                raise ValueError(f"{name} must be a subset of target_mask")
        if (
            bool((artificial & query).any())
            or bool((artificial & unsupported).any())
            or bool((query & unsupported).any())
        ):
            raise ValueError("target-origin masks must be disjoint")
        if not torch.equal(artificial | query | unsupported, self.target_mask):
            raise ValueError("target-origin masks must exactly partition target_mask")
        if bool((self.visible_mask & self.target_mask).any()):
            raise ValueError("visible and target masks must be disjoint")
        if bool((self.visible_mask & self.natural_missing_mask).any()):
            raise ValueError("visible and natural-missing masks must be disjoint")
        if self.graph is not None:
            batch, n_rows, _ = self.values.shape
            if self.graph.shape not in {(n_rows, n_rows), (batch, n_rows, n_rows)}:
                raise ValueError("graph must be [N,N] or [B,N,N]")
        if self.graph_topology_hash is not None and (
            len(self.graph_topology_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.graph_topology_hash)
        ):
            raise ValueError("graph_topology_hash must be a lowercase SHA-256")
        if self.graph_direction is not None and self.graph_direction not in {
            "directed",
            "undirected",
        }:
            raise ValueError("graph_direction must be directed or undirected")
        row_ids = tuple(self.row_ids)
        if row_ids and len(row_ids) != self.values.shape[1]:
            raise ValueError("row_ids must match the row axis")
        object.__setattr__(self, "row_ids", row_ids)
        if self.target_feature is not None and not 0 <= self.target_feature < self.values.shape[2]:
            raise ValueError("target_feature is outside the feature axis")
        specs = tuple(self.feature_specs)
        if specs and len(specs) != self.values.shape[2]:
            raise ValueError("feature_specs must match the feature axis")
        object.__setattr__(self, "feature_specs", specs)

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> DenseModelInput:
        """Materialize every tensor field on one model execution device.

        ``EvidenceEpisode`` deliberately owns detached CPU storage so its
        canonical hash and audit representation do not depend on execution
        hardware.  Models, however, must be able to consume the same episode
        when their parameters live on CUDA (or another device).  This method
        is the explicit boundary between those two representations; metadata,
        feature declarations, and provenance remain unchanged.
        """

        resolved_device = torch.device(device)

        def move(value: Tensor | None) -> Tensor | None:
            return (
                None
                if value is None
                else value.to(
                    device=resolved_device,
                    non_blocking=non_blocking,
                )
            )

        return type(self)(
            values=self.values.to(
                device=resolved_device,
                dtype=DEFAULT_FLOAT_DTYPE,
                non_blocking=non_blocking,
            ),
            visible_mask=self.visible_mask.to(device=resolved_device, non_blocking=non_blocking),
            target_mask=self.target_mask.to(device=resolved_device, non_blocking=non_blocking),
            natural_missing_mask=self.natural_missing_mask.to(
                device=resolved_device, non_blocking=non_blocking
            ),
            artificial_target_mask=move(self.artificial_target_mask),
            query_target_mask=move(self.query_target_mask),
            unsupported_target_mask=move(self.unsupported_target_mask),
            feature_specs=self.feature_specs,
            graph=move(self.graph),
            graph_topology_hash=self.graph_topology_hash,
            graph_direction=self.graph_direction,
            row_ids=self.row_ids,
            target_feature=self.target_feature,
            episode_id=self.episode_id,
            metadata=self.metadata,
            squeezed_batch=self.squeezed_batch,
        )

    @classmethod
    def from_any(
        cls,
        value: Any,
        *,
        visible_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        natural_missing_mask: Tensor | None = None,
        unsupported_target_mask: Tensor | None = None,
        graph: Tensor | None = None,
        target_feature: int | None = None,
        episode_id: str | None = None,
    ) -> DenseModelInput:
        if isinstance(value, cls):
            if any(
                override is not None
                for override in (
                    visible_mask,
                    target_mask,
                    natural_missing_mask,
                    unsupported_target_mask,
                    graph,
                    target_feature,
                    episode_id,
                )
            ):
                raise ValueError("do not override fields of an existing DenseModelInput")
            return value

        raw_values = value if isinstance(value, Tensor) else None
        if raw_values is None:
            for name in ("forward_values", "values", "model_values"):
                candidate = getattr(value, name, None)
                if candidate is not None:
                    raw_values = torch.as_tensor(candidate)
                    break
        if raw_values is None:
            raise TypeError("model input must be a Tensor, DenseModelInput, or evidence episode")
        if raw_values.is_complex():
            raise ValueError("input values must be real-valued")
        squeezed = raw_values.ndim == 2
        if squeezed:
            raw_values = raw_values.unsqueeze(0)
        if raw_values.ndim != 3:
            raise ValueError("input values must be [N,M] or [B,N,M]")
        source_finite = torch.isfinite(raw_values)
        raw_values = raw_values.to(DEFAULT_FLOAT_DTYPE)
        float32_finite = torch.isfinite(raw_values)

        external_visible = visible_mask
        external_target = target_mask
        external_missing = natural_missing_mask
        external_artificial = None
        external_query = None
        external_unsupported = unsupported_target_mask
        origins = None if isinstance(value, Tensor) else getattr(value, "origin_states", None)
        roles = None if isinstance(value, Tensor) else getattr(value, "forward_roles", None)

        if external_visible is None and not isinstance(value, Tensor):
            external_visible = getattr(value, "source_mask", None)
        if external_visible is None and not isinstance(value, Tensor):
            external_visible = getattr(value, "visible_mask", None)
        if external_target is None and not isinstance(value, Tensor):
            external_target = getattr(value, "target_mask", None)
        if external_missing is None and not isinstance(value, Tensor):
            external_missing = getattr(value, "natural_missing_mask", None)

        if origins is not None or roles is not None:
            from tabu_lab.contracts import ForwardRole, OriginState, origin_code

            origin_tensor = None if origins is None else torch.as_tensor(origins)
            role_tensor = None if roles is None else torch.as_tensor(roles)
            if origin_tensor is not None and origin_tensor.ndim == 2:
                origin_tensor = origin_tensor.unsqueeze(0)
            if role_tensor is not None and role_tensor.ndim == 2:
                role_tensor = role_tensor.unsqueeze(0)
            value_bearing = None
            if origin_tensor is not None:
                value_bearing = (origin_tensor == origin_code(OriginState.OBSERVED)) | (
                    origin_tensor == origin_code(OriginState.INTERVENTION)
                )
            if external_visible is None:
                if role_tensor is not None:
                    external_visible = (role_tensor & int(ForwardRole.SOURCE)) != 0
                    if value_bearing is not None:
                        external_visible = external_visible & value_bearing
                elif value_bearing is not None:
                    external_visible = value_bearing
            if external_target is None and role_tensor is not None:
                external_target = (role_tensor & int(ForwardRole.TARGET)) != 0
            if external_missing is None and origin_tensor is not None:
                external_missing = origin_tensor == origin_code(OriginState.NATURAL_MISSING)
            if external_unsupported is None and origin_tensor is not None:
                external_unsupported = origin_tensor == origin_code(OriginState.NATURAL_MISSING)
            if origin_tensor is not None:
                external_artificial = origin_tensor == origin_code(OriginState.ARTIFICIAL_MASK)
                external_query = origin_tensor == origin_code(OriginState.QUERY)

        if external_visible is None:
            external_visible = source_finite
        if external_target is None:
            external_target = torch.zeros_like(raw_values, dtype=torch.bool)
        if external_missing is None:
            external_missing = ~(
                torch.as_tensor(external_visible) | torch.as_tensor(external_target)
            )
        if external_unsupported is None:
            external_unsupported = torch.zeros_like(raw_values, dtype=torch.bool)
        if external_artificial is None:
            external_artificial = torch.as_tensor(external_target) & ~torch.as_tensor(
                external_unsupported
            )
        if external_query is None:
            external_query = torch.zeros_like(raw_values, dtype=torch.bool)

        def batch_mask(mask: Any, name: str) -> Tensor:
            tensor = torch.as_tensor(mask, device=raw_values.device)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            return _as_bool_mask(tensor, like=raw_values, name=name)

        resolved_visible = batch_mask(external_visible, "visible_mask") & source_finite
        if bool((resolved_visible & ~float32_finite).any()):
            raise ValueError("visible input values must fit in float32")
        resolved_target = batch_mask(external_target, "target_mask")
        resolved_missing = batch_mask(external_missing, "natural_missing_mask")
        resolved_unsupported = (
            batch_mask(external_unsupported, "unsupported_target_mask") & resolved_target
        )
        resolved_artificial = (
            batch_mask(external_artificial, "artificial_target_mask") & resolved_target
        )
        resolved_query = batch_mask(external_query, "query_target_mask") & resolved_target
        forward_values = torch.nan_to_num(raw_values).masked_fill(~resolved_visible, 0.0)

        resolved_graph = graph
        graph_topology = (
            None if isinstance(value, Tensor) else getattr(value, "graph_topology", None)
        )
        graph_topology_hash = None
        graph_direction = None
        if graph_topology is not None:
            if resolved_graph is not None:
                raise ValueError("do not combine typed graph_topology with a graph override")
            resolved_graph = graph_topology.adjacency
            graph_topology_hash = str(graph_topology.topology_hash)
            graph_direction = str(graph_topology.direction)
        if resolved_graph is None and not isinstance(value, Tensor):
            resolved_graph = getattr(value, "graph", None)
        if resolved_graph is not None:
            resolved_graph = torch.as_tensor(resolved_graph, device=raw_values.device).to(
                torch.bool
            )
        resolved_feature = target_feature
        if resolved_feature is None and not isinstance(value, Tensor):
            resolved_feature = getattr(value, "target_feature", None)
        resolved_episode = episode_id
        if resolved_episode is None and not isinstance(value, Tensor):
            resolved_episode = getattr(value, "episode_id", None)
        metadata = {} if isinstance(value, Tensor) else dict(getattr(value, "metadata", {}) or {})
        feature_specs = (
            () if isinstance(value, Tensor) else tuple(getattr(value, "feature_specs", ()) or ())
        )
        row_ids = () if isinstance(value, Tensor) else tuple(getattr(value, "row_ids", ()) or ())
        return cls(
            values=forward_values,
            visible_mask=resolved_visible,
            target_mask=resolved_target,
            natural_missing_mask=resolved_missing,
            artificial_target_mask=resolved_artificial,
            query_target_mask=resolved_query,
            unsupported_target_mask=resolved_unsupported,
            feature_specs=feature_specs,
            graph=resolved_graph,
            graph_topology_hash=graph_topology_hash,
            graph_direction=graph_direction,
            row_ids=row_ids,
            target_feature=None if resolved_feature is None else int(resolved_feature),
            episode_id=str(resolved_episode or "dense-episode"),
            metadata=metadata,
            squeezed_batch=squeezed,
        )


@dataclass(frozen=True)
class FeatureLayout:
    """Device-local masks and code domains derived only from declared schema."""

    kinds: tuple[str, ...]
    roles: tuple[str, ...]
    domains: tuple[tuple[str, ...], ...]
    codebook_ids: tuple[str | None, ...]
    numeric_mask: Tensor
    categorical_mask: Tensor
    domain_values: Tensor
    domain_mask: Tensor


def feature_layout(inputs: DenseModelInput) -> FeatureLayout:
    """Resolve typed feature families without consulting a truth sidecar.

    Categorical and ordinal values use the zero-based codes defined by the
    ordered ``FeatureSpec.domain``.  The labels themselves remain provenance;
    model-facing values never attempt to parse or infer them.
    """

    from tabu_lab.contracts import FeatureKind, FeatureRole

    n_features = inputs.values.shape[2]
    if inputs.feature_specs:
        kinds = tuple(str(spec.kind) for spec in inputs.feature_specs)
        roles = tuple(str(spec.role) for spec in inputs.feature_specs)
        domains = tuple(tuple(spec.domain) for spec in inputs.feature_specs)
        codebook_ids = tuple(spec.codebook_id for spec in inputs.feature_specs)
    else:
        kinds = (FeatureKind.NUMERIC.value,) * n_features
        roles = (FeatureRole.PREDICTOR.value,) * n_features
        domains = ((),) * n_features
        codebook_ids = (None,) * n_features
    numeric_flags = tuple(kind == FeatureKind.NUMERIC.value for kind in kinds)
    categorical_flags = tuple(not flag for flag in numeric_flags)
    device = inputs.values.device
    numeric_mask = torch.tensor(numeric_flags, dtype=torch.bool, device=device)
    categorical_mask = torch.tensor(categorical_flags, dtype=torch.bool, device=device)
    max_domain = max((len(domain) for domain in domains), default=0)
    domain_values = torch.zeros(
        n_features,
        max_domain,
        dtype=inputs.values.dtype,
        device=device,
    )
    domain_mask = torch.zeros(
        n_features,
        max_domain,
        dtype=torch.bool,
        device=device,
    )
    for feature, domain in enumerate(domains):
        if not domain:
            continue
        domain_values[feature, : len(domain)] = torch.arange(
            len(domain), dtype=inputs.values.dtype, device=device
        )
        domain_mask[feature, : len(domain)] = True
    return FeatureLayout(
        kinds=kinds,
        roles=roles,
        domains=domains,
        codebook_ids=codebook_ids,
        numeric_mask=numeric_mask,
        categorical_mask=categorical_mask,
        domain_values=domain_values,
        domain_mask=domain_mask,
    )


@dataclass(frozen=True)
class DenseTraceEvent:
    stage: str
    shape: tuple[int, ...]
    input_hash: str
    output_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DesignOpenBuild:
    model_id: str
    status: str = "design_open"
    reason: str = (
        "The model-factory source keeps intervention semantics and causal identification open; "
        "a forward implementation would silently invent a contract."
    )
    open_questions: tuple[str, ...] = (
        "scenario versus intervention semantics",
        "allowed factual evidence and post-treatment variables",
        "paired-outcome or structural-identification assumptions",
        "typed prediction, contrast, and abstention contract",
    )


def hash_dense_input(inputs: DenseModelInput) -> str:
    digest = hashlib.sha256()
    for tensor in (
        inputs.values,
        inputs.visible_mask,
        inputs.target_mask,
        inputs.natural_missing_mask,
        inputs.artificial_target_mask,
        inputs.query_target_mask,
        inputs.unsupported_target_mask,
    ):
        detached = tensor.detach().cpu().contiguous()
        if detached.dtype is torch.bfloat16:
            detached = detached.float()
        digest.update(str(tuple(detached.shape)).encode())
        digest.update(str(detached.dtype).encode())
        digest.update(detached.numpy().tobytes())
    if inputs.graph is not None:
        detached_graph = inputs.graph.detach().cpu().contiguous()
        digest.update(str(tuple(detached_graph.shape)).encode())
        digest.update(detached_graph.numpy().tobytes())
    digest.update(str(inputs.graph_topology_hash or "").encode())
    digest.update(str(inputs.graph_direction or "").encode())
    digest.update(json.dumps(inputs.row_ids, separators=(",", ":")).encode())
    schema = [
        {
            "name": str(getattr(spec, "name", "")),
            "kind": str(getattr(spec, "kind", "numeric")),
            "role": str(getattr(spec, "role", "predictor")),
            "domain": list(getattr(spec, "domain", ())),
            "codebook_id": getattr(spec, "codebook_id", None),
        }
        for spec in inputs.feature_specs
    ]
    digest.update(json.dumps(schema, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


__all__ = [
    "DenseModelInput",
    "DenseTraceEvent",
    "DesignOpenBuild",
    "DynamicsBlockKind",
    "FeatureLayout",
    "ReferenceConfig",
    "feature_layout",
    "hash_dense_input",
]
