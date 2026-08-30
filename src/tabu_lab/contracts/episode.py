"""Truth-free forward evidence and physically separate supervision sidecars."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .canonical import canonical_hash, require_sha256, to_canonical_data
from .features import FeatureSpec, normalize_feature_specs
from .roles import (
    ForwardRole,
    OriginState,
    encode_forward_roles,
    encode_origin_states,
    forward_role_mask,
    origin_mask,
    origin_value_mask,
)
from .topology import GraphTopology

_FORBIDDEN_TRUTH_KEYS = {
    "ground_truth",
    "label",
    "labels",
    "query_values",
    "supervision",
    "target",
    "target_value",
    "target_values",
    "targets",
    "truth",
    "truth_sidecar",
    "truths",
    "y",
    "y_true",
}


def _normalized_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def assert_truth_free(value: Any, *, path: str = "payload") -> None:
    """Reject truth-bearing names from any model-facing metadata tree."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for item in dataclasses.fields(value):
            key = _normalized_key(item.name)
            if key in _FORBIDDEN_TRUTH_KEYS:
                raise ValueError(f"truth-bearing field {item.name!r} is forbidden at {path}")
            assert_truth_free(getattr(value, item.name), path=f"{path}.{item.name}")
        return
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        assert_truth_free(
            model_dump(mode="python", by_alias=True, exclude_none=False),
            path=path,
        )
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"truth-free metadata keys must be strings at {path}")
            if _normalized_key(key) in _FORBIDDEN_TRUTH_KEYS:
                raise ValueError(f"truth-bearing key {key!r} is forbidden at {path}")
            assert_truth_free(nested, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            assert_truth_free(nested, path=f"{path}[{index}]")


def _freeze_truth_free_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = dict(value or {})
    assert_truth_free(payload, path="metadata")
    to_canonical_data(payload)
    return MappingProxyType(payload)


@dataclass(frozen=True, slots=True)
class EvidenceEpisode:
    """The complete model-facing episode, intentionally incapable of holding truth.

    Only ``SOURCE`` cells with a value-bearing origin may carry non-zero values.
    Receiver and target status are independent bits and never imply visibility.
    """

    episode_id: str
    dataset_id: str
    source_partition: str
    fit_partition: str
    row_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    forward_values: torch.Tensor
    origin_states: torch.Tensor
    forward_roles: torch.Tensor
    feature_specs: tuple[FeatureSpec, ...] = ()
    graph_topology: GraphTopology | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.forward_values).detach().clone()
        if values.ndim != 2 or not values.is_floating_point() or values.is_complex():
            raise ValueError("EvidenceEpisode.forward_values must be a rank-2 real float tensor")
        values = values.to(dtype=DEFAULT_FLOAT_DTYPE)
        origins = encode_origin_states(self.origin_states).to(device=values.device)
        roles = encode_forward_roles(self.forward_roles).to(device=values.device)
        if tuple(values.shape) != tuple(origins.shape) or tuple(values.shape) != tuple(roles.shape):
            raise ValueError("forward values, origin states, and roles must have identical shapes")
        feature_specs = normalize_feature_specs(
            width=values.shape[1],
            feature_specs=self.feature_specs,
            feature_names=self.feature_names,
        )
        feature_names = tuple(spec.name for spec in feature_specs)
        if len(self.row_ids) != values.shape[0]:
            raise ValueError("row_ids and feature_names must match the episode shape")
        if len(set(self.row_ids)) != len(self.row_ids):
            raise ValueError("episode row and feature identifiers must be unique")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("EvidenceEpisode.forward_values must be finite")
        graph_topology = self.graph_topology
        if graph_topology is not None:
            if not isinstance(graph_topology, GraphTopology):
                raise TypeError("EvidenceEpisode.graph_topology must be GraphTopology or None")
            if graph_topology.node_ids != self.row_ids:
                raise ValueError("EvidenceEpisode.graph_topology node_ids must match row_ids")
            assert_truth_free(graph_topology, path="graph_topology")

        receiver = forward_role_mask(roles, ForwardRole.RECEIVER)
        source = forward_role_mask(roles, ForwardRole.SOURCE)
        target = forward_role_mask(roles, ForwardRole.TARGET)
        value_bearing = origin_value_mask(origins)
        target_origin = (
            origin_mask(origins, OriginState.ARTIFICIAL_MASK)
            | origin_mask(origins, OriginState.QUERY)
            | origin_mask(origins, OriginState.NATURAL_MISSING)
        )
        if bool((source & ~value_bearing).any()):
            raise ValueError("SOURCE role requires OBSERVED or INTERVENTION origin")
        if bool((target & ~receiver).any()):
            raise ValueError("TARGET role also requires RECEIVER")
        if bool((target & source).any()):
            raise ValueError("TARGET cells cannot carry SOURCE role")
        if bool((target & ~target_origin).any()):
            raise ValueError(
                "TARGET bits require ARTIFICIAL_MASK, QUERY, or NATURAL_MISSING origin"
            )
        hidden = values[~source]
        if hidden.numel() and not bool((hidden == 0).all()):
            raise ValueError("all non-SOURCE forward values must be physically zero")

        object.__setattr__(self, "forward_values", values)
        object.__setattr__(self, "origin_states", origins)
        object.__setattr__(self, "forward_roles", roles)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "feature_specs", feature_specs)
        object.__setattr__(self, "graph_topology", graph_topology)
        object.__setattr__(self, "metadata", _freeze_truth_free_metadata(self.metadata))

    @property
    def values(self) -> torch.Tensor:
        """Compatibility alias for model adapters."""

        return self.forward_values

    @property
    def visible_values(self) -> torch.Tensor:
        return self.forward_values

    @property
    def receiver_mask(self) -> torch.Tensor:
        return forward_role_mask(self.forward_roles, ForwardRole.RECEIVER)

    @property
    def source_mask(self) -> torch.Tensor:
        return forward_role_mask(self.forward_roles, ForwardRole.SOURCE)

    @property
    def target_mask(self) -> torch.Tensor:
        return forward_role_mask(self.forward_roles, ForwardRole.TARGET)

    @property
    def evidence_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.evidence-episode.v3",
                "episode_id": self.episode_id,
                "dataset_id": self.dataset_id,
                "source_partition": self.source_partition,
                "fit_partition": self.fit_partition,
                "row_ids": self.row_ids,
                "feature_specs": self.feature_specs,
                "graph_topology": self.graph_topology,
                "forward_values": self.forward_values,
                "origin_states": self.origin_states,
                "forward_roles": self.forward_roles,
                "metadata": self.metadata,
            }
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> EvidenceEpisode:
        """Return a truth-free execution copy on ``device``.

        Compilers still emit CPU-canonical episodes.  Device transfer is an
        explicit execution boundary and never introduces a TruthSidecar or
        changes the canonical evidence hash.
        """

        resolved = torch.device(device)
        return type(self)(
            episode_id=self.episode_id,
            dataset_id=self.dataset_id,
            source_partition=self.source_partition,
            fit_partition=self.fit_partition,
            row_ids=self.row_ids,
            feature_names=self.feature_names,
            feature_specs=self.feature_specs,
            forward_values=self.forward_values.to(
                device=resolved,
                dtype=DEFAULT_FLOAT_DTYPE,
                non_blocking=non_blocking,
            ),
            origin_states=self.origin_states.to(device=resolved, non_blocking=non_blocking),
            forward_roles=self.forward_roles.to(device=resolved, non_blocking=non_blocking),
            graph_topology=self.graph_topology,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class TruthSidecar:
    """Loss/evaluation-only values, never passed to a model forward method."""

    episode_id: str
    recipe_hash: str
    row_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    target_values: torch.Tensor
    target_mask: torch.Tensor
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.target_values).detach().clone()
        mask = (
            torch.as_tensor(self.target_mask)
            .detach()
            .clone()
            .to(dtype=torch.bool, device=values.device)
        )
        if values.ndim != 2 or not values.is_floating_point() or values.is_complex():
            raise ValueError("TruthSidecar.target_values must be a rank-2 real float tensor")
        values = values.to(dtype=DEFAULT_FLOAT_DTYPE)
        if tuple(values.shape) != tuple(mask.shape):
            raise ValueError("target_values and target_mask must have identical shapes")
        if len(self.row_ids) != values.shape[0] or len(self.feature_names) != values.shape[1]:
            raise ValueError("row_ids and feature_names must match sidecar shape")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("TruthSidecar.target_values must be finite")
        hidden = values[~mask]
        if hidden.numel() and not bool((hidden == 0).all()):
            raise ValueError("non-target TruthSidecar values must be physically zero")
        to_canonical_data(dict(self.metadata))
        object.__setattr__(
            self,
            "recipe_hash",
            require_sha256(self.recipe_hash, field_name="recipe_hash"),
        )
        object.__setattr__(self, "target_values", values)
        object.__setattr__(self, "target_mask", mask)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def target_count(self) -> int:
        return int(self.target_mask.sum().item())

    @property
    def query_values(self) -> torch.Tensor:
        """Compatibility alias; the frozen contract name is ``target_values``."""

        return self.target_values

    @property
    def query_mask(self) -> torch.Tensor:
        return self.target_mask

    @property
    def query_count(self) -> int:
        return self.target_count

    @property
    def truth_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.truth-sidecar.v1",
                "episode_id": self.episode_id,
                "recipe_hash": self.recipe_hash,
                "row_ids": self.row_ids,
                "feature_names": self.feature_names,
                "target_values": self.target_values,
                "target_mask": self.target_mask,
                "metadata": self.metadata,
            }
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> TruthSidecar:
        """Return an objective-only execution copy on ``device``."""

        resolved = torch.device(device)
        return type(self)(
            episode_id=self.episode_id,
            recipe_hash=self.recipe_hash,
            row_ids=self.row_ids,
            feature_names=self.feature_names,
            target_values=self.target_values.to(
                device=resolved,
                dtype=DEFAULT_FLOAT_DTYPE,
                non_blocking=non_blocking,
            ),
            target_mask=self.target_mask.to(device=resolved, non_blocking=non_blocking),
            metadata=self.metadata,
        )


__all__ = ["EvidenceEpisode", "TruthSidecar", "assert_truth_free"]
