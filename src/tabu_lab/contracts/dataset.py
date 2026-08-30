"""Dataset, split, and episode-recipe contracts.

The compiler only accepts :class:`SplitView`, never :class:`RawDataset`.  That
small type boundary makes split-before-compile observable and fail-closed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    origin_code,
    origin_value_mask,
)
from .topology import GraphTopology


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = dict(value or {})
    # Validate now rather than discovering an unsupported value during a run.
    to_canonical_data(payload)
    return MappingProxyType(payload)


def _nonempty_identifier(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class RawDataset:
    """One physically normalized two-dimensional table.

    Categorical values may be represented by numeric codes; their schema is
    carried by ``feature_specs``.  Non-observed cells must be zero, while the
    independent ``origin_states`` channel records their source semantics.
    """

    dataset_id: str
    values: torch.Tensor
    origin_states: torch.Tensor | Sequence[Sequence[OriginState | str | int]]
    row_ids: tuple[str, ...] = ()
    feature_names: tuple[str, ...] = ()
    feature_specs: tuple[FeatureSpec, ...] = ()
    graph_topology: GraphTopology | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dataset_id = _nonempty_identifier(self.dataset_id, field_name="dataset_id")
        values = torch.as_tensor(self.values).detach().clone().to(device="cpu")
        if values.ndim != 2 or values.numel() == 0:
            raise ValueError("RawDataset.values must be a non-empty rank-2 tensor")
        if values.is_complex():
            raise ValueError("RawDataset.values must be real-valued")
        values = values.to(dtype=DEFAULT_FLOAT_DTYPE)
        origins = encode_origin_states(self.origin_states).to(device=values.device)
        if tuple(origins.shape) != tuple(values.shape):
            raise ValueError("origin_states must match values shape")
        value_bearing = origin_value_mask(origins)
        if not bool(torch.isfinite(values[value_bearing]).all()):
            raise ValueError("value-bearing RawDataset values must be finite")
        hidden = values[~value_bearing]
        if hidden.numel() and not bool((hidden == 0).all()):
            raise ValueError("non-value-bearing RawDataset values must be physically zero")

        rows = self.row_ids or tuple(f"row-{index}" for index in range(values.shape[0]))
        feature_specs = normalize_feature_specs(
            width=values.shape[1],
            feature_specs=self.feature_specs,
            feature_names=self.feature_names,
        )
        features = tuple(spec.name for spec in feature_specs)
        if len(rows) != values.shape[0] or len(set(rows)) != len(rows):
            raise ValueError("row_ids must be unique and match the row dimension")
        if len(features) != values.shape[1] or len(set(features)) != len(features):
            raise ValueError("feature_names must be unique and match the feature dimension")
        if any(not row_id.strip() for row_id in rows):
            raise ValueError("row_ids cannot contain empty identifiers")
        if any(not feature.strip() for feature in features):
            raise ValueError("feature_names cannot contain empty identifiers")
        graph_topology = self.graph_topology
        if graph_topology is not None:
            if not isinstance(graph_topology, GraphTopology):
                raise TypeError("RawDataset.graph_topology must be GraphTopology or None")
            if graph_topology.node_ids != tuple(rows):
                raise ValueError("RawDataset.graph_topology node_ids must match row_ids")

        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin_states", origins)
        object.__setattr__(self, "row_ids", tuple(rows))
        object.__setattr__(self, "feature_names", tuple(features))
        object.__setattr__(self, "feature_specs", feature_specs)
        object.__setattr__(self, "graph_topology", graph_topology)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def from_values(
        cls,
        *,
        dataset_id: str,
        values: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        row_ids: Sequence[str] = (),
        feature_names: Sequence[str] = (),
        feature_specs: Sequence[FeatureSpec] = (),
        graph_topology: GraphTopology | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RawDataset:
        """Normalize NaN/Inf or an explicit observed mask into physical zeros."""

        source = torch.as_tensor(values).detach().clone().to(device="cpu")
        if source.is_complex():
            raise ValueError("RawDataset.values must be real-valued")
        source_finite = torch.isfinite(source)
        if observed_mask is None:
            observed = source_finite
        else:
            observed = torch.as_tensor(
                observed_mask,
                device=source.device,
                dtype=torch.bool,
            )
            if observed.shape != source.shape:
                raise ValueError("observed_mask must match values shape")
            observed = observed & source_finite
        raw = source.to(dtype=DEFAULT_FLOAT_DTYPE)
        if bool((observed & ~torch.isfinite(raw)).any()):
            raise ValueError("observed RawDataset values must fit in float32")
        if observed.shape != raw.shape:
            raise ValueError("observed_mask must match values shape")
        normalized = torch.where(observed, raw, torch.zeros_like(raw))
        origins = torch.where(
            observed,
            torch.full_like(observed, origin_code(OriginState.OBSERVED), dtype=torch.uint8),
            torch.full_like(
                observed,
                origin_code(OriginState.NATURAL_MISSING),
                dtype=torch.uint8,
            ),
        )
        return cls(
            dataset_id=dataset_id,
            values=normalized,
            origin_states=origins,
            row_ids=tuple(row_ids),
            feature_names=tuple(feature_names),
            feature_specs=tuple(feature_specs),
            graph_topology=graph_topology,
            metadata=dict(metadata or {}),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.values.shape)  # type: ignore[return-value]

    @property
    def dataset_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.raw-dataset.v3",
                "dataset_id": self.dataset_id,
                "values": self.values,
                "origin_states": self.origin_states,
                "row_ids": self.row_ids,
                "feature_specs": self.feature_specs,
                "graph_topology": self.graph_topology,
                "metadata": self.metadata,
            }
        )

    @property
    def content_hash(self) -> str:
        return self.dataset_hash


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Content-bound row partitions selected before episode compilation."""

    dataset_id: str
    dataset_hash: str
    partitions: Mapping[str, tuple[str, ...]]
    split_id: str = "default"
    fit_partition: str = "train"
    strategy: str = "explicit"
    seed: int | None = None
    require_complete: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_partitions: dict[str, tuple[str, ...]] = {}
        for name, members in self.partitions.items():
            partition_name = _nonempty_identifier(name, field_name="partition name")
            if partition_name in normalized_partitions:
                raise ValueError(f"duplicate normalized partition name: {partition_name!r}")
            row_ids = tuple(members)
            if not row_ids or len(row_ids) != len(set(row_ids)):
                raise ValueError(f"partition {partition_name!r} must contain unique row ids")
            if any(not row_id.strip() for row_id in row_ids):
                raise ValueError(f"partition {partition_name!r} contains an empty row id")
            normalized_partitions[partition_name] = row_ids
        if not normalized_partitions:
            raise ValueError("SplitManifest needs at least one partition")
        if self.fit_partition not in normalized_partitions:
            raise ValueError("fit_partition must name one manifest partition")
        all_rows = [row_id for members in normalized_partitions.values() for row_id in members]
        if len(all_rows) != len(set(all_rows)):
            raise ValueError("SplitManifest partitions must be pairwise disjoint")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")

        object.__setattr__(
            self,
            "dataset_id",
            _nonempty_identifier(self.dataset_id, field_name="dataset_id"),
        )
        object.__setattr__(
            self,
            "dataset_hash",
            require_sha256(self.dataset_hash, field_name="dataset_hash"),
        )
        object.__setattr__(
            self,
            "split_id",
            _nonempty_identifier(self.split_id, field_name="split_id"),
        )
        object.__setattr__(
            self,
            "strategy",
            _nonempty_identifier(self.strategy, field_name="strategy"),
        )
        object.__setattr__(
            self,
            "partitions",
            MappingProxyType(dict(sorted(normalized_partitions.items()))),
        )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def create(
        cls,
        dataset: RawDataset,
        partitions: Mapping[str, Iterable[str | int]],
        *,
        split_id: str = "default",
        fit_partition: str = "train",
        strategy: str = "explicit",
        seed: int | None = None,
        require_complete: bool = True,
        metadata: Mapping[str, Any] | None = None,
    ) -> SplitManifest:
        normalized: dict[str, tuple[str, ...]] = {}
        for name, members in partitions.items():
            row_ids: list[str] = []
            for member in members:
                if isinstance(member, int):
                    if member < 0:
                        raise ValueError(f"row index {member} is outside the dataset")
                    try:
                        row_ids.append(dataset.row_ids[member])
                    except IndexError as exc:
                        raise ValueError(f"row index {member} is outside the dataset") from exc
                else:
                    row_ids.append(member)
            normalized[name] = tuple(row_ids)
        manifest = cls(
            dataset_id=dataset.dataset_id,
            dataset_hash=dataset.dataset_hash,
            partitions=normalized,
            split_id=split_id,
            fit_partition=fit_partition,
            strategy=strategy,
            seed=seed,
            require_complete=require_complete,
            metadata=dict(metadata or {}),
        )
        manifest.validate_dataset(dataset)
        return manifest

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.split-manifest.v1",
                "split_id": self.split_id,
                "dataset_id": self.dataset_id,
                "dataset_hash": self.dataset_hash,
                "partitions": self.partitions,
                "fit_partition": self.fit_partition,
                "strategy": self.strategy,
                "seed": self.seed,
                "require_complete": self.require_complete,
                "metadata": self.metadata,
            }
        )

    @property
    def split_hash(self) -> str:
        return self.manifest_hash

    @property
    def definition_hash(self) -> str:
        """Split topology identity, intentionally independent of cell values."""

        return canonical_hash(
            {
                "schema": "tabu.split-definition.v1",
                "split_id": self.split_id,
                "dataset_id": self.dataset_id,
                "partitions": self.partitions,
                "fit_partition": self.fit_partition,
                "strategy": self.strategy,
                "seed": self.seed,
                "require_complete": self.require_complete,
            }
        )

    def validate_dataset(self, dataset: RawDataset) -> None:
        if dataset.dataset_id != self.dataset_id or dataset.dataset_hash != self.dataset_hash:
            raise ValueError("SplitManifest is not bound to this RawDataset content")
        dataset_rows = set(dataset.row_ids)
        manifest_rows = {
            row_id for members in self.partitions.values() for row_id in members
        }
        unknown = manifest_rows - dataset_rows
        if unknown:
            raise ValueError(f"SplitManifest contains unknown row ids: {sorted(unknown)}")
        if self.require_complete and manifest_rows != dataset_rows:
            missing = sorted(dataset_rows - manifest_rows)
            raise ValueError(f"complete SplitManifest omits row ids: {missing}")


@dataclass(frozen=True, slots=True)
class SplitView:
    """A validated partition view; the only source type accepted by the compiler."""

    dataset: RawDataset
    manifest: SplitManifest
    partition: str

    def __post_init__(self) -> None:
        self.assert_bound()

    def assert_bound(self) -> None:
        self.manifest.validate_dataset(self.dataset)
        if self.partition not in self.manifest.partitions:
            raise ValueError(f"unknown split partition: {self.partition!r}")

    @property
    def row_ids(self) -> tuple[str, ...]:
        return self.manifest.partitions[self.partition]

    @property
    def row_indices(self) -> tuple[int, ...]:
        index = {row_id: position for position, row_id in enumerate(self.dataset.row_ids)}
        return tuple(index[row_id] for row_id in self.row_ids)

    @property
    def values(self) -> torch.Tensor:
        return self.dataset.values[list(self.row_indices)].clone()

    @property
    def origin_states(self) -> torch.Tensor:
        return self.dataset.origin_states[list(self.row_indices)].clone()

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.dataset.feature_names

    @property
    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        return self.dataset.feature_specs

    @property
    def graph_topology(self) -> GraphTopology | None:
        topology = self.dataset.graph_topology
        return topology.induced(self.row_ids) if topology is not None else None

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.row_ids), len(self.feature_names)

    @property
    def view_hash(self) -> str:
        self.assert_bound()
        return canonical_hash(
            {
                "schema": "tabu.split-view.v3",
                "dataset_id": self.dataset.dataset_id,
                "split_definition_hash": self.manifest.definition_hash,
                "partition": self.partition,
                "row_ids": self.row_ids,
                "feature_specs": self.feature_specs,
                "graph_topology": self.graph_topology,
                "values": self.values,
                "origin_states": self.origin_states,
            }
        )


@dataclass(frozen=True, slots=True)
class EpisodeRecipe:
    """Truth-free episode intent bound to source and fit partitions."""

    recipe_id: str
    split_manifest_hash: str
    source_partition: str
    source_view_hash: str
    fit_partition: str
    fit_view_hash: str
    origin_states: torch.Tensor | Sequence[Sequence[OriginState | str | int]]
    forward_roles: torch.Tensor | Sequence[Sequence[ForwardRole | str | int]]
    graph_topology: GraphTopology | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        origins = encode_origin_states(self.origin_states)
        roles = encode_forward_roles(self.forward_roles)
        if tuple(origins.shape) != tuple(roles.shape):
            raise ValueError("EpisodeRecipe origin_states and forward_roles must match")
        if not bool(forward_role_mask(roles, ForwardRole.TARGET).any()):
            raise ValueError("EpisodeRecipe needs at least one TARGET cell")
        if self.graph_topology is not None:
            if not isinstance(self.graph_topology, GraphTopology):
                raise TypeError("EpisodeRecipe.graph_topology must be GraphTopology or None")
            if len(self.graph_topology.node_ids) != origins.shape[0]:
                raise ValueError("EpisodeRecipe.graph_topology must match the row dimension")
        object.__setattr__(
            self,
            "recipe_id",
            _nonempty_identifier(self.recipe_id, field_name="recipe_id"),
        )
        object.__setattr__(
            self,
            "split_manifest_hash",
            require_sha256(self.split_manifest_hash, field_name="split_manifest_hash"),
        )
        object.__setattr__(
            self,
            "source_view_hash",
            require_sha256(self.source_view_hash, field_name="source_view_hash"),
        )
        object.__setattr__(
            self,
            "fit_view_hash",
            require_sha256(self.fit_view_hash, field_name="fit_view_hash"),
        )
        object.__setattr__(
            self,
            "source_partition",
            _nonempty_identifier(self.source_partition, field_name="source_partition"),
        )
        object.__setattr__(
            self,
            "fit_partition",
            _nonempty_identifier(self.fit_partition, field_name="fit_partition"),
        )
        object.__setattr__(self, "origin_states", origins)
        object.__setattr__(self, "forward_roles", roles)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @classmethod
    def create(
        cls,
        source_view: SplitView,
        fit_view: SplitView,
        forward_roles: torch.Tensor | Sequence[Sequence[ForwardRole | str | int]],
        *,
        origin_states: torch.Tensor | Sequence[Sequence[OriginState | str | int]] | None = None,
        target_origin: OriginState = OriginState.ARTIFICIAL_MASK,
        graph_topology: GraphTopology | None = None,
        recipe_id: str = "episode",
        metadata: Mapping[str, Any] | None = None,
    ) -> EpisodeRecipe:
        source_view.assert_bound()
        fit_view.assert_bound()
        if source_view.manifest.manifest_hash != fit_view.manifest.manifest_hash:
            raise ValueError("source and fit views must come from the same SplitManifest")
        if fit_view.partition != source_view.manifest.fit_partition:
            raise ValueError("fit_view must be the manifest's declared fit partition")
        roles = encode_forward_roles(forward_roles)
        if tuple(roles.shape) != source_view.shape:
            raise ValueError("forward_roles must match the source SplitView shape")
        if target_origin not in {
            OriginState.ARTIFICIAL_MASK,
            OriginState.QUERY,
            OriginState.NATURAL_MISSING,
        }:
            raise ValueError(
                "target_origin must be ARTIFICIAL_MASK, QUERY, or NATURAL_MISSING"
            )
        if origin_states is None:
            origins = source_view.origin_states
            target = forward_role_mask(roles, ForwardRole.TARGET)
            origins[target] = origin_code(target_origin)
        else:
            origins = encode_origin_states(origin_states)
            if tuple(origins.shape) != source_view.shape:
                raise ValueError("origin_states must match the source SplitView shape")
        source_topology = source_view.graph_topology
        if graph_topology is not None and not isinstance(graph_topology, GraphTopology):
            raise TypeError("graph_topology must be GraphTopology or None")
        if graph_topology is not None and graph_topology.node_ids != source_view.row_ids:
            raise ValueError("graph_topology node_ids must match the source SplitView row_ids")
        if (
            graph_topology is not None
            and source_topology is not None
            and graph_topology.topology_hash != source_topology.topology_hash
        ):
            raise ValueError("explicit graph_topology conflicts with RawDataset topology")
        resolved_topology = graph_topology or source_topology
        return cls(
            recipe_id=recipe_id,
            split_manifest_hash=source_view.manifest.manifest_hash,
            source_partition=source_view.partition,
            source_view_hash=source_view.view_hash,
            fit_partition=fit_view.partition,
            fit_view_hash=fit_view.view_hash,
            origin_states=origins,
            forward_roles=roles,
            graph_topology=resolved_topology,
            metadata=dict(metadata or {}),
        )

    from_roles = create

    @property
    def recipe_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.episode-recipe.v2",
                "recipe_id": self.recipe_id,
                "split_manifest_hash": self.split_manifest_hash,
                "source_partition": self.source_partition,
                "source_view_hash": self.source_view_hash,
                "fit_partition": self.fit_partition,
                "fit_view_hash": self.fit_view_hash,
                "origin_states": self.origin_states,
                "forward_roles": self.forward_roles,
                "graph_topology": self.graph_topology,
                "metadata": self.metadata,
            }
        )

    def validate_binding(self, source_view: SplitView, fit_view: SplitView) -> None:
        source_view.assert_bound()
        fit_view.assert_bound()
        expected = {
            "split_manifest_hash": source_view.manifest.manifest_hash,
            "source_partition": source_view.partition,
            "source_view_hash": source_view.view_hash,
            "fit_partition": fit_view.partition,
            "fit_view_hash": fit_view.view_hash,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError("EpisodeRecipe is not bound to the supplied source/fit views")
        if fit_view.partition != source_view.manifest.fit_partition:
            raise ValueError("fit_view must be the manifest's declared fit partition")
        if source_view.manifest.manifest_hash != fit_view.manifest.manifest_hash:
            raise ValueError("source and fit views must share one SplitManifest")
        if tuple(self.forward_roles.shape) != source_view.shape or tuple(
            self.origin_states.shape
        ) != source_view.shape:
            raise ValueError("EpisodeRecipe state grids no longer match the source view")
        if self.graph_topology is not None and self.graph_topology.node_ids != source_view.row_ids:
            raise ValueError("EpisodeRecipe graph topology no longer matches the source view")


__all__ = ["EpisodeRecipe", "RawDataset", "SplitManifest", "SplitView"]
