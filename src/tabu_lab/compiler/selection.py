"""Fit-bound explicit feature selection manifests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from tabu_lab.contracts import FeatureSpec, SplitView, canonical_hash, require_sha256

from .episode import FitPartitionBindingError
from .statistics import _require_fit_view, _require_same_fit_binding


@dataclass(frozen=True, slots=True)
class SelectedFeatureView:
    """Truth-free projection produced by a validated selection manifest."""

    row_ids: tuple[str, ...]
    feature_specs: tuple[FeatureSpec, ...]
    values: torch.Tensor
    origin_states: torch.Tensor

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.values).detach().clone().cpu()
        origins = torch.as_tensor(self.origin_states).detach().clone().to(torch.uint8).cpu()
        expected = (len(self.row_ids), len(self.feature_specs))
        if tuple(values.shape) != expected or tuple(origins.shape) != expected:
            raise ValueError("selected values/origins must match row and feature schema")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin_states", origins)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.feature_specs)


@dataclass(frozen=True, slots=True)
class FeatureSelectionManifest:
    """An explicit ordered selection bound to one fit-partition snapshot."""

    selected_indices: tuple[int, ...]
    selected_feature_specs: tuple[FeatureSpec, ...]
    fit_view_hash: str
    split_definition_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        indices = tuple(self.selected_indices)
        specs = tuple(self.selected_feature_specs)
        if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
            raise ValueError("selected feature indices must be non-empty, unique, and non-negative")
        if len(indices) != len(specs):
            raise ValueError("selected indices and feature specs must align")
        for field_name in ("fit_view_hash", "split_definition_hash", "config_hash"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_feature_specs", specs)
        expected_config_hash = canonical_hash(
            {
                "kind": "explicit_feature_selection",
                "selected_indices": indices,
                "selected_feature_specs": specs,
            }
        )
        if self.config_hash != expected_config_hash:
            raise ValueError("feature selection config_hash does not match its selection")

    @classmethod
    def fit(
        cls,
        view: SplitView,
        selected: Sequence[int | str],
    ) -> FeatureSelectionManifest:
        _require_fit_view(view)
        indices: list[int] = []
        for item in selected:
            if isinstance(item, str):
                try:
                    index = view.feature_names.index(item)
                except ValueError as exc:
                    raise ValueError(f"unknown selected feature: {item!r}") from exc
            else:
                index = int(item)
                if index < 0 or index >= len(view.feature_names):
                    raise ValueError("selected feature index is outside the table")
            indices.append(index)
        selected_indices = tuple(indices)
        if not selected_indices or len(selected_indices) != len(set(selected_indices)):
            raise ValueError("selected features must be non-empty and unique")
        specs = tuple(view.feature_specs[index] for index in selected_indices)
        config_hash = canonical_hash(
            {
                "kind": "explicit_feature_selection",
                "selected_indices": selected_indices,
                "selected_feature_specs": specs,
            }
        )
        return cls(
            selected_indices=selected_indices,
            selected_feature_specs=specs,
            fit_view_hash=view.view_hash,
            split_definition_hash=view.manifest.definition_hash,
            config_hash=config_hash,
        )

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.selected_feature_specs)

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.feature-selection-manifest.v1",
                "selected_indices": self.selected_indices,
                "selected_feature_specs": self.selected_feature_specs,
                "fit_view_hash": self.fit_view_hash,
                "split_definition_hash": self.split_definition_hash,
                "config_hash": self.config_hash,
            }
        )

    def apply(self, view: SplitView) -> SelectedFeatureView:
        _require_same_fit_binding(
            view,
            split_definition_hash=self.split_definition_hash,
            fit_view_hash=self.fit_view_hash,
        )
        if any(index >= len(view.feature_specs) for index in self.selected_indices):
            raise FitPartitionBindingError("selected feature index is outside transform view")
        actual_specs = tuple(view.feature_specs[index] for index in self.selected_indices)
        if actual_specs != self.selected_feature_specs:
            raise FitPartitionBindingError("selected feature schema differs from fit")
        index = list(self.selected_indices)
        return SelectedFeatureView(
            row_ids=view.row_ids,
            feature_specs=actual_specs,
            values=view.values[:, index],
            origin_states=view.origin_states[:, index],
        )

    def transform(self, view: SplitView) -> torch.Tensor:
        return self.apply(view).values


__all__ = ["FeatureSelectionManifest", "SelectedFeatureView"]
