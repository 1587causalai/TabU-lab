"""Fit-bound categorical codebooks with an explicit OOV code."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tabu_lab.contracts import SplitView, canonical_hash, origin_value_mask, require_sha256

from .episode import FitPartitionBindingError
from .statistics import _require_fit_view, _require_same_fit_binding


@dataclass(frozen=True, slots=True)
class CategoricalCodebook:
    feature_index: int
    feature_name: str
    vocabulary: tuple[float, ...]
    oov_code: int
    fit_view_hash: str
    split_definition_hash: str
    config_hash: str

    def __post_init__(self) -> None:
        if self.feature_index < 0 or not self.feature_name.strip():
            raise ValueError("categorical feature identity is invalid")
        vocabulary = tuple(float(value) for value in self.vocabulary)
        if tuple(sorted(set(vocabulary))) != vocabulary:
            raise ValueError("categorical vocabulary must be sorted and unique")
        if self.oov_code != len(vocabulary):
            raise ValueError("oov_code must be the first code after the fitted vocabulary")
        object.__setattr__(self, "vocabulary", vocabulary)
        for field_name in ("fit_view_hash", "split_definition_hash", "config_hash"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field_name=field_name),
            )

    @classmethod
    def fit(
        cls,
        view: SplitView,
        feature: int | str,
    ) -> CategoricalCodebook:
        _require_fit_view(view)
        if isinstance(feature, str):
            try:
                feature_index = view.feature_names.index(feature)
            except ValueError as exc:
                raise ValueError(f"unknown categorical feature: {feature!r}") from exc
        else:
            feature_index = int(feature)
            if feature_index < 0 or feature_index >= len(view.feature_names):
                raise ValueError("categorical feature index is outside the table")
        feature_name = view.feature_names[feature_index]
        visible = origin_value_mask(view.origin_states)[:, feature_index]
        values = view.values[:, feature_index][visible]
        vocabulary = tuple(sorted(float(value) for value in torch.unique(values).tolist()))
        config_hash = canonical_hash(
            {
                "kind": "categorical_codebook",
                "feature_index": feature_index,
                "feature_name": feature_name,
                "oov_policy": "explicit_last_code",
            }
        )
        return cls(
            feature_index=feature_index,
            feature_name=feature_name,
            vocabulary=vocabulary,
            oov_code=len(vocabulary),
            fit_view_hash=view.view_hash,
            split_definition_hash=view.manifest.definition_hash,
            config_hash=config_hash,
        )

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.categorical-codebook.v1",
                "feature_index": self.feature_index,
                "feature_name": self.feature_name,
                "vocabulary": self.vocabulary,
                "oov_code": self.oov_code,
                "fit_view_hash": self.fit_view_hash,
                "split_definition_hash": self.split_definition_hash,
                "config_hash": self.config_hash,
            }
        )

    def transform(self, view: SplitView) -> torch.Tensor:
        _require_same_fit_binding(
            view,
            split_definition_hash=self.split_definition_hash,
            fit_view_hash=self.fit_view_hash,
        )
        if (
            self.feature_index >= len(view.feature_names)
            or view.feature_names[self.feature_index] != self.feature_name
        ):
            raise FitPartitionBindingError("categorical feature order differs from fit")
        values = view.values[:, self.feature_index]
        visible = origin_value_mask(view.origin_states)[:, self.feature_index]
        encoded = torch.full((values.shape[0],), self.oov_code, dtype=torch.int64)
        for code, category in enumerate(self.vocabulary):
            encoded[visible & (values == category)] = code
        return encoded


__all__ = ["CategoricalCodebook"]
