"""Fit-partition-bound, kind-aware missing-value imputation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tabu_lab.contracts import (
    FeatureKind,
    SplitView,
    canonical_hash,
    origin_mask,
    origin_value_mask,
    require_sha256,
)
from tabu_lab.contracts.roles import OriginState

from .episode import FitPartitionBindingError
from .statistics import _require_fit_view, _require_same_fit_binding


@dataclass(frozen=True, slots=True)
class FittedImputation:
    """Per-feature fill values learned exclusively from the fit partition."""

    fit_view_hash: str
    split_definition_hash: str
    config_hash: str
    feature_names: tuple[str, ...]
    counts: torch.Tensor
    fill_values: torch.Tensor

    def __post_init__(self) -> None:
        counts = torch.as_tensor(self.counts).detach().clone().to(torch.int64).cpu()
        fill_values = (
            torch.as_tensor(self.fill_values).detach().clone().to(torch.float64).cpu()
        )
        expected = (len(self.feature_names),)
        if tuple(counts.shape) != expected or tuple(fill_values.shape) != expected:
            raise ValueError("fitted imputation needs one count and fill value per feature")
        if bool((counts < 0).any()) or not bool(torch.isfinite(fill_values).all()):
            raise ValueError("fitted imputation values must be finite with non-negative counts")
        for field_name in ("fit_view_hash", "split_definition_hash", "config_hash"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "fill_values", fill_values)

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.fitted-imputation.v1",
                "fit_view_hash": self.fit_view_hash,
                "split_definition_hash": self.split_definition_hash,
                "config_hash": self.config_hash,
                "feature_names": self.feature_names,
                "counts": self.counts,
                "fill_values": self.fill_values,
            }
        )


@dataclass(frozen=True, slots=True)
class Imputer:
    """Kind-aware mean/mode imputer with an immutable fit binding."""

    fitted: FittedImputation
    strategy: str = "kind_aware_mean_mode"

    def __post_init__(self) -> None:
        if self.strategy != "kind_aware_mean_mode":
            raise ValueError("unsupported imputation strategy")
        expected = canonical_hash({"kind": "imputer", "strategy": self.strategy})
        if self.fitted.config_hash != expected:
            raise ValueError("imputer configuration does not match fitted artifact")

    @classmethod
    def fit(
        cls,
        view: SplitView,
        *,
        strategy: str = "kind_aware_mean_mode",
    ) -> Imputer:
        _require_fit_view(view)
        if strategy != "kind_aware_mean_mode":
            raise ValueError("unsupported imputation strategy")
        values = view.values.to(torch.float64)
        visible = origin_value_mask(view.origin_states)
        counts = visible.sum(dim=0).to(torch.int64)
        fill_values = torch.zeros(values.shape[1], dtype=torch.float64)
        for feature_index, spec in enumerate(view.feature_specs):
            observed = values[:, feature_index][visible[:, feature_index]]
            if not observed.numel():
                continue
            if spec.kind is FeatureKind.NUMERIC:
                fill_values[feature_index] = observed.mean()
            else:
                categories, category_counts = torch.unique(
                    observed,
                    sorted=True,
                    return_counts=True,
                )
                fill_values[feature_index] = categories[category_counts.argmax()]
        config_hash = canonical_hash({"kind": "imputer", "strategy": strategy})
        fitted = FittedImputation(
            fit_view_hash=view.view_hash,
            split_definition_hash=view.manifest.definition_hash,
            config_hash=config_hash,
            feature_names=view.feature_names,
            counts=counts,
            fill_values=fill_values,
        )
        return cls(fitted=fitted, strategy=strategy)

    @property
    def artifact_hash(self) -> str:
        return self.fitted.artifact_hash

    def transform(self, view: SplitView) -> torch.Tensor:
        _require_same_fit_binding(
            view,
            split_definition_hash=self.fitted.split_definition_hash,
            fit_view_hash=self.fitted.fit_view_hash,
        )
        if view.feature_names != self.fitted.feature_names:
            raise FitPartitionBindingError("imputation feature order differs from fit")
        values = view.values.to(torch.float64)
        natural_missing = origin_mask(view.origin_states, OriginState.NATURAL_MISSING)
        fills = self.fitted.fill_values.unsqueeze(0).expand_as(values)
        return torch.where(natural_missing, fills, values)


__all__ = ["FittedImputation", "Imputer"]
