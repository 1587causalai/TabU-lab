"""Fit-partition-bound numeric statistics and normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tabu_lab.contracts import (
    FeatureKind,
    SplitView,
    canonical_hash,
    origin_value_mask,
    require_sha256,
)

from .episode import FitPartitionBindingError


def _require_fit_view(view: SplitView) -> None:
    if not isinstance(view, SplitView):
        raise TypeError("statistics fit requires a SplitView")
    view.assert_bound()
    if view.partition != view.manifest.fit_partition:
        raise FitPartitionBindingError(
            "statistics may only fit the SplitManifest's declared fit partition"
        )


def _require_same_split_definition(view: SplitView, expected_hash: str) -> None:
    if not isinstance(view, SplitView):
        raise TypeError("transform requires a SplitView")
    view.assert_bound()
    if view.manifest.definition_hash != expected_hash:
        raise FitPartitionBindingError(
            "transform view does not share the fitted split definition"
        )


def _require_same_fit_binding(
    view: SplitView,
    *,
    split_definition_hash: str,
    fit_view_hash: str,
) -> None:
    _require_same_split_definition(view, split_definition_hash)
    current_fit = SplitView(
        dataset=view.dataset,
        manifest=view.manifest,
        partition=view.manifest.fit_partition,
    )
    if current_fit.view_hash != fit_view_hash:
        raise FitPartitionBindingError("transform dataset does not match fitted partition content")


@dataclass(frozen=True, slots=True)
class FittedStatistics:
    """Portable numeric fit artifact with no non-fit observations."""

    fit_view_hash: str
    split_definition_hash: str
    config_hash: str
    fit_value_mask_hash: str
    feature_names: tuple[str, ...]
    feature_kinds: tuple[FeatureKind | str, ...]
    counts: torch.Tensor
    means: torch.Tensor
    scales: torch.Tensor

    def __post_init__(self) -> None:
        counts = torch.as_tensor(self.counts).detach().clone().to(torch.int64).cpu()
        means = torch.as_tensor(self.means).detach().clone().to(torch.float64).cpu()
        scales = torch.as_tensor(self.scales).detach().clone().to(torch.float64).cpu()
        kinds = tuple(FeatureKind(kind) for kind in self.feature_kinds)
        expected = (len(self.feature_names),)
        if len(kinds) != len(self.feature_names):
            raise ValueError("fitted statistics require one kind per feature")
        if tuple(counts.shape) != expected or tuple(means.shape) != expected or tuple(
            scales.shape
        ) != expected:
            raise ValueError("fitted statistics must have one value per feature")
        if bool((counts < 0).any()):
            raise ValueError("fitted statistic counts cannot be negative")
        if not bool(torch.isfinite(means).all()) or not bool(torch.isfinite(scales).all()):
            raise ValueError("fitted statistics must be finite")
        if bool((scales <= 0).any()):
            raise ValueError("fitted statistic scales must be positive")
        object.__setattr__(
            self,
            "fit_view_hash",
            require_sha256(self.fit_view_hash, field_name="fit_view_hash"),
        )
        object.__setattr__(
            self,
            "split_definition_hash",
            require_sha256(
                self.split_definition_hash,
                field_name="split_definition_hash",
            ),
        )
        object.__setattr__(
            self,
            "config_hash",
            require_sha256(self.config_hash, field_name="config_hash"),
        )
        object.__setattr__(
            self,
            "fit_value_mask_hash",
            require_sha256(
                self.fit_value_mask_hash,
                field_name="fit_value_mask_hash",
            ),
        )
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "feature_kinds", kinds)

    @property
    def artifact_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.fitted-statistics.v2",
                "fit_view_hash": self.fit_view_hash,
                "split_definition_hash": self.split_definition_hash,
                "config_hash": self.config_hash,
                "fit_value_mask_hash": self.fit_value_mask_hash,
                "feature_names": self.feature_names,
                "feature_kinds": self.feature_kinds,
                "counts": self.counts,
                "means": self.means,
                "scales": self.scales,
            }
        )


@dataclass(frozen=True, slots=True)
class NumericNormalizer:
    statistics: FittedStatistics
    epsilon: float = 1.0e-8
    shared_numeric_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("NumericNormalizer.epsilon must be positive")
        groups = tuple(tuple(group) for group in self.shared_numeric_groups)
        flattened = tuple(name for group in groups for name in group)
        if any(len(group) < 2 or len(group) != len(set(group)) for group in groups):
            raise ValueError(
                "shared numeric normalization groups need at least two unique features"
            )
        if len(flattened) != len(set(flattened)):
            raise ValueError("shared numeric normalization groups must be disjoint")
        expected = canonical_hash(
            {
                "kind": "numeric_normalizer",
                "epsilon": float(self.epsilon),
                "shared_numeric_groups": groups,
            }
        )
        if self.statistics.config_hash != expected:
            raise ValueError("normalizer configuration does not match fitted statistics")
        object.__setattr__(self, "shared_numeric_groups", groups)

    @classmethod
    def fit(
        cls,
        view: SplitView,
        *,
        epsilon: float = 1.0e-8,
        excluded_mask: torch.Tensor | None = None,
        shared_numeric_groups: tuple[tuple[str, ...], ...] = (),
    ) -> NumericNormalizer:
        _require_fit_view(view)
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        values = view.values.to(torch.float64)
        excluded = (
            torch.zeros(view.shape, dtype=torch.bool)
            if excluded_mask is None
            else torch.as_tensor(excluded_mask).detach().clone().cpu().bool()
        )
        if tuple(excluded.shape) != view.shape:
            raise ValueError("excluded_mask must match the fit SplitView shape")
        numeric_features = torch.tensor(
            tuple(spec.kind is FeatureKind.NUMERIC for spec in view.feature_specs),
            dtype=torch.bool,
        )
        visible = (
            origin_value_mask(view.origin_states)
            & ~excluded
            & numeric_features.view(1, -1)
        )
        counts = visible.sum(dim=0).to(torch.int64)
        sums = torch.where(visible, values, torch.zeros_like(values)).sum(dim=0)
        safe_counts = counts.clamp_min(1).to(torch.float64)
        means = sums / safe_counts
        centered = torch.where(visible, values - means, torch.zeros_like(values))
        variances = centered.square().sum(dim=0) / safe_counts
        scales = variances.sqrt()
        scales = torch.where((counts > 1) & (scales > epsilon), scales, torch.ones_like(scales))
        means = torch.where(counts > 0, means, torch.zeros_like(means))
        groups = tuple(tuple(group) for group in shared_numeric_groups)
        flattened = tuple(name for group in groups for name in group)
        if any(len(group) < 2 or len(group) != len(set(group)) for group in groups):
            raise ValueError(
                "shared numeric normalization groups need at least two unique features"
            )
        if len(flattened) != len(set(flattened)):
            raise ValueError("shared numeric normalization groups must be disjoint")
        feature_index = {name: index for index, name in enumerate(view.feature_names)}
        for group in groups:
            unknown = set(group) - set(feature_index)
            if unknown:
                raise ValueError(
                    "shared numeric normalization group contains unknown features: "
                    + ", ".join(sorted(unknown))
                )
            indices = tuple(feature_index[name] for name in group)
            if any(not bool(numeric_features[index]) for index in indices):
                raise ValueError("shared normalization groups may contain numeric features only")
            group_visible = visible[:, list(indices)]
            group_values = values[:, list(indices)]
            group_count = group_visible.sum().to(torch.int64)
            safe_group_count = group_count.clamp_min(1).to(torch.float64)
            group_sum = torch.where(
                group_visible,
                group_values,
                torch.zeros_like(group_values),
            ).sum()
            group_mean = group_sum / safe_group_count
            group_centered = torch.where(
                group_visible,
                group_values - group_mean,
                torch.zeros_like(group_values),
            )
            group_scale = (group_centered.square().sum() / safe_group_count).sqrt()
            group_scale = torch.where(
                (group_count > 1) & (group_scale > epsilon),
                group_scale,
                torch.ones_like(group_scale),
            )
            group_mean = torch.where(
                group_count > 0,
                group_mean,
                torch.zeros_like(group_mean),
            )
            counts[list(indices)] = group_count
            means[list(indices)] = group_mean
            scales[list(indices)] = group_scale
        config_hash = canonical_hash(
            {
                "kind": "numeric_normalizer",
                "epsilon": float(epsilon),
                "shared_numeric_groups": groups,
            }
        )
        statistics = FittedStatistics(
            fit_view_hash=view.view_hash,
            split_definition_hash=view.manifest.definition_hash,
            config_hash=config_hash,
            fit_value_mask_hash=canonical_hash(
                {
                    "schema": "tabu.numeric-fit-value-mask.v1",
                    "fit_view_hash": view.view_hash,
                    "value_mask": visible,
                }
            ),
            feature_names=view.feature_names,
            feature_kinds=tuple(spec.kind for spec in view.feature_specs),
            counts=counts,
            means=means,
            scales=scales,
        )
        return cls(
            statistics=statistics,
            epsilon=float(epsilon),
            shared_numeric_groups=groups,
        )

    @property
    def artifact_hash(self) -> str:
        return self.statistics.artifact_hash

    def require_fit_value_mask(
        self,
        view: SplitView,
        *,
        excluded_mask: torch.Tensor | None = None,
    ) -> None:
        """Require the fitted mask to match the compiler's target exclusion."""

        _require_fit_view(view)
        excluded = (
            torch.zeros(view.shape, dtype=torch.bool)
            if excluded_mask is None
            else torch.as_tensor(excluded_mask).detach().clone().cpu().bool()
        )
        if tuple(excluded.shape) != view.shape:
            raise ValueError("excluded_mask must match the fit SplitView shape")
        numeric_features = torch.tensor(
            tuple(spec.kind is FeatureKind.NUMERIC for spec in view.feature_specs),
            dtype=torch.bool,
        )
        visible = (
            origin_value_mask(view.origin_states)
            & ~excluded
            & numeric_features.view(1, -1)
        )
        expected = canonical_hash(
            {
                "schema": "tabu.numeric-fit-value-mask.v1",
                "fit_view_hash": view.view_hash,
                "value_mask": visible,
            }
        )
        if self.statistics.fit_value_mask_hash != expected:
            raise FitPartitionBindingError(
                "numeric normalizer target exclusion does not match this episode"
            )

    def transform(self, view: SplitView) -> torch.Tensor:
        _require_same_fit_binding(
            view,
            split_definition_hash=self.statistics.split_definition_hash,
            fit_view_hash=self.statistics.fit_view_hash,
        )
        if view.feature_names != self.statistics.feature_names:
            raise FitPartitionBindingError("transform feature order differs from fitted statistics")
        feature_kinds = tuple(spec.kind for spec in view.feature_specs)
        if feature_kinds != self.statistics.feature_kinds:
            raise FitPartitionBindingError("transform feature kinds differ from fitted statistics")
        values = view.values.to(torch.float64)
        visible = origin_value_mask(view.origin_states)
        normalized = (values - self.statistics.means) / self.statistics.scales
        numeric = torch.tensor(
            tuple(kind is FeatureKind.NUMERIC for kind in feature_kinds),
            dtype=torch.bool,
        ).view(1, -1)
        typed = torch.where(numeric, normalized, values)
        return torch.where(visible, typed, torch.zeros_like(typed))


__all__ = ["FittedStatistics", "NumericNormalizer"]
