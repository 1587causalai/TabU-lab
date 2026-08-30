"""Model-, loss-, and evaluation-boundary bundles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import torch

from .canonical import canonical_hash, require_sha256, to_canonical_data
from .episode import assert_truth_free


def _frozen_metadata(value: Mapping[str, Any], *, truth_free: bool) -> Mapping[str, Any]:
    payload = dict(value)
    if truth_free:
        assert_truth_free(payload, path="metadata")
    to_canonical_data(payload)
    return MappingProxyType(payload)


class PredictionKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DISTRIBUTION = "distribution"
    ABSTENTION = "abstention"


class PredictionStatus(StrEnum):
    OK = "ok"
    NO_SUPPORT = "no_support"
    UNSUPPORTED = "unsupported"
    ABSTAIN = "abstain"
    DESIGN_OPEN = "design_open"


def _empty_support_ids() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64)


def _empty_support_weights() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class PredictionEntry:
    """One typed public prediction channel and its explicit support ledger."""

    kind: PredictionKind
    status: PredictionStatus
    values: torch.Tensor | None = None
    support_ids: torch.Tensor = field(default_factory=_empty_support_ids)
    support_weights: torch.Tensor = field(default_factory=_empty_support_weights)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = PredictionKind(self.kind)
        status = PredictionStatus(self.status)
        values = None if self.values is None else torch.as_tensor(self.values)
        if status is PredictionStatus.OK and values is None:
            raise ValueError("PredictionStatus.OK requires prediction values")
        if values is not None and values.is_floating_point() and not bool(
            torch.isfinite(values).all()
        ):
            raise ValueError("prediction values must be finite")

        support_ids = torch.as_tensor(self.support_ids)
        support_weights = torch.as_tensor(self.support_weights)
        if (
            support_ids.dtype is torch.bool
            or support_ids.is_floating_point()
            or support_ids.is_complex()
        ):
            raise ValueError("PredictionEntry.support_ids must use an integer dtype")
        if not support_weights.is_floating_point() or support_weights.is_complex():
            raise ValueError("PredictionEntry.support_weights must use a real float dtype")
        if tuple(support_ids.shape) != tuple(support_weights.shape):
            raise ValueError("support_ids and support_weights must have identical shapes")
        if not bool(torch.isfinite(support_weights).all()) or bool(
            (support_weights < 0).any()
        ):
            raise ValueError("support_weights must be finite and non-negative")
        if status is PredictionStatus.NO_SUPPORT and (
            support_ids.numel() or support_weights.numel()
        ):
            raise ValueError("NO_SUPPORT predictions require empty support ids and weights")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "support_ids", support_ids)
        object.__setattr__(self, "support_weights", support_weights)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=True))

    @property
    def entry_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.prediction-entry.v1",
                "kind": self.kind,
                "status": self.status,
                "values": self.values,
                "support_ids": self.support_ids,
                "support_weights": self.support_weights,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class TraceEvent:
    name: str
    component: str
    duration_ns: int | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.component.strip():
            raise ValueError("TraceEvent name and component cannot be empty")
        if self.duration_ns is not None and self.duration_ns < 0:
            raise ValueError("TraceEvent.duration_ns cannot be negative")
        if self.input_hash is not None:
            object.__setattr__(
                self,
                "input_hash",
                require_sha256(self.input_hash, field_name="input_hash"),
            )
        if self.output_hash is not None:
            object.__setattr__(
                self,
                "output_hash",
                require_sha256(self.output_hash, field_name="output_hash"),
            )
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=True))


@dataclass(frozen=True, slots=True)
class ForwardTrace:
    """Truth-free provenance emitted by a model adapter."""

    trace_id: str
    episode_id: str
    model_id: str
    input_hash: str
    model_hash: str | None = None
    events: tuple[TraceEvent, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trace_id.strip() or not self.episode_id.strip() or not self.model_id.strip():
            raise ValueError("ForwardTrace identifiers cannot be empty")
        object.__setattr__(
            self,
            "input_hash",
            require_sha256(self.input_hash, field_name="input_hash"),
        )
        if self.model_hash is not None:
            object.__setattr__(
                self,
                "model_hash",
                require_sha256(self.model_hash, field_name="model_hash"),
            )
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=True))

    @property
    def trace_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.forward-trace.v1",
                "trace_id": self.trace_id,
                "episode_id": self.episode_id,
                "model_id": self.model_id,
                "model_hash": self.model_hash,
                "input_hash": self.input_hash,
                "events": self.events,
                "metadata": self.metadata,
            }
        )


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    """Named model outputs with no loss-side truth attached."""

    episode_id: str
    model_id: str
    entries: Mapping[str, PredictionEntry]
    contract_version: str = "prediction.v1"
    auxiliaries: Mapping[str, torch.Tensor] = field(default_factory=dict)
    trace: ForwardTrace | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id.strip() or not self.model_id.strip():
            raise ValueError("PredictionBundle identifiers cannot be empty")
        if not self.contract_version.strip():
            raise ValueError("PredictionBundle.contract_version cannot be empty")
        if not self.entries:
            raise ValueError("PredictionBundle.entries cannot be empty")
        entries: dict[str, PredictionEntry] = {}
        for name, entry in self.entries.items():
            if not name.strip():
                raise ValueError("prediction entry names cannot be empty")
            if name.strip().lower() in {"truth", "target_values", "y_true", "supervision"}:
                raise ValueError(f"truth-bearing prediction entry name is forbidden: {name!r}")
            if not isinstance(entry, PredictionEntry):
                raise TypeError("PredictionBundle.entries values must be PredictionEntry")
            entries[name] = entry
        auxiliaries: dict[str, torch.Tensor] = {}
        for name, tensor in self.auxiliaries.items():
            if not name.strip() or name in entries:
                raise ValueError("auxiliary names must be non-empty and distinct from entries")
            value = torch.as_tensor(tensor)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"prediction auxiliary {name!r} must be finite")
            auxiliaries[name] = value
        if self.trace is not None and (
            self.trace.episode_id != self.episode_id or self.trace.model_id != self.model_id
        ):
            raise ValueError("PredictionBundle trace identifiers do not match")
        object.__setattr__(self, "entries", MappingProxyType(dict(sorted(entries.items()))))
        object.__setattr__(
            self,
            "auxiliaries",
            MappingProxyType(dict(sorted(auxiliaries.items()))),
        )
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=True))

    @property
    def outputs(self) -> Mapping[str, torch.Tensor]:
        """Compatibility projection for objectives and evaluators."""

        values = dict(self.auxiliaries)
        for name, entry in self.entries.items():
            if entry.values is not None:
                values[name] = entry.values
        if len(self.entries) == 1:
            entry = next(iter(self.entries.values()))
            values.setdefault("support_ids", entry.support_ids)
            values.setdefault("support_weights", entry.support_weights)
        return MappingProxyType(values)

    @property
    def prediction_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.prediction-bundle.v1",
                "episode_id": self.episode_id,
                "model_id": self.model_id,
                "contract_version": self.contract_version,
                "entries": self.entries,
                "auxiliaries": self.auxiliaries,
                "trace_hash": None if self.trace is None else self.trace.trace_hash,
                "metadata": self.metadata,
            }
        )

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.outputs[name]


@dataclass(frozen=True, slots=True)
class LossBundle:
    """Differentiable aggregate loss without retaining a TruthSidecar."""

    episode_id: str
    total: torch.Tensor
    components: Mapping[str, torch.Tensor] = field(default_factory=dict)
    counts: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        total = torch.as_tensor(self.total)
        if (
            total.numel() != 1
            or not total.is_floating_point()
            or not bool(torch.isfinite(total).all())
        ):
            raise ValueError("LossBundle.total must be one finite floating scalar tensor")
        components: dict[str, torch.Tensor] = {}
        for name, component in self.components.items():
            value = torch.as_tensor(component)
            if (
                value.numel() != 1
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"loss component {name!r} must be one finite floating scalar"
                )
            components[name] = value
        counts = dict(self.counts)
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise ValueError("LossBundle counts must be non-negative integers")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "components", MappingProxyType(dict(sorted(components.items()))))
        object.__setattr__(self, "counts", MappingProxyType(dict(sorted(counts.items()))))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=False))


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    """Receipt-ready aggregate metrics, never a claim by itself."""

    evaluation_id: str
    episode_ids: tuple[str, ...]
    metrics: Mapping[str, float | None]
    counts: Mapping[str, int] = field(default_factory=dict)
    excluded: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evaluation_id.strip() or not self.episode_ids:
            raise ValueError("EvaluationBundle needs an id and at least one episode")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("EvaluationBundle episode_ids must be unique")
        metrics = dict(self.metrics)
        if not metrics:
            raise ValueError("EvaluationBundle.metrics cannot be empty")
        for name, value in metrics.items():
            if not name.strip() or (value is not None and not math.isfinite(float(value))):
                raise ValueError(f"metric {name!r} must have a finite value or None")
        counts = dict(self.counts)
        excluded = dict(self.excluded)
        all_counts = (*counts.values(), *excluded.values())
        if any(type(value) is not int or value < 0 for value in all_counts):
            raise ValueError("evaluation counts must be non-negative integers")
        object.__setattr__(self, "episode_ids", tuple(self.episode_ids))
        object.__setattr__(self, "metrics", MappingProxyType(dict(sorted(metrics.items()))))
        object.__setattr__(self, "counts", MappingProxyType(dict(sorted(counts.items()))))
        object.__setattr__(self, "excluded", MappingProxyType(dict(sorted(excluded.items()))))
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, truth_free=False))

    @property
    def evaluation_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.evaluation-bundle.v1",
                "evaluation_id": self.evaluation_id,
                "episode_ids": self.episode_ids,
                "metrics": self.metrics,
                "counts": self.counts,
                "excluded": self.excluded,
                "metadata": self.metadata,
            }
        )


__all__ = [
    "EvaluationBundle",
    "ForwardTrace",
    "LossBundle",
    "PredictionBundle",
    "PredictionEntry",
    "PredictionKind",
    "PredictionStatus",
    "TraceEvent",
]
