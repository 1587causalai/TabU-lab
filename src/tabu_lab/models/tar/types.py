"""Truth-free typed episodes. Discrete values are indices in declared domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class TARFeature:
    kind: Literal["numeric", "nominal", "ordinal"] = "numeric"
    domain: tuple[str, ...] = ()
    column_id: int = 0

    def __post_init__(self):
        if self.kind not in ("numeric", "nominal", "ordinal"):
            raise ValueError("invalid-input: unknown feature type")
        if type(self.column_id) is not int or self.column_id < 0:
            raise ValueError("column_id must be a stable nonnegative integer")
        if self.kind != "numeric" and (
            not self.domain or len(set(self.domain)) != len(self.domain)
        ):
            raise ValueError("discrete features require a nonempty unique ordered domain")


@dataclass(frozen=True)
class TAREpisode:
    values: Tensor  # [N,M], physical zeros outside visible
    visible: Tensor  # bool [N,M]
    queries: Tensor  # bool [N,M]
    features: tuple[TARFeature, ...]
    codebook_seed: int = 0
    codebooks: dict[int, Tensor] = field(default_factory=dict)  # visible class -> row mapping below
    codebook_classes: dict[int, tuple[int, ...]] = field(default_factory=dict)

    @classmethod
    def from_table(cls, values, visible, queries, features, **kwargs):
        """Explicit ingestion boundary: erase hidden values before constructing input."""
        return cls(
            torch.where(visible, values, torch.zeros_like(values)),
            visible.clone(),
            queries.clone(),
            tuple(features),
            **kwargs,
        )

    def validate(self):
        if (
            self.values.ndim != 2
            or min(self.values.shape) < 1
            or not self.values.is_floating_point()
        ):
            raise ValueError("invalid-input: values must be a nonempty floating matrix")
        for mask in (self.visible, self.queries):
            if mask.shape != self.values.shape or mask.dtype != torch.bool:
                raise ValueError("invalid-input: boolean masks must match values")
            if mask.device != self.values.device:
                raise ValueError("invalid-input: masks and values must share device")
        if bool((self.visible & self.queries).any()):
            raise ValueError("invalid-input: visible and query sets overlap")
        if bool((self.values[~self.visible] != 0).any()):
            raise ValueError("invalid-input: hidden values must be physically erased")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("invalid-input: nonfinite visible values")
        if len(self.features) != self.values.shape[1]:
            raise ValueError("invalid-input: schema width mismatch")
        ids = [f.column_id for f in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("invalid-input: column IDs must be unique")
        if type(self.codebook_seed) is not int or self.codebook_seed < 0:
            raise ValueError("invalid-input: codebook_seed must be nonnegative")
        for a, f in enumerate(self.features):
            x = self.values[self.visible[:, a], a]
            if f.kind != "numeric" and bool(
                ((x != x.round()) | (x < 0) | (x >= len(f.domain))).any()
            ):
                raise ValueError("invalid-input: discrete values must be declared domain indices")


@dataclass
class TARPrediction:
    address: tuple[int, int]
    status: str
    support_count: int
    value: Tensor | None = None
    probabilities: Tensor | None = None
    scale: Tensor | None = None
    weights: Tensor | None = None


@dataclass
class TAROutput:
    predictions: tuple[TARPrediction, ...]
    carriers: Tensor
    responses: Tensor
    codebooks: dict[int, Tensor]
    codebook_classes: dict[int, tuple[int, ...]]
    model_id: str = "tabu.tar"
