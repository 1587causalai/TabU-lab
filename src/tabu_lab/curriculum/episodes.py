"""A small consumer interface; truth is a separate result, never forward data."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol

from .catalog import BoundTable


@dataclass(frozen=True)
class EpisodeSeeds:
    masks: int
    codes: int
    windows: int
    evaluation: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in asdict(self).values()):
            raise ValueError("episode seeds must be nonnegative integers")


@dataclass(frozen=True)
class EpisodeRequest:
    """Unencoded task request; no model, codec, optimizer or labels are stored.

    Recipe names are extensible. Each factory validates its supported recipes;
    subclasses may carry additional frozen, serializable recipe parameters.
    Model/order seeds belong to execution, separate from the four episode streams.
    """

    seeds: EpisodeSeeds
    index: int = 0
    kind: str = "random_cell"
    fraction: float | None = 0.25
    partition: str = "train"
    purpose: str = "training"
    target_column: int | None = None
    window_rows: int | None = None

    def __post_init__(self):
        if not isinstance(self.seeds, EpisodeSeeds):
            raise ValueError("seeds must be EpisodeSeeds")
        if type(self.index) is not int or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("episode kind must be a nonempty recipe identifier")
        if self.fraction is not None and (
            type(self.fraction) not in (int, float) or not math.isfinite(self.fraction)
            or not 0 < self.fraction < 1
        ):
            raise ValueError("fraction must be finite and between zero and one")
        purposes = {"train": {"training", "fit", "retention"},
                    "validation": {"validation"}, "test": {"final_test", "retrospective"}}
        if self.partition not in purposes or self.purpose not in purposes[self.partition]:
            raise ValueError("partition and purpose do not permit this episode")
        if self.target_column is not None and (
            type(self.target_column) is not int or self.target_column < 0
        ):
            raise ValueError("target_column must be a nonnegative integer or None")
        if self.window_rows is not None and (
            type(self.window_rows) is not int or self.window_rows < 3
        ):
            raise ValueError("window_rows must be at least three or None")


class EpisodeFactory(Protocol):
    def __call__(self, table: BoundTable, request: EpisodeRequest) -> tuple:
        """Return (visible_inputs, target_request, scorer_truth, audit).

        Consumers pass only the first two objects to forward. Each adapter must
        qualify visible-only statistics and hidden-truth invariance itself.
        A Protocol annotation is not a process or security isolation boundary.
        """
        ...
