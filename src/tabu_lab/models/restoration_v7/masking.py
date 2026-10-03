"""Deterministic episode-level single/joint/mixed masking for one shared model."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tables import TypedTable, table_mask_task


@dataclass(frozen=True)
class MaskingSpec:
    mode: str = "mixed"
    columns: tuple[int | str, ...] | None = None  # None: all columns; "target": table target
    joint_probability: float = 0.5  # fraction of episodes, not of Query cells
    joint_columns: int | None = None  # None: all eligible columns

    def __post_init__(self):
        if self.mode not in ("single", "joint", "mixed"):
            raise ValueError("masking.mode must be single, joint or mixed")
        if isinstance(self.joint_probability, bool) or not 0 <= self.joint_probability <= 1:
            raise ValueError("joint_probability must lie in [0, 1]")
        if self.joint_columns is not None and (
            type(self.joint_columns) is not int or self.joint_columns < 2
        ):
            raise ValueError("joint_columns must be at least two or null (all eligible columns)")
        if self.columns is not None:
            object.__setattr__(self, "columns", tuple(self.columns))
            if not self.columns:
                raise ValueError("masking.columns must not be empty")

    def eligible(self, table: TypedTable) -> tuple[int, ...]:
        result = []
        keys = {column.key: i for i, column in enumerate(table.schema)}
        for value in self.columns if self.columns is not None else range(len(table.schema)):
            if value == "target":
                column = table.target
            elif type(value) is str and value in keys:
                column = keys[value]
            elif type(value) is int:
                column = value
            else:
                raise ValueError(f"{table.name}: invalid masking column {value!r}")
            if not 0 <= column < len(table.schema) or column in result:
                raise ValueError(f"{table.name}: masking columns must be distinct and in range")
            result.append(column)
        needs_joint = self.mode == "joint" or (self.mode == "mixed" and self.joint_probability > 0)
        if needs_joint and len(result) < (self.joint_columns or 2):
            raise ValueError(f"{table.name}: not enough eligible columns for joint masking")
        return tuple(result)

    def select(self, table: TypedTable, rng: random.Random) -> tuple[str, tuple[int, ...]]:
        eligible = self.eligible(table)
        joint = self.mode == "joint" or (
            self.mode == "mixed" and rng.random() < self.joint_probability
        )
        count = (self.joint_columns or len(eligible)) if joint else 1
        return ("joint" if joint else "single"), tuple(sorted(rng.sample(eligible, count)))


def sample_task(
    table: TypedTable,
    spec: MaskingSpec,
    *,
    seed: int,
    step: int,
    window_rows: int,
    query_rows: int,
    device="cpu",
    dtype=None,
):
    """Sample from train rows only; addressable by (seed, step), including on resume.

    The same Query rows are hidden in every selected column. ``table_mask_task``
    also supports arbitrary per-column row sets for callers needing sparse masks.
    """
    import torch

    if type(seed) is not int or type(step) is not int or step < 0:
        raise ValueError("seed and nonnegative step must be integers")
    if type(window_rows) is not int or type(query_rows) is not int or query_rows < 1:
        raise ValueError("window_rows and positive query_rows must be integers")
    count = min(window_rows, len(table.train_rows))
    if count - query_rows < 2:
        raise ValueError(f"{table.name}: query_rows must leave at least two train supports")
    rng = random.Random(f"v7-mask-v1:{seed}:{step}")
    mode, columns = spec.select(table, rng)
    rows = rng.sample(table.train_rows, count)
    query = rng.sample(rows, query_rows)
    code_seed, donor_seed = rng.randrange(2**63), rng.randrange(2**63)
    task = table_mask_task(
        table,
        rows,
        {column: query for column in columns},
        code_seed=code_seed,
        donor_seed=donor_seed,
        device=device,
        dtype=dtype or torch.float64,
    )
    receipt = dict(
        table=table.name,
        mode=mode,
        columns=list(columns),
        rows=rows,
        query_rows=query,
        query_cells=len(columns) * len(query),
        code_seed=code_seed,
        donor_seed=donor_seed,
    )
    return task, receipt
