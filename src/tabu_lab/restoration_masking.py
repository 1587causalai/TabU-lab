"""Table-wide Query sampling with visible support for every observed class.

Class support is selected first, uniformly within each observed class.  Given
those protected cells, a candidate is a uniform fixed-size subset of all
remaining cells.  Whole-candidate rejection enforces the two-visible-cells
minimum without introducing a per-column Query quota or a biased local repair.
This conditional uniformity does not mean uniformity over masks before the
class-support protection: rare classes intentionally have less Query exposure.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tabu_lab.restoration_joint_fit import TablePlan


def global_query_mask(
    table: TablePlan, fraction: float, seed: int,
) -> tuple[torch.Tensor, dict]:
    """Sample an exact global Query budget, retaining all observed classes.

    The budget is ``max(1, floor(rows * columns * fraction + 0.5))``.  Columns
    may receive zero Query cells.  Invalid or unsupported budgets raise rather
    than silently reducing the requested count.  Rejection is bounded so even
    pathological high fractions cannot hang an experiment.
    """
    if (isinstance(fraction, bool) or not isinstance(fraction, int | float)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("query_fraction must be a finite number strictly between 0 and 1")
    n, width = table.train_rows, len(table.schema)
    if type(n) is not int or n < 3 or width < 1:
        raise ValueError("global query masking requires at least three rows and one column")
    if len(table.values) != width or any(column.ndim != 1 or len(column) != n
                                         for column in table.values):
        raise ValueError("table values must match the declared rows and columns")

    count = max(1, math.floor(n * width * fraction + 0.5))
    rng = random.Random(seed)
    eligible = []
    eligible_counts = []
    singleton_per_column = []
    protected = 0
    for column, spec in enumerate(table.schema):
        keep = set()
        singleton_count = 0
        if spec.kind != "numeric":
            rows_by_class = {}
            for row, label in enumerate(table.values[column].tolist()):
                rows_by_class.setdefault(int(label), []).append(row)
            keep = {rng.choice(rows) for rows in rows_by_class.values()}
            protected += len(keep)
            singleton_count = sum(len(rows) == 1 for rows in rows_by_class.values())
        rows = [row for row in range(n) if row not in keep]
        eligible.extend(row * width + column for row in rows)
        eligible_counts.append(len(rows))
        singleton_per_column.append(singleton_count)

    capacities = [min(available, n - 2) for available in eligible_counts]
    if count > sum(capacities):
        raise ValueError(
            f"{table.name}: global query budget {count} exceeds safely maskable "
            f"capacity {sum(capacities)}"
        )
    for attempt in range(1024):
        sampling_attempts = attempt + 1
        sampled = rng.sample(eligible, count)
        per_column = [0] * width
        for cell in sampled:
            per_column[cell % width] += 1
        if all(value <= n - 2 for value in per_column):
            break
    else:
        raise ValueError(
            f"{table.name}: could not sample the requested global query budget "
            "within 1024 attempts while retaining two visible cells per column"
        )

    query = torch.zeros(n * width, dtype=torch.bool)
    query[sampled] = True
    query = query.reshape(n, width)
    singleton_classes = sum(singleton_per_column)
    return query, dict(
        query_count=count,
        query_fraction=float(fraction),
        query_per_column=per_column,
        eligible_per_column=eligible_counts,
        capacity_per_column=capacities,
        protected_discrete_cells=protected,
        unmaskable_discrete_classes=singleton_classes,
        singleton_discrete_classes=singleton_classes,
        singleton_classes_per_column=singleton_per_column,
        query_coverage=count / (n * width),
        sampling_attempts=sampling_attempts,
    )
