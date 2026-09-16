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


def numeric_tail_protection(
    table: TablePlan, guard: dict, scale_floor: float,
) -> tuple[torch.Tensor, dict]:
    """Mark numeric cells excluded from Query sampling using training rows only.

    These statistics define the corruption candidate pool, not the model's
    numeric codec. They are computed before masking and never passed to the
    model; protected cells remain visible and participate in the all-cell loss.
    The audit contains counts and declared policy, never observed values or
    fitted numeric coordinates.
    """
    if (not isinstance(guard, dict)
            or set(guard) != {"kind", "max_abs_robust_z"}
            or guard.get("kind") != "median_half_iqr"):
        raise ValueError("numeric_query_guard must declare "
                         "kind=median_half_iqr and max_abs_robust_z")
    threshold = guard["max_abs_robust_z"]
    if (isinstance(threshold, bool) or not isinstance(threshold, int | float)
            or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError("numeric_query_guard max_abs_robust_z must be finite and positive")
    if (isinstance(scale_floor, bool) or not isinstance(scale_floor, int | float)
            or not math.isfinite(scale_floor) or scale_floor <= 0):
        raise ValueError("numeric_query_guard scale_floor must be finite and positive")
    n, width = table.train_rows, len(table.schema)
    if (type(n) is not int or n < 3 or width < 1 or len(table.values) != width
            or any(value.ndim != 1 or len(value) != n for value in table.values)):
        raise ValueError("numeric query guard values must match declared training rows and columns")
    protected = torch.zeros((n, width), dtype=torch.bool)
    for column, spec in enumerate(table.schema):
        if spec.kind != "numeric":
            continue
        values = table.values[column].detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("numeric query guard requires finite training values")
        quantiles = torch.quantile(values, values.new_tensor([.25, .5, .75]),
                                   interpolation="linear")
        scale = torch.clamp((quantiles[2] - quantiles[0]) / 2, min=scale_floor)
        cutoff = threshold * scale
        if not bool(torch.isfinite(quantiles).all() & torch.isfinite(cutoff)):
            raise ValueError("nonfinite numeric query guard calibration")
        # A cell exactly at the declared boundary is still eligible.
        protected[:, column] = (values - quantiles[1]).abs() > cutoff
    per_column = protected.sum(0).tolist()
    return protected, {
        "protected_numeric_tail_cells": sum(per_column),
        "protected_numeric_tail_per_column": per_column,
        "numeric_query_guard": {
            "kind": "median_half_iqr", "max_abs_robust_z": float(threshold),
            "scale_floor": float(scale_floor), "comparison": "strictly_greater",
            "quantile_interpolation": "linear",
            "reference_scope": "all training rows before masking",
        },
    }


def global_query_mask(
    table: TablePlan, fraction: float, seed: int,
    *, numeric_query_guard: dict | None = None, numeric_scale_floor: float | None = None,
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

    numeric_protected = None
    numeric_audit = {}
    if numeric_query_guard is not None:
        numeric_protected, numeric_audit = numeric_tail_protection(
            table, numeric_query_guard, numeric_scale_floor,
        )
    elif numeric_scale_floor is not None:
        raise ValueError("numeric_scale_floor requires an explicit numeric_query_guard")

    count = max(1, math.floor(n * width * fraction + 0.5))
    rng = random.Random(seed)
    eligible = []
    eligible_counts = []
    singleton_per_column = []
    protected = 0
    for column, spec in enumerate(table.schema):
        keep = set()
        singleton_count = 0
        if spec.kind == "numeric" and numeric_protected is not None:
            keep = set(numeric_protected[:, column].nonzero().flatten().tolist())
        elif spec.kind != "numeric":
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
    if numeric_protected is not None:
        numeric_audit["query_numeric_tail_cells"] = int((query & numeric_protected).sum())
        if numeric_audit["query_numeric_tail_cells"]:
            raise AssertionError("numeric tail cell entered Query despite guard")
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
        **numeric_audit,
    )
