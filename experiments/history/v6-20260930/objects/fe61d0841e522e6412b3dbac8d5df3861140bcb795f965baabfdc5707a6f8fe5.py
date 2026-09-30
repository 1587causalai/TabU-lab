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


def default_v53_query_guard(kind: str) -> dict:
    """Current V5.3 policy; historical callers keep their explicit recipes."""
    if kind == "random_cell":
        return {"kind": "std_iqr_column", "max_std_iqr_ratio": 2.0}
    if kind == "supervised_row":
        return {"kind": "none"}
    raise ValueError("unknown V5.3 masking kind")


def validate_numeric_query_guard(guard: dict) -> dict:
    """Validate a recorded sampling policy, retaining historical cell guards."""
    fields = {"median_half_iqr": "max_abs_robust_z", "std_iqr_column": "max_std_iqr_ratio"}
    if (not isinstance(guard, dict) or not isinstance(guard.get("kind"), str)
            or guard["kind"] not in fields):
        raise ValueError("numeric_query_guard kind must be median_half_iqr or std_iqr_column")
    field = fields[guard["kind"]]
    if set(guard) != {"kind", field}:
        raise ValueError(f"numeric_query_guard must declare kind and {field}")
    threshold = guard[field]
    if (isinstance(threshold, bool) or not isinstance(threshold, int | float)
            or not math.isfinite(threshold) or threshold <= 0):
        raise ValueError(f"numeric_query_guard {field} must be finite and positive")
    return {"kind": guard["kind"], field: float(threshold)}


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
    guard = validate_numeric_query_guard(guard)
    column_guard = guard["kind"] == "std_iqr_column"
    threshold = guard["max_std_iqr_ratio" if column_guard else "max_abs_robust_z"]
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
        spread = quantiles[2] - quantiles[0]
        scale = torch.clamp(spread if column_guard else spread / 2, min=scale_floor)
        cutoff = threshold * scale
        if not bool(torch.isfinite(quantiles).all() & torch.isfinite(cutoff)):
            raise ValueError("nonfinite numeric query guard calibration")
        # Equality stays eligible. Column policy uses population std and full IQR.
        if column_guard:
            std = torch.std(values, correction=0)
            if not bool(torch.isfinite(std)):
                raise ValueError("nonfinite numeric query guard standard deviation")
            protected[:, column] = std > cutoff
        else:
            protected[:, column] = (values - quantiles[1]).abs() > cutoff
    per_column = protected.sum(0).tolist()
    audit = {
        "protected_numeric_tail_cells": sum(per_column),
        "protected_numeric_tail_per_column": per_column,
        "numeric_query_guard": {
            **guard,
            "scale_floor": float(scale_floor), "comparison": "strictly_greater",
            "quantile_interpolation": "linear",
            "reference_scope": "all training rows before masking",
        },
    }
    if column_guard:
        audit["numeric_query_guard"].update(std_correction=0, scale="full_iqr")
        audit["protected_numeric_columns"] = sum(value == n for value in per_column)
        audit["protected_numeric_column_cells"] = sum(per_column)
    return protected, audit


def global_query_mask(
    table: TablePlan, fraction: float, seed: int,
    *, numeric_query_guard: dict | None = None, numeric_scale_floor: float | None = None,
    protect_ordinal_classes: bool = True, require_numeric_diversity: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Sample an exact global Query budget, retaining all observed classes.

    The budget is ``max(1, floor(rows * columns * fraction + 0.5))``.  Columns
    may receive zero Query cells.  Invalid or unsupported budgets raise rather
    than silently reducing the requested count.  Rejection is bounded so even
    pathological high fractions cannot hang an experiment.
    Historical callers protect every discrete class. A declared-rank ordinal
    codec can disable ordinal class protection; numeric affine supervision can
    require two distinct visible values via whole-candidate rejection.
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
    if type(protect_ordinal_classes) is not bool or type(require_numeric_diversity) is not bool:
        raise ValueError("support policies must be explicit booleans")
    numeric_values = {
        a: value.detach().cpu() for a, (spec, value) in
        enumerate(zip(table.schema, table.values, strict=True))
        if spec.kind == "numeric" and require_numeric_diversity
    }
    for column, values in numeric_values.items():
        if not bool(torch.isfinite(values).all()) or len(torch.unique(values)) < 2:
            raise ValueError(
                f"no-valid-episode: {table.name} numeric column {column} "
                "needs two distinct finite visible values"
            )

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
        elif spec.kind == "nominal" or (spec.kind == "ordinal" and protect_ordinal_classes):
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
        if not all(value <= n - 2 for value in per_column):
            continue
        hidden = set(sampled)
        if any(len(torch.unique(values[[row * width + column not in hidden
                                         for row in range(n)]])) < 2
               for column, values in numeric_values.items()):
            continue
        break
    else:
        raise ValueError(
            f"no-valid-episode: {table.name}: could not sample the requested global query budget "
            "within 1024 attempts while retaining required visible support"
        )

    query = torch.zeros(n * width, dtype=torch.bool)
    query[sampled] = True
    query = query.reshape(n, width)
    if numeric_protected is not None:
        numeric_audit["query_numeric_tail_cells"] = int((query & numeric_protected).sum())
        if numeric_audit["query_numeric_tail_cells"]:
            raise AssertionError("numeric tail cell entered Query despite guard")
        if numeric_query_guard["kind"] == "std_iqr_column":
            numeric_audit["query_protected_numeric_column_cells"] = (
                numeric_audit["query_numeric_tail_cells"]
            )
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
