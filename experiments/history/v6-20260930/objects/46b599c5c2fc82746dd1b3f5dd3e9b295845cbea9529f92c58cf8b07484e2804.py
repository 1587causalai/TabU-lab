# Derived from tabuf-episode-api (MIT, Copyright 2026 Heyang Gong).
# Modified: package imports and separation of synthetic mechanisms; see PROVENANCE.json.
from __future__ import annotations
from typing import Any
import numpy as np
from .discoscm import _json_bool_grid, canonical_query_mode


def _masks(
    rng: np.random.Generator,
    n: int,
    d: int,
    missing_frac: float,
    query_frac: float,
    query_mode: str,
    query_column: int | None = None,
):
    n_cells = n * d
    n_miss = max(0, min(int(round(float(missing_frac) * n_cells)), n_cells))
    missing_flat = np.zeros(n_cells, dtype=bool)
    if n_miss:
        missing_flat[rng.choice(n_cells, size=n_miss, replace=False)] = True
    missing_mask = missing_flat.reshape(n, d)
    query_mask = np.zeros((n, d), dtype=bool)
    mode = canonical_query_mode(query_mode)
    held = None
    if mode == "label_cell":
        held = d - 1 if query_column is None else int(query_column)
        held = int(max(0, min(held, d - 1)))
        n_query = max(1, min(int(round(float(query_frac) * n)), n))
        rows = rng.choice(n, size=n_query, replace=False)
        query_mask[rows, held] = True
    else:
        n_query = max(1, min(int(round(float(query_frac) * n_cells)), n_cells))
        query_flat = np.zeros(n_cells, dtype=bool)
        query_flat[rng.choice(n_cells, size=n_query, replace=False)] = True
        query_mask = query_flat.reshape(n, d)
    return missing_mask, query_mask, mode, held


def pack_grid(
    Y: np.ndarray,
    rng: np.random.Generator,
    *,
    missing_frac: float,
    query_frac: float,
    column_types: list[str],
    n_classes: list[int | None],
    seed: int | None,
    return_mechanism: bool,
    mechanism: dict[str, Any] | None,
    query_mode: str = "any_cell",
    source: str | None = None,
    mechanism_field: str = "mechanism",
    missing_mask: np.ndarray | None = None,
    query_mask: np.ndarray | None = None,
    query_column: int | None = None,
) -> dict[str, Any]:
    n, d = int(Y.shape[0]), int(Y.shape[1])
    if missing_mask is not None and query_mask is not None:
        missing_mask = np.asarray(missing_mask, dtype=bool).reshape(n, d)
        query_mask = np.asarray(query_mask, dtype=bool).reshape(n, d)
        mode = canonical_query_mode(query_mode)
        held = None
        if mode == "label_cell":
            held = d - 1 if query_column is None else int(query_column)
            held = int(max(0, min(held, d - 1)))
    else:
        missing_mask, query_mask, mode, held = _masks(
            rng, n, d, missing_frac, query_frac, query_mode, query_column
        )
    values: list[list[float | int]] = []
    for i in range(n):
        row: list[float | int] = []
        for j in range(d):
            v = Y[i, j]
            if n_classes[j] is not None and np.isfinite(v):
                row.append(int(v))
            else:
                row.append(float(v))
        values.append(row)
    table = {
        "n_units": n,
        "n_rows": n,
        "n_features": d,
        "values": values,
        "missing_mask": _json_bool_grid(missing_mask),
        "query_mask": _json_bool_grid(query_mask),
        "column_types": column_types,
        "n_classes": n_classes,
        "shapes": {
            "values": [n, d],
            "missing_mask": [n, d],
            "query_mask": [n, d],
            "n_missing": int(missing_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_query_and_missing": int((query_mask & missing_mask).sum()),
        },
        "query_mode": mode,
        "query_column": held,
        "source": source,
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}
    if return_mechanism and mechanism is not None:
        payload[mechanism_field] = mechanism
    return payload
