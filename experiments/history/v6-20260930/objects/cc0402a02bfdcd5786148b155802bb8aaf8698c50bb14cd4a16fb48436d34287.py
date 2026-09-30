# Derived from tabuf-episode-api (MIT, Copyright 2026 Heyang Gong).
# Modified: package imports and separation of synthetic mechanisms; see PROVENANCE.json.
from __future__ import annotations
from typing import Any
import numpy as np
from .grid import pack_grid


def scm_anm(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    sigma: float,
    query_mode: str = "label_cell",
    source: str | None = None,
) -> dict[str, Any]:
    """Additive-noise SCM: i.i.d. rows from a random DAG over features.

    Last column is the designated prediction target (literature: sample an
    SCM, then pick a node to predict). Rows are observational draws, not units.
    """
    n, d = int(n_units), int(n_features)
    X = np.zeros((n, d), dtype=np.float64)
    edges: list[list[int]] = []
    node_fn: list[str] = []
    for j in range(d):
        e = rng.normal(0.0, float(sigma), size=n)
        if j == 0:
            X[:, j] = e
            node_fn.append("root_noise")
            continue
        n_pa = int(rng.integers(0, min(3, j) + 1))
        pa = rng.choice(j, size=n_pa, replace=False).tolist() if n_pa else []
        if not pa:
            X[:, j] = e
            node_fn.append("root_noise")
            continue
        w = rng.normal(0.0, 1.0, size=n_pa)
        lin = X[:, np.array(pa, dtype=int)] @ w
        kind = "linear" if rng.random() < 0.5 else "tanh"
        X[:, j] = (lin if kind == "linear" else np.tanh(lin)) + e
        node_fn.append(kind)
        for p in pa:
            edges.append([int(p), j])
    mech = {
        "framework": "ANM-SCM",
        "edges": edges,
        "node_fn": node_fn,
        "target_col": d - 1,
    }
    return pack_grid(
        X,
        rng,
        missing_frac=missing_frac,
        query_frac=query_frac,
        column_types=["numeric"] * d,
        n_classes=[None] * d,
        seed=seed,
        return_mechanism=return_mechanism,
        mechanism=mech,
        query_mode=query_mode,
        source=source or "scm",
    )
