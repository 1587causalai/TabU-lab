# Derived from tabuf-episode-api (MIT, Copyright 2026 Heyang Gong).
# Modified: package imports and separation of synthetic mechanisms; see PROVENANCE.json.
from __future__ import annotations
from typing import Any
import numpy as np
from .grid import pack_grid

SKLEARN_SYNTH_CANONICAL = {
    "sklearn_make_classification": "make_classification",
    "sklearn_make_regression": "make_regression",
    "sklearn_friedman1": "make_friedman1",
    "sklearn_low_rank": "make_low_rank_matrix",
}
SKLEARN_SYNTH_MAKER_TO_SOURCE = {v: k for k, v in SKLEARN_SYNTH_CANONICAL.items()}


def _fit_supervised(
    Y: np.ndarray, n: int, d: int, rng: np.random.Generator
) -> np.ndarray:
    """Keep the last column as label; subsample rows and the other features."""
    r, c = Y.shape
    rows = rng.choice(r, size=n, replace=(r < n))
    ylab = Y[rows, -1]
    if d <= 1:
        return ylab.reshape(-1, 1)
    feat = max(c - 1, 1)
    need = d - 1
    cols = rng.choice(feat, size=need, replace=(feat < need))
    return np.column_stack([Y[rows][:, cols], ylab])


def sklearn_synthetic(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    source_name: str | None = None,
    query_mode: str = "label_cell",
    source: str | None = None,
) -> dict[str, Any]:
    from sklearn.datasets import (
        make_classification,
        make_friedman1,
        make_low_rank_matrix,
        make_regression,
    )

    makers = {
        "make_classification": make_classification,
        "make_regression": make_regression,
        "make_friedman1": make_friedman1,
        "make_low_rank_matrix": make_low_rank_matrix,
    }
    name = source_name or str(rng.choice(list(makers)))
    if name not in makers:
        raise ValueError(f"unknown sklearn synthetic maker: {name}")
    rs = int(seed or 0)
    n, d = int(n_units), int(n_features)
    if name == "make_classification":
        X, y = make_classification(
            n_samples=n,
            n_features=max(d - 1, 2),
            n_informative=max(d // 2, 1),
            n_redundant=0,
            n_classes=2,
            random_state=rs,
        )
        if d >= 2:
            Y = np.column_stack([X[:, : d - 1], y])
        else:
            Y = y.reshape(-1, 1).astype(np.float64)
        types = ["numeric"] * (d - 1) + ["binary"] if d >= 2 else ["binary"]
        n_classes = [None] * (d - 1) + [2] if d >= 2 else [2]
    elif name == "make_regression":
        X, y = make_regression(
            n_samples=n, n_features=max(d - 1, 1), noise=0.3, random_state=rs
        )
        Y = np.column_stack([X[:, : max(d - 1, 1)], y])[:, :d]
        types = ["numeric"] * d
        n_classes = [None] * d
    elif name == "make_friedman1":
        nf = max(5, d)
        X, y = make_friedman1(n_samples=n, n_features=nf, noise=0.3, random_state=rs)
        Y = _fit_supervised(np.column_stack([X, y]), n, d, rng)
        types = ["numeric"] * d
        n_classes = [None] * d
    else:
        Y = make_low_rank_matrix(n_samples=n, n_features=d, random_state=rs)
        types = ["numeric"] * d
        n_classes = [None] * d
    source_key = source or SKLEARN_SYNTH_MAKER_TO_SOURCE.get(name, "sklearn_synthetic")
    mech = {"framework": "sklearn_synthetic", "maker": name}
    return pack_grid(
        np.asarray(Y, dtype=np.float64),
        rng,
        missing_frac=missing_frac,
        query_frac=query_frac,
        column_types=types,
        n_classes=n_classes,
        seed=seed,
        return_mechanism=return_mechanism,
        mechanism=mech,
        query_mode=query_mode,
        source=source_key,
    )
