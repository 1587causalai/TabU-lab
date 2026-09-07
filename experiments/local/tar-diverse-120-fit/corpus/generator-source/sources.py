"""Pluggable priors that all compile into the same episode table contract.

The shared contract is the WIRE envelope only: a complete table plus
missing_mask and query_mask. Each source has its own row semantics.
DiscoSCM language (units, population, response_law) applies ONLY to
source=discoscm.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from generator import _json_bool_grid, DEFAULT_MISSING_FRAC, DEFAULT_QUERY_FRAC, canonical_query_mode

SKLEARN_SYNTH_CANONICAL = {
    "sklearn_make_classification": "make_classification",
    "sklearn_make_regression": "make_regression",
    "sklearn_friedman1": "make_friedman1",
    "sklearn_low_rank": "make_low_rank_matrix",
}
SKLEARN_SYNTH_MAKER_TO_SOURCE = {v: k for k, v in SKLEARN_SYNTH_CANONICAL.items()}

SKLEARN_REAL_CANONICAL = {
    "sklearn_iris": "iris",
    "sklearn_wine": "wine",
    "sklearn_breast_cancer": "breast_cancer",
    "sklearn_diabetes": "diabetes",
}
SKLEARN_REAL_DS_TO_SOURCE = {v: k for k, v in SKLEARN_REAL_CANONICAL.items()}

SKLEARN_REAL = ("iris", "wine", "breast_cancer", "diabetes")

# OpenML-CTR23 regression suite (study 353). 35 tables in study data_id order.
# Zip with study task_id order: 44980 (kin8nm) → 361258.
OPENML_CTR23 = (
    44956, 44957, 44958, 44959, 44963, 44964, 44965, 44966, 44969,
    44971, 44972, 44973, 44974, 44975, 44976, 44977, 44978, 44979,
    44980, 44981, 44983, 44984, 44987, 44989, 44990, 44992, 44993,
    45012, 41021, 44960, 44962, 44967, 44970, 44994, 45402,
)
OPENML_CTR23_TASK_IDS = (
    361234, 361235, 361236, 361237, 361241, 361242, 361243, 361244, 361247,
    361249, 361250, 361251, 361252, 361253, 361254, 361255, 361256, 361257,
    361258, 361259, 361260, 361261, 361264, 361266, 361267, 361268, 361269,
    361272, 361616, 361617, 361618, 361619, 361621, 361622, 361623,
)
OPENML_CTR23_TASKS = dict(zip(OPENML_CTR23, OPENML_CTR23_TASK_IDS))
OPENML_CTR23_DEFAULT = 44970  # QSAR_fish_toxicity
OPENML_ROW_CAP = 100000
assert OPENML_CTR23_TASKS[44980] == 361258

CANONICAL_SOURCES = (
    "discoscm",
    "scm",
    *SKLEARN_SYNTH_CANONICAL.keys(),
    *SKLEARN_REAL_CANONICAL.keys(),
    "openml",
    "recsys",
)

ALIAS_SOURCES = ("sklearn_synthetic", "sklearn_real")

SOURCES = CANONICAL_SOURCES + ALIAS_SOURCES

_SHARED = (
    "n_units",
    "n_features",
    "seed",
    "n_episodes",
    "batch_size",
    "source",
    "missing_frac",
    "query_frac",
    "query_mode",
    "query_column",
    "return_mechanism",
)
_DISCOSCM_ONLY = ("unit_dim", "type_weights", "independent_frac", "dag_edge_p", "max_parents", "token_heritability", "beta_min", "beta_max", "graph_family")
_SIGMA = ("sigma",)
_DEBUG = ("debug",)
_SOURCE_NAME = ("source_name",)


def _rf(used: tuple[str, ...] | list[str], ignored: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    return {
        "request_fields_used": list(used),
        "request_fields_ignored": list(ignored),
    }


_SKLEARN_USED = _SHARED
_SKLEARN_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG + _SOURCE_NAME
_ALIAS_USED = _SHARED + _SOURCE_NAME
_ALIAS_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG
_PLACEHOLDER_USED = _SHARED + _SOURCE_NAME
_PLACEHOLDER_IGNORED = _DISCOSCM_ONLY + _SIGMA + _DEBUG

SOURCE_PROFILES: dict[str, dict[str, Any]] = {
    "discoscm": {
        "status": "ready",
        "family": "discoscm",
        "row_meaning": "unit",
        "query_mode": "any_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": True,
        **_rf(_SHARED + _DISCOSCM_ONLY + _SIGMA + _DEBUG, _SOURCE_NAME),
        "note": "Rows are units with latent u_i; cell-wise missing/query; unit-specific response law. See data-generation.pdf.",
    },
    "scm": {
        "status": "ready",
        "family": "scm",
        "row_meaning": "iid_sample",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SHARED + _SIGMA, _DISCOSCM_ONLY + _DEBUG + _SOURCE_NAME),
        "note": "i.i.d. rows from a random additive-noise DAG over features; last column is the prediction target; missing 0.05.",
    },
    "sklearn_make_classification": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn make_classification; last column is binary y; missing 0.05 / query 0.15 of the label column.",
    },
    "sklearn_make_regression": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn make_regression; last column is continuous y; missing 0.05 / query 0.15 of the label column.",
    },
    "sklearn_friedman1": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn Friedman #1; last column is continuous y; missing 0.05 / query 0.15 of the label column.",
    },
    "sklearn_low_rank": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "any_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "sklearn low-rank matrix; no designated label; cell-wise missing/query like matrix completion.",
    },
    "sklearn_iris": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled iris table; last column is the class label; missing 0.05.",
    },
    "sklearn_wine": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled wine table; last column is the class label; missing 0.05.",
    },
    "sklearn_breast_cancer": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled breast_cancer table; last column is the class label; missing 0.05.",
    },
    "sklearn_diabetes": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_SKLEARN_USED, _SKLEARN_IGNORED),
        "note": "Bundled diabetes table; last column is the continuous target; missing 0.05.",
    },
    "sklearn_synthetic": {
        "status": "ready",
        "family": "sklearn_synthetic",
        "row_meaning": "iid_sample",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_ALIAS_USED, _ALIAS_IGNORED),
        "note": "Alias: pass source_name to pick a maker, or one is drawn at random. Per-maker masks follow the canonical profile.",
        "makers": list(SKLEARN_SYNTH_CANONICAL.values()),
        "alias_of": list(SKLEARN_SYNTH_CANONICAL.keys()),
    },
    "sklearn_real": {
        "status": "ready",
        "family": "sklearn_real",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_ALIAS_USED, _ALIAS_IGNORED),
        "note": "Alias: pass source_name to pick a bundled table, or one is drawn at random.",
        "datasets": list(SKLEARN_REAL),
        "alias_of": list(SKLEARN_REAL_CANONICAL.keys()),
    },
    "openml": {
        "status": "ready",
        "family": "openml",
        "row_meaning": "entity_row",
        "query_mode": "label_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": 0.0,
        "uses_unit_token": False,
        **_rf(_SHARED + _SOURCE_NAME, _DISCOSCM_ONLY + _SIGMA + _DEBUG),
        "note": "Official OpenML-CTR23 task split (study 353); no injected missing; full native table; episode i uses official fold i. source_name is a data_id; omit → 44970. n_features/n_units ignored for shape (row cap 100000). query_mode label_cell; query = official TEST rows of y.",
        "suite": "CTR23",
        "data_ids": list(OPENML_CTR23),
        "task_ids": list(OPENML_CTR23_TASK_IDS),
    },
    "recsys": {
        "status": "placeholder",
        "family": "recsys",
        "row_meaning": "user",
        "query_mode": "any_cell",
        "query_frac": DEFAULT_QUERY_FRAC,
        "missing_frac": DEFAULT_MISSING_FRAC,
        "uses_unit_token": False,
        **_rf(_PLACEHOLDER_USED, _PLACEHOLDER_IGNORED),
        "note": "User×Item ratings; compiler missing 0.05 until unobserved-as-missing is wired; returns 501 until cached.",
    },
}

# /v1 changes only the SCM generator contract. Other source semantics are
# delegated to their /v0 implementations and remain separately owned.
SOURCE_PROFILES_V1: dict[str, dict[str, Any]] = {
    source: dict(profile) for source, profile in SOURCE_PROFILES.items()
}
SOURCE_PROFILES_V1["scm"] = {
    "status": "ready",
    "family": "scm",
    "contract_version": "scm.mixed_observational.v1",
    "row_meaning": "iid_sample",
    "query_mode": "label_cell",
    "query_frac": DEFAULT_QUERY_FRAC,
    "missing_frac": DEFAULT_MISSING_FRAC,
    "uses_unit_token": False,
    "safety_limits": {
        "max_n_units": 10000,
        "max_n_features": 128,
        "max_n_episodes": 32,
        "max_generated_cells": 250000,
        "max_parents": 8,
        "max_mechanism_edge_slots": 1024,
    },
    **_rf(
        _SHARED + _SIGMA + ("scm_options",),
        _DISCOSCM_ONLY + _DEBUG + _SOURCE_NAME,
    ),
    "note": (
        "Versioned mixed-type observational SCM. The target is a non-root "
        "sink and may be numeric, binary, ordinal, categorical, or explicitly "
        "high-cardinality. Observational only; no Unit-token, intervention, "
        "or counterfactual contract."
    ),
}


def resolve_profile(
    source: str,
    *,
    query_mode: str | None = None,
    missing_frac: float | None = None,
    query_frac: float | None = None,
) -> dict[str, Any]:
    src = (source or "discoscm").lower()
    if src not in SOURCE_PROFILES:
        raise ValueError(
            "unknown source %r; use %s" % (src, ", ".join(CANONICAL_SOURCES))
        )
    base = dict(SOURCE_PROFILES[src])
    if query_mode:
        base["query_mode"] = query_mode
    if missing_frac is not None:
        base["missing_frac"] = missing_frac
    if query_frac is not None:
        base["query_frac"] = query_frac
    if base.get("missing_frac") is None:
        base["missing_frac"] = DEFAULT_MISSING_FRAC
    if base.get("query_frac") is None:
        base["query_frac"] = DEFAULT_QUERY_FRAC
    return base


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


def _fit_size(X: np.ndarray, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample or pad-with-repeat to requested n x d."""
    r, c = X.shape
    rows = rng.choice(r, size=n, replace=(r < n))
    cols = rng.choice(c, size=d, replace=(c < d))
    return np.asarray(X[np.ix_(rows, cols)], dtype=np.float64)


def _fit_supervised(Y: np.ndarray, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
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
            n_samples=n, n_features=max(d - 1, 2), n_informative=max(d // 2, 1),
            n_redundant=0, n_classes=2, random_state=rs,
        )
        if d >= 2:
            Y = np.column_stack([X[:, : d - 1], y])
        else:
            Y = y.reshape(-1, 1).astype(np.float64)
        types = ["numeric"] * (d - 1) + ["binary"] if d >= 2 else ["binary"]
        n_classes = [None] * (d - 1) + [2] if d >= 2 else [2]
    elif name == "make_regression":
        X, y = make_regression(n_samples=n, n_features=max(d - 1, 1), noise=0.3, random_state=rs)
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
        np.asarray(Y, dtype=np.float64), rng,
        missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source_key,
    )


def sklearn_real(
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
    from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "diabetes": load_diabetes,
    }
    name = source_name or str(rng.choice(list(loaders)))
    if name not in loaders:
        raise ValueError(f"unknown sklearn real dataset: {name}. try {SKLEARN_REAL}")
    bunch = loaders[name]()
    X = np.asarray(bunch.data, dtype=np.float64)
    y = np.asarray(bunch.target)
    Y0 = np.column_stack([X, y.reshape(-1, 1)])
    Y = _fit_supervised(Y0, n_units, n_features, rng)
    types = ["numeric"] * n_features
    n_classes: list[int | None] = [None] * n_features
    source_key = source or SKLEARN_REAL_DS_TO_SOURCE.get(name, "sklearn_real")
    mech = {
        "framework": "sklearn_real",
        "dataset": name,
        "n_rows_original": int(Y0.shape[0]),
        "n_cols_original": int(Y0.shape[1]),
    }
    return pack_grid(
        Y, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source_key,
    )


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
        X, rng, missing_frac=missing_frac, query_frac=query_frac,
        column_types=["numeric"] * d, n_classes=[None] * d, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode=query_mode, source=source or "scm",
    )


def _parse_openml_splits_arff(text: str) -> list[tuple[str, int, int, int]]:
    """Parse OpenML task-splits ARFF into (type, rowid, repeat, fold)."""
    rows: list[tuple[str, int, int, int]] = []
    in_data = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower().startswith("@data"):
                in_data = True
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        typ = parts[0]
        try:
            rowid = int(float(parts[1]))
            repeat = int(float(parts[2]))
            fold = int(float(parts[3]))
        except ValueError:
            continue
        rows.append((typ, rowid, repeat, fold))
    return rows


def fetch_openml_splits(task_id: int) -> list[tuple[str, int, int, int]]:
    """Fetch+cache official OpenML task splits ARFF.

    Cache: ``~/scikit_learn_data/openml-splits/task_{id}_splits.arff``.
    URL: ``https://openml.org/api_splits/get/{task_id}/Task_{task_id}_splits.arff``.
    """
    task_id = int(task_id)
    cache_dir = Path.home() / "scikit_learn_data" / "openml-splits"
    path = cache_dir / ("task_%s_splits.arff" % task_id)
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        url = "https://openml.org/api_splits/get/%s/Task_%s_splits.arff" % (
            task_id,
            task_id,
        )
        urllib.request.urlretrieve(url, path)
    return _parse_openml_splits_arff(path.read_text(encoding="utf-8", errors="replace"))


def _openml_estimation_note(n_rows: int) -> str:
    """OpenML study 353 estimation procedure from table size."""
    n = int(n_rows)
    if n < 1000:
        return "10x repeated 10-fold CV (n < 1000; OpenML study 353)"
    if n > 10000:
        return "33% holdout (n > 10000; OpenML study 353)"
    return "10-fold CV (OpenML study 353)"


def openml_table(
    rng: np.random.Generator,
    *,
    n_units: int,
    n_features: int | None,
    missing_frac: float,
    query_frac: float,
    seed: int | None,
    return_mechanism: bool,
    source_name: str | None = None,
    query_mode: str = "label_cell",
    source: str | None = None,
    fold_index: int = 0,
) -> dict[str, Any]:
    """One OpenML-CTR23 table compiled with the official task split.

    Full native table (no row/column subsample). Compiler missing is not
    injected. Query cells are the official TEST rows of y for
    (repeat, fold) selected from ``fold_index``.
    ``n_units`` / ``n_features`` / ``missing_frac`` / ``query_frac`` are ignored
    for shape and masks (row cap ``OPENML_ROW_CAP``).
    """
    del rng, n_units, n_features, missing_frac, query_frac  # official protocol
    import pandas as pd
    from sklearn.datasets import fetch_openml

    raw = source_name if source_name is not None else str(OPENML_CTR23_DEFAULT)
    try:
        data_id = int(str(raw).strip())
    except ValueError as exc:
        raise ValueError(
            "openml source_name must be an OpenML data_id (e.g. 44970). got %r" % raw
        ) from exc
    task_id = OPENML_CTR23_TASKS.get(data_id)
    if task_id is None:
        raise ValueError(
            "openml data_id %s is not in OpenML-CTR23 (study 353); no official task split"
            % data_id
        )
    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    from pandas.api.types import (
        is_bool_dtype,
        is_numeric_dtype,
        is_object_dtype,
        is_string_dtype,
    )

    X = bunch.data
    if X is None:
        X = pd.DataFrame()
    else:
        X = pd.DataFrame(X).copy()
    y = bunch.target
    if y is None:
        raise ValueError("OpenML data_id %s has no target column" % data_id)
    if isinstance(y, pd.DataFrame):
        if y.shape[1] == 0:
            raise ValueError("OpenML data_id %s has an empty target frame" % data_id)
        y = y.iloc[:, 0]
    y = pd.Series(y)

    def _encode_non_numeric(s: pd.Series) -> pd.Series:
        codes = s.astype("category").cat.codes
        out = codes.astype("float64")
        out = out.where(out >= 0.0, np.nan)
        return out

    encoded = {}
    for col in X.columns:
        s = X[col]
        non_num = (
            is_bool_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or is_object_dtype(s)
            or is_string_dtype(s)
            or not is_numeric_dtype(s)
        )
        if non_num:
            encoded[col] = _encode_non_numeric(s)
        else:
            encoded[col] = pd.to_numeric(s, errors="coerce")
    X = pd.DataFrame(encoded, index=X.index) if encoded else pd.DataFrame(index=X.index)
    y = pd.to_numeric(y, errors="coerce")
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.shape[1] == 0:
        Y0 = y_arr.reshape(-1, 1)
    else:
        x_arr = np.asarray(X, dtype=np.float64)
        n_align = min(x_arr.shape[0], y_arr.shape[0])
        if n_align == 0:
            raise ValueError("OpenML data_id %s returned an empty table" % data_id)
        Y0 = np.column_stack([x_arr[:n_align], y_arr[:n_align].reshape(-1, 1)])
    if Y0.size == 0 or Y0.shape[0] == 0:
        raise ValueError("OpenML data_id %s returned an empty table" % data_id)
    n_rows, n_cols = int(Y0.shape[0]), int(Y0.shape[1])
    if n_rows > OPENML_ROW_CAP:
        raise ValueError(
            "OpenML data_id %s has %s rows; hard safety cap is %s"
            % (data_id, n_rows, OPENML_ROW_CAP)
        )
    Y = Y0
    d = int(Y.shape[1])
    types = ["numeric"] * d
    n_classes: list[int | None] = [None] * d

    splits = fetch_openml_splits(int(task_id))
    if not splits:
        raise ValueError("OpenML task %s returned empty splits" % task_id)
    n_folds = len({f for _t, _i, _r, f in splits})
    n_repeats = len({rep for _t, _i, rep, _f in splits})
    n_folds = max(int(n_folds), 1)
    n_repeats = max(int(n_repeats), 1)
    fi = int(fold_index)
    fold = fi % n_folds
    repeat = (fi // n_folds) % n_repeats
    test_rows = [
        rowid
        for typ, rowid, r, f in splits
        if str(typ).upper() == "TEST" and int(r) == repeat and int(f) == fold
    ]
    train_rows = [
        rowid
        for typ, rowid, r, f in splits
        if str(typ).upper() == "TRAIN" and int(r) == repeat and int(f) == fold
    ]
    n_test = len(test_rows)
    n_train = len(train_rows)
    missing_mask = np.zeros((n_rows, d), dtype=bool)
    query_mask = np.zeros((n_rows, d), dtype=bool)
    held = d - 1
    for rowid in test_rows:
        if 0 <= int(rowid) < n_rows:
            query_mask[int(rowid), held] = True

    details = getattr(bunch, "details", None) or {}
    name = details.get("name") if isinstance(details, dict) else None
    mech = {
        "framework": "openml-ctr23",
        "data_id": int(data_id),
        "task_id": int(task_id),
        "name": str(name) if name else str(data_id),
        "n_rows_original": n_rows,
        "n_cols_original": n_cols,
        "fold": int(fold),
        "repeat": int(repeat),
        "n_folds": int(n_folds),
        "n_repeats": int(n_repeats),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "estimation": _openml_estimation_note(n_rows),
        "target": "last_column",
        "task": "regression",
        "in_ctr23": data_id in OPENML_CTR23,
    }
    dummy_rng = np.random.default_rng(0)
    return pack_grid(
        Y, dummy_rng, missing_frac=0.0, query_frac=0.0,
        column_types=types, n_classes=n_classes, seed=seed,
        return_mechanism=return_mechanism, mechanism=mech,
        query_mode="label_cell", source=source or "openml",
        missing_mask=missing_mask, query_mask=query_mask, query_column=held,
    )


def recsys_table(*_a, **_k) -> dict[str, Any]:
    raise NotImplementedError(
        "recsys source is specified (MovieLens / similar) but not cached yet."
    )
