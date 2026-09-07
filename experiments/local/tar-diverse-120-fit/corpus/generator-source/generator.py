"""Observational v0 TabUF episode generator.

DiscoSCM (default source) draws a population mixture: M ~ P(m)∝1/m on
{1, ..., floor(sqrt(n))}, each component Gaussian (Cauchy mixture is off by default).
Then column types i.i.d. with weights 70/5/10/5/5
(numeric, ordinal, binary, categorical, high_cardinality), then normalized.
Each column is independently independent-of-rest with probability 0.05;
those columns skip the feature-token DAG and keep type-specific marginals.
Non-independent columns get tokens t_j in R^k from a unit-sphere SCM on a
sparse DAG (usual) or, with small probability, 1-2 star hubs (nonlinear φ on the parent mix, heterogeneous
noise). Cells use type-specific g_j linear in <u, t_j>; K-class is softmax
of linear logits. Nonlinear φ stays on the token SEM only.
Column noise is Gaussian-only by default; ``enable_heavy_tails=True`` restores the
student-t/Cauchy alternatives for the token and observation noise paths.
Other sources (scm, sklearn_*, openml, recsys) have their own semantics;
see sources.SOURCE_PROFILES. The shared contract is the table wire only.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable
import os

import numpy as np

DEFAULT_UNIT_DIM = None  # sample from prior; typical ~16, support [2, 1024]
UNIT_DIM_MIN = 2
UNIT_DIM_MAX = 1024
DEFAULT_SIGMA = 0.3
DEFAULT_QUERY_FRAC = 0.15
DEFAULT_MISSING_FRAC = 0.05
COL_TYPES = ("numeric", "ordinal", "binary", "categorical", "high_cardinality")
DEFAULT_TYPE_WEIGHTS = {
    "numeric": 70.0,
    "ordinal": 5.0,
    "binary": 10.0,
    "categorical": 5.0,
    "high_cardinality": 5.0,
}
DEFAULT_INDEP_FRAC = 0.05
DEFAULT_DAG_EDGE_P = 0.3  # legacy; the hub DAG sampler does not use Bernoulli(p)
DEFAULT_DAG_MAX_PARENTS = 6  # typical in-degree cap when max_parents is None
DEFAULT_STAR_PROB = 0.15  # P(star overlay | max_parents is None); else common sparse DAG
DEFAULT_BETA_MIN = 0.5
DEFAULT_BETA_MAX = 2.0
DEFAULT_TOKEN_HERITABILITY = 0.75  # α
DEFAULT_N_FEATURES_FALLBACK = 20  # non-discoscm when n_features is None
N_FEATURES_MIN = 2
N_FEATURES_MAX = 1000

ACTIVATIONS = ("identity", "tanh", "leaky_relu", "sin")
ACTIVATION_PROBS = (0.50, 0.20, 0.20, 0.10)
NOISE_FAMS = ("gaussian", "student_t", "cauchy")
NOISE_PROBS = (0.50, 0.30, 0.20)
STUDENT_T_DFS = (3, 4, 5, 6)


def _type_probs(weights: dict[str, float] | None) -> np.ndarray:
    src = DEFAULT_TYPE_WEIGHTS if weights is None else weights
    w = np.array([float(src.get(k, 0.0)) for k in COL_TYPES], dtype=np.float64)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0:
        raise ValueError("type_weights must sum to a positive number")
    return w / s


def _json_bool_grid(arr: np.ndarray) -> list[list[bool]]:
    return arr.astype(bool).tolist()


def _json_float_grid(arr: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in arr]


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def _n_classes_for(kind: str, n_units: int, rng: np.random.Generator) -> int | None:
    if kind == "numeric":
        return None
    if kind == "binary":
        return 2
    if kind == "ordinal":
        return int(rng.integers(3, 9))
    if kind == "categorical":
        return int(rng.integers(8, 33))
    hi = int(max(48, min(256, max(n_units, 48))))
    lo = int(max(32, min(64, hi - 1)))
    return int(rng.integers(lo, hi + 1))


def _sample_n_features(rng: np.random.Generator) -> int:
    """Mixture prior: typical tables around 20, support [2, 1000]."""
    u = rng.random()
    if u < 0.80:
        # typical tables: lognormal around 20, clip [4, 64]
        d = int(round(np.exp(rng.normal(np.log(20.0), 0.50))))
        d = int(np.clip(d, 4, 64))
    elif u < 0.95:
        # medium: log-uniform [20, 200]
        d = int(round(np.exp(rng.uniform(np.log(20.0), np.log(200.0)))))
    else:
        # rare wide: log-uniform [200, 1000]
        d = int(round(np.exp(rng.uniform(np.log(200.0), np.log(1000.0)))))
    d = int(np.clip(d, N_FEATURES_MIN, N_FEATURES_MAX))
    return d


def _resolve_n_features(n_features: int | None, rng: np.random.Generator) -> tuple[int, bool]:
    if n_features is None:
        return _sample_n_features(rng), True
    return int(np.clip(int(n_features), N_FEATURES_MIN, N_FEATURES_MAX)), False


def _sample_unit_dim(rng: np.random.Generator) -> int:
    """Mixture prior: typical k around 16, support [2, 1024]. Hand-tuned."""
    u = rng.random()
    if u < 0.80:
        k = int(round(np.exp(rng.normal(np.log(16.0), 0.45))))
        k = int(np.clip(k, 2, 64))
    elif u < 0.95:
        k = int(round(np.exp(rng.uniform(np.log(32.0), np.log(256.0)))))
    else:
        k = int(round(np.exp(rng.uniform(np.log(256.0), np.log(1024.0)))))
    return int(np.clip(k, UNIT_DIM_MIN, UNIT_DIM_MAX))


def _resolve_unit_dim(unit_dim: int | None, rng: np.random.Generator) -> tuple[int, bool]:
    if unit_dim is None:
        return _sample_unit_dim(rng), True
    return int(np.clip(int(unit_dim), UNIT_DIM_MIN, UNIT_DIM_MAX)), False



def _sample_n_mixture_components(rng: np.random.Generator, n: int) -> int:
    """M on {1, ..., floor(sqrt(n))} with P(M=m) ∝ 1/m."""
    m_max = max(1, int(np.floor(np.sqrt(max(int(n), 1)))))
    ms = np.arange(1, m_max + 1, dtype=np.int64)
    p = 1.0 / ms.astype(np.float64)
    p = p / p.sum()
    return int(rng.choice(ms, p=p))


def _sample_population_U(
    rng: np.random.Generator, n: int, k: int, *, enable_heavy_tails: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unit representations from a small location-scale mixture.

    M ~ 1/m on 1..floor(sqrt(n)). Default: every component is Gaussian.
    Cauchy is kept behind enable_heavy_tails for later exploration. M=1 is
    centered at 0 (Gaussian recovers N(0,I)). M>1 puts locations on a
    scaled sphere so clusters separate.
    """
    n, k = int(n), int(k)
    M = _sample_n_mixture_components(rng, n)
    pi = rng.dirichlet(np.ones(M, dtype=np.float64))
    assign = rng.choice(M, size=n, p=pi)
    U = np.zeros((n, k), dtype=np.float64)
    components: list[dict[str, Any]] = []
    for m in range(M):
        family = (
            str(rng.choice(("gaussian", "cauchy")))
            if enable_heavy_tails
            else "gaussian"
        )
        if M == 1:
            mu = np.zeros(k, dtype=np.float64)
            scale = 1.0
        else:
            sep = float(rng.uniform(1.5, 4.0))
            mu = sep * _l2_normalize(rng.standard_normal(k))
            scale = float(np.exp(rng.normal(0.0, 0.35)))
            scale = float(np.clip(scale, 0.15, 4.0))
        idx = np.flatnonzero(assign == m)
        n_m = int(idx.size)
        if n_m:
            if family == "gaussian":
                noise = rng.standard_normal((n_m, k))
            else:
                noise = rng.standard_cauchy(size=(n_m, k))
            U[idx] = mu[None, :] + scale * noise
        components.append(
            {
                "family": family,
                "location": [float(x) for x in mu],
                "scale": scale,
                "weight": float(pi[m]),
                "n_assigned": n_m,
            }
        )
    spec = {
        "form": "location_scale_mixture",
        "n_components": M,
        "m_max": max(1, int(np.floor(np.sqrt(max(n, 1))))),
        "sampling": "P(M=m) proportional to 1/m on 1..m_max",
        "weights": [float(x) for x in pi],
        "assignment": [int(x) for x in assign],
        "components": components,
        "enable_heavy_tails": bool(enable_heavy_tails),
    }
    return U, spec

def _apply_activation(v: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return v
    if name == "tanh":
        return np.tanh(v)
    if name == "leaky_relu":
        return np.where(v > 0, v, 0.2 * v)
    if name == "sin":
        return np.sin(v)
    raise ValueError("unknown activation %r" % name)


def _draw_vec(
    rng: np.random.Generator,
    k: int,
    family: str,
    df: int | float = 4,
) -> np.ndarray:
    k = int(k)
    if family == "gaussian":
        return rng.standard_normal(k).astype(np.float64)
    if family == "student_t":
        return rng.standard_t(float(df), size=k).astype(np.float64)
    return rng.standard_cauchy(size=k).astype(np.float64)


def _sample_noise_spec(
    rng: np.random.Generator, d: int, *, enable_heavy_tails: bool = False
) -> tuple[list[str], list[int | None]]:
    if enable_heavy_tails:
        choices = NOISE_FAMS
        probabilities = NOISE_PROBS
    else:
        # The default prior is strictly Gaussian.  The opt-in branch restores
        # the complete heterogeneous noise mixture.
        choices = ("gaussian",)
        probabilities = (1.0,)
    families: list[str] = []
    dfs: list[int | None] = []
    for _ in range(d):
        fam = str(rng.choice(choices, p=probabilities))
        df = int(rng.choice(STUDENT_T_DFS)) if fam == "student_t" else None
        families.append(fam)
        dfs.append(df)
    return families, dfs


def _sample_token_dag(
    rng: np.random.Generator,
    *,
    d: int,
    independent: np.ndarray,
    edge_p: float = DEFAULT_DAG_EDGE_P,
    max_parents: int | None = None,
    graph_family: str | None = None,
) -> tuple[list[list[int]], list[int], list[bool], list[int], list[int], str]:
    """Sparse DAG over non-independent features; stars are a rare mixture arm.

    Random permutation of active nodes is the topological order. Independent
    columns have no parents and are never used as parents.

    ``edge_p`` / ``dag_edge_p`` is a legacy field: accepted and stored, but
    this sampler does **not** draw Bernoulli(p) per potential edge.

    Sparse backbone: Poisson λ=2.2, typical in-degree cap = 6 (or the hard
    ``max_parents`` cap), preferential attachment P(p) ∝ outdeg[p]+1.

    ``max_parents``:
      * ``None`` — common sparse backbone (Poisson λ=2.2, cap 6, PA).
        With probability ``DEFAULT_STAR_PROB`` (0.15), or if
        ``graph_family="star"``, overlay 1–2 stars: frac ~ Unif[0.45, 0.85],
        out-hub in the first third of topo, in-hub in the last third.
        ``graph_family="sparse"`` never overlays stars.
      * positive int — hard cap for every node; no stars that exceed it.
    """
    del edge_p  # legacy; not used
    parents: list[list[int]] = [[] for _ in range(d)]
    is_hub = [False] * d
    out_hubs: list[int] = []
    in_hubs: list[int] = []
    active = [j for j in range(d) if not bool(independent[j])]
    if not active:
        return parents, [], is_hub, out_hubs, in_hubs, "sparse"
    n_active = len(active)
    hard_cap = None if max_parents is None else max(1, int(max_parents))
    typical_cap = DEFAULT_DAG_MAX_PARENTS if hard_cap is None else hard_cap
    topo = [int(x) for x in rng.permutation(np.asarray(active, dtype=np.int64))]
    outdeg = np.zeros(d, dtype=np.int64)
    for i, j in enumerate(topo):
        earlier = topo[:i]
        if not earlier:
            continue
        n_earlier = len(earlier)
        cap = min(n_earlier, typical_cap)
        if cap <= 0:
            continue
        k_pa = int(np.clip(int(rng.poisson(2.2)), 0, cap))
        if k_pa <= 0:
            continue
        earlier_arr = np.asarray(earlier, dtype=np.int64)
        weights = outdeg[earlier_arr].astype(np.float64) + 1.0
        weights = weights / weights.sum()
        chosen = rng.choice(earlier_arr, size=k_pa, replace=False, p=weights)
        chosen_list = [int(x) for x in np.atleast_1d(chosen)]
        parents[j] = chosen_list
        for p in chosen_list:
            outdeg[p] += 1

    fam = (graph_family or "").strip().lower() or None
    if fam not in (None, "sparse", "star"):
        raise ValueError("graph_family must be None, 'sparse', or 'star'")
    if hard_cap is not None or n_active < 8:
        stars_on = False
        fam_used = "sparse"
    elif fam == "star":
        stars_on = True
        fam_used = "star"
    elif fam == "sparse":
        stars_on = False
        fam_used = "sparse"
    else:
        stars_on = bool(rng.random() < DEFAULT_STAR_PROB)
        fam_used = "star" if stars_on else "sparse"
    if stars_on:
        frac = float(rng.uniform(0.45, 0.85))
        n_third = max(1, n_active // 3)
        early = topo[:n_third]
        late = topo[-n_third:]
        place_out = True
        place_in = True
        if n_active < 12:
            if float(rng.random()) < 0.5:
                place_in = False
            else:
                place_out = False
        if place_out:
            out_hub = int(rng.choice(np.asarray(early, dtype=np.int64)))
            out_hubs.append(out_hub)
            is_hub[out_hub] = True
            hub_pos = topo.index(out_hub)
            for j in topo[hub_pos + 1 :]:
                if float(rng.random()) < frac and out_hub not in parents[j]:
                    parents[j].append(out_hub)
        if place_in:
            in_hub = int(rng.choice(np.asarray(late, dtype=np.int64)))
            in_hubs.append(in_hub)
            is_hub[in_hub] = True
            hub_pos = topo.index(in_hub)
            earlier = topo[:hub_pos]
            n_earlier = len(earlier)
            if n_earlier > 0:
                k = int(np.clip(int(round(frac * n_earlier)), 1, n_earlier))
                earlier_arr = np.asarray(earlier, dtype=np.int64)
                chosen = rng.choice(earlier_arr, size=k, replace=False)
                parents[in_hub] = [int(x) for x in np.atleast_1d(chosen)]
    else:
        n_edges = sum(len(parents[j]) for j in active)
        limit = 4 * n_active
        if n_edges > limit:
            edges: list[tuple[int, int]] = [
                (j, p) for j in active for p in parents[j]
            ]
            keep_idx = set(
                int(x) for x in rng.choice(len(edges), size=limit, replace=False)
            )
            new_parents: list[list[int]] = [[] for _ in range(d)]
            for idx, (j, p) in enumerate(edges):
                if idx in keep_idx:
                    new_parents[j].append(p)
            parents = new_parents
    return parents, topo, is_hub, out_hubs, in_hubs, fam_used


def _orthogonal_unit(
    rng: np.random.Generator,
    s: np.ndarray,
    *,
    eta_raw: np.ndarray | None = None,
    draw_fn: Callable[[], np.ndarray] | None = None,
) -> np.ndarray:
    """Unit vector in the orthogonal complement of unit vector ``s``."""
    k = int(s.shape[0])

    def _draw() -> np.ndarray:
        if draw_fn is not None:
            return np.asarray(draw_fn(), dtype=np.float64).reshape(-1)
        return rng.standard_normal(k).astype(np.float64)

    used_pre = False
    for _ in range(16):
        if eta_raw is not None and not used_pre:
            raw = np.asarray(eta_raw, dtype=np.float64).reshape(-1)
            used_pre = True
        else:
            raw = _draw()
        eta = raw - float(np.dot(raw, s)) * s
        nrm = float(np.linalg.norm(eta))
        if nrm >= 1e-8:
            return eta / nrm
    # Degenerate draw: a standard-basis vector of the orthogonal complement.
    i = int(np.argmin(np.abs(s)))
    e = np.zeros(k, dtype=np.float64)
    e[i] = 1.0
    eta = e - float(np.dot(e, s)) * s
    nrm = float(np.linalg.norm(eta))
    if nrm < 1e-8 and k > 1:
        e = np.zeros(k, dtype=np.float64)
        e[(i + 1) % k] = 1.0
        eta = e - float(np.dot(e, s)) * s
        nrm = float(np.linalg.norm(eta))
    return eta / max(nrm, 1e-8)


def _sample_feature_tokens(
    rng: np.random.Generator,
    *,
    d: int,
    k: int,
    parents: list[list[int]],
    topo_order: list[int],
    beta_min: float = DEFAULT_BETA_MIN,
    beta_max: float = DEFAULT_BETA_MAX,
    token_heritability: float = DEFAULT_TOKEN_HERITABILITY,
    noise_families: list[str] | None = None,
    noise_dfs: list[int | None] | None = None,
) -> tuple[list[np.ndarray | None], list[list[float]], list[str | None]]:
    """Unit-sphere SEM on feature tokens t_j in R^k (not on Y).

    β are signed mixing weights (Σ|β|=1); causal strength is α.
    Relative |β| ~ Unif[beta_min, beta_max] (NOTEARS-style), independent sign ±1,
    then L1-normalized so Σ|β|=1. Single parent ⇒ β = ±1 always.
    Roots: t_j = normalize(η) from the column's noise family; φ = identity.
    Children: s_raw = Σ β t_p; s = normalize(φ_j(s_raw)); η ⊥ s on the sphere;
      t_j = normalize(α s + sqrt(1-α²) η), α = token_heritability in (0.05, 0.95).
    Independent columns are absent from ``topo_order`` and get no token
    (activation None).
    """
    tokens: list[np.ndarray | None] = [None] * d
    betas: list[list[float]] = [[] for _ in range(d)]
    activations: list[str | None] = [None] * d
    fams = noise_families if noise_families is not None else ["gaussian"] * d
    dfs = noise_dfs if noise_dfs is not None else [None] * d
    alpha = float(np.clip(token_heritability, 0.05, 0.95))
    noise_w = float(np.sqrt(max(1.0 - alpha * alpha, 0.0)))
    for j in topo_order:
        pa = parents[j]
        fam = fams[j]
        df = dfs[j] if dfs[j] is not None else 4
        if not pa:
            eta = _draw_vec(rng, k, fam, df)
            tokens[j] = _l2_normalize(eta)
            betas[j] = []
            activations[j] = "identity"
            continue
        bj: list[float] = []
        for p in pa:
            mag = float(rng.uniform(beta_min, beta_max))
            sign = 1.0 if float(rng.random()) < 0.5 else -1.0
            bj.append(sign * mag)
        l1 = sum(abs(b) for b in bj)
        bj = [b / l1 for b in bj]
        s_raw = np.zeros(k, dtype=np.float64)
        for b, p in zip(bj, pa):
            parent_t = tokens[p]
            assert parent_t is not None
            s_raw = s_raw + b * parent_t
        phi = str(rng.choice(ACTIVATIONS, p=ACTIVATION_PROBS))
        s = _l2_normalize(_apply_activation(s_raw, phi))
        eta = _orthogonal_unit(
            rng, s, draw_fn=lambda fam=fam, df=df: _draw_vec(rng, k, fam, df)
        )
        t = alpha * s + noise_w * eta
        tokens[j] = _l2_normalize(t)
        betas[j] = bj
        activations[j] = phi
    return tokens, betas, activations


def _realize_column(
    rng: np.random.Generator,
    *,
    kind: str,
    independent: bool,
    n_classes: int | None,
    U: np.ndarray,
    sigma: float,
    token: np.ndarray | None = None,
    noise_family: str = "gaussian",
    noise_df: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Unit-specific response for one feature.

    Non-independent columns use the feature token t_j in place of the old w_j.
    The cell response is linear in u; nonlinear φ lives only on the token SEM
    (s_raw = Σ β t_p). Type-specific g_j:
      numeric:    Y = scale * <u, t_j> + shift + e          (affine)
      binary:     1{<u, t_j> + e > 0}                       (linear probit)
      ordinal:    ordered probit, cuts on <u, t_j> + e      (linear index)
      categorical / high_cardinality: K-class linear softmax
          C_c = a_c t_j + 0.25 * residual_c, residual ⊥ t_j
          logits = U @ C.T, p = softmax(logits), Y ~ Categorical(p)
    independent: drop u and the token SCM; draw from a column-wise marginal
    of the same type. Observation noise e uses the column's noise family
    (gaussian / student_t / cauchy), scaled by sigma.
    """
    n, k = int(U.shape[0]), int(U.shape[1])
    df = int(noise_df) if noise_df is not None else 4
    meta: dict[str, Any] = {
        "type": kind,
        "independent": bool(independent),
        "n_classes": n_classes,
        "noise_family": noise_family,
        "noise_df": int(noise_df) if noise_family == "student_t" else None,
    }

    def obs_e(scale: float) -> np.ndarray:
        return float(scale) * _draw_vec(rng, n, noise_family, df)

    if kind == "numeric":
        if independent:
            mu = float(rng.normal(0.0, 1.0))
            sig = float(np.exp(rng.normal(-0.2, 0.4)))
            y = mu + obs_e(sig)
            meta["g"] = "independent_gaussian"
            meta["mu"] = mu
            meta["sigma"] = sig
            return y.astype(np.float64), meta
        assert token is not None
        t = _l2_normalize(np.asarray(token, dtype=np.float64))
        scale = float(np.exp(rng.normal(0.0, 0.35)))
        shift = float(rng.normal(0.0, 0.5))
        e = obs_e(float(sigma))
        y = scale * (U @ t) + shift + e
        meta["g"] = "affine_factor"
        meta["t"] = [float(x) for x in t]
        meta["scale"] = scale
        meta["shift"] = shift
        meta["sigma"] = float(sigma)
        return y.astype(np.float64), meta

    if kind == "binary":
        if independent:
            p = float(rng.uniform(0.1, 0.9))
            y = rng.binomial(1, p, size=n)
            meta["g"] = "independent_bernoulli"
            meta["p"] = p
            return y.astype(np.int64), meta
        assert token is not None
        t = _l2_normalize(np.asarray(token, dtype=np.float64))
        e = obs_e(float(sigma))
        y = (U @ t + e > 0.0).astype(np.int64)
        meta["g"] = "latent_threshold"
        meta["t"] = [float(x) for x in t]
        meta["sigma"] = float(sigma)
        return y, meta

    assert n_classes is not None
    L = int(n_classes)

    if independent:
        alpha = np.ones(L) * (0.4 if kind == "high_cardinality" else 0.9)
        p = rng.dirichlet(alpha)
        y = rng.choice(L, size=n, p=p)
        meta["g"] = "independent_multinomial"
        meta["probs"] = [float(x) for x in p]
        return y.astype(np.int64), meta

    assert token is not None
    t = _l2_normalize(np.asarray(token, dtype=np.float64))

    if kind == "ordinal":
        e = obs_e(float(sigma))
        s = U @ t + e
        cuts = np.sort(rng.normal(0.0, 1.0, size=L - 1))
        y = np.sum(s[:, None] > cuts[None, :], axis=1)
        meta["g"] = "ordered_probit"
        meta["t"] = [float(x) for x in t]
        meta["thresholds"] = [float(x) for x in cuts]
        meta["sigma"] = float(sigma)
        return y.astype(np.int64), meta

    # Unordered: column token is t_j.
    #   C_c = a_c * t_j + 0.25 * residual_c
    # with residual_c in the orthogonal complement of t_j (so C_c · t_j = a_c).
    # High-card scales stay heterogeneous so a few classes dominate (Zipf-like).
    # logits = U @ C.T (linear in u); Y ~ Categorical(softmax(logits)).
    a = rng.standard_normal(L)
    residual = rng.standard_normal((L, k))
    residual = residual - np.outer(residual @ t, t)
    residual_scale = 0.25
    if kind == "high_cardinality":
        scales = np.exp(rng.normal(0.0, 0.9, size=L))
        a = a * scales
        residual = residual * scales[:, None]
        meta["class_scales"] = [float(x) for x in scales]
    C = np.outer(a, t) + residual_scale * residual
    logits = U @ C.T
    p = _softmax_rows(logits)
    cdf = np.cumsum(p, axis=1)
    cdf[:, -1] = 1.0
    y = (rng.random(n)[:, None] <= cdf).argmax(axis=1)
    meta["g"] = "linear_softmax"
    meta["t"] = [float(x) for x in t]
    meta["class_embeddings"] = _json_float_grid(C)
    return y.astype(np.int64), meta



def canonical_query_mode(query_mode: str | None) -> str:
    """Two modes: any_cell (whole grid) and label_cell (one target column).

    Legacy aliases: cells / observed_cells → any_cell; label_column → label_cell.
    """
    raw = (query_mode or "any_cell").strip().lower()
    if raw in ("label_cell", "label_column"):
        return "label_cell"
    return "any_cell"


def _compile_masks(
    rng: np.random.Generator,
    n: int,
    d: int,
    missing_frac: float | None,
    query_frac: float | None,
    query_mode: str | None = None,
    query_column: int | None = None,
) -> tuple[np.ndarray, np.ndarray, str, int | None]:
    """MCAR missing over the grid; query is any_cell or label_cell at query_frac."""
    n_cells = n * d
    if missing_frac is None:
        missing_frac = DEFAULT_MISSING_FRAC
    if query_frac is None:
        query_frac = DEFAULT_QUERY_FRAC
    missing_frac = float(np.clip(missing_frac, 0.0, 0.95))
    query_frac = float(np.clip(query_frac, 0.0, 0.95))
    n_miss = max(0, min(int(round(missing_frac * n_cells)), n_cells))
    missing_flat = np.zeros(n_cells, dtype=bool)
    if n_miss:
        missing_flat[rng.choice(n_cells, size=n_miss, replace=False)] = True
    missing_mask = missing_flat.reshape(n, d)
    mode = canonical_query_mode(query_mode)
    query_mask = np.zeros((n, d), dtype=bool)
    held = None
    if mode == "label_cell":
        held = d - 1 if query_column is None else int(query_column)
        held = int(np.clip(held, 0, d - 1))
        n_query = max(1, min(int(round(query_frac * n)), n))
        rows = rng.choice(n, size=n_query, replace=False)
        query_mask[rows, held] = True
    else:
        n_query = max(1, min(int(round(query_frac * n_cells)), n_cells))
        query_flat = np.zeros(n_cells, dtype=bool)
        query_flat[rng.choice(n_cells, size=n_query, replace=False)] = True
        query_mask = query_flat.reshape(n, d)
    return missing_mask, query_mask, mode, held


def sample_episode(
    rng: np.random.Generator,
    *,
    n_units: int = 1000,
    n_features: int | None = None,
    unit_dim: int | None = None,
    query_frac: float | None = None,
    missing_frac: float | None = None,
    sigma: float = DEFAULT_SIGMA,
    enable_heavy_tails: bool = False,
    column_normalize: bool = True,
    debug: bool = False,
    return_mechanism: bool = False,
    seed: int | None = 0,
    type_weights: dict[str, float] | None = None,
    independent_frac: float = DEFAULT_INDEP_FRAC,
    dag_edge_p: float = DEFAULT_DAG_EDGE_P,
    max_parents: int | None = None,
    token_heritability: float = DEFAULT_TOKEN_HERITABILITY,
    beta_min: float = DEFAULT_BETA_MIN,
    beta_max: float = DEFAULT_BETA_MAX,
    graph_family: str | None = None,
    query_mode: str | None = None,
    query_column: int | None = None,
) -> dict[str, Any]:
    n = int(n_units)
    d, n_features_sampled = _resolve_n_features(n_features, rng)
    k, unit_dim_sampled = _resolve_unit_dim(unit_dim, rng)
    probs = _type_probs(type_weights)
    indep_p = float(np.clip(independent_frac, 0.0, 1.0))
    used_weights = {
        key: float((type_weights or DEFAULT_TYPE_WEIGHTS).get(key, DEFAULT_TYPE_WEIGHTS[key]))
        for key in COL_TYPES
    }

    U, mix_spec = _sample_population_U(
        rng, n, k, enable_heavy_tails=enable_heavy_tails
    )
    kinds = rng.choice(COL_TYPES, size=d, p=probs)
    independent = rng.random(d) < indep_p
    n_classes = [_n_classes_for(str(kinds[j]), n, rng) for j in range(d)]
    noise_families, noise_dfs = _sample_noise_spec(
        rng, d, enable_heavy_tails=enable_heavy_tails
    )

    parents, topo_order, is_hub, out_hubs, in_hubs, graph_kind = _sample_token_dag(
        rng,
        d=d,
        independent=independent,
        edge_p=dag_edge_p,
        max_parents=max_parents,
        graph_family=graph_family,
    )
    tokens, betas, activations = _sample_feature_tokens(
        rng,
        d=d,
        k=k,
        parents=parents,
        topo_order=topo_order,
        beta_min=beta_min,
        beta_max=beta_max,
        token_heritability=token_heritability,
        noise_families=noise_families,
        noise_dfs=noise_dfs,
    )

    Y_cols: list[np.ndarray] = []
    col_maps: list[dict[str, Any]] = []
    for j in range(d):
        yj, meta = _realize_column(
            rng,
            kind=str(kinds[j]),
            independent=bool(independent[j]),
            n_classes=n_classes[j],
            U=U,
            sigma=sigma,
            token=tokens[j],
            noise_family=noise_families[j],
            noise_df=noise_dfs[j],
        )
        meta["j"] = j
        Y_cols.append(yj)
        col_maps.append(meta)

    values: list[list[float | int]] = []
    for i in range(n):
        row: list[float | int] = []
        for j in range(d):
            v = Y_cols[j][i]
            row.append(int(v) if n_classes[j] is not None else float(v))
        values.append(row)

    missing_mask, query_mask, qmode, held = _compile_masks(
        rng, n, d, missing_frac, query_frac, query_mode, query_column
    )

    table: dict[str, Any] = {
        "n_units": n,
        "n_rows": n,
        "n_features": d,
        "unit_dim": k,
        "source": "discoscm",
        "values": values,
        "missing_mask": _json_bool_grid(missing_mask),
        "query_mask": _json_bool_grid(query_mask),
        "column_types": [str(x) for x in kinds],
        "n_classes": n_classes,
        "shapes": {
            "values": [n, d],
            "missing_mask": [n, d],
            "query_mask": [n, d],
            "n_missing": int(missing_mask.sum()),
            "n_query": int(query_mask.sum()),
            "n_query_and_missing": int((query_mask & missing_mask).sum()),
        },
        "query_mode": qmode,
        "query_column": held,
    }
    payload: dict[str, Any] = {"seed": seed, "table": table}

    if return_mechanism or debug:
        typical_cap = (
            DEFAULT_DAG_MAX_PARENTS if max_parents is None else int(max_parents)
        )
        max_in_degree = max((len(pa) for pa in parents), default=0)
        payload["population"] = {
            "id": f"pop-{seed}",
            "n_units": n,
            "unit_dim": k,
            "prior": mix_spec["form"],
            "note": "This episode sampled its own population. Row i is unit u_i.",
            "mixture": {k: v for k, v in mix_spec.items() if k != "assignment"},
            "assignment": mix_spec["assignment"],
            "representations": _json_float_grid(U),
        }
        payload["response_law"] = {
            "form": "feature_token_linear_scm",
            "type_weights": used_weights,
            "column_independent_prob": indep_p,
            "dag_edge_p": float(dag_edge_p),
            "max_parents": None if max_parents is None else int(max_parents),
            "typical_max_parents": int(typical_cap),
            "max_in_degree": int(max_in_degree),
            "token_heritability": float(np.clip(token_heritability, 0.05, 0.95)),
            "beta_min": float(beta_min),
            "beta_max": float(beta_max),
            "n_features_sampled": bool(n_features_sampled),
            "unit_dim_sampled": bool(unit_dim_sampled),
            "parents": parents,
            "betas": betas,
            "tokens": [
                None if t is None else [float(x) for x in t] for t in tokens
            ],
            "activations": activations,
            "noise_families": list(noise_families),
            "enable_heavy_tails": bool(enable_heavy_tails),
            "hubs": [j for j, h in enumerate(is_hub) if h],
            "graph_family": graph_kind,
            "out_hubs": list(out_hubs),
            "in_hubs": list(in_hubs),
            "columns": col_maps,
        }
    return payload


def _sample_one_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    """One episode. Module-level so a thread/process pool can call it."""
    from sources import (
        SKLEARN_REAL_CANONICAL,
        SKLEARN_SYNTH_CANONICAL,
        openml_table,
        recsys_table,
        resolve_profile,
        scm_anm,
        sklearn_real,
        sklearn_synthetic,
    )

    ep_seed = int(job["ep_seed"])
    rng = np.random.default_rng(ep_seed)
    resolved_src = job["resolved_src"]
    resolved_name = job["resolved_name"]
    prof = resolve_profile(
        resolved_src,
        query_mode=job.get("query_mode"),
        missing_frac=job.get("missing_frac"),
        query_frac=job.get("query_frac"),
    )
    mf = float(prof["missing_frac"])
    qf = float(prof["query_frac"])
    qmode = str(prof["query_mode"])
    n_units = job["n_units"]
    n_features = job["n_features"]
    unit_dim = job["unit_dim"]
    if resolved_src == "discoscm":
        return sample_episode(
            rng,
            n_units=n_units,
            n_features=n_features,
            unit_dim=unit_dim,
            query_frac=qf,
            missing_frac=mf,
            sigma=job["sigma"],
            enable_heavy_tails=job["enable_heavy_tails"],
            column_normalize=job["column_normalize"],
            debug=job["debug"],
            return_mechanism=job["return_mechanism"],
            seed=ep_seed,
            type_weights=job["type_weights"],
            independent_frac=job["independent_frac"],
            dag_edge_p=job["dag_edge_p"],
            max_parents=job["max_parents"],
            token_heritability=job["token_heritability"],
            beta_min=job["beta_min"],
            beta_max=job["beta_max"],
            graph_family=job["graph_family"],
            query_mode=qmode,
            query_column=job.get("query_column"),
        )
    if resolved_src == "scm" and job.get("api_version") == "v1":
        from scm_v1 import sample_mixed_scm_episode

        scm_options = dict(job.get("scm_options") or {})
        return sample_mixed_scm_episode(
            n_units=n_units,
            n_features=n_features,
            missing_frac=mf,
            query_frac=qf,
            sigma=job["sigma"],
            seed=ep_seed,
            return_mechanism=job["return_mechanism"],
            query_mode=qmode,
            query_column=job.get("query_column"),
            type_weights=scm_options.get("type_weights"),
            target_type=scm_options.get("target_type", "auto"),
            max_parents=scm_options.get("max_parents", 3),
        )
    kw = dict(
        rng=rng,
        n_units=n_units,
        n_features=n_features,
        missing_frac=mf,
        query_frac=qf,
        seed=ep_seed,
        return_mechanism=job["return_mechanism"],
        query_mode=qmode,
        source=resolved_src,
    )
    if resolved_src in SKLEARN_SYNTH_CANONICAL:
        ep = sklearn_synthetic(source_name=resolved_name, **kw)
    elif resolved_src in SKLEARN_REAL_CANONICAL:
        ep = sklearn_real(source_name=resolved_name, **kw)
    elif resolved_src == "scm":
        ep = scm_anm(sigma=job["sigma"], **kw)
    elif resolved_src == "openml":
        ep = openml_table(
            source_name=resolved_name,
            fold_index=job.get("episode_index", 0),
            **kw,
        )
    elif resolved_src == "recsys":
        ep = recsys_table(source_name=job.get("source_name"), **kw)
    else:
        raise ValueError("unknown source %r" % resolved_src)
    ep.pop("population", None)
    ep.pop("response_law", None)
    return ep


def _shape_rng(base: int, batch_index: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(base), 0x53485045, int(batch_index)])
    )


def group_batches(
    episodes: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Split a flat episode list into same-shape training batches."""
    bs = max(1, int(batch_size))
    out: list[dict[str, Any]] = []
    for b, start in enumerate(range(0, len(episodes), bs)):
        chunk = episodes[start : start + bs]
        t0 = chunk[0]["table"]
        out.append(
            {
                "batch_index": b,
                "n_units": t0["n_units"],
                "n_features": t0["n_features"],
                "unit_dim": t0.get("unit_dim"),
                "shape": [t0["n_units"], t0["n_features"]],
                "n_episodes": len(chunk),
                "episodes": chunk,
            }
        )
    return out


def sample_episodes(
    *,
    n_units: int = 1000,
    n_features: int | None = None,
    unit_dim: int | None = None,
    query_frac: float | None = None,
    missing_frac: float | None = None,
    sigma: float = DEFAULT_SIGMA,
    enable_heavy_tails: bool = False,
    seed: int | None = 0,
    n_episodes: int | None = None,
    batch_size: int | None = None,
    debug: bool = False,
    return_mechanism: bool = False,
    column_normalize: bool = True,
    type_weights: dict[str, float] | None = None,
    independent_frac: float = DEFAULT_INDEP_FRAC,
    dag_edge_p: float = DEFAULT_DAG_EDGE_P,
    max_parents: int | None = None,
    token_heritability: float = DEFAULT_TOKEN_HERITABILITY,
    beta_min: float = DEFAULT_BETA_MIN,
    beta_max: float = DEFAULT_BETA_MAX,
    graph_family: str | None = None,
    source: str = "discoscm",
    source_name: str | None = None,
    query_mode: str | None = None,
    query_column: int | None = None,
    api_version: str = "v0",
    scm_options: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Draw distinct episodes, packed into same-shape batches.

    ``n_episodes`` is how many different worlds to draw (default 8).
    ``batch_size`` is the GPU pack (default 1): within a batch, n/d/k are
    shared; the next batch re-samples d and k (if they were left null).
    Episode content uses seed, seed+1, ... independently of the shape stream.
    """
    n_ep = 8 if n_episodes is None else int(n_episodes)
    bs = 1 if batch_size is None else int(batch_size)
    n_ep = max(1, min(n_ep, 4096))
    bs = max(1, min(bs, 512))
    if seed is None:
        base = int(np.random.default_rng().integers(0, 2**31 - 1))
    else:
        base = int(seed)
    from sources import (
        CANONICAL_SOURCES,
        OPENML_CTR23_DEFAULT,
        SKLEARN_REAL_CANONICAL,
        SKLEARN_REAL_DS_TO_SOURCE,
        SKLEARN_SYNTH_CANONICAL,
        SKLEARN_SYNTH_MAKER_TO_SOURCE,
        SOURCES,
    )

    src = (source or "discoscm").lower()
    version = str(api_version or "v0").lower()
    if version not in ("v0", "v1"):
        raise ValueError("api_version must be v0 or v1")
    if src not in SOURCES:
        raise ValueError(
            "unknown source %r; use %s" % (src, ", ".join(CANONICAL_SOURCES))
        )

    src_rng = np.random.default_rng(np.random.SeedSequence([base, 0x535243]))
    resolved_src = src
    resolved_name = source_name
    if src == "sklearn_synthetic":
        makers = list(SKLEARN_SYNTH_CANONICAL.values())
        resolved_name = source_name or str(src_rng.choice(makers))
        if resolved_name not in SKLEARN_SYNTH_MAKER_TO_SOURCE:
            raise ValueError("unknown sklearn synthetic maker: %s" % resolved_name)
        resolved_src = SKLEARN_SYNTH_MAKER_TO_SOURCE[resolved_name]
    elif src == "sklearn_real":
        datasets = list(SKLEARN_REAL_CANONICAL.values())
        resolved_name = source_name or str(src_rng.choice(datasets))
        if resolved_name not in SKLEARN_REAL_DS_TO_SOURCE:
            raise ValueError("unknown sklearn real dataset: %s" % resolved_name)
        resolved_src = SKLEARN_REAL_DS_TO_SOURCE[resolved_name]
    elif src in SKLEARN_SYNTH_CANONICAL:
        resolved_name = SKLEARN_SYNTH_CANONICAL[src]
    elif src in SKLEARN_REAL_CANONICAL:
        resolved_name = SKLEARN_REAL_CANONICAL[src]
    elif src == "openml":
        # one table for the whole request, like sklearn_real
        if source_name is None:
            resolved_name = str(int(OPENML_CTR23_DEFAULT))
        else:
            resolved_name = str(source_name).strip()

    job_base = {
        "api_version": version,
        "scm_options": scm_options,
        "resolved_src": resolved_src,
        "resolved_name": resolved_name,
        "source_name": source_name,
        "n_units": int(n_units),
        "query_mode": query_mode,
        "missing_frac": missing_frac,
        "query_frac": query_frac,
        "sigma": sigma,
        "enable_heavy_tails": bool(enable_heavy_tails),
        "column_normalize": column_normalize,
        "debug": debug,
        "return_mechanism": return_mechanism,
        "type_weights": type_weights,
        "independent_frac": independent_frac,
        "dag_edge_p": dag_edge_p,
        "max_parents": max_parents,
        "token_heritability": token_heritability,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "graph_family": graph_family,
        "query_column": query_column,
    }
    jobs: list[dict[str, Any]] = []
    n_batches = (n_ep + bs - 1) // bs
    for b in range(n_batches):
        start = b * bs
        stop = min(start + bs, n_ep)
        shape_rng = _shape_rng(base, b)
        if resolved_src == "discoscm":
            d_use, _ = _resolve_n_features(n_features, shape_rng)
            k_use, _ = _resolve_unit_dim(unit_dim, shape_rng)
        elif resolved_src == "openml":
            # official protocol: always native width; n_units ignored for shape
            d_use = None
            k_use = None
        else:
            d_use = (
                DEFAULT_N_FEATURES_FALLBACK
                if n_features is None
                else int(np.clip(int(n_features), N_FEATURES_MIN, N_FEATURES_MAX))
            )
            k_use = None
        for e in range(start, stop):
            jobs.append(
                {
                    **job_base,
                    "n_features": None if d_use is None else int(d_use),
                    "unit_dim": k_use,
                    "ep_seed": base + e,
                    "episode_index": e,
                }
            )
    workers = min(len(jobs), os.cpu_count() or 1)
    if len(jobs) == 1 or workers <= 1:
        return [_sample_one_episode_job(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_sample_one_episode_job, jobs))
