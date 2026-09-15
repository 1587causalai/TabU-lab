"""The explicit small synthetic prior and address-only mask sampler in the design."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import torch

from .types import TAREpisode, TARFeature


def mask_observed(observed, *, rng, fraction=0.15, maximum_queries=64):
    if not 0 < fraction <= 1 or maximum_queries < 1:
        raise ValueError("invalid mask configuration")
    observed = np.asarray(observed, dtype=bool)
    remaining = observed.sum(axis=0).copy()
    coords = np.argwhere(observed)
    goal = min(maximum_queries, max(1, math.floor(fraction * len(coords))))
    query = np.zeros_like(observed)
    count = 0
    for idx in rng.permutation(len(coords)):
        r, a = coords[idx]
        if remaining[a] > 2:
            query[r, a] = True
            remaining[a] -= 1
            count += 1
            if count == goal:
                break
    if not count:
        raise ValueError("no-valid-episode")
    return query


def synthetic_episode(
    episode_id, *, seed=20260906, split_id=0, rows=(64, 128, 256, 512), columns=(4, 8, 16, 32)
):
    """Return (truth-free episode, separate truth) with disjoint deterministic RNGs."""
    streams = np.random.SeedSequence([seed, split_id, episode_id]).spawn(4)
    rng, missing, mask, codes = [np.random.default_rng(s) for s in streams]
    n, m = int(rng.choice(rows)), int(rng.choice(columns))
    if n < 3 or m < 1:
        raise ValueError("synthetic prior needs >=3 rows and >=1 columns")
    hidden = 8
    z = np.empty((n, hidden + m), dtype=np.float64)

    def loguniform(lo, hi):
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    for j in range(hidden + m):
        count = int(rng.integers(0, min(3, j) + 1))
        parents = rng.choice(j, size=count, replace=False)
        w = rng.normal(0, 1 / math.sqrt(max(1, count)), size=count)
        bias = rng.uniform(-0.5, 0.5)
        amplitude, noise = loguniform(0.5, 2), loguniform(0.05, 0.5)
        fn = int(rng.integers(4))
        eps = rng.normal(size=n)
        if not count:
            z[:, j] = eps
        else:
            x = bias + z[:, parents] @ w
            fx = (x, np.tanh(x), np.sin(x), np.maximum(x, 0))[fn]
            z[:, j] = amplitude * fx + noise * eps
    values = z[:, hidden:].copy()
    features = []
    for a in range(m):
        kind = str(rng.choice(["numeric", "nominal", "ordinal"], p=[0.5, 0.3, 0.2]))
        domain = ()
        if kind == "numeric":
            values[:, a] = values[:, a] * loguniform(0.1, 10) + rng.uniform(-2, 2)
        else:
            d = int(rng.choice([2, 4, 8, 16]))
            domain = tuple(f"class-{i}" for i in range(d))
            thresholds = [NormalDist().inv_cdf(i / d) for i in range(1, d)]
            classes = np.searchsorted(thresholds, values[:, a], side="right")
            values[:, a] = rng.permutation(d)[classes] if kind == "nominal" else classes
        features.append(TARFeature(kind, domain, a))
    observed = missing.random((n, m)) >= missing.uniform(0, 0.2, size=(1, m))
    rr, cc = rng.permutation(n), rng.permutation(m)
    values = values[rr][:, cc]
    observed = observed[rr][:, cc]
    features = tuple(features[a] for a in cc)
    query = mask_observed(observed, rng=mask)
    visible = observed & ~query
    truth = {(int(r), int(a)): float(values[r, a]) for r, a in np.argwhere(query)}
    # Keep original input precision; model preprocessing performs FP64 accumulation.
    episode = TAREpisode.from_table(
        torch.tensor(values),
        torch.tensor(visible),
        torch.tensor(query),
        features,
        codebook_seed=int(codes.integers(0, 2**63 - 1)),
    )
    return episode, truth
