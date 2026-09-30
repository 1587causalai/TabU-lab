"""Stateless episode sampling over a fixed training pool, separate from the model."""

from __future__ import annotations

import hashlib
import json

import torch

from .types import TAREpisode


def episode_seed(seed, namespace, episode_id, stream="codebook"):
    """New IDs have independent streams; replaying an ID reproduces its realization."""
    if any(type(x) is not int or x < 0 for x in (seed, episode_id)):
        raise ValueError("seed and episode_id must be nonnegative integers")
    if not isinstance(namespace, str) or not namespace or not stream:
        raise ValueError("episode RNG requires named namespaces and streams")
    key = json.dumps([seed, namespace, episode_id, stream], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % (2**63 - 1)


def supervised_episode(context, queries, features, *, codebook_seed):
    """Last-column supervised wedge: query truth leaves the forward input here."""
    if context.ndim != 2 or queries.ndim != 2 or context.shape[1] != queries.shape[1]:
        raise ValueError("context and queries must have matching matrix schemas")
    if len(context) < 2 or len(queries) < 1:
        raise ValueError("need at least two context rows and one query")
    values = torch.cat((context, queries))
    visible = torch.ones_like(values, dtype=torch.bool)
    visible[len(context) :, -1] = False
    mask = torch.zeros_like(visible)
    mask[len(context) :, -1] = True
    episode = TAREpisode.from_table(values, visible, mask, features, codebook_seed=codebook_seed)
    episode.validate()
    truth = {
        (len(context) + r, values.shape[1] - 1): float(queries[r, -1]) for r in range(len(queries))
    }
    return episode, truth


def sample_supervised_episode(
    training_values, features, *, row_ids, context_size, seed, namespace, episode_id
):
    """Caller supplies only the frozen training pool; no held-out data is accepted.

    Uniform sampling uses addresses, never labels. All remaining pool rows are
    queries. Domain indices keep their meaning; nominal embedding vectors change.
    """
    n = len(training_values)
    if len(row_ids) != n or len(set(row_ids)) != n:
        raise ValueError("training row IDs must be unique and match the pool")
    if type(context_size) is not int or not 2 <= context_size < n:
        raise ValueError("invalid context size")
    generator = torch.Generator(device="cpu").manual_seed(
        episode_seed(seed, namespace, episode_id, "row_roles")
    )
    order = torch.randperm(n, generator=generator).tolist()
    c, q = order[:context_size], order[context_size:]
    book_seed = episode_seed(seed, namespace, episode_id)
    episode, truth = supervised_episode(
        training_values[c], training_values[q], features, codebook_seed=book_seed
    )
    record = dict(
        namespace=namespace,
        episode_id=episode_id,
        codebook_seed=book_seed,
        context_row_ids=[row_ids[i] for i in c],
        query_row_ids=[row_ids[i] for i in q],
    )
    return episode, truth, record


def covering_fit_episodes(training_values, features, *, row_ids, query_size, seed,
                          namespace, count):
    """Fixed cyclic masks cover every training row, without consulting labels."""
    n = len(training_values)
    if len(row_ids) != n or len(set(row_ids)) != n:
        raise ValueError("training row IDs must be unique and match the pool")
    if not 1 <= query_size <= n - 2 or count * query_size < n:
        raise ValueError("fit bank must cover all rows with at least two context rows")
    gen = torch.Generator().manual_seed(episode_seed(seed, namespace, 0, "row_roles"))
    order = torch.randperm(n, generator=gen).tolist()
    bank = []
    for i in range(count):
        q = [order[(i * query_size + j) % n] for j in range(query_size)]
        selected = set(q)
        c = [j for j in order if j not in selected]
        book_seed = episode_seed(seed, namespace, i)
        episode, truth = supervised_episode(training_values[c], training_values[q], features,
                                           codebook_seed=book_seed)
        bank.append((episode, truth, dict(
            namespace=namespace, episode_id=i, codebook_seed=book_seed,
            context_row_ids=[row_ids[j] for j in c], query_row_ids=[row_ids[j] for j in q],
        )))
    return bank
