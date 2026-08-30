from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from tabu_lab.contracts import OriginState, origin_code
from tabu_lab.experiments import tabubase_real_benchmark as real_benchmark
from tabu_lab.experiments.tabubase_real_benchmark import (
    RealDataset,
    _real_episode,
    evaluate_tabubase_on_indices,
    evaluation_context_indices,
    prepare_real_task,
    temperature_scale_probabilities,
    training_episode_indices,
)
from tabu_lab.experiments.tabubase_scale import build_tabubase_scale_model


def test_real_preparation_is_paired_and_train_only() -> None:
    rng = np.random.default_rng(7)
    dataset = RealDataset(
        dataset_id="fixture",
        task="classification",
        features=rng.normal(size=(200, 6)).astype(np.float32),
        response=np.tile(np.arange(2), 100),
        source="fixture",
    )
    first = prepare_real_task(dataset, budget=64, seed=1729)
    replay = prepare_real_task(dataset, budget=64, seed=1729)
    assert np.array_equal(first.label_indices, replay.label_indices)
    assert np.array_equal(first.test_indices, replay.test_indices)
    assert len(first.label_indices) == 64
    assert not set(first.label_indices) & set(first.test_indices)


def test_classification_evaluation_uses_full_budget_and_covers_every_class() -> None:
    rng = np.random.default_rng(11)
    dataset = RealDataset(
        dataset_id="ordered-three-class-fixture",
        task="classification",
        features=rng.normal(size=(180, 5)).astype(np.float32),
        # Source rows are deliberately class-blocked. A sorted-prefix context
        # would reproduce the Wine failure and omit the final class.
        response=np.repeat(np.arange(3), 60),
        source="fixture",
    )
    task = prepare_real_task(dataset, budget=90, seed=1729)
    context = evaluation_context_indices(task)
    assert np.array_equal(context, task.label_indices)
    assert set(task.response[context].tolist()) == {0, 1, 2}


def test_classification_training_episode_is_disjoint_and_class_covered() -> None:
    rng = np.random.default_rng(13)
    dataset = RealDataset(
        dataset_id="episode-fixture",
        task="classification",
        features=rng.normal(size=(240, 7)).astype(np.float32),
        response=np.repeat(np.arange(3), 80),
        source="fixture",
    )
    task = prepare_real_task(dataset, budget=120, seed=2718)
    context, query = training_episode_indices(task, seed=2718, update=17)
    assert not np.intersect1d(context, query).size
    assert set(task.response[context].tolist()) == {0, 1, 2}
    assert set(task.response[query].tolist()) == {0, 1, 2}

    evidence, truth = _real_episode(
        task,
        context_indices=context,
        query_indices=query,
        episode_id="truth-firewall-fixture",
    )
    query_start = len(context)
    assert torch.count_nonzero(evidence.forward_values[query_start:, -1]) == 0
    assert torch.all(evidence.origin_states[query_start:, -1] == origin_code(OriginState.QUERY))
    assert torch.equal(
        truth.target_values[query_start:, -1],
        torch.as_tensor(task.response[query], dtype=torch.float32),
    )
    assert torch.count_nonzero(truth.target_values[:query_start, -1]) == 0


def test_classification_evaluation_fails_closed_without_class_support() -> None:
    rng = np.random.default_rng(19)
    dataset = RealDataset(
        dataset_id="missing-support-fixture",
        task="classification",
        features=rng.normal(size=(180, 5)).astype(np.float32),
        response=np.repeat(np.arange(3), 60),
        source="fixture",
    )
    task = prepare_real_task(dataset, budget=90, seed=31415)
    incomplete = replace(
        task,
        label_indices=task.label_indices[task.response[task.label_indices] != 2],
    )
    with pytest.raises(ValueError, match="lacks response-class support"):
        evaluation_context_indices(incomplete)


def test_evaluation_chunking_does_not_rebuild_transductive_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(23)
    dataset = RealDataset(
        dataset_id="single-episode-fixture",
        task="classification",
        features=rng.normal(size=(240, 5)).astype(np.float32),
        response=np.tile(np.arange(2), 120),
        source="fixture",
    )
    task = prepare_real_task(dataset, budget=64, seed=1729, test_limit=None)
    model = build_tabubase_scale_model(seed=1729, device=torch.device("cpu"))
    calls = 0
    original = real_benchmark._real_episode

    def traced_episode(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(real_benchmark, "_real_episode", traced_episode)
    wide_chunks = evaluate_tabubase_on_indices(
        model,
        task,
        device=torch.device("cpu"),
        query_indices=task.test_indices,
        query_partition="test",
        query_readout_chunk_rows=64,
    )
    row_chunks = evaluate_tabubase_on_indices(
        model,
        task,
        device=torch.device("cpu"),
        query_indices=task.test_indices,
        query_partition="test",
        query_readout_chunk_rows=1,
    )

    assert calls == 2
    assert wide_chunks == pytest.approx(row_chunks)


def test_temperature_scaling_preserves_class_ranking_and_normalization() -> None:
    probabilities = np.asarray([[0.99, 0.01], [0.2, 0.8]], dtype=np.float64)
    softened = temperature_scale_probabilities(probabilities, 5.0)
    assert np.array_equal(softened.argmax(axis=1), probabilities.argmax(axis=1))
    np.testing.assert_allclose(softened.sum(axis=1), np.ones(2))
    assert softened[0, 0] < probabilities[0, 0]
    with pytest.raises(ValueError, match="positive"):
        temperature_scale_probabilities(probabilities, 0.0)
