from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from tabu_lab.experiments.query_row_real_benchmark import _model_prediction
from tabu_lab.experiments.query_row_real_coordinates import (
    query_row_real_regression_loss,
    task_scale_to_raw,
)
from tabu_lab.experiments.tabubase_real_benchmark import (
    _real_episode,
    evaluation_context_indices,
    load_real_dataset,
    prepare_real_task,
    training_episode_indices,
)
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig


def _row_model(seed: int = 1729) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_model(
        "tabu.query.row",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=4,
            max_features=256,
        ),
        profile="supervised.label_broadcast.v1",
        row_token_count=4,
    )


def test_numeric_raw_identity_uses_episode_context_statistics() -> None:
    task = prepare_real_task(load_real_dataset("diabetes"), budget=32, seed=1729, test_limit=8)
    model = _row_model()
    context_a, query = training_episode_indices(task, seed=1729, update=0)
    context_b = np.roll(context_a, 1)
    evidence_a, _ = _real_episode(task, context_indices=context_a, query_indices=query, episode_id="a")
    evidence_b, _ = _real_episode(task, context_indices=context_b, query_indices=query, episode_id="b")
    with torch.no_grad():
        prediction_a = model(evidence_a)
        prediction_b = model(evidence_b)
    for prediction in (prediction_a, prediction_b):
        numeric = prediction["numeric"]
        raw = prediction["numeric_raw_prediction"]
        scale = prediction["numeric_context_scale"]
        mean = prediction["numeric_context_mean"]
        assert torch.allclose(raw, numeric * scale + mean, atol=1.0e-6, rtol=1.0e-6)
    assert not torch.equal(
        prediction_a["numeric_context_mean"], prediction_b["numeric_context_mean"]
    )
    assert not torch.equal(
        prediction_a["numeric_context_scale"], prediction_b["numeric_context_scale"]
    )


def test_oracle_task_scale_prediction_has_zero_real_regression_error() -> None:
    task = prepare_real_task(load_real_dataset("diabetes"), budget=32, seed=2718, test_limit=8)
    context, query = training_episode_indices(task, seed=2718, update=0)
    evidence, truth = _real_episode(task, context_indices=context, query_indices=query, episode_id="oracle")
    model = _row_model(2718)
    prediction = model._forward_dense(evidence.to("cpu"), emit_trace=False)
    auxiliaries = dict(prediction.auxiliaries)
    auxiliaries["numeric_raw_prediction"] = truth.target_values.clone()
    oracle = replace(prediction, auxiliaries=auxiliaries)
    loss = query_row_real_regression_loss(oracle, truth)
    assert loss.total.item() == 0.0


def test_task_scale_inverse_is_affine_invariant_and_matches_eval_boundary() -> None:
    task = prepare_real_task(load_real_dataset("diabetes"), budget=32, seed=31415, test_limit=8)
    model = _row_model(31415)
    predicted_raw, _ = _model_prediction(model, task)
    context, query = training_episode_indices(task, seed=31415, update=0)
    task_scale = np.asarray(predicted_raw - task.response_mean, dtype=np.float64) / task.response_scale
    assert np.allclose(
        task_scale_to_raw(
            task_scale,
            response_mean=task.response_mean,
            response_scale=task.response_scale,
        ),
        predicted_raw,
    )
    a, b = 3.0, 7.0
    transformed = task_scale_to_raw(
        task_scale,
        response_mean=a * task.response_mean + b,
        response_scale=a * task.response_scale,
    )
    assert np.allclose(transformed, a * predicted_raw + b)
    assert context.size > 0 and query.size > 0 and not np.intersect1d(context, query).size


def test_classification_prediction_path_remains_distribution_only() -> None:
    task = prepare_real_task(load_real_dataset("iris"), budget=32, seed=1729, test_limit=8)
    model = _row_model()
    predicted, truth = _model_prediction(model, task)
    assert predicted.shape[0] == len(truth)
    assert predicted.shape[1] == 3
    assert np.isfinite(predicted).all()
    assert np.isfinite(truth).all()
