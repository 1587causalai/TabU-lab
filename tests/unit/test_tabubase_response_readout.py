from __future__ import annotations

import pytest
import torch

from tabu_lab.experiments.tabubase_expanded_synthetic import (
    build_expanded_synthetic_episode,
)
from tabu_lab.experiments.tabubase_response_readout import (
    query_response_objective_loss,
    query_response_readout,
)
from tabu_lab.experiments.tabubase_scale import build_tabubase_scale_model
from tabu_lab.models.components import CellTokenizer
from tabu_lab.training import Objective


def _model() -> torch.nn.Module:
    return build_tabubase_scale_model(
        seed=1729,
        device=torch.device("cpu"),
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
    )


@pytest.mark.parametrize("world_index", range(4))
def test_query_response_loss_and_gradients_match_dense_terminal(world_index: int) -> None:
    """G-D4: cover numeric, binary, ordinal, and categorical responses."""

    episode, truth, _ = build_expanded_synthetic_episode(
        root_seed=1729,
        world_index=world_index,
        context_rows=8,
        query_rows=8,
    )
    dense_model = _model()
    query_model = _model()

    dense_prediction = dense_model._forward_dense(episode, emit_trace=False)
    dense_loss = Objective()(dense_prediction, truth).total
    query_loss = query_response_objective_loss(
        query_model,
        episode,
        truth,
        context_rows=8,
        query_readout_chunk_rows=3,
    )
    torch.testing.assert_close(query_loss, dense_loss, atol=1.0e-6, rtol=1.0e-6)

    dense_loss.backward()
    query_loss.backward()
    dense_parameters = dict(dense_model.named_parameters())
    query_parameters = dict(query_model.named_parameters())
    assert dense_parameters.keys() == query_parameters.keys()
    for name in dense_parameters:
        dense_gradient = dense_parameters[name].grad
        query_gradient = query_parameters[name].grad
        assert (dense_gradient is None) == (query_gradient is None), name
        if dense_gradient is not None and query_gradient is not None:
            torch.testing.assert_close(
                query_gradient,
                dense_gradient,
                atol=5.0e-5,
                rtol=2.0e-4,
                msg=lambda message, parameter=name: f"{parameter}: {message}",
            )


@pytest.mark.parametrize("world_index", (0, 3))
def test_query_response_readout_is_chunk_invariant(world_index: int) -> None:
    episode, _, _ = build_expanded_synthetic_episode(
        root_seed=1729,
        world_index=world_index,
        context_rows=16,
        query_rows=11,
    )
    model = _model().eval()
    with torch.inference_mode():
        rowwise = query_response_readout(
            model,
            episode,
            context_rows=16,
            query_readout_chunk_rows=1,
        )
        complete = query_response_readout(
            model,
            episode,
            context_rows=16,
            query_readout_chunk_rows=64,
        )
    assert torch.equal(rowwise.support_available, complete.support_available)
    for left, right in (
        (rowwise.numeric_values, complete.numeric_values),
        (rowwise.probabilities, complete.probabilities),
        (rowwise.log_probabilities, complete.log_probabilities),
    ):
        assert (left is None) == (right is None)
        if left is not None and right is not None:
            torch.testing.assert_close(left, right, atol=1.0e-6, rtol=1.0e-6)


def test_query_response_path_never_calls_dense_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    episode, truth, _ = build_expanded_synthetic_episode(
        root_seed=1729,
        world_index=0,
        context_rows=16,
        query_rows=8,
    )
    model = _model()

    def forbidden_dense_terminal(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("query-response-only path called the dense terminal")

    monkeypatch.setattr(type(model), "_forward_dense", forbidden_dense_terminal)
    loss = query_response_objective_loss(
        model,
        episode,
        truth,
        context_rows=16,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
        for parameter in model.parameters()
    )
