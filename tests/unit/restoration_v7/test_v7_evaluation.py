"""Original-value metric boundaries for V7 column codecs."""

from __future__ import annotations

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput, make_episode
from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    V7Task,
    build_g64_codec,
    evaluate_task,
)
from tabu_lab.models.restoration_v7.codec import G64ColumnCodec
from tabu_lab.models.restoration_v7.evaluation import value_metrics


def ordinal_codec(order=None):
    schema = (ColumnSchema("rank", "ordinal", domain_size=3, order=order),)
    values = (torch.tensor([0, 1, 2]),)
    visible = torch.ones(3, 1, dtype=torch.bool)
    inputs = RestorationInput(schema, values, visible, torch.zeros_like(visible), 3)
    return build_g64_codec(inputs).columns[0]


def test_ordinal_metric_uses_the_declared_rank_instead_of_category_indices():
    column = ordinal_codec(order=(2, 0, 1))
    assert torch.equal(column.rank_positions, torch.tensor([1, 2, 0]))
    metrics = value_metrics(column, torch.tensor([0]), torch.tensor([2]))
    assert metrics == {"accuracy": 0.0, "mean_rank_distance": 1.0}
    metrics = value_metrics(column, torch.tensor([0, 1]), torch.tensor([2, 2]))
    assert metrics["mean_rank_distance"] == 1.5


def test_ordinal_codec_former_constructor_defaults_to_identity_order():
    original = ordinal_codec()
    # The additional rank field is optional and leaves the old positional API intact.
    column = G64ColumnCodec(
        "ordinal",
        original.base,
        original.direction,
        original.categories,
        original.codes,
    )
    assert column.rank_positions is None and column.domain_size is None
    metrics = value_metrics(column, torch.tensor([0]), torch.tensor([2]))
    assert metrics == {"accuracy": 0.0, "mean_rank_distance": 2.0}


def numeric_codec(scale=1.0):
    return G64ColumnCodec("numeric", torch.zeros(64), torch.ones(64) / 8, None, None, scale=scale)


@pytest.mark.parametrize(
    ("prediction", "truth", "scale", "mae", "rmse"),
    (
        ([1e200, -1e200], [0.0, 0.0], 1e200, 1e200, 1e200),
        ([1.7e308, 0.0, 0.0, 0.0], [-1.7e308, 0.0, 0.0, 0.0], 1.7e308, 8.5e307, 1.7e308),
        ([1e-200], [0.0], 1e-200, 1e-200, 1e-200),
    ),
)
def test_numeric_metrics_avoid_unnecessary_intermediate_overflow_and_underflow(
    prediction, truth, scale, mae, rmse
):
    metrics = value_metrics(
        numeric_codec(scale),
        torch.tensor(prediction, dtype=torch.float64),
        torch.tensor(truth, dtype=torch.float64),
    )
    assert metrics["mae"] == pytest.approx(mae, rel=1e-14, abs=0)
    assert metrics["rmse"] == pytest.approx(rmse, rel=1e-14, abs=0)
    assert metrics["standardized_rmse"] == pytest.approx(rmse / scale)


def test_unrepresentable_numeric_metrics_fail_explicitly():
    with pytest.raises(FloatingPointError, match="V7 numeric metrics"):
        value_metrics(
            numeric_codec(1e308),
            torch.tensor([1.7e308], dtype=torch.float64),
            torch.tensor([-1.7e308], dtype=torch.float64),
        )


def test_numeric_evaluation_reuses_the_finite_codec_mean_for_its_marginal_baseline():
    schema = (ColumnSchema("y", "numeric"),)
    values = (torch.full((3,), 1e308, dtype=torch.float64),)
    observed = torch.ones(3, 1, dtype=torch.bool)
    query = torch.tensor([[False], [False], [True]])
    inputs, _, truth = make_episode(schema, values, observed, query, code_seed=3)
    config = V7Config(
        codec="G64",  # This numerical-extreme fixture uses four dimensions.
        code_dim=4,
        backbone=dict(width=4, layers=1, heads=2, ff_width=8, slots=2),
        rounds=1,
        coupling_hidden=(8,),
    )
    report = evaluate_task(V7Model(config).double(), V7Task(inputs, truth, donor_seed=1))
    assert report.baselines["support_marginal"] == {
        "mae": 0.0,
        "rmse": 0.0,
        "standardized_rmse": 0.0,
        "code_mean_loss": 0.0,
    }
