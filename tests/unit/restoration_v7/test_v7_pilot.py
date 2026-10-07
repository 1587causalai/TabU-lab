"""The old120 pilot must distinguish unavailable comparisons from ties."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema
from tabu_lab.models.restoration_v7 import V7Config, V7Model, table_task
from tabu_lab.models.restoration_v7.tables import TypedTable

PILOT_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/local/v7-old120-pilot-20260930/run_v7_old120.py"
)
spec = importlib.util.spec_from_file_location("v7_old120_pilot", PILOT_PATH)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def toy_table():
    x = torch.arange(8, dtype=torch.float64)
    return TypedTable(
        "toy",
        (ColumnSchema("x", "numeric"), ColumnSchema("y", "numeric")),
        (x, 2 * x),
        1,
        tuple(range(6)),
        (6, 7),
        "synthetic",
    )


def test_versus_counts_only_finite_pairs_as_wins_ties_or_losses():
    result = pilot.versus(
        [0.8, 0.1, 0.5, 0.5 + 5e-10, math.nan, 0.5, math.inf, 0.5],
        [0.2, 0.9, 0.5, 0.5, 0.5, math.nan, 0.5, -math.inf],
    )
    assert result == {"win": 1, "tie": 2, "loss": 1, "not_compared": 4}
    with pytest.raises(ValueError):
        pilot.versus([0.5, 0.6], [0.5])


def test_constant_query_r2_is_not_reported_as_a_tie():
    truth = torch.ones(2, dtype=torch.float64)
    score = pilot.value_score("numeric", truth, truth)
    assert math.isnan(score)
    assert pilot.versus([score], [score]) == {
        "win": 0,
        "tie": 0,
        "loss": 0,
        "not_compared": 1,
    }


def test_extreme_finite_values_have_the_same_r2_after_rescaling():
    truth = torch.tensor([1e200, 2e200], dtype=torch.float64)
    prediction = torch.full_like(truth, 1.5e200)
    score = pilot.value_score("numeric", prediction, truth)
    assert score == pytest.approx(0.0, abs=1e-14)
    assert score == pytest.approx(pilot.value_score("numeric", prediction / 1e200, truth / 1e200))
    assert pilot.value_score("numeric", truth, truth) == 1.0


def test_numeric_marginal_avoids_overflowing_the_support_sum():
    table = TypedTable(
        "extreme",
        (ColumnSchema("y", "numeric"),),
        (torch.tensor([1e308, 1.2e308, 1.1e308], dtype=torch.float64),),
        0,
        (0, 1),
        (2,),
        "synthetic",
    )
    task = table_task(table, [0, 1, 2], [2], code_seed=0, donor_seed=0)
    assert pilot.marginal_prediction(task, "numeric").tolist() == pytest.approx([1.1e308])


def test_no_xgboost_evaluation_reports_unavailable_comparison(monkeypatch):
    def unexpected_xgboost(*args, **kwargs):
        pytest.fail("--no-xgboost must not invoke the baseline")

    monkeypatch.setattr(pilot, "xgboost_prediction", unexpected_xgboost)
    tables = [toy_table()]
    banks = pilot.build_banks(tables, 5, 1, "cpu", torch.float64)
    baselines = pilot.baseline_scores(tables, banks, use_xgboost=False)
    config = V7Config(
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        rounds=1,
        coupling_hidden=(16, 16),
    )
    report = pilot.evaluate(V7Model(config).double(), tables, banks, baselines, config.rounds)
    for bank in banks:
        assert report[bank]["final_vs_xgboost"] == {
            "win": 0,
            "tie": 0,
            "loss": 0,
            "not_compared": 1,
        }
        assert report[bank]["macro"]["xgboost"]["numeric"] is None
        assert report[bank]["final_vs_support_marginal"]["not_compared"] == 0


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_macro_does_not_include_unavailable_or_nonfinite_scores(score):
    assert pilot.aggregate([toy_table()], {"method": [score]})["method"]["numeric"] is None
