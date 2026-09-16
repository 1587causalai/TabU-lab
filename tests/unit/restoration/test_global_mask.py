import math

import pytest
import torch

from tabu_lab.models.restoration import (
    ColumnSchema,
    RestorationConfig,
    RestorationModel,
    score_episode,
)
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.encoding import EncoderConfig
from tabu_lab.restoration_joint_fit import TablePlan, _episode
from tabu_lab.restoration_masking import global_query_mask


def _numeric_table(rows=204, columns=8):
    return TablePlan(
        "numeric", tuple(range(rows)),
        tuple(torch.arange(rows, dtype=torch.float64) for _ in range(columns)),
        tuple(ColumnSchema(f"numeric/{index}", "numeric") for index in range(columns)),
        {"numeric": columns}, rows, 52,
    )


@pytest.mark.parametrize("columns", [6, 8, 12, 20, 32])
def test_global_budget_matches_fraction_without_equal_column_quotas(columns):
    table = _numeric_table(columns=columns)
    query, info = global_query_mask(table, 0.025, 1729)
    expected = math.floor(204 * columns * 0.025 + 0.5)
    assert int(query.sum()) == expected
    assert info["query_count"] == expected
    assert info["query_coverage"] == expected / (204 * columns)
    assert info["query_per_column"] == query.sum(0).tolist()
    assert len(set(info["query_per_column"])) > 1
    assert (query.sum(0) <= 202).all()


def test_global_budget_is_reproducible_and_changes_with_seed():
    table = _numeric_table()
    query, info = global_query_mask(table, 0.025, 1729)
    replay, replay_info = global_query_mask(table, 0.025, 1729)
    other, _ = global_query_mask(table, 0.025, 1730)
    assert torch.equal(query, replay)
    assert info == replay_info
    assert not torch.equal(query, other)


def test_support_protection_keeps_singletons_and_randomizes_repeated_classes():
    table = TablePlan(
        "rare", tuple(range(8)),
        (torch.arange(8, dtype=torch.float64), torch.tensor([0, 1, 1, 1, 2, 2, 2, 2]),
         torch.arange(8, dtype=torch.long)),
        (ColumnSchema("rare/numeric", "numeric"),
         ColumnSchema("rare/nominal", "nominal", 3),
         ColumnSchema("rare/ordinal", "ordinal", 8)),
        {"numeric": 1, "nominal": 1, "ordinal": 1}, 8, 2,
    )
    exposed = torch.zeros(8, dtype=torch.bool)
    for seed in range(40):
        query, info = global_query_mask(table, 0.25, seed)
        assert query.sum() == 6
        assert not query[0, 1]  # The sole observation of nominal class 0.
        assert not query[:, 2].any()  # Every ordinal class is a singleton.
        assert set(table.values[1][~query[:, 1]].tolist()) == {0, 1, 2}
        assert info["protected_discrete_cells"] == 11
        assert info["singleton_classes_per_column"] == [0, 1, 8]
        assert info["unmaskable_discrete_classes"] == 9
        assert info["eligible_per_column"] == [8, 5, 0]
        exposed |= query[:, 1]
    # No fixed representative of a repeated class is permanently excluded.
    assert exposed[1:].all()


def test_global_sampler_keeps_two_visible_cells_even_for_small_tables():
    table = _numeric_table(rows=3, columns=2)
    attempts = []
    for seed in range(20):
        query, info = global_query_mask(table, 1 / 3, seed)
        assert query.sum() == 2
        assert query.sum(0).tolist() == [1, 1]
        attempts.append(info["sampling_attempts"])
    assert max(attempts) > 1  # Invalid whole candidates are rejected, never patched.


def test_global_sampler_fails_closed_for_an_impossible_budget():
    table = _numeric_table(rows=3, columns=2)
    with pytest.raises(ValueError, match="exceeds safely maskable capacity"):
        global_query_mask(table, 0.9, 1729)


@pytest.mark.parametrize("fraction", [0, -0.01, 1, 1.1, float("inf"), float("nan"), True])
def test_global_sampler_rejects_invalid_fractions(fraction):
    with pytest.raises(ValueError, match="query_fraction"):
        global_query_mask(_numeric_table(), fraction, 1729)


def test_zero_query_columns_still_score_all_cells_and_have_finite_gradients():
    table = _numeric_table(rows=6, columns=3)
    query, info = global_query_mask(table, 0.025, 1729)
    assert query.sum() == 1
    assert info["query_per_column"].count(0) == 2
    model = RestorationModel(RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    )).double()
    score = score_episode(model, *_episode(table, query, 1732, "cpu"))
    score.loss.backward()
    assert len(score.per_target) == 18
    assert score.by_state["query"]["count"] == 1
    assert torch.isfinite(score.loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
