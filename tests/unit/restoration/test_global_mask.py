import math
from dataclasses import replace

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
from tabu_lab.restoration_masking import global_query_mask, numeric_tail_protection

GUARD = {"kind": "median_half_iqr", "max_abs_robust_z": 8.0}


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


def _guard_table(values):
    return replace(_numeric_table(rows=len(values), columns=1),
                   values=(torch.tensor(values, dtype=torch.float64),))


def test_tail_guard_strict_boundary_floor_and_count_only_audit():
    table = _guard_table([0] * 12 + [-8, 8, -8.125, 8.125])
    protected, audit = numeric_tail_protection(table, GUARD, 1.0)
    assert protected[:, 0].tolist() == [False] * 14 + [True, True]
    assert audit["protected_numeric_tail_cells"] == 2
    assert audit["protected_numeric_tail_per_column"] == [2]
    assert audit["numeric_query_guard"]["scale_floor"] == 1
    assert audit["numeric_query_guard"]["comparison"] == "strictly_greater"
    # The artifact contains policy/counts, not fitted medians/scales or raw values.
    assert set(audit) == {
        "protected_numeric_tail_cells", "protected_numeric_tail_per_column", "numeric_query_guard"
    }
    queried_boundary = torch.zeros(2, dtype=torch.bool)
    for seed in range(20):
        query, info = global_query_mask(table, .5, seed, numeric_query_guard=GUARD,
                                        numeric_scale_floor=1.0)
        assert query.sum() == 8
        assert not (query & protected).any()
        assert info["query_numeric_tail_cells"] == 0
        queried_boundary |= query[12:14, 0]
    assert queried_boundary.all()


def test_tail_guard_zero_iqr_and_all_constant_columns():
    constant = _guard_table([4] * 16)
    protected, info = numeric_tail_protection(constant, GUARD, 1e-6)
    assert not protected.any()
    assert info["protected_numeric_tail_cells"] == 0
    query, _ = global_query_mask(constant, .025, 1729, numeric_query_guard=GUARD,
                                numeric_scale_floor=1e-6)
    assert query.sum() == 1
    impulse = _guard_table([0] * 15 + [9])
    strict, _ = numeric_tail_protection(impulse, GUARD, 1.0)
    larger_floor, _ = numeric_tail_protection(impulse, GUARD, 2.0)
    assert strict[-1, 0]
    assert not larger_floor.any()


@pytest.mark.parametrize("factor", [-2.0, 2.0])
def test_tail_guard_affine_units_are_invariant_when_scale_floor_is_scaled(factor):
    table = _guard_table([0] * 12 + [-8, 8, -8.125, 8.125])
    transformed = replace(table, values=(table.values[0] * factor + 16,))
    first, _ = global_query_mask(table, .5, 1729, numeric_query_guard=GUARD,
                                numeric_scale_floor=1.0)
    moved, _ = global_query_mask(transformed, .5, 1729, numeric_query_guard=GUARD,
                                numeric_scale_floor=abs(factor))
    assert torch.equal(first, moved)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_tail_guard_rejects_nonfinite_numeric_values(bad):
    with pytest.raises(ValueError, match="finite training values"):
        numeric_tail_protection(_guard_table([0, 1, bad]), GUARD, 1e-6)


@pytest.mark.parametrize("threshold", [0, -1, float("nan"), float("inf"), True])
def test_tail_guard_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="max_abs_robust_z"):
        numeric_tail_protection(_numeric_table(), dict(GUARD, max_abs_robust_z=threshold), 1e-6)


@pytest.mark.parametrize("floor", [0, -1, float("nan"), float("inf"), True, None])
def test_tail_guard_rejects_invalid_scale_floor(floor):
    with pytest.raises(ValueError, match="scale_floor"):
        numeric_tail_protection(_numeric_table(), GUARD, floor)


def test_tail_guard_uses_median_half_iqr_and_preserves_discrete_policy():
    numeric = [-100, -4, -3, -2, -1, 0, 1, 2, 3, 4, 100]
    table = replace(
        _guard_table(numeric),
        values=(torch.tensor(numeric, dtype=torch.float64), torch.arange(11)),
        schema=(ColumnSchema("n", "numeric"), ColumnSchema("d", "nominal", 11)),
        kind_counts={"numeric": 1, "nominal": 1},
    )
    protected, _ = numeric_tail_protection(table, GUARD, 1e-6)
    assert protected[:, 0].tolist() == [True] + [False] * 9 + [True]
    assert not protected[:, 1].any()
    query, info = global_query_mask(table, .025, 1729, numeric_query_guard=GUARD,
                                    numeric_scale_floor=1e-6)
    assert query.sum() == 1
    assert not query[:, 1].any()
    assert info["protected_discrete_cells"] == 11
    assert info["protected_numeric_tail_per_column"] == [2, 0]


def test_tail_guard_fails_closed_without_reducing_global_budget():
    table = _guard_table([-9] * 3 + [0] * 10 + [9] * 3)
    with pytest.raises(ValueError, match="exceeds safely maskable capacity"):
        global_query_mask(table, .8, 1729, numeric_query_guard=GUARD, numeric_scale_floor=1.0)


def test_tail_guard_protected_cells_remain_visible_and_in_all_cell_loss():
    table = _guard_table([-100, -4, -3, -2, -1, 0, 1, 2, 3, 4, 100])
    query, _ = global_query_mask(table, .2, 1729, numeric_query_guard=GUARD,
                                numeric_scale_floor=1e-6)
    episode = _episode(table, query, 1732, "cpu")
    assert episode[0].visible[0, 0] and episode[0].visible[-1, 0]
    assert episode[1].targets.shape[0] == 11
    model = RestorationModel(RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    )).double()
    score = score_episode(model, *episode)
    assert score.by_state["query"]["count"] == 2
    assert score.by_state["retained"]["count"] == 9
    score.loss.backward()
    assert torch.isfinite(score.loss)


def test_legacy_global_masks_keep_identical_rng_and_metadata_with_guard_disabled():
    table = _guard_table([-100, -4, -3, -2, -1, 0, 1, 2, 3, 4, 100])
    expected, info = global_query_mask(table, .2, 1729)
    explicit_off, explicit_info = global_query_mask(table, .2, 1729, numeric_query_guard=None)
    assert torch.equal(expected, explicit_off)
    assert info == explicit_info
    assert "numeric_query_guard" not in info
    assert "protected_numeric_tail_cells" not in info
