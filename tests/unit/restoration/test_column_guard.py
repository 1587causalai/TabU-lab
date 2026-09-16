import json
from dataclasses import replace

import pytest
import torch

from tabu_lab import restoration_curriculum_fit as curriculum
from tabu_lab.models.restoration import ColumnSchema, RestorationModel, score_episode
from tabu_lab.models.restoration.end_to_end_checks import small_config
from tabu_lab.restoration_masking import global_query_mask, numeric_tail_protection

GUARD = {"kind": "std_iqr_column", "max_std_iqr_ratio": 4.0}


def _table(values=None):
    values = [0.] * 7 + [100.] if values is None else values
    n = len(values)
    return curriculum.CurriculumTable(
        "mixed", "old120", curriculum.SYNTHETIC,
        (torch.tensor(values, dtype=torch.float64), torch.arange(n, dtype=torch.float64),
         torch.tensor([0] + [1] * (n - 1))),
        (ColumnSchema("heavy", "numeric"), ColumnSchema("normal", "numeric"),
         ColumnSchema("category", "nominal", 2)), tuple(range(n)), 2,
    )


def test_column_guard_excludes_entire_column_and_preserves_global_budget():
    table = _table()
    protected, audit = numeric_tail_protection(table, GUARD, 1e-6)
    assert protected[:, 0].all() and not protected[:, 1:].any()
    assert audit["protected_numeric_columns"] == 1
    assert audit["protected_numeric_column_cells"] == 8
    assert audit["numeric_query_guard"]["std_correction"] == 0
    assert audit["numeric_query_guard"]["scale"] == "full_iqr"
    masks = []
    for seed in range(20):
        query, info = global_query_mask(table, .25, seed, numeric_query_guard=GUARD,
                                       numeric_scale_floor=1e-6)
        assert int(query.sum()) == 6  # All cells, not eligible cells, define the budget.
        assert not query[:, 0].any()
        assert not query[0, 2]  # Discrete singleton still visible.
        assert info["query_protected_numeric_column_cells"] == 0
        assert info["eligible_per_column"][0] == 0
        replay, replay_info = global_query_mask(table, .25, seed, numeric_query_guard=GUARD,
                                               numeric_scale_floor=1e-6)
        assert torch.equal(query, replay) and info == replay_info
        masks.append(query)
    assert any(not torch.equal(masks[0], mask) for mask in masks[1:])


def test_column_guard_uses_population_std_strict_boundary_and_zero_iqr_floor():
    # IQR=0; population std=4 exactly, sample std>4. Equality must stay eligible.
    at_boundary = _table([-8.] + [0.] * 6 + [8.])
    protected, _ = numeric_tail_protection(at_boundary, GUARD, 1.)
    assert not protected[:, 0].any()
    above = _table([-8.01] + [0.] * 6 + [8.01])
    protected, _ = numeric_tail_protection(above, GUARD, 1.)
    assert protected[:, 0].all()
    constant, _ = numeric_tail_protection(_table([42.] * 8), GUARD, 1e-6)
    assert not constant[:, 0].any()


def test_full_iqr_is_not_half_iqr_and_cell_guard_is_not_silently_combined():
    # Full IQR=1, std~2.04: passes full-IQR k4, would fail half-IQR k4.
    table = _table([0., 0., 0., 0., 1., 1., 1., 6.])
    protected, _ = numeric_tail_protection(table, GUARD, 1e-6)
    assert not protected[:, 0].any()
    old, _ = numeric_tail_protection(
        table, {"kind": "median_half_iqr", "max_abs_robust_z": 4.}, 1e-6,
    )
    assert old[-1, 0]
    exposure = False
    for seed in range(40):
        query, _ = global_query_mask(table, .25, seed, numeric_query_guard=GUARD,
                                    numeric_scale_floor=1e-6)
        exposure |= bool(query[-1, 0])
    assert exposure


def test_all_columns_protected_fails_without_shrinking_budget():
    table = _table()
    table = replace(table, values=table.values[:1], schema=table.schema[:1])
    with pytest.raises(ValueError, match="exceeds safely maskable capacity 0"):
        global_query_mask(table, .025, 1, numeric_query_guard=GUARD, numeric_scale_floor=1e-6)


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_column_guard_rejects_invalid_threshold(value):
    with pytest.raises(ValueError, match="max_std_iqr_ratio"):
        numeric_tail_protection(_table(), dict(GUARD, max_std_iqr_ratio=value), 1e-6)


@pytest.mark.parametrize("values", [[0.] * 7 + [float("nan")],
                                  [0.] * 7 + [float("inf")],
                                  [0.] * 7 + [1e308]])
def test_column_guard_fails_closed_for_nonfinite_values_or_statistics(values):
    with pytest.raises(ValueError, match="finite"):
        numeric_tail_protection(_table(values), GUARD, 1e-6)


def test_supervised_rows_ignore_column_guard_and_random_cells_keep_all_targets():
    table = _table()
    config = small_config()
    seeds = {"windows": 1733, "masks": 1731}
    stage = {"name": "old120", "mask": "random_cell", "mask_fraction": .25}
    for evaluation in (False, True):
        episode, info = curriculum._episode_for(
            table, 0, stage, seeds, config, "cpu", evaluation=evaluation,
            numeric_query_guard=GUARD,
        )
        assert episode[0].visible[:, 0].all()
        assert episode[1].targets.shape == (24, 2)
        assert info["protected_numeric_columns"] == 1
    model = RestorationModel(config).double()
    score = score_episode(model, *episode,
                          loss_config=curriculum._loss_config({"objective": {
                              "kind": "state_weighted", "retained_weight": .05,
                              "query_weight": .95}}))
    score.loss.backward()
    assert torch.isfinite(score.loss) and len(score.per_target) == 24
    real = replace(table, kind=curriculum.REAL, values=(table.values[1], table.values[0]),
                   schema=(table.schema[1], table.schema[0]))
    supervised = {"name": "openml12_mixed", "mask": "supervised_row", "mask_fraction": .5}
    for seed in range(20):
        unguarded, _ = curriculum._episode_for(real, seed, supervised, seeds, config, "cpu")
        supplied, info = curriculum._episode_for(
            real, seed, supervised, seeds, config, "cpu", numeric_query_guard=GUARD,
        )
        assert torch.equal(unguarded[0].query, supplied[0].query)
        assert supplied[0].query[:, 1].any() and info["tail_guard_enabled"] is False


def test_reserved_values_do_not_influence_column_selection(tmp_path):
    data = {"values": [[float(i), float(i)] for i in range(8)] + [[1e12, 0.]],
            "splits": {"train": list(range(8)), "test": [8]}}
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    table = curriculum.load_table(path, "reserved", "old120", curriculum.SYNTHETIC)
    protected, _ = numeric_tail_protection(table, GUARD, 1e-6)
    assert not protected.any()
    assert table.train_rows == 8 and table.reserved_rows == 1


def test_column_guard_identity_is_separate_from_legacy_policy(tmp_path):
    prereg = tmp_path / "recipe.yaml"
    prereg.write_text("test")
    legacy = {"model": {}}
    column = {"model": {}, "numeric_query_guard": GUARD}
    assert curriculum._identity(legacy, prereg, [_table()], {}) != (
        curriculum._identity(column, prereg, [_table()], {})
    )
    assert curriculum._numeric_query_guard(legacy)["kind"] == "median_half_iqr"
    assert curriculum._numeric_query_guard(column) == GUARD
