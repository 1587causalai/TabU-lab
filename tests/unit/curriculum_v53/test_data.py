import hashlib
import json
from dataclasses import replace

import pytest
import torch

from tabu_lab.curriculum_v53.data import build_episode, data_fingerprint, load_table

SEEDS = {name: i + 11 for i, name in enumerate(
    ("model", "order", "masks", "codes", "windows", "evaluation")
)}


def write_table_fixture(tmp_path, *, values=None, features=None, splits=None, **entry_fields):
    """Small legacy table fixture, usable by curriculum runner tests."""
    data = {
        "values": values if values is not None else [[float(i), i % 2] for i in range(12)],
        "features": features if features is not None else [
            {"kind": "numeric", "domain": []}, {"kind": "nominal", "domain": [0, 1]}
        ],
        "splits": splits if splits is not None else {
            "train": [0, 1, 2, 3, 4, 5, 6, 7], "validation": [8, 9], "test": [10, 11]
        },
    }
    path = tmp_path / "table.json"
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    entry = {"id": "small", "cohort": "fixture", "kind": "synthetic",
             "path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), **entry_fields}
    return entry, data


def test_strict_load_preserves_only_training_tensors_and_identity(tmp_path):
    entry, _ = write_table_fixture(tmp_path, target_column=0, role="probe")
    table = load_table(entry, tmp_path)
    assert table.row_ids == tuple(range(8)) and table.reserved_rows == 4
    assert table.target_column == 0 and table.role == "probe"
    assert table.values[0].tolist() == list(range(8))
    assert table.holdout_values["test"][0].tolist() == [10, 11]
    json.dumps(table.summary(), allow_nan=False)
    assert data_fingerprint([table]) == data_fingerprint([replace(table, path=tmp_path / "moved")])
    assert data_fingerprint([table]) != data_fingerprint([replace(table, role="train")])
    with pytest.raises(ValueError, match="digest"):
        load_table({**entry, "sha256": "0" * 64}, tmp_path)


@pytest.mark.parametrize("splits", [
    {"train": [0, 1, 1], "test": list(range(2, 12))},
    {"train": [0, 1, True], "test": list(range(3, 12))},
    {"train": [0, 1, 2], "test": list(range(2, 12))},
    {"train": [0, 1, 2], "test": list(range(4, 12))},
    {"train": [0, 1], "test": list(range(2, 12))},
    {"train": [0, 1, 2], "test": list(range(3, 13))},
])
def test_rejects_ambiguous_splits(tmp_path, splits):
    entry, _ = write_table_fixture(tmp_path, splits=splits)
    with pytest.raises(ValueError):
        load_table(entry, tmp_path)


@pytest.mark.parametrize("column,value", [(0, True), (0, float("nan")), (1, 1.0), (1, 2)])
def test_validates_even_reserved_values(tmp_path, column, value):
    values = [[float(i), i % 2] for i in range(12)]
    values[-1][column] = value
    entry, _ = write_table_fixture(tmp_path, values=values)
    with pytest.raises(ValueError):
        load_table(entry, tmp_path)


@pytest.mark.parametrize("kind", ["random_cell", "supervised_row"])
def test_local_rng_repeatability_class_protection_and_training_isolation(tmp_path, kind):
    entry, _ = write_table_fixture(tmp_path, window_rows=6)
    table = load_table(entry, tmp_path)
    recipe = {"kind": kind, "fraction": 0.25}
    state = torch.get_rng_state().clone()
    first = build_episode(table, recipe, 1, SEEDS, "cpu")
    repeat = build_episode(table, recipe, 1, SEEDS, "cpu")
    assert torch.equal(torch.get_rng_state(), state)
    assert first[3] == repeat[3]
    assert torch.equal(first[0].query, repeat[0].query)
    assert first[3]["numeric_query_guard"]["kind"] == (
        "std_iqr_column" if kind == "random_cell" else "none"
    )
    assert set(first[3]["row_ids"]) <= set(table.row_ids)
    assert min(first[3]["visible_per_column"]) >= 2
    for a, spec in enumerate(table.schema):
        if spec.kind != "numeric":
            visible = set(first[0].values[a][first[0].visible[:, a]].tolist())
            assert set(first[2].values[a].tolist()) <= visible
    poisoned = replace(table, holdout_values={"test": None, "validation": None})
    assert build_episode(poisoned, recipe, 1, SEEDS, "cpu")[3] == first[3]
    evaluation = build_episode(table, recipe, 1, SEEDS, "cpu", evaluation=True)
    assert first[3]["code_seed"] != evaluation[3]["code_seed"]
    assert first[3]["window_seed"] != evaluation[3]["window_seed"]


def test_current_random_cell_default_excludes_whole_tail_column(tmp_path):
    entry, _ = write_table_fixture(
        tmp_path, values=[[float(i) if i != 19 else 10000.0, float(i)] for i in range(22)],
        features=[{"kind": "numeric"}, {"kind": "numeric"}],
        splits={"train": list(range(20)), "test": [20, 21]},
    )
    table = load_table(entry, tmp_path)
    recipe = {"kind": "random_cell", "fraction": 0.2}
    inputs, _, _, audit = build_episode(table, recipe, 0, SEEDS, "cpu")
    assert not inputs.query[:, 0].any()
    assert inputs.visible[:, 0].all()
    assert int(inputs.query.sum()) == 8
    assert audit["protected_numeric_columns"] == 1
    assert audit["protected_numeric_column_cells"] == 20
    recipe["numeric_query_guard"] = {"kind": "none"}
    unguarded = build_episode(table, recipe, 0, SEEDS, "cpu")
    assert unguarded[3]["numeric_query_guard"] == {"kind": "none"}
    assert unguarded[0].query[:, 0].any()


def test_supervised_does_not_drop_unsupported_queries_or_reduce_budget(tmp_path):
    entry, _ = write_table_fixture(
        tmp_path, values=[[float(i), i] for i in range(6)],
        features=[{"kind": "numeric"}, {"kind": "nominal", "domain": list(range(6))}],
        splits={"train": [0, 1, 2, 3], "test": [4, 5]},
    )
    table = load_table(entry, tmp_path)
    recipe = {"kind": "supervised_row", "fraction": 0.25}
    with pytest.raises(ValueError, match="capacity"):
        build_episode(table, recipe, 0, SEEDS, "cpu")
    with pytest.raises(ValueError, match="no-answer-code"):
        build_episode(table, recipe, 0, SEEDS, "cpu", evaluation=True, partition="test")


def test_holdout_target_is_zeroed_forward_and_all_rows_are_scored(tmp_path):
    entry, _ = write_table_fixture(tmp_path, target_column=0)
    table = load_table(entry, tmp_path)
    recipe = {"kind": "supervised_row", "fraction": 0.25}
    inputs, request, truth, info = build_episode(
        table, recipe, 0, SEEDS, "cpu", evaluation=True, partition="test"
    )
    assert info["row_ids"] == [*range(8), 10, 11]
    assert info["query_addresses"] == [[10, 0], [11, 0]]
    assert inputs.values[0][-2:].tolist() == [0, 0]
    assert truth.values[0][-2:].tolist() == [10, 11]
    assert inputs.visible[-2:, 1].all() and info["transductive"]
    assert len(request.targets) == 20
    with pytest.raises(ValueError, match="evaluation=True"):
        build_episode(table, recipe, 0, SEEDS, "cpu", partition="test")
    with pytest.raises(ValueError, match="window_rows"):
        build_episode(replace(table, window_rows=5), recipe, 0, SEEDS, "cpu",
                      evaluation=True, partition="test")


def test_numeric_guard_is_explicit_and_uses_registered_epsilon(tmp_path):
    entry, _ = write_table_fixture(tmp_path, target_column=0)
    table = load_table(entry, tmp_path)
    recipe = {"kind": "supervised_row", "fraction": 0.25,
              "numeric_query_guard": {"kind": "median_half_iqr", "max_abs_robust_z": 1.5}}
    info = build_episode(table, recipe, 0, SEEDS, "cpu", epsilon=1e-4)[3]
    assert info["numeric_query_guard"]["scale_floor"] == 1e-4
    assert info["protected_numeric_tail_cells"] > 0
    with pytest.raises(ValueError, match="filter"):
        build_episode(table, recipe, 0, SEEDS, "cpu", evaluation=True, partition="test")


@pytest.mark.parametrize("codec_version", ["unit_gaussian_v2", "constant_weight_v1"])
def test_declared_ordinal_domain_allows_unseen_query_ranks(tmp_path, codec_version):
    entry, _ = write_table_fixture(
        tmp_path, values=[[float(i), i] for i in range(6)],
        features=[{"kind": "numeric"}, {"kind": "ordinal", "domain": list(range(6))}],
        splits={"train": [0, 1, 2, 3], "test": [4, 5]},
    )
    table = load_table(entry, tmp_path)
    recipe = {"kind": "supervised_row", "fraction": .25}
    inputs, _, truth, info = build_episode(table, recipe, 0, SEEDS, "cpu",
                                         codec_version=codec_version)
    assert info["query_count"] == 1
    assert info["protected_discrete_cells"] == 0
    assert not info["support_policy"]["protect_ordinal_classes"]
    assert len(set(truth.values[1].tolist()) -
               set(inputs.values[1][inputs.visible[:, 1]].tolist())) == 1
    heldout = build_episode(table, recipe, 0, SEEDS, "cpu", evaluation=True, partition="test",
                            codec_version=codec_version)
    assert heldout[3]["query_addresses"] == [[4, 1], [5, 1]]
    assert heldout[0].values[1][-2:].tolist() == [0, 0]
    with pytest.raises(ValueError, match="capacity"):
        build_episode(table, recipe, 0, SEEDS, "cpu", codec_version="legacy_v53")
    with pytest.raises(ValueError, match="no-answer-code"):
        build_episode(table, recipe, 0, SEEDS, "cpu", evaluation=True, partition="test",
                      codec_version="legacy_v53")


def test_numeric_support_rejects_constant_and_resamples_degenerate_masks(tmp_path):
    entry, _ = write_table_fixture(tmp_path, target_column=0)
    table = load_table(entry, tmp_path)
    recipe = {"kind": "supervised_row", "fraction": .5}
    constant = replace(table, values=(torch.zeros_like(table.values[0]), table.values[1]))
    with pytest.raises(ValueError, match="no-valid-episode"):
        build_episode(constant, recipe, 0, SEEDS, "cpu")
    values = torch.zeros_like(table.values[0])
    values[0] = 1
    repeated = replace(table, values=(values, table.values[1]))
    attempts = []
    for index in range(20):
        inputs, _, _, info = build_episode(repeated, recipe, index, SEEDS, "cpu")
        assert info["query_count"] == 4
        assert len(inputs.values[0][inputs.visible[:, 0]].unique()) == 2
        attempts.append(info["sampling_attempts"])
    assert max(attempts) > 1
