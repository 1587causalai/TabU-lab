"""Typed-fit table loading and row-restricted V7 task construction."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from tabu_lab.models.restoration_v7 import load_typed_table, table_task


def write_table(path, **overrides):
    payload = {
        "schema": "tabu.tar.typed-fit-table.1",
        "dataset": "toy",
        "values": [[0.5 * i, i % 2, i % 3, float(i)] for i in range(10)],
        "features": [
            {"kind": "numeric", "domain": []},
            {"kind": "nominal", "domain": ["a", "b"]},
            {"kind": "ordinal", "domain": ["lo", "mid", "hi"], "order": [0, 1, 2]},
            {"kind": "numeric", "domain": []},
        ],
        "splits": {"train": list(range(7)), "test": [7, 8, 9]},
    } | overrides
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("schema_id", ["tabu.tar.typed-fit-table.1", "tfm-data.sparse-relevance-table.1"])
def test_loader_types_columns_and_checks_digest(tmp_path, schema_id):
    path = tmp_path / "toy.json"
    digest = write_table(path, schema=schema_id)
    table = load_typed_table(path, target=3, expected_sha256=digest)
    assert [s.kind for s in table.schema] == ["numeric", "nominal", "ordinal", "numeric"]
    assert table.schema[2].domain_size == 3 and table.schema[2].order == (0, 1, 2)
    assert table.values[0].dtype == torch.float64 and table.values[1].dtype == torch.long
    assert table.train_rows == tuple(range(7)) and table.test_rows == (7, 8, 9)
    assert table.schema[0].key == "toy/column-0"
    with pytest.raises(ValueError, match="digest"):
        load_typed_table(path, target=3, expected_sha256="0" * 64)


@pytest.mark.parametrize("schema_id", ["tabu.tar.typed-fit-table.1", "tfm-data.sparse-relevance-table.1"])
def test_loader_rejects_discrete_values_outside_the_domain(tmp_path, schema_id):
    path = tmp_path / "bad.json"
    write_table(path, schema=schema_id, values=[[0.0, 2, 0, 1.0]] * 4)
    with pytest.raises(ValueError, match="declared domain"):
        load_typed_table(path, target=3)


def test_task_restricts_rows_and_hides_only_query_targets(tmp_path):
    path = tmp_path / "toy.json"
    write_table(path)
    table = load_typed_table(path, target=3)
    task = table_task(table, table.train_rows, [1, 4], code_seed=5, donor_seed=6)
    inputs = task.inputs
    assert inputs.visible.shape == (7, 4)
    assert inputs.query.nonzero().tolist() == [[1, 3], [4, 3]]
    assert not bool(inputs.visible[[1, 4], 3].any()) and bool(inputs.visible[:, :3].all())
    assert float(inputs.values[3][1]) == 0.0
    assert task.truth.values[3][[1, 4]].tolist() == [1.0, 4.0]
    with pytest.raises(ValueError, match="subset"):
        table_task(table, table.train_rows, [8], code_seed=0, donor_seed=0)


def test_loader_accepts_openml12_payloads_without_typed_schema(tmp_path):
    payload = {
        "column_names": ["x", "y"],
        "features": [{"kind": "numeric", "domain": []}, {"kind": "numeric", "domain": []}],
        "values": [[float(i), 2.0 * i] for i in range(8)],
        "splits": {"train": list(range(6)), "test": [6, 7]},
        "target_kind": "numeric",
    }
    path = tmp_path / "airfoil.json"
    path.write_text(json.dumps(payload))
    table = load_typed_table(path)
    assert table.target == 1 and table.schema[0].key == "x"
    bad = tmp_path / "other.json"
    bad.write_text(json.dumps(payload | {"schema": "other.v1"}))
    with pytest.raises(ValueError, match="unsupported table schema"):
        load_typed_table(bad)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize(
    ("bad_rows", "message"),
    [
        ([-1], "out of range"),
        ([10], "out of range"),
        ([1.0], "integers"),
        ([True], "integers"),
        ([1, 1], "distinct"),
    ],
)
def test_loader_rejects_invalid_split_addresses(tmp_path, split, bad_rows, message):
    path = tmp_path / "bad-split.json"
    splits = {"train": list(range(7)), "test": [7, 8, 9]}
    splits[split] = bad_rows
    write_table(path, splits=splits)
    with pytest.raises(ValueError, match=message):
        load_typed_table(path)


def test_loader_rejects_split_overlap_but_allows_partial_empty_splits(tmp_path):
    path = tmp_path / "split.json"
    write_table(path, splits={"train": [0, 1, 2, 9], "test": [9]})
    with pytest.raises(ValueError, match="overlap"):
        load_typed_table(path)
    write_table(path, splits={"train": [4, 2, 0], "test": []})
    table = load_typed_table(path)
    assert table.train_rows == (4, 2, 0) and table.test_rows == ()


@pytest.mark.parametrize(
    ("rows", "query", "message"),
    [
        ([-1, 0, 1, 9], [9], "out of range"),
        ([0, 1, 2, 10], [2], "out of range"),
        ([0, 1.5, 2], [2], "integers"),
        ([False, 1, 2], [2], "integers"),
        ([0, 1, 2, 2], [2], "distinct"),
        ([0, 1, 2], [-1], "out of range"),
        ([0, 1, 2], [10], "out of range"),
        ([0, 1, 2], [1.5], "integers"),
        ([0, 1, 2], [True], "integers"),
    ],
)
def test_task_rejects_aliases_and_coercion_before_selecting_rows(tmp_path, rows, query, message):
    path = tmp_path / "toy.json"
    write_table(path)
    table = load_typed_table(path)
    with pytest.raises(ValueError, match=message):
        table_task(table, rows, query, code_seed=0, donor_seed=0)


def test_task_accepts_integer_tensor_addresses_and_preserves_row_order(tmp_path):
    path = tmp_path / "toy.json"
    write_table(path)
    table = load_typed_table(path)
    task = table_task(table, torch.tensor([4, 0, 2]), torch.tensor([2]), code_seed=0, donor_seed=0)
    assert task.truth.values[table.target].tolist() == [4.0, 0.0, 2.0]
    assert task.inputs.query.nonzero().tolist() == [[2, table.target]]
