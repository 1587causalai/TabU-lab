"""Single-table fit probes must hide Query labels without duplicating physical rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema
from tabu_lab.models.restoration_v7.tables import TypedTable

RUNNER_PATH = (
    Path(__file__).resolve().parents[3]
    / "experiments/v7-openml12-single-table-20260930/run_v7_openml12_single.py"
)
spec = importlib.util.spec_from_file_location("v7_openml12_single", RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def table():
    x = torch.arange(8, dtype=torch.float64)
    return TypedTable(
        "toy",
        (ColumnSchema("x", "numeric"), ColumnSchema("y", "numeric")),
        (x, 10 * x),
        1,
        tuple(range(6)),
        (6, 7),
        "synthetic",
    )


def test_train_fit_query_is_removed_from_support_and_each_row_occurs_once(table):
    task = runner.full_task(table, table.train_rows, [1, 4], 7, "cpu", torch.float64)
    assert task.inputs.values[0].tolist() == [0, 2, 3, 5, 1, 4]
    assert task.inputs.visible[:, table.target].tolist() == [True] * 4 + [False] * 2
    assert task.truth.values[table.target][-2:].tolist() == [10, 40]
    assert task.inputs.values[table.target][-2:].tolist() == [0, 0]


def test_disjoint_heldout_query_keeps_the_original_support_order(table):
    task = runner.full_task(table, table.train_rows, table.test_rows, 7, "cpu", torch.float64)
    assert task.inputs.values[0].tolist() == list(range(8))
    assert task.inputs.visible[:, table.target].tolist() == [True] * 6 + [False] * 2


def test_duplicate_query_rows_are_still_rejected(table):
    with pytest.raises(ValueError, match="distinct"):
        runner.full_task(table, table.train_rows, [1, 1], 7, "cpu", torch.float64)
