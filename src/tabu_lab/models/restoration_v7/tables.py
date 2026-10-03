"""Typed-fit tables (``tabu.tar.typed-fit-table.1``) as masked V7 tasks.

Column typing follows the V5.5 curriculum loader: numeric columns carry no
domain, discrete columns declare ``domain`` and optionally ``order``; discrete
cells hold integer domain indices.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import torch
from torch import Tensor

from ..restoration.contracts import ColumnSchema, make_episode
from .runner import V7Task

TABLE_SCHEMA = "tabu.tar.typed-fit-table.1"


def _row_ids(rows: Sequence[int], count: int, label: str) -> tuple[int, ...]:
    """Validate physical row addresses before tensor indexing can coerce or alias them."""
    result = []
    for row in rows:
        if isinstance(row, Tensor):
            if (
                row.ndim != 0
                or row.dtype == torch.bool
                or row.is_floating_point()
                or row.is_complex()
            ):
                raise ValueError(f"{label}: row IDs must be integers")
            row = row.item()
        if isinstance(row, bool) or not isinstance(row, Integral):
            raise ValueError(f"{label}: row IDs must be integers")
        if not 0 <= row < count:
            raise ValueError(f"{label}: row ID out of range")
        result.append(int(row))
    return tuple(result)


@dataclass(frozen=True)
class TypedTable:
    name: str
    schema: tuple[ColumnSchema, ...]
    values: tuple[Tensor, ...]  # per column; float64 numeric, int64 domain index
    target: int
    train_rows: tuple[int, ...]
    test_rows: tuple[int, ...]
    sha256: str


def _schema(
    name: str, features: list[dict], keys: list[str] | None = None
) -> tuple[ColumnSchema, ...]:
    if keys is not None and len(keys) != len(features):
        raise ValueError(f"{name}: column_names must match the feature list")
    result = []
    for index, feature in enumerate(features):
        kind = feature.get("kind")
        key = keys[index] if keys is not None else f"{name}/column-{index}"
        if kind == "numeric":
            result.append(ColumnSchema(key, kind))
        elif kind in ("nominal", "ordinal"):
            domain = feature.get("domain")
            if not isinstance(domain, list) or not domain:
                raise ValueError(f"{name}: discrete column {index} needs a declared domain")
            order = feature.get("order")
            result.append(
                ColumnSchema(key, kind, len(domain), tuple(order) if order is not None else None)
            )
        else:
            raise ValueError(f"{name}: unsupported feature kind at column {index}")
    if len({s.key for s in result}) != len(result):
        raise ValueError(f"{name}: column keys must be unique")
    return tuple(result)


def load_typed_table(
    path: str | Path,
    *,
    target: int | None = None,
    expected_sha256: str | None = None,
    name: str | None = None,
) -> TypedTable:
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"dataset digest mismatch: {path}")
    payload = json.loads(raw)
    schema_id = payload.get("schema")
    if schema_id not in (TABLE_SCHEMA, None):
        raise ValueError(f"unsupported table schema: {schema_id}")
    name = name or payload.get("dataset") or Path(path).stem
    rows = payload["values"]
    keys = payload.get("column_names")
    if keys is not None and (
        not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys)
    ):
        raise ValueError(f"{name}: column_names must be nonempty strings")
    schema = _schema(name, payload["features"], keys)
    if not rows or any(len(row) != len(schema) for row in rows):
        raise ValueError(f"{name}: values must be rectangular and match the features")
    if target is None:
        target = len(schema) - 1
    if not 0 <= target < len(schema):
        raise ValueError(f"{name}: target column out of range")
    matrix = torch.tensor(rows, dtype=torch.float64)
    values = []
    for a, column in enumerate(schema):
        if column.kind == "numeric":
            values.append(matrix[:, a].contiguous())
            continue
        index = matrix[:, a]
        if not bool((index == index.round()).all()) or not bool(
            ((index >= 0) & (index < column.domain_size)).all()
        ):
            raise ValueError(f"{name}: column {a} holds values outside its declared domain")
        values.append(index.to(torch.long))
    splits = payload["splits"]
    train_rows = _row_ids(splits["train"], len(rows), f"{name}/train")
    test_rows = _row_ids(splits["test"], len(rows), f"{name}/test")
    if len(set(train_rows)) != len(train_rows) or len(set(test_rows)) != len(test_rows):
        raise ValueError(f"{name}: split rows must be distinct")
    if set(train_rows) & set(test_rows):
        raise ValueError(f"{name}: train and test rows must not overlap")
    return TypedTable(
        name,
        schema,
        tuple(values),
        target,
        train_rows,
        test_rows,
        digest,
    )


def table_task(
    table: TypedTable,
    rows: Sequence[int],
    query: Sequence[int],
    *,
    code_seed: int,
    donor_seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> V7Task:
    """Sub-table on ``rows`` with the target hidden on ``query`` (a subset of ``rows``).

    Every other cell of the chosen rows is visible; rows outside ``rows`` do not
    enter the episode at all.
    """
    return table_mask_task(
        table,
        rows,
        {table.target: query},
        code_seed=code_seed,
        donor_seed=donor_seed,
        device=device,
        dtype=dtype,
    )


def table_mask_task(
    table: TypedTable,
    rows: Sequence[int],
    query_by_column: Mapping[int, Sequence[int]],
    *,
    code_seed: int,
    donor_seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> V7Task:
    """Hide explicit physical row addresses per column before fitting any codec."""
    row_ids = _row_ids(rows, len(table.values[0]), "episode")
    position = {r: i for i, r in enumerate(row_ids)}
    if not row_ids or len(position) != len(row_ids):
        raise ValueError("episode rows must be nonempty and distinct")
    rows = torch.tensor(row_ids, dtype=torch.long)
    observed = torch.ones(len(rows), len(table.schema), dtype=torch.bool)
    mask = torch.zeros_like(observed)
    for column, query in query_by_column.items():
        if type(column) is not int or not 0 <= column < len(table.schema):
            raise ValueError("query column must be an in-range integer")
        query_ids = _row_ids(query, len(table.values[0]), "query")
        if len(set(query_ids)) != len(query_ids) or any(r not in position for r in query_ids):
            raise ValueError("query rows must be a distinct subset of episode rows")
        mask[[position[r] for r in query_ids], column] = True
    if not bool(mask.any()):
        raise ValueError("at least one Query cell is required")
    values = tuple(
        (v[rows].to(dtype) if v.is_floating_point() else v[rows]).to(device) for v in table.values
    )
    inputs, _, truth = make_episode(
        table.schema, values, observed.to(device), mask.to(device), code_seed=code_seed
    )
    return V7Task(inputs, truth, donor_seed)


__all__ = ["TABLE_SCHEMA", "TypedTable", "load_typed_table", "table_mask_task", "table_task"]
