"""Frozen typed tables and deterministic, split-isolated V5.3 episodes.

Training tensors contain only the registered training rows. Reserved tensors
live in a separate field and are touched only by explicit reserved evaluation.
Windows are deterministic samples, not an assertion of full row coverage.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

from tabu_lab.models.restoration_v53 import (
    CODEC_VERSIONS,
    DEFAULT_CODEC_VERSION,
    ColumnSchema,
    make_episode,
)
from tabu_lab.restoration_masking import (
    default_v53_query_guard,
    global_query_mask,
    validate_numeric_query_guard,
)


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _seed(seed: int, *identity) -> int:
    if type(seed) is not int:
        raise ValueError("seed streams must contain integers")
    return int.from_bytes(hashlib.sha256(_canonical([seed, *identity])).digest()[:8], "little")


@dataclass(frozen=True)
class Table:
    name: str
    cohort: str
    kind: str
    schema: tuple[ColumnSchema, ...]
    values: tuple[torch.Tensor, ...]
    row_ids: tuple[int, ...]
    reserved_rows: int
    window_rows: int | None
    target_column: int
    role: str
    path: Path
    digest: str
    holdout_values: dict[str, tuple[torch.Tensor, ...]]
    holdout_row_ids: dict[str, tuple[int, ...]]

    @property
    def id(self) -> str:
        return self.name

    @property
    def sha256(self) -> str:
        return self.digest

    @property
    def train_rows(self) -> int:
        return len(self.row_ids)

    @property
    def width(self) -> int:
        return len(self.schema)

    def summary(self) -> dict:
        return table_summary(self)


def table_summary(table: Table) -> dict:
    """JSON-safe identity and split counts, without raw values."""
    return {
        "id": table.name, "cohort": table.cohort, "kind": table.kind, "role": table.role,
        "path": str(table.path), "sha256": table.digest, "train_rows": table.train_rows,
        "reserved_rows": table.reserved_rows, "width": table.width,
        "validation_rows": len(table.holdout_row_ids["validation"]),
        "test_rows": len(table.holdout_row_ids["test"]),
        "window_rows": table.window_rows, "target_column": table.target_column,
        "schema": [{"key": s.key, "kind": s.kind, "domain_size": s.domain_size,
                    "order": list(s.order) if s.order is not None else None}
                   for s in table.schema],
        "train_row_ids": list(table.row_ids),
        "holdout_row_ids": {k: list(v) for k, v in table.holdout_row_ids.items()},
    }


def data_fingerprint(tables) -> str:
    """Bind order, table bytes, split addresses and data-recipe metadata.

    Absolute relocation paths do not change data identity.
    """
    summaries = []
    for table in tables:
        summary = table_summary(table)
        del summary["path"]
        summaries.append(summary)
    return hashlib.sha256(_canonical(summaries)).hexdigest()


def _schema(name: str, features, width: int) -> tuple[ColumnSchema, ...]:
    if not isinstance(features, list) or len(features) != width:
        raise ValueError(f"{name}: features must declare every column")
    result = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("kind") not in (
            "numeric", "nominal", "ordinal"
        ):
            raise ValueError(f"{name}: unsupported feature kind at column {index}")
        kind = feature["kind"]
        domain = feature.get("domain")
        order = feature.get("order")
        key = feature.get("key", f"{name}/column-{index}")
        if kind == "numeric":
            if domain not in (None, []) or order is not None:
                raise ValueError(f"{name}: numeric columns cannot declare a domain or order")
            result.append(ColumnSchema(key, kind))
        else:
            if not isinstance(domain, list) or not domain:
                raise ValueError(f"{name}: discrete columns need a nonempty declared domain")
            if any(not isinstance(x, (str, int, float, bool)) for x in domain):
                raise ValueError(f"{name}: domain labels must be scalar JSON values")
            if len({_canonical(x) for x in domain}) != len(domain):
                raise ValueError(f"{name}: domain labels must be unique")
            result.append(ColumnSchema(key, kind, len(domain), order))
    if len({s.key for s in result}) != len(result):
        raise ValueError(f"{name}: column keys must be unique")
    return tuple(result)


def _split_rows(splits, rows: int, name: str) -> dict[str, tuple[int, ...]]:
    if (not isinstance(splits, dict) or not {"train", "test"} <= set(splits)
            or set(splits) - {"train", "validation", "test"}):
        raise ValueError(f"{name}: explicit train/test splits and optional validation required")
    result, seen = {}, set()
    for partition in ("train", "validation", "test"):
        addresses = splits.get(partition, [])
        if (not isinstance(addresses, list)
                or any(type(r) is not int or not 0 <= r < rows for r in addresses)
                or len(set(addresses)) != len(addresses)):
            raise ValueError(f"{name}: split row addresses must be unique in-bounds integers")
        if seen.intersection(addresses):
            raise ValueError(f"{name}: train/validation/test rows must be disjoint")
        seen.update(addresses)
        result[partition] = tuple(addresses)
    if seen != set(range(rows)):
        raise ValueError(f"{name}: splits must cover every original row")
    if len(result["train"]) < 3:
        raise ValueError(f"{name}: training split needs at least three rows")
    return result


def load_table(entry: dict, base_dir: Path) -> Table:
    """Load legacy values/features/splits JSON after validating its byte digest."""
    required = {"id", "path", "sha256", "cohort", "kind"}
    allowed = required | {"window_rows", "target_column", "role"}
    if not isinstance(entry, dict) or not required <= set(entry) or set(entry) - allowed:
        raise ValueError("table entry needs id/path/sha256/cohort/kind and known optional fields")
    for field in ("id", "path", "cohort"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"table entry {field} must be a nonempty string")
    if entry["kind"] not in ("synthetic", "real"):
        raise ValueError("table kind must be synthetic or real")
    role = entry.get("role", "train")
    if role not in ("train", "probe"):
        raise ValueError("table role must be train or probe")
    relative = Path(entry["path"])
    if relative.is_absolute():
        raise ValueError("table paths must be relative to the manifest directory")
    digest = entry["sha256"]
    if (not isinstance(digest, str) or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("table sha256 must be a lowercase hexadecimal SHA-256")
    path = (Path(base_dir) / relative).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"dataset digest mismatch: {entry['id']}")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("table JSON must be a mapping")
    values = data.get("values")
    if (not isinstance(values, list) or not values or not isinstance(values[0], list)
            or not values[0]):
        raise ValueError("values must be a nonempty rectangular matrix")
    width = len(values[0])
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError("values must be a nonempty rectangular matrix")
    schema = _schema(entry["id"], data.get("features"), width)
    splits = _split_rows(data.get("splits"), len(values), entry["id"])
    target = entry.get("target_column", width - 1)
    if type(target) is not int or not 0 <= target < width:
        raise ValueError("target_column must be an in-bounds column index")
    window = entry.get("window_rows")
    if window is not None and (type(window) is not int or window < 3):
        raise ValueError("window_rows must be an integer at least three or null")
    columns = []
    for a, spec in enumerate(schema):
        raw_column = [row[a] for row in values]
        if spec.kind == "numeric":
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in raw_column):
                raise ValueError("numeric values must be finite real numbers")
            dtype = torch.float64
        else:
            if any(type(v) is not int or not 0 <= v < spec.domain_size for v in raw_column):
                raise ValueError("discrete values must be integer declared-domain indices")
            dtype = torch.long
        columns.append(torch.tensor(raw_column, dtype=dtype))
    partitions = {p: tuple(c[list(ids)].clone() for c in columns) for p, ids in splits.items()}
    return Table(entry["id"], entry["cohort"], entry["kind"], schema, partitions["train"],
                 splits["train"], len(values) - len(splits["train"]), window, target, role,
                 path, digest, {p: partitions[p] for p in ("validation", "test")},
                 {p: splits[p] for p in ("validation", "test")})


@dataclass(frozen=True)
class _MaskTable:
    name: str
    values: tuple[torch.Tensor, ...]
    schema: tuple[ColumnSchema, ...]

    @property
    def train_rows(self):
        return len(self.values[0])


def _recipe(recipe: dict) -> tuple[str, float, dict]:
    if (not isinstance(recipe, dict) or not {"kind", "fraction"} <= set(recipe)
            or set(recipe) - {"kind", "fraction", "numeric_query_guard"}):
        raise ValueError("episode recipe needs kind/fraction and optional numeric_query_guard")
    kind, fraction = recipe["kind"], recipe["fraction"]
    if kind not in ("random_cell", "supervised_row"):
        raise ValueError("episode kind must be random_cell or supervised_row")
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 < fraction < 1):
        raise ValueError("episode fraction must be strictly between zero and one")
    guard = recipe.get("numeric_query_guard", default_v53_query_guard(kind))
    if guard != {"kind": "none"}:
        guard = validate_numeric_query_guard(guard)
    return kind, float(fraction), dict(guard)


def build_episode(table: Table, recipe: dict, index: int, seeds: dict, device: str, *,
                  evaluation: bool = False, partition: str = "train", epsilon: float = 1e-6,
                  codec_version: str = DEFAULT_CODEC_VERSION):
    """Build one episode; return inputs/request/truth plus original-address audit.

    Supervised training uses an exact rounded Query count and protects every
    nominal class before sampling. The legacy codec also protects ordinal
    classes; the default ordinal codec uses the complete declared rank domain.
    Reserved evaluation is transductive: all
    non-target heldout features are visible. Its target truth is scorer-only.
    """
    kind, fraction, guard = _recipe(recipe)
    if codec_version not in CODEC_VERSIONS:
        raise ValueError("unknown V5.3 codec_version")
    support_policy = {
        "protect_ordinal_classes": codec_version == "legacy_v53",
        "require_numeric_diversity": codec_version != "legacy_v53",
    }
    if type(index) is not int or index < 0:
        raise ValueError("episode index must be a nonnegative integer")
    if partition not in ("train", "validation", "test"):
        raise ValueError("partition must be train, validation or test")
    if partition != "train" and not evaluation:
        raise ValueError("reserved rows require explicit evaluation=True")
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    namespace = ["evaluation" if evaluation else "training", table.name, partition, index]
    evaluation_seed = seeds["evaluation"] if evaluation else 0
    mask_seed = _seed(seeds["masks"], *namespace, evaluation_seed, "mask")
    code_seed = _seed(seeds["codes"], *namespace, evaluation_seed, "code")
    window_seed = _seed(seeds["windows"], *namespace, evaluation_seed, "window")
    audit = {}
    if partition == "train":
        size = min(table.window_rows or table.train_rows, table.train_rows)
        if size == table.train_rows:
            selected = list(range(size))
        else:
            generator = torch.Generator(device="cpu").manual_seed(window_seed)
            selected = torch.randperm(table.train_rows, generator=generator)[:size].tolist()
        values = tuple(v[selected] for v in table.values)
        row_ids = tuple(table.row_ids[r] for r in selected)
        if kind == "random_cell":
            plan = _MaskTable(table.name, values, table.schema)
            query, audit = global_query_mask(
                plan, fraction, mask_seed,
                numeric_query_guard=None if guard["kind"] == "none" else guard,
                numeric_scale_floor=None if guard["kind"] == "none" else epsilon,
                **support_policy,
            )
        else:
            a = table.target_column
            plan = _MaskTable(table.name, (values[a],), (table.schema[a],))
            target_query, audit = global_query_mask(
                plan, fraction, mask_seed,
                numeric_query_guard=None if guard["kind"] == "none" else guard,
                numeric_scale_floor=None if guard["kind"] == "none" else epsilon,
                **support_policy,
            )
            query = torch.zeros(size, table.width, dtype=torch.bool)
            query[:, a] = target_query[:, 0]
            for key in ("eligible_per_column", "capacity_per_column",
                        "singleton_classes_per_column", "protected_numeric_tail_per_column"):
                if key in audit:
                    audit[key] = [audit[key][0] if column == a else 0
                                  for column in range(table.width)]
        if "numeric_query_guard" in audit:
            audit["numeric_query_guard"]["reference_scope"] = (
                "selected training window before masking" if size < table.train_rows
                else "all training rows before masking"
            )
    else:
        if kind != "supervised_row":
            raise ValueError("reserved evaluation supports supervised_row only")
        if table.window_rows is not None:
            raise ValueError(
                "reserved evaluation does not support window_rows; register full context"
            )
        if guard["kind"] != "none":
            raise ValueError("reserved evaluation cannot filter Query targets with a numeric guard")
        if not table.holdout_row_ids[partition]:
            raise ValueError(f"{table.name}: {partition} split is empty")
        values = tuple(torch.cat((train, heldout)) for train, heldout in
                       zip(table.values, table.holdout_values[partition], strict=True))
        row_ids = table.row_ids + table.holdout_row_ids[partition]
        query = torch.zeros(len(row_ids), table.width, dtype=torch.bool)
        query[table.train_rows:, table.target_column] = True
        spec = table.schema[table.target_column]
        if spec.kind == "nominal" or (spec.kind == "ordinal" and codec_version == "legacy_v53"):
            support = set(table.values[table.target_column].tolist())
            labels = set(table.holdout_values[partition][table.target_column].tolist())
            if not labels <= support:
                raise ValueError("no-answer-code: reserved target class has no training support")
    query = query.to(device)
    inputs, request, truth = make_episode(
        table.schema, tuple(v.to(device) for v in values), torch.ones_like(query), query,
        code_seed=code_seed,
    )
    addresses = [[row_ids[r], a] for r, a in query.cpu().nonzero().tolist()]
    info = {
        **audit, "table": table.name, "cohort": table.cohort, "kind": table.kind,
        "role": table.role, "partition": partition, "evaluation": evaluation,
        "index": index, "mask_mode": kind, "fraction": fraction,
        "row_ids": list(row_ids), "query_addresses": addresses,
        "query_count": len(addresses), "query_per_column": query.sum(0).cpu().tolist(),
        "query_cell_fraction": len(addresses) / (len(row_ids) * table.width),
        "query_row_fraction": len({r for r, _ in addresses}) / len(row_ids),
        "visible_per_column": inputs.visible.sum(0).cpu().tolist(),
        "numeric_query_guard": audit.get("numeric_query_guard", guard),
        "codec_version": codec_version, "support_policy": support_policy,
        "protected_numeric_tail_cells": audit.get("protected_numeric_tail_cells", 0),
        "protected_discrete_cells": audit.get("protected_discrete_cells", 0),
        "mask_seed": mask_seed, "code_seed": code_seed, "window_seed": window_seed,
        "selected_rows": len(row_ids), "train_pool_rows": table.train_rows,
        "row_coverage": len(row_ids) / table.train_rows if partition == "train" else 1.0,
        "query_row_ids": sorted({address[0] for address in addresses}),
        "transductive": partition != "train",
        "context_protocol": "visible holdout non-target features" if partition != "train"
                            else "training rows only",
    }
    return inputs, request, truth, info
