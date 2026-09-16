"""Bounded multi-table fitting for the table-restoration reference model.

This runner is deliberately separate from the legacy TAR joint-fit runner.  It
uses the restoration scorer for every observed cell, keeps reserved rows out of
the model, and binds every resume to the corpus, source and preregistration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import subprocess
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from tabu_lab.models.restoration import (
    ColumnSchema,
    RestorationConfig,
    RestorationModel,
    make_episode,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)
from tabu_lab.restoration_masking import global_query_mask
from tabu_lab.tar_data import validate_full_dataset

SCHEMA = "tabu.restoration.joint-fit.v1"
PROTOCOL = "old120_all_train_rows_mixed_types_all_observed_targets_v2"
GLOBAL_MASK_PROTOCOL = "old120_all_train_rows_mixed_types_global_fraction_all_observed_targets_v3"


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_mapping(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"mapping expected: {path.name}")
    return dict(value)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def _seed(seed: int, *parts: object) -> int:
    raw = json.dumps([seed, *parts], separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class TablePlan:
    name: str
    row_ids: tuple[int, ...]
    values: tuple[torch.Tensor, ...]
    schema: tuple[ColumnSchema, ...]
    kind_counts: dict[str, int]
    train_rows: int
    reserved_rows: int


def _table_schema(name: str, features: list[dict]) -> tuple[ColumnSchema, ...]:
    schema = []
    for index, feature in enumerate(features):
        kind = feature.get("kind")
        if kind not in ("numeric", "nominal", "ordinal"):
            raise ValueError(f"{name}: unsupported feature kind")
        if kind == "numeric":
            schema.append(ColumnSchema(f"{name}/column-{index}", kind))
        else:
            domain = feature.get("domain")
            if not isinstance(domain, list) or not domain:
                raise ValueError(f"{name}: discrete column needs a declared domain")
            order = feature.get("order")
            if order is not None and not isinstance(order, list | tuple):
                raise ValueError(f"{name}: ordinal order must be a sequence")
            schema.append(ColumnSchema(
                f"{name}/column-{index}", kind, len(domain),
                tuple(order) if order is not None else None,
            ))
    return tuple(schema)


def _load_table(name: str, path: Path, expected_rows: int, expected_hash: str) -> TablePlan:
    if not path.is_file() or _hash(path) != expected_hash:
        raise ValueError(f"dataset digest mismatch: {name}")
    data = _load_mapping(path)
    coverage = validate_full_dataset(data, expected_rows)
    values = data.get("values")
    features = data.get("features")
    if not isinstance(values, list) or not values or not isinstance(features, list):
        raise ValueError(f"{name}: typed table requires values and features")
    width = len(features)
    if any(not isinstance(row, list) or len(row) != width for row in values):
        raise ValueError(f"{name}: values must be rectangular")
    target = features[-1]
    if data.get("target_kind") != target.get("kind"):
        raise ValueError(f"{name}: target kind metadata mismatch")
    if target.get("kind") != "numeric" and data.get("domain") != target.get("domain"):
        raise ValueError(f"{name}: target domain metadata mismatch")
    schema = _table_schema(name, features)
    columns = []
    for column, spec in enumerate(schema):
        raw = [row[column] for row in values]
        if spec.kind == "numeric":
            if any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x)
                   for x in raw):
                raise ValueError(f"{name}: numeric values must be finite real numbers")
            columns.append(torch.tensor([raw[i] for i in data["splits"]["train"]],
                                        dtype=torch.float64))
        else:
            if any(type(x) is not int or not 0 <= x < spec.domain_size for x in raw):
                raise ValueError(f"{name}: discrete value outside declared domain")
            columns.append(torch.tensor([raw[i] for i in data["splits"]["train"]],
                                        dtype=torch.long))
    counts = Counter(spec.kind for spec in schema)
    return TablePlan(
        name,
        tuple(data["splits"]["train"]),
        tuple(columns),
        schema,
        dict(counts),
        coverage["train_rows"],
        coverage["test_rows"],
    )


def _query_mask(table: TablePlan, count: int, seed: int) -> tuple[torch.Tensor, dict]:
    """Sample exactly ``count`` Query cells while keeping every class supported."""
    n, width = table.train_rows, len(table.schema)
    if not 1 <= count <= n - 2:
        raise ValueError("query_count must leave at least two visible rows")
    query = torch.zeros(n, width, dtype=torch.bool)
    protected = 0
    singleton_classes = 0
    eligible_counts = []
    singleton_per_column = []
    for column, spec in enumerate(table.schema):
        rng = random.Random(f"{seed}/{column}")
        if spec.kind == "numeric":
            eligible = list(range(n))
            singleton_per_column.append(0)
        else:
            labels = table.values[column]
            occurrences = Counter(labels.tolist())
            rows_by_class = {}
            for row, label in enumerate(labels.tolist()):
                rows_by_class.setdefault(int(label), []).append(row)
            keep = {rng.choice(rows) for rows in rows_by_class.values()}
            eligible = [row for row in range(n) if row not in keep]
            protected += len(keep)
            singleton = sum(count == 1 for count in occurrences.values())
            singleton_classes += singleton
            singleton_per_column.append(singleton)
            if len(keep) > n - count:
                raise ValueError(f"{table.name}: query mask cannot preserve all classes")
        if len(eligible) < count:
            raise ValueError(f"{table.name}: fewer than query_count safely maskable cells")
        rng.shuffle(eligible)
        query[eligible[:count], column] = True
        eligible_counts.append(len(eligible))
    return query, dict(
        query_per_column=count,
        eligible_per_column=eligible_counts,
        protected_discrete_cells=protected,
        unmaskable_discrete_classes=singleton_classes,
        singleton_discrete_classes=singleton_classes,
        singleton_classes_per_column=singleton_per_column,
        query_coverage=float(count * width) / (n * width),
    )


def _source_identity() -> dict:
    root = Path(__file__).parent
    files = {"restoration_joint_fit.py": _hash(Path(__file__)), "cli.py": _hash(root / "cli.py")}
    files.update({f"models/restoration/{p.name}": _hash(p)
                  for p in sorted((root / "models" / "restoration").glob("*.py"))})
    files["tar_data.py"] = _hash(root / "tar_data.py")
    files["restoration_masking.py"] = _hash(root / "restoration_masking.py")
    preflight = root / "restoration_joint_preflight.py"
    if preflight.exists():
        files["restoration_joint_preflight.py"] = _hash(preflight)
    observer = root / "observers" / "restoration.py"
    if observer.exists():
        files["observers/restoration.py"] = _hash(observer)
    return {"files": files, "sha256": _digest(files)}


@dataclass(frozen=True)
class FitPlan:
    spec: dict
    corpus: Path
    corpus_spec: dict
    tables: tuple[TablePlan, ...]
    config: RestorationConfig
    source: dict
    identity: dict

    @property
    def table_count(self):
        return len(self.tables)


def _require_committed(path: Path) -> None:
    path = path.resolve()
    try:
        root = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=path.parent,
            text=True, stderr=subprocess.DEVNULL).strip())
        relative = path.relative_to(root).as_posix()
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"],
                                            cwd=root, stderr=subprocess.DEVNULL)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("execute requires a committed preregistration in Git HEAD") from error
    if committed != path.read_bytes():
        raise ValueError("preregistration bytes differ from committed Git HEAD")


def prepare_plan(preregistration: Path, corpus: Path, device="cpu") -> FitPlan:
    spec = _load_mapping(Path(preregistration))
    if spec.get("schema") != SCHEMA or spec.get("status") != "local_unissued":
        raise ValueError(f"preregistration schema/status must be {SCHEMA}/local_unissued")
    protocol = spec.get("protocol")
    if protocol not in (PROTOCOL, GLOBAL_MASK_PROTOCOL):
        raise ValueError(f"preregistration protocol must be {PROTOCOL} or {GLOBAL_MASK_PROTOCOL}")
    corpus = Path(corpus).resolve()
    corpus_spec_path = corpus / "preregistration.yaml"
    corpus_spec = _load_mapping(corpus_spec_path)
    if spec.get("corpus_preregistration_sha256") != _hash(corpus_spec_path):
        raise ValueError("corpus preregistration digest mismatch")
    if corpus_spec.get("schema") != "tabu.tar.joint-training-fit.1":
        raise ValueError("unexpected old120 corpus schema")
    if ("corpus_manifest_sha256" in corpus_spec
            and corpus_spec["corpus_manifest_sha256"] != _hash(corpus / "manifest.json")):
        raise ValueError("corpus manifest digest mismatch")
    datasets = corpus_spec.get("datasets")
    expected = corpus_spec.get("expected_rows")
    if not isinstance(datasets, dict) or not isinstance(expected, dict):
        raise ValueError("corpus preregistration needs datasets and expected_rows")
    names = spec.get("dataset_names")
    if names is None:
        names = list(datasets)
    if not isinstance(names, list) or len(names) != spec.get("table_count"):
        raise ValueError("dataset_names must contain the declared table_count")
    if len(set(names)) != len(names) or set(names) != set(datasets):
        raise ValueError("dataset_names must cover the frozen corpus exactly")
    if protocol == GLOBAL_MASK_PROTOCOL:
        if "query_count" in spec:
            raise ValueError("global mask protocol declares mask_fraction, not query_count")
        if _number(spec.get("mask_fraction"), "mask_fraction") >= 1:
            raise ValueError("mask_fraction must be below one")
    else:
        if "mask_fraction" in spec:
            raise ValueError("legacy per-column protocol cannot declare mask_fraction")
        if _integer(spec.get("query_count"), "query_count") >= 204:
            raise ValueError("query_count must be below the frozen 204 train rows")
    _integer(spec.get("evaluation_masks"), "evaluation_masks")
    max_rounds = _integer(spec.get("max_rounds"), "max_rounds")
    max_seconds = _number(spec.get("max_seconds"), "max_seconds")
    evaluate_every = _integer(spec.get("evaluate_every_rounds", 1), "evaluate_every_rounds")
    reserve = _number(spec.get("final_reserve_seconds", min(300, max_seconds / 10)),
                      "final_reserve_seconds", zero=True)
    if reserve >= max_seconds:
        raise ValueError("final_reserve_seconds must be below max_seconds")
    if _integer(spec.get("checkpoint_every_round"), "checkpoint_every_round") != 1:
        raise ValueError("checkpoint_every_round must be 1 for resumable joint fit")
    seeds = spec.get("seeds")
    required_seeds = {"model", "order", "masks", "codes"}
    if not isinstance(seeds, dict) or set(seeds) != required_seeds:
        raise ValueError("seeds must declare model, order, masks and codes")
    for seed_name in required_seeds:
        _integer(seeds[seed_name], f"seed:{seed_name}", minimum=0)
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    config = RestorationConfig.from_dict(spec["model"])
    optimizer = spec.get("optimizer")
    required_optimizer = {
        "kind", "learning_rate", "weight_decay", "betas", "eps", "grad_clip"
    }
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != required_optimizer
        or optimizer["kind"] != "adamw"
    ):
        raise ValueError("optimizer must explicitly declare AdamW fields")
    _number(optimizer["learning_rate"], "learning_rate")
    _number(optimizer["weight_decay"], "weight_decay", zero=True)
    _number(optimizer["eps"], "eps")
    _number(optimizer["grad_clip"], "grad_clip")
    if (
        not isinstance(optimizer["betas"], list)
        or len(optimizer["betas"]) != 2
        or any(not 0 <= _number(x, "beta", zero=True) < 1 for x in optimizer["betas"])
    ):
        raise ValueError("betas must contain two values in [0,1)")
    tables = tuple(_load_table(name, corpus / "data" / f"{name}.json",
                               expected[name], datasets[name]) for name in names)
    if len(tables) == 120 and (
        {table.train_rows for table in tables} != {204}
        or {table.reserved_rows for table in tables} != {52}
    ):
        raise ValueError("old120 corpus must have 204 train and 52 reserved rows per table")
    identity = {
        "schema": SCHEMA,
        "protocol": protocol,
        "device": device,
        "dtype": "float64",
        "preregistration_sha256": _hash(preregistration),
        "corpus_preregistration_sha256": _hash(corpus_spec_path),
        "corpus_manifest_sha256": _hash(corpus / "manifest.json"),
        "dataset_sha256": {t.name: datasets[t.name] for t in tables},
        "source": _source_identity(),
        "config": config.as_dict(),
    }
    summary = {
        "status": "local_unissued",
        "outcome": "planned",
        "execution_started": False,
        "protocol": protocol,
        "table_count": len(tables),
        "train_rows_per_table": sorted({t.train_rows for t in tables}),
        "reserved_rows_per_table": sorted({t.reserved_rows for t in tables}),
        "type_counts": dict(sum((Counter(t.kind_counts) for t in tables), Counter())),
        "singleton_class_tables": sum(
            any(
                spec.kind != "numeric" and any(count == 1 for count in Counter(
                    table.values[column].tolist()
                ).values())
                for column, spec in enumerate(table.schema)
            )
            for table in tables
        ),
        "singleton_class_columns": sum(
            sum(
                spec.kind != "numeric" and any(count == 1 for count in Counter(
                    table.values[column].tolist()
                ).values())
                for column, spec in enumerate(table.schema)
            )
            for table in tables
        ),
        **({"mask_fraction": spec["mask_fraction"]} if protocol == GLOBAL_MASK_PROTOCOL
           else {"query_count": spec["query_count"]}),
        "evaluation_masks": spec["evaluation_masks"],
        "max_rounds": max_rounds,
        "max_updates": max_rounds * len(tables),
        "max_seconds": max_seconds,
        "evaluate_every_rounds": evaluate_every,
        "final_reserve_seconds": reserve,
        "model": config.as_dict(),
        "optimizer": optimizer,
        "data_scope": "all training rows; reserved rows excluded",
        "mask_policy": (
            "global cell sampling without equal per-column counts; randomly preserve one "
            "visible sample per observed discrete class; retain at least two cells per column"
            if protocol == GLOBAL_MASK_PROTOCOL else
            "equal per-column counts; randomly preserve one visible sample "
            "per observed discrete class"
        ),
    }
    # Validate the mask budget against every frozen table before starting a run.
    for table in tables:
        if protocol == GLOBAL_MASK_PROTOCOL:
            global_query_mask(table, spec["mask_fraction"], seeds["masks"])
        else:
            _query_mask(table, spec["query_count"], seeds["masks"])
    return FitPlan(dict(spec, _summary=summary), corpus, corpus_spec, tables, config,
                   _source_identity(), identity)


def _episode(table: TablePlan, query: torch.Tensor, code_seed: int, device: str):
    values = tuple(value.to(device) for value in table.values)
    query = query.to(device)
    observed = torch.ones_like(query)
    return make_episode(table.schema, values, observed, query, code_seed=code_seed)


def _fixed_bank(plan: FitPlan, table: TablePlan, device: str):
    for index in range(plan.spec["evaluation_masks"]):
        query, mask_info = _plan_query_mask(
            plan, table, _seed(plan.spec["seeds"]["masks"], table.name, index)
        )
        code_seed = _seed(plan.spec["seeds"]["codes"], table.name, index)
        yield (_episode(table, query, code_seed, device), mask_info)


def _plan_query_mask(plan: FitPlan, table: TablePlan, seed: int):
    if plan.spec["protocol"] == GLOBAL_MASK_PROTOCOL:
        return global_query_mask(table, plan.spec["mask_fraction"], seed)
    return _query_mask(table, plan.spec["query_count"], seed)


def _distribution(values, *, eligible_table_count=None):
    """Equal-table summaries with linearly interpolated median and P95."""
    ordered = sorted(float(value) for value in values)
    count = len(ordered)

    def quantile(probability):
        if not count:
            return None
        position = (count - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    result = {"mean": sum(ordered) / count if count else None,
              "median": quantile(.5), "p95": quantile(.95), "table_count": count}
    if eligible_table_count is not None:
        result["eligible_table_count"] = eligible_table_count
    return result


def _table_macro(tables, by_table, by_table_type, completed_tables):
    """Combine masks within each table before giving every table equal weight.

    Tables without any scored cells for a metric are excluded from that metric.
    ``eligible_table_count`` is the number of corpus tables containing the
    relevant type, so sparse Query coverage and partial evaluation stay visible.
    """
    result = {}
    for state in ("retained", "query"):
        result[state] = {}
        for group, metrics in (
            ("all", ("encoding_mse",)),
            ("numeric", ("encoding_mse", "numeric_mse")),
            ("discrete", ("encoding_mse", "discrete_accuracy", "discrete_error_rate")),
        ):
            values = {metric: [] for metric in metrics}
            target_counts = {metric: 0 for metric in metrics}
            eligible = 0
            for table in tables:
                if group == "all":
                    has_type = True
                    item = by_table[table.name][state]
                elif group == "numeric":
                    has_type = bool(table.kind_counts.get("numeric", 0))
                    item = by_table_type[table.name]["numeric"][state]
                else:
                    has_type = bool(table.kind_counts.get("nominal", 0)
                                    or table.kind_counts.get("ordinal", 0))
                    parts = [by_table_type[table.name][kind][state]
                             for kind in ("nominal", "ordinal")]
                    count = sum(part["count"] for part in parts)
                    item = {
                        metric: (sum(part[metric] * part["count"] for part in parts
                                     if part["count"]) / count if count else None)
                        for metric in metrics
                    }
                    item["count"] = count
                eligible += int(has_type)
                if table.name not in completed_tables:
                    continue
                for metric in metrics:
                    if item[metric] is not None:
                        values[metric].append(item[metric])
                        target_counts[metric] += item["count"]
            result[state][group] = {
                metric: dict(_distribution(items, eligible_table_count=eligible),
                             target_count=target_counts[metric])
                for metric, items in values.items()
            }
    return result


class _PreparedCache:
    """Bounded LRU for fixed evaluation episodes.

    The cache holds only prepared visible/scorer facts.  It never holds an
    autograd graph or learned state, and a new mask or code seed uses a new key.
    Eight entries cap the host/device footprint while evaluation walks all
    tables.
    """

    def __init__(self, model: RestorationModel, capacity=8):
        self.model = model
        self.capacity = capacity
        self.entries = OrderedDict()
        self.identities = {}

    def get(self, key, episode):
        inputs, request, truth = episode
        identity = hashlib.sha256(repr((inputs.schema, inputs.code_seed)).encode())
        for tensor in (*inputs.values, inputs.visible, inputs.query, request.targets,
                       *truth.values, truth.states):
            identity.update(str((tensor.shape, tensor.dtype, tensor.device)).encode())
            identity.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        fingerprint = identity.hexdigest()
        if key not in self.entries or self.identities[key] != fingerprint:
            self.entries[key] = prepare_episode(self.model, *episode)
            self.identities[key] = fingerprint
            if len(self.entries) > self.capacity:
                evicted, _ = self.entries.popitem(last=False)
                del self.identities[evicted]
        self.entries.move_to_end(key)
        return self.entries[key]


def _metrics(plan: FitPlan, model: RestorationModel, device: str, banks=None,
             prepared_cache=None, *, deadline=None, progress=None) -> dict:
    """Reduce metrics on device; copy one small column summary per episode.

    Evaluation episodes are regenerated from fixed CPU seeds one table at a
    time. A deadline produces an explicit partial receipt, never a complete
    score for the full corpus. The check is between episodes (one forward pass
    is the smallest interruptible unit).
    """
    model.eval()
    fields = ("count", "encoding_sse", "numeric_count", "numeric_sse",
              "discrete_count", "discrete_correct")

    def blank():
        return dict.fromkeys(fields, 0)

    def states():
        return {state: blank() for state in ("retained", "query")}

    totals = states()
    by_type = {kind: states() for kind in ("numeric", "nominal", "ordinal")}
    by_table = {table.name: states() for table in plan.tables}
    by_table_type = {table.name: {kind: states() for kind in by_type} for table in plan.tables}
    by_source = {table.name.rsplit("_", 1)[0]: states() for table in plan.tables}
    losses = []
    episodes_by_table = Counter()
    coverage = dict.fromkeys(("query_cells", "total_cells", "protected_discrete_cells",
                             "unmaskable_discrete_classes", "singleton_discrete_classes"), 0)
    expected = plan.table_count * plan.spec["evaluation_masks"]
    complete = True
    longest_episode = 0.0
    with torch.no_grad():
        for table in plan.tables:
            # Only this table's masks/episodes occupy device memory. The prepared
            # cache is independently bounded, including scorer-only clean truth.
            if deadline is not None and time.monotonic() + longest_episode >= deadline:
                complete = False
                break
            bank = banks[table.name] if banks is not None else _fixed_bank(plan, table, device)
            for index, ((inputs, request, truth), info) in enumerate(bank):
                tick = time.monotonic()
                if deadline is not None and tick + longest_episode >= deadline:
                    complete = False
                    break
                if prepared_cache is None:
                    score = score_episode(model, inputs, request, truth)
                else:
                    prepared = prepared_cache.get((table.name, index),
                                                   (inputs, request, truth))
                    score = score_prepared_episode(model, prepared, decode=True, report=True)
                targets = request.targets
                target_states = truth.states[targets[:, 0], targets[:, 1]]
                reduced = []
                kinds = []
                for column in score.output.columns:
                    positions = column.target_indices
                    rows = targets[positions, 0]
                    kind = table.schema[column.column].kind
                    kinds.append(kind)
                    membership = torch.stack((target_states[positions] == 0,
                                              target_states[positions] == 1)).to(torch.float64)
                    count = membership.sum(1)
                    error = (membership * score.per_target[positions]).sum(1)
                    actual = truth.values[column.column][rows]
                    if kind == "numeric":
                        numeric = (membership * (column.decoded - actual).square()).sum(1)
                        zero = torch.zeros_like(count)
                        reduced.append(torch.stack((count, error, count, numeric, zero, zero), 1))
                    else:
                        correct = (membership * (column.decoded == actual)).sum(1)
                        zero = torch.zeros_like(count)
                        reduced.append(torch.stack((count, error, zero, zero, count, correct), 1))
                # One transfer replaces thousands of per-cell CUDA synchronizations.
                compact = torch.cat((score.loss.reshape(1), torch.stack(reduced).reshape(-1)))
                values = compact.detach().cpu().tolist()
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError("nonfinite evaluation metric")
                losses.append(values[0])
                episodes_by_table[table.name] += 1
                source = table.name.rsplit("_", 1)[0]
                for column_index, kind in enumerate(kinds):
                    for state_index, state in enumerate(("retained", "query")):
                        offset = 1 + (column_index * 2 + state_index) * len(fields)
                        row = values[offset:offset + len(fields)]
                        for item in (totals[state], by_type[kind][state],
                                     by_table[table.name][state], by_source[source][state],
                                     by_table_type[table.name][kind][state]):
                            for field, value in zip(fields, row, strict=True):
                                item[field] += value
                per_column = info["query_per_column"]
                coverage["query_cells"] += (sum(per_column) if isinstance(per_column, list)
                                            else per_column * len(table.schema))
                coverage["total_cells"] += len(targets)
                for name in ("protected_discrete_cells", "unmaskable_discrete_classes",
                             "singleton_discrete_classes"):
                    coverage[name] += info[name]
                longest_episode = max(longest_episode, time.monotonic() - tick)
            if progress is not None:
                progress({"completed_episodes": len(losses), "total_episodes": expected,
                          "table": table.name})
            if not complete:
                break

    def finish(item):
        count = int(item["count"])
        item["count"] = count
        item["encoding_mse"] = item.pop("encoding_sse") / count if count else None
        numeric_count = int(item.pop("numeric_count"))
        numeric_sse = item.pop("numeric_sse")
        item["numeric_mse"] = numeric_sse / numeric_count if numeric_count else None
        discrete_count = int(item.pop("discrete_count"))
        discrete_correct = item.pop("discrete_correct")
        item["discrete_accuracy"] = discrete_correct / discrete_count if discrete_count else None
        item["discrete_error_rate"] = (1 - item["discrete_accuracy"]
                                       if discrete_count else None)

    table_type_states = [group for table_types in by_table_type.values()
                         for group in table_types.values()]
    for group in (totals, *by_type.values(), *by_table.values(), *by_source.values(),
                  *table_type_states):
        for item in group.values():
            finish(item)
    coverage["query_fraction"] = (coverage["query_cells"] / coverage["total_cells"]
                                  if coverage["total_cells"] else None)
    completed_tables = {name for name, count in episodes_by_table.items()
                        if count == plan.spec["evaluation_masks"]}
    complete = complete and len(losses) == expected
    return {"loss": sum(losses) / len(losses) if losses else None, "by_state": totals,
            "by_type": by_type, "by_table": by_table, "by_source": by_source,
            "by_table_type": by_table_type,
            "table_macro": _table_macro(plan.tables, by_table, by_table_type, completed_tables),
            "table_macro_complete": complete,
            "table_macro_scope": "equal table weights after pooling fixed masks within table",
            "completed_tables": len(completed_tables), "expected_tables": plan.table_count,
            "episodes_by_table": {table.name: episodes_by_table[table.name]
                                  for table in plan.tables},
            "coverage": coverage, "complete": complete,
            "completed_episodes": len(losses), "expected_episodes": expected,
            "stop_reason": None if complete else "wall_limit",
            "scope": "fixed masks on training rows; no reserved evaluation"}


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, list | tuple):
        return all(_finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def _checkpoint(path: Path, model, optimizer, identity, round_index, update, cursor, elapsed,
                *, replace=False, evaluation=None, training_metrics=None):
    state = {
        "schema": "tabu.restoration.joint-fit-checkpoint.v3",
        "identity": identity,
        "config": model.config.as_dict(),
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state_all()
            if identity["device"] == "cuda:0" and torch.cuda.is_available()
            else []
        ),
        "round": round_index,
        "update": update,
        "cursor": cursor,
        "elapsed_seconds": elapsed,
        "evaluation": evaluation or {},
        "training_metrics": training_metrics or {"partial_round_losses": {}, "round_summaries": []},
    }
    if not _finite_state(state["model"]) or not _finite_state(state["optimizer"]):
        raise FloatingPointError("nonfinite checkpoint state")
    target = path
    temporary = path.with_name(f".{path.name}.tmp") if replace else None
    if temporary is not None:
        target = temporary
    with target.open("wb" if replace else "xb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if temporary is not None:
        os.replace(temporary, path)


def _load_checkpoint(path: Path, model, optimizer, plan: FitPlan):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != "tabu.restoration.joint-fit-checkpoint.v3":
        raise ValueError("unsupported joint-fit checkpoint schema")
    if state.get("identity") != plan.identity or state.get("config") != model.config.as_dict():
        raise ValueError("checkpoint identity or source drift")
    if not _finite_state(state.get("model")) or not _finite_state(state.get("optimizer")):
        raise ValueError("checkpoint contains nonfinite state")
    round_index = _integer(state.get("round"), "checkpoint round", minimum=0)
    cursor = _integer(state.get("cursor"), "checkpoint cursor", minimum=0)
    update = _integer(state.get("update"), "checkpoint update", minimum=0)
    _number(state.get("elapsed_seconds"), "checkpoint elapsed_seconds", zero=True)
    if (round_index > plan.spec["max_rounds"] or cursor >= plan.table_count
            or update != round_index * plan.table_count + cursor
            or (round_index == plan.spec["max_rounds"] and cursor)):
        raise ValueError("checkpoint round/cursor/update are inconsistent")
    training_metrics = state.get("training_metrics", {})
    partial = training_metrics.get("partial_round_losses", {})
    summaries = training_metrics.get("round_summaries", [])
    order = list(range(plan.table_count))
    random.Random(f"{plan.spec['seeds']['order']}/{round_index}").shuffle(order)
    expected_partial = {plan.tables[index].name for index in order[:cursor]}
    if (not isinstance(partial, dict) or set(partial) != expected_partial
            or not all(isinstance(value, float) and math.isfinite(value)
                       for value in partial.values())
            or not isinstance(summaries, list) or len(summaries) != round_index
            or any(item.get("training_round") != index + 1
                   for index, item in enumerate(summaries))):
        raise ValueError("checkpoint training round metrics are inconsistent")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if plan.identity["device"] == "cuda:0":
        if len(state["torch_cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG count mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    elif state["torch_cuda_rng"]:
        raise ValueError("CPU checkpoint unexpectedly contains CUDA RNG state")
    return state


class WallLimit(RuntimeError):
    pass


def run_joint_fit(args, observer=None):
    """Execute a preregistered run; local receipts remain authoritative.

    ``observer`` is an optional callable receiving JSON-safe event mappings.
    Observer failures are recorded without discarding a valid training boundary.
    The wall budget is cumulative across attempts, including evaluation and
    checkpointing. A CUDA forward/update is the smallest non-preemptible unit.
    """
    prereg = Path(args.preregistration)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("output-root must be a new attempt directory")
    plan = prepare_plan(prereg, Path(args.corpus), args.device)
    if not args.execute:
        return dict(plan.spec["_summary"], corpus=str(Path(args.corpus).resolve()),
                    source=plan.source, identity=plan.identity)
    _require_committed(prereg)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA, "status": "local_unissued", "outcome": "started",
               "execution_started": False, "round": 0, "update": 0,
               "preregistration_sha256": _hash(prereg)}
    started = time.monotonic()
    previous_signal = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        KeyboardInterrupt(f"signal {signum}")))
    model = optimizer = None
    round_index = update = cursor = 0
    prior_elapsed = 0.0
    deadline = started + plan.spec["max_seconds"]
    latest_checkpoint = None
    evaluation = {"initial": None, "latest": None, "latest_update": -1}
    training_metrics = {"partial_round_losses": {}, "round_summaries": []}
    prepared_cache = None
    stop_round = getattr(args, "stop_after_round", None) or plan.spec["max_rounds"]
    evaluate_every = plan.spec["_summary"]["evaluate_every_rounds"]
    reserve = plan.spec["_summary"]["final_reserve_seconds"]
    save_reserve = min(5.0, reserve / 10)

    def elapsed():
        return prior_elapsed + time.monotonic() - started

    def emit(event, **payload):
        if observer is not None:
            try:
                observer(dict(event=event, round=round_index, update=update,
                              elapsed_seconds=elapsed(), **payload))
            except Exception as error:
                failures = receipt.setdefault("observer_errors", [])
                if len(failures) < 10:
                    failures.append({"event": event, "error_type": type(error).__name__})

    def checkpoint(path, *, replace=False):
        nonlocal latest_checkpoint
        _checkpoint(path, model, optimizer, plan.identity, round_index, update,
                    cursor, elapsed(), replace=replace, evaluation=evaluation,
                    training_metrics=training_metrics)
        latest_checkpoint = path

    def evaluate(stage, limit):
        emit("phase", stage=stage)
        result = _metrics(
            plan, model, args.device, prepared_cache=prepared_cache, deadline=limit,
            progress=lambda values: emit("evaluation_progress", stage=stage, **values),
        )
        result["at_update"] = update
        result["at_round"] = round_index
        _write_json(output / f"{stage}-metrics.json", result)
        if result["complete"]:
            evaluation["latest"] = result
            evaluation["latest_update"] = update
            if update == 0 and evaluation["initial"] is None:
                evaluation["initial"] = result
            checkpoint(output / "checkpoint-progress.pt", replace=True)
        emit("phase", stage=f"{stage}_complete", metrics=result)
        return result

    try:
        _write_json(output / "started.json", receipt)
        if args.device == "cuda:0":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no CPU fallback")
            receipt["cuda_device"] = {"name": torch.cuda.get_device_name(0),
                                      "capability": list(torch.cuda.get_device_capability(0))}
            torch.cuda.reset_peak_memory_stats()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(1)
        torch.manual_seed(plan.spec["seeds"]["model"])
        model = RestorationModel(plan.config).to(device=args.device, dtype=torch.float64)
        opt = plan.spec["optimizer"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt["learning_rate"],
                                      weight_decay=opt["weight_decay"],
                                      betas=tuple(opt["betas"]), eps=opt["eps"])
        resume_path = getattr(args, "resume_checkpoint", None)
        if resume_path:
            state = _load_checkpoint(Path(resume_path), model, optimizer, plan)
            round_index, update, cursor = state["round"], state["update"], state["cursor"]
            prior_elapsed = state["elapsed_seconds"]
            evaluation = state["evaluation"]
            training_metrics = state["training_metrics"]
            receipt["resumed_from_update"] = update
        if not 0 < stop_round <= plan.spec["max_rounds"] or stop_round < round_index:
            raise ValueError("stop_after_round must be within the remaining round budget")
        if stop_round == round_index and cursor:
            raise ValueError("stop_after_round precedes the checkpoint table position")
        deadline = started + max(0.0, plan.spec["max_seconds"] - prior_elapsed)
        training_deadline = deadline - reserve
        receipt.update(execution_started=True, identity=plan.identity,
                       model_parameters=sum(p.numel() for p in model.parameters()),
                       type_counts=plan.spec["_summary"]["type_counts"],
                       evaluate_every_rounds=evaluate_every, final_reserve_seconds=reserve)
        prepared_cache = _PreparedCache(model, capacity=8)
        _write_json(output / "resolved.json",
                    {"preregistration": plan.spec, "identity": plan.identity})
        # CPU-only identities are sufficient to regenerate the fixed bank.
        masks_record = {}
        for table in plan.tables:
            masks_record[table.name] = []
            for index in range(plan.spec["evaluation_masks"]):
                mask_seed = _seed(plan.spec["seeds"]["masks"], table.name, index)
                query, info = _plan_query_mask(plan, table, mask_seed)
                masks_record[table.name].append({
                    "query": query.tolist(), "info": info, "mask_seed": mask_seed,
                    "code_seed": _seed(plan.spec["seeds"]["codes"], table.name, index),
                })
        _write_json(output / "evaluation-masks.json", masks_record)
        checkpoint(output / "checkpoint-initial.pt")
        emit("phase", stage="start", config=plan.spec, identity=plan.identity,
             model_parameters=receipt["model_parameters"])
        if evaluation["initial"] is None:
            if update:
                raise ValueError("checkpoint lacks the original initial evaluation")
            receipt["initial"] = evaluate("initial", training_deadline)
            if not receipt["initial"]["complete"]:
                receipt["final"] = receipt["initial"]
                raise WallLimit("initial evaluation did not fit within the training budget")
        else:
            receipt["initial"] = evaluation["initial"]
            _write_json(output / "initial-metrics.json", receipt["initial"])
        longest_update = 0.0
        with ((output / "updates.jsonl").open("x", encoding="utf-8") as curve,
              (output / "round-metrics.jsonl").open("x", encoding="utf-8") as round_curve):
            # A new attempt carries the complete prior round history exactly
            # once; partial rounds remain only in its durable accumulator.
            for summary in training_metrics["round_summaries"]:
                round_curve.write(json.dumps(summary, allow_nan=False) + "\n")
            round_curve.flush()
            os.fsync(round_curve.fileno())
            while round_index < stop_round:
                # On resume, complete an interrupted scheduled evaluation before
                # advancing to the next training round.
                if (cursor == 0 and round_index > 0 and round_index % evaluate_every == 0
                        and evaluation["latest_update"] != update):
                    metrics = evaluate(f"round-{round_index:04d}", training_deadline)
                    if not metrics["complete"]:
                        raise WallLimit("scheduled evaluation reached the training deadline")
                current_round = round_index
                order = list(range(plan.table_count))
                random.Random(f"{plan.spec['seeds']['order']}/{current_round}").shuffle(order)
                for position in range(cursor, plan.table_count):
                    if time.monotonic() + longest_update >= training_deadline:
                        raise WallLimit("training stopped to preserve final evaluation/save time")
                    tick = time.monotonic()
                    table = plan.tables[order[position]]
                    query, mask_info = _plan_query_mask(
                        plan, table,
                        _seed(plan.spec["seeds"]["masks"], table.name, "train", current_round),
                    )
                    code_seed = _seed(
                        plan.spec["seeds"]["codes"], table.name, "train", current_round,
                    )
                    episode = _episode(table, query, code_seed, args.device)
                    model.train()
                    prepared = prepare_episode(model, *episode)
                    optimizer.zero_grad(set_to_none=True)
                    score = score_prepared_episode(model, prepared)
                    if not bool(torch.isfinite(score.loss)):
                        raise FloatingPointError("nonfinite training loss")
                    score.loss.backward()
                    params = [p for p in model.parameters() if p.grad is not None]
                    if not params or any(not bool(torch.isfinite(p.grad).all()) for p in params):
                        raise FloatingPointError("missing or nonfinite training gradient")
                    norm = torch.nn.utils.clip_grad_norm_(params, opt["grad_clip"],
                                                          error_if_nonfinite=True)
                    optimizer.step()
                    if (not _finite_state(model.state_dict())
                            or not _finite_state(optimizer.state_dict())):
                        raise FloatingPointError("nonfinite parameters or optimizer state")
                    if args.device == "cuda:0":
                        torch.cuda.synchronize()
                    loss = float(score.loss.detach())
                    partial = training_metrics["partial_round_losses"]
                    if table.name in partial:
                        raise ValueError("duplicate table in training round metrics")
                    partial[table.name] = loss
                    # Counters always describe the NEXT update; round is the
                    # number of complete rounds, including at the last table.
                    update += 1
                    round_index = current_round + int(position + 1 == plan.table_count)
                    cursor = (position + 1) % plan.table_count
                    round_summary = None
                    if cursor == 0:
                        if set(partial) != {item.name for item in plan.tables}:
                            raise ValueError("complete training round does not cover all tables")
                        round_summary = {
                            "training_round": round_index, "completed_round": round_index,
                            "update": update,
                            "train_round": {
                                "loss": _distribution(partial.values()),
                                "completed_tables": len(partial), "total_tables": plan.table_count,
                                "complete": True,
                                "scope": "one pre-update loss per table during sequential training",
                            },
                        }
                        training_metrics["round_summaries"].append(round_summary)
                        training_metrics["partial_round_losses"] = {}
                    checkpoint(output / "checkpoint-progress.pt", replace=True)
                    row = {"round": current_round + 1, "update": update, "table": table.name,
                           "loss": loss, "gradient_norm": float(norm),
                           "mask": mask_info, "code_seed": code_seed,
                           "elapsed_seconds": elapsed()}
                    if args.device == "cuda:0":
                        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    # Includes episode preparation and durable checkpointing.
                    row["update_seconds"] = time.monotonic() - tick
                    curve.write(json.dumps(row, allow_nan=False) + "\n")
                    curve.flush()
                    os.fsync(curve.fileno())
                    longest_update = max(longest_update, time.monotonic() - tick)
                    if round_summary is not None:
                        round_curve.write(json.dumps(round_summary, allow_nan=False) + "\n")
                        round_curve.flush()
                        os.fsync(round_curve.fileno())
                        emit("round_summary", **{key: value for key, value in round_summary.items()
                                                 if key != "update"})
                    emit("update", **{k: v for k, v in row.items()
                                      if k not in ("round", "update", "elapsed_seconds")},
                         training_round=current_round + 1)
                checkpoint(output / f"checkpoint-round-{round_index:04d}.pt")
            if evaluation["latest_update"] != update:
                receipt["final"] = evaluate(f"round-{round_index:04d}", deadline - save_reserve)
            else:
                receipt["final"] = evaluation["latest"]
        if not receipt["final"]["complete"]:
            raise WallLimit("final evaluation reached max_seconds")
        receipt["outcome"] = ("completed" if stop_round == plan.spec["max_rounds"]
                              else "segment_completed")
    except WallLimit as error:
        receipt.update(outcome="wall_limit", error_type=type(error).__name__, error=str(error))
        # Training is at a validated boundary. Spend the explicitly reserved
        # time on a final evaluation; leave an honest partial receipt if needed.
        if model is not None and evaluation["initial"] is not None and "final" not in receipt:
            try:
                receipt["final"] = (evaluation["latest"] if evaluation["latest_update"] == update
                                    else evaluate("final", deadline - save_reserve))
            except KeyboardInterrupt:
                receipt["final_evaluation_interrupted"] = True
            except Exception as evaluation_error:
                receipt["final_evaluation_error_type"] = type(evaluation_error).__name__
    except KeyboardInterrupt as error:
        receipt.update(outcome="interrupted", error_type=type(error).__name__, error=str(error))
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        # Always restore the latest durable boundary, even when interruption or
        # an optimizer failure left live tensors partly changed/nonfinite.
        if latest_checkpoint is not None:
            try:
                saved = torch.load(latest_checkpoint, map_location="cpu", weights_only=True)
                round_index, update, cursor = saved["round"], saved["update"], saved["cursor"]
                saved["elapsed_seconds"] = elapsed()
                with (output / "checkpoint.pt").open("xb") as handle:
                    torch.save(saved, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                receipt["checkpoint"] = "checkpoint.pt"
            except Exception as error:
                receipt["checkpoint_error_type"] = type(error).__name__
                receipt["outcome"] = "failed"
        receipt.update(round=round_index, cursor=cursor, update=update, elapsed_seconds=elapsed())
        if args.device == "cuda:0" and receipt.get("execution_started"):
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["budget_exhausted"] = receipt["elapsed_seconds"] >= plan.spec["max_seconds"]
        emit("summary", metrics=receipt)
        _write_json(output / "terminal.json", receipt)
    return receipt


__all__ = ["SCHEMA", "prepare_plan", "run_joint_fit"]
