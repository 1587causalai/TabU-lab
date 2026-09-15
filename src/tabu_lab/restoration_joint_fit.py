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
from tabu_lab.tar_data import validate_full_dataset

SCHEMA = "tabu.restoration.joint-fit.v1"
PROTOCOL = "old120_all_train_rows_mixed_types_all_observed_targets_v1"


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
        if spec.kind == "numeric":
            eligible = list(range(n))
            singleton_per_column.append(0)
        else:
            labels = table.values[column]
            occurrences = Counter(labels.tolist())
            first = {}
            for row, label in enumerate(labels.tolist()):
                first.setdefault(int(label), row)
            keep = set(first.values())
            eligible = [row for row in range(n) if row not in keep]
            protected += len(keep)
            singleton = sum(count == 1 for count in occurrences.values())
            singleton_classes += singleton
            singleton_per_column.append(singleton)
            if len(keep) > n - count:
                raise ValueError(f"{table.name}: query mask cannot preserve all classes")
        if len(eligible) < count:
            raise ValueError(f"{table.name}: fewer than query_count safely maskable cells")
        random.Random(f"{seed}/{column}").shuffle(eligible)
        query[eligible[:count], column] = True
        eligible_counts.append(len(eligible))
    return query, dict(
        query_per_column=count,
        eligible_per_column=eligible_counts,
        protected_discrete_cells=protected,
        unmaskable_discrete_classes=protected,
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
    corpus = Path(corpus).resolve()
    corpus_spec_path = corpus / "preregistration.yaml"
    corpus_spec = _load_mapping(corpus_spec_path)
    if spec.get("corpus_preregistration_sha256") != _hash(corpus_spec_path):
        raise ValueError("corpus preregistration digest mismatch")
    if corpus_spec.get("schema") != "tabu.tar.joint-training-fit.1":
        raise ValueError("unexpected old120 corpus schema")
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
    if _integer(spec.get("query_count"), "query_count") >= 204:
        raise ValueError("query_count must be below the frozen 204 train rows")
    _integer(spec.get("evaluation_masks"), "evaluation_masks")
    max_rounds = _integer(spec.get("max_rounds"), "max_rounds")
    max_seconds = _number(spec.get("max_seconds"), "max_seconds")
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
        "protocol": PROTOCOL,
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
        "protocol": PROTOCOL,
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
        "query_count": spec["query_count"],
        "evaluation_masks": spec["evaluation_masks"],
        "max_rounds": max_rounds,
        "max_updates": max_rounds * len(tables),
        "max_seconds": max_seconds,
        "model": config.as_dict(),
        "optimizer": optimizer,
        "data_scope": "all training rows; reserved rows excluded",
        "mask_policy": "preserve one visible sample per observed discrete class",
    }
    return FitPlan(dict(spec, _summary=summary), corpus, corpus_spec, tables, config,
                   _source_identity(), identity)


def _episode(table: TablePlan, query: torch.Tensor, code_seed: int, device: str):
    values = tuple(value.to(device) for value in table.values)
    query = query.to(device)
    observed = torch.ones_like(query)
    return make_episode(table.schema, values, observed, query, code_seed=code_seed)


def _fixed_bank(plan: FitPlan, table: TablePlan, device: str):
    bank = []
    for index in range(plan.spec["evaluation_masks"]):
        query, mask_info = _query_mask(
            table, plan.spec["query_count"], _seed(plan.spec["seeds"]["masks"], table.name, index)
        )
        code_seed = _seed(plan.spec["seeds"]["codes"], table.name, index)
        bank.append((_episode(table, query, code_seed, device), mask_info))
    return bank


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

    def get(self, key, episode):
        if key not in self.entries:
            self.entries[key] = prepare_episode(self.model, *episode)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        self.entries.move_to_end(key)
        return self.entries[key]


def _metrics(plan: FitPlan, model: RestorationModel, device: str, banks=None,
             prepared_cache=None) -> dict:
    model.eval()
    def blank():
        return {"count": 0, "encoding_sse": 0.0, "numeric_count": 0,
                "numeric_sse": 0.0, "discrete_count": 0, "discrete_correct": 0}

    totals = {state: blank() for state in ("retained", "query")}
    by_type = {kind: {state: blank()
                      for state in ("retained", "query")}
               for kind in ("numeric", "nominal", "ordinal")}
    by_table = {
        table.name: {state: blank() for state in ("retained", "query")}
        for table in plan.tables
    }
    groups = sorted({table.name.rsplit("_", 1)[0] for table in plan.tables})
    by_source = {
        group: {state: blank() for state in ("retained", "query")}
        for group in groups
    }
    losses = []
    coverage = {
        "query_cells": 0,
        "total_cells": 0,
        "protected_discrete_cells": 0,
        "unmaskable_discrete_classes": 0,
        "singleton_discrete_classes": 0,
    }
    banks = banks or {t.name: _fixed_bank(plan, t, device) for t in plan.tables}
    with torch.no_grad():
        for table in plan.tables:
            for index, ((inputs, request, truth), info) in enumerate(banks[table.name]):
                if prepared_cache is None:
                    score = score_episode(model, inputs, request, truth)
                else:
                    prepared = prepared_cache.get((table.name, index),
                                                   (inputs, request, truth))
                    score = score_prepared_episode(
                        model, prepared, decode=True, report=True
                    )
                losses.append(float(score.loss))
                targets = request.targets
                states = truth.states[targets[:, 0], targets[:, 1]]
                coverage["query_cells"] += int((states == 1).sum())
                coverage["total_cells"] += len(states)
                coverage["protected_discrete_cells"] += info["protected_discrete_cells"]
                coverage["unmaskable_discrete_classes"] += info["unmaskable_discrete_classes"]
                coverage["singleton_discrete_classes"] += info["singleton_discrete_classes"]
                columns = {item.column: item for item in score.output.columns}
                for position, (row, column) in enumerate(targets.tolist()):
                    state = ("retained", "query")[int(states[position] == 1)]
                    kind = table.schema[column].kind
                    source = table.name.rsplit("_", 1)[0]
                    items = (
                        totals[state], by_type[kind][state],
                        by_table[table.name][state], by_source[source][state],
                    )
                    for item in items:
                        item["count"] += 1
                    error = float(score.per_target[position])
                    for item in items:
                        item["encoding_sse"] += error
                    prediction = columns[column].decoded
                    local = int((columns[column].target_indices == position).nonzero()[0])
                    actual = truth.values[column][row]
                    if kind == "numeric":
                        squared = float((prediction[local] - actual).square())
                        for item in items:
                            item["numeric_count"] += 1
                            item["numeric_sse"] += squared
                    else:
                        correct = int(prediction[local] == actual)
                        for item in items:
                            item["discrete_count"] += 1
                            item["discrete_correct"] += correct
    def finish(item):
        count = item["count"]
        item["encoding_mse"] = item.pop("encoding_sse") / count if count else None
        numeric_count = item.pop("numeric_count")
        numeric_sse = item.pop("numeric_sse")
        item["numeric_mse"] = numeric_sse / numeric_count if numeric_count else None
        discrete_count = item.pop("discrete_count")
        discrete_correct = item.pop("discrete_correct")
        item["discrete_accuracy"] = discrete_correct / discrete_count if discrete_count else None
        return item
    for state in totals:
        finish(totals[state])
    for kind in by_type:
        for state in by_type[kind]:
            finish(by_type[kind][state])
    for table in by_table.values():
        for state in table:
            finish(table[state])
    for source in by_source.values():
        for state in source:
            finish(source[state])
    coverage["query_fraction"] = coverage["query_cells"] / coverage["total_cells"]
    return {"loss": sum(losses) / len(losses), "by_state": totals,
            "by_type": by_type, "by_table": by_table, "by_source": by_source,
            "coverage": coverage,
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
                *, replace=False):
    state = {
        "schema": "tabu.restoration.joint-fit-checkpoint.v1",
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
    if state.get("schema") != "tabu.restoration.joint-fit-checkpoint.v1":
        raise ValueError("unsupported joint-fit checkpoint schema")
    if state.get("identity") != plan.identity or state.get("config") != model.config.as_dict():
        raise ValueError("checkpoint identity or source drift")
    if not _finite_state(state.get("model")) or not _finite_state(state.get("optimizer")):
        raise ValueError("checkpoint contains nonfinite state")
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


def run_joint_fit(args):
    prereg = Path(args.preregistration)
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("output-root must be a new attempt directory")
    plan = prepare_plan(prereg, Path(args.corpus), args.device)
    if not args.execute:
        summary = dict(plan.spec["_summary"], corpus=str(Path(args.corpus).resolve()),
                       source=plan.source, identity=plan.identity)
        return summary
    _require_committed(prereg)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": SCHEMA, "status": "local_unissued", "outcome": "started",
               "execution_started": False, "round": 0, "update": 0,
               "preregistration_sha256": _hash(prereg)}
    started = time.monotonic()
    previous_signal = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        KeyboardInterrupt(f"signal {signum}")))
    model = optimizer = None
    boundary = None
    round_index = update = cursor = 0
    prior_elapsed = 0.0
    stop_round = None
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
        state = _load_checkpoint(Path(resume_path), model, optimizer, plan) if resume_path else None
        round_index = int(state["round"]) if state else 0
        update = int(state["update"]) if state else 0
        cursor = int(state["cursor"]) if state else 0
        if round_index < 0 or round_index > plan.spec["max_rounds"]:
            raise ValueError("checkpoint round is outside the declared budget")
        if cursor < 0 or cursor > plan.table_count:
            raise ValueError("checkpoint table cursor is invalid")
        prior_elapsed = float(state["elapsed_seconds"]) if state else 0.0
        deadline = started + max(0.0, plan.spec["max_seconds"] - prior_elapsed)
        receipt.update(execution_started=True, round=round_index, update=update)
        eval_bank = {t.name: _fixed_bank(plan, t, args.device) for t in plan.tables}
        prepared_cache = _PreparedCache(model, capacity=8)
        receipt["identity"] = plan.identity
        receipt["model_parameters"] = sum(p.numel() for p in model.parameters())
        receipt["type_counts"] = plan.spec["_summary"]["type_counts"]
        receipt["initial"] = _metrics(plan, model, args.device, eval_bank, prepared_cache)
        _write_json(
            output / "resolved.json",
            {"preregistration": plan.spec, "identity": plan.identity},
        )
        _write_json(output / "initial-metrics.json", receipt["initial"])
        # Store the deterministic mask/code identities without serializing tensors.
        masks_record = {}
        for name, bank in eval_bank.items():
            masks_record[name] = []
            for (inputs, _request, _truth), info in bank:
                masks_record[name].append({"query": inputs.query.cpu().tolist(), "info": info,
                                           "code_seed": inputs.code_seed})
        _write_json(output / "evaluation-masks.json", masks_record)
        _checkpoint(output / "checkpoint-initial.pt", model, optimizer, plan.identity,
                    round_index, update, cursor, prior_elapsed)
        with (output / "updates.jsonl").open("x", encoding="utf-8") as curve:
            stop_round = getattr(args, "stop_after_round", None) or plan.spec["max_rounds"]
            if not 0 < stop_round <= plan.spec["max_rounds"] or stop_round <= round_index:
                raise ValueError("stop_after_round must advance within the declared round budget")
            for current_round in range(round_index, stop_round):
                if time.monotonic() >= deadline:
                    raise WallLimit("max_seconds reached before round")
                order = list(range(plan.table_count))
                random.Random(f"{plan.spec['seeds']['order']}/{current_round}").shuffle(order)
                start_position = cursor if current_round == round_index else 0
                for position, table_index in enumerate(order):
                    if position < start_position:
                        continue
                    if time.monotonic() >= deadline:
                        raise WallLimit("max_seconds reached between table updates")
                    table = plan.tables[table_index]
                    query, mask_info = _query_mask(
                        table, plan.spec["query_count"],
                        _seed(plan.spec["seeds"]["masks"], table.name, "train", current_round),
                    )
                    code_seed = _seed(
                        plan.spec["seeds"]["codes"], table.name, "train", current_round
                    )
                    episode = _episode(table, query, code_seed, args.device)
                    model.train()
                    prepared = prepare_episode(model, *episode)
                    tick = time.monotonic()
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
                    update += 1
                    cursor = position + 1
                    if not _finite_state(model.state_dict()) or not _finite_state(
                        optimizer.state_dict()
                    ):
                        raise FloatingPointError("nonfinite parameters or optimizer state")
                    if args.device == "cuda:0":
                        torch.cuda.synchronize()
                    row = {"round": current_round + 1, "update": update, "table": table.name,
                           "loss": float(score.loss.detach()), "gradient_norm": float(norm),
                           "update_seconds": time.monotonic() - tick, "mask": mask_info,
                           "code_seed": code_seed}
                    if args.device == "cuda:0":
                        row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    curve.write(json.dumps(row, allow_nan=False) + "\n")
                    curve.flush()
                    _checkpoint(output / "checkpoint-progress.pt", model, optimizer,
                                plan.identity, current_round, update, cursor,
                                prior_elapsed + time.monotonic() - started, replace=True)
                    receipt.update(round=current_round, cursor=cursor, update=update)
                cursor = 0
                boundary = output / f"checkpoint-round-{current_round + 1:04d}.pt"
                _checkpoint(boundary, model, optimizer, plan.identity, current_round + 1,
                            update, cursor, prior_elapsed + time.monotonic() - started)
                metrics = _metrics(plan, model, args.device, eval_bank, prepared_cache)
                _write_json(output / f"metrics-round-{current_round + 1:04d}.json", metrics)
        receipt["round"] = stop_round
        receipt["cursor"] = 0
        receipt["update"] = update
        receipt["final"] = _metrics(plan, model, args.device, eval_bank, prepared_cache)
        receipt["outcome"] = (
            "completed" if stop_round == plan.spec["max_rounds"] else "segment_completed"
        )
    except WallLimit as error:
        receipt.update(outcome="wall_limit", error_type=type(error).__name__, error=str(error))
    except KeyboardInterrupt as error:
        receipt.update(outcome="interrupted", error_type=type(error).__name__, error=str(error))
    except Exception as error:
        receipt.update(outcome="failed", error_type=type(error).__name__, error=str(error))
    finally:
        signal.signal(signal.SIGTERM, previous_signal)
        if receipt.get("outcome") in ("completed", "segment_completed") and stop_round is not None:
            receipt["round"] = stop_round
            receipt["cursor"] = 0
        else:
            receipt["round"] = round_index
            receipt["cursor"] = cursor
        receipt["update"] = update if model is not None else receipt.get("update", 0)
        receipt["elapsed_seconds"] = prior_elapsed + time.monotonic() - started \
            if model is not None else time.monotonic() - started
        if model is not None and optimizer is not None:
            try:
                _checkpoint(output / "checkpoint.pt", model, optimizer, plan.identity,
                            receipt["round"], receipt["update"], receipt["cursor"],
                            receipt["elapsed_seconds"])
                receipt["checkpoint"] = "checkpoint.pt"
            except Exception as error:
                receipt["checkpoint_error_type"] = type(error).__name__
                receipt["outcome"] = "failed"
        if args.device == "cuda:0" and receipt.get("execution_started"):
            receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        _write_json(output / "terminal.json", receipt)
    return receipt


__all__ = ["SCHEMA", "prepare_plan", "run_joint_fit"]
