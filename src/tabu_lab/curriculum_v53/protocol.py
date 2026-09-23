"""Versioned portable curriculum plans and directly addressable schedules."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from tabu_lab.models.restoration_v53 import BackboneConfig, V53Config, V53LossConfig
from tabu_lab.restoration_masking import default_v53_query_guard, validate_numeric_query_guard
from tabu_lab.restoration_optimizers import OptimizerConfig

from . import loss_replay

SCHEMA = "tabu.curriculum.v53.v1"
V54_SCHEMA = "tabu.curriculum.v54.v1"
_SEEDS = {"model", "order", "masks", "codes", "windows", "evaluation"}
_TOP_FIELDS = {
    "schema", "experiment_id", "seeds", "model", "optimizer", "tables", "probes",
    "stages", "description",
}
_OPT_DEFAULTS = {
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "grad_clip": 1.0,
    "muon_momentum": 0.95,
    "muon_nesterov": True,
    "muon_ns_steps": 5,
    "muon_adjust_lr_fn": "match_rms_adamw",
}


@dataclass(frozen=True)
class Plan:
    spec: dict
    tables: tuple
    config: V53Config
    optimizer: OptimizerConfig
    identity: dict
    summary: dict
    path: Path


def _mapping(value: Any, name: str, allowed: set[str], required=()) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = set(value) - allowed
    missing = set(required) - set(value)
    if unknown:
        raise ValueError(f"{name}: unknown fields {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name}: missing fields {sorted(missing)}")
    return dict(value)


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, minimum=0.0, inclusive=False):
    if (isinstance(value, bool) or not isinstance(value, int | float)
            or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        relation = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be a finite number {relation} {minimum}")
    return float(value)


def _list(value, name, *, empty=False):
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if empty else 'nonempty '}list")
    return value


def _names(value, name, *, empty=False):
    values = [_name(item, name) for item in _list(value, name, empty=empty)]
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate names")
    return values


def _model(value, schema=SCHEMA):
    if schema == V54_SCHEMA:
        from tabu_lab.models.restoration_v54 import V54Config

        value = _mapping(value, "model", {field.name for field in fields(V54Config)})
        try:
            config = V54Config.from_dict(value)
        except (TypeError, AttributeError) as error:
            raise ValueError(f"invalid model configuration: {error}") from error
        if config.slope_source != "shared_ll":
            raise ValueError("curriculum requires serializable shared_ll; "
                             "feature slope is unsupported")
        return config
    value = _mapping(value, "model", {field.name for field in fields(V53Config)})
    backbone = _mapping(value.get("backbone", {}), "model.backbone",
                        {field.name for field in fields(BackboneConfig)})
    try:
        resolved = V53Config().as_dict()
        resolved.update(value)
        resolved["backbone"] = asdict(BackboneConfig(**backbone))
        config = V53Config.from_dict(resolved)
    except (TypeError, AttributeError) as error:
        raise ValueError(f"invalid model configuration: {error}") from error
    if config.codec_version not in (
        "legacy_v53", "unit_gaussian_v1", "unit_gaussian_v2", "constant_weight_v1"
    ):
        raise ValueError("V5.3 manifests require a V5.3 codec; use curriculum-v54 for composition")
    if config.slope_source != "shared_ll":
        raise ValueError("curriculum requires serializable shared_ll; feature slope is unsupported")
    return config


def _optimizer(value):
    value = _mapping(value, "optimizer", set(_OPT_DEFAULTS))
    resolved = dict(_OPT_DEFAULTS, **value)
    for key in ("learning_rate", "eps", "grad_clip"):
        resolved[key] = _number(resolved[key], f"optimizer.{key}")
    resolved["weight_decay"] = _number(
        resolved["weight_decay"], "optimizer.weight_decay", inclusive=True,
    )
    betas = resolved["betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.betas must contain two numbers in [0,1)")
    betas = tuple(_number(beta, "optimizer.betas", inclusive=True) for beta in betas)
    if any(beta >= 1 for beta in betas):
        raise ValueError("optimizer.betas must contain two numbers in [0,1)")
    resolved["betas"] = betas
    momentum = _number(resolved["muon_momentum"], "optimizer.muon_momentum", inclusive=True)
    if momentum >= 1:
        raise ValueError("optimizer.muon_momentum must be in [0,1)")
    resolved["muon_momentum"] = momentum
    if type(resolved["muon_nesterov"]) is not bool:
        raise ValueError("optimizer.muon_nesterov must be boolean")
    _integer(resolved["muon_ns_steps"], "optimizer.muon_ns_steps")
    if resolved["muon_adjust_lr_fn"] not in (None, "original", "match_rms_adamw"):
        raise ValueError("optimizer.muon_adjust_lr_fn is unsupported")
    return OptimizerConfig(**resolved)


def _recipe(value, name, *, default_kind=None):
    value = _mapping(value, name, {"kind", "fraction", "numeric_query_guard"},
                     {"fraction"} if default_kind is not None else {"kind", "fraction"})
    if default_kind is not None:
        value.setdefault("kind", default_kind)
    if value["kind"] not in ("random_cell", "supervised_row"):
        raise ValueError(f"{name}.kind must be random_cell or supervised_row")
    fraction = _number(value["fraction"], f"{name}.fraction")
    if fraction >= 1:
        raise ValueError(f"{name}.fraction must be strictly between zero and one")
    value["fraction"] = fraction
    if "numeric_query_guard" in value:
        guard = value["numeric_query_guard"]
        if value["kind"] != "random_cell" and guard != {"kind": "none"}:
            raise ValueError(f"{name}: numeric_query_guard only applies to random_cell")
        value["numeric_query_guard"] = (
            {"kind": "none"} if guard == {"kind": "none"}
            else validate_numeric_query_guard(guard)
        )
    else:
        value["numeric_query_guard"] = default_v53_query_guard(value["kind"])
    return value


def _table_entries(value):
    entries, seen = [], set()
    allowed = {"id", "path", "sha256", "cohort", "kind", "window_rows", "target_column", "role"}
    for raw in _list(value, "tables"):
        entry = _mapping(raw, "table", allowed, {"id", "path", "sha256", "cohort", "kind"})
        for key in ("id", "path", "cohort"):
            _name(entry[key], f"table.{key}")
        if entry["id"] in seen:
            raise ValueError(f"duplicate table id: {entry['id']}")
        seen.add(entry["id"])
        digest = entry["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
            raise ValueError("table.sha256 must be a SHA256 hex digest")
        entry["sha256"] = digest.lower()
        if entry["kind"] not in ("synthetic", "real"):
            raise ValueError("table.kind must be synthetic or real")
        entry.setdefault("role", "train")
        if entry["role"] not in ("train", "probe"):
            raise ValueError("table.role must be train or probe")
        if entry.get("window_rows") is not None:
            _integer(entry["window_rows"], "table.window_rows", minimum=3)
        if "target_column" in entry:
            _integer(entry["target_column"], "table.target_column", minimum=0)
        entries.append(entry)
    return entries


def _probes(value, cohorts, *, default_kind=None):
    probes, seen = [], set()
    allowed = {"name", "cohorts", "partition", "purpose", "recipe", "masks"}
    purposes = {"fit", "retention", "transfer", "validation", "final_test", "retrospective"}
    for raw in _list(value, "probes", empty=True):
        probe = _mapping(raw, "probe", allowed, allowed)
        name = _name(probe["name"], "probe.name")
        if name in seen:
            raise ValueError(f"duplicate probe name: {name}")
        seen.add(name)
        probe["cohorts"] = _names(probe["cohorts"], f"probe {name}.cohorts")
        if set(probe["cohorts"]) - cohorts:
            raise ValueError(f"probe {name} references unknown cohort")
        if probe["partition"] not in ("train", "validation", "test"):
            raise ValueError(f"probe {name}: invalid partition")
        if probe["purpose"] not in purposes:
            raise ValueError(f"probe {name}: invalid purpose")
        probe["recipe"] = _recipe(probe["recipe"], f"probe {name}.recipe",
                                   default_kind=default_kind)
        _integer(probe["masks"], f"probe {name}.masks")
        if probe["partition"] != "train" and (
            probe["recipe"]["kind"] != "supervised_row" or probe["masks"] != 1
        ):
            raise ValueError(f"probe {name}: reserved partitions require "
                             "supervised_row and masks=1")
        if probe["purpose"] == "final_test" and probe["partition"] != "test":
            raise ValueError(f"probe {name}: final_test requires the test partition")
        if probe["purpose"] == "validation" and probe["partition"] != "validation":
            raise ValueError(f"probe {name}: validation purpose requires the validation partition")
        if probe["purpose"] == "retrospective" and probe["partition"] != "test":
            raise ValueError(f"probe {name}: retrospective purpose requires the test partition")
        probes.append(probe)
    return probes


def _stages(value, tables, probes, *, default_kind=None):
    result, seen = [], set()
    probe_by_name = {probe["name"]: probe for probe in probes}
    cohort_kinds: dict[str, set[str]] = {}
    for table in tables:
        if table["role"] == "train":
            cohort_kinds.setdefault(table["cohort"], set()).add(table["kind"])
    required = {"name", "question", "max_updates", "max_seconds", "sampling", "recipe",
                "optimizer", "evaluate_every", "checkpoint_every", "probes"}
    allowed = required | {"gate", "loss", "loss_replay"}
    previous_optimizer = "adamw"
    for raw in _list(value, "stages"):
        stage = _mapping(raw, "stage", allowed, required)
        name = _name(stage["name"], "stage.name")
        if name in seen:
            raise ValueError(f"duplicate stage name: {name}")
        seen.add(name)
        _name(stage["question"], f"stage {name}.question")
        for key in ("max_updates", "evaluate_every", "checkpoint_every"):
            _integer(stage[key], f"stage {name}.{key}")
        stage["max_seconds"] = _number(stage["max_seconds"], f"stage {name}.max_seconds")
        sampling, sampled, kinds = [], set(), set()
        for item in _list(stage["sampling"], f"stage {name}.sampling"):
            item = _mapping(item, "sampling entry", {"cohort", "episodes"}, {"cohort", "episodes"})
            cohort = _name(item["cohort"], "sampling.cohort")
            if cohort in sampled:
                raise ValueError(f"stage {name}: duplicate sampling cohort {cohort}")
            if cohort not in cohort_kinds:
                raise ValueError(f"stage {name}: unknown or probe-only sampling cohort {cohort}")
            sampled.add(cohort)
            kinds.update(cohort_kinds[cohort])
            _integer(item["episodes"], "sampling.episodes")
            sampling.append(item)
        stage["sampling"] = sampling
        if "loss_replay" in stage:
            if default_kind != "supervised_row":
                raise ValueError("loss_replay requires a V5.4 plan")
            policy = _mapping(stage["loss_replay"], "loss_replay",
                              {"kind", "normal_max_updates", "start_normal_cursor",
                               "start_extra_updates"},
                              {"kind", "normal_max_updates"})
            extra_cycle = loss_replay.extras_per_cycle(policy["kind"])
            if policy["kind"] == loss_replay.V1:
                if "start_extra_updates" in policy:
                    raise ValueError("loss_replay v1 does not accept start_extra_updates")
                start_extra = 0
            else:
                if "start_extra_updates" not in policy:
                    raise ValueError("loss_replay v2 requires explicit start_extra_updates")
                start_extra = _integer(policy["start_extra_updates"],
                                       "start_extra_updates", minimum=0)
            normal = _integer(policy["normal_max_updates"], "normal_max_updates")
            start = _integer(policy.get("start_normal_cursor", 0),
                             "start_normal_cursor", minimum=0)
            if normal % 120 or start % 120 or start >= normal:
                raise ValueError("loss_replay needs complete remaining 120-table cycles")
            selected = [table for table in tables
                        if table["role"] == "train" and table["cohort"] in sampled]
            if (len(selected) != 120 or len(sampling) != 1
                    or sampling[0]["episodes"] != 120):
                raise ValueError("loss_replay requires one cohort with exactly 120 train tables")
            actual = normal + start_extra + (normal - start) // 120 * extra_cycle
            if stage["max_updates"] != actual:
                raise ValueError(f"loss_replay max_updates must be {actual} actual optimizer steps")
            policy["start_normal_cursor"] = start
            stage["loss_replay"] = policy
        recipes = _mapping(stage["recipe"], f"stage {name}.recipe", {"synthetic", "real"}, kinds)
        stage["recipe"] = {kind: _recipe(recipe, f"stage {name}.recipe.{kind}",
                                         default_kind=default_kind)
                           for kind, recipe in recipes.items()}
        if stage["optimizer"] not in ("adamw", "muon"):
            raise ValueError(f"stage {name}: optimizer must be adamw or muon")
        if previous_optimizer == "muon" and stage["optimizer"] == "adamw":
            raise ValueError("Muon to AdamW reversal is unsupported")
        previous_optimizer = stage["optimizer"]
        stage["probes"] = _names(stage["probes"], f"stage {name}.probes", empty=True)
        for probe_name in stage["probes"]:
            if probe_name not in probe_by_name:
                raise ValueError(f"stage {name}: unknown probe {probe_name}")
            probe = probe_by_name[probe_name]
            if probe["purpose"] == "final_test":
                raise ValueError("final_test probes are independent evaluate only")
            if probe["partition"] == "test" and probe["purpose"] != "retrospective":
                raise ValueError("stage test probes require explicit retrospective purpose")
        if "gate" in stage:
            gate = _mapping(stage["gate"], f"stage {name}.gate",
                            {"probe", "metric", "mode", "threshold"},
                            {"probe", "metric", "mode", "threshold"})
            if not isinstance(gate["probe"], str) or gate["probe"] not in stage["probes"]:
                raise ValueError(f"stage {name}: gate probe must appear in stage.probes")
            if probe_by_name[gate["probe"]]["partition"] != "validation":
                raise ValueError("stage gates require validation probes; "
                                 "train and test cannot gate")
            if gate["metric"] not in ("query_numeric_mse", "query_discrete_accuracy",
                                      "query_numeric_normalized_mse"):
                raise ValueError("unsupported gate metric")
            if gate["mode"] not in ("min", "max"):
                raise ValueError("gate.mode must be min or max")
            threshold = gate["threshold"]
            if (isinstance(threshold, bool) or not isinstance(threshold, int | float)
                    or not math.isfinite(threshold)):
                raise ValueError("gate.threshold must be finite")
            gate["threshold"] = float(threshold)
            stage["gate"] = gate
        loss = _mapping(stage.get("loss", {}), f"stage {name}.loss",
                        {"discrete_weight", "state_weights"})
        if "discrete_weight" in loss:
            loss["discrete_weight"] = _number(loss["discrete_weight"], "loss.discrete_weight")
        if loss.get("state_weights") is not None:
            weights = loss["state_weights"]
            if (not isinstance(weights, (list, tuple)) or len(weights) != 4
                    or any(isinstance(w, bool) or not isinstance(w, int | float) for w in weights)):
                raise ValueError("loss.state_weights must contain four nonnegative finite numbers")
            loss["state_weights"] = tuple(float(w) for w in weights)
        resolved_loss = V53LossConfig(**loss)
        if resolved_loss.state_weights is not None:
            retained, query, null, corrupted = resolved_loss.state_weights
            if null != 0 or corrupted != 0 or not (retained > 0 or query > 0):
                raise ValueError("curriculum loss supports retained/query states only: "
                                 "states 2/3 must be zero and states 0/1 need positive weight")
        stage["loss"] = asdict(resolved_loss)
        result.append(stage)
    if len(result) != 1 and any("loss_replay" in stage for stage in result):
        raise ValueError("loss_replay supports a single bounded stage")
    return result


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _source_digest(schema=SCHEMA):
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in (
        "cli.py", "restoration_masking.py", "restoration_optimizers.py", "tar_data.py",
    )]
    directories = ["curriculum_v53", "models/restoration_v53", "models/restoration"]
    if schema == V54_SCHEMA:
        directories.append("models/restoration_v54")
    for relative in directories:
        paths.extend(sorted((root / relative).rglob("*.py")))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(set(paths))}
    return {"files": hashes, "sha256": _digest(hashes)}


def _summary(spec):
    stages = []
    for stage in spec["stages"]:
        length = sum(item["episodes"] for item in stage["sampling"])
        exposure = []
        for item in stage["sampling"]:
            cohort = item["cohort"]
            tables = [table for table in spec["tables"]
                      if table["cohort"] == cohort and table["role"] == "train"]
            exposure.append({"cohort": cohort, "train_tables": len(tables),
                             "episodes_per_cycle": item["episodes"],
                             "expected_episode_share": item["episodes"] / length})
        stages.append({"name": stage["name"], "question": stage["question"],
                       "max_updates": stage["max_updates"], "max_seconds": stage["max_seconds"],
                       "cycle_updates": length, "exposure": exposure,
                       "optimizer": stage["optimizer"], "loss": stage["loss"],
                       "probes": stage["probes"], "gate": stage.get("gate")})
        if "loss_replay" in stage:
            policy = stage["loss_replay"]
            extra_cycle = loss_replay.extras_per_cycle(policy["kind"])
            stages[-1].update(loss_replay=policy, normal_cycle_updates=120,
                              extra_cycle_updates=extra_cycle,
                              actual_cycle_updates=120 + extra_cycle,
                              extra_max_updates=stage["max_updates"]
                              - policy["normal_max_updates"])
            if policy["kind"] == loss_replay.V2:
                stages[-1]["new_extra_max_updates"] = (
                    stage["max_updates"] - policy["normal_max_updates"]
                    - policy["start_extra_updates"])
    return {
        "schema": spec["schema"], "status": "local_unissued", "outcome": "planned",
        "execution_started": False, "experiment_id": spec["experiment_id"],
        "model": spec["model"], "optimizer": spec["optimizer"], "stages": stages,
        "table_count": len(spec["tables"]), "probes": spec["probes"],
        "codec_boundary": {
            "version": spec["model"]["codec_version"],
            "numeric_scaling": spec["model"]["numeric_scaling"],
            "statistics": "forward-visible values only, with epsilon scale floor",
            "ordinal_domain": {
                "legacy_v53": "visible class codes",
                "unit_gaussian_v1": "complete declared ranks; shared origin",
                "unit_gaussian_v2": "complete declared identity-plus-rank codes",
                "constant_weight_v1": "complete declared identity-plus-rank codes",
                "unit_gaussian_composition_v1": "affine composition of column identity and rank",
                "constant_weight_composition_v1": "affine composition of column identity and rank",
            }[spec["model"]["codec_version"]],
            "scope": "shared input/answer codec within each episode",
        },
        "training_boundary": "train-role tables and train rows only; no reserved truth in forward",
        "probe_boundary": "train fit is not held-out evidence; validation gates; "
                          "test is independent final evaluation "
                          "or explicitly retrospective monitoring",
        "evaluation_loss": asdict(V53LossConfig()),
    }


def load_plan(path: Path | str, *, expected_schema: str = SCHEMA) -> Plan:
    """Read and validate a local manifest without running a model or downloading data."""
    from .data import data_fingerprint, load_table, table_summary

    path = Path(path).resolve()
    raw = path.read_text(encoding="utf-8")
    # PyYAML's YAML 1.1 resolver misreads JSON numbers such as 1e-06 as strings.
    document = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    spec = _mapping(document, "manifest", _TOP_FIELDS,
                    {"schema", "experiment_id", "seeds", "tables", "probes", "stages"})
    if expected_schema not in (SCHEMA, V54_SCHEMA):
        raise ValueError("unsupported curriculum schema")
    if spec["schema"] != expected_schema:
        raise ValueError(f"manifest.schema must be {expected_schema}")
    _name(spec["experiment_id"], "experiment_id")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError("description must be a string")
    spec["seeds"] = _mapping(spec["seeds"], "seeds", _SEEDS, _SEEDS)
    for name, seed in spec["seeds"].items():
        _integer(seed, f"seeds.{name}", minimum=0)
    config = _model(spec.get("model", {}), expected_schema)
    optimizer = _optimizer(spec.get("optimizer", {}))
    spec["model"], spec["optimizer"] = config.as_dict(), asdict(optimizer)
    spec["tables"] = _table_entries(spec["tables"])
    cohorts = {entry["cohort"] for entry in spec["tables"]}
    default_kind = "supervised_row" if expected_schema == V54_SCHEMA else None
    spec["probes"] = _probes(spec["probes"], cohorts, default_kind=default_kind)
    spec["stages"] = _stages(spec["stages"], spec["tables"], spec["probes"],
                              default_kind=default_kind)
    # JSON-native normalized output also removes mutable caller-owned tuples/lists.
    spec = json.loads(json.dumps(spec, allow_nan=False))
    tables = tuple(load_table(entry, path.parent) for entry in spec["tables"])
    for entry, table in zip(spec["tables"], tables, strict=True):
        entry["target_column"] = table.target_column
        entry["window_rows"] = table.window_rows
    for probe in spec["probes"]:
        if probe["partition"] == "train":
            continue
        for table in tables:
            if table.cohort in probe["cohorts"]:
                if not table.holdout_row_ids[probe["partition"]]:
                    raise ValueError(f"probe {probe['name']}: table {table.name} has no "
                                     f"{probe['partition']} rows")
                if table.window_rows is not None:
                    raise ValueError("reserved probes require full context without window_rows")
    portable = dict(spec)
    portable["tables"] = [{key: value for key, value in entry.items() if key != "path"}
                          for entry in spec["tables"]]
    identity = {
        "schema": expected_schema, "manifest_sha256": _digest(portable),
        "source": (_source_digest() if expected_schema == SCHEMA
                   else _source_digest(expected_schema)),
        "datasets": {entry["id"]: entry["sha256"] for entry in spec["tables"]},
        "data_sha256": data_fingerprint(tables),
    }
    identity["sha256"] = _digest(identity)
    summary = _summary(spec)
    summary["tables"] = [
        {key: value for key, value in table_summary(table).items() if key != "path"}
        for table in tables
    ]
    return Plan(spec, tables, config, optimizer, identity, summary, path)


def load_v54_plan(path: Path | str) -> Plan:
    """Load only V5.4 manifests, with its explicit model and recipe defaults."""
    return load_plan(path, expected_schema=V54_SCHEMA)


def _seed(*parts):
    digest = hashlib.sha256(json.dumps(parts, ensure_ascii=True).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def schedule_entry(plan: Plan, stage_index: int, cursor: int):
    """Return (table, that table's episode index) without replaying earlier updates.

    A cohort has a fixed seeded table permutation for this stage. Its slots walk
    that permutation continuously across cycles, so e.g. 30 of 120 tables cover
    the complete cohort in four cycles. Only the current cycle is shuffled.
    """
    _integer(stage_index, "stage_index", minimum=0)
    _integer(cursor, "cursor", minimum=0)
    if stage_index >= len(plan.spec["stages"]):
        raise ValueError("stage_index is out of bounds")
    stage = plan.spec["stages"][stage_index]
    if cursor >= stage["max_updates"]:
        raise ValueError("cursor is outside the stage update budget")
    cycle_size = sum(item["episodes"] for item in stage["sampling"])
    cycle, position = divmod(cursor, cycle_size)
    seed = plan.spec["seeds"]["order"]
    entries = []
    for item in stage["sampling"]:
        cohort = item["cohort"]
        tables = sorted((table for table in plan.tables
                         if table.cohort == cohort and table.role == "train"),
                        key=lambda table: table.name)
        if not tables:
            raise ValueError(f"sampling cohort has no train tables: {cohort}")
        random.Random(_seed(seed, stage["name"], cohort, "cohort_order")).shuffle(tables)
        for offset in range(item["episodes"]):
            absolute = cycle * item["episodes"] + offset
            entries.append((tables[absolute % len(tables)], absolute // len(tables)))
    random.Random(_seed(seed, stage["name"], cycle, "cycle_order")).shuffle(entries)
    return entries[position]


__all__ = ["SCHEMA", "V54_SCHEMA", "Plan", "load_plan", "load_v54_plan", "schedule_entry"]
