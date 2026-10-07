"""Bounded, YAML-configured V7 training on typed tables; no implicit device fallback."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .config import V7Config, _version_defaults
from .evaluation import evaluate_joint_task
from .masking import MaskingSpec, sample_task
from .model import V7Model
from .runner import (
    OptimizerSpec,
    checkpoint_state,
    load_checkpoint,
    make_optimizer,
    save_checkpoint,
    train_step,
)
from .tables import load_typed_table, table_mask_task

RUN_SCHEMA = "tabu.restoration.v7-fit.v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_digest():
    root = Path(__file__).resolve().parents[2]
    paths = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _known(values, allowed, label):
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(values) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)}")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _json(path, payload):
    with Path(path).open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def resolve_run(path):
    """Validate inputs and resolve all paths against the YAML directory."""
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    _known(
        cfg,
        {
            "schema",
            "tables",
            "model",
            "masking",
            "optimizer",
            "steps",
            "seed",
            "device",
            "dtype",
            "window_rows",
            "query_rows",
            "eval_query_rows",
            "output_dir",
            "init_checkpoint",
            "grad_clip_norm",
            "num_threads",
        },
        "run",
    )
    if cfg.get("schema") != RUN_SCHEMA:
        raise ValueError(f"schema must be {RUN_SCHEMA}")

    def absolute(value):
        return (path.parent / value).resolve()

    for required in (
        "steps",
        "window_rows",
        "query_rows",
        "device",
        "dtype",
        "tables",
        "output_dir",
    ):
        if required not in cfg:
            raise ValueError(f"missing required run field: {required}")
    for key in ("steps", "window_rows", "query_rows"):
        _positive_int(cfg[key], key)
    seed = cfg.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    threads = _positive_int(cfg.get("num_threads", 2), "num_threads")
    eval_rows = _positive_int(cfg.get("eval_query_rows", cfg["query_rows"]), "eval_query_rows")
    if cfg["window_rows"] - cfg["query_rows"] < 2:
        raise ValueError("window_rows must leave at least two supports after query_rows")
    device = torch.device(cfg["device"])
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    if cfg["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    dtype = getattr(torch, cfg["dtype"])
    if device.type == "mps" and dtype != torch.float32:
        raise ValueError("MPS requires explicit float32")
    clip = cfg.get("grad_clip_norm", 1.0)
    if clip is not None and (isinstance(clip, bool) or not math.isfinite(clip) or clip <= 0):
        raise ValueError("grad_clip_norm must be positive and finite, or null")
    _known(cfg.get("masking", {}), MaskingSpec.__dataclass_fields__, "masking")
    _known(cfg.get("optimizer", {}), OptimizerSpec.__dataclass_fields__, "optimizer")
    masking = MaskingSpec(**cfg.get("masking", {}))
    optimizer = OptimizerSpec(**cfg.get("optimizer", {}))
    if not isinstance(cfg["tables"], list) or not cfg["tables"]:
        raise ValueError("tables must be a nonempty list")
    tables, table_masks = [], []
    for item in cfg["tables"]:
        _known(item, {"path", "name", "target", "sha256", "columns"}, "table")
        table = load_typed_table(
            absolute(item["path"]),
            target=item.get("target"),
            name=item.get("name"),
            expected_sha256=item.get("sha256"),
        )
        if min(cfg["window_rows"], len(table.train_rows)) - cfg["query_rows"] < 2:
            raise ValueError(f"{table.name}: insufficient training rows for the requested window")
        spec = MaskingSpec(
            **(asdict(masking) | ({"columns": item["columns"]} if "columns" in item else {}))
        )
        spec.eligible(table)
        tables.append(table)
        table_masks.append(spec)
    if len({table.name for table in tables}) != len(tables):
        raise ValueError("table names must be unique")
    init_path = absolute(cfg["init_checkpoint"]) if cfg.get("init_checkpoint") else None
    parent = torch.load(init_path, map_location="cpu", weights_only=True) if init_path else None
    if parent is not None and not all(key in parent for key in ("config", "model")):
        raise ValueError("weights-only initialization requires a V7 config and model state")
    overrides = cfg.get("model", {})
    _known(overrides, V7Config.__dataclass_fields__, "model")
    parent_config = V7Config.from_dict(parent["config"]) if parent else None
    if parent_config is not None and (
        parent.get("model_version", parent_config.model_version) != parent_config.model_version
    ):
        raise ValueError("initial checkpoint model_version differs from its stored ModelSpec")
    version = overrides.get(
        "model_version", parent_config.model_version if parent_config is not None else "v7"
    )
    if parent_config is None:
        model_values = V7Config.for_version(version).as_dict()
    else:
        model_values = parent_config.as_dict()
        if version != parent_config.model_version:
            # Preserve the parent's parameter layout. Unit depth and sharing
            # change state keys; for unshared rounds, K also counts modules.
            # Shared recurrence can adopt the target version's default K.
            version_defaults = _version_defaults(version)
            version_defaults.pop("unit_layers")
            version_defaults.pop("share_rounds")
            if not parent_config.share_rounds:
                version_defaults.pop("rounds")
            model_values.update(version_defaults)
    model = V7Config.from_dict(model_values | overrides)
    if parent_config is not None and parent_config.value_encoder != model.value_encoder:
        # Strict all-key initialization cannot change the value encoder.
        raise ValueError(
            "value_encoder differs from init_checkpoint; use the explicit weights-only "
            "migration row_dual_stream_from_checkpoint or row_reversible64_from_checkpoint and initialize from its checkpoint"
        )
    init = (
        None
        if parent is None
        else dict(
            path=str(init_path),
            sha256=_sha(init_path),
            kind="weights-only; fresh optimizer/RNG/sampler",
            source_model_version=parent_config.model_version,
            target_model_version=model.model_version,
            source_config=parent["config"],
            configuration_changes={
                name: {"from": value, "to": model.as_dict()[name]}
                for name, value in parent_config.as_dict().items()
                if value != model.as_dict()[name]
            },
            weight_transfer_plan={
                "policy": "strict-all-keys; reject any missing, extra or incompatible tensor",
                "inherit_state_keys": sorted(parent["model"]),
                "reinitialize_state_keys": [],
                "optimizer_rng_sampler": "fresh",
            },
        )
    )
    manifest = dict(
        schema=RUN_SCHEMA,
        model=model.as_dict(),
        tables=[
            dict(
                name=t.name,
                sha256=t.sha256,
                target=t.target,
                train_rows=list(t.train_rows),
                test_rows=list(t.test_rows),
                masking=asdict(s),
            )
            for t, s in zip(tables, table_masks, strict=True)
        ],
        optimizer=asdict(optimizer),
        seed=seed,
        window_rows=cfg["window_rows"],
        query_rows=cfg["query_rows"],
        eval_query_rows=eval_rows,
        grad_clip_norm=clip,
        device=str(device),
        dtype=str(dtype),
        num_threads=threads,
        initialization=init,
        sampler="v7-mask-v1; shuffled equal table visits; independent addressable episode draws",
        source_sha256=_source_digest(),
        evaluation=(
            "fixed same-table heldout rows; target-only and configured joint mask; no selection"
        ),
    )
    return dict(
        manifest=manifest,
        model=model,
        tables=tables,
        masks=table_masks,
        optimizer=optimizer,
        parent=parent,
        device=device,
        dtype=dtype,
        output=absolute(cfg["output_dir"]),
        steps=cfg["steps"],
        seed=seed,
    )


def _table_index(seed, step, count):
    cycle, index = divmod(step, count)
    order = list(range(count))
    random.Random(f"v7-tables-v1:{seed}:{cycle}").shuffle(order)
    return order[index]


def _evaluate(model, run):
    reports = []
    manifest = run["manifest"]
    for table, spec in zip(run["tables"], run["masks"], strict=True):
        if not table.test_rows:
            reports.append(dict(table=table.name, status="no-heldout-rows"))
            continue
        rng = random.Random(f"v7-evaluation-v1:{run['seed']}:{table.name}")
        supports = rng.sample(
            table.train_rows,
            min(len(table.train_rows), manifest["window_rows"] - manifest["query_rows"]),
        )
        queries = rng.sample(
            table.test_rows, min(len(table.test_rows), manifest["eval_query_rows"])
        )
        eligible = spec.eligible(table)
        joint = tuple(
            sorted(rng.sample(eligible, min(len(eligible), spec.joint_columns or len(eligible))))
        )
        masks = {"target_only": (table.target,)}
        if len(joint) > 1:
            masks["joint"] = joint
        code_seed, donor_seed = rng.randrange(2**63), rng.randrange(2**63)
        for mode, columns in masks.items():
            task = table_mask_task(
                table,
                supports + queries,
                {c: queries for c in columns},
                code_seed=code_seed,
                donor_seed=donor_seed,
                device=run["device"],
                dtype=run["dtype"],
            )
            result = evaluate_joint_task(model, task)
            reports.append(
                dict(
                    table=table.name,
                    mode=mode,
                    support_rows=supports,
                    query_rows=queries,
                    code_seed=code_seed,
                    donor_seed=donor_seed,
                    columns={
                        str(c): dict(
                            name=table.schema[c].key,
                            kind=r.kind,
                            status=r.status,
                            code_losses=r.code_losses,
                            state_changes=r.state_changes,
                            metrics=r.metrics,
                            baselines=r.baselines,
                            predictions=[p.detach().cpu().tolist() for p in r.predictions],
                            candidates=None
                            if r.candidates is None
                            else r.candidates.cpu().tolist(),
                            probabilities=None
                            if r.probabilities is None
                            else r.probabilities.cpu().tolist(),
                            probability_temperature=r.probability_temperature,
                        )
                        for c, r in result.columns.items()
                    },
                )
            )
    return reports


def run_fit(config_path, *, execute=False, resume=None, output_dir=None):
    run = resolve_run(config_path)
    manifest = run["manifest"]
    output = Path(output_dir).resolve() if output_dir else run["output"]
    if not execute:
        return dict(
            status="validated-not-run",
            output_dir=str(output),
            steps=run["steps"],
            resume=str(Path(resume).resolve()) if resume else None,
            manifest=manifest,
        )
    device = run["device"]
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable; no fallback")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable; no fallback")
    torch.set_num_threads(manifest["num_threads"])
    torch.manual_seed(run["seed"])
    model = V7Model(run["model"]).to(device=device, dtype=run["dtype"])
    if run["parent"] is not None:
        model.load_state_dict(run["parent"]["model"], strict=True)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError("initial checkpoint contains nonfinite weights")
    optimizer = make_optimizer(model, run["optimizer"])
    start = 0
    totals = dict(single_episodes=0, joint_episodes=0, query_cells=0)
    if resume:
        start = load_checkpoint(resume, model, optimizer, manifest=manifest)
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        totals = previous["sampling_totals"].copy()
    if run["steps"] <= start:
        raise ValueError("steps must exceed the resumed checkpoint step")
    output.mkdir(parents=True, exist_ok=False)
    _json(
        output / "resolved-config.json",
        dict(
            manifest=manifest,
            steps=run["steps"],
            resume=str(Path(resume).resolve()) if resume else None,
            start_step=start,
        ),
    )
    step = start
    try:
        with (output / "updates.jsonl").open("x") as log:
            for index in range(start, run["steps"]):
                table_index = _table_index(run["seed"], index, len(run["tables"]))
                task, receipt = sample_task(
                    run["tables"][table_index],
                    run["masks"][table_index],
                    seed=run["seed"],
                    step=index,
                    window_rows=manifest["window_rows"],
                    query_rows=manifest["query_rows"],
                    device=device,
                    dtype=run["dtype"],
                )
                record = train_step(
                    model, optimizer, [task], grad_clip_norm=manifest["grad_clip_norm"]
                )
                step = index + 1
                totals[f"{receipt['mode']}_episodes"] += 1
                totals["query_cells"] += receipt["query_cells"]
                log.write(
                    json.dumps(dict(step=step, **receipt, **asdict(record)), allow_nan=False) + "\n"
                )
                log.flush()
        state = checkpoint_state(model, optimizer, step=step, manifest=manifest)
        state["sampling_totals"] = totals
        save_checkpoint(output / "final.pt", state)
        _json(output / "evaluation.json", _evaluate(model, run))
        result = dict(
            status="completed",
            step=step,
            sampling_totals=totals,
            checkpoint=str(output / "final.pt"),
            checkpoint_sha256=_sha(output / "final.pt"),
            evaluation=str(output / "evaluation.json"),
        )
        _json(output / "terminal.json", result)
        return result
    except Exception as error:
        _json(
            output / "terminal.json",
            dict(status="failed", completed_step=step, error=f"{type(error).__name__}: {error}"),
        )
        raise
