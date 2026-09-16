"""Bounded numeric restoration fit diagnostic; execution is an explicit opt-in.

Training episodes come from the fixed mask bank (default) or from TAR-style
stateless resampled supervised episodes (``training_episode_mode =
"resampled_supervised"``): update ``i`` then reproduces TAR joint-fit round
``i`` for this table (same episode seed stream, context/query split and
codebook seed). Initial/final scores always use the fixed mask bank on sampled
training rows. With ``reserved_test`` enabled, the snapshot's reserved rows
additionally appear in exactly one forward-only evaluation episode (target
column queried, predictors visible); they never enter a training episode or a
gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections import OrderedDict
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

SCHEMA = "tabu.restoration.pilot-fit.v1"
PROTOCOL = "fixed_train_rows_fixed_cell_masks_numeric_all_observed_targets_v1"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_spec(path):
    text = Path(path).read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _execution_dtype(spec):
    execution = spec.get("execution", {})
    if not isinstance(execution, dict) or set(execution) - {"dtype"}:
        raise ValueError("execution may only declare dtype")
    dtype = execution.get("dtype", "float64")
    if dtype not in ("float64", "float32"):
        raise ValueError("execution dtype must be float64 or float32")
    return dtype


def _execution_identity(device, dtype="float64"):
    return {
        "device": device,
        "dtype": dtype,
        "torch": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
    }


def _configure_backend():
    if torch.cuda.is_initialized() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA already initialized without the declared deterministic workspace")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


def source_identity():
    root = Path(__file__).parent
    paths = [Path(__file__), root / "cli.py", root / "tar_data.py"]
    paths += sorted((root / "models" / "restoration").glob("*.py"))
    files = {str(path.relative_to(root)): _hash(path) for path in paths}
    return {"files": files, "sha256": _digest(files)}


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"invalid {name}")
    return float(value)


def _resampled_supervised_query(count, width, hidden, seed, namespace, episode_id):
    """Reproduce TAR joint-fit training episode ``episode_id`` for this table.

    Same stateless stream as ``models.tar.episodes.sample_supervised_episode``:
    one randperm keyed by (seed, namespace, episode_id, "row_roles"), the first
    ``count - hidden`` rows form the visible context, the remaining ``hidden``
    rows are queried on the final (target) column only; the codebook seed is the
    default "codebook" stream of the same episode id.
    """
    from tabu_lab.models.tar.episodes import episode_seed

    generator = torch.Generator().manual_seed(
        episode_seed(seed, namespace, episode_id, "row_roles")
    )
    order = torch.randperm(count, generator=generator).tolist()
    query = torch.zeros(count, width, dtype=torch.bool)
    for row in order[count - hidden :]:
        query[row, width - 1] = True
    return query, episode_seed(seed, namespace, episode_id)


@dataclass(frozen=True)
class FitPlan:
    spec: dict
    config: RestorationConfig
    schema: tuple
    values: tuple
    queries: tuple
    code_seeds: tuple
    dtype: torch.dtype
    identity: dict
    summary: dict
    training_mode: str = "fixed_bank"
    evaluate_every: int | None = None
    test: dict | None = None

    def episode(self, index, device="cpu"):
        bank_index = index % len(self.queries)
        query = self.queries[bank_index].to(device)
        return make_episode(
            self.schema,
            tuple(value.to(device) for value in self.values),
            torch.ones_like(query),
            query,
            code_seed=self.code_seeds[bank_index],
        )

    def training_episode(self, index, device="cpu"):
        """The episode consumed by update ``index`` (sampler cursor value)."""
        if self.training_mode == "fixed_bank":
            return self.episode(index, device)
        count = len(self.values[0])
        hidden = math.ceil(count * self.spec["mask_fraction"])
        query, code_seed = _resampled_supervised_query(
            count,
            len(self.schema),
            hidden,
            _integer(self.spec.get("episode_seed"), "episode_seed", 0),
            self.spec["training_episode_namespace"],
            index,
        )
        query = query.to(device)
        return make_episode(
            self.schema,
            tuple(value.to(device) for value in self.values),
            torch.ones_like(query),
            query,
            code_seed=code_seed,
        )

    def test_episode(self, device="cpu"):
        """The single forward-only reserved-row episode (target column queried)."""
        if self.test is None:
            raise ValueError("this plan declares no reserved-row test episode")
        query = self.test["query"].to(device)
        return make_episode(
            self.schema,
            tuple(value.to(device) for value in self.test["values"]),
            torch.ones_like(query),
            query,
            code_seed=self.test["code_seed"],
        )


class _PreparedBank:
    """Run-local LRU of at most eight fixed mask/seed episodes on one device.

    No learned state or autograd graph is cached. This is rebuilt on resume;
    only the existing sampler cursor and mask/code identities are checkpointed.
    """

    def __init__(self, plan, model, device, capacity=8):
        self.plan, self.model, self.device = plan, model, device
        self.capacity = capacity
        self.values = tuple(value.to(device) for value in plan.values)
        self.entries = OrderedDict()

    def get(self, index):
        key = index % len(self.plan.queries)
        if key not in self.entries:
            query = self.plan.queries[key].to(self.device)
            episode = make_episode(
                self.plan.schema, self.values, torch.ones_like(query), query,
                code_seed=self.plan.code_seeds[key],
            )
            prepared = prepare_episode(self.model, *episode)
            if len(self.entries) == self.capacity:
                self.entries.popitem(last=False)
            self.entries[key] = prepared
        self.entries.move_to_end(key)
        return self.entries[key]


class _ResampledEpisodes:
    """Stateless TAR-style training episodes; each cursor value is used once.

    Episode masks and codebook seeds are pure functions of the sampler cursor,
    so resume just regenerates the same episode from the checkpointed cursor.
    """

    def __init__(self, plan, model, device):
        self.plan, self.model, self.device = plan, model, device

    def get(self, index):
        return prepare_episode(self.model, *self.plan.training_episode(index, self.device))


def prepare_plan(preregistration, dataset, device="cpu"):
    if device not in ("cpu", "cuda:0"):
        raise ValueError("device must be cpu or cuda:0")
    spec = _load_spec(preregistration)
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise ValueError(f"preregistration schema must be {SCHEMA}")
    if spec.get("status") != "local_unissued":
        raise ValueError("this runner only supports local_unissued diagnostics")
    if _hash(dataset) != spec.get("dataset_sha256"):
        raise ValueError("dataset snapshot digest mismatch")
    data = json.loads(Path(dataset).read_text())
    coverage = validate_full_dataset(data, _integer(spec.get("expected_rows"), "expected_rows"))
    rows = data.get("values")
    if not rows or not isinstance(rows[0], list) or not rows[0]:
        raise ValueError("snapshot needs a nonempty values matrix")
    width = len(rows[0])
    if any(not isinstance(row, list) or len(row) != width for row in rows):
        raise ValueError("snapshot values must be rectangular")
    features = data.get("features")
    if features is None:
        # Exact legacy snapshot convention: all predictors numeric, final target typed.
        features = [{"kind": "numeric", "domain": []} for _ in range(width - 1)]
        features.append({"kind": data.get("target_kind"), "domain": data.get("domain")})
    if not isinstance(features, list) or len(features) != width:
        raise ValueError("snapshot feature schema width mismatch")
    for feature in features:
        if not isinstance(feature, dict) or feature.get("kind") not in (
            "numeric",
            "nominal",
            "ordinal",
        ):
            raise ValueError("snapshot has an invalid feature kind")
    if "target_kind" in data and features[-1]["kind"] != data["target_kind"]:
        raise ValueError("snapshot target metadata contradicts feature schema")
    if "domain" in data and features[-1].get("domain", []) != data["domain"]:
        raise ValueError("snapshot target domain contradicts feature schema")
    columns = spec.get("columns")
    if (
        not isinstance(columns, list)
        or not columns
        or any(type(col) is not int or not 0 <= col < width for col in columns)
        or len(set(columns)) != len(columns)
    ):
        raise ValueError("columns must explicitly list distinct valid numeric column indices")
    if any(features[col]["kind"] != "numeric" for col in columns):
        raise ValueError("pilot columns must all be numeric; categorical columns are not dropped")
    count = _integer(spec.get("row_count"), "row_count", 3)
    if count > len(data["splits"]["train"]):
        raise ValueError("row_count exceeds training split; no reserved-row fallback")
    seeds = spec.get("seeds")
    if not isinstance(seeds, dict) or set(seeds) != {"model", "rows", "masks", "codes"}:
        raise ValueError("seeds must declare model, rows, masks and codes")
    for name, seed in seeds.items():
        _integer(seed, f"seeds.{name}", 0)
        if seed >= 2**63:
            raise ValueError("seeds must be below 2**63")
    fraction = _positive(spec.get("mask_fraction"), "mask_fraction")
    hidden = math.ceil(count * fraction)
    if not 0 < fraction < 1 or hidden > count - 2:
        raise ValueError("mask_fraction must leave at least two visible supports per column")
    bank_size = _integer(spec.get("mask_bank_size"), "mask_bank_size")
    _integer(spec.get("max_updates"), "max_updates")
    _positive(spec.get("max_wall_seconds"), "max_wall_seconds")
    _integer(spec.get("gradient_accumulation", 1), "gradient_accumulation")
    _integer(spec.get("checkpoint_every", 1), "checkpoint_every")
    evaluate_every = spec.get("evaluate_every")
    if evaluate_every is not None:
        _integer(evaluate_every, "evaluate_every")
    training_mode = spec.get("training_episode_mode", "fixed_bank")
    if training_mode not in ("fixed_bank", "resampled_supervised"):
        raise ValueError("training_episode_mode must be fixed_bank or resampled_supervised")
    training_namespace = spec.get("training_episode_namespace")
    if training_mode == "resampled_supervised":
        if not isinstance(training_namespace, str) or not training_namespace:
            raise ValueError("resampled_supervised requires a training_episode_namespace")
        _integer(spec.get("episode_seed"), "episode_seed", 0)
    elif training_namespace is not None:
        raise ValueError("training_episode_namespace requires resampled_supervised")
    reserved_test = spec.get("reserved_test", False)
    if type(reserved_test) is not bool:
        raise ValueError("reserved_test must be a boolean")
    config = RestorationConfig.from_dict(spec["model"])
    exec_dtype = _execution_dtype(spec)
    optimizer = spec.get("optimizer")
    if not isinstance(optimizer, dict) or optimizer.get("kind") != "adamw":
        raise ValueError("pilot optimizer must be explicit adamw")
    allowed = {"kind", "learning_rate", "weight_decay", "betas", "eps", "grad_clip"}
    if set(optimizer) != allowed:
        raise ValueError(
            "optimizer must declare kind, learning_rate, weight_decay, betas, eps, grad_clip"
        )
    _positive(optimizer["learning_rate"], "learning_rate")
    _positive(optimizer["weight_decay"], "weight_decay", zero=True)
    _positive(optimizer["eps"], "eps")
    _positive(optimizer["grad_clip"], "grad_clip")
    betas = optimizer["betas"]
    if (
        not isinstance(betas, list)
        or len(betas) != 2
        or any(not 0 <= _positive(beta, "beta", zero=True) < 1 for beta in betas)
    ):
        raise ValueError("betas must be two finite values in [0,1)")
    row_order = spec.get("row_order", "shuffle_seeded")
    if row_order == "split":
        # Keep the dataset's train-split order, required for reproducing the
        # TAR covering-fit bank (TAR indexes its pool in split order).
        if count != len(data["splits"]["train"]):
            raise ValueError("row_order=split requires row_count to cover the full train split")
        order = list(range(len(data["splits"]["train"])))
    elif row_order == "shuffle_seeded":
        generator = torch.Generator().manual_seed(seeds["rows"])
        order = torch.randperm(len(data["splits"]["train"]), generator=generator)[:count].tolist()
    else:
        raise ValueError("row_order must be shuffle_seeded or split")
    row_ids = [data["splits"]["train"][index] for index in order]
    selected = [[rows[row][col] for col in columns] for row in row_ids]
    if any(
        isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
        for row in selected
        for value in row
    ):
        raise ValueError("selected numeric observations must be finite real values")
    matrix = torch.tensor(selected, dtype=torch.float64)
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("selected numeric values must fit FP64")
    schema = tuple(ColumnSchema(f"column-{col}", "numeric") for col in columns)
    values = tuple(matrix[:, index].clone() for index in range(len(columns)))
    mask_mode = spec.get("mask_mode", "cell_bank")
    if mask_mode == "tar_covering_fit_labels":
        # Supervised row masking is value-independent: unlike the current
        # random-cell recipe, it must not implicitly protect extreme labels.
        # Reproduce the TAR joint-fit evaluation bank exactly: cyclic
        # hidden-row chunks over a randperm keyed by
        # episode_seed(episode_seed, mask_namespace, 0, "row_roles"), with only
        # the final (target) column masked. Query row sets then coincide with
        # TAR's covering-fit episodes for the same table.
        from tabu_lab.models.tar.episodes import episode_seed

        episode_seed_value = _integer(spec.get("episode_seed"), "episode_seed", 0)
        namespace = spec.get("mask_namespace")
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("mask_mode=tar_covering_fit_labels requires a mask_namespace")
        generator = torch.Generator().manual_seed(
            episode_seed(episode_seed_value, namespace, 0, "row_roles")
        )
        order = torch.randperm(count, generator=generator).tolist()
        queries = []
        for index in range(bank_size):
            query = torch.zeros(count, len(columns), dtype=torch.bool)
            for offset in range(hidden):
                query[order[(index * hidden + offset) % count], len(columns) - 1] = True
            queries.append(query)
    elif mask_mode == "cell_bank":
        generator = torch.Generator().manual_seed(seeds["masks"])
        queries = []
        for _ in range(bank_size):
            query = torch.zeros(count, len(columns), dtype=torch.bool)
            for column in range(len(columns)):
                query[torch.randperm(count, generator=generator)[:hidden], column] = True
            queries.append(query)
    else:
        raise ValueError("mask_mode must be cell_bank or tar_covering_fit_labels")
    generator = torch.Generator().manual_seed(seeds["codes"])
    code_seeds = tuple(torch.randint(2**63 - 1, (bank_size,), generator=generator).tolist())
    test_pack = None
    if reserved_test:
        test_ids = list(data["splits"]["test"])
        panel_ids = list(row_ids) + test_ids
        panel = [[rows[row][col] for col in columns] for row in panel_ids]
        if any(
            isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            for row in panel
            for value in row
        ):
            raise ValueError("reserved-row test panel must be finite real values")
        panel_matrix = torch.tensor(panel, dtype=torch.float64)
        test_query = torch.zeros(len(panel_ids), len(columns), dtype=torch.bool)
        test_query[len(row_ids) :, len(columns) - 1] = True
        test_pack = {
            "values": tuple(panel_matrix[:, index].clone() for index in range(len(columns))),
            "query": test_query,
            # One more draw from the declared codes stream; bank seeds unchanged.
            "code_seed": int(torch.randint(2**63 - 1, (1,), generator=generator)[0]),
            "train_row_ids": list(row_ids),
            "test_row_ids": test_ids,
        }
    bank = [
        {"query": query.tolist(), "code_seed": seed}
        for query, seed in zip(queries, code_seeds, strict=True)
    ]
    identity = {
        "preregistration_sha256": _hash(preregistration),
        "dataset_sha256": _hash(dataset),
        "split_sha256": _digest(data["splits"]),
        "schema_sha256": _digest(features),
        "selected_columns": columns,
        "selected_train_row_ids": row_ids,
        "mask_bank_sha256": _digest(bank),
        "protocol": PROTOCOL,
        "source": source_identity(),
        "execution": _execution_identity(device, exec_dtype),
        "training_episode_mode": training_mode,
    }
    if training_mode == "resampled_supervised":
        identity["training_episode_namespace"] = training_namespace
    if test_pack is not None:
        identity["reserved_test"] = {
            "train_row_ids": test_pack["train_row_ids"],
            "test_row_ids": test_pack["test_row_ids"],
            "episode_sha256": _digest(
                {
                    "rows": panel_ids,
                    "columns": columns,
                    "query": test_pack["query"].tolist(),
                    "code_seed": test_pack["code_seed"],
                }
            ),
        }
    summary = {
        "status": "local_unissued",
        "outcome": "planned",
        "execution_started": False,
        "identity": identity,
        "coverage": coverage,
        "selected_rows": count,
        "selected_columns": columns,
        "excluded_columns": [i for i in range(width) if i not in columns],
        "row_order": row_order,
        "mask_mode": mask_mode,
        "training_episode_mode": training_mode,
        "evaluate_every": evaluate_every,
        "query_per_column": hidden,
        "visible_per_column": count - hidden,
        "mask_bank": bank,
        "reserved_rows_used": 0,
        "evaluation_scope": "fixed fit masks on training rows; no reserved evaluation"
        if test_pack is None
        else "fixed fit masks on training rows; reserved-row test episode is forward-only",
        "reserved_test": None
        if test_pack is None
        else {
            "panel_rows": len(panel_ids),
            "test_rows": len(test_ids),
            "query_cells": int(test_pack["query"].sum()),
            "queried_column": columns[-1],
            "scope": "forward-only evaluation; reserved rows never enter training episodes",
        },
        "cuda_hardware_identity": "bound during execute"
        if device == "cuda:0"
        else "not applicable",
        "max_updates": spec["max_updates"],
        "max_wall_seconds": spec["max_wall_seconds"],
        "model": config.as_dict(),
        "optimizer": optimizer,
    }
    return FitPlan(
        spec, config, schema, values, tuple(queries), code_seeds,
        getattr(torch, exec_dtype), identity, summary,
        training_mode, evaluate_every, test_pack,
    )


def _require_committed_preregistration(path):
    path = Path(path).resolve()
    try:
        root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path.parent,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        relative = path.relative_to(root).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError("execute requires a committed preregistration in its Git HEAD") from error
    if committed != path.read_bytes():
        raise ValueError("preregistration bytes differ from committed Git HEAD")


class WallLimit(RuntimeError):
    pass


def _check_wall(deadline):
    if time.monotonic() >= deadline:
        raise WallLimit("max_wall_seconds reached between bounded operations")


@torch.no_grad()
def _evaluate_episodes(model, episodes, deadline, scope):
    model.eval()
    totals = {
        name: {"count": 0, "encoding_sse": 0.0, "numeric_sse": 0.0}
        for name in ("retained", "query")
    }
    losses = []
    for episode in episodes:
        _check_wall(deadline)
        score = score_episode(model, *episode)
        if not bool(torch.isfinite(score.loss)):
            raise FloatingPointError("nonfinite evaluation loss")
        losses.append(float(score.loss))
        targets = episode[1].targets
        states = episode[2].states[targets[:, 0], targets[:, 1]]
        for state, name in enumerate(("retained", "query")):
            selected = states == state
            totals[name]["count"] += int(selected.sum())
            totals[name]["encoding_sse"] += float(score.per_target[selected].sum())
            for prediction in score.output.columns:
                positions = prediction.target_indices
                chosen = selected[positions]
                truth = episode[2].values[prediction.column][targets[positions, 0]]
                residual = prediction.decoded - truth
                squared_error = float(residual[chosen].square().sum())
                if not math.isfinite(squared_error):
                    raise FloatingPointError("nonfinite original-unit evaluation error")
                totals[name]["numeric_sse"] += squared_error
                if not math.isfinite(totals[name]["numeric_sse"]):
                    raise FloatingPointError("original-unit evaluation aggregation overflow")
    for result in totals.values():
        result["encoding_mse"] = result.pop("encoding_sse") / result["count"]
        result["numeric_mse"] = result.pop("numeric_sse") / result["count"]
    report = {
        "loss": sum(losses) / len(losses),
        "by_state": totals,
        "scope": scope,
    }
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as error:
        raise FloatingPointError("nonfinite evaluation report") from error
    return report


def evaluate(model, plan, device, deadline):
    """Fixed-bank fit diagnostic over the declared training rows."""
    episodes = [plan.episode(index, device) for index in range(len(plan.queries))]
    return _evaluate_episodes(model, episodes, deadline, plan.summary["evaluation_scope"])


def evaluate_test(model, plan, device, deadline):
    """Forward-only reserved-row test episode; None when not declared."""
    if plan.test is None:
        return None
    return _evaluate_episodes(
        model,
        [plan.test_episode(device)],
        deadline,
        plan.summary["reserved_test"]["scope"],
    )


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_copy(item) for item in value)
    return value


def _device_copy(value):
    """Structural clone that stays on the source device (no host sync)."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _device_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_device_copy(item) for item in value)
    return value


def _finite_state(value):
    """Same detection semantics as before, but tensor flags are accumulated
    device-side and reduced with a single host sync instead of one sync per
    tensor (matters on MPS/CUDA, where each bool() stalls the pipeline)."""
    tensor_flags = []
    python_finite = True

    def walk(item):
        nonlocal python_finite
        if isinstance(item, torch.Tensor):
            tensor_flags.append(torch.isfinite(item).all().reshape(()))
        elif isinstance(item, dict):
            for sub in item.values():
                walk(sub)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
        elif isinstance(item, float) and not math.isfinite(item):
            python_finite = False

    walk(value)
    if not python_finite:
        return False
    if not tensor_flags:
        return True
    devices = {flag.device for flag in tensor_flags}
    if len(devices) > 1:
        # Mixed-device state: fall back to per-tensor checks.
        return all(bool(flag) for flag in tensor_flags)
    return bool(torch.stack(tensor_flags).all())


def checkpoint_state(model, optimizer, plan, step, cursor, elapsed):
    # Capture only a fully completed, finite update. Assignment of the returned
    # independent snapshot is atomic; an interrupted copy preserves its predecessor.
    # The snapshot is cloned on-device (no host sync per update); the host copy
    # happens once inside save_checkpoint, i.e. at actual save points.
    return {
        "schema": "tabu.restoration.pilot-checkpoint.v1",
        "identity": plan.identity,
        "config": model.config.as_dict(),
        "model": _device_copy(model.state_dict()),
        "optimizer": _device_copy(optimizer.state_dict()),
        "step": step,
        "sampler_cursor": cursor,
        "elapsed_seconds": elapsed,
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all()
        if plan.identity["execution"]["device"] == "cuda:0"
        else [],
    }


def save_checkpoint(path, state):
    with Path(path).open("xb") as handle:
        torch.save(_cpu_copy(state), handle)
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path, model, optimizer, plan):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != "tabu.restoration.pilot-checkpoint.v1":
        raise ValueError("unsupported restoration checkpoint schema")
    if state.get("identity") != plan.identity or state.get("config") != plan.config.as_dict():
        raise ValueError(
            "checkpoint identity drift: preregistration, data, masks, source or execution"
        )
    step = _integer(state.get("step"), "checkpoint step", 0)
    cursor = _integer(state.get("sampler_cursor"), "checkpoint sampler cursor", 0)
    if step > plan.spec["max_updates"] or cursor != step * plan.spec.get(
        "gradient_accumulation", 1
    ):
        raise ValueError("checkpoint sampler cursor or update budget mismatch")
    elapsed = _positive(state.get("elapsed_seconds"), "checkpoint elapsed time", zero=True)
    if not _finite_state(state["model"]) or not _finite_state(state["optimizer"]):
        raise ValueError("checkpoint model or optimizer contains nonfinite state")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if plan.identity["execution"]["device"] == "cuda:0":
        if len(state["torch_cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
    elif state["torch_cuda_rng"]:
        raise ValueError("CPU checkpoint contains unexpected CUDA RNG state")
    return step, cursor, elapsed


def _wandb_start(args, plan, receipt):
    """Start a passive W&B mirror of the fit stream.

    updates.jsonl stays authoritative; the mirror never feeds back into
    training. Missing declaration or a missing wandb install is fatal (the
    caller asked explicitly); a mirror *runtime* failure is recorded in the
    receipt and training continues.
    """
    project = getattr(args, "wandb_project", None)
    if project is None:
        return None
    if plan.spec.get("telemetry", {}).get("wandb_mirror") is not True:
        raise ValueError("preregistration does not declare telemetry.wandb_mirror: true")
    try:
        import wandb
    except ImportError:
        raise RuntimeError("wandb mirror requested but wandb is not installed (extra: telemetry)")
    try:
        run = wandb.init(
            project=project,
            entity=getattr(args, "wandb_entity", None),
            name=f"{Path(args.preregistration).parent.name}-{plan.identity['preregistration_sha256'][:8]}",
            config={
                "preregistration_sha256": plan.identity["preregistration_sha256"],
                "dataset_sha256": plan.identity["dataset_sha256"],
                "mask_bank_sha256": plan.identity["mask_bank_sha256"],
                "execution": plan.identity["execution"],
                "max_updates": plan.spec["max_updates"],
            },
            tags=["restoration", "passive-mirror"],
        )
    except Exception as error:
        receipt["wandb"] = {"mode": "passive_mirror", "error_type": type(error).__name__}
        return None
    receipt["wandb"] = {"mode": "passive_mirror", "url": run.url, "id": run.id}
    return run


def _wandb_log_metrics(run, stage, metrics, step):
    run.log(
        {
            f"{stage}.loss": metrics["loss"],
            **{
                f"{stage}.{state}.encoding_mse": values["encoding_mse"]
                for state, values in metrics.get("by_state", {}).items()
                if "encoding_mse" in values
            },
        },
        step=step,
    )


def run_fit(args):
    output = Path(args.output_root)
    if output.exists():
        raise FileExistsError("output-root must be a new attempt directory")
    if not args.execute:
        plan = prepare_plan(args.preregistration, args.dataset, args.device)
        return dict(plan.summary, resume_requested=args.resume is not None)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": SCHEMA,
        "status": "local_unissued",
        "outcome": "started",
        "execution_started": False,
        "completed_updates": 0,
        "reserved_rows_used": 0,
    }
    started = time.monotonic()
    previous_signal = None
    wandb_run = None
    model = optimizer = plan = None
    boundary = None
    step = cursor = 0
    prior_elapsed = 0.0
    try:
        _json(output / "started.json", receipt)

        def interrupted(signum, frame):
            raise KeyboardInterrupt("received termination signal")

        previous_signal = signal.signal(signal.SIGTERM, interrupted)
        plan = prepare_plan(args.preregistration, args.dataset, args.device)
        receipt["identity"] = plan.identity
        receipt["training_episode_mode"] = plan.training_mode
        receipt["reserved_test_rows_evaluated"] = (
            len(plan.test["test_row_ids"]) if plan.test is not None else 0
        )
        _require_committed_preregistration(args.preregistration)
        _configure_backend()
        if args.device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        if args.device == "cuda:0":
            torch.cuda.reset_peak_memory_stats()
            receipt["cuda_device"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
            }
            plan.identity["execution"]["cuda_device"] = receipt["cuda_device"]
        _json(output / "resolved.json", {"preregistration": plan.spec, "plan": plan.summary})
        wandb_run = _wandb_start(args, plan, receipt)
        torch.manual_seed(plan.spec["seeds"]["model"])
        model = RestorationModel(plan.config).to(device=args.device, dtype=plan.dtype)
        cfg = plan.spec["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            betas=tuple(cfg["betas"]),
            eps=cfg["eps"],
        )
        if args.resume is not None:
            step, cursor, prior_elapsed = load_checkpoint(args.resume, model, optimizer, plan)
            receipt["parent_checkpoint_sha256"] = _hash(args.resume)
        boundary = checkpoint_state(
            model, optimizer, plan, step, cursor, prior_elapsed + time.monotonic() - started
        )
        deadline = started + max(0, plan.spec["max_wall_seconds"] - prior_elapsed)

        receipt["execution_started"] = True
        receipt["initial"] = evaluate(model, plan, args.device, deadline)
        _json(output / "initial-metrics.json", receipt["initial"])
        if wandb_run is not None:
            _wandb_log_metrics(wandb_run, "initial", receipt["initial"], 0)
        if plan.test is not None:
            receipt["initial_test"] = evaluate_test(model, plan, args.device, deadline)
            _json(output / "initial-test-metrics.json", receipt["initial_test"])
            if wandb_run is not None:
                _wandb_log_metrics(wandb_run, "initial_test", receipt["initial_test"], 0)
        accumulation = plan.spec.get("gradient_accumulation", 1)
        episodes = (
            _PreparedBank(plan, model, args.device)
            if plan.training_mode == "fixed_bank"
            else _ResampledEpisodes(plan, model, args.device)
        )
        while step < plan.spec["max_updates"]:
            _check_wall(deadline)
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset in range(accumulation):
                _check_wall(deadline)
                score = score_prepared_episode(model, episodes.get(cursor + offset))
                if not bool(torch.isfinite(score.loss)):
                    raise FloatingPointError("nonfinite training loss")
                (score.loss / accumulation).backward()
                losses.append(float(score.loss.detach()))
                del score
            parameters = [
                parameter for parameter in model.parameters() if parameter.grad is not None
            ]
            if not parameters:
                raise FloatingPointError("missing or nonfinite training gradients")
            # Same detection, one host sync: accumulate per-parameter flags on
            # device and reduce once, instead of one bool() sync per parameter.
            grad_flags = [torch.isfinite(p.grad).all().reshape(()) for p in parameters]
            if len({flag.device for flag in grad_flags}) > 1:
                grads_finite = all(bool(flag) for flag in grad_flags)
            else:
                grads_finite = bool(torch.stack(grad_flags).all())
            if not grads_finite:
                raise FloatingPointError("missing or nonfinite training gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters, cfg["grad_clip"], error_if_nonfinite=True
            )
            _check_wall(deadline)
            optimizer.step()
            if not _finite_state(model.state_dict()) or not _finite_state(optimizer.state_dict()):
                raise FloatingPointError("nonfinite parameters or optimizer state after update")
            boundary = checkpoint_state(
                model,
                optimizer,
                plan,
                step + 1,
                cursor + accumulation,
                prior_elapsed + time.monotonic() - started,
            )
            step, cursor = boundary["step"], boundary["sampler_cursor"]
            if args.device == "cuda:0":
                torch.cuda.synchronize()
            metric = {
                "step": step,
                "sampler_cursor": cursor,
                "loss": sum(losses) / accumulation,
                "gradient_norm": float(gradient_norm),
                "update_seconds": time.monotonic() - tick,
            }
            with (output / "updates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metric, allow_nan=False) + "\n")
                handle.flush()
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update.loss": metric["loss"],
                        "update.gradient_norm": metric["gradient_norm"],
                        "update.update_seconds": metric["update_seconds"],
                    },
                    step=step,
                )
            if step % plan.spec.get("checkpoint_every", 1) == 0:
                save_checkpoint(output / f"checkpoint-{step:08d}.pt", boundary)
            if (
                plan.evaluate_every
                and step % plan.evaluate_every == 0
                and step < plan.spec["max_updates"]
            ):
                point = {"step": step, "bank": evaluate(model, plan, args.device, deadline)}
                if wandb_run is not None:
                    _wandb_log_metrics(wandb_run, "eval.bank", point["bank"], step)
                if plan.test is not None:
                    point["test"] = evaluate_test(model, plan, args.device, deadline)
                    if wandb_run is not None:
                        _wandb_log_metrics(wandb_run, "eval.test", point["test"], step)
                with (output / "evaluations.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(point, allow_nan=False) + "\n")
                    handle.flush()
        receipt["final"] = evaluate(model, plan, args.device, deadline)
        if wandb_run is not None:
            _wandb_log_metrics(wandb_run, "final", receipt["final"], step)
        if plan.test is not None:
            receipt["final_test"] = evaluate_test(model, plan, args.device, deadline)
            _json(output / "final-test-metrics.json", receipt["final_test"])
            if wandb_run is not None:
                _wandb_log_metrics(wandb_run, "final_test", receipt["final_test"], step)
        receipt["outcome"] = "completed"
    except WallLimit as error:
        receipt.update(outcome="wall_limit", error_type=type(error).__name__, error=str(error))
    except KeyboardInterrupt:
        receipt.update(outcome="interrupted", error_type="KeyboardInterrupt")
    except Exception as error:
        # Error types and bounded messages suffice; never persist private filesystem paths.
        message = str(error)
        for path in (args.preregistration, args.dataset, args.output_root, args.resume):
            if path is not None:
                message = message.replace(str(Path(path).resolve()), "<artifact>")
        receipt.update(
            outcome="failed",
            error_type=type(error).__name__,
            error="artifact I/O failure" if isinstance(error, OSError) else message,
        )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish(exit_code=0 if receipt["outcome"] == "completed" else 1)
            except Exception:
                pass  # mirror teardown must never mask the authoritative receipt
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        receipt["completed_updates"] = boundary["step"] if boundary is not None else step
        receipt["sampler_cursor"] = boundary["sampler_cursor"] if boundary is not None else cursor
        receipt["attempt_elapsed_seconds"] = time.monotonic() - started
        receipt["elapsed_seconds"] = prior_elapsed + receipt["attempt_elapsed_seconds"]
        if boundary is not None:
            try:
                # A failed/partial optimizer step is discarded; resume starts at
                # the last completed finite update, using its matching RNG state.
                boundary["elapsed_seconds"] = receipt["elapsed_seconds"]
                save_checkpoint(output / "checkpoint.pt", boundary)
                receipt["checkpoint_step"] = boundary["step"]
                receipt["checkpoint_sha256"] = _hash(output / "checkpoint.pt")
            except Exception as error:
                receipt["checkpoint_error_type"] = type(error).__name__
                receipt["outcome"] = "failed"
        if args.device == "cuda:0" and receipt["execution_started"]:
            try:
                receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            except Exception as error:
                receipt["cuda_metrics_error_type"] = type(error).__name__
        _json(output / "terminal.json", receipt)
    return receipt
