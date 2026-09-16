"""Bounded numeric restoration fit diagnostic; execution is an explicit opt-in.

The snapshot's reserved rows never enter an episode. Initial/final scores use the
same fixed mask bank on sampled training rows and are fit diagnostics only.
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
    generator = torch.Generator().manual_seed(seeds["rows"])
    order = torch.randperm(len(data["splits"]["train"]), generator=generator)[:count].tolist()
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
    generator.manual_seed(seeds["masks"])
    queries = []
    for _ in range(bank_size):
        query = torch.zeros(count, len(columns), dtype=torch.bool)
        for column in range(len(columns)):
            query[torch.randperm(count, generator=generator)[:hidden], column] = True
        queries.append(query)
    generator.manual_seed(seeds["codes"])
    code_seeds = tuple(torch.randint(2**63 - 1, (bank_size,), generator=generator).tolist())
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
        "query_per_column": hidden,
        "visible_per_column": count - hidden,
        "mask_bank": bank,
        "reserved_rows_used": 0,
        "evaluation_scope": "fixed fit masks on training rows; no reserved evaluation",
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
def evaluate(model, plan, device, deadline):
    model.eval()
    totals = {
        name: {"count": 0, "encoding_sse": 0.0, "numeric_sse": 0.0}
        for name in ("retained", "query")
    }
    losses = []
    for index in range(len(plan.queries)):
        _check_wall(deadline)
        episode = plan.episode(index, device)
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
        "scope": plan.summary["evaluation_scope"],
    }
    try:
        json.dumps(report, allow_nan=False)
    except ValueError as error:
        raise FloatingPointError("nonfinite evaluation report") from error
    return report


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
        accumulation = plan.spec.get("gradient_accumulation", 1)
        prepared_bank = _PreparedBank(plan, model, args.device)
        while step < plan.spec["max_updates"]:
            _check_wall(deadline)
            tick = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for offset in range(accumulation):
                _check_wall(deadline)
                score = score_prepared_episode(model, prepared_bank.get(cursor + offset))
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
            if step % plan.spec.get("checkpoint_every", 1) == 0:
                save_checkpoint(output / f"checkpoint-{step:08d}.pt", boundary)
        receipt["final"] = evaluate(model, plan, args.device, deadline)
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
