"""Bounded stages, frozen probes, exact resume, and durable terminal receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import signal
import threading
import time
from pathlib import Path

import torch

from tabu_lab.models.restoration_v53 import (
    V53LossConfig,
    prepare_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.restoration_optimizers import adamw, switch_to_muon

from .artifacts import (
    append_event,
    atomic_json,
    finite_state,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    sha256,
)
from .data import build_episode
from .evaluation import BudgetExhausted, evaluate_probe
from .factory import artifact_schema, make_model
from .protocol import schedule_entry
from .reporting import write_report


def configure_runtime(device):
    if device not in ("cpu", "cuda:0", "mps"):
        raise ValueError("qualified execution interface requires cpu, cuda:0 or mps")
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; CPU fallback is forbidden")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=0 must be set before importing torch")
        if os.environ.get("PYTORCH_MPS_FAST_MATH", "0") != "0":
            raise RuntimeError("PYTORCH_MPS_FAST_MATH must be disabled")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.use_deterministic_algorithms(True, warn_only=device == "mps")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    result = {
        "torch": str(torch.__version__),
        "device": device,
        "dtype": str(execution_dtype(device)).removeprefix("torch."),
        "deterministic_algorithms": True,
        "threads": 1,
        "interop_threads": torch.get_num_interop_threads(),
        "platform": [platform.system(), platform.release(), platform.machine()],
        "python": platform.python_version(),
        "torch_build_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
    }
    if device == "cuda:0":
        result["cuda_device"] = torch.cuda.get_device_name(0)
        result["cuda_capability"] = list(torch.cuda.get_device_capability(0))
        result["cuda_runtime"] = torch.version.cuda
        result["cudnn"] = torch.backends.cudnn.version()
        result["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        result["mps_available"] = True
        result["mps_cpu_fallback"] = False
    return result


def _seed_model(plan):
    seed = plan.spec["seeds"]["model"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_source(identity):
    source = identity.get("source", {})
    files = source.get("files", {})
    return {key: value for key, value in files.items() if key.startswith("models/")}


def _loss_terms(score, prepared, config):
    losses = score.per_target.detach()
    names = ("retained_contribution", "query_contribution")
    result = {name: losses.new_zeros(()) for name in names}
    for mask, weight in ((prepared.numeric, 1.0), (~prepared.numeric, config.discrete_weight)):
        # Keep population counts on-device. The old int(mask.sum()) and
        # int(selected.sum()) branches synchronized MPS/CUDA once per type and
        # state before the actual metric transfer below.
        total = mask.sum().clamp_min(1).to(losses.dtype)
        for state, name in ((0, "retained"), (1, "query")):
            selected = mask & (prepared.states == state)
            count = selected.sum().clamp_min(1).to(losses.dtype)
            term = losses[selected].sum() / (count if config.state_weights else total)
            multiplier = config.state_weights[state] if config.state_weights else 1.0
            result[f"{name}_contribution"] += term * multiplier * weight
    # Transfer the small metric vector once instead of synchronizing once per
    # state/type term. The loss itself is already detached above.
    values = torch.stack([result[name] for name in names]).cpu().tolist()
    return dict(zip(names, values))


def train_step(model, optimizer, plan, table, recipe, index, device, loss_config, *, namespace=""):
    started = time.monotonic()
    seeds = dict(plan.spec["seeds"])
    for stream in ("masks", "codes", "windows"):
        seeds[stream] = int.from_bytes(
            hashlib.sha256(f"{seeds[stream]}/{namespace}".encode()).digest()[:8], "little"
        )
    inputs, request, truth, info = build_episode(
        table, recipe, index, seeds, device, epsilon=plan.config.epsilon,
        codec_version=plan.config.codec_version,
    )
    model.train()
    prepared = prepare_episode(model, inputs, request, truth)
    optimizer.zero_grad(set_to_none=True)
    score = score_prepared_episode(model, prepared, loss_config=loss_config)
    score.loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    # Reduce all gradient finiteness flags on-device and synchronize once.
    # Checking every parameter with a separate bool() serializes MPS/CUDA
    # launches and was a measurable part of the per-step cost.
    if not parameters or not finite_state([parameter.grad for parameter in parameters]):
        raise FloatingPointError("missing or nonfinite gradients; update rejected")
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, plan.optimizer.grad_clip, error_if_nonfinite=True
    )
    post_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
    optimizer.step()
    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
        raise FloatingPointError("nonfinite parameters/optimizer; last valid checkpoint retained")
    return {
        "table": table.name,
        "cohort": table.cohort,
        "table_episode_index": index,
        "loss": float(score.loss.detach()),
        **_loss_terms(score, prepared, loss_config),
        "gradient_norm": float(norm),
        "clipped_gradient_norm": float(post_norm),
        "seconds": time.monotonic() - started,
        "episode": info,
    }


def _exposure(state, row):
    item = state["exposure"].setdefault(
        row["table"],
        {
            "updates": 0,
            "query_cells": 0,
            "row_visits": {},
            "query_row_visits": {},
        },
    )
    item["updates"] += 1
    item["query_cells"] += row["episode"]["query_count"]
    for address in row["episode"]["row_ids"]:
        key = str(address)
        item["row_visits"][key] = item["row_visits"].get(key, 0) + 1
    for address, _column in row["episode"]["query_addresses"]:
        key = str(address)
        item["query_row_visits"][key] = item["query_row_visits"].get(key, 0) + 1


def _new_state(stages):
    return {
        "stage_index": 0,
        "cursor": 0,
        "update": 0,
        "phase": "entry",
        "stage_seconds": [0.0] * len(stages),
        "total_seconds": 0.0,
        "optimizer_kind": "adamw",
        "evaluated_cursor": -1,
        "last_evaluation": {},
        "exposure": {},
        "stage_verdicts": [],
    }


def _validate_resume(payload, plan, runtime):
    if payload.get("purpose") != "training":
        raise ValueError("qualification artifacts are not training checkpoints")
    if payload["identity"] != plan.identity or payload["model_config"] != plan.config.as_dict():
        raise ValueError("resume identity drift: model, data, source or protocol changed")
    if payload["runtime"] != runtime:
        raise ValueError("resume runtime drift; exact resume requires the same runtime profile")
    state = payload["state"]
    index = state["stage_index"]
    if (
        type(index) is not int
        or not 0 <= index <= len(plan.spec["stages"])
        or state["phase"] not in ("entry", "train", "completed")
        or len(state["stage_seconds"]) != len(plan.spec["stages"])
        or state["optimizer_kind"] not in ("adamw", "muon")
        or not finite_state(state)
    ):
        raise ValueError("invalid checkpoint scheduler state")
    for key in ("cursor", "update", "evaluated_cursor"):
        if type(state[key]) is not int or state[key] < (-1 if key == "evaluated_cursor" else 0):
            raise ValueError(f"invalid checkpoint {key}")
    if (
        any(value < 0 for value in state["stage_seconds"])
        or state["total_seconds"] + 1e-9 < sum(state["stage_seconds"])
        or payload["optimizer_kind"] != state["optimizer_kind"]
        or state["evaluated_cursor"] > state["cursor"]
    ):
        raise ValueError("invalid checkpoint counters or optimizer")
    expected_update = sum(stage["max_updates"] for stage in plan.spec["stages"][:index])
    if state["update"] != expected_update + state["cursor"]:
        raise ValueError("checkpoint update/cursor mismatch")
    if (state["phase"] == "completed") != (index == len(plan.spec["stages"])) or (
        state["phase"] in ("entry", "completed") and state["cursor"] != 0
    ):
        raise ValueError("checkpoint phase/cursor mismatch")
    if (
        index < len(plan.spec["stages"])
        and not 0 <= state["cursor"] <= plan.spec["stages"][index]["max_updates"]
    ):
        raise ValueError("checkpoint cursor outside the stage")


def run(
    plan,
    output,
    *,
    device="cpu",
    resume=None,
    initialize_from=None,
    max_updates_this_invocation=None,
):
    """Use a NEW attempt directory on every invocation, including resume.

    A failed/in-flight optimizer step is never serialized as a valid checkpoint.
    Wall exhaustion ends this attempt and never implicitly allocates another stage.
    Update limits may stop at a boundary without changing the experiment identity.
    """
    if resume and initialize_from:
        raise ValueError("resume and initialize_from are mutually exclusive")
    if max_updates_this_invocation is not None and (
        type(max_updates_this_invocation) is not int or max_updates_this_invocation < 0
    ):
        raise ValueError("invocation update limit must be a nonnegative integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    stages = plan.spec["stages"]
    state, lineage = _new_state(stages), []
    model = optimizer = None
    runtime = {"device": device}
    last_clock = time.monotonic()
    checkpoint_digest = None
    durable_update = None
    stop_requested = []
    previous_signals = {}
    result = {
        "schema": artifact_schema(plan.spec["schema"], "terminal"),
        "status": "local_unissued",
        "identity": plan.identity,
        "outcome": "started",
    }
    atomic_json(output / "started.json", result)
    atomic_json(
        output / "resolved.json",
        {"spec": plan.spec, "summary": plan.summary, "identity": plan.identity},
    )

    def charge(attribution=None):
        nonlocal last_clock
        current = time.monotonic()
        elapsed = current - last_clock
        state["total_seconds"] += elapsed
        index = state["stage_index"] if attribution is None else attribution
        if index < len(stages):
            state["stage_seconds"][index] += elapsed
        last_clock = current

    def save(name="checkpoint-progress.pt", *, attribution=None):
        nonlocal checkpoint_digest, durable_update
        charge(attribution)
        digest = save_checkpoint(
            output / name,
            plan=plan,
            model=model,
            optimizer=optimizer,
            state=state,
            runtime=runtime,
            lineage=lineage,
        )
        if name == "checkpoint-progress.pt":
            checkpoint_digest, durable_update = digest, state["update"]
        charge(attribution)

    def budget():
        charge()
        index = state["stage_index"]
        for earlier in range(min(index + 1, len(stages))):
            if state["stage_seconds"][earlier] >= stages[earlier]["max_seconds"]:
                raise BudgetExhausted(f"stage {stages[earlier]['name']} wall budget exhausted")
        if index == len(stages):
            return None
        return time.monotonic() + stages[index]["max_seconds"] - state["stage_seconds"][index]

    def evaluate(stage, label):
        deadline = budget()
        values = {}
        for name in stage["probes"]:
            probe = next(item for item in plan.spec["probes"] if item["name"] == name)
            values[name] = evaluate_probe(model, plan, probe, device, deadline=deadline)
        charge()
        path = output / f"evaluation-{state['stage_index']:03d}-{state['cursor']:09d}-{label}.json"
        atomic_json(
            path,
            {
                "stage": stage["name"],
                "update": state["update"],
                "cursor": state["cursor"],
                "probes": values,
            },
        )
        state["last_evaluation"], state["evaluated_cursor"] = values, state["cursor"]
        append_event(
            output / "events.jsonl",
            {"event": "evaluation", "file": path.name, "update": state["update"]},
        )
        return values

    try:
        runtime = configure_runtime(device)
        _seed_model(plan)
        model = make_model(plan).to(device=device, dtype=execution_dtype(device))
        optimizer = adamw(model, plan.optimizer)
        parent = resume or initialize_from
        if parent:
            payload, digest = load_checkpoint(parent)
            lineage = [
                *payload.get("lineage", []),
                {
                    "mode": "resume" if resume else "weights_only_initialization",
                    "checkpoint_sha256": digest,
                    "parent_update": payload["state"]["update"],
                    "parent_identity": payload["identity"],
                },
            ]
            if resume:
                _validate_resume(payload, plan, runtime)
                state = copy.deepcopy(payload["state"])
                # Charge failed/discarded work recorded by the parent terminal too.
                generation = Path(parent).resolve()
                attempt = (
                    generation.parent.parent
                    if generation.parent.name == "checkpoints"
                    else Path(parent).parent
                )
                terminal = attempt / "terminal.json"
                if terminal.exists():
                    receipt = json.loads(terminal.read_text())
                    if (
                        receipt.get("identity") == payload["identity"]
                        and (receipt.get("durable_update") or 0) >= state["update"]
                    ):
                        state["total_seconds"] = max(
                            state["total_seconds"], receipt["total_seconds"]
                        )
                        state["stage_seconds"] = [
                            max(a, b)
                            for a, b in zip(
                                state["stage_seconds"], receipt["stage_seconds"], strict=True
                            )
                        ]
                if state["optimizer_kind"] == "muon":
                    optimizer, _ = switch_to_muon(optimizer, model, plan.optimizer)
                optimizer.load_state_dict(payload["optimizer"])
                model.load_state_dict(payload["model"])
                restore_rng(payload["rng"])
            else:
                if (
                    payload.get("purpose") != "training"
                    or payload["identity"].get("schema") != plan.spec["schema"]
                    or payload["model_config"] != plan.config.as_dict()
                    or not _model_source(plan.identity)
                    or _model_source(payload["identity"]) != _model_source(plan.identity)
                ):
                    raise ValueError(
                        "weights-only initialization requires identical model contract"
                    )
                model.load_state_dict(payload["model"])
                # Explicit initialization resets optimizer, RNG and scheduling.
                _seed_model(plan)
        first_update = state["update"]
        save()
        # Also check a resumed completed cursor: final checkpoint IO may have
        # exhausted the last stage, as recorded by the parent terminal receipt.
        budget()
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_signals[signum] = signal.signal(
                    signum, lambda received, _frame: stop_requested.append(received)
                )
        while state["stage_index"] < len(stages):
            stage = stages[state["stage_index"]]
            if stop_requested or (
                max_updates_this_invocation is not None
                and state["update"] - first_update >= max_updates_this_invocation
            ):
                save()
                result["outcome"] = "interrupted" if stop_requested else "stopped"
                break
            budget()
            if state["phase"] == "entry":
                if stage["optimizer"] == "muon" and state["optimizer_kind"] == "adamw":
                    optimizer, partition = switch_to_muon(optimizer, model, plan.optimizer)
                    state["optimizer_kind"] = "muon"
                    atomic_json(
                        output / f"optimizer-stage-{state['stage_index']:03d}.json", partition
                    )
                evaluate(stage, "initial")
                state["phase"] = "train"
                save()
            # A checkpoint after an update precedes its due evaluation. Resuming
            # here must execute the pending bank before admitting another update.
            if (
                state["cursor"]
                and state["evaluated_cursor"] != state["cursor"]
                and (
                    state["cursor"] % stage["evaluate_every"] == 0
                    or state["cursor"] == stage["max_updates"]
                )
            ):
                evaluate(stage, "periodic" if state["cursor"] < stage["max_updates"] else "final")
                budget()
                save()
            if state["cursor"] == stage["max_updates"]:
                budget()
                verdict = "completed_no_performance_gate"
                gate = stage.get("gate")
                if gate:
                    value = state["last_evaluation"][gate["probe"]]["macro"].get(gate["metric"])
                    passed = value is not None and (
                        value <= gate["threshold"]
                        if gate["mode"] == "min"
                        else value >= gate["threshold"]
                    )
                    verdict = "passed" if passed else "failed"
                state["stage_verdicts"].append(
                    {"stage": stage["name"], "verdict": verdict, "update": state["update"]}
                )
                append_event(
                    output / "events.jsonl",
                    {"event": "stage_completed", **state["stage_verdicts"][-1]},
                )
                if verdict == "failed":
                    save()
                    result["outcome"] = "gate_failed"
                    break
                # Attribute final evaluation / gate IO to the completed stage.
                charge()
                old_index = state["stage_index"]
                state.update(
                    stage_index=old_index + 1,
                    cursor=0,
                    phase="entry",
                    evaluated_cursor=-1,
                    last_evaluation={},
                )
                if state["stage_index"] == len(stages):
                    state["phase"] = "completed"
                save(f"stage-{old_index:03d}-final.pt", attribution=old_index)
                save(attribution=old_index)
                if state["stage_seconds"][old_index] >= stage["max_seconds"]:
                    raise BudgetExhausted(f"stage {stage['name']} final saving exceeded budget")
                continue
            budget()
            table, index = schedule_entry(plan, state["stage_index"], state["cursor"])
            row = train_step(
                model,
                optimizer,
                plan,
                table,
                stage["recipe"][table.kind],
                index,
                device,
                V53LossConfig(**stage.get("loss", {})),
                namespace=stage["name"],
            )
            state["update"] += 1
            state["cursor"] += 1
            _exposure(state, row)
            charge()
            append_event(
                output / "updates.jsonl",
                {
                    **row,
                    "stage": stage["name"],
                    "update": state["update"],
                    "cursor": state["cursor"],
                    "total_seconds": state["total_seconds"],
                },
            )
            if (
                state["cursor"] % stage["checkpoint_every"] == 0
                or state["cursor"] % stage["evaluate_every"] == 0
                or state["cursor"] == stage["max_updates"]
            ):
                save()
        else:
            result["outcome"] = "completed"
    except BudgetExhausted as error:
        # No update is in flight at budget admission checks/evaluation boundaries.
        result.update(outcome="budget_exhausted", error=str(error))
        if model is not None and optimizer is not None and checkpoint_digest is not None:
            save()
    except (KeyboardInterrupt, Exception) as error:
        # Never serialize potentially half-updated tensors after arbitrary failure.
        result.update(
            outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    finally:
        for signum, previous in previous_signals.items():
            signal.signal(signum, previous)
        charge()
        result.update(
            runtime=runtime,
            update=state["update"],
            durable_update=durable_update,
            stage_index=state["stage_index"],
            cursor=state["cursor"],
            phase=state["phase"],
            total_seconds=state["total_seconds"],
            stage_seconds=state["stage_seconds"],
            exposure=state["exposure"],
            stage_verdicts=state["stage_verdicts"],
            lineage=lineage,
            checkpoint="checkpoint-progress.pt" if checkpoint_digest else None,
            checkpoint_sha256=checkpoint_digest,
        )
        if device == "cuda:0" and torch.cuda.is_available():
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        atomic_json(output / "terminal.json", result)
        _report(output, plan, result)
    return result


def _report(output, plan, result):
    # Reporting is a rebuildable projection; preserve the authoritative receipt
    # even if the disk becomes unavailable during this final presentation step.
    try:
        write_report(output, plan, result)
    except (OSError, ValueError, TypeError) as error:
        result["report_error"] = f"{type(error).__name__}: {error}"
        atomic_json(output / "terminal.json", result)


def evaluate_checkpoint(plan, checkpoint, output, *, device="cpu", probes=None):
    """Read-only frozen checkpoint evaluation in a separate, non-overwriting attempt."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": artifact_schema(plan.spec["schema"], "evaluation"),
        "outcome": "started",
        "status": "local_unissued",
        "identity": plan.identity,
    }
    atomic_json(output / "started.json", result)
    try:
        runtime = configure_runtime(device)
        payload, digest = load_checkpoint(checkpoint)
        if payload.get("purpose") != "training" or payload["identity"] != plan.identity:
            raise ValueError("evaluation identity drift")
        selected = (
            probes if probes is not None else [probe["name"] for probe in plan.spec["probes"]]
        )
        if len(selected) != len(set(selected)) or not set(selected) <= {
            probe["name"] for probe in plan.spec["probes"]
        }:
            raise ValueError("unknown or duplicate probe names")
        model = make_model(plan).to(device=device, dtype=execution_dtype(device))
        model.load_state_dict(payload["model"])
        reports = {
            probe["name"]: evaluate_probe(model, plan, probe, device)
            for probe in plan.spec["probes"]
            if probe["name"] in selected
        }
        if sha256(checkpoint) != digest:
            raise ValueError("checkpoint changed during frozen evaluation")
        result.update(
            outcome="completed",
            runtime=runtime,
            checkpoint_sha256=digest,
            checkpoint_update=payload["state"]["update"],
            checkpoint_source=str(Path(checkpoint).resolve()),
            probes=reports,
        )
    except (KeyboardInterrupt, Exception) as error:
        result.update(
            outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
    atomic_json(output / "terminal.json", result)
    _report(output, plan, result)
    return result
