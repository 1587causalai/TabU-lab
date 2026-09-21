"""Atomic local artifacts and exact checkpoint identity; no external telemetry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import uuid
from pathlib import Path

import torch

from .factory import artifact_schema
from .protocol import SCHEMA, V54_SCHEMA

CHECKPOINT_SCHEMA = "tabu.curriculum.v53.checkpoint.v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finite_state(value):
    """Return whether nested tensor state is finite with minimal host sync.

    Tensor-by-tensor ``bool(torch.isfinite(...).all())`` checks force a device
    synchronization for every parameter and optimizer slot. A curriculum
    update can contain hundreds of tensors, which is particularly expensive on
    MPS and also stalls CUDA launch overlap. Accumulate one scalar flag per
    device and synchronize only once per device while preserving the same
    fail-closed semantics, including mixed-device state dictionaries.
    """
    tensor_flags = {}
    python_finite = True

    def walk(item):
        nonlocal python_finite
        if isinstance(item, torch.Tensor):
            tensor_flags.setdefault(item.device, []).append(
                torch.isfinite(item).all().reshape(())
            )
        elif isinstance(item, dict):
            for sub in item.values():
                walk(sub)
        elif isinstance(item, list | tuple):
            for sub in item:
                walk(sub)
        elif isinstance(item, float) and not math.isfinite(item):
            python_finite = False

    walk(value)
    if not python_finite:
        return False
    for flags in tensor_flags.values():
        if not bool(torch.stack(flags).all()):
            return False
    return True


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, *, plan, model, optimizer, state, runtime, lineage, purpose="training"):
    if not finite_state(model.state_dict()) or not finite_state(optimizer.state_dict()):
        raise FloatingPointError("refusing to checkpoint nonfinite model or optimizer state")
    schema = artifact_schema(plan.spec["schema"], "checkpoint")
    payload = {
        "schema": schema,
        "purpose": purpose,
        "identity": plan.identity,
        "model_config": model.config.as_dict(),
        "runtime": runtime,
        "model": {key: tensor.detach().cpu() for key, tensor in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "optimizer_kind": state["optimizer_kind"],
        "rng": rng_state(),
        "state": state,
        "lineage": lineage,
    }
    path = Path(path)
    generations = path.parent / "checkpoints"
    generations.mkdir(exist_ok=True)
    temporary = generations / f".{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = sha256(temporary)
    generation = generations / f"{digest}.pt"
    # A generation is immutable. Its complete bytes exist before either index.
    if generation.exists():
        if sha256(generation) != digest:
            raise ValueError("content-addressed checkpoint was corrupted")
        temporary.unlink()
    else:
        os.replace(temporary, generation)
    descriptor = os.open(generations, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    atomic_json(
        path.with_suffix(".json"),
        {
            "schema": schema,
            "sha256": digest,
            "update": state["update"],
            "stage_index": state["stage_index"],
            "cursor": state["cursor"],
            "identity": plan.identity,
        },
    )
    # The only commit point is one atomic pointer replacement. A sidecar failure
    # leaves the previous pointer usable; orphan generations are safe to retain.
    pointer = path.with_name(f".{path.name}.{uuid.uuid4().hex}.link")
    pointer.symlink_to(generation.relative_to(path.parent))
    os.replace(pointer, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return digest


def load_checkpoint(path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    digest = sha256(resolved)
    # Sidecars are projections. The committed generation verifies itself through
    # its content-addressed name, even if a newer sidecar write preceded a crash.
    expected = (
        resolved.stem
        if re.fullmatch(r"[0-9a-f]{64}", resolved.stem)
        else (json.loads(path.with_suffix(".json").read_text()).get("sha256"))
    )
    if expected != digest:
        raise ValueError("checkpoint checksum mismatch")
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    identity_schema = payload.get("identity", {}).get("schema")
    if identity_schema not in (SCHEMA, V54_SCHEMA) or payload.get("schema") != artifact_schema(
        identity_schema, "checkpoint"
    ):
        raise ValueError("checkpoint schema or identity mismatch")
    if not finite_state(payload["model"]) or not finite_state(payload["optimizer"]):
        raise FloatingPointError("nonfinite checkpoint state")
    return payload, digest
