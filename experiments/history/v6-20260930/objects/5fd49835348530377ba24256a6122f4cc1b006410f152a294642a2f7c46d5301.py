"""Program-level checkpoint envelope over the existing full-state Trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field, field_validator, model_validator

from tabu_lab.contracts import canonical_hash, canonical_json, require_sha256
from tabu_lab.training import Trainer

from .models import (
    EvidenceStatus,
    ProgramInitialization,
    ProgramLane,
    ResolvedProgramSnapshot,
    StrictModel,
)
from .policy import SamplingPolicyEngine, SamplingPolicyState


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ProgramCheckpointMetadata(StrictModel):
    schema_version: Literal["tabu.program-checkpoint.v1"] = "tabu.program-checkpoint.v1"
    resolved_snapshot: ResolvedProgramSnapshot
    lane: ProgramLane
    evidence_status: EvidenceStatus
    initialization: ProgramInitialization
    run_identity_hash: str
    trainer_checkpoint_sha256: str
    trainer_checkpoint_schema: Literal["tabu.training-checkpoint.v3"] = (
        "tabu.training-checkpoint.v3"
    )
    update_cursor: int = Field(ge=0)
    world_cursor: int = Field(ge=0)
    target_steps: int = Field(gt=0)
    policy_state: SamplingPolicyState
    scheduler_state: dict[str, Any] | None
    metadata_hash: str

    @field_validator("run_identity_hash", "trainer_checkpoint_sha256", "metadata_hash")
    @classmethod
    def _valid_hashes(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _consistent_identity(self) -> ProgramCheckpointMetadata:
        if self.world_cursor != self.policy_state.step:
            raise ValueError("world_cursor must equal serialized sampling-policy step")
        if self.update_cursor != self.world_cursor:
            raise ValueError("first-slice runner requires one world per optimizer update")
        if self.update_cursor > self.target_steps:
            raise ValueError("checkpoint cursor cannot exceed target_steps")
        if (
            self.lane is ProgramLane.GROW
            and self.evidence_status is not EvidenceStatus.LOCAL_UNISSUED
        ):
            raise ValueError("grow checkpoint cannot carry evidence status")
        payload = self.model_dump(mode="python", exclude={"metadata_hash"})
        if canonical_hash(payload) != self.metadata_hash:
            raise ValueError("metadata_hash does not match program checkpoint metadata")
        return self


def program_sidecar_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.suffix != ".safetensors":
        raise ValueError("program training checkpoint must use .safetensors")
    return path.with_suffix(".program.json")


def save_program_checkpoint(
    trainer: Trainer,
    checkpoint: str | Path,
    *,
    resolved_snapshot: ResolvedProgramSnapshot,
    lane: ProgramLane,
    evidence_status: EvidenceStatus,
    policy: SamplingPolicyEngine,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    initialization: ProgramInitialization,
    target_steps: int,
) -> ProgramCheckpointMetadata:
    destination = Path(checkpoint)
    sidecar = program_sidecar_path(destination)
    if destination.exists() or sidecar.exists():
        raise ValueError("refusing to overwrite an existing program checkpoint")
    if trainer.run_identity is None:
        raise ValueError("program checkpoint requires a bound Trainer RunIdentity")
    if trainer.step != policy.state.step:
        raise ValueError("trainer and sampling policy cursors differ")
    trainer.save_checkpoint(destination)
    scheduler_state = None if scheduler is None else scheduler.state_dict()
    payload = {
        "schema_version": "tabu.program-checkpoint.v1",
        "resolved_snapshot": resolved_snapshot,
        "lane": lane,
        "evidence_status": evidence_status,
        "initialization": initialization,
        "run_identity_hash": trainer.run_identity.identity_hash,
        "trainer_checkpoint_sha256": file_sha256(destination),
        "trainer_checkpoint_schema": trainer.checkpoint_version,
        "update_cursor": trainer.step,
        "world_cursor": policy.state.step,
        "target_steps": target_steps,
        "policy_state": policy.state,
        "scheduler_state": scheduler_state,
    }
    metadata = ProgramCheckpointMetadata(**payload, metadata_hash=canonical_hash(payload))
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary.write_text(
        canonical_json(metadata.model_dump(mode="python")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return metadata


def read_program_checkpoint(checkpoint: str | Path) -> ProgramCheckpointMetadata:
    source = Path(checkpoint)
    sidecar = program_sidecar_path(source)
    if not source.is_file() or not sidecar.is_file():
        raise ValueError("exact resume requires both trainer checkpoint and program sidecar")
    try:
        metadata = ProgramCheckpointMetadata.model_validate(
            json.loads(sidecar.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid program checkpoint sidecar: {exc}") from exc
    if file_sha256(source) != metadata.trainer_checkpoint_sha256:
        raise ValueError("trainer checkpoint bytes do not match program sidecar")
    return metadata


def load_program_checkpoint(
    trainer: Trainer,
    checkpoint: str | Path,
    *,
    resolved_snapshot: ResolvedProgramSnapshot,
    lane: ProgramLane,
    evidence_status: EvidenceStatus,
    policy: SamplingPolicyEngine,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    initialization: ProgramInitialization,
) -> ProgramCheckpointMetadata:
    metadata = read_program_checkpoint(checkpoint)
    if metadata.resolved_snapshot.snapshot_hash != resolved_snapshot.snapshot_hash:
        raise ValueError("program checkpoint snapshot does not match exact-resume target")
    if metadata.lane is not lane or metadata.evidence_status is not evidence_status:
        raise ValueError("program checkpoint lane identity does not match target")
    if metadata.initialization != initialization:
        raise ValueError("program checkpoint initialization identity does not match target")
    if trainer.run_identity is None or (
        metadata.run_identity_hash != trainer.run_identity.identity_hash
    ):
        raise ValueError("program checkpoint RunIdentity does not match target")
    if (scheduler is None) != (metadata.scheduler_state is None):
        raise ValueError("program checkpoint scheduler presence differs from target")
    trainer.load_checkpoint(checkpoint)
    policy.restore(metadata.policy_state)
    if scheduler is not None:
        assert metadata.scheduler_state is not None
        scheduler.load_state_dict(metadata.scheduler_state)
    if trainer.step != policy.state.step:
        raise ValueError("resumed trainer and sampling policy cursors differ")
    return metadata


def read_checkpoint_model_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Read model tensors without optimizer/RNG state for an explicit warm start."""

    from safetensors import safe_open

    source = Path(checkpoint)
    if source.suffix != ".safetensors" or not source.is_file():
        raise ValueError("warm-start checkpoint must be an existing .safetensors file")
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        keys = tuple(handle.keys())
        if any(key.startswith("model.") for key in keys):
            model_keys = tuple(key for key in keys if key.startswith("model."))
            return {key.removeprefix("model."): handle.get_tensor(key) for key in model_keys}
        return {key: handle.get_tensor(key) for key in keys}


__all__ = [
    "ProgramCheckpointMetadata",
    "file_sha256",
    "load_program_checkpoint",
    "program_sidecar_path",
    "read_checkpoint_model_state",
    "read_program_checkpoint",
    "save_program_checkpoint",
]
