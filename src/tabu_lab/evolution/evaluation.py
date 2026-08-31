"""Independent, non-overwriting evaluation receipts for program checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import (
    EvidenceEpisode,
    TruthSidecar,
    canonical_hash,
    canonical_json,
    require_sha256,
)

from .checkpoint import (
    file_sha256,
    program_sidecar_path,
    read_checkpoint_model_state,
    read_program_checkpoint,
)
from .models import (
    ComponentGraphNode,
    EvaluationProtocolNode,
    EvidenceStatus,
    GeneratorNode,
    ObjectiveBundleNode,
    ProgramArtifact,
    ProgramRunReceipt,
    ProgramRunStatus,
    StrictModel,
    TrainingRecipeNode,
    WorldMixtureNode,
)
from .repository import EvolutionRepository
from .runtime import _build_runtime_model, _import_symbol, _objective

_REQUEST_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


class ProgramCheckpointEvaluationRequest(StrictModel):
    """Checked-in selection and panel identity for one checkpoint evaluation."""

    schema_version: Literal["tabu.program-checkpoint-evaluation-request.v1"] = (
        "tabu.program-checkpoint-evaluation-request.v1"
    )
    request_id: str
    version: str
    evidence_status: Literal[EvidenceStatus.LOCAL_UNISSUED] = EvidenceStatus.LOCAL_UNISSUED
    claim_boundary: str
    program_ref: str
    snapshot_hash: str
    run_identity_hash: str
    training_run_receipt_hash: str
    source_training_commit: str
    source_repository_hash: str
    checkpoint_name: str
    checkpoint_step: int = Field(ge=0)
    checkpoint_sha256: str
    checkpoint_sidecar_sha256: str
    selection_receipt_hash: str
    selection_panel_addresses_hash: str
    selection_rule: str
    evaluation_protocol_ref: str
    evaluation_mode: Literal["synthetic_fit"] = "synthetic_fit"
    metric: Literal["loss"] = "loss"
    partition: Literal["validation"] = "validation"
    root_seed: int = Field(ge=0)
    worlds: int = Field(gt=0)
    address_prefix: str

    @field_validator("request_id")
    @classmethod
    def _valid_request_id(cls, value: str) -> str:
        if _REQUEST_ID.fullmatch(value) is None:
            raise ValueError("request_id must be a namespaced lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("evaluation request version must be semantic-versioned")
        return value

    @field_validator(
        "snapshot_hash",
        "run_identity_hash",
        "training_run_receipt_hash",
        "source_repository_hash",
        "checkpoint_sha256",
        "checkpoint_sidecar_sha256",
        "selection_receipt_hash",
        "selection_panel_addresses_hash",
    )
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @field_validator("source_training_commit")
    @classmethod
    def _valid_source_commit(cls, value: str) -> str:
        if _GIT_REVISION.fullmatch(value) is None:
            raise ValueError("source_training_commit must be a full Git revision")
        return value

    @model_validator(mode="after")
    def _request_is_complete(self) -> ProgramCheckpointEvaluationRequest:
        if "@" not in self.program_ref or "@" not in self.evaluation_protocol_ref:
            raise ValueError("program and evaluation protocol refs must be versioned")
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("checkpoint_name must be a basename")
        if not self.claim_boundary.strip() or not self.selection_rule.strip():
            raise ValueError("claim boundary and selection rule cannot be blank")
        if not self.address_prefix.strip():
            raise ValueError("address_prefix cannot be blank")
        return self

    @property
    def ref(self) -> str:
        return f"{self.request_id}@{self.version}"

    @property
    def request_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class ProgramEvaluationPanel(StrictModel):
    partition: Literal["validation"] = "validation"
    root_seed: int = Field(ge=0)
    worlds: int = Field(gt=0)
    address_prefix: str
    addresses_hash: str
    coverage: dict[str, tuple[JsonValue, ...]]

    @field_validator("addresses_hash")
    @classmethod
    def _valid_addresses_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="addresses_hash")


class ProgramEvaluationMetrics(StrictModel):
    mean_loss: float
    median_loss: float
    minimum_loss: float
    maximum_loss: float
    per_world_loss: tuple[float, ...]
    scored_targets: int = Field(ge=0)
    abstained_targets: int = Field(ge=0)

    @model_validator(mode="after")
    def _metrics_are_finite_and_consistent(self) -> ProgramEvaluationMetrics:
        values = self.per_world_loss
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("per-world losses must be non-empty and finite")
        summaries = (
            self.mean_loss,
            self.median_loss,
            self.minimum_loss,
            self.maximum_loss,
        )
        if not all(math.isfinite(value) for value in summaries):
            raise ValueError("evaluation summaries must be finite")
        expected = (
            sum(values) / len(values),
            statistics.median(values),
            min(values),
            max(values),
        )
        if any(
            not math.isclose(observed, target, rel_tol=0.0, abs_tol=1.0e-12)
            for observed, target in zip(summaries, expected, strict=True)
        ):
            raise ValueError("evaluation summaries do not match per-world losses")
        return self


class ProgramCheckpointEvaluationReceipt(StrictModel):
    """Self-hashing receipt for one checkpoint on one immutable evaluation panel."""

    schema_version: Literal["tabu.program-checkpoint-evaluation-receipt.v1"] = (
        "tabu.program-checkpoint-evaluation-receipt.v1"
    )
    evidence_status: Literal[EvidenceStatus.LOCAL_UNISSUED] = EvidenceStatus.LOCAL_UNISSUED
    claim_boundary: str
    request: ProgramCheckpointEvaluationRequest
    request_hash: str
    source_repository_hash: str
    evaluation_protocol_ref: str
    evaluation_protocol_hash: str
    objective_bundle_ref: str
    objective_bundle_hash: str
    world_mixture_ref: str
    world_mixture_hash: str
    generator_ref: str
    generator_hash: str
    snapshot_hash: str
    run_identity_hash: str
    checkpoint: ProgramArtifact
    checkpoint_sidecar: ProgramArtifact
    checkpoint_metadata_hash: str
    checkpoint_step: int = Field(ge=0)
    evaluation_tool: ProgramArtifact
    evaluation_source_revision: str
    evaluation_source_archive_sha256: str
    execution: dict[str, JsonValue]
    panel: ProgramEvaluationPanel
    metrics: ProgramEvaluationMetrics
    model_state_hash_before: str
    model_state_hash_after: str
    receipt_hash: str

    @field_validator(
        "request_hash",
        "source_repository_hash",
        "evaluation_protocol_hash",
        "objective_bundle_hash",
        "world_mixture_hash",
        "generator_hash",
        "snapshot_hash",
        "run_identity_hash",
        "checkpoint_metadata_hash",
        "evaluation_source_archive_sha256",
        "model_state_hash_before",
        "model_state_hash_after",
        "receipt_hash",
    )
    @classmethod
    def _valid_receipt_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @field_validator("evaluation_source_revision")
    @classmethod
    def _valid_evaluation_revision(cls, value: str) -> str:
        if _GIT_REVISION.fullmatch(value) is None:
            raise ValueError("evaluation_source_revision must be a full Git revision")
        return value

    @model_validator(mode="after")
    def _receipt_is_consistent(self) -> ProgramCheckpointEvaluationReceipt:
        if self.request_hash != self.request.request_hash:
            raise ValueError("request_hash does not match evaluation request")
        if self.claim_boundary != self.request.claim_boundary:
            raise ValueError("receipt claim boundary does not match request")
        if self.source_repository_hash != self.request.source_repository_hash:
            raise ValueError("receipt repository hash does not match request")
        if self.evaluation_protocol_ref != self.request.evaluation_protocol_ref:
            raise ValueError("receipt evaluation protocol does not match request")
        if self.snapshot_hash != self.request.snapshot_hash:
            raise ValueError("receipt snapshot does not match request")
        if self.run_identity_hash != self.request.run_identity_hash:
            raise ValueError("receipt run identity does not match request")
        if self.checkpoint.name != self.request.checkpoint_name:
            raise ValueError("receipt checkpoint name does not match request")
        if self.checkpoint.sha256 != self.request.checkpoint_sha256:
            raise ValueError("receipt checkpoint hash does not match request")
        if self.checkpoint_sidecar.sha256 != self.request.checkpoint_sidecar_sha256:
            raise ValueError("receipt checkpoint sidecar hash does not match request")
        if self.checkpoint_step != self.request.checkpoint_step:
            raise ValueError("receipt checkpoint step does not match request")
        if (
            self.panel.partition != self.request.partition
            or self.panel.root_seed != self.request.root_seed
            or self.panel.worlds != self.request.worlds
            or self.panel.address_prefix != self.request.address_prefix
        ):
            raise ValueError("receipt panel does not match request")
        if len(self.metrics.per_world_loss) != self.panel.worlds:
            raise ValueError("per-world losses do not cover the immutable panel")
        if self.model_state_hash_before != self.model_state_hash_after:
            raise ValueError("evaluation mutated model state")
        payload = self.model_dump(mode="python", exclude={"receipt_hash"})
        if canonical_hash(payload) != self.receipt_hash:
            raise ValueError("receipt_hash does not match evaluation receipt")
        return self


@dataclass(frozen=True)
class ProgramCheckpointEvaluationResult:
    receipt: ProgramCheckpointEvaluationReceipt
    receipt_path: Path


def load_program_evaluation_request(
    path: str | Path,
) -> ProgramCheckpointEvaluationRequest:
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid checkpoint evaluation request: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("checkpoint evaluation request root must be a mapping")
    return ProgramCheckpointEvaluationRequest.model_validate(payload)


def _load_training_run_receipt(path: str | Path) -> ProgramRunReceipt:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training run receipt: {exc}") from exc
    return ProgramRunReceipt.model_validate(payload)


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(canonical_json(tuple(value.shape)).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def evaluate_program_checkpoint(
    repository: EvolutionRepository,
    *,
    request: ProgramCheckpointEvaluationRequest,
    checkpoint: str | Path,
    training_run_receipt: str | Path,
    output: str | Path,
    device: str = "cpu",
    evaluation_source_revision: str,
    evaluation_source_archive_sha256: str,
) -> ProgramCheckpointEvaluationResult:
    """Evaluate one selected checkpoint without optimizer state or in-place output."""

    destination = Path(output)
    if destination.exists():
        raise ValueError(f"refusing to overwrite evaluation receipt: {destination}")
    if _GIT_REVISION.fullmatch(evaluation_source_revision) is None:
        raise ValueError("evaluation_source_revision must be a full Git revision")
    require_sha256(
        evaluation_source_archive_sha256,
        field_name="evaluation_source_archive_sha256",
    )
    if repository.repository_hash != request.source_repository_hash:
        raise ValueError("evaluation repository hash does not match frozen request")

    resolved = repository.resolve(request.program_ref)
    if resolved.snapshot_hash != request.snapshot_hash:
        raise ValueError("evaluation program snapshot does not match frozen request")
    graph = repository.node(resolved.slots["component_graph"].ref)
    mixture = repository.node(resolved.slots["world_mixture"].ref)
    bundle = repository.node(resolved.slots["objective_bundle"].ref)
    recipe = repository.node(resolved.slots["training_recipe"].ref)
    evaluation = repository.node(resolved.slots["evaluation_protocol"].ref)
    if not isinstance(graph, ComponentGraphNode):
        raise TypeError("program component graph is invalid")
    if not isinstance(mixture, WorldMixtureNode) or len(mixture.entries) != 1:
        raise ValueError("checkpoint evaluation requires one immutable generator entry")
    if not isinstance(bundle, ObjectiveBundleNode) or not isinstance(recipe, TrainingRecipeNode):
        raise TypeError("program objective or training recipe is invalid")
    if not isinstance(evaluation, EvaluationProtocolNode):
        raise TypeError("program evaluation protocol is invalid")
    if evaluation.ref != request.evaluation_protocol_ref:
        raise ValueError("resolved evaluation protocol does not match frozen request")
    if request.evaluation_mode not in evaluation.modes or request.metric not in evaluation.metrics:
        raise ValueError("frozen evaluation mode or metric is absent from protocol")
    if "validation" not in evaluation.split_authority:
        raise ValueError("evaluation protocol does not authorize validation partition")

    generator = repository.node(mixture.entries[0].generator.ref)
    if not isinstance(generator, GeneratorNode):
        raise TypeError("world mixture entry is not a generator")
    generator_runtime = _import_symbol(generator.runtime_ref)
    if not callable(generator_runtime):
        raise TypeError("generator runtime is not callable")
    options = {**generator.immutable_config, **recipe.episode_options}

    checkpoint_path = Path(checkpoint).resolve()
    if checkpoint_path.name != request.checkpoint_name:
        raise ValueError("checkpoint basename does not match frozen request")
    sidecar_path = program_sidecar_path(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    sidecar_hash = file_sha256(sidecar_path)
    if checkpoint_hash != request.checkpoint_sha256:
        raise ValueError("checkpoint hash does not match frozen request")
    if sidecar_hash != request.checkpoint_sidecar_sha256:
        raise ValueError("checkpoint sidecar hash does not match frozen request")
    metadata = read_program_checkpoint(checkpoint_path)
    if metadata.resolved_snapshot.snapshot_hash != request.snapshot_hash:
        raise ValueError("checkpoint snapshot does not match frozen request")
    if metadata.run_identity_hash != request.run_identity_hash:
        raise ValueError("checkpoint run identity does not match frozen request")
    if metadata.update_cursor != request.checkpoint_step:
        raise ValueError("checkpoint step does not match frozen request")

    training_receipt = _load_training_run_receipt(training_run_receipt)
    if training_receipt.receipt_hash != request.training_run_receipt_hash:
        raise ValueError("training run receipt hash does not match frozen request")
    if training_receipt.status is not ProgramRunStatus.COMPLETED:
        raise ValueError("selected checkpoint must descend from a completed training run")
    if (
        training_receipt.snapshot_hash != request.snapshot_hash
        or training_receipt.run_identity_hash != request.run_identity_hash
    ):
        raise ValueError("training run receipt lineage does not match selected checkpoint")
    if training_receipt.evidence_status is not EvidenceStatus.LOCAL_UNISSUED:
        raise ValueError("Grow evaluation cannot inherit a non-local evidence status")

    torch.manual_seed(1729)
    model = _build_runtime_model(graph, device)
    model.load_state_dict(read_checkpoint_model_state(checkpoint_path), strict=True)
    model.eval()
    state_before = _model_state_hash(model)
    objective = _objective(bundle)
    addresses = tuple(
        f"{request.address_prefix}-{index:08d}" for index in range(request.worlds)
    )
    coverage: dict[str, set[JsonValue]] = {
        "context_rows": set(),
        "noise_level": set(),
        "predictor_regime": set(),
        "width": set(),
        "world_family": set(),
    }
    losses: list[float] = []
    scored_targets = 0
    abstained_targets = 0
    target_device = torch.device(device)
    with torch.no_grad():
        for index, world_id in enumerate(addresses):
            generated = generator_runtime(
                root_seed=request.root_seed,
                world_id=world_id,
                partition=request.partition,
                **options,
            )
            evidence = getattr(generated, "evidence", None)
            truth = getattr(generated, "sidecar", None)
            if not isinstance(evidence, EvidenceEpisode) or not isinstance(truth, TruthSidecar):
                raise TypeError("generator did not return EvidenceEpisode plus TruthSidecar")
            prediction = model(evidence)
            loss = objective(prediction, truth.to(target_device))
            value = float(loss.total.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("non-finite independent evaluation loss")
            losses.append(value)
            scored_targets += int(
                loss.counts["numeric_scored_targets"]
                + loss.counts["categorical_scored_targets"]
            )
            abstained_targets += int(loss.counts["abstained_targets"])
            for key in coverage:
                item = evidence.metadata.get(key)
                if item is not None and isinstance(item, (str, int, float, bool)):
                    coverage[key].add(item)
            del prediction, loss, truth, evidence, generated
            if (index + 1) % 16 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    state_after = _model_state_hash(model)
    if state_before != state_after:
        raise RuntimeError("checkpoint evaluation mutated model state")

    panel = ProgramEvaluationPanel(
        partition=request.partition,
        root_seed=request.root_seed,
        worlds=request.worlds,
        address_prefix=request.address_prefix,
        addresses_hash=canonical_hash(addresses),
        coverage={
            key: tuple(sorted(values, key=lambda item: canonical_json(item)))
            for key, values in sorted(coverage.items())
        },
    )
    metrics = ProgramEvaluationMetrics(
        mean_loss=sum(losses) / len(losses),
        median_loss=statistics.median(losses),
        minimum_loss=min(losses),
        maximum_loss=max(losses),
        per_world_loss=tuple(losses),
        scored_targets=scored_targets,
        abstained_targets=abstained_targets,
    )
    evaluation_path = Path(__file__).resolve()
    execution: dict[str, JsonValue] = {
        "architecture": platform.machine().lower(),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "device": str(target_device),
        "numpy_version": str(np.__version__),
        "operating_system": platform.system().lower(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
    }
    checkpoint_artifact = ProgramArtifact(
        name=checkpoint_path.name,
        sha256=checkpoint_hash,
        size_bytes=checkpoint_path.stat().st_size,
    )
    sidecar_artifact = ProgramArtifact(
        name=sidecar_path.name,
        sha256=sidecar_hash,
        size_bytes=sidecar_path.stat().st_size,
    )
    tool_artifact = ProgramArtifact(
        name="src/tabu_lab/evolution/evaluation.py",
        sha256=file_sha256(evaluation_path),
        size_bytes=evaluation_path.stat().st_size,
    )
    payload = {
        "schema_version": "tabu.program-checkpoint-evaluation-receipt.v1",
        "evidence_status": EvidenceStatus.LOCAL_UNISSUED,
        "claim_boundary": request.claim_boundary,
        "request": request.model_dump(mode="python"),
        "request_hash": request.request_hash,
        "source_repository_hash": repository.repository_hash,
        "evaluation_protocol_ref": evaluation.ref,
        "evaluation_protocol_hash": evaluation.node_hash,
        "objective_bundle_ref": bundle.ref,
        "objective_bundle_hash": bundle.node_hash,
        "world_mixture_ref": mixture.ref,
        "world_mixture_hash": mixture.node_hash,
        "generator_ref": generator.ref,
        "generator_hash": generator.node_hash,
        "snapshot_hash": resolved.snapshot_hash,
        "run_identity_hash": metadata.run_identity_hash,
        "checkpoint": checkpoint_artifact.model_dump(mode="python"),
        "checkpoint_sidecar": sidecar_artifact.model_dump(mode="python"),
        "checkpoint_metadata_hash": metadata.metadata_hash,
        "checkpoint_step": metadata.update_cursor,
        "evaluation_tool": tool_artifact.model_dump(mode="python"),
        "evaluation_source_revision": evaluation_source_revision,
        "evaluation_source_archive_sha256": evaluation_source_archive_sha256,
        "execution": execution,
        "panel": panel.model_dump(mode="python"),
        "metrics": metrics.model_dump(mode="python"),
        "model_state_hash_before": state_before,
        "model_state_hash_after": state_after,
    }
    receipt = ProgramCheckpointEvaluationReceipt(
        **payload,
        receipt_hash=canonical_hash(payload),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_json(receipt.model_dump(mode="python")) + "\n",
        encoding="utf-8",
    )
    return ProgramCheckpointEvaluationResult(receipt=receipt, receipt_path=destination)


__all__ = [
    "ProgramCheckpointEvaluationReceipt",
    "ProgramCheckpointEvaluationRequest",
    "ProgramCheckpointEvaluationResult",
    "ProgramEvaluationMetrics",
    "ProgramEvaluationPanel",
    "evaluate_program_checkpoint",
    "load_program_evaluation_request",
]
