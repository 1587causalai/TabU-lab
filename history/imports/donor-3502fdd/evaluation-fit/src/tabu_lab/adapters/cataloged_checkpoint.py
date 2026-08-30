"""Content-verify a cataloged TabU safetensors checkpoint without executing it.

Verification is intentionally separate from model evaluation.  A v2
``ModelArtifact`` binds the checkpoint bytes, ModelSpec, resolved semantic
configuration, compiler manifest, producer RunIdentity, and embedded resume
contract.  The verified record can therefore be handed to the isolated
truth-free Evaluation Foundry checkpoint adapter without trusting Python code
stored in the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tabu_lab.catalog import (
    CatalogIndex,
    CatalogObjectKind,
    ModelArtifact,
    load_catalog,
)
from tabu_lab.contracts import canonical_hash, require_sha256
from tabu_lab.evidence import RunIdentity


class CatalogedCheckpointError(ValueError):
    """Checkpoint bytes or embedded identity disagree with the catalog."""


@dataclass(frozen=True, slots=True)
class VerifiedCatalogedCheckpoint:
    artifact_id: str
    contract_id: str
    contract_version: str
    producer_run_id: str
    checkpoint_sha256: str
    checkpoint_schema_version: str
    experiment_spec_sha256: str
    source_code_sha256: str
    compiler_sha256: str
    model_spec_sha256: str
    semantic_config_sha256: str
    run_identity_sha256: str
    step: int
    model_tensor_names: tuple[str, ...]


_HEADER_FIELDS = frozenset(
    {
        "schema",
        "resume_contract",
        "run_identity",
        "training_config",
        "execution_config",
        "step",
        "optimizer_param_groups",
        "optimizer_state_scalars",
        "optimizer_tensor_fields",
        "rng",
    }
)
_RESUME_FIELDS = frozenset(
    {
        "checkpoint_schema_version",
        "contract_version",
        "execution_config_hash",
        "model_id",
        "model_spec_hash",
        "model_state_schema_version",
        "objective_config",
        "objective_type",
        "optimizer_defaults",
        "optimizer_state_schema_version",
        "optimizer_type",
        "optimizer_version",
        "rng_state_schema_version",
        "run_id",
        "run_identity_hash",
        "semantic_config_hash",
        "training_config_hash",
    }
)
_OPTIONAL_RESUME_FIELDS = frozenset({"model_identity"})
_MODEL_IDENTITY_FIELDS = frozenset(
    {
        "bandwidth",
        "contract_version",
        "label_broadcast",
        "label_broadcast_tau",
        "ll_ridge",
        "model_id",
        "profile_id",
        "reference_config",
        "terminal",
        "tokenizer_version",
        "variant_hash",
        "variant_ref",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogedCheckpointError(f"checkpoint {field_name} must be a JSON object")
    return value


def resolve_model_artifact(
    catalog: CatalogIndex | str | Path,
    artifact_id: str,
) -> ModelArtifact:
    """Resolve one typed ``ModelArtifact`` from a catalog index or file."""

    index = catalog if isinstance(catalog, CatalogIndex) else load_catalog(Path(catalog))
    entry = index.show(artifact_id)
    if entry.kind is not CatalogObjectKind.MODEL_ARTIFACT:
        raise CatalogedCheckpointError(f"catalog object {artifact_id!r} is not a model artifact")
    return ModelArtifact.model_validate(entry.data)


def verify_artifact_checkpoint(
    artifact: ModelArtifact,
    checkpoint_path: str | Path,
) -> VerifiedCatalogedCheckpoint:
    """Verify bytes and embedded training identity for one artifact record."""

    source = Path(checkpoint_path)
    if artifact.checkpoint_format.casefold() != "safetensors" or source.suffix != ".safetensors":
        raise CatalogedCheckpointError("cataloged TabU checkpoints must use safetensors")
    if not source.is_file():
        raise CatalogedCheckpointError(f"checkpoint file does not exist: {source}")
    observed_sha256 = _sha256_file(source)
    if observed_sha256 != artifact.checkpoint.sha256:
        raise CatalogedCheckpointError("checkpoint byte hash does not match ModelArtifact")

    from safetensors import safe_open

    try:
        with safe_open(str(source), framework="pt", device="cpu") as checkpoint:
            metadata = checkpoint.metadata() or {}
            encoded_header = metadata.get("tabu_training_state")
            tensor_names = tuple(sorted(checkpoint.keys()))
    except Exception as error:
        raise CatalogedCheckpointError("checkpoint is not a readable safetensors file") from error
    if not isinstance(encoded_header, str):
        raise CatalogedCheckpointError("checkpoint is missing TabU training metadata")
    try:
        header = json.loads(encoded_header)
    except json.JSONDecodeError as error:
        raise CatalogedCheckpointError("checkpoint training metadata is not valid JSON") from error
    header = _mapping(header, field_name="header")
    if set(header) != _HEADER_FIELDS:
        raise CatalogedCheckpointError("checkpoint header is missing or has unknown fields")
    resume = _mapping(header["resume_contract"], field_name="resume_contract")
    if not _RESUME_FIELDS.issubset(resume) or set(resume) - (
        _RESUME_FIELDS | _OPTIONAL_RESUME_FIELDS
    ):
        raise CatalogedCheckpointError("checkpoint resume contract is incomplete")
    try:
        identity = RunIdentity.model_validate(header["run_identity"])
    except ValueError as error:
        raise CatalogedCheckpointError("checkpoint RunIdentity is invalid") from error

    if header["schema"] != resume["checkpoint_schema_version"]:
        raise CatalogedCheckpointError("checkpoint schema disagrees with its resume contract")
    if artifact.checkpoint_schema_version != resume["checkpoint_schema_version"]:
        raise CatalogedCheckpointError("checkpoint schema disagrees with ModelArtifact")
    if artifact.model_state_schema_version != resume["model_state_schema_version"]:
        raise CatalogedCheckpointError("model-state schema disagrees with ModelArtifact")
    if resume["model_id"] != artifact.contract_id:
        raise CatalogedCheckpointError("checkpoint model id disagrees with ModelArtifact")
    if resume["contract_version"] != artifact.contract_version:
        raise CatalogedCheckpointError("checkpoint contract version disagrees with ModelArtifact")
    if resume["run_id"] != artifact.producer_run_id or identity.run_id != artifact.producer_run_id:
        raise CatalogedCheckpointError("checkpoint run id disagrees with producer run")
    if resume["run_identity_hash"] != identity.identity_hash:
        raise CatalogedCheckpointError("checkpoint resume contract does not bind RunIdentity")
    if canonical_hash(header["training_config"]) != identity.training_config_hash:
        raise CatalogedCheckpointError("checkpoint training config does not bind RunIdentity")
    if canonical_hash(header["execution_config"]) != identity.execution_config_hash:
        raise CatalogedCheckpointError("checkpoint execution config does not bind RunIdentity")
    if resume["training_config_hash"] != identity.training_config_hash:
        raise CatalogedCheckpointError("resume training hash disagrees with RunIdentity")
    if resume["execution_config_hash"] != identity.execution_config_hash:
        raise CatalogedCheckpointError("resume execution hash disagrees with RunIdentity")

    model_spec_sha256 = require_sha256(resume["model_spec_hash"], field_name="model_spec_hash")
    semantic_sha256 = require_sha256(
        resume["semantic_config_hash"], field_name="semantic_config_hash"
    )
    if artifact.model_spec.sha256 != model_spec_sha256:
        raise CatalogedCheckpointError("checkpoint ModelSpec hash disagrees with ModelArtifact")
    if artifact.semantic_config.sha256 != semantic_sha256:
        raise CatalogedCheckpointError("checkpoint semantic hash disagrees with ModelArtifact")
    if artifact.compiler_manifest.sha256 != identity.compiler_hash:
        raise CatalogedCheckpointError("checkpoint compiler hash disagrees with ModelArtifact")
    if "model_identity" in resume:
        model_identity = _mapping(resume["model_identity"], field_name="model_identity")
        if set(model_identity) != _MODEL_IDENTITY_FIELDS:
            raise CatalogedCheckpointError("checkpoint model identity is incomplete")
        variant_ref = _mapping(model_identity["variant_ref"], field_name="variant_ref")
        if (
            model_identity["model_id"] != artifact.contract_id
            or model_identity["contract_version"] != artifact.contract_version
            or variant_ref.get("contract_id") != artifact.contract_id
            or variant_ref.get("contract_version") != artifact.contract_version
            or variant_ref.get("model_spec_hash") != model_spec_sha256
            or variant_ref.get("semantic_config_hash") != semantic_sha256
        ):
            raise CatalogedCheckpointError(
                "checkpoint model identity disagrees with its bound contract"
            )
    step = header["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CatalogedCheckpointError("checkpoint step must be a non-negative integer")
    model_tensors = tuple(name for name in tensor_names if name.startswith("model."))
    if not model_tensors:
        raise CatalogedCheckpointError("checkpoint contains no model tensors")

    return VerifiedCatalogedCheckpoint(
        artifact_id=artifact.artifact_id,
        contract_id=artifact.contract_id,
        contract_version=artifact.contract_version,
        producer_run_id=artifact.producer_run_id,
        checkpoint_sha256=observed_sha256,
        checkpoint_schema_version=str(header["schema"]),
        experiment_spec_sha256=identity.spec_hash,
        source_code_sha256=identity.code_hash,
        compiler_sha256=identity.compiler_hash,
        model_spec_sha256=model_spec_sha256,
        semantic_config_sha256=semantic_sha256,
        run_identity_sha256=identity.identity_hash,
        step=step,
        model_tensor_names=model_tensors,
    )


def verify_cataloged_checkpoint(
    *,
    catalog: CatalogIndex | str | Path,
    artifact_id: str,
    checkpoint_path: str | Path,
) -> VerifiedCatalogedCheckpoint:
    """Resolve a catalog record and verify the corresponding local bytes."""

    return verify_artifact_checkpoint(
        resolve_model_artifact(catalog, artifact_id),
        checkpoint_path,
    )


__all__ = [
    "CatalogedCheckpointError",
    "VerifiedCatalogedCheckpoint",
    "resolve_model_artifact",
    "verify_artifact_checkpoint",
    "verify_cataloged_checkpoint",
]
