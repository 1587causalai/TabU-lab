"""Deterministic discovery, generation, and verification of ``catalog.json``."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ValidationError

from tabu_lab.contracts.canonical import (
    canonical_hash,
    canonical_json,
    to_canonical_data,
)
from tabu_lab.evaluation.foundry.contracts import (
    ComparisonReport,
    EvalResult,
    EvalSuiteSpec,
    ProducerProvenance,
)
from tabu_lab.evidence.receipt_io import ReceiptEnvelope, read_receipt
from tabu_lab.evidence.schemas import (
    ClaimLedger,
    Preregistration,
    Receipt,
    ReceiptStatus,
    RunBundle,
)
from tabu_lab.registry import ModelSpec

from .models import (
    ArtifactStatusEvent,
    CatalogCheckIssue,
    CatalogCheckReport,
    CatalogEntry,
    CatalogIndex,
    CatalogObjectKind,
    ClaimRecord,
    DatasetAdapter,
    DatasetSnapshotSpec,
    EvidencePointer,
    ExperimentRecord,
    ExperimentStatus,
    FailureCategory,
    LineageEdge,
    LineageRelation,
    ModelArtifact,
    ObjectRef,
    ReviewRecord,
    RunAttemptRecord,
    RunAttemptStatus,
    RunRecord,
    RunStatus,
    StatusEvent,
)
from .source_revision import CatalogSourceRevision

if TYPE_CHECKING:
    from tabu_lab.evidence.formal_authorization import FormalAuthorizationReplaySession


class CatalogBuildError(ValueError):
    """A canonical catalog source cannot be loaded or linked safely."""


_SOURCE_DIRECTORIES = (
    "specs/models",
    "experiments",
    "runs",
    "artifacts",
    "datasets",
    "evaluations/suites",
    "evaluations/results",
    "claims",
    "reviews",
    "verification/suites",
    "verification/results",
)
_SOURCE_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


@dataclass(frozen=True, slots=True)
class _RunProjection:
    run: RunRecord
    entries: tuple[CatalogEntry, ...]
    edges: tuple[LineageEdge, ...]
    source: dict[str, Any]
    source_path: str


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise CatalogBuildError(f"cannot read catalog source {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise CatalogBuildError(f"catalog source must be a mapping: {path}")
    return raw


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise CatalogBuildError(f"catalog source escapes repository root: {path}") from exc


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths: list[Path] = []
    for directory_name in _SOURCE_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        candidates = tuple(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in _SOURCE_SUFFIXES
        )
        if directory_name == "specs/models":
            # Nested manifests are immutable registry history.  Until catalog
            # ModelContract identities become fully version-qualified, only
            # current aliases are public catalog sources.  Including history
            # under the legacy bare contract_id would either collide or,
            # worse, silently misattribute old evidence to a newer contract.
            # The IMPLEMENTS hash gate in CatalogIndex keeps an alias advance
            # fail-closed during this transition.
            candidates = tuple(path for path in candidates if path.parent == directory)
        if directory_name == "runs":
            # Attempt directories contain many JSON payloads, but only the
            # receipt envelope is a canonical catalog input. Seed aggregates
            # are derived summaries and are independently checksum-verifiable;
            # they do not create evidence objects or public claims.
            # Direct children remain available for explicit Run/Attempt
            # manifests. Unknown nested payloads cannot silently become
            # catalog objects.
            paths.extend(
                path
                for path in candidates
                if path.parent == directory or path.name == "receipt.json"
            )
        else:
            paths.extend(candidates)
    return tuple(sorted(set(paths), key=lambda item: _relative(item, root)))


def _json_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    payload: Any
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", by_alias=False, exclude_none=False)
    else:
        payload = to_canonical_data(value)
    if not isinstance(payload, dict):  # pragma: no cover - caller contract
        raise TypeError("catalog entry data must be a mapping")
    return payload


def _entry(
    *,
    kind: CatalogObjectKind,
    object_id: str,
    value: BaseModel | Mapping[str, Any],
    source: Mapping[str, Any],
    source_path: str,
    status: str | None = None,
) -> CatalogEntry:
    data = _json_mapping(value)
    schema_version = data.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise CatalogBuildError(f"catalog object {object_id!r} lacks schema_version")
    return CatalogEntry(
        kind=kind,
        object_id=object_id,
        object_schema_version=schema_version,
        object_hash=canonical_hash(data),
        source_hash=canonical_hash(source),
        source_path=source_path,
        status=status,
        data=data,
    )


def _edge(
    source_kind: CatalogObjectKind,
    source_id: str,
    relation: LineageRelation,
    target_kind: CatalogObjectKind,
    target_id: str,
    *,
    evidence_hash: str | None = None,
) -> LineageEdge:
    return LineageEdge(
        source=ObjectRef(kind=source_kind, object_id=source_id),
        relation=relation,
        target=ObjectRef(kind=target_kind, object_id=target_id),
        evidence_hash=evidence_hash,
    )


def _fit_experiment(
    raw: dict[str, Any],
    source_path: str,
    *,
    current_model_hashes: Mapping[str, str] | None = None,
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    # Import lazily so catalog inspection remains usable independently of the fit runner.
    from tabu_lab.experiments.contracts import FitExperimentSpec

    spec = FitExperimentSpec.model_validate(raw)
    preregistration_hash = canonical_hash(raw)
    split_hash = canonical_hash(spec.split)
    dataset_snapshot_id = f"{spec.dataset.dataset_id}-{split_hash[:16]}"
    dataset = DatasetSnapshotSpec(
        dataset_snapshot_id=dataset_snapshot_id,
        dataset_id=spec.dataset.dataset_id,
        source_uri=spec.dataset.source_uri,
        source_sha256=spec.dataset.source_sha256,
        content_sha256=spec.dataset.dataset_hash,
        license_id=spec.dataset.license_id,
        split_manifest_sha256=split_hash,
        fit_partition=spec.split.fit_partition,
        adapter=DatasetAdapter(
            adapter_id=spec.dataset.adapter.adapter_id,
            adapter_version=spec.dataset.adapter.adapter_version,
        ),
        episode_recipe_hashes=spec.episode_schedule.recipe_hashes,
        mask_boundary=(
            "targets are compiler receivers only; artificial-mask truth enters the loss sidecar"
        ),
        contamination_boundary=(
            "split-before-compile; statistics, codebooks, schedules, and transforms use only "
            f"the {spec.split.fit_partition} partition"
        ),
    )
    experiment = ExperimentRecord(
        experiment_id=spec.experiment_id,
        contract_id=spec.contract_id,
        hypothesis=(
            f"{spec.stage.value} bounded support-realizable fit for {spec.contract_id} "
            "under the frozen preregistration"
        ),
        claim_boundary=(
            "frozen-episode fit evidence only; not generalization, pretraining, benchmark, "
            "supported-model, or foundation-model evidence"
        ),
        status=ExperimentStatus.DRAFT,
        status_history=(
            StatusEvent(
                status=ExperimentStatus.DRAFT.value,
                evidence_hashes=(preregistration_hash,),
                note="local preregistration source; promotion requires reviewed commit evidence",
            ),
        ),
        preregistration=EvidencePointer(
            uri=source_path,
            sha256=preregistration_hash,
            media_type="application/yaml",
        ),
        dataset_snapshot_ids=(dataset_snapshot_id,),
        supersedes_experiment_ids=spec.supersedes_experiment_ids,
        revision_rationale=spec.revision_rationale,
    )
    # A preregistration is allowed to retain an immutable historical ModelSpec
    # while the bare contract id advances to a newer canonical source.  Keep
    # that historical identity in the catalog under a content-qualified id;
    # otherwise the IMPLEMENTS edge would silently attribute old evidence to
    # the newer contract and the graph validator must (correctly) reject it.
    # The alias is a projection of the embedded spec, not a second source of
    # truth, and equivalent aliases are coalesced deterministically below.
    historical_contract_id: str | None = None
    if current_model_hashes is not None and current_model_hashes.get(spec.contract_id) not in {
        None,
        spec.model_spec_hash,
    }:
        historical_contract_id = f"{spec.contract_id}@{spec.model_spec_hash}"

    entries = [
        _entry(
            kind=CatalogObjectKind.EXPERIMENT,
            object_id=experiment.experiment_id,
            value=experiment,
            source=raw,
            source_path=source_path,
            status=experiment.status.value,
        ),
        _entry(
            kind=CatalogObjectKind.DATASET_SNAPSHOT,
            object_id=dataset.dataset_snapshot_id,
            value=dataset,
            source={"dataset": raw["dataset"], "split": raw["split"]},
            source_path=source_path,
        ),
    ]
    if historical_contract_id is not None:
        entries.append(
            _entry(
                kind=CatalogObjectKind.MODEL_CONTRACT,
                object_id=historical_contract_id,
                value=spec.model_spec,
                source=raw["model_spec"],
                source_path=source_path,
                status=spec.model_spec.maturity.stage.value,
            )
        )
    edges = [
        _edge(
            CatalogObjectKind.EXPERIMENT,
            experiment.experiment_id,
            LineageRelation.IMPLEMENTS,
            CatalogObjectKind.MODEL_CONTRACT,
            historical_contract_id or experiment.contract_id,
            evidence_hash=spec.model_spec_hash,
        ),
        _edge(
            CatalogObjectKind.EXPERIMENT,
            experiment.experiment_id,
            LineageRelation.USES_DATA,
            CatalogObjectKind.DATASET_SNAPSHOT,
            dataset.dataset_snapshot_id,
            evidence_hash=dataset.content_sha256,
        ),
    ]
    edges.extend(
        _edge(
            CatalogObjectKind.EXPERIMENT,
            experiment.experiment_id,
            LineageRelation.SUPERSEDES,
            CatalogObjectKind.EXPERIMENT,
            previous_id,
            evidence_hash=preregistration_hash,
        )
        for previous_id in experiment.supersedes_experiment_ids
    )
    return entries, edges


def _legacy_preregistration(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    prereg = Preregistration.model_validate(raw)
    source_hash = canonical_hash(raw)
    experiment = ExperimentRecord(
        experiment_id=prereg.experiment_id,
        contract_id=prereg.contract.contract_id,
        hypothesis=prereg.hypothesis,
        claim_boundary=prereg.claim_boundary,
        status=ExperimentStatus.DRAFT,
        status_history=(
            StatusEvent(
                status=ExperimentStatus.DRAFT.value,
                evidence_hashes=(source_hash,),
                note="local preregistration source; promotion requires reviewed commit evidence",
            ),
        ),
        preregistration=EvidencePointer(
            uri=source_path,
            sha256=source_hash,
            media_type="application/yaml",
        ),
    )
    return (
        [
            _entry(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id=experiment.experiment_id,
                value=experiment,
                source=raw,
                source_path=source_path,
                status=experiment.status.value,
            )
        ],
        [
            _edge(
                CatalogObjectKind.EXPERIMENT,
                experiment.experiment_id,
                LineageRelation.IMPLEMENTS,
                CatalogObjectKind.MODEL_CONTRACT,
                experiment.contract_id,
            )
        ],
    )


def _explicit_experiment(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    record = ExperimentRecord.model_validate(raw)
    edges = [
        _edge(
            CatalogObjectKind.EXPERIMENT,
            record.experiment_id,
            LineageRelation.IMPLEMENTS,
            CatalogObjectKind.MODEL_CONTRACT,
            record.contract_id,
        )
    ]
    edges.extend(
        _edge(
            CatalogObjectKind.EXPERIMENT,
            record.experiment_id,
            LineageRelation.USES_DATA,
            CatalogObjectKind.DATASET_SNAPSHOT,
            dataset_id,
        )
        for dataset_id in record.dataset_snapshot_ids
    )
    edges.extend(
        _edge(
            CatalogObjectKind.EXPERIMENT,
            record.experiment_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RUN,
            run_id,
        )
        for run_id in record.run_ids
    )
    edges.extend(
        _edge(
            CatalogObjectKind.EXPERIMENT,
            record.experiment_id,
            LineageRelation.SUPERSEDES,
            CatalogObjectKind.EXPERIMENT,
            previous_id,
            evidence_hash=(
                record.preregistration.sha256 if record.preregistration is not None else None
            ),
        )
        for previous_id in record.supersedes_experiment_ids
    )
    return (
        [
            _entry(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id=record.experiment_id,
                value=record,
                source=raw,
                source_path=source_path,
                status=record.status.value,
            )
        ],
        edges,
    )


def _run_record(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    record = RunRecord.model_validate(raw)
    edges = [
        _edge(
            CatalogObjectKind.EXPERIMENT,
            record.experiment_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RUN,
            record.run_id,
        )
    ]
    edges.extend(
        _edge(
            CatalogObjectKind.RUN,
            record.run_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RUN_ATTEMPT,
            attempt_id,
        )
        for attempt_id in record.attempt_ids
    )
    edges.extend(
        _edge(
            CatalogObjectKind.RUN,
            record.run_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact_id,
        )
        for artifact_id in record.artifact_ids
    )
    if record.resumes_from_run_id is not None:
        edges.append(
            _edge(
                CatalogObjectKind.RUN,
                record.run_id,
                LineageRelation.RESUMES_FROM,
                CatalogObjectKind.RUN,
                record.resumes_from_run_id,
            )
        )
    return (
        [
            _entry(
                kind=CatalogObjectKind.RUN,
                object_id=record.run_id,
                value=record,
                source=raw,
                source_path=source_path,
                status=record.status.value,
            )
        ],
        edges,
    )


def _attempt_record(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    record = RunAttemptRecord.model_validate(raw)
    return (
        [
            _entry(
                kind=CatalogObjectKind.RUN_ATTEMPT,
                object_id=record.attempt_id,
                value=record,
                source=raw,
                source_path=source_path,
                status=record.status.value,
            )
        ],
        [
            _edge(
                CatalogObjectKind.RUN,
                record.run_id,
                LineageRelation.PRODUCED,
                CatalogObjectKind.RUN_ATTEMPT,
                record.attempt_id,
            )
        ],
    )


def _model_artifact(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    artifact = ModelArtifact.model_validate(raw)
    edges = [
        _edge(
            CatalogObjectKind.RUN,
            artifact.producer_run_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact.artifact_id,
            evidence_hash=artifact.producer_receipt.sha256,
        ),
        _edge(
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact.artifact_id,
            LineageRelation.IMPLEMENTS,
            CatalogObjectKind.MODEL_CONTRACT,
            artifact.contract_id,
        ),
    ]
    edges.extend(
        _edge(
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact.artifact_id,
            LineageRelation.EVALUATED_BY,
            CatalogObjectKind.EVAL_RESULT,
            result_id,
        )
        for result_id in artifact.evaluation_result_ids
    )
    return (
        [
            _entry(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id=artifact.artifact_id,
                value=artifact,
                source=raw,
                source_path=source_path,
                status=artifact.status.value,
            )
        ],
        edges,
    )


def _review_record(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    review = ReviewRecord.model_validate(raw)
    return (
        [
            _entry(
                kind=CatalogObjectKind.REVIEW,
                object_id=review.review_id,
                value=review,
                source=raw,
                source_path=source_path,
                status=review.decision.value,
            )
        ],
        [],
    )


def _claim_record(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    claim = ClaimRecord.model_validate(raw)
    edges = [
        _edge(
            reference.kind,
            reference.object_id,
            LineageRelation.SUPPORTS,
            CatalogObjectKind.CLAIM,
            claim.claim_id,
        )
        for reference in claim.evidence
        if reference.kind
        in {
            CatalogObjectKind.EVAL_RESULT,
            CatalogObjectKind.REVIEW,
            CatalogObjectKind.RECEIPT,
        }
    ]
    return (
        [
            _entry(
                kind=CatalogObjectKind.CLAIM,
                object_id=claim.claim_id,
                value=claim,
                source=raw,
                source_path=source_path,
                status=claim.status.value,
            )
        ],
        edges,
    )


def _claim_ledger(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    ledger = ClaimLedger.model_validate(raw)
    entries: list[CatalogEntry] = []
    for claim in ledger.claims:
        # Legacy proposed claims remain proposals.  Accepted legacy claims already
        # carry their own receipt/review/approval gate in ClaimLedger validation.
        data = claim.model_dump(mode="json", exclude_none=False)
        data["schema_version"] = "tabu.claim-record.v1"
        entries.append(
            _entry(
                kind=CatalogObjectKind.CLAIM,
                object_id=claim.claim_id,
                value=data,
                source=raw,
                source_path=source_path,
                status=claim.status.value,
            )
        )
    return entries, []


def _receipt(raw: dict[str, Any], source_path: str) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    receipt = Receipt.model_validate(raw)
    return (
        [
            _entry(
                kind=CatalogObjectKind.RECEIPT,
                object_id=receipt.receipt_id,
                value=receipt,
                source=raw,
                source_path=source_path,
                status=receipt.status.value,
            )
        ],
        [],
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_contract(
    checkpoint_path: Path,
    *,
    run_id: str,
    model_id: str,
    model_spec_hash: str,
    semantic_config_hash: str,
) -> tuple[str, str]:
    """Read inert safetensors metadata and bind it to the producer identity."""

    from safetensors import SafetensorError, safe_open

    try:
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            encoded = (checkpoint.metadata() or {}).get("tabu_training_state")
        if not isinstance(encoded, str):
            raise ValueError("missing tabu_training_state")
        header = json.loads(encoded)
        resume = header["resume_contract"]
    except (OSError, KeyError, TypeError, ValueError, SafetensorError) as exc:
        raise CatalogBuildError("formal checkpoint has invalid training metadata") from exc
    required = {
        "checkpoint_schema_version",
        "model_state_schema_version",
        "model_id",
        "model_spec_hash",
        "run_id",
        "semantic_config_hash",
    }
    if (
        not isinstance(header, dict)
        or not isinstance(resume, dict)
        or not required.issubset(resume)
    ):
        raise CatalogBuildError("formal checkpoint resume contract is incomplete")
    if (
        header.get("schema") != resume["checkpoint_schema_version"]
        or resume["run_id"] != run_id
        or resume["model_id"] != model_id
        or resume["model_spec_hash"] != model_spec_hash
        or resume["semantic_config_hash"] != semantic_config_hash
    ):
        raise CatalogBuildError("formal checkpoint identity differs from its producer run")
    checkpoint_schema = resume["checkpoint_schema_version"]
    model_state_schema = resume["model_state_schema_version"]
    if not isinstance(checkpoint_schema, str) or not checkpoint_schema:
        raise CatalogBuildError("formal checkpoint schema version is invalid")
    if not isinstance(model_state_schema, str) or not model_state_schema:
        raise CatalogBuildError("formal model-state schema version is invalid")
    return checkpoint_schema, model_state_schema


def _verify_attempt_checksums(directory: Path) -> None:
    if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob("*")):
        raise CatalogBuildError("run attempt artifacts cannot contain symlinks")
    checksum_path = directory / "artifacts.sha256"
    if not checksum_path.is_file():
        raise CatalogBuildError("run attempt is missing artifacts.sha256")
    listed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        candidate = (directory / relative).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exc:
            raise CatalogBuildError("run attempt checksum path escapes its directory") from exc
        if (
            separator != "  "
            or not relative
            or relative in listed
            or not candidate.is_file()
            or _sha256_file(candidate) != digest
        ):
            raise CatalogBuildError("run attempt checksum manifest is invalid or has drift")
        listed[relative] = digest
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifacts.sha256"
    }
    if set(listed) != actual:
        raise CatalogBuildError("run attempt checksum manifest does not cover exact files")


def _project_failure_category(bundle: RunBundle) -> FailureCategory:
    phase = str(bundle.metadata.get("failure_phase", "infrastructure"))
    if phase in {"dataset", "split", "compile", "compiler"}:
        return FailureCategory.DATA
    if phase in {"initial_evaluation", "final_evaluation", "evaluation"}:
        return FailureCategory.EVALUATOR
    if phase in {"checkpoint", "artifact"}:
        return FailureCategory.ARTIFACT
    if phase in {"build", "train"}:
        return FailureCategory.MODEL
    return FailureCategory.INFRASTRUCTURE


def _fit_receipt_projection(
    *,
    raw: dict[str, Any],
    source_path: str,
    path: Path,
    authorization_replay: FormalAuthorizationReplaySession | None,
) -> _RunProjection:
    envelope = ReceiptEnvelope.model_validate(raw)
    receipt = read_receipt(path)
    if receipt != envelope.receipt:
        raise CatalogBuildError("receipt envelope read-back differs from its catalog payload")
    directory = path.parent
    _verify_attempt_checksums(directory)
    try:
        bundle = RunBundle.model_validate_json(
            (directory / "run_bundle.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise CatalogBuildError("receipt attempt has an invalid RunBundle") from exc
    if (
        bundle.run_id != receipt.run_id
        or bundle.identity.identity_hash != receipt.run_identity_hash
        or bundle.run_bundle_hash != receipt.run_bundle_hash
        or bundle.artifacts != receipt.artifacts
    ):
        raise CatalogBuildError("receipt and RunBundle identity bindings differ")
    for artifact in receipt.artifacts:
        candidate = (directory / artifact.uri).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exc:
            raise CatalogBuildError("receipt artifact path escapes its attempt") from exc
        if not candidate.is_file() or _sha256_file(candidate) != artifact.sha256:
            raise CatalogBuildError("receipt artifact bytes do not match their digest")

    experiment_id = bundle.metadata.get("experiment_id")
    attempt_id = receipt.metadata.get("attempt_id")
    if not isinstance(experiment_id, str) or not isinstance(attempt_id, str):
        raise CatalogBuildError("fit receipt lacks experiment_id or attempt_id")
    if bundle.metadata.get("attempt_id") != attempt_id:
        raise CatalogBuildError("receipt and RunBundle attempt ids differ")
    issuance_status = receipt.metadata.get("issuance_status")
    if issuance_status != bundle.metadata.get("issuance_status") or issuance_status not in {
        "formal",
        "local_unissued",
    }:
        raise CatalogBuildError("receipt issuance status is invalid or unbound")
    if issuance_status == "formal" and authorization_replay is None:
        raise CatalogBuildError("formal receipt projection requires canonical Git-history replay")
    from tabu_lab.evaluation.fit_artifacts import verify_fit_attempt_artifacts

    try:
        verified_receipt = verify_fit_attempt_artifacts(
            directory,
            formal_authorization_replay=authorization_replay,
        )
    except (OSError, ValueError) as exc:
        raise CatalogBuildError(
            "fit receipt failed complete artifact and authorization replay"
        ) from exc
    if verified_receipt != receipt:
        raise CatalogBuildError("fit receipt replay returned a different receipt")

    preregistration_path = directory / "preregistration.yaml"
    try:
        preregistration_raw = yaml.safe_load(preregistration_path.read_text(encoding="utf-8"))
        from tabu_lab.experiments.contracts import FitExperimentSpec

        spec = FitExperimentSpec.model_validate(preregistration_raw)
    except (OSError, yaml.YAMLError, ValidationError, ValueError) as exc:
        raise CatalogBuildError("fit receipt has an invalid frozen preregistration") from exc
    if spec.experiment_id != experiment_id or spec.contract_id != bundle.model_id:
        raise CatalogBuildError("receipt metadata differs from its frozen preregistration")
    split_hash = canonical_hash(spec.split)
    dataset_snapshot_id = f"{spec.dataset.dataset_id}-{split_hash[:16]}"

    receipt_pointer = EvidencePointer(
        uri=source_path,
        sha256=receipt.receipt_hash,
        media_type="application/json",
    )
    if receipt.status is ReceiptStatus.SUCCEEDED:
        run_status = RunStatus.SUCCEEDED
        attempt_status = RunAttemptStatus.SUCCEEDED
        failure_category = None
    elif receipt.status is ReceiptStatus.FAILED:
        run_status = RunStatus.FAILED
        attempt_status = RunAttemptStatus.FAILED
        failure_category = _project_failure_category(bundle)
    elif receipt.status is ReceiptStatus.CANCELLED:
        run_status = RunStatus.KILLED
        attempt_status = RunAttemptStatus.KILLED
        failure_category = FailureCategory.KILL_CONDITION
    else:
        raise CatalogBuildError("canonical fit attempt receipts must be terminal")

    artifact: ModelArtifact | None = None
    artifact_ids: tuple[str, ...] = ()
    if issuance_status == "formal" and receipt.status is ReceiptStatus.SUCCEEDED:
        checkpoints = tuple(item for item in receipt.artifacts if item.kind == "checkpoint")
        if len(checkpoints) != 1:
            raise CatalogBuildError("formal successful fit receipt needs one checkpoint")
        checkpoint = checkpoints[0]
        contract_version = bundle.metadata.get("contract_version")
        license_id = bundle.metadata.get("checkpoint_license_id")
        if not isinstance(contract_version, str) or not isinstance(license_id, str):
            raise CatalogBuildError("formal checkpoint lacks contract version or license")
        semantic_path = directory / "resolved-configs" / "semantic.json"
        compiler_path = directory / "compiler-manifest.json"
        semantic_source = _read_mapping(semantic_path)
        compiler_source = _read_mapping(compiler_path)
        if canonical_hash(semantic_source) != bundle.identity.semantic_config_hash:
            raise CatalogBuildError("formal semantic config differs from RunIdentity")
        if canonical_hash(compiler_source) != bundle.identity.compiler_hash:
            raise CatalogBuildError("formal compiler manifest differs from RunIdentity")
        receipt_artifacts = {item.uri: item for item in receipt.artifacts}
        semantic_artifact = receipt_artifacts.get("resolved-configs/semantic.json")
        compiler_artifact = receipt_artifacts.get("compiler-manifest.json")
        if semantic_artifact is None or compiler_artifact is None:
            raise CatalogBuildError(
                "formal receipt does not retain semantic config and compiler manifest"
            )
        checkpoint_schema, model_state_schema = _checkpoint_contract(
            directory / checkpoint.uri,
            run_id=receipt.run_id,
            model_id=bundle.model_id,
            model_spec_hash=spec.model_spec_hash,
            semantic_config_hash=bundle.identity.semantic_config_hash,
        )
        artifact_id = f"{receipt.run_id}.{attempt_id}.checkpoint"
        attempt_source = Path(source_path).parent
        checkpoint_uri = (attempt_source / checkpoint.uri).as_posix()
        artifact = ModelArtifact(
            artifact_id=artifact_id,
            contract_id=bundle.model_id,
            contract_version=contract_version,
            producer_run_id=receipt.run_id,
            producer_receipt=receipt_pointer,
            checkpoint=EvidencePointer(
                uri=checkpoint_uri,
                sha256=checkpoint.sha256,
                media_type=checkpoint.media_type,
            ),
            checkpoint_format="safetensors",
            checkpoint_schema_version=checkpoint_schema,
            model_state_schema_version=model_state_schema,
            model_spec=EvidencePointer(
                uri=f"specs/models/{bundle.model_id}.yaml",
                sha256=spec.model_spec_hash,
                media_type="application/yaml",
            ),
            semantic_config=EvidencePointer(
                uri=(attempt_source / "resolved-configs/semantic.json").as_posix(),
                sha256=semantic_artifact.sha256,
                media_type="application/json",
            ),
            compiler_manifest=EvidencePointer(
                uri=(attempt_source / "compiler-manifest.json").as_posix(),
                sha256=compiler_artifact.sha256,
                media_type="application/json",
            ),
            license_id=license_id,
            status_history=(
                ArtifactStatusEvent(
                    status="produced",
                    evidence=receipt_pointer,
                    note="projected from a verified formal fit receipt",
                ),
            ),
        )
        artifact_ids = (artifact_id,)

    run = RunRecord(
        run_id=receipt.run_id,
        experiment_id=experiment_id,
        status=run_status,
        status_history=(
            StatusEvent(status="planned"),
            StatusEvent(status="running"),
            StatusEvent(status=run_status.value, evidence_hashes=(receipt.receipt_hash,)),
        ),
        attempt_ids=(attempt_id,),
        receipt=receipt_pointer,
        failure_category=failure_category,
        artifact_ids=artifact_ids,
    )
    attempt = RunAttemptRecord(
        attempt_id=attempt_id,
        run_id=receipt.run_id,
        status=attempt_status,
        receipt=receipt_pointer,
        failure_category=failure_category,
    )
    entries = [
        _entry(
            kind=CatalogObjectKind.RECEIPT,
            object_id=receipt.receipt_id,
            value=receipt,
            source=raw,
            source_path=source_path,
            status=receipt.status.value,
        ),
        _entry(
            kind=CatalogObjectKind.RUN_ATTEMPT,
            object_id=attempt.attempt_id,
            value=attempt,
            source=raw,
            source_path=source_path,
            status=attempt.status.value,
        ),
    ]
    edges = [
        _edge(
            CatalogObjectKind.EXPERIMENT,
            experiment_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RUN,
            receipt.run_id,
            evidence_hash=receipt.receipt_hash,
        ),
        _edge(
            CatalogObjectKind.RUN,
            receipt.run_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RUN_ATTEMPT,
            attempt_id,
            evidence_hash=receipt.receipt_hash,
        ),
        _edge(
            CatalogObjectKind.RUN,
            receipt.run_id,
            LineageRelation.PRODUCED,
            CatalogObjectKind.RECEIPT,
            receipt.receipt_id,
            evidence_hash=receipt.receipt_hash,
        ),
        _edge(
            CatalogObjectKind.RUN,
            receipt.run_id,
            LineageRelation.USES_DATA,
            CatalogObjectKind.DATASET_SNAPSHOT,
            dataset_snapshot_id,
            evidence_hash=spec.dataset.dataset_hash,
        ),
    ]
    if artifact is not None:
        entries.append(
            _entry(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id=artifact.artifact_id,
                value=artifact,
                source=raw,
                source_path=source_path,
                status=artifact.status.value,
            )
        )
        edges.extend(
            (
                _edge(
                    CatalogObjectKind.RUN,
                    receipt.run_id,
                    LineageRelation.PRODUCED,
                    CatalogObjectKind.MODEL_ARTIFACT,
                    artifact.artifact_id,
                    evidence_hash=receipt.receipt_hash,
                ),
                _edge(
                    CatalogObjectKind.MODEL_ARTIFACT,
                    artifact.artifact_id,
                    LineageRelation.IMPLEMENTS,
                    CatalogObjectKind.MODEL_CONTRACT,
                    bundle.model_id,
                ),
            )
        )
    return _RunProjection(
        run=run,
        entries=tuple(entries),
        edges=tuple(edges),
        source=raw,
        source_path=source_path,
    )


def _eval_suite(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    suite = EvalSuiteSpec.model_validate(raw)
    return (
        [
            _entry(
                kind=CatalogObjectKind.EVAL_SUITE,
                object_id=suite.suite_id,
                value=suite,
                source=raw,
                source_path=source_path,
                status=suite.status,
            )
        ],
        [],
    )


def _eval_result(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    result = EvalResult.model_validate(raw)
    receipt = result.execution_receipt
    if receipt is None or not receipt.publication_eligible:
        raise CatalogBuildError(
            "only a result bound to its own formal evaluation receipt, independent of the "
            "immutable producer receipt, may enter the catalog"
        )
    edges: list[LineageEdge] = []
    if result.producer.provenance is ProducerProvenance.RECEIPTED_RUN:
        assert result.producer.run_id is not None
        edges.append(
            _edge(
                CatalogObjectKind.RUN,
                result.producer.run_id,
                LineageRelation.EVALUATED_BY,
                CatalogObjectKind.EVAL_RESULT,
                result.result_id,
                evidence_hash=receipt.content_hash,
            )
        )
    if result.adapter.artifact_id is not None:
        edges.append(
            _edge(
                CatalogObjectKind.MODEL_ARTIFACT,
                result.adapter.artifact_id,
                LineageRelation.EVALUATED_BY,
                CatalogObjectKind.EVAL_RESULT,
                result.result_id,
                evidence_hash=receipt.content_hash,
            )
        )
    return (
        [
            _entry(
                kind=CatalogObjectKind.EVAL_RESULT,
                object_id=result.result_id,
                value=result,
                source=raw,
                source_path=source_path,
                status=result.status.value,
            )
        ],
        edges,
    )


def _eval_comparison(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    comparison = ComparisonReport.model_validate(raw)
    if not comparison.publication_eligible:
        raise CatalogBuildError("unissued evaluation comparisons cannot enter the catalog")
    return (
        [
            _entry(
                kind=CatalogObjectKind.EVAL_COMPARISON,
                object_id=comparison.comparison_id,
                value=comparison,
                source=raw,
                source_path=source_path,
                status="publication_eligible",
            )
        ],
        [],
    )


def _verification_suite(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    from tabu_lab.verification.contracts import VerificationSuite
    from tabu_lab.verification.registry import get_check

    suite = VerificationSuite.model_validate(raw)
    for check in suite.checks:
        get_check(check.check_id)
    return (
        [
            _entry(
                kind=CatalogObjectKind.VERIFICATION_SUITE,
                object_id=suite.suite_id,
                value=suite,
                source=raw,
                source_path=source_path,
            )
        ],
        [],
    )


def _verification_result(
    raw: dict[str, Any], source_path: str
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    from tabu_lab.contracts import canonical_hash
    from tabu_lab.registry import get_model_spec
    from tabu_lab.verification.contracts import VerificationResult
    from tabu_lab.verification.registry import get_check
    from tabu_lab.verification.runner import list_suites

    result = VerificationResult.model_validate(raw)
    suites = {suite.suite_id: suite for suite in list_suites()}
    suite = suites.get(result.suite_id)
    if (
        suite is None
        or suite.suite_version != result.suite_version
        or suite.suite_hash != result.suite_hash
    ):
        raise CatalogBuildError("verification result is not bound to the exact verification suite")
    declared_checks = {item.check_id for item in suite.checks}
    for check in result.checks:
        try:
            get_check(check.check_id)
        except ValueError as exc:
            raise CatalogBuildError(str(exc)) from exc
        if check.check_id not in declared_checks:
            raise CatalogBuildError("verification result contains a check absent from its suite")
    spec = get_model_spec(result.contract_id)
    if (
        spec.contract_version != result.contract_version
        or canonical_hash(spec) != result.model_spec_hash
    ):
        raise CatalogBuildError("verification result is not bound to the exact ModelSpec")
    return (
        [
            _entry(
                kind=CatalogObjectKind.VERIFICATION_RESULT,
                object_id=result.result_id,
                value=result,
                source=raw,
                source_path=source_path,
                status=result.outcome.value,
            )
        ],
        [
            _edge(
                CatalogObjectKind.MODEL_CONTRACT,
                result.contract_id,
                LineageRelation.VERIFIED_BY,
                CatalogObjectKind.VERIFICATION_RESULT,
                result.result_id,
                evidence_hash=result.model_spec_hash,
            )
        ],
    )


def _load_source(
    raw: dict[str, Any],
    source_path: str,
    *,
    current_model_hashes: Mapping[str, str] | None = None,
) -> tuple[list[CatalogEntry], list[LineageEdge]]:
    schema = raw.get("schema_version")
    if schema == "1.0.0" and "contract_id" in raw:
        spec = ModelSpec.model_validate(raw)
        return (
            [
                _entry(
                    kind=CatalogObjectKind.MODEL_CONTRACT,
                    object_id=spec.contract_id,
                    value=spec,
                    source=raw,
                    source_path=source_path,
                    status=spec.maturity.stage.value,
                )
            ],
            [],
        )
    if schema == "tabu.fit-experiment.v1":
        return _fit_experiment(
            raw,
            source_path,
            current_model_hashes=current_model_hashes,
        )
    if schema == "tabu.verification-suite.v1":
        return _verification_suite(raw, source_path)
    if schema == "tabu.verification-result.v1":
        return _verification_result(raw, source_path)
    if schema in {
        "tabu.synthetic-prior.v1",
        "tabu.pretrain-experiment.v1",
        "tabu.finetune-experiment.v1",
        "tabu.transfer-comparison.v1",
        "tabu.transfer-panel-manifest.v1",
        "tabu.transfer-split-manifest.v1",
        "tabu.icl-harness.v1",
        "tabu.transfer-base-pretrain.v2",
        "tabu.transfer-base-icl.v2",
        "tabu.transfer-base-finetune.v2",
    }:
        # Transfer protocol specs are validated by ``experiments validate`` and
        # remain outside the evidence catalog until a receipt-backed adapter
        # emits a catalog-native experiment/run record.
        return [], []
    if schema == "tabu-lab.preregistration.v1":
        return _legacy_preregistration(raw, source_path)
    if schema == "tabu.catalog-experiment.v1":
        return _explicit_experiment(raw, source_path)
    if schema == "tabu.catalog-run.v1":
        return _run_record(raw, source_path)
    if schema == "tabu.catalog-run-attempt.v1":
        return _attempt_record(raw, source_path)
    if schema == "tabu.model-artifact.v2":
        return _model_artifact(raw, source_path)
    if schema in {"tabu.dataset-snapshot.v2", "tabu.dataset-snapshot.v3"}:
        snapshot = DatasetSnapshotSpec.model_validate(raw)
        return (
            [
                _entry(
                    kind=CatalogObjectKind.DATASET_SNAPSHOT,
                    object_id=snapshot.dataset_snapshot_id,
                    value=snapshot,
                    source=raw,
                    source_path=source_path,
                )
            ],
            [],
        )
    if schema == "tabu.review.v1":
        return _review_record(raw, source_path)
    if schema == "tabu.catalog-claim.v1":
        return _claim_record(raw, source_path)
    if schema == "tabu.claim-ledger.v1":
        return _claim_ledger(raw, source_path)
    if schema == "tabu.receipt.v1":
        return _receipt(raw, source_path)
    if schema == "tabu.eval-suite.v1":
        return _eval_suite(raw, source_path)
    if schema == "tabu.eval-result.v2":
        return _eval_result(raw, source_path)
    if schema == "tabu.eval-comparison.v2":
        return _eval_comparison(raw, source_path)
    if schema == "tabu.receipt-envelope.v1":
        raise CatalogBuildError("receipt envelopes require attempt-directory verification")
    if not isinstance(schema, str) or not schema:
        raise CatalogBuildError(f"catalog source lacks schema_version: {source_path}")
    raise CatalogBuildError(f"unsupported catalog source schema_version {schema!r}: {source_path}")


def _deduplicate_entries(entries: Iterable[CatalogEntry]) -> tuple[CatalogEntry, ...]:
    by_id: dict[str, CatalogEntry] = {}
    preregistration_names = {
        "preregistration.json",
        "preregistration.yaml",
        "preregistration.yml",
    }
    for entry in entries:
        previous = by_id.get(entry.object_id)
        if previous is None:
            by_id[entry.object_id] = entry
            continue
        if previous == entry:
            continue
        if (
            previous.kind is CatalogObjectKind.DATASET_SNAPSHOT
            and entry.kind is CatalogObjectKind.DATASET_SNAPSHOT
            and Path(previous.source_path).name in preregistration_names
            and Path(entry.source_path).name in preregistration_names
            and previous.model_copy(update={"source_path": entry.source_path}) == entry
        ):
            # Fit preregistrations may deliberately share one immutable dataset
            # and split.  They project the same DatasetSnapshot from different
            # source files; retain one deterministic source pointer while every
            # experiment keeps its own USES_DATA edge.  Only byte-equivalent
            # derived objects are coalesced.  Any divergent duplicate still
            # fails below.
            by_id[entry.object_id] = min(
                (previous, entry),
                key=lambda candidate: candidate.source_path,
            )
            continue
        if (
            previous.kind is CatalogObjectKind.MODEL_CONTRACT
            and entry.kind is CatalogObjectKind.MODEL_CONTRACT
            and previous.object_hash == entry.object_hash
            and previous.model_copy(update={"source_path": entry.source_path}) == entry
        ):
            # Historical ModelSpec aliases can be embedded by multiple
            # preregistrations.  They are one immutable identity; retain the
            # lexicographically earliest source pointer without collapsing
            # them into the newer bare contract id.
            by_id[entry.object_id] = min(
                (previous, entry),
                key=lambda candidate: candidate.source_path,
            )
            continue
        if (
            previous.kind is CatalogObjectKind.EXPERIMENT
            and entry.kind is CatalogObjectKind.EXPERIMENT
        ):
            previous_is_prereg = Path(previous.source_path).name in preregistration_names
            entry_is_prereg = Path(entry.source_path).name in preregistration_names
            if previous_is_prereg != entry_is_prereg:
                draft_entry = previous if previous_is_prereg else entry
                explicit_entry = entry if previous_is_prereg else previous
                draft = ExperimentRecord.model_validate(draft_entry.data)
                explicit = ExperimentRecord.model_validate(explicit_entry.data)
                if (
                    draft.preregistration is None
                    or explicit.preregistration is None
                    or draft.preregistration.sha256 != explicit.preregistration.sha256
                    or explicit.preregistration.uri != draft_entry.source_path
                    or explicit.supersedes_experiment_ids != draft.supersedes_experiment_ids
                    or explicit.revision_rationale != draft.revision_rationale
                ):
                    raise CatalogBuildError(
                        f"experiment record {entry.object_id!r} does not bind its canonical "
                        "preregistration source, hash, and revision lineage"
                    )
                by_id[entry.object_id] = explicit_entry
                continue
        raise CatalogBuildError(
            f"duplicate catalog object id {entry.object_id!r}: "
            f"{previous.source_path} and {entry.source_path}"
        )
    return tuple(sorted(by_id.values(), key=lambda item: (item.kind.value, item.object_id)))


def _deduplicate_edges(edges: Iterable[LineageEdge]) -> tuple[LineageEdge, ...]:
    by_id = {edge.edge_id: edge for edge in edges}
    return tuple(sorted(by_id.values(), key=lambda item: item.edge_id))


def _bind_projected_artifact_evaluations(
    entries: tuple[CatalogEntry, ...],
) -> tuple[CatalogEntry, ...]:
    """Attach discovered result ids to receipt-projected artifacts.

    A receipt projection is intentionally immutable with respect to checkpoint
    identity, but evaluation results arrive later as separate canonical
    sources.  Their reverse references are therefore a deterministic catalog
    projection.  Explicit artifact manifests must carry their own exact refs
    and are validated without mutation.
    """

    results_by_artifact: dict[str, set[str]] = {}
    for entry in entries:
        if entry.kind is not CatalogObjectKind.EVAL_RESULT:
            continue
        result = EvalResult.model_validate(entry.data)
        if result.adapter.artifact_id is not None:
            results_by_artifact.setdefault(result.adapter.artifact_id, set()).add(result.result_id)

    resolved: list[CatalogEntry] = []
    for entry in entries:
        if (
            entry.kind is not CatalogObjectKind.MODEL_ARTIFACT
            or Path(entry.source_path).name != "receipt.json"
        ):
            resolved.append(entry)
            continue
        artifact = ModelArtifact.model_validate(entry.data)
        result_ids = tuple(sorted(results_by_artifact.get(artifact.artifact_id, set())))
        if artifact.evaluation_result_ids == result_ids:
            resolved.append(entry)
            continue
        projected = artifact.model_copy(update={"evaluation_result_ids": result_ids})
        data = _json_mapping(projected)
        resolved.append(
            entry.model_copy(
                update={
                    "object_hash": canonical_hash(data),
                    "data": data,
                }
            )
        )
    return tuple(sorted(resolved, key=lambda item: (item.kind.value, item.object_id)))


def _evaluation_dataset_edges(entries: tuple[CatalogEntry, ...]) -> tuple[LineageEdge, ...]:
    """Derive exact result-to-snapshot lineage when all frozen hashes agree."""

    suites = {
        entry.object_id: EvalSuiteSpec.model_validate(entry.data)
        for entry in entries
        if entry.kind is CatalogObjectKind.EVAL_SUITE
    }
    snapshots = tuple(
        DatasetSnapshotSpec.model_validate(entry.data)
        for entry in entries
        if entry.kind is CatalogObjectKind.DATASET_SNAPSHOT
    )
    derived: list[LineageEdge] = []
    for entry in entries:
        if entry.kind is not CatalogObjectKind.EVAL_RESULT:
            continue
        result = EvalResult.model_validate(entry.data)
        suite = suites.get(result.suite_id)
        if suite is None:
            continue
        scenarios = tuple(
            scenario for scenario in suite.scenarios if scenario.scenario_id == result.scenario_id
        )
        if len(scenarios) != 1:
            continue
        scenario = scenarios[0]
        matches = tuple(
            snapshot
            for snapshot in snapshots
            if snapshot.dataset_id == scenario.dataset.dataset_id
            and snapshot.source_uri == scenario.dataset.source_uri
            and snapshot.license_id == scenario.dataset.license_id
            and snapshot.source_sha256 == result.source_sha256
            and snapshot.split_manifest_sha256 == result.split_sha256
            and result.recipe_sha256 in snapshot.episode_recipe_hashes
            and snapshot.evaluation_scenario_id == result.scenario_id
            and snapshot.truth_sidecar_sha256 == result.truth_sidecar_sha256
        )
        if len(matches) == 1:
            snapshot = matches[0]
            derived.append(
                _edge(
                    CatalogObjectKind.EVAL_RESULT,
                    result.result_id,
                    LineageRelation.USES_DATA,
                    CatalogObjectKind.DATASET_SNAPSHOT,
                    snapshot.dataset_snapshot_id,
                    evidence_hash=snapshot.content_sha256,
                )
            )
    return tuple(derived)


def _source_tree_hash(entries: tuple[CatalogEntry, ...], edges: tuple[LineageEdge, ...]) -> str:
    return canonical_hash(
        {
            "schema": "tabu.catalog-source-tree.v1",
            "sources": [
                {
                    "kind": entry.kind.value,
                    "object_id": entry.object_id,
                    "source_hash": entry.source_hash,
                    "source_path": entry.source_path,
                }
                for entry in entries
            ],
            "lineage": [edge.model_dump(mode="json") for edge in edges],
        }
    )


def build_catalog(
    repo_root: str | Path,
    output_path: str | Path | None = None,
    *,
    source_revision: CatalogSourceRevision | None = None,
    authorization_replay: FormalAuthorizationReplaySession | None = None,
) -> CatalogIndex:
    """Discover canonical manifests and optionally write a deterministic index."""

    root = Path(repo_root).resolve()
    if not root.is_dir():
        raise CatalogBuildError(f"catalog repository root does not exist: {root}")

    entries: list[CatalogEntry] = []
    edges: list[LineageEdge] = []
    receipt_projections: dict[str, list[_RunProjection]] = {}
    replay = authorization_replay
    current_model_hashes: dict[str, str] = {}
    model_directory = root / "specs" / "models"
    if model_directory.is_dir():
        for model_path in sorted(model_directory.iterdir()):
            if not model_path.is_file() or model_path.suffix.lower() not in _SOURCE_SUFFIXES:
                continue
            model_raw = _read_mapping(model_path)
            if model_raw.get("schema_version") != "1.0.0":
                continue
            model_spec = ModelSpec.model_validate(model_raw)
            current_model_hashes[model_spec.contract_id] = canonical_hash(model_spec)
    for path in _source_paths(root):
        relative = _relative(path, root)
        try:
            raw = _read_mapping(path)
            if raw.get("schema_version") == "tabu.receipt-envelope.v1":
                receipt_metadata = raw.get("receipt", {}).get("metadata", {})
                if receipt_metadata.get("issuance_status") == "formal" and replay is None:
                    from tabu_lab.evidence.formal_authorization import (
                        FormalAuthorizationError,
                        FormalAuthorizationReplaySession,
                    )

                    try:
                        replay = FormalAuthorizationReplaySession(root)
                    except FormalAuthorizationError as exc:
                        raise CatalogBuildError(
                            "formal receipt catalog requires an independent canonical Git root"
                        ) from exc
                projection = _fit_receipt_projection(
                    raw=raw,
                    source_path=relative,
                    path=path,
                    authorization_replay=replay,
                )
                receipt_projections.setdefault(projection.run.run_id, []).append(projection)
                loaded_entries = list(projection.entries)
                loaded_edges = list(projection.edges)
            else:
                loaded_entries, loaded_edges = _load_source(
                    raw,
                    relative,
                    current_model_hashes=current_model_hashes,
                )
        except (ValidationError, ValueError) as exc:
            if isinstance(exc, CatalogBuildError):
                raise
            raise CatalogBuildError(f"invalid catalog source {relative}: {exc}") from exc
        entries.extend(loaded_entries)
        edges.extend(loaded_edges)

    # Formal replay must observe the same compatible experiment coalescing as
    # the final catalog.  Fit preregistrations and their explicit
    # ExperimentRecord intentionally project the same experiment id; replaying
    # the raw list would falsely treat that legal pair as an ambiguous source
    # authority.
    replay_entries = _deduplicate_entries(entries)
    formal_eval_results = tuple(
        EvalResult.model_validate(entry.data)
        for entry in replay_entries
        if entry.kind is CatalogObjectKind.EVAL_RESULT
        and (entry.data.get("execution_receipt") or {}).get("issuance_status") == "formal"
    )
    if formal_eval_results:
        if replay is None:
            from tabu_lab.evidence.formal_authorization import (
                FormalAuthorizationError,
                FormalAuthorizationReplaySession,
            )

            try:
                replay = FormalAuthorizationReplaySession(root)
            except FormalAuthorizationError as exc:
                raise CatalogBuildError(
                    "formal evaluation catalog requires an independent canonical Git root"
                ) from exc
        from tabu_lab.evaluation.formal_receipt import (
            FormalEvaluationReceiptError,
            replay_formal_evaluation_source_authorization,
        )

        for result in formal_eval_results:
            try:
                replay_formal_evaluation_source_authorization(
                    result,
                    repository=root,
                    entries=replay_entries,
                    replay=replay,
                )
            except (FormalEvaluationReceiptError, FormalAuthorizationError) as exc:
                raise CatalogBuildError(
                    "formal evaluation receipt failed evaluator-source authorization replay"
                ) from exc

    explicit_runs = {
        entry.object_id: RunRecord.model_validate(entry.data)
        for entry in entries
        if entry.kind is CatalogObjectKind.RUN
    }
    for run_id, projections in sorted(receipt_projections.items()):
        discovered_attempts = {
            RunAttemptRecord.model_validate(entry.data).attempt_id
            for projection in projections
            for entry in projection.entries
            if entry.kind is CatalogObjectKind.RUN_ATTEMPT
        }
        discovered_artifacts = {
            ModelArtifact.model_validate(entry.data).artifact_id
            for projection in projections
            for entry in projection.entries
            if entry.kind is CatalogObjectKind.MODEL_ARTIFACT
        }
        explicit = explicit_runs.get(run_id)
        if explicit is not None:
            if set(explicit.attempt_ids) != discovered_attempts:
                raise CatalogBuildError(
                    f"explicit run {run_id!r} does not list its exact discovered attempts"
                )
            if set(explicit.artifact_ids) != discovered_artifacts:
                raise CatalogBuildError(
                    f"explicit run {run_id!r} does not list its exact projected artifacts"
                )
            continue
        if len(projections) != 1:
            raise CatalogBuildError(
                f"run {run_id!r} has multiple attempt receipts; add an explicit RunRecord "
                "that selects the terminal receipt and records the complete attempt lineage"
            )
        projection = projections[0]
        entries.append(
            _entry(
                kind=CatalogObjectKind.RUN,
                object_id=run_id,
                value=projection.run,
                source=projection.source,
                source_path=projection.source_path,
                status=projection.run.status.value,
            )
        )

    stable_entries = _bind_projected_artifact_evaluations(_deduplicate_entries(entries))
    stable_edges = _deduplicate_edges((*edges, *_evaluation_dataset_edges(stable_entries)))
    try:
        catalog = CatalogIndex(
            source_tree_hash=_source_tree_hash(stable_entries, stable_edges),
            source_revision=source_revision,
            entries=stable_entries,
            lineage=stable_edges,
        )
    except ValidationError as exc:
        raise CatalogBuildError(f"catalog graph validation failed: {exc}") from exc

    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(canonical_json(catalog) + "\n", encoding="utf-8")
    return catalog


def load_catalog(path: str | Path) -> CatalogIndex:
    """Load and revalidate a generated catalog, including hashes and lineage."""

    source = Path(path)
    try:
        return CatalogIndex.model_validate_json(source.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise CatalogBuildError(f"invalid generated catalog {source}: {exc}") from exc


def check_catalog(
    repo_root: str | Path,
    catalog_path: str | Path | None = None,
    *,
    authorization_replay: FormalAuthorizationReplaySession | None = None,
) -> CatalogCheckReport:
    """Validate sources and, when present, fail on generated-index hash drift."""

    root = Path(repo_root).resolve()
    try:
        rebuilt = build_catalog(root, authorization_replay=authorization_replay)
        candidate_path: Path | None
        if catalog_path is not None:
            candidate_path = Path(catalog_path)
            if not candidate_path.is_absolute():
                candidate_path = root / candidate_path
        else:
            default = root / "catalog.json"
            candidate_path = default if default.is_file() else None

        if candidate_path is not None:
            if not candidate_path.is_file():
                raise CatalogBuildError(f"catalog file does not exist: {candidate_path}")
            checked_in = load_catalog(candidate_path)
            if checked_in.source_revision is not None:
                rebuilt = build_catalog(
                    root,
                    source_revision=checked_in.source_revision,
                    authorization_replay=authorization_replay,
                )
            if checked_in != rebuilt:
                raise CatalogBuildError(
                    "catalog hash drift: generated catalog does not match canonical sources"
                )
        return CatalogCheckReport(ok=True, catalog_hash=rebuilt.catalog_hash)
    except (CatalogBuildError, ValidationError, ValueError) as exc:
        public_message = str(exc).replace(root.as_posix(), "<repo>")
        return CatalogCheckReport(
            ok=False,
            issues=(CatalogCheckIssue(code="catalog_invalid", message=public_message),),
        )


def collections(catalog: CatalogIndex) -> dict[str, tuple[CatalogEntry, ...]]:
    """Stable functional adapter for CLI and projection consumers."""

    return catalog.collections()


def show(catalog: CatalogIndex, object_id: str) -> CatalogEntry:
    """Return one globally unique catalog entry."""

    return catalog.show(object_id)


__all__ = [
    "CatalogBuildError",
    "build_catalog",
    "check_catalog",
    "collections",
    "load_catalog",
    "show",
]
