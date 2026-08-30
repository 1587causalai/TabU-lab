"""Offline, create-once registration workflow for Evaluation v0 data.

The workflow deliberately separates three evidence boundaries:

* a caller-authored authority request that declares the exact retained bytes,
  raw format, exhaustive split, and any mask or topology choices;
* a private prepared bundle that retains source bytes and evaluator truth; and
* a public ``DatasetSnapshotSpec`` containing only content-addressed metadata.

No function in this module downloads data.  Registration can only consume a
self-verifying private bundle, and checking never writes state.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from tabu_lab.catalog import DatasetSnapshotSpec
from tabu_lab.contracts import canonical_hash, require_sha256, to_canonical_data
from tabu_lab.evaluation.foundry import EvalSuiteSpec, PreparedScenario, TaskKind, load_suite
from tabu_lab.evidence.schemas import EvidenceSchema

from .eval_snapshot import dataset_snapshot_from_prepared, write_dataset_snapshot_manifest
from .real_eval_data import (
    CompletionMaskAuthority,
    DelimitedTableAuthority,
    KarateAuthority,
    MovieLensAuthority,
    RealEvalDataError,
    materialize_karate,
    materialize_movielens,
    materialize_table_completion,
    materialize_table_supervised,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
AuthorityManifest = Annotated[
    DelimitedTableAuthority | KarateAuthority | MovieLensAuthority,
    Field(discriminator="schema_version"),
]
SourceMediaType = Literal[
    "application/json",
    "application/zip",
    "text/csv",
    "text/tab-separated-values",
]

_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_MAX_BUNDLE_BYTES = 192 * 1024 * 1024
_SCENARIO_CONTRACTS: dict[str, tuple[str, TaskKind, type[EvidenceSchema]]] = {
    "adult-v2-task-7592-classification-micro": (
        "table-supervised-micro-v0",
        TaskKind.SUPERVISED_CLASSIFICATION,
        DelimitedTableAuthority,
    ),
    "adult-v2-task-7592-classification-micro-base": (
        "table-supervised-micro-v1",
        TaskKind.SUPERVISED_CLASSIFICATION,
        DelimitedTableAuthority,
    ),
    "sklearn-diabetes-regression-micro": (
        "table-supervised-micro-v0",
        TaskKind.SUPERVISED_REGRESSION,
        DelimitedTableAuthority,
    ),
    "sklearn-diabetes-regression-micro-base": (
        "table-supervised-micro-v1",
        TaskKind.SUPERVISED_REGRESSION,
        DelimitedTableAuthority,
    ),
    "adult-v2-feature-completion-micro": (
        "table-completion-micro-v0",
        TaskKind.TABLE_COMPLETION,
        DelimitedTableAuthority,
    ),
    "adult-v2-feature-completion-micro-base": (
        "table-completion-micro-v1",
        TaskKind.TABLE_COMPLETION,
        DelimitedTableAuthority,
    ),
    "sklearn-diabetes-feature-completion-micro": (
        "table-completion-micro-v0",
        TaskKind.TABLE_COMPLETION,
        DelimitedTableAuthority,
    ),
    "sklearn-diabetes-feature-completion-micro-base": (
        "table-completion-micro-v1",
        TaskKind.TABLE_COMPLETION,
        DelimitedTableAuthority,
    ),
    "zachary-karate-club-label-completion": (
        "graph-completion-micro-v0",
        TaskKind.GRAPH_COMPLETION,
        KarateAuthority,
    ),
    "movielens-100k-interaction-completion": (
        "recsys-completion-micro-v0",
        TaskKind.RECSYS_COMPLETION,
        MovieLensAuthority,
    ),
}


class EvalDataWorkflowError(ValueError):
    """An offline evaluation-data request or bundle is not self-consistent."""


def _authority_source_sha256(authority: AuthorityManifest) -> str:
    if isinstance(authority, MovieLensAuthority):
        return authority.source_sha256
    return authority.split.source_sha256


def _authority_dataset_id(authority: AuthorityManifest) -> str:
    if isinstance(authority, MovieLensAuthority):
        return authority.dataset_id
    return authority.split.dataset_id


def _authority_source_version(authority: AuthorityManifest) -> str:
    if isinstance(authority, MovieLensAuthority):
        return authority.source_version
    return authority.split.source_version


def _expected_media_type(authority: AuthorityManifest) -> SourceMediaType:
    if isinstance(authority, DelimitedTableAuthority):
        return "text/csv" if authority.delimiter == "," else "text/tab-separated-values"
    if isinstance(authority, KarateAuthority):
        return "application/json"
    return "application/zip"


def _authority_envelope_hash(
    *,
    authority: AuthorityManifest,
    completion_mask_authority: CompletionMaskAuthority | None,
) -> str:
    return canonical_hash(
        {
            "schema_version": "tabu.eval-data-authority-envelope.v1",
            "authority": authority,
            "completion_mask_authority": completion_mask_authority,
        }
    )


class EvalDataPreparationRequest(EvidenceSchema):
    """Strict authority/source manifest for one frozen Evaluation v0 scenario."""

    schema_version: Literal["tabu.eval-data-preparation-request.v1"] = (
        "tabu.eval-data-preparation-request.v1"
    )
    suite_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    scenario_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    network_access: Literal[False] = False
    source_sha256: Sha256
    source_size_bytes: int = Field(gt=0, le=_MAX_SOURCE_BYTES)
    source_media_type: SourceMediaType
    authority: AuthorityManifest
    completion_mask_authority: CompletionMaskAuthority | None = None

    @field_validator("source_sha256")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source_sha256")

    @model_validator(mode="after")
    def _request_is_closed(self) -> EvalDataPreparationRequest:
        expected = _SCENARIO_CONTRACTS.get(self.scenario_id)
        if expected is None:
            raise ValueError(
                "preparation request scenario is not one of the frozen evaluation scenarios"
            )
        suite_id, task, authority_type = expected
        if self.suite_id != suite_id:
            raise ValueError("preparation request scenario is bound to another frozen suite")
        if type(self.authority) is not authority_type:
            raise ValueError("preparation request uses the wrong authority schema")
        if _authority_source_sha256(self.authority) != self.source_sha256:
            raise ValueError("request source_sha256 differs from its authority manifest")
        if self.source_media_type != _expected_media_type(self.authority):
            raise ValueError("request media type differs from its authority format")
        if task is TaskKind.TABLE_COMPLETION:
            if self.completion_mask_authority is None:
                raise ValueError("table completion request requires a mask authority")
        elif self.completion_mask_authority is not None:
            raise ValueError("mask authority is only valid for table completion")
        return self

    @property
    def authority_sha256(self) -> str:
        return _authority_envelope_hash(
            authority=self.authority,
            completion_mask_authority=self.completion_mask_authority,
        )


class PreparedEvalDataBundle(EvidenceSchema):
    """Private source-and-truth retaining bundle; never a public catalog object."""

    schema_version: Literal["tabu.eval-data-prepared-bundle.v1"] = (
        "tabu.eval-data-prepared-bundle.v1"
    )
    visibility: Literal["private_evaluator_input"] = "private_evaluator_input"
    publication_eligible: Literal[False] = False
    request: EvalDataPreparationRequest
    request_sha256: Sha256
    authority_sha256: Sha256
    suite_sha256: Sha256
    source_sha256: Sha256
    source_size_bytes: int = Field(gt=0, le=_MAX_SOURCE_BYTES)
    prepared_sha256: Sha256
    prepared: PreparedScenario

    @field_validator(
        "request_sha256",
        "authority_sha256",
        "suite_sha256",
        "source_sha256",
        "prepared_sha256",
    )
    @classmethod
    def _valid_hash(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @model_validator(mode="after")
    def _bundle_is_self_verifying(self) -> PreparedEvalDataBundle:
        if self.request_sha256 != self.request.content_hash:
            raise ValueError("request_sha256 does not bind the preparation request")
        if self.authority_sha256 != self.request.authority_sha256:
            raise ValueError("authority_sha256 does not bind the authority envelope")
        if self.source_sha256 != self.request.source_sha256:
            raise ValueError("bundle source_sha256 differs from its request")
        if self.prepared.scenario_id != self.request.scenario_id:
            raise ValueError("prepared scenario differs from its request")
        if self.prepared.binding.dataset_id != _authority_dataset_id(self.request.authority):
            raise ValueError("prepared dataset differs from its authority")
        if self.prepared.binding.source_sha256 != self.source_sha256:
            raise ValueError("prepared source hash differs from its request")
        if self.prepared.source_material.media_type != self.request.source_media_type:
            raise ValueError("prepared source media type differs from its request")
        if len(self.prepared.source_material.content_bytes) != self.source_size_bytes:
            raise ValueError("source_size_bytes does not bind the retained source bytes")
        if self.source_size_bytes != self.request.source_size_bytes:
            raise ValueError("bundle source size differs from its request")
        if self.prepared.content_hash != self.prepared_sha256:
            raise ValueError("prepared_sha256 does not bind the PreparedScenario")
        preprocessing = self.prepared.preparation.preprocessing
        if preprocessing.get("source_authority_sha256") != self.request.authority.content_hash:
            raise ValueError("prepared recipe does not bind the source authority")
        mask_authority = self.request.completion_mask_authority
        execution = preprocessing.get("execution")
        if mask_authority is not None:
            if not isinstance(execution, Mapping):
                raise ValueError("prepared completion recipe lacks mask execution")
            if execution.get("mask_authority_sha256") != mask_authority.content_hash:
                raise ValueError("prepared recipe does not bind the mask authority")
        return self


class EvalDataCheckReport(EvidenceSchema):
    """Read-only validation report for a private bundle and optional snapshot."""

    schema_version: Literal["tabu.eval-data-check-report.v1"] = "tabu.eval-data-check-report.v1"
    valid: Literal[True] = True
    suite_id: str
    suite_sha256: Sha256
    scenario_id: str
    request_sha256: Sha256
    authority_sha256: Sha256
    source_sha256: Sha256
    prepared_sha256: Sha256
    dataset_snapshot_id: str
    dataset_snapshot_sha256: Sha256
    snapshot_checked: bool
    snapshot_matches: bool | None = None


def _load_mapping(path: str | os.PathLike[str], *, size_limit: int) -> Mapping[str, object]:
    source = Path(path)
    if not source.is_file():
        raise EvalDataWorkflowError(f"manifest must name a regular local file: {source}")
    if source.stat().st_size > size_limit:
        raise EvalDataWorkflowError(f"manifest exceeds the offline size limit: {source}")
    text = source.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise EvalDataWorkflowError(f"manifest is not valid YAML or JSON: {source}") from error
    if not isinstance(payload, Mapping):
        raise EvalDataWorkflowError(f"manifest must contain a mapping: {source}")
    return payload


def _nearest_existing_directory(path: Path) -> Path:
    """Return an existing directory from which Git can classify ``path``."""

    candidate = path.resolve(strict=False)
    if candidate.exists() and not candidate.is_dir():
        candidate = candidate.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate if candidate.is_dir() else candidate.parent


def _git_worktree_root(path: Path) -> Path | None:
    """Return the containing Git worktree root without mutating repository state."""

    directory = _nearest_existing_directory(path)
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(directory), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without resolving any symlink component."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_no_repository_symlink(path: Path, *, role: str) -> None:
    """Reject private evidence paths that use a symlink as an escape hatch.

    Git classifies the resolved target of a symlink differently from the path
    the caller supplied.  Without this lexical check, an unignored link in the
    repository could point either outside the checkout or at an ignored target
    and bypass the private-input gate.  Inspect components with ``lstat`` so no
    target bytes are read while deciding.
    """

    lexical = _lexical_absolute(path)
    try:
        if stat.S_ISLNK(lexical.lstat().st_mode):
            raise EvalDataWorkflowError(f"{role} must not be a symlink")
    except FileNotFoundError:
        pass
    except OSError as error:
        raise EvalDataWorkflowError(f"cannot verify {role} symlink safety") from error

    for root in _active_repository_roots():
        if not _is_within(lexical, root):
            continue
        current = root
        for component in lexical.relative_to(root).parts:
            current /= component
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                break
            except OSError as error:
                raise EvalDataWorkflowError(
                    f"cannot verify {role} symlink safety"
                ) from error
            if stat.S_ISLNK(metadata.st_mode):
                raise EvalDataWorkflowError(
                    f"{role} must not traverse a symlink inside an active Git worktree"
                )


def _git_path_is_ignored(path: Path, *, worktree_root: Path) -> bool:
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(worktree_root)
    except ValueError as error:  # pragma: no cover - callers bind the containing root
        raise EvalDataWorkflowError("Git-ignore check escaped its worktree") from error
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(worktree_root),
                "check-ignore",
                "--quiet",
                "--",
                relative.as_posix(),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise EvalDataWorkflowError("cannot verify Git-ignore safety") from error
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise EvalDataWorkflowError("cannot verify Git-ignore safety")


def _active_repository_roots() -> tuple[Path, ...]:
    """Identify the caller checkout and this editable source checkout, if present."""

    roots = {
        root
        for candidate in (Path.cwd(), Path(__file__).resolve())
        if (root := _git_worktree_root(candidate)) is not None
    }
    return tuple(sorted(roots, key=lambda item: item.as_posix()))


def _require_retained_source_safe(path: Path) -> None:
    """Prevent raw evaluator input from becoming an unignored repo file."""

    _require_no_repository_symlink(path, role="retained source")
    resolved = path.resolve(strict=False)
    for root in _active_repository_roots():
        if _is_within(resolved, root) and not _git_path_is_ignored(
            resolved,
            worktree_root=root,
        ):
            raise EvalDataWorkflowError(
                "retained source inside the active Git worktree must be Git-ignored"
            )


def _require_private_output_safe(path: Path) -> None:
    """Require private source-and-truth bundles to be ignored in any worktree."""

    _require_no_repository_symlink(path, role="private prepared bundle")
    resolved = path.resolve(strict=False)
    root = _git_worktree_root(resolved)
    if root is not None and not _git_path_is_ignored(resolved, worktree_root=root):
        raise EvalDataWorkflowError(
            "private prepared bundle inside a Git worktree must be Git-ignored"
        )


def load_eval_data_request(path: str | os.PathLike[str]) -> EvalDataPreparationRequest:
    """Load one strict YAML/JSON authority request."""

    payload = _load_mapping(path, size_limit=16 * 1024 * 1024)
    return EvalDataPreparationRequest.model_validate(payload)


def load_prepared_eval_bundle(path: str | os.PathLike[str]) -> PreparedEvalDataBundle:
    """Load and fully self-verify one private prepared bundle."""

    payload = _load_mapping(path, size_limit=_MAX_BUNDLE_BYTES)
    return PreparedEvalDataBundle.model_validate(payload)


def _load_bound_suite(
    request: EvalDataPreparationRequest,
    *,
    suite_directory: Path | None,
) -> tuple[EvalSuiteSpec, object]:
    suite = load_suite(request.suite_id, directory=suite_directory)
    scenarios = tuple(item for item in suite.scenarios if item.scenario_id == request.scenario_id)
    if len(scenarios) != 1:
        raise EvalDataWorkflowError(
            "frozen suite does not contain the requested scenario exactly once"
        )
    scenario = scenarios[0]
    expected_suite, expected_task, _ = _SCENARIO_CONTRACTS[request.scenario_id]
    if suite.suite_id != expected_suite or scenario.task is not expected_task:
        raise EvalDataWorkflowError("live suite task identity drifted from the v0 workflow")
    if scenario.dataset.dataset_id != _authority_dataset_id(request.authority):
        raise EvalDataWorkflowError("authority dataset differs from the frozen scenario")
    if scenario.dataset.source_version != _authority_source_version(request.authority):
        raise EvalDataWorkflowError("authority source version differs from the frozen scenario")
    return suite, scenario


def _read_retained_source(
    source: str | os.PathLike[str],
    *,
    request: EvalDataPreparationRequest,
) -> bytes:
    raw = os.fspath(source)
    if "://" in raw:
        raise EvalDataWorkflowError(
            "source must be a local retained file; network URLs are forbidden"
        )
    path = Path(raw)
    _require_retained_source_safe(path)
    try:
        metadata = path.lstat()
    except OSError as error:
        raise EvalDataWorkflowError("source must be a regular local retained file") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise EvalDataWorkflowError("source must be a regular local retained file")
    if metadata.st_size > _MAX_SOURCE_BYTES:
        raise EvalDataWorkflowError("retained source exceeds the offline size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvalDataWorkflowError("source must be a regular local retained file") from error
    with os.fdopen(descriptor, "rb") as stream:
        opened_metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise EvalDataWorkflowError("source must be a regular local retained file")
        if opened_metadata.st_size > _MAX_SOURCE_BYTES:
            raise EvalDataWorkflowError("retained source exceeds the offline size limit")
        content = stream.read(_MAX_SOURCE_BYTES + 1)
    if len(content) > _MAX_SOURCE_BYTES:
        raise EvalDataWorkflowError("retained source exceeds the offline size limit")
    if not content:
        raise EvalDataWorkflowError("retained source bytes cannot be empty")
    digest = hashlib.sha256(content).hexdigest()
    if digest != request.source_sha256:
        raise EvalDataWorkflowError("source_sha256 does not bind the retained source bytes")
    if len(content) != request.source_size_bytes:
        raise EvalDataWorkflowError("source_size_bytes does not bind the retained source bytes")
    return content


def prepare_eval_data_bundle(
    *,
    request: EvalDataPreparationRequest,
    source: str | os.PathLike[str],
    suite_directory: Path | None = None,
) -> PreparedEvalDataBundle:
    """Materialize one frozen scenario from exact caller-retained local bytes."""

    request = EvalDataPreparationRequest.model_validate(request.model_dump(mode="python"))
    suite, scenario = _load_bound_suite(request, suite_directory=suite_directory)
    content = _read_retained_source(source, request=request)
    authority = request.authority
    if isinstance(authority, DelimitedTableAuthority):
        if scenario.task is TaskKind.TABLE_COMPLETION:
            mask_authority = request.completion_mask_authority
            if mask_authority is None:  # guarded by the request schema; keeps dispatch fail-closed
                raise EvalDataWorkflowError("table completion request lacks mask authority")
            prepared = materialize_table_completion(
                scenario=scenario,
                source=content,
                authority=authority,
                mask_authority=mask_authority,
            )
        else:
            prepared = materialize_table_supervised(
                scenario=scenario,
                source=content,
                authority=authority,
            )
    elif isinstance(authority, KarateAuthority):
        prepared = materialize_karate(
            scenario=scenario,
            source=content,
            authority=authority,
        )
    elif isinstance(authority, MovieLensAuthority):
        prepared = materialize_movielens(
            scenario=scenario,
            source=content,
            authority=authority,
        )
    else:  # pragma: no cover - the discriminated schema excludes this branch
        raise EvalDataWorkflowError("unsupported Evaluation v0 authority schema")
    return PreparedEvalDataBundle(
        request=request,
        request_sha256=request.content_hash,
        authority_sha256=request.authority_sha256,
        suite_sha256=suite.suite_hash,
        source_sha256=request.source_sha256,
        source_size_bytes=len(content),
        prepared_sha256=prepared.content_hash,
        prepared=prepared,
    )


def _canonical_json(value: object) -> str:
    return (
        json.dumps(
            to_canonical_data(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_create_once(value: object, destination: str | os.PathLike[str]) -> Path:
    target = Path(destination)
    canonical = _canonical_json(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or target.read_text(encoding="utf-8") != canonical:
            raise FileExistsError(f"output already exists with different content: {target}")
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical)
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            if not target.is_file() or target.read_text(encoding="utf-8") != canonical:
                raise FileExistsError(
                    f"output already exists with different content: {target}"
                ) from error
        return target
    finally:
        temporary.unlink(missing_ok=True)


def write_prepared_eval_bundle(
    bundle: PreparedEvalDataBundle,
    destination: str | os.PathLike[str],
) -> Path:
    """Write a private bundle deterministically without overwriting evidence."""

    _require_private_output_safe(Path(destination))
    verified = PreparedEvalDataBundle.model_validate(bundle.model_dump(mode="python"))
    return _write_create_once(verified, destination)


def validate_prepared_eval_bundle(
    bundle: PreparedEvalDataBundle,
    *,
    suite_directory: Path | None = None,
) -> tuple[PreparedEvalDataBundle, DatasetSnapshotSpec]:
    """Revalidate a bundle against the live frozen suite and derive its snapshot."""

    verified = PreparedEvalDataBundle.model_validate(bundle.model_dump(mode="python"))
    suite, _ = _load_bound_suite(verified.request, suite_directory=suite_directory)
    if suite.suite_hash != verified.suite_sha256:
        raise EvalDataWorkflowError("prepared bundle suite hash drifted from the live frozen suite")
    snapshot = dataset_snapshot_from_prepared(
        suite=suite,
        scenario_id=verified.request.scenario_id,
        prepared=verified.prepared,
        request_sha256=verified.request_sha256,
        authority_sha256=verified.authority_sha256,
    )
    return verified, snapshot


def register_prepared_eval_bundle(
    *,
    bundle_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    suite_directory: Path | None = None,
) -> tuple[PreparedEvalDataBundle, DatasetSnapshotSpec, Path]:
    """Register only a self-verifying bundle as a public create-once snapshot."""

    _require_private_output_safe(Path(bundle_path))
    bundle = load_prepared_eval_bundle(bundle_path)
    verified, snapshot = validate_prepared_eval_bundle(
        bundle,
        suite_directory=suite_directory,
    )
    output = write_dataset_snapshot_manifest(snapshot, destination)
    return verified, snapshot, output


def _load_dataset_snapshot(path: str | os.PathLike[str]) -> DatasetSnapshotSpec:
    payload = _load_mapping(path, size_limit=4 * 1024 * 1024)
    return DatasetSnapshotSpec.model_validate(payload)


def check_prepared_eval_bundle(
    *,
    bundle_path: str | os.PathLike[str],
    snapshot_path: str | os.PathLike[str] | None = None,
    suite_directory: Path | None = None,
) -> EvalDataCheckReport:
    """Read and verify a bundle and optional public snapshot without writing."""

    bundle = load_prepared_eval_bundle(bundle_path)
    verified, expected_snapshot = validate_prepared_eval_bundle(
        bundle,
        suite_directory=suite_directory,
    )
    snapshot_matches: bool | None = None
    if snapshot_path is not None:
        actual_snapshot = _load_dataset_snapshot(snapshot_path)
        if actual_snapshot != expected_snapshot:
            raise EvalDataWorkflowError(
                "registered DatasetSnapshot differs from the verified prepared bundle"
            )
        snapshot_matches = True
    return EvalDataCheckReport(
        suite_id=verified.request.suite_id,
        suite_sha256=verified.suite_sha256,
        scenario_id=verified.request.scenario_id,
        request_sha256=verified.request_sha256,
        authority_sha256=verified.authority_sha256,
        source_sha256=verified.source_sha256,
        prepared_sha256=verified.prepared_sha256,
        dataset_snapshot_id=expected_snapshot.dataset_snapshot_id,
        dataset_snapshot_sha256=expected_snapshot.content_hash,
        snapshot_checked=snapshot_path is not None,
        snapshot_matches=snapshot_matches,
    )


def prepare_and_write_eval_data(
    *,
    request_path: str | os.PathLike[str],
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    suite_directory: Path | None = None,
) -> tuple[PreparedEvalDataBundle, Path]:
    """Load a request, materialize offline, and write one create-once bundle."""

    request = load_eval_data_request(request_path)
    try:
        bundle = prepare_eval_data_bundle(
            request=request,
            source=source,
            suite_directory=suite_directory,
        )
    except RealEvalDataError as error:
        raise EvalDataWorkflowError(str(error)) from error
    output = write_prepared_eval_bundle(bundle, destination)
    return bundle, output


__all__ = [
    "EvalDataCheckReport",
    "EvalDataPreparationRequest",
    "EvalDataWorkflowError",
    "PreparedEvalDataBundle",
    "check_prepared_eval_bundle",
    "load_eval_data_request",
    "load_prepared_eval_bundle",
    "prepare_and_write_eval_data",
    "prepare_eval_data_bundle",
    "register_prepared_eval_bundle",
    "validate_prepared_eval_bundle",
    "write_prepared_eval_bundle",
]
