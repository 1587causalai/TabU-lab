"""Replayable, repository-backed authorization for formal fit receipts.

The public catalog is an index, not an authority by itself.  Formal issuance
therefore starts from a caller-supplied canonical repository checkout and its
checked-in ``catalog.json``.  This module rebuilds the catalog from canonical
sources, requires byte-identical canonical output, and closes the experiment,
review, report, preregistration, and source-identity evidence chain.

Local paths are verification inputs only.  Receipts retain only
``FormalAuthorizationSummary``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tabu_lab.catalog import (
    CatalogBuildError,
    CatalogIndex,
    CatalogObjectKind,
    ExperimentRecord,
    ExperimentStatus,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    build_catalog,
)
from tabu_lab.contracts.canonical import canonical_hash, canonical_json
from tabu_lab.evidence.source_identity import SourceIdentity


class FormalAuthorizationError(ValueError):
    """A canonical formal-authorization replay failed closed."""


class FormalAuthorizationSummary(BaseModel):
    """Path-free authorization identity retained by a formal receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tabu.formal-run-authorization.v3"] = (
        "tabu.formal-run-authorization.v3"
    )
    canonical_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    catalog_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_source_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    experiment_id: str = Field(min_length=1)
    experiment_status: str = Field(min_length=1)
    preregistration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    review_ids: tuple[str, ...] = Field(min_length=1)
    review_report_sha256s: tuple[str, ...] = Field(min_length=1)
    gong_approval_sha256s: tuple[str, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class FormalAuthorizationContext:
    """Private replay inputs; never serialize this object into public evidence."""

    repository: Path
    catalog: Path
    experiment_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", Path(self.repository))
        object.__setattr__(self, "catalog", Path(self.catalog))
        if not isinstance(self.experiment_id, str) or not self.experiment_id:
            raise ValueError("formal authorization context requires experiment_id")


@dataclass(frozen=True, slots=True)
class VerifiedFormalAuthorization:
    """Replay result used by the runner and artifact writers."""

    summary: FormalAuthorizationSummary
    source_identity: SourceIdentity
    catalog: CatalogIndex


@dataclass(frozen=True, slots=True)
class _RemoteAuthority:
    """Live remote closure used for both issuance and historical replay."""

    repository_uri: str
    remote_name: str
    tracking_ref: str
    branch_ref: str
    remote_oid: str


@dataclass(slots=True)
class FormalAuthorizationReplaySession:
    """Replay path-free receipt summaries from one canonical Git history.

    The session reads immutable Git objects only.  It never checks out a commit
    into the caller's working tree.  Successful closures are cached, while a
    second visit to an in-progress summary is rejected as a history cycle.
    """

    repository: Path
    _cache: dict[str, VerifiedFormalAuthorization] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _visiting: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        supplied = Path(self.repository)
        repository = supplied.resolve()
        if supplied.is_symlink() or not repository.is_dir():
            raise FormalAuthorizationError(
                "formal authorization history requires a real repository directory"
            )
        git_root = Path(
            _git(repository, "rev-parse", "--show-toplevel").stdout.decode("utf-8").strip()
        ).resolve()
        if git_root != repository:
            raise FormalAuthorizationError(
                "formal authorization history currently supports only scope=. at an "
                "independent Git root"
            )
        object.__setattr__(self, "repository", repository)

    def verify(
        self,
        summary: FormalAuthorizationSummary | Mapping[str, Any],
        *,
        preregistration_text: str,
        live_source_identity: SourceIdentity,
    ) -> VerifiedFormalAuthorization:
        """Resolve and replay one recorded authorization summary."""

        try:
            expected = (
                summary
                if isinstance(summary, FormalAuthorizationSummary)
                else FormalAuthorizationSummary.model_validate(summary)
            )
        except ValueError as exc:
            raise FormalAuthorizationError(
                "recorded formal authorization summary is invalid"
            ) from exc
        if not isinstance(preregistration_text, str):
            raise TypeError("formal authorization preregistration_text must be a string")
        if not isinstance(live_source_identity, SourceIdentity):
            raise TypeError("formal authorization live source must be a SourceIdentity")
        supplied_preregistration = _preregistration_mapping(preregistration_text)
        if canonical_hash(supplied_preregistration) != expected.preregistration_sha256:
            raise FormalAuthorizationError(
                "receipt preregistration differs from its authorization summary"
            )
        if canonical_hash(live_source_identity) != expected.source_identity_sha256:
            raise FormalAuthorizationError(
                "receipt SourceIdentity differs from its authorization summary"
            )

        cache_key = canonical_hash(expected)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if cache_key in self._visiting:
            raise FormalAuthorizationError(
                "formal authorization Git history contains a replay cycle"
            )
        self._visiting.add(cache_key)
        try:
            commit = _resolve_history_commit(self.repository, expected.canonical_commit)
            with tempfile.TemporaryDirectory(prefix="tabu-formal-auth-history-") as directory:
                snapshot = Path(directory) / "repository"
                snapshot.mkdir()
                _materialize_git_commit(self.repository, commit, snapshot)
                verified = _verify_materialized_authorization(
                    repository=snapshot,
                    commit=commit,
                    preregistration_text=preregistration_text,
                    live_source_identity=live_source_identity,
                    expected_summary=expected,
                    replay=self,
                )
            self._cache[cache_key] = verified
            return verified
        finally:
            self._visiting.discard(cache_key)


_RUNNABLE_STATUSES = frozenset(
    {
        ExperimentStatus.RUNNABLE,
        ExperimentStatus.RUNNING,
        ExperimentStatus.SUCCEEDED,
        ExperimentStatus.FAILED,
        ExperimentStatus.KILLED,
        ExperimentStatus.REVIEWED,
    }
)


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FormalAuthorizationError(
            "canonical authorization Git context is unavailable"
        ) from exc
    if check and result.returncode != 0:
        raise FormalAuthorizationError("canonical authorization Git verification failed")
    return result


_SCP_REMOTE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _normalized_public_repository_uri(raw: str) -> str:
    """Normalize one public GitHub remote without credentials or mutability."""

    value = raw.strip()
    scp = _SCP_REMOTE.fullmatch(value)
    if scp is not None and "://" not in value:
        value = urlunsplit(
            ("https", scp.group("host").lower(), "/" + scp.group("path").lstrip("/"), "", "")
        )
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"https", "ssh"}
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
    ):
        raise FormalAuthorizationError(
            "formal authorization requires a public GitHub repository remote"
        )
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = PurePosixPath(path).parts
    if len(parts) != 3 or parts[0] != "/" or not parts[1] or not parts[2]:
        raise FormalAuthorizationError(
            "formal authorization remote must identify exactly one GitHub repository"
        )
    return urlunsplit(("https", "github.com", f"/{parts[1]}/{parts[2]}", "", ""))


def _remote_authority(repository: Path) -> _RemoteAuthority:
    """Resolve and query the actual configured upstream, never a declared field."""

    try:
        tracking_ref = (
            _git(repository, "rev-parse", "--symbolic-full-name", "@{upstream}")
            .stdout.decode("utf-8")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise FormalAuthorizationError(
            "formal authorization upstream ref is not valid UTF-8"
        ) from exc
    if not tracking_ref.startswith("refs/remotes/"):
        raise FormalAuthorizationError(
            "formal authorization requires a configured remote-tracking upstream"
        )
    remote_parts = tracking_ref.removeprefix("refs/remotes/").split("/", 1)
    if len(remote_parts) != 2 or not all(remote_parts):
        raise FormalAuthorizationError("formal authorization upstream ref is invalid")
    remote_name, branch = remote_parts
    branch_ref = f"refs/heads/{branch}"
    try:
        raw_uri = (
            _git(repository, "config", "--get", f"remote.{remote_name}.url")
            .stdout.decode("utf-8")
            .strip()
        )
    except UnicodeDecodeError as exc:
        raise FormalAuthorizationError(
            "formal authorization remote URI is not valid UTF-8"
        ) from exc
    repository_uri = _normalized_public_repository_uri(raw_uri)
    remote_oid = _public_remote_ref_oid(repository, repository_uri, branch_ref)
    tracking_oid = (
        _git(repository, "rev-parse", "--verify", f"{tracking_ref}^{{commit}}")
        .stdout.decode("utf-8")
        .strip()
    )
    if tracking_oid != remote_oid:
        raise FormalAuthorizationError(
            "formal authorization remote-tracking ref is stale"
        )
    _resolve_exact_commit(repository, remote_oid, label="remote head")
    return _RemoteAuthority(
        repository_uri=repository_uri,
        remote_name=remote_name,
        tracking_ref=tracking_ref,
        branch_ref=branch_ref,
        remote_oid=remote_oid,
    )


def _public_remote_ref_oid(
    repository: Path,
    repository_uri: str,
    branch_ref: str,
) -> str:
    """Query public HTTPS directly with all ambient Git rewrites disabled.

    ``repository`` is accepted so tests can substitute a real local bare-remote
    probe without changing the production command.  The production path never
    uses it as a working directory or Git configuration source.
    """

    del repository
    return _probe_public_https_ref_oid(repository_uri, branch_ref)


def _probe_public_https_ref_oid(repository_uri: str, branch_ref: str) -> str:
    """Execute the sanitized public-network part of the remote closure."""

    environment = os.environ.copy()
    for key in tuple(environment):
        if key in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_CONFIG",
            "GIT_CONFIG_COUNT",
            "GIT_CONFIG_PARAMETERS",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        } or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
            "GCM_INTERACTIVE": "never",
            "GIT_ASKPASS": "/usr/bin/false",
            "SSH_ASKPASS": "/usr/bin/false",
        }
    )
    with tempfile.TemporaryDirectory(prefix="tabu-public-remote-probe-") as directory:
        try:
            result = subprocess.run(
                (
                    "git",
                    "-c",
                    "protocol.file.allow=never",
                    "ls-remote",
                    "--exit-code",
                    repository_uri,
                    branch_ref,
                ),
                check=False,
                capture_output=True,
                cwd=directory,
                env=environment,
                timeout=15.0,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise FormalAuthorizationError(
                "formal authorization public remote probe is unavailable"
            ) from exc
    if result.returncode != 0:
        raise FormalAuthorizationError(
            "formal authorization public remote ref is not live retrievable"
        )
    for raw_line in result.stdout.splitlines():
        try:
            oid, ref = raw_line.decode("utf-8").split("\t", 1)
        except (UnicodeDecodeError, ValueError):
            continue
        if ref == branch_ref and _GIT_OBJECT.fullmatch(oid):
            return oid
    raise FormalAuthorizationError(
        "formal authorization public remote ref did not return an exact commit"
    )


def _resolve_exact_commit(repository: Path, declared: str, *, label: str) -> str:
    result = _git(
        repository,
        "rev-parse",
        "--verify",
        f"{declared}^{{commit}}",
        check=False,
    )
    if result.returncode != 0:
        raise FormalAuthorizationError(f"{label} does not name an existing exact Git commit")
    try:
        resolved = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise FormalAuthorizationError(
            f"{label} exact Git commit is not valid UTF-8"
        ) from exc
    if resolved != declared:
        raise FormalAuthorizationError(f"{label} does not name an exact Git commit")
    return resolved


def _require_remote_reachable_commit(
    repository: Path,
    declared: str,
    *,
    remote: _RemoteAuthority,
    label: str,
) -> str:
    resolved = _resolve_exact_commit(repository, declared, label=label)
    reachable = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        resolved,
        remote.remote_oid,
        check=False,
    )
    if reachable.returncode != 0:
        raise FormalAuthorizationError(f"{label} is not reachable from the live remote ref")
    return resolved


def _resolve_history_commit(repository: Path, declared: str) -> str:
    resolved = (
        _git(repository, "rev-parse", "--verify", f"{declared}^{{commit}}")
        .stdout.decode("utf-8")
        .strip()
    )
    if resolved != declared:
        raise FormalAuthorizationError(
            "formal authorization summary does not name an exact canonical commit"
        )
    reachable = _git(
        repository,
        "merge-base",
        "--is-ancestor",
        resolved,
        "HEAD",
        check=False,
    )
    if reachable.returncode != 0:
        raise FormalAuthorizationError(
            "formal authorization commit is not reachable from the canonical Git history"
        )
    return resolved


def _safe_archive_relative(name: str) -> PurePosixPath:
    if not name or "\\" in name:
        raise FormalAuthorizationError("formal authorization Git archive has an unsafe path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise FormalAuthorizationError("formal authorization Git archive has an unsafe path")
    return relative


def _materialize_git_commit(repository: Path, commit: str, destination: Path) -> None:
    archive_bytes = _git(repository, "archive", "--format=tar", commit).stdout
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
            seen: set[str] = set()
            for member in archive.getmembers():
                relative = _safe_archive_relative(member.name)
                normalized = relative.as_posix().rstrip("/")
                if normalized in seen:
                    raise FormalAuthorizationError(
                        "formal authorization Git archive has duplicate members"
                    )
                seen.add(normalized)
                target = destination.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise FormalAuthorizationError(
                        "formal authorization Git archive contains a non-regular member"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise FormalAuthorizationError(
                        "formal authorization Git archive member is unreadable"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
    except (OSError, tarfile.TarError) as exc:
        raise FormalAuthorizationError(
            "formal authorization Git commit could not be materialized"
        ) from exc


def _verify_catalog_remote_binding(
    *,
    repository: Path,
    catalog: CatalogIndex,
    authorization_commit: str,
    remote: _RemoteAuthority,
    replay: FormalAuthorizationReplaySession | None = None,
) -> None:
    revision = catalog.source_revision
    if revision is None:
        raise FormalAuthorizationError(
            "formal authorization catalog lacks CatalogSourceRevision"
        )
    declared_uri = _normalized_public_repository_uri(revision.repository_uri)
    if declared_uri != remote.repository_uri:
        raise FormalAuthorizationError(
            "catalog source revision does not match the configured public remote"
        )
    if revision.catalog_source_tree_hash != catalog.source_tree_hash:
        raise FormalAuthorizationError(
            "catalog source revision does not bind the catalog source tree"
        )
    revision_commit = _require_remote_reachable_commit(
        repository,
        revision.commit,
        remote=remote,
        label="catalog source revision commit",
    )
    _require_remote_reachable_commit(
        repository,
        authorization_commit,
        remote=remote,
        label="authorization canonical commit",
    )
    with tempfile.TemporaryDirectory(prefix="tabu-catalog-source-revision-") as directory:
        snapshot = Path(directory) / "repository"
        snapshot.mkdir()
        _materialize_git_commit(repository, revision_commit, snapshot)
        try:
            rebuilt = build_catalog(
                snapshot,
                authorization_replay=replay or FormalAuthorizationReplaySession(repository),
            )
        except (CatalogBuildError, ValueError) as exc:
            raise FormalAuthorizationError(
                "catalog source revision commit cannot be independently rebuilt"
            ) from exc
    if rebuilt.source_tree_hash != revision.catalog_source_tree_hash:
        raise FormalAuthorizationError(
            "catalog source revision hash does not match its immutable commit sources"
        )


def _repository_source_files(source_root: Path) -> tuple[dict[str, object], ...]:
    """Rebuild the runner's repository-mode source manifest from immutable bytes."""

    candidates: list[Path] = []
    for relative in ("src/tabu_lab", "specs/models", "schemas"):
        directory = source_root / relative
        if directory.is_dir():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            )
    candidates.extend(
        path for path in (source_root / "pyproject.toml", source_root / "uv.lock") if path.is_file()
    )
    if not candidates:
        raise FormalAuthorizationError(
            "formal Git source commit has no executable source manifest files"
        )
    files: list[dict[str, object]] = []
    for path in sorted(set(candidates)):
        if path.is_symlink() or not path.is_file():
            raise FormalAuthorizationError(
                "formal Git source manifest contains a non-regular file"
            )
        payload = path.read_bytes()
        files.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        )
    return tuple(files)


def _verify_git_source_identity(
    *,
    authority_repository: Path,
    remote: _RemoteAuthority,
    source_identity: SourceIdentity,
    preregistration_relative: str,
) -> None:
    """Independently replay every formal Git identity field from Git objects."""

    if source_identity.source_kind != "git":
        raise FormalAuthorizationError("formal authorization source is not Git-backed")
    required = (
        source_identity.repository_uri,
        source_identity.repository_subdirectory,
        source_identity.commit,
        source_identity.remote_ref,
        source_identity.git_tree_oid,
        source_identity.source_tree_hash,
        source_identity.preregistration_blob_hash,
        source_identity.lock_hash,
    )
    if any(value is None for value in required):
        raise FormalAuthorizationError("formal Git SourceIdentity lacks a required binding")
    assert source_identity.repository_uri is not None
    assert source_identity.repository_subdirectory is not None
    assert source_identity.commit is not None
    assert source_identity.remote_ref is not None
    assert source_identity.git_tree_oid is not None
    assert source_identity.source_tree_hash is not None
    assert source_identity.preregistration_blob_hash is not None
    assert source_identity.lock_hash is not None
    if _normalized_public_repository_uri(source_identity.repository_uri) != remote.repository_uri:
        raise FormalAuthorizationError(
            "formal Git SourceIdentity repository differs from the configured public remote"
        )
    if source_identity.remote_ref != remote.tracking_ref:
        raise FormalAuthorizationError(
            "formal Git SourceIdentity remote ref differs from the configured upstream"
        )
    source_commit = _require_remote_reachable_commit(
        authority_repository,
        source_identity.commit,
        remote=remote,
        label="formal Git source commit",
    )
    tree_expression = (
        f"{source_commit}^{{tree}}"
        if source_identity.repository_subdirectory == "."
        else f"{source_commit}:{source_identity.repository_subdirectory}"
    )
    actual_tree_oid = (
        _git(authority_repository, "rev-parse", "--verify", tree_expression)
        .stdout.decode("utf-8")
        .strip()
    )
    if actual_tree_oid != source_identity.git_tree_oid:
        raise FormalAuthorizationError("formal Git SourceIdentity tree oid is not authentic")

    with tempfile.TemporaryDirectory(prefix="tabu-formal-source-") as directory:
        snapshot = Path(directory) / "repository"
        snapshot.mkdir()
        _materialize_git_commit(authority_repository, source_commit, snapshot)
        source_root = (
            snapshot
            if source_identity.repository_subdirectory == "."
            else snapshot / source_identity.repository_subdirectory
        )
        if source_root.is_symlink() or not source_root.is_dir():
            raise FormalAuthorizationError(
                "formal Git SourceIdentity subdirectory is absent from its source commit"
            )
        files = _repository_source_files(source_root)
        actual_source_tree_hash = canonical_hash(
            {
                "schema_version": "tabu.source-tree-preimage.v1",
                "mode": "repository",
                "root_label": "repository",
                "files": files,
            }
        )
        if actual_source_tree_hash != source_identity.source_tree_hash:
            raise FormalAuthorizationError(
                "formal Git SourceIdentity source-tree manifest does not match commit bytes"
            )
        lock_path = source_root / "uv.lock"
        if lock_path.is_symlink() or not lock_path.is_file():
            raise FormalAuthorizationError("formal Git source commit lacks uv.lock")
        if hashlib.sha256(lock_path.read_bytes()).hexdigest() != source_identity.lock_hash:
            raise FormalAuthorizationError(
                "formal Git SourceIdentity uv.lock hash is not authentic"
            )

        preregistration_path = snapshot.joinpath(*PurePosixPath(preregistration_relative).parts)
        try:
            preregistration_path.relative_to(source_root)
        except ValueError as exc:
            raise FormalAuthorizationError(
                "formal Git preregistration lies outside the declared source scope"
            ) from exc
        if preregistration_path.is_symlink() or not preregistration_path.is_file():
            raise FormalAuthorizationError(
                "formal Git source commit lacks the canonical preregistration blob"
            )
        if (
            hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
            != source_identity.preregistration_blob_hash
        ):
            raise FormalAuthorizationError(
                "formal Git SourceIdentity preregistration blob hash is not authentic"
            )


def _canonical_repository(context: FormalAuthorizationContext) -> tuple[Path, Path, str, str]:
    repository = context.repository.resolve()
    if not repository.is_dir() or context.repository.is_symlink():
        raise FormalAuthorizationError("canonical authorization repository is not a real directory")
    git_root = Path(
        _git(repository, "rev-parse", "--show-toplevel").stdout.decode("utf-8").strip()
    ).resolve()
    try:
        scope = repository.relative_to(git_root)
    except ValueError as exc:
        raise FormalAuthorizationError(
            "canonical authorization repository escapes its Git checkout"
        ) from exc
    repository_scope = "." if scope == Path(".") else scope.as_posix()
    if repository_scope != ".":
        raise FormalAuthorizationError(
            "formal authorization currently supports only scope=. at an independent Git root"
        )
    status = _git(
        git_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        repository_scope,
    ).stdout
    if status.strip():
        raise FormalAuthorizationError("canonical authorization repository is not clean")
    commit = _git(git_root, "rev-parse", "HEAD").stdout.decode("utf-8").strip()
    return repository, git_root, repository_scope, commit


def _git_path(repository_scope: str, relative: str) -> str:
    return relative if repository_scope == "." else f"{repository_scope}/{relative}"


def _repository_relative_evidence_uri(uri: str, *, label: str) -> str:
    if (
        not isinstance(uri, str)
        or not uri
        or "://" in uri
        or "\\" in uri
        or any(ord(character) < 32 or ord(character) == 127 for character in uri)
    ):
        raise FormalAuthorizationError(f"{label} must be a repository-relative evidence pointer")
    pure = PurePosixPath(uri.removeprefix("./"))
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise FormalAuthorizationError(f"{label} has an unsafe repository-relative pointer")
    return pure.as_posix()


def _relative_evidence_path(repository: Path, uri: str, *, label: str) -> tuple[Path, str]:
    repository = repository.resolve()
    relative = _repository_relative_evidence_uri(uri, label=label)
    candidate = repository / relative
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repository)
    except ValueError as exc:
        raise FormalAuthorizationError(f"{label} escapes the canonical repository") from exc
    if candidate.is_symlink() or not candidate.is_file() or resolved != candidate.absolute():
        raise FormalAuthorizationError(f"{label} source is missing or not a regular file")
    return candidate, relative


def _committed_bytes(
    *,
    git_root: Path,
    repository_scope: str,
    path: Path,
    relative: str,
    label: str,
) -> bytes:
    try:
        current = path.read_bytes()
    except OSError as exc:
        raise FormalAuthorizationError(f"cannot read canonical {label} source") from exc
    committed = _git(git_root, "show", f"HEAD:{_git_path(repository_scope, relative)}").stdout
    if current != committed:
        raise FormalAuthorizationError(f"canonical {label} source differs from HEAD")
    return current


def _read_mapping_bytes(payload: bytes, *, suffix: str, label: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(text) if suffix.lower() == ".json" else yaml.safe_load(text)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise FormalAuthorizationError(f"canonical {label} source is invalid") from exc
    if not isinstance(value, dict):
        raise FormalAuthorizationError(f"canonical {label} source must contain one mapping")
    return value


def _evidence_digest(path: Path, payload: bytes, *, label: str) -> str:
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return canonical_hash(
            _read_mapping_bytes(payload, suffix=path.suffix, label=label)
        )
    return hashlib.sha256(payload).hexdigest()


def verify_committed_evidence_pointer(
    repository: str | Path,
    commit: str,
    *,
    uri: str,
    sha256: str,
    label: str,
) -> str:
    """Verify one path-safe regular Git blob at an exact canonical commit.

    JSON and YAML pointers use the catalog's canonical-mapping digest semantics;
    all other files use raw SHA-256.  The lookup reads the immutable Git object,
    not the caller's working tree, so historical replay is unaffected by later
    edits while symlinks, submodules, missing files, and path tricks fail closed.
    """

    supplied = Path(repository)
    root = supplied.resolve()
    if supplied.is_symlink() or not root.is_dir():
        raise FormalAuthorizationError(
            f"{label} verification requires a real canonical Git repository"
        )
    git_root = Path(
        _git(root, "rev-parse", "--show-toplevel").stdout.decode("utf-8").strip()
    ).resolve()
    if git_root != root:
        raise FormalAuthorizationError(
            f"{label} verification currently requires the canonical Git root"
        )
    resolved_commit = _resolve_exact_commit(root, commit, label=f"{label} commit")
    relative = _repository_relative_evidence_uri(uri, label=label)
    listing = _git(
        root,
        "ls-tree",
        "-z",
        "--full-tree",
        resolved_commit,
        "--",
        f":(literal){relative}",
    ).stdout
    records = tuple(record for record in listing.split(b"\0") if record)
    if len(records) != 1:
        raise FormalAuthorizationError(
            f"{label} is missing or not uniquely committed at the authorization commit"
        )
    try:
        metadata, encoded_path = records[0].split(b"\t", 1)
        mode, object_type, object_oid = metadata.decode("ascii").split(" ", 2)
        committed_path = encoded_path.decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise FormalAuthorizationError(f"{label} has an invalid Git tree entry") from exc
    if committed_path != relative:
        raise FormalAuthorizationError(f"{label} Git tree entry differs from its pointer")
    if mode not in {"100644", "100755"} or object_type != "blob":
        raise FormalAuthorizationError(f"{label} must name one committed regular file")
    payload = _git(root, "cat-file", "blob", object_oid).stdout
    digest = _evidence_digest(Path(relative), payload, label=label)
    if digest != sha256:
        raise FormalAuthorizationError(f"{label} source digest differs from its pointer")
    return digest


def _preregistration_mapping(preregistration_text: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(preregistration_text)
    except yaml.YAMLError as exc:
        raise FormalAuthorizationError("supplied preregistration text is invalid") from exc
    if not isinstance(payload, dict):
        raise FormalAuthorizationError("supplied preregistration must contain one mapping")
    return payload


def _authorization_source_bytes(
    *,
    path: Path,
    relative: str,
    label: str,
    git_root: Path | None,
    repository_scope: str | None,
) -> bytes:
    if git_root is not None and repository_scope is not None:
        return _committed_bytes(
            git_root=git_root,
            repository_scope=repository_scope,
            path=path,
            relative=relative,
            label=label,
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FormalAuthorizationError(f"cannot read canonical {label} source") from exc


def _load_catalog_context(
    context: FormalAuthorizationContext,
) -> tuple[Path, Path, str, str, CatalogIndex, _RemoteAuthority]:
    repository, git_root, repository_scope, commit = _canonical_repository(context)
    catalog_path = context.catalog.resolve()
    if catalog_path != repository / "catalog.json":
        raise FormalAuthorizationError(
            "authorization catalog must be the canonical repository catalog.json"
        )
    try:
        catalog_relative = catalog_path.relative_to(repository).as_posix()
    except ValueError as exc:
        raise FormalAuthorizationError(
            "authorization catalog must be checked into the canonical repository"
        ) from exc
    if context.catalog.is_symlink() or not context.catalog.is_file():
        raise FormalAuthorizationError("authorization catalog is missing or not a regular file")
    catalog_bytes = _committed_bytes(
        git_root=git_root,
        repository_scope=repository_scope,
        path=context.catalog,
        relative=catalog_relative,
        label="catalog",
    )
    try:
        supplied = CatalogIndex.model_validate_json(catalog_bytes)
    except ValueError as exc:
        raise FormalAuthorizationError("authorization catalog is not a valid CatalogIndex") from exc
    try:
        rebuilt = build_catalog(
            repository,
            source_revision=supplied.source_revision,
        )
    except (CatalogBuildError, ValueError) as exc:
        raise FormalAuthorizationError("canonical authorization catalog rebuild failed") from exc
    expected_bytes = (canonical_json(rebuilt) + "\n").encode("utf-8")
    if catalog_bytes != expected_bytes:
        raise FormalAuthorizationError(
            "authorization catalog bytes differ from the canonical repository rebuild"
        )
    if supplied != rebuilt:
        raise FormalAuthorizationError(
            "authorization catalog canonical object differs from the repository rebuild"
        )
    remote = _remote_authority(git_root)
    _verify_catalog_remote_binding(
        repository=git_root,
        catalog=rebuilt,
        authorization_commit=commit,
        remote=remote,
    )
    for entry in rebuilt.entries:
        source, relative = _relative_evidence_path(
            repository,
            entry.source_path,
            label="catalog object",
        )
        _committed_bytes(
            git_root=git_root,
            repository_scope=repository_scope,
            path=source,
            relative=relative,
            label="catalog object",
        )
    return repository, git_root, repository_scope, commit, rebuilt, remote


def _verify_catalog_authorization(
    *,
    repository: Path,
    catalog: CatalogIndex,
    commit: str,
    experiment_id: str,
    preregistration_text: str,
    live_source_identity: SourceIdentity | None,
    expected_summary: FormalAuthorizationSummary | Mapping[str, Any] | None,
    git_root: Path | None,
    repository_scope: str | None,
    authority_git_repository: Path,
    remote_authority: _RemoteAuthority,
    historical_replay: bool,
) -> VerifiedFormalAuthorization:
    try:
        experiment_entry = catalog.show(experiment_id)
    except KeyError as exc:
        raise FormalAuthorizationError(
            "authorization catalog does not contain the target ExperimentRecord"
        ) from exc
    if experiment_entry.kind is not CatalogObjectKind.EXPERIMENT:
        raise FormalAuthorizationError("authorization target is not an ExperimentRecord")
    experiment = ExperimentRecord.model_validate(experiment_entry.data)
    if experiment_entry.status != experiment.status.value:
        raise FormalAuthorizationError("ExperimentRecord status projection differs")
    if experiment.status not in _RUNNABLE_STATUSES:
        raise FormalAuthorizationError("formal authorization requires runnable status or later")
    if experiment.preregistration is None:
        raise FormalAuthorizationError("ExperimentRecord lacks canonical preregistration evidence")

    preregistration_path, preregistration_relative = _relative_evidence_path(
        repository,
        experiment.preregistration.uri,
        label="preregistration",
    )
    preregistration_bytes = _authorization_source_bytes(
        path=preregistration_path,
        relative=preregistration_relative,
        label="preregistration",
        git_root=git_root,
        repository_scope=repository_scope,
    )
    preregistration_payload = _read_mapping_bytes(
        preregistration_bytes,
        suffix=preregistration_path.suffix,
        label="preregistration",
    )
    preregistration_sha256 = canonical_hash(preregistration_payload)
    if preregistration_sha256 != experiment.preregistration.sha256:
        raise FormalAuthorizationError("canonical preregistration digest differs from its pointer")
    supplied_preregistration = _preregistration_mapping(preregistration_text)
    if canonical_hash(supplied_preregistration) != preregistration_sha256:
        raise FormalAuthorizationError("supplied preregistration differs from canonical source")
    if (
        supplied_preregistration.get("experiment_id") != experiment.experiment_id
        or supplied_preregistration.get("contract_id") != experiment.contract_id
    ):
        raise FormalAuthorizationError(
            "canonical preregistration identity differs from ExperimentRecord"
        )

    if experiment.source_identity is None:
        raise FormalAuthorizationError("ExperimentRecord lacks source_identity evidence")
    source_path, source_relative = _relative_evidence_path(
        repository,
        experiment.source_identity.uri,
        label="SourceIdentity",
    )
    source_bytes = _authorization_source_bytes(
        path=source_path,
        relative=source_relative,
        label="SourceIdentity",
        git_root=git_root,
        repository_scope=repository_scope,
    )
    source_payload = _read_mapping_bytes(
        source_bytes,
        suffix=source_path.suffix,
        label="SourceIdentity",
    )
    try:
        source_identity = SourceIdentity.model_validate(source_payload)
    except ValueError as exc:
        raise FormalAuthorizationError("canonical SourceIdentity source is invalid") from exc
    source_identity_sha256 = canonical_hash(source_identity)
    if source_identity_sha256 != experiment.source_identity.sha256:
        raise FormalAuthorizationError("canonical SourceIdentity digest differs from its pointer")
    if source_identity.issuance_status != "formal" or not source_identity.reviewed:
        raise FormalAuthorizationError("canonical SourceIdentity is not formal and reviewed")
    if source_identity.source_kind == "distribution":
        if historical_replay:
            raise FormalAuthorizationError(
                "historical distribution formal authorization cannot be independently replayed"
            )
        raise FormalAuthorizationError(
            "distribution formal authorization requires immutable archive bytes; unsupported"
        )
    _verify_git_source_identity(
        authority_repository=authority_git_repository,
        remote=remote_authority,
        source_identity=source_identity,
        preregistration_relative=preregistration_relative,
    )
    if live_source_identity is not None and live_source_identity != source_identity:
        raise FormalAuthorizationError("live SourceIdentity differs from canonical source object")

    if experiment.preregistration_review is None or not experiment.review_ids:
        raise FormalAuthorizationError("ExperimentRecord lacks independent review evidence")
    subject = ObjectRef(
        kind=CatalogObjectKind.EXPERIMENT,
        object_id=experiment.experiment_id,
    )
    matched_reviews: list[ReviewRecord] = []
    for review_id in experiment.review_ids:
        try:
            review_entry = catalog.show(review_id)
        except KeyError as exc:
            raise FormalAuthorizationError(
                "ExperimentRecord review is absent from catalog"
            ) from exc
        if review_entry.kind is not CatalogObjectKind.REVIEW:
            raise FormalAuthorizationError("ExperimentRecord review id names a non-review object")
        review = ReviewRecord.model_validate(review_entry.data)
        if (
            review.decision is ReviewDecision.APPROVED
            and subject in review.subjects
            and review.report == experiment.preregistration_review
            and review.developer_identity.strip().casefold()
            != review.reviewer_identity.strip().casefold()
        ):
            matched_reviews.append(review)
    if not matched_reviews:
        raise FormalAuthorizationError(
            "formal authorization requires a matching approved independent ReviewRecord"
        )

    report_hashes: list[str] = []
    gong_approval_hashes: list[str] = []
    for review in matched_reviews:
        if review.gong_approval is None:
            raise FormalAuthorizationError(
                "formal authorization review lacks gong approval evidence"
            )
        report_path, report_relative = _relative_evidence_path(
            repository,
            review.report.uri,
            label="review report",
        )
        report_bytes = _authorization_source_bytes(
            path=report_path,
            relative=report_relative,
            label="review report",
            git_root=git_root,
            repository_scope=repository_scope,
        )
        report_sha256 = _evidence_digest(
            report_path,
            report_bytes,
            label="review report",
        )
        if report_sha256 != review.report.sha256:
            raise FormalAuthorizationError("review report source digest differs from its pointer")
        report_hashes.append(report_sha256)
        gong_path, gong_relative = _relative_evidence_path(
            repository,
            review.gong_approval.uri,
            label="gong approval",
        )
        gong_bytes = _authorization_source_bytes(
            path=gong_path,
            relative=gong_relative,
            label="gong approval",
            git_root=git_root,
            repository_scope=repository_scope,
        )
        gong_sha256 = _evidence_digest(
            gong_path,
            gong_bytes,
            label="gong approval",
        )
        if gong_sha256 != review.gong_approval.sha256:
            raise FormalAuthorizationError(
                "gong approval source digest differs from its pointer"
            )
        gong_approval_hashes.append(gong_sha256)

    summary = FormalAuthorizationSummary(
        canonical_commit=commit,
        catalog_hash=catalog.catalog_hash,
        catalog_source_tree_hash=catalog.source_tree_hash,
        experiment_id=experiment.experiment_id,
        experiment_status=experiment.status.value,
        preregistration_sha256=preregistration_sha256,
        source_identity_sha256=source_identity_sha256,
        review_ids=tuple(sorted(review.review_id for review in matched_reviews)),
        review_report_sha256s=tuple(sorted(report_hashes)),
        gong_approval_sha256s=tuple(sorted(gong_approval_hashes)),
    )
    if expected_summary is not None:
        try:
            expected = (
                expected_summary
                if isinstance(expected_summary, FormalAuthorizationSummary)
                else FormalAuthorizationSummary.model_validate(expected_summary)
            )
        except ValueError as exc:
            raise FormalAuthorizationError(
                "receipt formal authorization summary is invalid"
            ) from exc
        if expected != summary:
            raise FormalAuthorizationError(
                "receipt formal authorization summary differs from canonical replay"
            )
    return VerifiedFormalAuthorization(
        summary=summary,
        source_identity=source_identity,
        catalog=catalog,
    )


def _verify_materialized_authorization(
    *,
    repository: Path,
    commit: str,
    preregistration_text: str,
    live_source_identity: SourceIdentity,
    expected_summary: FormalAuthorizationSummary,
    replay: FormalAuthorizationReplaySession,
) -> VerifiedFormalAuthorization:
    catalog_path = repository / "catalog.json"
    if catalog_path.is_symlink() or not catalog_path.is_file():
        raise FormalAuthorizationError(
            "formal authorization commit lacks a checked-in catalog.json"
        )
    try:
        catalog_bytes = catalog_path.read_bytes()
        supplied = CatalogIndex.model_validate_json(catalog_bytes)
    except (OSError, ValueError) as exc:
        raise FormalAuthorizationError(
            "formal authorization commit has an invalid catalog.json"
        ) from exc
    remote = _remote_authority(replay.repository)
    _verify_catalog_remote_binding(
        repository=replay.repository,
        catalog=supplied,
        authorization_commit=commit,
        remote=remote,
        replay=replay,
    )
    if (
        supplied.catalog_hash != expected_summary.catalog_hash
        or supplied.source_tree_hash != expected_summary.catalog_source_tree_hash
    ):
        raise FormalAuthorizationError(
            "recorded catalog identity differs from its canonical Git commit"
        )
    try:
        rebuilt = build_catalog(
            repository,
            source_revision=supplied.source_revision,
            authorization_replay=replay,
        )
    except (CatalogBuildError, ValueError) as exc:
        raise FormalAuthorizationError(
            "formal authorization Git catalog rebuild failed"
        ) from exc
    expected_bytes = (canonical_json(rebuilt) + "\n").encode("utf-8")
    if catalog_bytes != expected_bytes or supplied != rebuilt:
        raise FormalAuthorizationError(
            "formal authorization Git catalog differs from its canonical rebuild"
        )
    for entry in rebuilt.entries:
        _relative_evidence_path(repository, entry.source_path, label="catalog object")
    return _verify_catalog_authorization(
        repository=repository,
        catalog=rebuilt,
        commit=commit,
        experiment_id=expected_summary.experiment_id,
        preregistration_text=preregistration_text,
        live_source_identity=live_source_identity,
        expected_summary=expected_summary,
        git_root=None,
        repository_scope=None,
        authority_git_repository=replay.repository,
        remote_authority=remote,
        historical_replay=True,
    )


def verify_formal_authorization(
    context: FormalAuthorizationContext,
    *,
    preregistration_text: str,
    live_source_identity: SourceIdentity | None = None,
    expected_summary: FormalAuthorizationSummary | Mapping[str, Any] | None = None,
) -> VerifiedFormalAuthorization:
    """Replay the complete canonical authorization closure.

    The first runner preflight may omit ``live_source_identity``; the canonical
    source object is still validated and returned.  Writers and final runner
    preflight pass the live identity and require exact equality.
    """

    if not isinstance(context, FormalAuthorizationContext):
        raise TypeError("formal authorization requires a FormalAuthorizationContext")
    if not isinstance(preregistration_text, str):
        raise TypeError("formal authorization preregistration_text must be a string")
    repository, git_root, repository_scope, commit, catalog, remote = _load_catalog_context(
        context
    )
    return _verify_catalog_authorization(
        repository=repository,
        catalog=catalog,
        commit=commit,
        experiment_id=context.experiment_id,
        preregistration_text=preregistration_text,
        live_source_identity=live_source_identity,
        expected_summary=expected_summary,
        git_root=git_root,
        repository_scope=repository_scope,
        authority_git_repository=git_root,
        remote_authority=remote,
        historical_replay=False,
    )


__all__ = [
    "FormalAuthorizationContext",
    "FormalAuthorizationError",
    "FormalAuthorizationReplaySession",
    "FormalAuthorizationSummary",
    "VerifiedFormalAuthorization",
    "verify_committed_evidence_pointer",
    "verify_formal_authorization",
]
