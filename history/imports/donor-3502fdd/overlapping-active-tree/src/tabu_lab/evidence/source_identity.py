"""Fail-closed source identities for formal and local-only experiment evidence."""

from __future__ import annotations

import hashlib
import io
import re
import stat
import subprocess
import tarfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tabu_lab.contracts import canonical_hash

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_OBJECT_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"
_SCP_REMOTE = re.compile(r"^(?:[^@/]+@)?(?P<host>[^:/]+):(?P<path>.+)$")
_DISTRIBUTION_SUFFIXES = (".whl", ".tar.gz", ".tar.bz2", ".tar.xz", ".zip")


class SourceIdentity(BaseModel):
    """Public, path-free identity of the source that produced an attempt.

    ``formal`` is deliberately a narrow state. A Git identity is formal only
    when the exact preregistration is committed, the declared source scope is
    clean, and the commit is reachable from a configured remote-tracking ref.
    The scope may be a repository root or a public repository subdirectory;
    unrelated parent-repository dirt is deliberately outside that identity.
    A distribution identity is formal only when a reviewer-approved immutable
    wheel/sdist URI, digest, dependency-lock digest, and exact installed-package
    source-tree binding are all present.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tabu.source-identity.v2"] = "tabu.source-identity.v2"
    source_kind: Literal["git", "distribution", "local"]
    issuance_status: Literal["formal", "local_unissued"]
    reviewed: bool = False

    repository_uri: str | None = None
    repository_subdirectory: str | None = None
    commit: str | None = Field(default=None, pattern=_GIT_OBJECT_PATTERN)
    remote_ref: str | None = None
    git_tree_oid: str | None = Field(default=None, pattern=_GIT_OBJECT_PATTERN)
    source_tree_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    preregistration_blob_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    distribution_uri: str | None = None
    distribution_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    lock_hash: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _coherent_boundary(self) -> SourceIdentity:
        if len(self.reasons) != len(set(self.reasons)):
            raise ValueError("SourceIdentity reasons must be unique")
        if any(not reason or reason != reason.strip() for reason in self.reasons):
            raise ValueError("SourceIdentity reasons must be non-empty normalized strings")
        if self.repository_subdirectory is not None:
            subdirectory = self.repository_subdirectory
            if (
                not subdirectory
                or subdirectory.startswith(("/", "\\"))
                or "\\" in subdirectory
                or subdirectory != Path(subdirectory).as_posix()
                or (subdirectory != "." and "." in Path(subdirectory).parts)
                or ".." in Path(subdirectory).parts
            ):
                raise ValueError(
                    "repository_subdirectory must be a normalized repository-relative path"
                )
        if self.issuance_status == "local_unissued":
            if not self.reasons:
                raise ValueError("local_unissued SourceIdentity requires an explicit reason")
            return self
        if self.reasons:
            raise ValueError("formal SourceIdentity cannot retain local-only reasons")
        if not self.reviewed:
            raise ValueError("formal SourceIdentity requires reviewed=True")
        if self.source_kind == "git":
            required = (
                self.repository_uri,
                self.repository_subdirectory,
                self.commit,
                self.remote_ref,
                self.git_tree_oid,
                self.source_tree_hash,
                self.preregistration_blob_hash,
                self.lock_hash,
            )
            if any(value is None for value in required):
                raise ValueError("formal Git SourceIdentity is missing a required binding")
            if self.distribution_uri is not None or self.distribution_sha256 is not None:
                raise ValueError("formal Git SourceIdentity cannot include distribution fields")
        elif self.source_kind == "distribution":
            required = (
                self.source_tree_hash,
                self.distribution_uri,
                self.distribution_sha256,
                self.lock_hash,
            )
            if any(value is None for value in required):
                raise ValueError("formal distribution SourceIdentity is missing a required binding")
            if any(
                value is not None
                for value in (
                    self.repository_uri,
                    self.repository_subdirectory,
                    self.commit,
                    self.remote_ref,
                    self.git_tree_oid,
                    self.preregistration_blob_hash,
                )
            ):
                raise ValueError(
                    "formal distribution SourceIdentity cannot include Git-only fields"
                )
        else:
            raise ValueError("local source kind cannot be promoted to formal")
        return self


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", "-C", str(root), *arguments),
        check=check,
        capture_output=True,
        timeout=timeout,
    )


def _public_repository_uri(raw: str) -> str | None:
    """Normalize common public Git remotes without retaining userinfo or query data."""

    value = raw.strip()
    scp = _SCP_REMOTE.fullmatch(value)
    if scp is not None and "://" not in value:
        host = scp.group("host").lower()
        path = "/" + scp.group("path").lstrip("/")
        return urlunsplit(("https", host, path, "", ""))
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https", "ssh"} or not parsed.hostname:
        return None
    scheme = "https" if parsed.scheme == "ssh" else parsed.scheme
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((scheme, f"{parsed.hostname.lower()}{port}", parsed.path, "", ""))


def _public_distribution_uri(raw: str) -> str | None:
    """Return a path-safe HTTPS wheel/sdist URI, never a local retrieval path."""

    value = raw.strip()
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None
    if not parsed.path.lower().endswith(_DISTRIBUTION_SUFFIXES):
        return None
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(("https", f"{parsed.hostname.lower()}{port}", parsed.path, "", ""))


def _remote_ref_oid(root: Path, remote_name: str, branch_ref: str) -> str | None:
    """Resolve a branch from the actual remote instead of trusting a local tracking ref."""

    result = _git(
        root,
        "ls-remote",
        "--exit-code",
        remote_name,
        branch_ref,
        check=False,
        timeout=10.0,
    )
    if result.returncode != 0:
        return None
    for raw_line in result.stdout.splitlines():
        fields = raw_line.decode().split("\t", 1)
        if len(fields) == 2 and fields[1] == branch_ref and re.fullmatch(
            _GIT_OBJECT_PATTERN, fields[0]
        ):
            return fields[0]
    return None


def _verification_bytes(value: bytes | str | Path | None) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if value is None:
        return None
    try:
        path = Path(value)
        if not path.is_file():
            return None
        return path.read_bytes()
    except (OSError, TypeError, ValueError):
        return None


def _scoped_git_path(repository_subdirectory: str, relative_path: str) -> str:
    if repository_subdirectory == ".":
        return relative_path
    return f"{repository_subdirectory}/{relative_path}"


def _verify_source_files_at_head(
    *,
    git_root: Path,
    source_root: Path,
    repository_subdirectory: str,
    source_files: Sequence[Mapping[str, object]] | None,
) -> tuple[str, ...]:
    """Verify that every hashed source-manifest entry is a regular HEAD blob.

    Scoped ``git status`` catches ordinary tracked and untracked changes, but an
    ignored source file or a tracked symlink can otherwise enter the source hash
    without being retrievable from the reviewed commit. The manifest entries are
    therefore checked independently against ``HEAD`` before formal issuance.
    Reasons are intentionally path-free because they are serialized publicly.
    """

    if source_files is None:
        return ("source_manifest_not_provided",)
    if not source_files:
        return ("source_manifest_empty",)
    reasons: list[str] = []
    seen: set[str] = set()
    for entry in source_files:
        relative_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        expected_size = entry.get("size")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith(("/", "\\"))
            or "\\" in relative_path
            or relative_path != Path(relative_path).as_posix()
            or "." in Path(relative_path).parts
            or ".." in Path(relative_path).parts
            or not isinstance(expected_sha256, str)
            or re.fullmatch(_SHA256_PATTERN, expected_sha256) is None
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            reasons.append("source_manifest_entry_invalid")
            continue
        if relative_path in seen:
            reasons.append("source_manifest_entry_duplicate")
            continue
        seen.add(relative_path)
        candidate = source_root / relative_path
        try:
            if (
                candidate.resolve() != candidate.absolute()
                or candidate.is_symlink()
                or not candidate.is_file()
            ):
                reasons.append("source_manifest_file_not_regular")
                continue
            current = candidate.read_bytes()
        except OSError:
            reasons.append("source_manifest_file_unreadable")
            continue
        if len(current) != expected_size or _sha256_bytes(current) != expected_sha256:
            reasons.append("source_manifest_file_drift")
            continue
        git_path = _scoped_git_path(repository_subdirectory, relative_path)
        try:
            committed = _git(git_root, "show", f"HEAD:{git_path}").stdout
        except subprocess.CalledProcessError:
            reasons.append("source_manifest_file_not_committed")
            continue
        if committed != current:
            reasons.append("source_manifest_file_differs_from_head")
    return tuple(dict.fromkeys(reasons))


def _normalized_archive_name(raw_name: str, *, directory: bool) -> str | None:
    """Return one safe POSIX archive name without extracting it to disk."""

    if (
        not raw_name
        or "\x00" in raw_name
        or "\\" in raw_name
        or raw_name.startswith("/")
        or re.match(r"^[A-Za-z]:", raw_name)
    ):
        return None
    stripped = raw_name.rstrip("/") if directory else raw_name
    if not stripped:
        return None
    path = PurePosixPath(stripped)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != stripped
    ):
        return None
    return path.as_posix()


def _zip_archive_members(payload: bytes) -> dict[str, bytes] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            if not infos:
                return None
            members: dict[str, bytes] = {}
            seen: set[str] = set()
            for info in infos:
                normalized = _normalized_archive_name(
                    info.filename,
                    directory=info.is_dir(),
                )
                if normalized is None or normalized in seen:
                    return None
                seen.add(normalized)
                mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(mode)
                if file_type == stat.S_IFLNK:
                    return None
                if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    return None
                if info.is_dir():
                    if file_type not in {0, stat.S_IFDIR}:
                        return None
                    continue
                if file_type == stat.S_IFDIR or info.flag_bits & 0x1:
                    return None
                members[normalized] = archive.read(info)
            return members
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        return None


def _tar_archive_members(payload: bytes) -> dict[str, bytes] | None:
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:*") as archive:
            archive_members = archive.getmembers()
            if not archive_members:
                return None
            members: dict[str, bytes] = {}
            seen: set[str] = set()
            for member in archive_members:
                normalized = _normalized_archive_name(
                    member.name,
                    directory=member.isdir(),
                )
                if normalized is None or normalized in seen:
                    return None
                seen.add(normalized)
                if member.issym() or member.islnk():
                    return None
                if member.isdir():
                    continue
                if not member.isfile():
                    return None
                extracted = archive.extractfile(member)
                if extracted is None:
                    return None
                members[normalized] = extracted.read()
            return members
    except (EOFError, OSError, RuntimeError, ValueError, tarfile.TarError):
        return None


def _contains_sdist_build_metadata(names: Sequence[str]) -> bool:
    return any(
        PurePosixPath(name).name in {"pyproject.toml", "setup.py", "setup.cfg"}
        for name in names
    )


def _normalized_package_source(
    members: Mapping[str, bytes],
) -> tuple[tuple[dict[str, object], ...], dict[str, bytes]] | None:
    """Normalize the sole ``tabu_lab`` package root inside an archive."""

    package_roots: set[tuple[str, ...]] = set()
    normalized_payloads: dict[str, bytes] = {}
    for archive_name, payload in members.items():
        parts = PurePosixPath(archive_name).parts
        positions = [index for index, part in enumerate(parts) if part == "tabu_lab"]
        if not positions:
            continue
        if len(positions) != 1:
            return None
        position = positions[0]
        relative_parts = parts[position + 1 :]
        if not relative_parts:
            return None
        if "__pycache__" in relative_parts or PurePosixPath(*relative_parts).suffix in {
            ".pyc",
            ".pyo",
        }:
            continue
        package_roots.add(parts[: position + 1])
        relative = PurePosixPath(*relative_parts).as_posix()
        if relative in normalized_payloads:
            return None
        normalized_payloads[relative] = payload
    if len(package_roots) != 1 or "__init__.py" not in normalized_payloads:
        return None
    manifest = tuple(
        {
            "path": relative,
            "sha256": _sha256_bytes(normalized_payloads[relative]),
            "size": len(normalized_payloads[relative]),
        }
        for relative in sorted(normalized_payloads)
    )
    return manifest, normalized_payloads


def _distribution_archive_source(
    uri: str,
    payload: bytes,
) -> tuple[tuple[dict[str, object], ...], dict[str, bytes]] | None:
    lowered = urlsplit(uri).path.lower()
    if lowered.endswith((".whl", ".zip")):
        members = _zip_archive_members(payload)
    elif lowered.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
        members = _tar_archive_members(payload)
    else:
        return None
    if members is None:
        return None
    names = tuple(members)
    if lowered.endswith(".whl"):
        if not any(name.endswith(".dist-info/WHEEL") for name in names) or not any(
            name.endswith(".dist-info/METADATA") for name in names
        ):
            return None
    elif not _contains_sdist_build_metadata(names):
        return None
    return _normalized_package_source(members)


def _live_package_source(
    source_root: str | Path | None,
) -> tuple[tuple[dict[str, object], ...], dict[str, bytes]] | None:
    if source_root is None:
        return None
    requested_root = Path(source_root)
    if requested_root.is_symlink() or not requested_root.is_dir():
        return None
    root = requested_root.resolve()
    payloads: dict[str, bytes] = {}
    try:
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root)
            if "__pycache__" in relative.parts or candidate.suffix in {".pyc", ".pyo"}:
                continue
            if candidate.is_symlink():
                return None
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                return None
            relative_name = relative.as_posix()
            if relative_name in payloads:
                return None
            payloads[relative_name] = candidate.read_bytes()
    except OSError:
        return None
    if not payloads or "__init__.py" not in payloads:
        return None
    manifest = tuple(
        {
            "path": relative,
            "sha256": _sha256_bytes(payloads[relative]),
            "size": len(payloads[relative]),
        }
        for relative in sorted(payloads)
    )
    return manifest, payloads


def _installed_source_tree_hash(files: Sequence[Mapping[str, object]]) -> str:
    return canonical_hash(
        {
            "schema_version": "tabu.source-tree-preimage.v1",
            "mode": "installed_package",
            "root_label": "tabu_lab_package",
            "files": files,
        }
    )


def git_source_identity(
    repository: str | Path,
    *,
    preregistration: str | Path | None,
    source_tree_hash: str,
    lock_hash: str | None,
    request_formal: bool,
    reviewed: bool,
    source_files: Sequence[Mapping[str, object]] | None = None,
) -> SourceIdentity:
    """Resolve a scoped Git source boundary without serializing local paths."""

    root = Path(repository).resolve()
    reasons: list[str] = []
    commit: str | None = None
    tree_oid: str | None = None
    repository_uri: str | None = None
    repository_subdirectory: str | None = None
    remote_ref: str | None = None
    preregistration_hash: str | None = None
    source_scope_clean = False

    try:
        top_level = Path(_git(root, "rev-parse", "--show-toplevel").stdout.decode().strip())
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return SourceIdentity(
            source_kind="local",
            issuance_status="local_unissued",
            source_tree_hash=source_tree_hash,
            lock_hash=lock_hash,
            reasons=("not_a_git_repository",),
        )
    git_root = top_level.resolve()
    try:
        scope = root.relative_to(git_root)
    except ValueError:
        reasons.append("source_scope_outside_repository")
        scope = Path(".")
    repository_subdirectory = "." if scope == Path(".") else scope.as_posix()
    try:
        commit = _git(git_root, "rev-parse", "HEAD").stdout.decode().strip()
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        reasons.append("git_head_unavailable")
    try:
        tree_expression = (
            "HEAD^{tree}"
            if repository_subdirectory == "."
            else f"HEAD:{repository_subdirectory}"
        )
        tree_oid = _git(git_root, "rev-parse", tree_expression).stdout.decode().strip()
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        reasons.append("source_scope_not_committed")
    try:
        status = _git(
            git_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            repository_subdirectory,
        ).stdout
        if status.strip():
            reasons.append("git_source_scope_not_clean")
        else:
            source_scope_clean = True
    except subprocess.CalledProcessError:
        reasons.append("git_status_unavailable")

    if request_formal and tree_oid is not None and source_scope_clean:
        reasons.extend(
            _verify_source_files_at_head(
                git_root=git_root,
                source_root=root,
                repository_subdirectory=repository_subdirectory,
                source_files=source_files,
            )
        )

    if preregistration is None:
        reasons.append("preregistration_not_provided")
    else:
        preregistration_path = Path(preregistration).resolve()
        try:
            relative = preregistration_path.relative_to(root).as_posix()
        except ValueError:
            reasons.append("preregistration_outside_repository")
        else:
            if not preregistration_path.is_file():
                reasons.append("preregistration_missing")
            else:
                preregistration_hash = _sha256_bytes(preregistration_path.read_bytes())
                try:
                    git_relative = _scoped_git_path(repository_subdirectory, relative)
                    committed = _git(git_root, "show", f"HEAD:{git_relative}").stdout
                except subprocess.CalledProcessError:
                    reasons.append("preregistration_not_committed")
                else:
                    if _sha256_bytes(committed) != preregistration_hash:
                        reasons.append("preregistration_differs_from_head")

    try:
        upstream = _git(
            git_root, "rev-parse", "--symbolic-full-name", "@{upstream}"
        ).stdout.decode().strip()
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        reasons.append("remote_tracking_ref_missing")
    else:
        if upstream.startswith("refs/remotes/"):
            remote_ref = upstream
            remote_parts = upstream.removeprefix("refs/remotes/").split("/", 1)
            remote_name = remote_parts[0]
            try:
                remote_raw = _git(
                    git_root, "config", "--get", f"remote.{remote_name}.url"
                ).stdout
                repository_uri = _public_repository_uri(remote_raw.decode())
            except (subprocess.CalledProcessError, UnicodeDecodeError, ValueError):
                repository_uri = None
            if repository_uri is None:
                reasons.append("public_remote_uri_unavailable")
            if len(remote_parts) != 2 or not remote_parts[1]:
                reasons.append("remote_tracking_ref_invalid")
            elif request_formal:
                branch_ref = f"refs/heads/{remote_parts[1]}"
                try:
                    remote_oid = _remote_ref_oid(git_root, remote_name, branch_ref)
                    tracking_oid = _git(git_root, "rev-parse", upstream).stdout.decode().strip()
                except (
                    OSError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                    UnicodeDecodeError,
                ):
                    remote_oid = None
                    tracking_oid = None
                if remote_oid is None:
                    reasons.append("remote_ref_not_retrievable")
                elif tracking_oid != remote_oid:
                    reasons.append("remote_tracking_ref_not_current")
            if commit is not None and (
                not request_formal or "remote_ref_not_retrievable" not in reasons
            ):
                reachable = _git(
                    git_root,
                    "merge-base",
                    "--is-ancestor",
                    commit,
                    upstream,
                    check=False,
                )
                if reachable.returncode != 0:
                    reasons.append("commit_not_reachable_from_remote_ref")
        else:
            reasons.append("remote_tracking_ref_invalid")

    if lock_hash is None:
        reasons.append("dependency_lock_hash_missing")
    if not request_formal:
        reasons.append("formal_issuance_not_requested")
    if not reviewed:
        reasons.append("source_review_not_attested")

    normalized_reasons = tuple(dict.fromkeys(reasons))
    formal = not normalized_reasons
    return SourceIdentity(
        source_kind="git",
        issuance_status="formal" if formal else "local_unissued",
        reviewed=reviewed if formal else False,
        repository_uri=repository_uri,
        repository_subdirectory=repository_subdirectory,
        commit=commit,
        remote_ref=remote_ref,
        git_tree_oid=tree_oid,
        source_tree_hash=source_tree_hash,
        preregistration_blob_hash=preregistration_hash,
        lock_hash=lock_hash,
        reasons=() if formal else normalized_reasons,
    )


def distribution_source_identity(
    *,
    uri: str,
    sha256: str,
    lock_hash: str,
    reviewed: bool,
    retrieved_distribution: bytes | str | Path | None = None,
    retrieved_lock: bytes | str | Path | None = None,
    source_tree_hash: str | None = None,
    live_source_root: str | Path | None = None,
) -> SourceIdentity:
    """Verify distribution, lock, and live installed-package bytes.

    Retrieval locations are verification inputs only and are never serialized. The
    public identity retains the immutable HTTPS URI and the digests of bytes that were
    actually read by this process. Formal issuance additionally requires the sole
    ``tabu_lab`` package in the wheel/sdist to match the live installed source root
    byte-for-byte and to produce the caller-supplied ``source_tree_hash``.
    """

    public_uri = _public_distribution_uri(uri)
    reasons: list[str] = []
    if public_uri is None:
        reasons.append("immutable_public_distribution_uri_required")
    distribution_bytes = _verification_bytes(retrieved_distribution)
    actual_distribution_sha256 = (
        _sha256_bytes(distribution_bytes) if distribution_bytes is not None else None
    )
    if distribution_bytes is None:
        reasons.append("distribution_bytes_not_verified")
    else:
        if actual_distribution_sha256 != sha256:
            reasons.append("distribution_digest_mismatch")
    archive_source = (
        _distribution_archive_source(public_uri, distribution_bytes)
        if public_uri is not None and distribution_bytes is not None
        else None
    )
    if distribution_bytes is not None and archive_source is None:
        reasons.append("distribution_archive_invalid")

    live_source = _live_package_source(live_source_root)
    if live_source is None:
        reasons.append("live_source_root_not_verified")
    elif archive_source is not None:
        archive_manifest, archive_payloads = archive_source
        live_manifest, live_payloads = live_source
        if archive_manifest != live_manifest or archive_payloads != live_payloads:
            reasons.append("distribution_source_manifest_mismatch")

    if source_tree_hash is None:
        reasons.append("source_tree_hash_missing")
    elif live_source is not None:
        live_manifest, _ = live_source
        if source_tree_hash != _installed_source_tree_hash(live_manifest):
            reasons.append("source_tree_hash_mismatch")
    lock_bytes = _verification_bytes(retrieved_lock)
    actual_lock_hash = _sha256_bytes(lock_bytes) if lock_bytes is not None else None
    if lock_bytes is None:
        reasons.append("dependency_lock_bytes_not_verified")
    elif actual_lock_hash != lock_hash:
        reasons.append("dependency_lock_digest_mismatch")
    if not reviewed:
        reasons.append("source_review_not_attested")
    normalized_reasons = tuple(dict.fromkeys(reasons))
    return SourceIdentity(
        source_kind="distribution",
        issuance_status="formal" if not normalized_reasons else "local_unissued",
        reviewed=reviewed if not normalized_reasons else False,
        source_tree_hash=source_tree_hash,
        distribution_uri=public_uri,
        distribution_sha256=actual_distribution_sha256 or sha256,
        lock_hash=actual_lock_hash or lock_hash,
        reasons=normalized_reasons,
    )


__all__ = [
    "SourceIdentity",
    "distribution_source_identity",
    "git_source_identity",
]
