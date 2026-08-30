from __future__ import annotations

import hashlib
import io
import stat
import subprocess
import tarfile
import warnings
import zipfile
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

import tabu_lab.evidence.source_identity as source_identity_module
from tabu_lab.contracts import canonical_hash
from tabu_lab.evidence.source_identity import (
    SourceIdentity,
    distribution_source_identity,
    git_source_identity,
    git_source_tree_hash,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _reviewed_repository(tmp_path: Path) -> tuple[Path, Path, str, str]:
    repository = tmp_path / "repo"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "TabU Test")
    _git(repository, "config", "user.email", "tabu@example.test")
    preregistration = repository / "experiments" / "F0" / "preregistration.yaml"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("schema_version: tabu.fit-experiment.v1\n", encoding="utf-8")
    lock = repository / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    source = repository / "src" / "tabu_lab" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = 'test'\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "reviewed source")
    _git(repository, "remote", "add", "origin", "https://example.test/wehub/tabu-lab.git")
    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repository, "branch", "--set-upstream-to=origin/main", "main")
    tree_hash = git_source_tree_hash(_source_files(repository))
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    return repository, preregistration, tree_hash, lock_hash


def _source_files(repository: Path) -> tuple[dict[str, object], ...]:
    paths = ("src/tabu_lab/__init__.py", "uv.lock")
    return tuple(
        {
            "path": relative,
            "sha256": hashlib.sha256((repository / relative).read_bytes()).hexdigest(),
            "size": (repository / relative).stat().st_size,
        }
        for relative in paths
    )


def test_reviewed_clean_remote_git_source_is_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, preregistration, tree_hash, lock_hash = _reviewed_repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: commit)

    identity = git_source_identity(
        repository,
        preregistration=preregistration,
        source_tree_hash=tree_hash,
        lock_hash=lock_hash,
        request_formal=True,
        reviewed=True,
        source_files=_source_files(repository),
    )

    assert identity.issuance_status == "formal"
    assert identity.repository_uri == "https://example.test/wehub/tabu-lab.git"
    assert identity.repository_subdirectory == "."
    assert identity.remote_ref == "refs/remotes/origin/main"
    assert identity.preregistration_blob_hash == hashlib.sha256(
        preregistration.read_bytes()
    ).hexdigest()
    assert str(tmp_path) not in identity.model_dump_json()


def test_unretrievable_remote_cannot_issue_formal_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, preregistration, tree_hash, lock_hash = _reviewed_repository(tmp_path)
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: None)

    identity = git_source_identity(
        repository,
        preregistration=preregistration,
        source_tree_hash=tree_hash,
        lock_hash=lock_hash,
        request_formal=True,
        reviewed=True,
        source_files=_source_files(repository),
    )

    assert identity.issuance_status == "local_unissued"
    assert "remote_ref_not_retrievable" in identity.reasons


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("dirty", "git_source_scope_not_clean"),
        ("untracked", "git_source_scope_not_clean"),
        ("preregistration", "preregistration_differs_from_head"),
    ),
)
def test_dirty_untracked_or_uncommitted_preregistration_is_local_unissued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    reason: str,
) -> None:
    repository, preregistration, tree_hash, lock_hash = _reviewed_repository(tmp_path)
    commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: commit)
    if mutation == "dirty":
        (repository / "src/tabu_lab/__init__.py").write_text("dirty = True\n", encoding="utf-8")
    elif mutation == "untracked":
        (repository / "untracked.txt").write_text("not reviewed\n", encoding="utf-8")
    else:
        preregistration.write_text("schema_version: changed\n", encoding="utf-8")

    identity = git_source_identity(
        repository,
        preregistration=preregistration,
        source_tree_hash=tree_hash,
        lock_hash=lock_hash,
        request_formal=True,
        reviewed=True,
        source_files=_source_files(repository),
    )

    assert identity.issuance_status == "local_unissued"
    assert reason in identity.reasons
    assert not identity.reviewed


def test_nested_source_scope_ignores_unrelated_parent_dirt_but_binds_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "workspace"
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "TabU Test")
    _git(repository, "config", "user.email", "tabu@example.test")
    scope = repository / "projects" / "TabU" / "tabu-lab"
    preregistration = scope / "experiments" / "F0" / "preregistration.yaml"
    preregistration.parent.mkdir(parents=True)
    preregistration.write_text("schema_version: tabu.fit-experiment.v1\n", encoding="utf-8")
    source = scope / "src" / "tabu_lab" / "__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = 'nested'\n", encoding="utf-8")
    lock = scope / "uv.lock"
    lock.write_text("version = 1\n", encoding="utf-8")
    sibling = repository / "unrelated.txt"
    sibling.write_text("reviewed sibling\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-m", "reviewed nested source")
    _git(repository, "remote", "add", "origin", "https://example.test/wehub/workspace.git")
    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(repository, "branch", "--set-upstream-to=origin/main", "main")
    commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: commit)

    sibling.write_text("dirty but outside TabU-lab\n", encoding="utf-8")
    (repository / "also-unrelated.tmp").write_text("untracked sibling\n", encoding="utf-8")
    files = _source_files(scope)
    identity = git_source_identity(
        scope,
        preregistration=preregistration,
        source_tree_hash=git_source_tree_hash(files),
        lock_hash=hashlib.sha256(lock.read_bytes()).hexdigest(),
        request_formal=True,
        reviewed=True,
        source_files=files,
    )

    assert identity.issuance_status == "formal"
    assert identity.repository_subdirectory == "projects/TabU/tabu-lab"
    assert identity.git_tree_oid == _git(
        repository, "rev-parse", "HEAD:projects/TabU/tabu-lab"
    )
    assert str(tmp_path) not in identity.model_dump_json()


def test_ignored_manifest_source_not_in_commit_is_local_unissued(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, preregistration, tree_hash, lock_hash = _reviewed_repository(tmp_path)
    (repository / ".gitignore").write_text("*.ignored.py\n", encoding="utf-8")
    _git(repository, "add", ".gitignore")
    _git(repository, "commit", "-m", "review ignore policy")
    _git(repository, "update-ref", "refs/remotes/origin/main", "HEAD")
    commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: commit)
    ignored = repository / "src" / "tabu_lab" / "shadow.ignored.py"
    ignored.write_text("not retrievable\n", encoding="utf-8")
    files = (*_source_files(repository), {
        "path": "src/tabu_lab/shadow.ignored.py",
        "sha256": hashlib.sha256(ignored.read_bytes()).hexdigest(),
        "size": ignored.stat().st_size,
    })

    identity = git_source_identity(
        repository,
        preregistration=preregistration,
        source_tree_hash=tree_hash,
        lock_hash=lock_hash,
        request_formal=True,
        reviewed=True,
        source_files=files,
    )

    assert identity.issuance_status == "local_unissued"
    assert "source_manifest_file_not_committed" in identity.reasons


def _wheel_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("tabu_lab/__init__.py", "__version__ = 'test'\n")
        archive.writestr(
            "tabu_lab-0.1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(
            "tabu_lab-0.1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: tabu-lab\nVersion: 0.1.0\n",
        )
    return buffer.getvalue()


def _installed_package(
    tmp_path: Path,
    files: Mapping[str, bytes] | None = None,
) -> Path:
    package = tmp_path / "site-packages" / "tabu_lab"
    for relative, payload in (files or {"__init__.py": b"__version__ = 'test'\n"}).items():
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return package


def _installed_tree_hash(package: Path) -> str:
    files = tuple(
        {
            "path": path.relative_to(package).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(
            candidate
            for candidate in package.rglob("*")
            if candidate.is_file()
            and "__pycache__" not in candidate.parts
            and candidate.suffix not in {".pyc", ".pyo"}
        )
    )
    return canonical_hash(
        {
            "schema_version": "tabu.source-tree-preimage.v1",
            "mode": "installed_package",
            "root_label": "tabu_lab_package",
            "files": files,
        }
    )


def test_formal_distribution_requires_verified_archive_lock_and_public_uri(
    tmp_path: Path,
) -> None:
    wheel = _wheel_bytes()
    lock = b"version = 1\n"
    package = _installed_package(tmp_path)
    tree_hash = _installed_tree_hash(package)
    formal = distribution_source_identity(
        uri="https://huggingface.co/wehub/tabu-lab/resolve/012345/package.whl",
        sha256=hashlib.sha256(wheel).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=wheel,
        retrieved_lock=lock,
        source_tree_hash=tree_hash,
        live_source_root=package,
    )
    local = distribution_source_identity(
        uri="/Users/alice/private/package.whl",
        sha256="a" * 64,
        lock_hash="b" * 64,
        reviewed=False,
    )

    assert formal.issuance_status == "formal"
    assert formal.source_tree_hash == tree_hash
    assert str(tmp_path) not in formal.model_dump_json()
    assert local.issuance_status == "local_unissued"
    assert "immutable_public_distribution_uri_required" in local.reasons
    assert "distribution_bytes_not_verified" in local.reasons
    assert "dependency_lock_bytes_not_verified" in local.reasons


def test_distribution_uri_alone_or_mismatched_retrieval_cannot_be_formal() -> None:
    wheel = _wheel_bytes()
    lock = b"version = 1\n"
    uri_only = distribution_source_identity(
        uri="https://example.test/releases/tabu_lab-0.1.0-py3-none-any.whl",
        sha256=hashlib.sha256(wheel).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
    )
    digest_mismatch = distribution_source_identity(
        uri="https://example.test/releases/tabu_lab-0.1.0-py3-none-any.whl",
        sha256="a" * 64,
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=wheel,
        retrieved_lock=lock,
    )
    unsuitable_uri = distribution_source_identity(
        uri="https://example.test/releases/package.bin",
        sha256=hashlib.sha256(wheel).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=wheel,
        retrieved_lock=lock,
    )

    assert uri_only.issuance_status == "local_unissued"
    assert "distribution_bytes_not_verified" in uri_only.reasons
    assert digest_mismatch.issuance_status == "local_unissued"
    assert "distribution_digest_mismatch" in digest_mismatch.reasons
    assert unsuitable_uri.issuance_status == "local_unissued"
    assert "immutable_public_distribution_uri_required" in unsuitable_uri.reasons


def test_distribution_requires_live_package_parity_and_expected_tree_hash(
    tmp_path: Path,
) -> None:
    wheel = _wheel_bytes()
    lock = b"version = 1\n"
    package = _installed_package(
        tmp_path,
        {"__init__.py": b"__version__ = 'drifted'\n"},
    )
    common = {
        "uri": "https://example.test/releases/tabu_lab-0.1.0-py3-none-any.whl",
        "sha256": hashlib.sha256(wheel).hexdigest(),
        "lock_hash": hashlib.sha256(lock).hexdigest(),
        "reviewed": True,
        "retrieved_distribution": wheel,
        "retrieved_lock": lock,
    }

    missing_binding = distribution_source_identity(**common)
    content_drift = distribution_source_identity(
        **common,
        source_tree_hash=_installed_tree_hash(package),
        live_source_root=package,
    )
    package.joinpath("__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    package.joinpath("unreleased.py").write_text("PRIVATE = True\n", encoding="utf-8")
    file_set_drift = distribution_source_identity(
        **common,
        source_tree_hash=_installed_tree_hash(package),
        live_source_root=package,
    )
    package.joinpath("unreleased.py").unlink()
    hash_drift = distribution_source_identity(
        **common,
        source_tree_hash="f" * 64,
        live_source_root=package,
    )

    assert missing_binding.issuance_status == "local_unissued"
    assert "live_source_root_not_verified" in missing_binding.reasons
    assert "source_tree_hash_missing" in missing_binding.reasons
    assert content_drift.issuance_status == "local_unissued"
    assert "distribution_source_manifest_mismatch" in content_drift.reasons
    assert file_set_drift.issuance_status == "local_unissued"
    assert "distribution_source_manifest_mismatch" in file_set_drift.reasons
    assert hash_drift.issuance_status == "local_unissued"
    assert "source_tree_hash_mismatch" in hash_drift.reasons


def _sdist_bytes(*, symlink: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        files = {
            "tabu-lab-0.1.0/pyproject.toml": b"[build-system]\nrequires = []\n",
            "tabu-lab-0.1.0/src/tabu_lab/__init__.py": b"__version__ = 'test'\n",
            "tabu-lab-0.1.0/src/tabu_lab/module.py": b"VALUE = 1\n",
        }
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        if symlink:
            member = tarfile.TarInfo("tabu-lab-0.1.0/src/tabu_lab/linked.py")
            member.type = tarfile.SYMTYPE
            member.linkname = "module.py"
            archive.addfile(member)
    return buffer.getvalue()


def test_formal_sdist_normalizes_src_layout_against_installed_package(
    tmp_path: Path,
) -> None:
    sdist = _sdist_bytes()
    lock = b"version = 1\n"
    package = _installed_package(
        tmp_path,
        {
            "__init__.py": b"__version__ = 'test'\n",
            "module.py": b"VALUE = 1\n",
        },
    )

    identity = distribution_source_identity(
        uri="https://example.test/releases/tabu-lab-0.1.0.tar.gz",
        sha256=hashlib.sha256(sdist).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=sdist,
        retrieved_lock=lock,
        source_tree_hash=_installed_tree_hash(package),
        live_source_root=package,
    )

    assert identity.issuance_status == "formal"


def _malformed_wheel(kind: str) -> bytes:
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("tabu_lab/__init__.py", "__version__ = 'test'\n")
            archive.writestr(
                "tabu_lab-0.1.0.dist-info/WHEEL",
                "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            archive.writestr(
                "tabu_lab-0.1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: tabu-lab\nVersion: 0.1.0\n",
            )
            if kind == "unsafe":
                archive.writestr("../escape.py", "PRIVATE = True\n")
            elif kind == "duplicate":
                archive.writestr("tabu_lab/__init__.py", "__version__ = 'test'\n")
            elif kind == "symlink":
                member = zipfile.ZipInfo("tabu_lab/linked.py")
                member.create_system = 3
                member.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(member, "__init__.py")
            else:  # pragma: no cover - guarded by the parametrization below
                raise AssertionError(kind)
    return buffer.getvalue()


@pytest.mark.parametrize("archive_kind", ("unsafe", "duplicate", "symlink", "tar_symlink"))
def test_distribution_rejects_unsafe_duplicate_or_link_archive_members(
    tmp_path: Path,
    archive_kind: str,
) -> None:
    distribution = (
        _sdist_bytes(symlink=True)
        if archive_kind == "tar_symlink"
        else _malformed_wheel(archive_kind)
    )
    suffix = "tar.gz" if archive_kind == "tar_symlink" else "whl"
    lock = b"version = 1\n"
    package = _installed_package(tmp_path)

    identity = distribution_source_identity(
        uri=f"https://example.test/releases/tabu-lab-0.1.0.{suffix}",
        sha256=hashlib.sha256(distribution).hexdigest(),
        lock_hash=hashlib.sha256(lock).hexdigest(),
        reviewed=True,
        retrieved_distribution=distribution,
        retrieved_lock=lock,
        source_tree_hash=_installed_tree_hash(package),
        live_source_root=package,
    )

    assert identity.issuance_status == "local_unissued"
    assert "distribution_archive_invalid" in identity.reasons


def test_schema_rejects_manual_formal_promotion_without_reviewed_bindings() -> None:
    with pytest.raises(ValidationError, match="reviewed=True"):
        SourceIdentity(
            source_kind="git",
            issuance_status="formal",
            reviewed=False,
        )


@pytest.mark.parametrize(
    ("tree_hash", "lock_hash", "reason"),
    (
        ("a" * 64, None, "source_tree_hash_mismatch"),
        (None, "b" * 64, "dependency_lock_digest_mismatch"),
    ),
)
def test_formal_git_identity_rejects_arbitrary_tree_or_lock_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree_hash: str | None,
    lock_hash: str | None,
    reason: str,
) -> None:
    repository, preregistration, expected_tree_hash, expected_lock_hash = (
        _reviewed_repository(tmp_path)
    )
    commit = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(source_identity_module, "_remote_ref_oid", lambda *args: commit)

    identity = git_source_identity(
        repository,
        preregistration=preregistration,
        source_tree_hash=tree_hash or expected_tree_hash,
        lock_hash=lock_hash or expected_lock_hash,
        request_formal=True,
        reviewed=True,
        source_files=_source_files(repository),
    )

    assert identity.issuance_status == "local_unissued"
    assert reason in identity.reasons


def test_source_identity_has_deterministic_content_hash() -> None:
    identity = SourceIdentity(
        source_kind="local",
        issuance_status="local_unissued",
        reasons=("formal_issuance_not_requested",),
    )

    assert identity.content_hash == identity.schema_hash
    assert len(identity.content_hash) == 64
