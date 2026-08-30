"""Unit-test support for public-remote verification without public network I/O."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import tabu_lab.evidence.formal_authorization as formal_authorization


@pytest.fixture(autouse=True)
def _probe_marked_formal_remotes_via_real_local_bare_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route only explicitly marked test repositories to their real bare remote.

    Production never reads ``tabu.tests.bareRemote``.  This fixture substitutes
    the public-network probe while preserving actual Git object/ref retrieval in
    unit tests; the production probe has separate environment-sanitization tests.
    """

    production_probe = formal_authorization._public_remote_ref_oid

    def probe(repository: Path, repository_uri: str, branch_ref: str) -> str:
        marker = subprocess.run(
            ("git", "-C", str(repository), "config", "--get", "tabu.tests.bareRemote"),
            check=False,
            capture_output=True,
            text=True,
        )
        if marker.returncode != 0 or not marker.stdout.strip():
            return production_probe(repository, repository_uri, branch_ref)
        bare = Path(marker.stdout.strip()).resolve()
        result = subprocess.run(
            (
                "git",
                "ls-remote",
                "--exit-code",
                f"file://{bare}",
                branch_ref,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            oid, ref = line.split("\t", 1)
            if ref == branch_ref and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", oid):
                return oid
        raise AssertionError("marked bare test remote did not return the requested ref")

    monkeypatch.setattr(formal_authorization, "_public_remote_ref_oid", probe)
