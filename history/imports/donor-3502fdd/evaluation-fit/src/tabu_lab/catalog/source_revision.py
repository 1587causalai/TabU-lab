"""Path-free, exact public source revisions for catalog projections."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, field_validator

from tabu_lab.contracts.canonical import require_sha256
from tabu_lab.evidence.schemas import EvidenceSchema

_GIT_OBJECT_PATTERN = r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$"


class CatalogSourceRevision(EvidenceSchema):
    """Reviewed GitHub revision that exactly binds one catalog source tree.

    The repository URI contains no file path or mutable branch. ``commit`` is a
    full Git object ID, and ``catalog_source_tree_hash`` binds the declaration to
    the canonical manifests indexed by :class:`CatalogIndex`.
    """

    schema_version: Literal["tabu.catalog-source-revision.v1"] = (
        "tabu.catalog-source-revision.v1"
    )
    repository_uri: str = Field(min_length=1)
    commit: str = Field(pattern=_GIT_OBJECT_PATTERN)
    catalog_source_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("repository_uri")
    @classmethod
    def _public_github_repository(cls, value: str) -> str:
        parsed = urlsplit(value.strip())
        if (
            parsed.scheme != "https"
            or parsed.hostname != "github.com"
            or parsed.netloc != "github.com"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("repository_uri must be a public HTTPS github.com repository")
        path = parsed.path.rstrip("/")
        if path.endswith(".git"):
            path = path[:-4]
        parts = PurePosixPath(path).parts
        if len(parts) != 3 or parts[0] != "/" or not parts[1] or not parts[2]:
            raise ValueError("repository_uri must identify exactly one GitHub repository")
        return urlunsplit(("https", "github.com", f"/{parts[1]}/{parts[2]}", "", ""))

    @field_validator("catalog_source_tree_hash")
    @classmethod
    def _source_tree_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="catalog_source_tree_hash")


__all__ = ["CatalogSourceRevision"]
