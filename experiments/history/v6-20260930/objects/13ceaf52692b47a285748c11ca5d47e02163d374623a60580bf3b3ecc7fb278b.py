"""Small, public-safe catalog records for the consolidated source tree."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts.canonical import canonical_hash, require_sha256
from tabu_lab.evidence.schemas import EvidenceSchema


class CatalogObjectKind(StrEnum):
    MODEL_CONTRACT = "model_contract"
    EVOLUTION_NODE = "evolution_node"
    COMPATIBILITY_EDGE = "compatibility_edge"
    PROGRAM_SNAPSHOT = "program_snapshot"


class CatalogEntry(EvidenceSchema):
    schema_version: Literal["tabu.catalog-entry.v1"] = "tabu.catalog-entry.v1"
    kind: CatalogObjectKind
    object_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
    source_path: str = Field(min_length=1)
    object_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    data: dict[str, JsonValue]

    @field_validator("source_path")
    @classmethod
    def _repository_relative_source(cls, value: str) -> str:
        if (
            value.startswith(("/", "\\", "../"))
            or "\\" in value
            or "/../" in value
            or value == ".."
        ):
            raise ValueError("catalog source_path must be repository-relative")
        return value.removeprefix("./")

    @field_validator("object_hash")
    @classmethod
    def _valid_content_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="object_hash")

    @model_validator(mode="after")
    def _content_identity_matches(self) -> CatalogEntry:
        if self.object_hash != canonical_hash(self.data):
            raise ValueError("catalog entry object_hash does not match data")
        if self.kind is CatalogObjectKind.MODEL_CONTRACT and (
            self.data.get("contract_id") != self.object_id
            or self.data.get("contract_version") != self.version
        ):
            raise ValueError("catalog entry wrapper identity does not match ModelSpec data")
        identity_fields = {
            CatalogObjectKind.EVOLUTION_NODE: ("node_id", "version"),
            CatalogObjectKind.COMPATIBILITY_EDGE: ("edge_id", "version"),
            CatalogObjectKind.PROGRAM_SNAPSHOT: ("program_id", "version"),
        }
        if self.kind in identity_fields:
            object_field, version_field = identity_fields[self.kind]
            if (
                self.data.get(object_field) != self.object_id
                or self.data.get(version_field) != self.version
            ):
                raise ValueError("catalog entry wrapper identity does not match manifest data")
        return self

    @property
    def entry_id(self) -> str:
        return f"{self.kind.value}:{self.object_id}@{self.version}"


class CatalogIndex(EvidenceSchema):
    schema_version: Literal["tabu.catalog-index.v1"] = "tabu.catalog-index.v1"
    entries: tuple[CatalogEntry, ...] = Field(min_length=1)
    source_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    formal_receipt_count: Literal[0] = 0
    accepted_claim_count: Literal[0] = 0
    claim_boundary: Literal["catalog projection; not evidence or claim acceptance"] = (
        "catalog projection; not evidence or claim acceptance"
    )

    @field_validator("source_tree_hash")
    @classmethod
    def _valid_source_tree_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source_tree_hash")

    @model_validator(mode="after")
    def _stable_unique_entries(self) -> CatalogIndex:
        ids = tuple(entry.entry_id for entry in self.entries)
        if ids != tuple(sorted(ids)):
            raise ValueError("catalog entries must be sorted by entry_id")
        if len(ids) != len(set(ids)):
            raise ValueError("catalog entry ids must be unique")
        expected = canonical_hash(tuple(entry.model_dump(mode="python") for entry in self.entries))
        if self.source_tree_hash != expected:
            raise ValueError("catalog source_tree_hash does not match entries")
        return self


__all__ = ["CatalogEntry", "CatalogIndex", "CatalogObjectKind"]
