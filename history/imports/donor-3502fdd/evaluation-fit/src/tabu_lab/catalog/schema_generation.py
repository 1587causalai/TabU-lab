"""Deterministic public JSON Schema generation for catalog contracts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from tabu_lab.verification.contracts import VerificationResult, VerificationSuite

from .models import (
    CatalogIndex,
    ClaimRecord,
    DatasetSnapshotSpec,
    ExperimentRecord,
    ModelArtifact,
    ReviewRecord,
    RunAttemptRecord,
    RunRecord,
)

_SCHEMA_BASE = "https://research.wehub.us/schemas"
PUBLIC_CATALOG_SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "catalog-index": CatalogIndex,
    "catalog-experiment": ExperimentRecord,
    "catalog-run": RunRecord,
    "catalog-run-attempt": RunAttemptRecord,
    "catalog-model-artifact": ModelArtifact,
    "catalog-dataset-snapshot": DatasetSnapshotSpec,
    "catalog-review": ReviewRecord,
    "catalog-claim-record": ClaimRecord,
    "verification-suite": VerificationSuite,
    "verification-result": VerificationResult,
}


def generate_catalog_schema(name: str) -> dict[str, Any]:
    try:
        model = PUBLIC_CATALOG_SCHEMA_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unknown public catalog schema: {name!r}") from exc
    generated = model.model_json_schema(mode="validation", by_alias=True)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"{_SCHEMA_BASE}/{name}.schema.json",
        **generated,
    }


__all__ = ["PUBLIC_CATALOG_SCHEMA_MODELS", "generate_catalog_schema"]
