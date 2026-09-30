"""Deterministic checked-in JSON Schema generation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .schemas import ClaimLedger, Preregistration, Receipt
from .source_identity import SourceIdentity

_SCHEMA_BASE = "https://research.wehub.us/schemas"
PUBLIC_SCHEMA_MODELS: dict[str, tuple[type[BaseModel], str]] = {
    "preregistration": (
        Preregistration,
        f"{_SCHEMA_BASE}/preregistration.schema.json",
    ),
    "receipt": (Receipt, f"{_SCHEMA_BASE}/receipt.schema.json"),
    "claim": (ClaimLedger, f"{_SCHEMA_BASE}/claim.schema.json"),
    "source-identity": (
        SourceIdentity,
        f"{_SCHEMA_BASE}/source-identity.schema.json",
    ),
}


def generate_public_schema(name: str) -> dict[str, Any]:
    try:
        model, schema_id = PUBLIC_SCHEMA_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"unknown public evidence schema: {name!r}") from exc
    generated = model.model_json_schema(mode="validation", by_alias=True)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": schema_id,
        **generated,
    }


__all__ = ["PUBLIC_SCHEMA_MODELS", "generate_public_schema"]
