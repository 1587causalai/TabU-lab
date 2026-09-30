"""Deterministic, dependency-light canonical encoding for contract identities.

Hashes in this module identify semantic payloads.  They are deliberately not a
replacement for an artifact byte hash: tensors and arrays are represented by
their dtype, shape, and contiguous bytes so that identities do not depend on a
pretty-printer or Python container insertion order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch


class CanonicalizationError(ValueError):
    """Raised when a value has no safe deterministic canonical representation."""


def _canonical_float(value: float) -> float:
    if not math.isfinite(value):
        raise CanonicalizationError(
            "canonical payloads must be finite; NaN and infinity are forbidden"
        )
    # JSON has two encodings for zero on common runtimes.  Collapse them.
    return 0.0 if value == 0.0 else value


def _tensor_payload(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise CanonicalizationError("canonical tensors must be finite")
    if tensor.is_complex() and not bool(
        torch.isfinite(tensor.real).all() and torch.isfinite(tensor.imag).all()
    ):
        raise CanonicalizationError("canonical tensors must be finite")
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return {
        "__tensor__": {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "data_hex": raw.hex(),
        }
    }


def _array_payload(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if np.issubdtype(array.dtype, np.inexact) and not bool(np.isfinite(array).all()):
        raise CanonicalizationError("canonical arrays must be finite")
    return {
        "__ndarray__": {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_hex": array.tobytes(order="C").hex(),
        }
    }


def to_canonical_data(value: Any) -> Any:
    """Convert supported Python objects into a deterministic JSON value.

    Mapping keys must be strings.  Sets and arbitrary object ``repr`` values are
    rejected because they would make evidence identities runtime-dependent.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _canonical_float(value)
    if isinstance(value, Enum):
        return to_canonical_data(value.value)
    if isinstance(value, torch.Tensor):
        return _tensor_payload(value)
    if isinstance(value, np.ndarray):
        return _array_payload(value)
    if isinstance(value, np.generic):
        return to_canonical_data(value.item())
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        normalized = value
        if value.tzinfo is not None:
            normalized = value.astimezone(UTC)
        return normalized.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_canonical_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if not field.name.startswith("_")
        }
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_canonical_data(model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise CanonicalizationError("canonical mapping keys must be strings")
        return {
            key: to_canonical_data(value[key])
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_canonical_data(item) for item in value]
    raise CanonicalizationError(
        f"unsupported canonical payload type: {type(value).__module__}.{type(value).__qualname__}"
    )


def canonical_json(value: Any) -> str:
    """Return UTF-8 JSON with stable key ordering and no insignificant spaces."""

    return json.dumps(
        to_canonical_data(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


# The shorter name is the public vocabulary used by manifests and receipts.
canonical_hash = canonical_sha256


def require_sha256(value: str, *, field_name: str = "sha256") -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return normalized


__all__ = [
    "CanonicalizationError",
    "canonical_hash",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "require_sha256",
    "to_canonical_data",
]
