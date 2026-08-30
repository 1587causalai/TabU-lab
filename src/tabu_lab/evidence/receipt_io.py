"""Crash-safe immutable receipt persistence with read-back verification."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from .canonical import canonical_json, require_sha256
from .schemas import EvidenceSchema, Receipt


class ReceiptIntegrityError(ValueError):
    """Raised when a receipt file is malformed, non-canonical, or tampered."""


class ReceiptEnvelope(EvidenceSchema):
    """Self-verifying disk envelope around the public Receipt payload."""

    schema_version: Literal["tabu.receipt-envelope.v1"] = "tabu.receipt-envelope.v1"
    receipt_hash: str
    receipt: Receipt

    @field_validator("receipt_hash")
    @classmethod
    def _valid_receipt_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="receipt_hash")

    @model_validator(mode="after")
    def _hash_matches_payload(self) -> ReceiptEnvelope:
        if self.receipt_hash != self.receipt.receipt_hash:
            raise ValueError("receipt_hash does not match the canonical Receipt payload")
        return self


def _canonical_envelope_bytes(envelope: ReceiptEnvelope) -> bytes:
    payload = envelope.model_dump(mode="python", by_alias=False)
    return (canonical_json(payload) + "\n").encode("utf-8")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_receipt(path: str | Path) -> Receipt:
    """Read and validate canonical JSON, strict schema, and embedded content hash."""

    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ReceiptIntegrityError("receipt path must be an existing regular file")
    try:
        raw = target.read_bytes()
        decoded = json.loads(raw)
        envelope = ReceiptEnvelope.model_validate(decoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReceiptIntegrityError(f"invalid receipt envelope: {exc}") from exc
    if raw != _canonical_envelope_bytes(envelope):
        raise ReceiptIntegrityError("receipt file is not canonical JSON")
    return envelope.receipt


def write_receipt(path: str | Path, receipt: Receipt) -> str:
    """Atomically publish one immutable receipt and verify it from disk.

    The target parent must already exist.  A same-directory temporary file is
    flushed and fsynced before an atomic hard-link publish.  Linking within the
    same directory is create-if-absent, so even a non-cooperating concurrent
    writer cannot be overwritten.
    """

    if not isinstance(receipt, Receipt):
        raise TypeError("write_receipt requires a validated Receipt")
    target = Path(path)
    parent = target.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"receipt parent directory does not exist: {parent}")
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"immutable receipt already exists: {target}")

    envelope = ReceiptEnvelope(
        receipt_hash=receipt.receipt_hash,
        receipt=receipt,
    )
    payload = _canonical_envelope_bytes(envelope)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.link(temporary_name, target)
        os.unlink(temporary_name)
        temporary_name = None
        _fsync_directory(parent)
    finally:
        if temporary_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)

    read_back = read_receipt(target)
    if read_back.receipt_hash != receipt.receipt_hash:
        raise ReceiptIntegrityError("receipt read-back hash does not match written payload")
    return receipt.receipt_hash


__all__ = [
    "ReceiptEnvelope",
    "ReceiptIntegrityError",
    "read_receipt",
    "write_receipt",
]
