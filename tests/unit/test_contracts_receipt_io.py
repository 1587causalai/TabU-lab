from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier

import pytest

from tabu_lab.evidence import (
    EnvironmentDisclosure,
    Receipt,
    ReceiptIntegrityError,
    ReceiptStatus,
    RunBundle,
    RunIdentity,
    canonical_json,
    read_receipt,
    write_receipt,
)


def _receipt() -> Receipt:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    identity = RunIdentity.create(
        spec_hash="1" * 64,
        code_hash="2" * 64,
        data_hash="3" * 64,
        split_hash="4" * 64,
        compiler_hash="5" * 64,
        semantic_config_hash="6" * 64,
        execution_config_hash="7" * 64,
        training_config_hash="8" * 64,
        seeds={"data_order": 17, "model_init": 23},
    )
    bundle = RunBundle(
        identity=identity,
        created_at=now,
        model_id="model-a",
        dataset_id="dataset-a",
        fit_partition="train",
        environment=EnvironmentDisclosure(
            environment_hash="9" * 64,
            host_class="workstation",
            operating_system="linux",
            device="cpu",
        ),
    )
    return Receipt.from_run_bundle(
        bundle,
        receipt_id="receipt-atomic-1",
        status=ReceiptStatus.SUCCEEDED,
        created_at=now,
        completed_at=now,
    )


def test_atomic_receipt_roundtrip_is_canonical_and_self_verifying(tmp_path) -> None:
    receipt = _receipt()
    path = tmp_path / "receipt.json"

    written_hash = write_receipt(path, receipt)
    loaded = read_receipt(path)
    parsed = json.loads(path.read_bytes())

    assert written_hash == receipt.receipt_hash
    assert loaded == receipt
    assert loaded.receipt_hash == receipt.receipt_hash
    assert path.read_text(encoding="utf-8") == canonical_json(parsed) + "\n"


def test_stale_temp_does_not_affect_canonical_target(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    stale = tmp_path / ".receipt.json.interrupted.tmp"
    stale.write_text("partial", encoding="utf-8")

    write_receipt(path, _receipt())

    assert read_receipt(path) == _receipt()
    assert stale.read_text(encoding="utf-8") == "partial"


def test_immutable_receipt_refuses_overwrite(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    receipt = _receipt()
    write_receipt(path, receipt)
    original = path.read_bytes()

    with pytest.raises(FileExistsError, match="already exists"):
        write_receipt(path, receipt.model_copy(update={"receipt_id": "replacement"}))

    assert path.read_bytes() == original


def test_tampered_receipt_fails_content_hash_validation(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    write_receipt(path, _receipt())
    payload = json.loads(path.read_bytes())
    payload["receipt"]["receipt_id"] = "tampered-but-schema-valid"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(ReceiptIntegrityError, match="receipt_hash"):
        read_receipt(path)


def test_noncooperating_target_created_during_publish_is_never_overwritten(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "receipt.json"
    original_link = os.link

    def inject_external_target(source, destination):  # type: ignore[no-untyped-def]
        Path(destination).write_bytes(b"external-writer")
        original_link(source, destination)

    monkeypatch.setattr(os, "link", inject_external_target)

    with pytest.raises(FileExistsError):
        write_receipt(path, _receipt())

    assert path.read_bytes() == b"external-writer"


def test_concurrent_publish_has_exactly_one_immutable_winner(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "receipt.json"
    first = _receipt()
    second = first.model_copy(update={"receipt_id": "receipt-atomic-2"})
    original_link = os.link
    barrier = Barrier(2)

    def synchronized_link(source, destination):  # type: ignore[no-untyped-def]
        barrier.wait(timeout=5)
        original_link(source, destination)

    def attempt(receipt: Receipt) -> tuple[str, str]:
        try:
            return "written", write_receipt(path, receipt)
        except FileExistsError:
            return "exists", receipt.receipt_hash

    monkeypatch.setattr(os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(attempt, (first, second)))

    assert sorted(status for status, _ in results) == ["exists", "written"]
    assert read_receipt(path) in (first, second)
    assert not tuple(tmp_path.glob("*.lock"))
