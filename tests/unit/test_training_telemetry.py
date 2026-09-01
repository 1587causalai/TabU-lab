from __future__ import annotations

import json
from pathlib import Path

from tabu_lab.evolution import EvolutionRepository, ProgramLane, file_sha256, run_program
from tabu_lab.training.telemetry import load_telemetry_protocol

ROOT = Path(__file__).resolve().parents[2]
BASE = "tabu.pretraining.query-base@1.0.0"


def test_checked_in_telemetry_protocol_is_versioned_and_content_addressed() -> None:
    protocol = load_telemetry_protocol(
        ROOT / "specs/telemetry/training-observer-1.0.0.yaml"
    )

    assert protocol.ref == "tabu.telemetry.pretraining-core@1.0.0"
    assert protocol.step_interval == 1
    assert len(protocol.protocol_hash) == 64
    assert "baseline_relative_skill" in protocol.metric_groups


def test_local_telemetry_is_passive_and_checkpoint_identical(tmp_path: Path) -> None:
    repository = EvolutionRepository.load(ROOT)
    unobserved = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "unobserved",
    )
    observed = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "observed",
        telemetry_mode="local",
    )

    assert observed.telemetry is not None
    assert observed.telemetry.status == "ok"
    assert file_sha256(observed.checkpoint) == file_sha256(unobserved.checkpoint)
    assert file_sha256(observed.checkpoint_sidecar) == file_sha256(
        unobserved.checkpoint_sidecar
    )
    assert observed.receipt.receipt_hash == unobserved.receipt.receipt_hash

    records = [
        json.loads(line)
        for line in observed.telemetry.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["step"] for record in records] == [1, 2]
    assert all(record["schema_version"] == "tabu.training-metric-record.v1" for record in records)
    assert all("loss/numeric_skill_vs_context_mean" in record for record in records)
    assert all("support/coverage" in record for record in records)
    forbidden = {"target_values", "truth_sidecar", "y_true", "supervision"}
    assert not any(key.lower() in forbidden for record in records for key in record)

    receipt = json.loads(observed.telemetry.receipt_path.read_text(encoding="utf-8"))
    assert receipt["observer_semantics"] == "non_authoritative_projection"
    assert receipt["run_identity_hash"] == observed.receipt.run_identity_hash
    assert receipt["metrics_sha256"] == file_sha256(observed.telemetry.metrics_path)
