from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabu_lab.evolution import (
    EvidenceStatus,
    EvolutionRepository,
    ProgramCheckpointEvaluationReceipt,
    ProgramCheckpointEvaluationRequest,
    ProgramLane,
    evaluate_program_checkpoint,
    file_sha256,
    load_program_evaluation_request,
    program_sidecar_path,
    run_program,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = "tabu.pretraining.query-base@1.0.0"


def test_checked_in_best_checkpoint_requests_freeze_a_new_shared_panel() -> None:
    base = load_program_evaluation_request(
        ROOT
        / "experiments/evolution/query-base-v3.1-best-step500-eval-1.0.0.yaml"
    )
    row = load_program_evaluation_request(
        ROOT
        / "experiments/evolution/query-row-v3.1-best-step1500-eval-1.0.0.yaml"
    )

    assert base.request_hash != row.request_hash
    assert base.root_seed == row.root_seed == 271828182
    assert base.worlds == row.worlds == 256
    assert base.address_prefix == row.address_prefix
    assert base.selection_panel_addresses_hash == row.selection_panel_addresses_hash
    assert base.checkpoint_step == 500
    assert row.checkpoint_step == 1500
    assert base.evidence_status is EvidenceStatus.LOCAL_UNISSUED


def test_checked_in_terminal_requests_forbid_metric_based_selection() -> None:
    base = load_program_evaluation_request(
        ROOT
        / "experiments/evolution/query-base-v3.1-terminal-step18500-eval-1.0.0.yaml"
    )
    row = load_program_evaluation_request(
        ROOT
        / "experiments/evolution/query-row-v3.1-terminal-step18500-eval-1.0.0.yaml"
    )

    assert base.request_hash != row.request_hash
    assert base.checkpoint_step == row.checkpoint_step == 18500
    assert base.address_prefix == row.address_prefix
    assert base.selection_panel_addresses_hash == row.selection_panel_addresses_hash
    assert base.selection_rule == row.selection_rule
    assert "no metric-based checkpoint selection" in base.selection_rule
    assert base.evaluation_protocol_ref == "tabu.eval.transfer-lanes-terminal@1.0.0"


def test_program_checkpoint_evaluation_is_independent_and_non_overwriting(
    tmp_path: Path,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    training = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "training",
    )
    resolved = repository.resolve(BASE)
    sidecar = program_sidecar_path(training.checkpoint)
    request = ProgramCheckpointEvaluationRequest(
        request_id="tabu.eval.test-base",
        version="1.0.0",
        claim_boundary="local unit-test evaluation only",
        program_ref=BASE,
        snapshot_hash=resolved.snapshot_hash,
        run_identity_hash=training.receipt.run_identity_hash,
        training_run_receipt_hash=training.receipt.receipt_hash,
        source_training_commit="0" * 40,
        source_repository_hash=repository.repository_hash,
        checkpoint_name=training.checkpoint.name,
        checkpoint_step=training.receipt.step,
        checkpoint_sha256=file_sha256(training.checkpoint),
        checkpoint_sidecar_sha256=file_sha256(sidecar),
        selection_receipt_hash="1" * 64,
        selection_panel_addresses_hash="2" * 64,
        selection_rule="lowest loss in the unit-test selection panel",
        evaluation_protocol_ref=resolved.slots["evaluation_protocol"].ref,
        root_seed=17,
        worlds=2,
        address_prefix="unit-independent-eval",
    )
    output = tmp_path / "evaluation.json"
    result = evaluate_program_checkpoint(
        repository,
        request=request,
        checkpoint=training.checkpoint,
        training_run_receipt=training.receipt_path,
        output=output,
        evaluation_source_revision="3" * 40,
        evaluation_source_archive_sha256="4" * 64,
    )

    assert output.is_file()
    assert result.receipt.evidence_status is EvidenceStatus.LOCAL_UNISSUED
    assert result.receipt.panel.worlds == 2
    assert len(result.receipt.metrics.per_world_loss) == 2
    assert result.receipt.metrics.scored_targets > 0
    assert result.receipt.metrics.abstained_targets == 0
    assert result.receipt.model_state_hash_before == result.receipt.model_state_hash_after
    replay = ProgramCheckpointEvaluationReceipt.model_validate(
        json.loads(output.read_text(encoding="utf-8"))
    )
    assert replay.receipt_hash == result.receipt.receipt_hash

    with pytest.raises(ValueError, match="refusing to overwrite"):
        evaluate_program_checkpoint(
            repository,
            request=request,
            checkpoint=training.checkpoint,
            training_run_receipt=training.receipt_path,
            output=output,
            evaluation_source_revision="3" * 40,
            evaluation_source_archive_sha256="4" * 64,
        )

    tampered = request.model_copy(update={"checkpoint_sha256": "5" * 64})
    with pytest.raises(ValueError, match="checkpoint hash"):
        evaluate_program_checkpoint(
            repository,
            request=tampered,
            checkpoint=training.checkpoint,
            training_run_receipt=training.receipt_path,
            output=tmp_path / "tampered.json",
            evaluation_source_revision="3" * 40,
            evaluation_source_archive_sha256="4" * 64,
        )
