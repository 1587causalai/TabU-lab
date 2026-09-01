from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tabu_lab.contracts import canonical_hash
from tabu_lab.evolution import (
    EvidenceStatus,
    EvolutionRepository,
    ProgramLane,
    ProgramRunStatus,
    file_sha256,
    program_sidecar_path,
    read_program_checkpoint,
    run_program,
)
from tabu_lab.evolution.checkpoint import read_checkpoint_model_state
from tabu_lab.evolution.models import ProgramRunReceipt
from tabu_lab.evolution.runtime import _build_runtime_model, _objective, identity_state_projection

ROOT = Path(__file__).resolve().parents[2]
BASE = "tabu.pretraining.query-base@1.0.0"
ROW = "tabu.pretraining.query-row@1.0.0"
GENERATOR_VNEXT = "tabu.pretraining.query-base-generator-v3@1.1.0-exercise"


def test_versioned_objective_selects_context_standardized_truth_coordinates() -> None:
    repository = EvolutionRepository.load(ROOT)
    objective = _objective(
        repository.node("tabu.objectives.context-standardized-supervised-response@1.0.0")
    )

    assert objective.resume_config["numeric_target_coordinate"] == "context_standardized"


def test_legacy_objective_keeps_raw_truth_coordinates() -> None:
    repository = EvolutionRepository.load(ROOT)
    objective = _objective(repository.node("tabu.objectives.supervised-response@1.0.0"))

    assert objective.resume_config["numeric_target_coordinate"] == "raw"


@pytest.mark.parametrize(
    ("source_graph_ref", "target_graph_ref"),
    (
        ("tabu.graph.query.base@1.0.0", "tabu.graph.query.base@1.1.0"),
        ("tabu.graph.query.row@1.0.0", "tabu.graph.query.row@1.1.0"),
    ),
)
def test_v3_scale_projection_preserves_full_model_state(
    source_graph_ref: str,
    target_graph_ref: str,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    source = _build_runtime_model(repository.node(source_graph_ref), "cpu")
    target = _build_runtime_model(repository.node(target_graph_ref), "cpu")

    assert source.config.max_features == 64
    assert target.config.max_features == 1024
    source_state = source.state_dict()
    target_state = target.state_dict()
    projected = identity_state_projection(source_state, target_state)
    assert set(projected) == set(target_state)
    assert all(projected[name].shape == target_state[name].shape for name in projected)
    assert all(torch.equal(projected[name], source_state[name]) for name in projected)


def test_tiny_base_exact_resume_is_byte_identical_to_uninterrupted_run(
    tmp_path: Path,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    full = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "full",
    )
    interrupted = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "interrupted",
        max_updates_this_invocation=1,
    )
    resumed = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "resumed",
        resume_checkpoint=interrupted.checkpoint,
    )

    assert interrupted.receipt.status is ProgramRunStatus.INTERRUPTED
    assert full.receipt.status is ProgramRunStatus.COMPLETED
    assert resumed.receipt.status is ProgramRunStatus.COMPLETED
    assert (tmp_path / "full" / "checkpoint-step-00000001.safetensors").is_file()
    assert program_sidecar_path(
        tmp_path / "full" / "checkpoint-step-00000001.safetensors"
    ).is_file()
    assert full.receipt.evidence_status is EvidenceStatus.LOCAL_UNISSUED
    assert resumed.receipt.run_identity_hash == full.receipt.run_identity_hash
    assert resumed.receipt.receipt_hash == full.receipt.receipt_hash
    assert full.receipt.resolved_snapshot == repository.resolve(BASE)
    assert dict(full.receipt.run_identity.seeds) == {
        "episode": 2718,
        "model": 1729,
        "sampler": 31415,
    }
    assert full.receipt.execution_config["device"] == "cpu"
    assert full.receipt.run_identity.code_hash != repository.repository_hash
    assert file_sha256(resumed.checkpoint) == file_sha256(full.checkpoint)
    assert file_sha256(resumed.checkpoint_sidecar) == file_sha256(
        full.checkpoint_sidecar
    )


def test_exact_resume_rejects_snapshot_change_and_missing_policy_state(
    tmp_path: Path,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    interrupted = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "interrupted",
        max_updates_this_invocation=1,
    )

    with pytest.raises(ValueError, match="snapshot does not match"):
        run_program(
            repository,
            lane=ProgramLane.GROW,
            program_ref=GENERATOR_VNEXT,
            output_root=tmp_path / "wrong-snapshot",
            resume_checkpoint=interrupted.checkpoint,
        )

    tampered = tmp_path / "tampered.safetensors"
    shutil.copy2(interrupted.checkpoint, tampered)
    source_sidecar = program_sidecar_path(interrupted.checkpoint)
    target_sidecar = program_sidecar_path(tampered)
    payload = json.loads(source_sidecar.read_text(encoding="utf-8"))
    payload.pop("policy_state")
    target_sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid program checkpoint sidecar"):
        read_program_checkpoint(tampered)


def test_tabur_runs_as_independent_sibling_without_base_checkpoint(tmp_path: Path) -> None:
    repository = EvolutionRepository.load(ROOT)
    base = repository.resolve(BASE)
    row = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=ROW,
        output_root=tmp_path / "row",
    )

    assert row.receipt.status is ProgramRunStatus.COMPLETED
    assert row.receipt.snapshot_hash != base.snapshot_hash
    assert repository.program(ROW).state_projection is None
    assert repository.program(ROW).parent is None


def test_identity_projection_is_strict() -> None:
    source = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    target = {"weight": torch.zeros(2, 2)}

    projected = identity_state_projection(source, target)
    assert torch.equal(projected["weight"], source["weight"])
    assert projected["weight"].data_ptr() != source["weight"].data_ptr()
    with pytest.raises(ValueError, match="keys"):
        identity_state_projection(source, {"other": target["weight"]})
    with pytest.raises(ValueError, match="tensor mismatch"):
        identity_state_projection(source, {"weight": torch.zeros(4)})


def test_identity_warm_start_creates_new_run(
    tmp_path: Path,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    source = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "source",
    )
    target = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=GENERATOR_VNEXT,
        output_root=tmp_path / "target",
        warm_start_checkpoint=source.checkpoint,
    )
    cold_target = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=GENERATOR_VNEXT,
        output_root=tmp_path / "cold-target",
    )
    interrupted_target = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=GENERATOR_VNEXT,
        output_root=tmp_path / "interrupted-target",
        warm_start_checkpoint=source.checkpoint,
        max_updates_this_invocation=1,
    )
    resumed_target = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=GENERATOR_VNEXT,
        output_root=tmp_path / "resumed-target",
        resume_checkpoint=interrupted_target.checkpoint,
    )

    assert target.receipt.status is ProgramRunStatus.COMPLETED
    assert target.receipt.evidence_status is EvidenceStatus.LOCAL_UNISSUED
    assert target.receipt.run_identity_hash != source.receipt.run_identity_hash
    assert target.receipt.run_identity_hash != cold_target.receipt.run_identity_hash
    assert target.receipt.snapshot_hash != source.receipt.snapshot_hash
    assert target.receipt.initialization.mode.value == "warm_start"
    assert cold_target.receipt.initialization.mode.value == "cold"
    assert resumed_target.receipt.receipt_hash == target.receipt.receipt_hash
    assert file_sha256(resumed_target.checkpoint) == file_sha256(target.checkpoint)

    promoted = target.receipt.model_dump(mode="python", exclude={"receipt_hash"})
    promoted["evidence_status"] = EvidenceStatus.EVIDENCE_CANDIDATE_UNREVIEWED
    with pytest.raises(ValueError, match="grow run receipt cannot claim evidence"):
        ProgramRunReceipt(
            **promoted,
            receipt_hash=canonical_hash(promoted),
        )


def test_weights_only_checkpoint_requires_explicit_projected_source_program(
    tmp_path: Path,
) -> None:
    repository = EvolutionRepository.load(ROOT)
    source = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=BASE,
        output_root=tmp_path / "source",
    )
    weights_only = tmp_path / "weights-only.safetensors"
    save_file(read_checkpoint_model_state(source.checkpoint), str(weights_only))

    with pytest.raises(ValueError, match="requires --warm-start-source-program"):
        run_program(
            repository,
            lane=ProgramLane.GROW,
            program_ref=GENERATOR_VNEXT,
            output_root=tmp_path / "missing-source",
            warm_start_checkpoint=weights_only,
        )
    target = run_program(
        repository,
        lane=ProgramLane.GROW,
        program_ref=GENERATOR_VNEXT,
        output_root=tmp_path / "target",
        warm_start_checkpoint=weights_only,
        warm_start_source_program=BASE,
    )

    assert target.receipt.initialization.source_checkpoint_kind.value == "weights_only"
    assert target.receipt.initialization.source_run_identity_hash is None
    assert target.receipt.evidence_status is EvidenceStatus.LOCAL_UNISSUED
    with pytest.raises(ValueError, match="both trainer checkpoint and program sidecar"):
        run_program(
            repository,
            lane=ProgramLane.GROW,
            program_ref=GENERATOR_VNEXT,
            output_root=tmp_path / "not-resumable",
            resume_checkpoint=weights_only,
        )
