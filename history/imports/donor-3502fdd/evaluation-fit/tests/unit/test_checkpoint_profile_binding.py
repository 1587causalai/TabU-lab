from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from tabu_lab.adapters.checkpoint_model import (
    CatalogedCheckpointError,
    CatalogedCheckpointModelAdapter,
    ExplicitEpisodeInputError,
    _model_profile_id,
    _reference_model,
)
from tabu_lab.catalog import (
    ArtifactStatusEvent,
    EvidencePointer,
    ModelArtifact,
    ModelArtifactStatus,
)
from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.foundry import load_suite
from tabu_lab.evidence import RunIdentity
from tabu_lab.experiments import ModelSemanticConfig
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.models.types import TabUCellBaseProfile
from tabu_lab.registry import get_model_spec

CONTRACT_ID = "tabu.cell.base"
CONTRACT_VERSION = "0.2.0"

_REFERENCE_VALUES: dict[str, Any] = {
    "backend": "dense_reference_v0",
    "d_model": 8,
    "n_heads": 2,
    "d_ff": 16,
    "n_blocks": 1,
    "inducing_slots": 2,
    "matched_slots": 2,
    "max_features": 16,
    "dropout": 0.0,
    "presence_tau": 1.0e-6,
    "denominator_epsilon": 1.0e-8,
    "routing_bandwidth": 1.0,
    "geometry_normalization": "none",
}


def _cell_base_semantic(*, profile_id: str | None) -> ModelSemanticConfig:
    values: dict[str, Any] = {"reference": dict(_REFERENCE_VALUES)}
    if profile_id is not None:
        values["profile_id"] = profile_id
    return ModelSemanticConfig(**values)


def _cell_base_artifact(
    *,
    sha256: str,
    identity: RunIdentity,
    model_spec_sha256: str,
) -> ModelArtifact:
    receipt = EvidencePointer(
        uri=f"runs/{identity.run_id}/receipt.json",
        sha256="8" * 64,
        media_type="application/json",
    )
    return ModelArtifact(
        artifact_id=f"artifact-{identity.run_id}",
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
        producer_run_id=identity.run_id,
        producer_receipt=receipt,
        checkpoint=EvidencePointer(
            uri=f"runs/{identity.run_id}/checkpoint/checkpoint.safetensors",
            sha256=sha256,
            media_type="application/x-safetensors",
        ),
        checkpoint_format="safetensors",
        checkpoint_schema_version="tabu.training-checkpoint.v3",
        model_state_schema_version="tabu.model-state.v1",
        model_spec=EvidencePointer(
            uri="specs/models/tabu.cell.base.yaml",
            sha256=model_spec_sha256,
            media_type="application/yaml",
        ),
        semantic_config=EvidencePointer(
            uri=f"runs/{identity.run_id}/resolved-configs/semantic.json",
            sha256=identity.semantic_config_hash,
            media_type="application/json",
        ),
        compiler_manifest=EvidencePointer(
            uri=f"runs/{identity.run_id}/compiler-manifest.json",
            sha256=identity.compiler_hash,
            media_type="application/json",
        ),
        license_id="Apache-2.0",
        status=ModelArtifactStatus.PRODUCED,
        status_history=(
            ArtifactStatusEvent(status=ModelArtifactStatus.PRODUCED, evidence=receipt),
        ),
    )


def _write_cell_base_checkpoint(
    path: Path,
    *,
    profile_id: str,
) -> tuple[ModelArtifact, dict[str, Any], ModelSemanticConfig, dict[str, Any]]:
    model_spec = get_model_spec(CONTRACT_ID)
    semantic = _cell_base_semantic(profile_id=profile_id)
    compiler_manifest: dict[str, Any] = {
        "schema": "tabu.fit-compiler-binding.test-v1",
        "projection": "rows_to_tabular_row_carrier",
    }
    training = {"learning_rate": 0.01, "max_updates": 1}
    execution = {"resolved_device": "cpu", "dtype": "float32"}
    identity = RunIdentity.create(
        spec_hash="1" * 64,
        code_hash="2" * 64,
        data_hash="3" * 64,
        split_hash="4" * 64,
        compiler_hash=canonical_hash(compiler_manifest),
        semantic_config_hash=semantic.content_hash,
        execution_config_hash=canonical_hash(execution),
        training_config_hash=canonical_hash(training),
        seeds={"model_init": 1729},
    )
    config_values = dict(_REFERENCE_VALUES)
    config_values.pop("backend")
    torch.manual_seed(1729)
    model = build_model(
        CONTRACT_ID,
        config=ReferenceConfig(**config_values),
        profile=profile_id,
    )
    model.semantic_config_hash = semantic.content_hash
    resume = {
        "checkpoint_schema_version": "tabu.training-checkpoint.v3",
        "contract_version": CONTRACT_VERSION,
        "execution_config_hash": identity.execution_config_hash,
        "model_id": CONTRACT_ID,
        "model_spec_hash": canonical_hash(model_spec.model_dump(mode="json")),
        "model_state_schema_version": "tabu.model-state.v1",
        "objective_config": {},
        "objective_type": "tabu_lab.training.objective.Objective",
        "optimizer_defaults": {},
        "optimizer_state_schema_version": "tabu.optimizer-state.v1",
        "optimizer_type": "torch.optim.adamw.AdamW",
        "optimizer_version": torch.__version__,
        "rng_state_schema_version": "tabu.rng-state.v2",
        "run_id": identity.run_id,
        "run_identity_hash": identity.identity_hash,
        "semantic_config_hash": semantic.content_hash,
        "training_config_hash": identity.training_config_hash,
    }
    header = {
        "schema": "tabu.training-checkpoint.v3",
        "resume_contract": resume,
        "run_identity": identity.model_dump(mode="json"),
        "training_config": training,
        "execution_config": execution,
        "step": 1,
        "optimizer_param_groups": [],
        "optimizer_state_scalars": {},
        "optimizer_tensor_fields": [],
        "rng": {},
    }
    save_file(
        {
            f"model.{name}": value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        },
        str(path),
        metadata={"tabu_training_state": json.dumps(header, sort_keys=True)},
    )
    artifact = _cell_base_artifact(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        identity=identity,
        model_spec_sha256=canonical_hash(model_spec.model_dump(mode="json")),
    )
    return artifact, model_spec.model_dump(mode="json"), semantic, compiler_manifest


def test_cell_base_without_a_declared_profile_resolves_to_completion() -> None:
    """Why the rebuild path must pass the profile explicitly.

    ``tabu.cell.base`` carries two disjoint evidence profiles under one contract
    id.  A build that omits the profile does not fail: it resolves to
    artificial-mask completion.  A supervised checkpoint rebuilt this way would
    therefore evaluate under the wrong profile and still produce numbers.
    """

    model = build_model(CONTRACT_ID, config=ReferenceConfig())
    assert model.profile is TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1


def test_cell_base_rebuild_honors_a_declared_supervised_profile() -> None:
    artifact = _cell_base_artifact(
        sha256="9" * 64,
        identity=_identity(),
        model_spec_sha256="a" * 64,
    )
    model = _reference_model(
        artifact=artifact,
        model_spec=get_model_spec(CONTRACT_ID),
        semantic=_cell_base_semantic(profile_id="supervised.label_broadcast.v1"),
    )
    assert model.profile is TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
    assert _model_profile_id(model) == "supervised.label_broadcast.v1"


def test_cell_base_rebuild_refuses_an_undeclared_profile() -> None:
    artifact = _cell_base_artifact(
        sha256="9" * 64,
        identity=_identity(),
        model_spec_sha256="a" * 64,
    )
    with pytest.raises(CatalogedCheckpointError, match="declared profile"):
        _reference_model(
            artifact=artifact,
            model_spec=get_model_spec(CONTRACT_ID),
            semantic=_cell_base_semantic(profile_id=None),
        )


def test_loaded_completion_checkpoint_reports_its_profile(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_cell_base_checkpoint(
        checkpoint,
        profile_id="completion.artificial_mask.v1",
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )
    assert adapter.profile_id == "completion.artificial_mask.v1"
    assert adapter.spec.profile_id == "completion.artificial_mask.v1"


def test_predict_refuses_a_checkpoint_whose_profile_the_scenario_excludes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_cell_base_checkpoint(
        checkpoint,
        profile_id="completion.artificial_mask.v1",
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )
    suite = load_suite("table-supervised-micro-v1")
    scenario = next(
        scenario
        for scenario in suite.scenarios
        if "supervised.label_broadcast.v1" in scenario.applicable_profiles
    )
    with pytest.raises(ExplicitEpisodeInputError, match="profile is not applicable"):
        adapter.predict(scenario=scenario, fit_examples=(), examples=(), seed=1729)


def test_predict_accepts_a_checkpoint_whose_profile_the_scenario_declares(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_cell_base_checkpoint(
        checkpoint,
        profile_id="completion.artificial_mask.v1",
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )
    suite = load_suite("table-completion-micro-v1")
    scenario = next(
        scenario
        for scenario in suite.scenarios
        if "completion.artificial_mask.v1" in scenario.applicable_profiles
    )
    assert adapter.predict(scenario=scenario, fit_examples=(), examples=(), seed=1729) == ()


def _identity() -> RunIdentity:
    return RunIdentity.create(
        spec_hash="1" * 64,
        code_hash="2" * 64,
        data_hash="3" * 64,
        split_hash="4" * 64,
        compiler_hash="5" * 64,
        semantic_config_hash="6" * 64,
        execution_config_hash="7" * 64,
        training_config_hash="8" * 64,
        seeds={"model_init": 1729},
    )
