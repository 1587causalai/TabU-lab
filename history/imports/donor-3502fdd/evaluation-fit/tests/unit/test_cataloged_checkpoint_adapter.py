from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tabu_lab.adapters import (
    EPISODE_PAYLOAD_KEY,
    EPISODE_PAYLOAD_SCHEMA,
    READOUT_SELECTOR_KEY,
    READOUT_SELECTOR_SCHEMA,
    CatalogedCheckpointError,
    CatalogedCheckpointModelAdapter,
    ExplicitEpisodeInputError,
    verify_artifact_checkpoint,
)
from tabu_lab.catalog import (
    ArtifactStatusEvent,
    EvidencePointer,
    ModelArtifact,
    ModelArtifactStatus,
)
from tabu_lab.contracts import FeatureSpec, canonical_hash
from tabu_lab.evaluation.foundry import BlindExample, PreparedExample, load_suite
from tabu_lab.evidence import RunIdentity
from tabu_lab.experiments import DynamicsSemanticConfig, ModelSemanticConfig
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.primitives import MAB
from tabu_lab.registry import get_model_spec


def _write_checkpoint(
    path: Path,
    *,
    model_id: str = "tabuf",
    with_model_identity: bool = False,
) -> tuple[str, RunIdentity]:
    training = {"learning_rate": 0.01, "max_updates": 1}
    execution = {"resolved_device": "cpu", "dtype": "float32"}
    identity = RunIdentity.create(
        spec_hash="1" * 64,
        code_hash="2" * 64,
        data_hash="3" * 64,
        split_hash="4" * 64,
        compiler_hash="5" * 64,
        semantic_config_hash="6" * 64,
        execution_config_hash=canonical_hash(execution),
        training_config_hash=canonical_hash(training),
        seeds={"model_init": 1729},
    )
    resume = {
        "checkpoint_schema_version": "tabu.training-checkpoint.v3",
        "contract_version": "0.1.0",
        "execution_config_hash": identity.execution_config_hash,
        "model_id": model_id,
        "model_spec_hash": "7" * 64,
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
        "semantic_config_hash": identity.semantic_config_hash,
        "training_config_hash": identity.training_config_hash,
    }
    if with_model_identity:
        resume["model_identity"] = {
            "bandwidth": 1.0,
            "contract_version": "0.1.0",
            "label_broadcast": False,
            "label_broadcast_tau": 1.0e-6,
            "ll_ridge": None,
            "model_id": model_id,
            "profile_id": "completion.artificial_mask.v1",
            "reference_config": {"d_model": 8},
            "terminal": "nadaraya_watson",
            "tokenizer_version": "cell-tokenizer.v1",
            "variant_hash": "9" * 64,
            "variant_ref": {
                "contract_id": model_id,
                "contract_version": "0.1.0",
                "model_spec_hash": "7" * 64,
                "profile_id": "completion.artificial_mask.v1",
                "semantic_config_hash": identity.semantic_config_hash,
                "source_identity": "a" * 64,
            },
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
        {"model.weight": torch.ones(2, dtype=torch.float32)},
        str(path),
        metadata={"tabu_training_state": json.dumps(header, sort_keys=True)},
    )
    return hashlib.sha256(path.read_bytes()).hexdigest(), identity


def _artifact(
    *,
    sha256: str,
    identity: RunIdentity,
    model_spec_sha256: str = "7" * 64,
) -> ModelArtifact:
    receipt = EvidencePointer(
        uri=f"runs/{identity.run_id}/receipt.json",
        sha256="8" * 64,
        media_type="application/json",
    )
    return ModelArtifact(
        artifact_id=f"artifact-{identity.run_id}",
        contract_id="tabuf",
        contract_version="0.1.0",
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
            uri="specs/models/tabuf.yaml",
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
            ArtifactStatusEvent(
                status=ModelArtifactStatus.PRODUCED,
                evidence=receipt,
            ),
        ),
    )


def _semantic_config(*, block_kind: str | None = None) -> ModelSemanticConfig:
    values: dict[str, object] = {
        "reference": {
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
        },
        "augmented_readout_geometry": "matched_uf",
    }
    if block_kind is not None:
        values["dynamics"] = DynamicsSemanticConfig(block_kind=block_kind)
    return ModelSemanticConfig(
        **values,
    )


def _write_executable_checkpoint(
    path: Path,
    *,
    block_kind: str | None = None,
) -> tuple[ModelArtifact, dict[str, object], ModelSemanticConfig, dict[str, object]]:
    model_spec = get_model_spec("tabuf")
    semantic = _semantic_config(block_kind=block_kind)
    compiler_manifest: dict[str, object] = {
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
    config_values = semantic.reference.model_dump(mode="python")
    config_values.pop("backend")
    config_values["block_kind"] = semantic.dynamics.block_kind
    torch.manual_seed(1729)
    model = build_model(
        "tabuf",
        config=ReferenceConfig(**config_values),
        readout_geometry="matched_uf",
    )
    model.semantic_config_hash = semantic.content_hash
    resume = {
        "checkpoint_schema_version": "tabu.training-checkpoint.v3",
        "contract_version": "0.1.0",
        "execution_config_hash": identity.execution_config_hash,
        "model_id": "tabuf",
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
    artifact = _artifact(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        identity=identity,
        model_spec_sha256=canonical_hash(model_spec.model_dump(mode="json")),
    )
    return artifact, model_spec.model_dump(mode="json"), semantic, compiler_manifest


def _numeric_episode_example(*, example_id: str = "test-0") -> BlindExample:
    feature_specs = (FeatureSpec(name="x"), FeatureSpec(name="y"))
    return BlindExample(
        example_id=example_id,
        target_kind="numeric",
        target_family="feature-y",
        features={
            EPISODE_PAYLOAD_KEY: {
                "schema_version": EPISODE_PAYLOAD_SCHEMA,
                "episode_id": f"episode-{example_id}",
                "dataset_id": "sklearn-diabetes",
                "source_partition": "test",
                "fit_partition": "train",
                "row_ids": ["support-0", "support-1", "query"],
                "feature_specs": [
                    {
                        "name": spec.name,
                        "kind": spec.kind.value,
                        "domain": list(spec.domain),
                        "codebook_id": spec.codebook_id,
                        "role": spec.role.value,
                    }
                    for spec in feature_specs
                ],
                "forward_values": [[0.0, 1.0], [1.0, 3.0], [2.0, 0.0]],
                "origin_states": [
                    ["observed", "observed"],
                    ["observed", "observed"],
                    ["observed", "artificial_mask"],
                ],
                "forward_roles": [[3, 3], [3, 3], [3, 5]],
                "graph_topology": None,
                "metadata": {"numeric_normalized": True},
            }
        },
        context={
            READOUT_SELECTOR_KEY: {
                "schema_version": READOUT_SELECTOR_SCHEMA,
                "row_id": "query",
                "feature_name": "y",
            }
        },
    )


def test_cataloged_checkpoint_binds_bytes_contract_and_producer(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    sha256, identity = _write_checkpoint(checkpoint)

    verified = verify_artifact_checkpoint(
        _artifact(sha256=sha256, identity=identity),
        checkpoint,
    )

    assert verified.checkpoint_sha256 == sha256
    assert verified.contract_id == "tabuf"
    assert verified.producer_run_id == identity.run_id
    assert verified.model_spec_sha256 == "7" * 64
    assert verified.semantic_config_sha256 == "6" * 64
    assert verified.compiler_sha256 == identity.compiler_hash
    assert verified.run_identity_sha256 == identity.identity_hash
    assert verified.model_tensor_names == ("model.weight",)


def test_cataloged_checkpoint_accepts_bound_model_variant_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    sha256, identity = _write_checkpoint(checkpoint, with_model_identity=True)

    verified = verify_artifact_checkpoint(
        _artifact(sha256=sha256, identity=identity),
        checkpoint,
    )

    assert verified.model_spec_sha256 == "7" * 64


def test_cataloged_checkpoint_rejects_tampered_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    sha256, identity = _write_checkpoint(checkpoint)
    artifact = _artifact(sha256=sha256, identity=identity)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(CatalogedCheckpointError, match="byte hash"):
        verify_artifact_checkpoint(artifact, checkpoint)


def test_cataloged_checkpoint_rejects_contract_identity_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    sha256, identity = _write_checkpoint(checkpoint, model_id="tabul")

    with pytest.raises(CatalogedCheckpointError, match="model id"):
        verify_artifact_checkpoint(
            _artifact(sha256=sha256, identity=identity),
            checkpoint,
        )


def test_checkpoint_model_adapter_executes_truth_free_episode(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_executable_checkpoint(
        checkpoint
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )
    scenario = load_suite("table-completion-micro-v0").scenarios[1]
    example = _numeric_episode_example()
    fit_a = (
        PreparedExample(
            example_id="fit-a",
            target_kind="numeric",
            target_family="feature-y",
            features={"x": 0.0},
            target=-1000.0,
        ),
    )
    fit_b = (
        fit_a[0].model_copy(update={"target": 1000.0}),
    )

    first = adapter.predict(
        scenario=scenario,
        fit_examples=fit_a,
        examples=(example,),
        seed=1729,
    )
    substituted_fit_truth = adapter.predict(
        scenario=scenario,
        fit_examples=fit_b,
        examples=(example,),
        seed=1729,
    )

    assert first == substituted_fit_truth
    assert len(first) == 1
    assert first[0].example_id == example.example_id
    assert isinstance(first[0].value, float)
    assert not first[0].abstained
    assert first[0].diagnostics["checkpoint_sha256"] == artifact.checkpoint.sha256
    assert first[0].diagnostics["compiler_sha256"] == canonical_hash(compiler_manifest)
    assert adapter.spec.artifact_id == artifact.artifact_id
    assert adapter.spec.fit_iterations == 0


def test_checkpoint_model_adapter_restores_configured_mab(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "mab-checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_executable_checkpoint(
        checkpoint,
        block_kind="mab",
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )

    assert semantic.dynamics.block_kind.value == "mab"
    assert adapter._model.config.block_kind.value == "mab"
    blocks = tuple(module for module in adapter._model.modules() if isinstance(module, MAB))
    assert blocks
    assert all(type(module) is MAB for module in blocks)


@pytest.mark.parametrize("binding", ("semantic", "compiler"))
def test_checkpoint_model_adapter_rejects_reconstruction_hash_drift(
    tmp_path: Path,
    binding: str,
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_executable_checkpoint(
        checkpoint
    )
    semantic_payload = semantic.model_dump(mode="json")
    if binding == "semantic":
        semantic_payload["reference"]["routing_bandwidth"] = 2.0
    else:
        compiler_manifest = {**compiler_manifest, "projection": "changed"}

    with pytest.raises(CatalogedCheckpointError, match=f"{binding}.*hash"):
        CatalogedCheckpointModelAdapter(
            artifact=artifact.model_dump(mode="json"),
            checkpoint_path=str(checkpoint),
            model_spec=model_spec,
            semantic_config=semantic_payload,
            compiler_manifest=compiler_manifest,
        )


def test_checkpoint_model_adapter_rejects_ambiguous_blind_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    artifact, model_spec, semantic, compiler_manifest = _write_executable_checkpoint(
        checkpoint
    )
    adapter = CatalogedCheckpointModelAdapter(
        artifact=artifact.model_dump(mode="json"),
        checkpoint_path=str(checkpoint),
        model_spec=model_spec,
        semantic_config=semantic.model_dump(mode="json"),
        compiler_manifest=compiler_manifest,
    )
    scenario = load_suite("table-completion-micro-v0").scenarios[1]
    example = _numeric_episode_example()
    ambiguous = BlindExample(
        **{
            **example.model_dump(mode="python"),
            "features": {**example.features, "ignored_feature": 1.0},
        }
    )

    with pytest.raises(ExplicitEpisodeInputError, match="exactly"):
        adapter.predict(
            scenario=scenario,
            fit_examples=(),
            examples=(ambiguous,),
            seed=1729,
        )
