from __future__ import annotations

import hashlib
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

import tabu_lab.evidence.formal_authorization as authorization_module
from tabu_lab.catalog import (
    CatalogObjectKind,
    CatalogSourceRevision,
    EvidencePointer,
    ExperimentRecord,
    ExperimentStatus,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    StatusEvent,
    build_catalog,
    load_catalog,
)
from tabu_lab.contracts import (
    EvaluationBundle,
    PredictionBundle,
    PredictionEntry,
    PredictionKind,
    PredictionStatus,
    canonical_hash,
    canonical_json,
)
from tabu_lab.evaluation import (
    assert_public_payload_safe,
    capture_environment,
    verify_fit_attempt_artifacts,
    write_fit_attempt_artifacts,
)
from tabu_lab.evidence import Receipt, ReceiptStatus, RunBundle, RunIdentity
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    FormalAuthorizationReplaySession,
    verify_formal_authorization,
)
from tabu_lab.evidence.source_identity import SourceIdentity
from tabu_lab.experiments import (
    FitEvaluationBundle,
    FitExperimentSpec,
    FitFamilyMetrics,
    FitMetricKind,
    FitStage,
    FitTargetFamily,
)
from tabu_lab.experiments.runner import source_tree_manifest
from tabu_lab.registry import get_model_spec


def _identity() -> RunIdentity:
    return RunIdentity.create(
        spec_hash="1" * 64,
        code_hash="2" * 64,
        data_hash="3" * 64,
        split_hash="4" * 64,
        compiler_hash="5" * 64,
        semantic_config_hash="6" * 64,
        execution_config_hash=canonical_hash({"device": "cpu"}),
        training_config_hash="7" * 64,
        seeds={"episode": 1729, "sampler": 1729},
    )


def _evaluation() -> tuple[PredictionBundle, EvaluationBundle]:
    prediction = PredictionBundle(
        episode_id="fit-artifacts",
        model_id="tabuf",
        contract_version="0.1.0",
        entries={
            "numeric": PredictionEntry(
                kind=PredictionKind.NUMERIC,
                status=PredictionStatus.OK,
                values=torch.tensor([[1.0]]),
                support_ids=torch.tensor([[[0]]]),
                support_weights=torch.tensor([[[1.0]]]),
            )
        },
        auxiliaries={
            "coordinates": torch.zeros(1, 1, 1),
            "target_mask": torch.ones(1, 1, dtype=torch.bool),
            "support_available": torch.ones(1, 1, dtype=torch.bool),
        },
    )
    evaluation = EvaluationBundle(
        evaluation_id="fit-artifacts-eval",
        episode_ids=(prediction.episode_id,),
        metrics={"coverage": 1.0, "mse": 0.0},
        counts={"targets": 1, "scored_targets": 1},
    )
    return prediction, evaluation


def _fit_evaluation() -> FitEvaluationBundle:
    return FitEvaluationBundle(
        evaluation_id="fit-artifacts-fit-eval",
        experiment_id="F0-001-tabuf-v1",
        stage=FitStage.F0,
        model_seed=1729,
        targets=1,
        scored_targets=1,
        coverage=1.0,
        families=(
            FitFamilyMetrics(
                family=FitTargetFamily.COMPLETION,
                kind=FitMetricKind.NUMERIC,
                targets=1,
                scored_targets=1,
                initial_loss=1.0,
                final_loss=0.0,
                mse=0.0,
            ),
        ),
        gradient_nonzero_by_step=1,
        parameter_delta_norm=1.0,
        checkpoint_reloaded=True,
    )


def test_fit_attempt_is_immutable_and_self_verifying(tmp_path) -> None:
    prediction, evaluation = _evaluation()
    output = tmp_path / "attempt"

    artifacts = write_fit_attempt_artifacts(
        output,
        attempt_id="attempt-001",
        run_identity=_identity(),
        model_id="tabuf",
        dataset_id="fixture",
        fit_partition="train",
        preregistration_text="schema_version: test\n",
        resolved_configs={"execution": {"device": "cpu"}, "training": {"lr": 0.01}},
        dataset_manifest={"dataset": "fixture"},
        split_manifest={"partition": "train"},
        compiler_manifest={"compiler": "fixture"},
        feasibility={"status": "feasible", "targets": 1},
        metrics={
            "history": (
                {"record_type": "step", "step": 1, "loss": 1.0},
                {"record_type": "step", "step": 2, "loss": 0.0},
            ),
            "summary": {
                "record_type": "summary",
                "fit_evaluation": _fit_evaluation(),
                "loss_ratio": 0.0,
            },
        },
        evaluation=evaluation,
        predictions=(prediction,),
        baselines={"mean": 1.0},
        verdict="# pass",
        status=ReceiptStatus.SUCCEEDED,
        command=("tabu-lab", "experiments", "run"),
        checkpoint_writer=lambda path: _write_checkpoint(path),
        metadata={
            "attempt_id": "must-not-override",
            "claim_boundary": "must-not-override",
        },
    )

    receipt = verify_fit_attempt_artifacts(output)
    assert receipt.receipt_hash == artifacts.receipt_hash
    assert receipt.status is ReceiptStatus.SUCCEEDED
    assert artifacts.checkpoint is not None and artifacts.checkpoint.is_file()
    assert not (output / "metrics.json").exists()
    metric_rows = [
        json.loads(line)
        for line in (output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["record_type"] for row in metric_rows] == ["step", "step", "summary"]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fit_evaluation_hash"] == _fit_evaluation().evaluation_hash
    bundle = json.loads((output / "run_bundle.json").read_text(encoding="utf-8"))
    assert bundle["metadata"]["attempt_id"] == "attempt-001"
    assert bundle["metadata"]["claim_boundary"] == "fit_attempt_only_no_accepted_claim"
    assert bundle["episode_recipe_hashes"] == []
    with pytest.raises(FileExistsError, match="already exists"):
        write_fit_attempt_artifacts(
            output,
            attempt_id="attempt-002",
            run_identity=_identity(),
            model_id="tabuf",
            dataset_id="fixture",
            fit_partition="train",
            preregistration_text="schema_version: test\n",
            resolved_configs={"execution": {"device": "cpu"}},
            dataset_manifest={},
            split_manifest={},
            compiler_manifest={},
            feasibility={},
            metrics={},
            evaluation=evaluation,
            predictions=(prediction,),
            baselines={},
            verdict="# pass",
            status=ReceiptStatus.SUCCEEDED,
            command=(),
            checkpoint_writer=lambda path: _write_checkpoint(path),
        )


def _write_checkpoint(path):  # type: ignore[no-untyped-def]
    path.write_bytes(b"safe-checkpoint-fixture")
    return path


def test_fit_attempt_records_explicit_recipe_hashes_without_model_metadata(tmp_path) -> None:
    prediction, evaluation = _evaluation()
    # A model-facing bundle is not a provenance carrier.  Even if legacy or
    # third-party code places a recipe-shaped value in metadata, the writer
    # must use only the explicit compiler/experiment boundary argument.
    prediction = replace(prediction, metadata={"recipe_hash": "f" * 64})
    recipe_hashes = ("a" * 64, "b" * 64)
    output = tmp_path / "explicit-recipes"

    write_fit_attempt_artifacts(
        output,
        attempt_id="attempt-explicit-recipes",
        run_identity=_identity(),
        model_id="tabuf",
        dataset_id="fixture",
        fit_partition="train",
        preregistration_text="schema_version: test\n",
        resolved_configs={"execution": {"device": "cpu"}},
        dataset_manifest={},
        split_manifest={},
        compiler_manifest={},
        feasibility={},
        metrics={},
        evaluation=evaluation,
        predictions=(prediction,),
        episode_recipe_hashes=recipe_hashes,
        baselines={},
        verdict="# failed",
        status=ReceiptStatus.FAILED,
        error="bounded failure",
        command=(),
    )

    verify_fit_attempt_artifacts(output)
    bundle = RunBundle.model_validate(
        json.loads((output / "run_bundle.json").read_text(encoding="utf-8"))
    )
    assert bundle.episode_recipe_hashes == recipe_hashes


@pytest.mark.parametrize(
    "recipe_hashes",
    (
        ("not-a-sha256",),
        ("a" * 64, "a" * 64),
        "a" * 64,
    ),
)
def test_fit_attempt_rejects_invalid_explicit_recipe_hashes(
    tmp_path: Path,
    recipe_hashes: object,
) -> None:
    prediction, evaluation = _evaluation()

    with pytest.raises((TypeError, ValueError)):
        write_fit_attempt_artifacts(
            tmp_path / "invalid-recipes",
            attempt_id="attempt-invalid-recipes",
            run_identity=_identity(),
            model_id="tabuf",
            dataset_id="fixture",
            fit_partition="train",
            preregistration_text="schema_version: test\n",
            resolved_configs={"execution": {"device": "cpu"}},
            dataset_manifest={},
            split_manifest={},
            compiler_manifest={},
            feasibility={},
            metrics={},
            evaluation=evaluation,
            predictions=(prediction,),
            episode_recipe_hashes=recipe_hashes,  # type: ignore[arg-type]
            baselines={},
            verdict="# failed",
            status=ReceiptStatus.FAILED,
            error="bounded failure",
            command=(),
        )
    assert not (tmp_path / "invalid-recipes").exists()


def _write_training_checkpoint(
    path: Path,
    *,
    identity: RunIdentity,
    spec: FitExperimentSpec,
    training: dict[str, object],
    execution: dict[str, object],
) -> Path:
    from safetensors.torch import save_file

    resume = {
        "checkpoint_schema_version": "tabu.training-checkpoint.v3",
        "contract_version": spec.contract_version,
        "execution_config_hash": identity.execution_config_hash,
        "model_id": spec.contract_id,
        "model_spec_hash": spec.model_spec_hash,
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
        {"model.weight": torch.ones(1, dtype=torch.float32)},
        str(path),
        metadata={"tabu_training_state": json.dumps(header, sort_keys=True)},
    )
    return path


def test_environment_disclosure_is_public_and_machine_anonymous() -> None:
    disclosure, payload = capture_environment("cpu")

    assert payload["schema_version"] == "tabu.fit-environment.v2"
    assert payload["host_class"] == "cpu-host"
    assert disclosure.host_class == "cpu-host"
    assert "host" not in payload
    assert "hostname" not in payload
    assert "python_executable" not in payload
    serialized = json.dumps(payload, sort_keys=True)
    assert "/Users/" not in serialized
    assert "/home/" not in serialized


def _authorization_context(
    tmp_path: Path,
    *,
    spec: FitExperimentSpec,
    preregistration: str,
) -> tuple[FormalAuthorizationContext, str, SourceIdentity, tuple[dict[str, object], ...]]:
    repository = tmp_path / "authorization-repo"
    remote_repository = tmp_path / "authorization-remote.git"
    remote_repository.mkdir()
    subprocess.run(
        ("git", "-C", remote_repository, "init", "--bare", "-b", "main"),
        check=True,
        capture_output=True,
    )
    (repository / "specs/models").mkdir(parents=True)
    (repository / ".gitignore").write_text(".local-runs/\n", encoding="utf-8")
    preregistration_path = (
        repository
        / "experiments/fit-first/F0"
        / spec.experiment_id
        / "preregistration.yaml"
    )
    preregistration_path.parent.mkdir(parents=True)
    preregistration_path.write_text(preregistration, encoding="utf-8")
    (repository / "specs/models" / f"{spec.contract_id}.yaml").write_text(
        yaml.safe_dump(
            get_model_spec(spec.contract_id).model_dump(mode="json"),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    source = repository / "src/tabu_lab/__init__.py"
    source.parent.mkdir(parents=True)
    source.write_text("__version__ = 'formal-artifact-test'\n", encoding="utf-8")
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'tabu-lab-formal-test'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    subprocess.run(("git", "init", "-b", "main", repository), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", repository, "config", "user.email", "tests@example.test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", repository, "config", "user.name", "TabU Tests"),
        check=True,
    )
    public_repository = "https://github.com/wehub-community/tabu-lab.git"
    subprocess.run(
        ("git", "-C", repository, "remote", "add", "origin", public_repository),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            repository,
            "config",
            f"url.file://{remote_repository.resolve()}.insteadOf",
            public_repository,
        ),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            repository,
            "config",
            "tabu.tests.bareRemote",
            str(remote_repository.resolve()),
        ),
        check=True,
    )
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "freeze executable source"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", repository, "push", "-u", "origin", "main"),
        check=True,
        capture_output=True,
    )
    manifest = source_tree_manifest(
        repository,
        preregistration=preregistration_path,
        request_formal=True,
        reviewed=True,
    )
    source_identity = SourceIdentity.model_validate(manifest["source_identity"])
    source_files = tuple(manifest["files"])
    evidence = repository / "authorization-evidence"
    evidence.mkdir()
    report_payload = {"review": spec.experiment_id, "decision": "approved"}
    report_path = evidence / "review-report.json"
    report_path.write_text(canonical_json(report_payload) + "\n", encoding="utf-8")
    gong_payload = {"approval": spec.experiment_id, "decision": "approved"}
    gong_path = evidence / "gong-approval.json"
    gong_path.write_text(canonical_json(gong_payload) + "\n", encoding="utf-8")
    source_path = evidence / "source-identity.json"
    source_path.write_text(canonical_json(source_identity) + "\n", encoding="utf-8")
    preregistration_hash = canonical_hash(yaml.safe_load(preregistration))
    report_pointer = EvidencePointer(
        uri=report_path.relative_to(repository).as_posix(),
        sha256=canonical_hash(report_payload),
    )
    source_pointer = EvidencePointer(
        uri=source_path.relative_to(repository).as_posix(),
        sha256=canonical_hash(source_identity),
    )
    review = ReviewRecord(
        review_id=f"review-{spec.experiment_id}",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id=spec.experiment_id,
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=report_pointer,
        gong_approval=EvidencePointer(
            uri=gong_path.relative_to(repository).as_posix(),
            sha256=canonical_hash(gong_payload),
        ),
    )
    review_path = repository / "reviews" / f"{review.review_id}.json"
    review_path.parent.mkdir()
    review_path.write_text(canonical_json(review) + "\n", encoding="utf-8")
    record = ExperimentRecord(
        experiment_id=spec.experiment_id,
        contract_id=spec.contract_id,
        hypothesis="bounded authorization fixture",
        claim_boundary="F0 fit only",
        status=ExperimentStatus.RUNNABLE,
        status_history=(
            StatusEvent(status=ExperimentStatus.DRAFT.value),
            StatusEvent(
                status=ExperimentStatus.PREREGISTERED.value,
                evidence_hashes=(preregistration_hash, report_pointer.sha256),
            ),
            StatusEvent(
                status=ExperimentStatus.RUNNABLE.value,
                evidence_hashes=(source_pointer.sha256,),
            ),
        ),
        preregistration=EvidencePointer(
            uri=preregistration_path.relative_to(repository).as_posix(),
            sha256=preregistration_hash,
            media_type="application/yaml",
        ),
        preregistration_review=report_pointer,
        source_identity=source_pointer,
        review_ids=(review.review_id,),
        supersedes_experiment_ids=spec.supersedes_experiment_ids,
        revision_rationale=spec.revision_rationale,
    )
    record_path = repository / "experiments/records" / f"{spec.experiment_id}.json"
    record_path.parent.mkdir()
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "review authorization"),
        check=True,
        capture_output=True,
    )
    revision_commit = subprocess.run(
        ("git", "-C", repository, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    draft = build_catalog(repository)
    source_revision = CatalogSourceRevision(
        repository_uri=public_repository,
        commit=revision_commit,
        catalog_source_tree_hash=draft.source_tree_hash,
    )
    catalog_path = repository / "catalog.json"
    catalog = build_catalog(repository, catalog_path, source_revision=source_revision)
    subprocess.run(("git", "-C", repository, "add", "catalog.json"), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "freeze catalog"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", repository, "push", "origin", "main"),
        check=True,
        capture_output=True,
    )
    return (
        FormalAuthorizationContext(
            repository=repository,
            catalog=catalog_path,
            experiment_id=spec.experiment_id,
        ),
        catalog.catalog_hash,
        source_identity,
        source_files,
    )


def _formal_inputs(tmp_path: Path) -> tuple[dict[str, object], FitExperimentSpec]:
    # Formal gate artifacts must describe the runtime that actually produced
    # them.  Keep deterministic algorithms enabled while the writer captures
    # its environment disclosure; diagnostic nondeterministic runs are never
    # accepted by the formal writer.
    torch.use_deterministic_algorithms(True)
    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/fit-first/F0/F0-001-tabuf-v1/preregistration.yaml"
    )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["execution"]["device"] = "cpu"
    payload["execution"]["device_index"] = None
    payload["execution"]["deterministic_algorithms"] = True
    payload["execution"]["evidence_mode"] = "gate"
    payload["training"]["exact_resume"] = True
    payload["semantic"]["augmented_readout_geometry"] = "matched_uf"
    payload["supersedes_experiment_ids"] = []
    payload["revision_rationale"] = None
    spec = FitExperimentSpec.model_validate(payload)
    preregistration = yaml.safe_dump(
        spec.model_dump(mode="json"), sort_keys=False, allow_unicode=True
    )
    authorization, catalog_hash, source_identity, source_files = _authorization_context(
        tmp_path,
        spec=spec,
        preregistration=preregistration,
    )
    _, environment = capture_environment("cpu")
    environment["deterministic_algorithms"] = True
    code = {
        "schema_version": "tabu.source-tree.v3",
        "mode": "repository",
        "root_label": "repository",
        "files": source_files,
        "source_identity": source_identity.model_dump(mode="json"),
    }
    training = spec.training.model_dump(mode="json")
    execution = {
        **spec.execution.model_dump(mode="json"),
        "resolved_device": "cpu",
        "environment_hash": canonical_hash(environment),
        "host_class": environment["host_class"],
        "torch_version": environment["torch_version"],
        "python_version": environment["python_version"],
    }
    seed = spec.seeds.model_seeds[0]
    seeds = {
        "episode": spec.seeds.data_seed,
        "model_init": seed,
        "numpy": seed,
        "python": seed,
        "sampler": spec.seeds.episode_order_seed,
        "torch_cpu": seed,
    }
    provenance = {
        "dataset_hash": spec.dataset.dataset_hash,
        "split_manifest_hash": spec.split.content_hash,
        "source_view_hash": "b" * 64,
        "fit_view_hash": "c" * 64,
        "recipe_hash": spec.episode_schedule.recipe_hashes[0],
        "graph_topology_hash": None,
    }
    normalizer_config_hash = canonical_hash(
        {
            "kind": "numeric_normalizer",
            "epsilon": 1e-8,
            "shared_numeric_groups": [],
        }
    )
    statistics = {
        "schema": "tabu.fitted-statistics.v2",
        "fit_view_hash": provenance["fit_view_hash"],
        "split_definition_hash": "d" * 64,
        "config_hash": normalizer_config_hash,
        "fit_value_mask_hash": "e" * 64,
        "feature_names": ("x",),
        "feature_kinds": ("numeric",),
        "counts": torch.tensor([2], dtype=torch.int64),
        "means": torch.tensor([0.0], dtype=torch.float64),
        "scales": torch.tensor([1.0], dtype=torch.float64),
    }
    normalizer = {
        "schema": "tabu.numeric-normalizer-binding.v1",
        **{name: value for name, value in statistics.items() if name != "schema"},
        "epsilon": 1e-8,
        "shared_numeric_groups": (),
        "artifact_hash": canonical_hash(statistics),
    }
    provenance["numeric_normalizer_hash"] = normalizer["artifact_hash"]
    compiler = {
        "schema": "tabu.fit-compiler-binding.v1",
        "typed_split_hash": spec.split.content_hash,
        "typed_split_kind": spec.split.kind.value,
        "fit_partition": spec.split.fit_partition,
        "compiler_provenance": provenance,
        "compiler_provenance_hash": canonical_hash(
            {"schema": "tabu.compilation-provenance.v2", **provenance}
        ),
        "numeric_normalizer": normalizer,
        "projection": "rows_to_tabular_row_carrier",
    }
    identity = RunIdentity.create(
        spec_hash=spec.spec_hash,
        code_hash=canonical_hash(code),
        data_hash=spec.dataset.dataset_hash,
        split_hash=spec.split.content_hash,
        compiler_hash=canonical_hash(compiler),
        semantic_config_hash=spec.semantic.content_hash,
        execution_config_hash=canonical_hash(execution),
        training_config_hash=canonical_hash(training),
        seeds=seeds,
    )
    prediction, evaluation = _evaluation()
    row_ids = spec.split.partition(spec.split.fit_partition).row_ids
    inputs: dict[str, object] = {
            "attempt_id": "attempt-formal-001",
            "run_identity": identity,
            "model_id": spec.contract_id,
            "dataset_id": spec.dataset.dataset_id,
            "fit_partition": spec.split.fit_partition,
            "preregistration_text": preregistration,
            "resolved_configs": {
                "code": code,
                "experiment": spec,
                "semantic": spec.semantic,
                "training": training,
                "execution": execution,
                "seeds": seeds,
            },
            "dataset_manifest": {
                "schema": "tabu.fit-dataset-manifest.v1",
                "dataset": spec.dataset,
                "dataset_id": spec.dataset.dataset_id,
                "dataset_hash": spec.dataset.dataset_hash,
                "feature_specs": (),
                "row_ids": row_ids,
                "metadata": {},
            },
            "split_manifest": spec.split,
            "compiler_manifest": compiler,
            "feasibility": {"status": "feasible", "targets": 1},
            "metrics": {
                "history": ({"record_type": "step", "step": 1, "loss": 0.0},),
                "summary": {
                    "record_type": "summary",
                    "fit_evaluation": _fit_evaluation(),
                },
            },
            "evaluation": evaluation,
            "predictions": (prediction,),
            "baselines": {"mean": 1.0},
            "verdict": "# pass",
            "status": ReceiptStatus.SUCCEEDED,
            "command": (
                "tabu-lab",
                "experiments",
                "run",
                "--output-root",
                "formal-staging://output",
                "--authorization-catalog",
                f"sha256:{catalog_hash}",
            ),
            "checkpoint_writer": lambda path: _write_training_checkpoint(
                path,
                identity=identity,
                spec=spec,
                training=training,
                execution=execution,
            ),
            "metadata": {
                "attempt_nonce": "formal-test-nonce",
                "experiment_id": spec.experiment_id,
                "stage": spec.stage.value,
                "model_seed": seed,
                "code_hash": identity.code_hash,
                "issuance_status": "formal",
                "source_identity_hash": canonical_hash(source_identity),
            },
            "formal_authorization": authorization,
        }
    return inputs, spec


def _replace_authorized_source_identity(
    context: FormalAuthorizationContext,
    source_identity: SourceIdentity,
) -> None:
    source_path = context.repository / "authorization-evidence/source-identity.json"
    source_path.write_text(canonical_json(source_identity) + "\n", encoding="utf-8")
    record_path = (
        context.repository / "experiments/records" / f"{context.experiment_id}.json"
    )
    record = ExperimentRecord.model_validate_json(record_path.read_bytes())
    source_hash = canonical_hash(source_identity)
    replaced = record.model_copy(
        update={
            "source_identity": EvidencePointer(
                uri="authorization-evidence/source-identity.json",
                sha256=source_hash,
            ),
            "status_history": tuple(
                event.model_copy(update={"evidence_hashes": (source_hash,)})
                if event.status == ExperimentStatus.RUNNABLE.value
                else event
                for event in record.status_history
            ),
        }
    )
    record_path.write_text(canonical_json(replaced) + "\n", encoding="utf-8")


def _refreeze_authorization_catalog(context: FormalAuthorizationContext) -> None:
    repository = context.repository
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "change authorization sources"),
        check=True,
        capture_output=True,
    )
    revision_commit = subprocess.run(
        ("git", "-C", repository, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    existing_revision = load_catalog(context.catalog).source_revision
    assert existing_revision is not None
    draft = build_catalog(repository)
    build_catalog(
        repository,
        context.catalog,
        source_revision=CatalogSourceRevision(
            repository_uri=existing_revision.repository_uri,
            commit=revision_commit,
            catalog_source_tree_hash=draft.source_tree_hash,
        ),
    )
    subprocess.run(("git", "-C", repository, "add", "catalog.json"), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "refreeze authorization catalog"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", repository, "push", "origin", "main"),
        check=True,
        capture_output=True,
    )


def test_formal_authorization_rejects_self_declared_nonexistent_source_commit(
    tmp_path: Path,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    forged_payload = source.model_dump(mode="python")
    forged_payload["commit"] = "f" * 40
    forged = SourceIdentity.model_validate(forged_payload)
    _replace_authorized_source_identity(context, forged)
    _refreeze_authorization_catalog(context)

    with pytest.raises(FormalAuthorizationError, match="exact Git commit"):
        verify_formal_authorization(
            context,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=forged,
        )


def test_formal_authorization_rejects_missing_configured_remote(tmp_path: Path) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    subprocess.run(
        ("git", "-C", context.repository, "remote", "remove", "origin"),
        check=True,
        capture_output=True,
    )

    with pytest.raises(FormalAuthorizationError, match="Git verification failed"):
        verify_formal_authorization(
            context,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source,
        )


def test_public_remote_probe_ignores_ambient_git_rewrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch_ref = "refs/heads/main"
    remote_oid = "a" * 40
    observed: dict[str, object] = {}
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.file:///attacker/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "attacker.git"))

    def fake_run(
        arguments: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        observed["arguments"] = arguments
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            arguments,
            0,
            stdout=f"{remote_oid}\t{branch_ref}\n".encode(),
            stderr=b"",
        )

    monkeypatch.setattr(authorization_module.subprocess, "run", fake_run)
    resolved = authorization_module._probe_public_https_ref_oid(
        "https://github.com/wehub-community/tabu-lab",
        branch_ref,
    )

    assert resolved == remote_oid
    arguments = observed["arguments"]
    assert isinstance(arguments, tuple)
    assert arguments == (
        "git",
        "-c",
        "protocol.file.allow=never",
        "ls-remote",
        "--exit-code",
        "https://github.com/wehub-community/tabu-lab",
        branch_ref,
    )
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_CONFIG_SYSTEM"] == os.devnull
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert "GIT_CONFIG_COUNT" not in environment
    assert "GIT_CONFIG_KEY_0" not in environment
    assert "GIT_CONFIG_VALUE_0" not in environment
    assert "GIT_DIR" not in environment
    assert Path(str(observed["cwd"])).resolve() != tmp_path.resolve()


def test_catalog_source_revision_commit_is_rebuilt_not_merely_reachable(
    tmp_path: Path,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    current = load_catalog(context.catalog)
    assert current.source_revision is not None
    forged = CatalogSourceRevision(
        repository_uri=current.source_revision.repository_uri,
        commit=source.commit,
        catalog_source_tree_hash=current.source_tree_hash,
    )
    build_catalog(context.repository, context.catalog, source_revision=forged)
    subprocess.run(("git", "-C", context.repository, "add", "catalog.json"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            context.repository,
            "commit",
            "-m",
            "forge catalog source revision",
        ),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", context.repository, "push", "origin", "main"),
        check=True,
        capture_output=True,
    )

    with pytest.raises(FormalAuthorizationError, match="immutable commit sources"):
        verify_formal_authorization(
            context,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source,
        )


def test_formal_authorization_requires_hashed_gong_approval_evidence(
    tmp_path: Path,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    review_path = context.repository / f"reviews/review-{context.experiment_id}.json"
    review = ReviewRecord.model_validate_json(review_path.read_bytes())
    review_path.write_text(
        canonical_json(review.model_copy(update={"gong_approval": None})) + "\n",
        encoding="utf-8",
    )
    _refreeze_authorization_catalog(context)

    with pytest.raises(FormalAuthorizationError, match="lacks gong approval"):
        verify_formal_authorization(
            context,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source,
        )


def test_formal_authorization_rejects_gong_approval_hash_drift(tmp_path: Path) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    gong = context.repository / "authorization-evidence/gong-approval.json"
    gong.write_text('{"approval":"forged"}\n', encoding="utf-8")
    _refreeze_authorization_catalog(context)

    with pytest.raises(FormalAuthorizationError, match="gong approval source digest"):
        verify_formal_authorization(
            context,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source,
        )


def test_historical_distribution_formal_authorization_fails_closed(
    tmp_path: Path,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    original = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    verified = verify_formal_authorization(
        context,
        preregistration_text=str(inputs["preregistration_text"]),
        live_source_identity=original,
    )
    distribution = SourceIdentity(
        source_kind="distribution",
        issuance_status="formal",
        reviewed=True,
        source_tree_hash=original.source_tree_hash,
        distribution_uri="https://github.com/wehub-community/tabu-lab/releases/source.whl",
        distribution_sha256="a" * 64,
        lock_hash="b" * 64,
    )
    _replace_authorized_source_identity(context, distribution)
    _refreeze_authorization_catalog(context)
    catalog = load_catalog(context.catalog)
    canonical_commit = subprocess.run(
        ("git", "-C", context.repository, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attacked = verified.summary.model_copy(
        update={
            "canonical_commit": canonical_commit,
            "catalog_hash": catalog.catalog_hash,
            "catalog_source_tree_hash": catalog.source_tree_hash,
            "source_identity_sha256": canonical_hash(distribution),
        }
    )

    with pytest.raises(FormalAuthorizationError, match="historical distribution"):
        FormalAuthorizationReplaySession(context.repository).verify(
            attacked,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=distribution,
        )


def _write_canonical_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _remint_integrity_envelope(output: Path, changed_uri: str) -> None:
    changed = output / changed_uri
    digest = hashlib.sha256(changed.read_bytes()).hexdigest()
    size = changed.stat().st_size
    bundle = json.loads((output / "run_bundle.json").read_text(encoding="utf-8"))
    for artifact in bundle["artifacts"]:
        if artifact["uri"] == changed_uri:
            artifact["sha256"] = digest
            artifact["size_bytes"] = size
    typed_bundle = RunBundle.model_validate(bundle)
    _write_canonical_json(output / "run_bundle.json", typed_bundle.model_dump(mode="json"))
    envelope = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    receipt_payload = envelope["receipt"]
    receipt_payload["artifacts"] = typed_bundle.model_dump(mode="json")["artifacts"]
    receipt_payload["run_bundle_hash"] = typed_bundle.run_bundle_hash
    typed_receipt = Receipt.model_validate(receipt_payload)
    (output / "receipt.json").write_text(
        canonical_json(
            {
                "schema_version": "tabu.receipt-envelope.v1",
                "receipt_hash": typed_receipt.receipt_hash,
                "receipt": typed_receipt,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["run_bundle_hash"] = typed_bundle.run_bundle_hash
    _write_canonical_json(output / "run_manifest.json", manifest)
    paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "artifacts.sha256"
    )
    (output / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


def _remint_formal_authorization_summary(
    output: Path,
    *,
    catalog_hash: str,
) -> None:
    bundle_payload = json.loads((output / "run_bundle.json").read_text(encoding="utf-8"))
    bundle_payload["metadata"]["formal_authorization"]["catalog_hash"] = catalog_hash
    bundle = RunBundle.model_validate(bundle_payload)
    _write_canonical_json(output / "run_bundle.json", bundle.model_dump(mode="json"))

    envelope = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    receipt_payload = envelope["receipt"]
    receipt_payload["metadata"]["formal_authorization"]["catalog_hash"] = catalog_hash
    receipt_payload["run_bundle_hash"] = bundle.run_bundle_hash
    receipt = Receipt.model_validate(receipt_payload)
    (output / "receipt.json").write_text(
        canonical_json(
            {
                "schema_version": "tabu.receipt-envelope.v1",
                "receipt_hash": receipt.receipt_hash,
                "receipt": receipt,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["run_bundle_hash"] = bundle.run_bundle_hash
    _write_canonical_json(output / "run_manifest.json", manifest)
    paths = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "artifacts.sha256"
    )
    (output / "artifacts.sha256").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
            f"{path.relative_to(output).as_posix()}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


def test_git_history_replay_rejects_reminted_catalog_hash(tmp_path: Path) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    output = tmp_path / "formal-reminted-authorization"
    write_fit_attempt_artifacts(output, **inputs)  # type: ignore[arg-type]
    _remint_formal_authorization_summary(output, catalog_hash="0" * 64)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)

    with pytest.raises(ValueError, match="canonical replay failed"):
        verify_fit_attempt_artifacts(
            output,
            formal_authorization_replay=FormalAuthorizationReplaySession(
                context.repository
            ),
        )


def test_git_history_replay_rejects_missing_review_report(tmp_path: Path) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source_identity = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    verified = verify_formal_authorization(
        context,
        preregistration_text=str(inputs["preregistration_text"]),
        live_source_identity=source_identity,
    )
    subprocess.run(
        (
            "git",
            "-C",
            context.repository,
            "rm",
            "authorization-evidence/review-report.json",
        ),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", context.repository, "commit", "-m", "remove review report"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", context.repository, "push", "origin", "main"),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", context.repository, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attacked = verified.summary.model_copy(update={"canonical_commit": commit})

    with pytest.raises(FormalAuthorizationError, match=r"review report.*missing"):
        FormalAuthorizationReplaySession(context.repository).verify(
            attacked,
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source_identity,
        )


def test_git_history_replay_rejects_recursive_cycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tabu_lab.evidence.formal_authorization as authorization_module

    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    preregistration = str(inputs["preregistration_text"])
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source_identity = SourceIdentity.model_validate(resolved["code"]["source_identity"])
    summary = verify_formal_authorization(
        context,
        preregistration_text=preregistration,
        live_source_identity=source_identity,
    ).summary
    replay = FormalAuthorizationReplaySession(context.repository)

    def recursive_replay(**_: object) -> object:
        return replay.verify(
            summary,
            preregistration_text=preregistration,
            live_source_identity=source_identity,
        )

    monkeypatch.setattr(
        authorization_module,
        "_verify_materialized_authorization",
        recursive_replay,
    )
    with pytest.raises(FormalAuthorizationError, match="replay cycle"):
        replay.verify(
            summary,
            preregistration_text=preregistration,
            live_source_identity=source_identity,
        )


def test_formal_fit_attempt_recomputes_all_identity_preimages(tmp_path) -> None:
    inputs, spec = _formal_inputs(tmp_path)
    output = tmp_path / "formal-attempt"

    artifacts = write_fit_attempt_artifacts(output, **inputs)  # type: ignore[arg-type]

    receipt = verify_fit_attempt_artifacts(
        output,
        formal_authorization=inputs["formal_authorization"],  # type: ignore[arg-type]
    )
    assert receipt.receipt_hash == artifacts.receipt_hash
    assert receipt.metadata["formal_authorization"]["catalog_hash"]
    assert "/Users/" not in " ".join(receipt.command)
    bundle = RunBundle.model_validate(
        json.loads((output / "run_bundle.json").read_text(encoding="utf-8"))
    )
    assert bundle.identity.spec_hash == spec.spec_hash
    assert bundle.identity.data_hash == spec.dataset.dataset_hash
    assert bundle.identity.split_hash == spec.split.content_hash


def test_formal_authorization_rejects_detached_self_consistent_catalog(tmp_path: Path) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    context = inputs["formal_authorization"]
    assert isinstance(context, FormalAuthorizationContext)
    detached = tmp_path / "self-consistent-catalog.json"
    detached.write_bytes(context.catalog.read_bytes())
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    source_identity = SourceIdentity.model_validate(resolved["code"]["source_identity"])

    with pytest.raises(FormalAuthorizationError, match=r"canonical repository catalog.json"):
        verify_formal_authorization(
            FormalAuthorizationContext(
                repository=context.repository,
                catalog=detached,
                experiment_id=context.experiment_id,
            ),
            preregistration_text=str(inputs["preregistration_text"]),
            live_source_identity=source_identity,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("hostname", "internal-build-host"),
        ("username", "alice"),
        ("access_token", "secret-shaped-value"),
        ("database_password", "secret-shaped-value"),
        ("training_hostname", "internal-build-host"),
        ("note", "/Users/alice/private/checkpoint.safetensors"),
    ),
)
def test_formal_writer_rejects_private_or_secret_shaped_metadata(
    tmp_path: Path,
    field_name: str,
    value: str,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    metadata = inputs["metadata"]
    assert isinstance(metadata, dict)
    metadata[field_name] = value

    with pytest.raises(ValueError, match=r"forbidden private field|private absolute path"):
        write_fit_attempt_artifacts(
            tmp_path / f"formal-private-{field_name}",
            **inputs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    (
        "/tmp/checkpoint.safetensors",
        "cache: /var/folders/ab/build.json",
        "mounted at /mnt/research/run-1",
        "service root=/srv/tabu",
        r"C:\\research\\checkpoint.safetensors",
        r"\\\\server\\share\\checkpoint.safetensors",
        "~/private/checkpoint.safetensors",
        "file:///opt/tabu/checkpoint.safetensors",
    ),
)
def test_public_payload_rejects_any_local_absolute_path(value: str) -> None:
    with pytest.raises(ValueError, match="private absolute path"):
        assert_public_payload_safe({"note": value})


@pytest.mark.parametrize(
    "value",
    (
        "https://huggingface.co/wehub/tabu/resolve/commit/checkpoint.safetensors",
        "hf://wehub/tabu@commit/checkpoint.safetensors",
        "git://tabu-lab/runs/run-1/receipt.json",
        "runs/run-1/receipt.json",
        "coverage/abstention",
        "RMSE / MAE",
    ),
)
def test_public_payload_allows_remote_uris_and_relative_text(value: str) -> None:
    assert_public_payload_safe({"note": value})


@pytest.mark.parametrize(
    "field_name",
    (
        "os_family",
        "host_class",
        "architecture",
        "device",
        "python_version",
        "torch_version",
        "cuda_version",
    ),
)
def test_public_environment_payload_rejects_secret_shaped_string_values(
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match="private identity or secret"):
        assert_public_payload_safe(
            {field_name: "Bearer abcdefghijklmnopqrstuvwxyz"}
        )


def test_public_environment_payload_accepts_generalized_versions_and_classes() -> None:
    assert_public_payload_safe(
        {
            "os_family": "Darwin",
            "host_class": "mps-host",
            "architecture": "arm64",
            "device": "mps",
            "python_version": "3.11.14",
            "torch_version": "2.8.0+cu128",
            "cuda_version": "12.8",
        }
    )


@pytest.mark.parametrize("artifact", ("baselines", "verdict"))
def test_formal_writer_scans_every_public_text_artifact(
    tmp_path: Path,
    artifact: str,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    if artifact == "baselines":
        inputs["baselines"] = {"mean": 1.0, "api_key": "must-not-be-published"}
    else:
        inputs["verdict"] = "debug checkpoint: /Users/alice/private/model.safetensors"

    with pytest.raises(ValueError, match=r"forbidden private field|private absolute path"):
        write_fit_attempt_artifacts(
            tmp_path / f"formal-private-{artifact}",
            **inputs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "binding",
    ("preregistration", "semantic", "code", "seeds", "dataset", "split", "compiler"),
)
def test_formal_fit_attempt_fails_closed_on_unbound_preimage(tmp_path, binding: str) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    resolved = inputs["resolved_configs"]
    assert isinstance(resolved, dict)
    if binding == "preregistration":
        preregistration = yaml.safe_load(str(inputs["preregistration_text"]))
        preregistration["experiment_id"] = "F0-001-tabuf-v2"
        inputs["preregistration_text"] = yaml.safe_dump(
            preregistration, sort_keys=False, allow_unicode=True
        )
    elif binding == "semantic":
        semantic = deepcopy(resolved["semantic"].model_dump(mode="json"))
        semantic["reference"]["routing_bandwidth"] = 2.0
        resolved["semantic"] = semantic
    elif binding == "code":
        code = deepcopy(resolved["code"])
        code["root_label"] = "tampered"
        resolved["code"] = code
    elif binding == "seeds":
        seeds = deepcopy(resolved["seeds"])
        seeds["sampler"] += 1
        resolved["seeds"] = seeds
    elif binding == "dataset":
        dataset = deepcopy(inputs["dataset_manifest"])
        dataset["dataset_hash"] = "0" * 64
        inputs["dataset_manifest"] = dataset
    elif binding == "split":
        split = inputs["split_manifest"].model_dump(mode="json")
        split["strategy"] = "tampered"
        inputs["split_manifest"] = split
    else:
        compiler = deepcopy(inputs["compiler_manifest"])
        compiler["projection"] = "tampered"
        inputs["compiler_manifest"] = compiler

    with pytest.raises(ValueError):
        write_fit_attempt_artifacts(
            tmp_path / f"formal-{binding}",
            **inputs,  # type: ignore[arg-type]
        )


def test_formal_verifier_rejects_rehashed_but_semantically_unbound_config(
    tmp_path,
) -> None:
    inputs, _ = _formal_inputs(tmp_path)
    output = tmp_path / "formal-remint"
    write_fit_attempt_artifacts(output, **inputs)  # type: ignore[arg-type]
    semantic_path = output / "resolved-configs" / "semantic.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["reference"]["routing_bandwidth"] = 2.0
    _write_canonical_json(semantic_path, semantic)
    _remint_integrity_envelope(output, "resolved-configs/semantic.json")

    with pytest.raises(ValueError, match="semantic config does not match"):
        verify_fit_attempt_artifacts(
            output,
            formal_authorization=inputs["formal_authorization"],  # type: ignore[arg-type]
        )


def test_checksum_tampering_is_detected(tmp_path) -> None:
    prediction, evaluation = _evaluation()
    output = tmp_path / "attempt"
    write_fit_attempt_artifacts(
        output,
        attempt_id="attempt-tamper",
        run_identity=_identity(),
        model_id="tabuf",
        dataset_id="fixture",
        fit_partition="train",
        preregistration_text="schema_version: test\n",
        resolved_configs={"execution": {"device": "cpu"}},
        dataset_manifest={},
        split_manifest={},
        compiler_manifest={},
        feasibility={},
        metrics={},
        evaluation=evaluation,
        predictions=(prediction,),
        baselines={},
        verdict="# failed",
        status=ReceiptStatus.FAILED,
        error="bounded failure",
        command=(),
    )
    metrics = output / "metrics.jsonl"
    metrics.write_text(
        metrics.read_text(encoding="utf-8") + '{"tampered":true}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_fit_attempt_artifacts(output)
