from __future__ import annotations

import copy
import json
import shutil
import subprocess
from itertools import pairwise
from pathlib import Path

import pytest
import yaml
from test_fit_artifacts import _formal_inputs

from tabu_lab.catalog import (
    PUBLIC_CATALOG_SCHEMA_MODELS,
    CatalogBuildError,
    CatalogEntry,
    CatalogObjectKind,
    CatalogSourceRevision,
    DatasetAdapter,
    DatasetSnapshotSpec,
    ExperimentRecord,
    ExperimentStatus,
    StatusEvent,
    build_catalog,
    check_catalog,
    generate_catalog_schema,
    load_catalog,
)
from tabu_lab.catalog.builder import _deduplicate_entries
from tabu_lab.cli import main
from tabu_lab.contracts.canonical import canonical_hash, canonical_json
from tabu_lab.evaluation.fit_artifacts import write_fit_attempt_artifacts
from tabu_lab.evaluation.foundry import (
    AdapterSpec,
    EvalProducerBinding,
    EvalResult,
    EvaluationFailure,
    EvaluationStatus,
    FailureCategory,
    TaskKind,
    load_suite,
)
from tabu_lab.evaluation.foundry.contracts import _result_id_from_components
from tabu_lab.evidence import RunIdentity, read_receipt
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationReplaySession,
    FormalAuthorizationSummary,
)
from tabu_lab.evidence.source_identity import SourceIdentity


def _write_snapshot(root: Path, *, contamination_boundary: str = "train-only") -> Path:
    snapshot = DatasetSnapshotSpec(
        dataset_snapshot_id="dataset-a-snapshot",
        dataset_id="dataset-a",
        source_uri="https://example.test/dataset-a.csv",
        source_sha256="a" * 64,
        content_sha256="b" * 64,
        license_id="CC-BY-4.0",
        split_manifest_sha256="c" * 64,
        fit_partition="train",
        adapter=DatasetAdapter(adapter_id="dataset-a", adapter_version="1.0.0"),
        mask_boundary="truth sidecar only",
        contamination_boundary=contamination_boundary,
    )
    path = root / "datasets/dataset-a.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(snapshot.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    return path


def _derived_snapshot_entry(
    snapshot: DatasetSnapshotSpec,
    *,
    source_path: str,
    source_hash: str = "d" * 64,
) -> CatalogEntry:
    return CatalogEntry(
        kind=CatalogObjectKind.DATASET_SNAPSHOT,
        object_id=snapshot.dataset_snapshot_id,
        object_schema_version=snapshot.schema_version,
        object_hash=canonical_hash(snapshot),
        source_hash=source_hash,
        source_path=source_path,
        data=snapshot.model_dump(mode="json"),
    )


def test_identical_preregistered_dataset_snapshots_coalesce_deterministically() -> None:
    snapshot = DatasetSnapshotSpec(
        dataset_snapshot_id="shared-dataset-snapshot",
        dataset_id="shared-dataset",
        source_uri="https://example.test/shared.csv",
        source_sha256="a" * 64,
        content_sha256="b" * 64,
        license_id="CC-BY-4.0",
        split_manifest_sha256="c" * 64,
        fit_partition="train",
        adapter=DatasetAdapter(adapter_id="shared-adapter", adapter_version="1.0.0"),
        mask_boundary="truth sidecar only",
        contamination_boundary="train-only",
    )
    later = _derived_snapshot_entry(
        snapshot,
        source_path="experiments/S1-002/preregistration.yaml",
    )
    earlier = _derived_snapshot_entry(
        snapshot,
        source_path="experiments/S1-001/preregistration.yaml",
    )

    forward = _deduplicate_entries((later, earlier))
    reverse = _deduplicate_entries((earlier, later))

    assert forward == reverse
    assert len(forward) == 1
    assert forward[0].source_path == "experiments/S1-001/preregistration.yaml"


def test_divergent_or_independently_declared_dataset_duplicates_fail_closed() -> None:
    snapshot = DatasetSnapshotSpec(
        dataset_snapshot_id="shared-dataset-snapshot",
        dataset_id="shared-dataset",
        source_uri="https://example.test/shared.csv",
        source_sha256="a" * 64,
        content_sha256="b" * 64,
        license_id="CC-BY-4.0",
        split_manifest_sha256="c" * 64,
        fit_partition="train",
        adapter=DatasetAdapter(adapter_id="shared-adapter", adapter_version="1.0.0"),
        mask_boundary="truth sidecar only",
        contamination_boundary="train-only",
    )
    first = _derived_snapshot_entry(
        snapshot,
        source_path="experiments/S1-001/preregistration.yaml",
    )
    changed = snapshot.model_copy(update={"contamination_boundary": "changed"})
    divergent = _derived_snapshot_entry(
        changed,
        source_path="experiments/S1-002/preregistration.yaml",
        source_hash="e" * 64,
    )
    with pytest.raises(CatalogBuildError, match="duplicate catalog object id"):
        _deduplicate_entries((first, divergent))

    declared_twice = (
        first.model_copy(update={"source_path": "datasets/shared-a.yaml"}),
        first.model_copy(update={"source_path": "datasets/shared-b.yaml"}),
    )
    with pytest.raises(CatalogBuildError, match="duplicate catalog object id"):
        _deduplicate_entries(declared_twice)


def test_build_is_byte_deterministic_and_loadable(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    first = build_catalog(tmp_path, "catalog.json")
    first_bytes = (tmp_path / "catalog.json").read_bytes()

    second = build_catalog(tmp_path, "catalog.json")
    second_bytes = (tmp_path / "catalog.json").read_bytes()

    assert first == second
    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n")
    assert load_catalog(tmp_path / "catalog.json") == first
    assert first.collections()[CatalogObjectKind.DATASET_SNAPSHOT.value][0].object_id == (
        "dataset-a-snapshot"
    )
    assert first.show("dataset-a-snapshot").source_path == "datasets/dataset-a.yaml"


def test_catalog_reads_current_model_alias_but_not_nested_registry_history(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    payload = (repository / "specs/models/tabu4rec.yaml").read_bytes()
    current = tmp_path / "specs/models/tabu4rec.yaml"
    history = tmp_path / "specs/models/tabu4rec/0.2.0.yaml"
    current.parent.mkdir(parents=True)
    history.parent.mkdir(parents=True)
    current.write_bytes(payload)
    history.write_bytes(payload)

    catalog = build_catalog(tmp_path)

    models = catalog.collections()[CatalogObjectKind.MODEL_CONTRACT.value]
    assert tuple(entry.object_id for entry in models) == ("tabu4rec",)
    assert models[0].source_path == "specs/models/tabu4rec.yaml"


def test_check_catalog_detects_canonical_source_hash_drift(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    build_catalog(tmp_path, "catalog.json")
    assert check_catalog(tmp_path).ok

    _write_snapshot(tmp_path, contamination_boundary="changed train-only policy")
    report = check_catalog(tmp_path)

    assert not report.ok
    assert report.issues[0].code == "catalog_invalid"
    assert "hash drift" in report.issues[0].message


def test_checked_catalog_json_has_stable_top_level_shape(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    catalog = build_catalog(tmp_path, "catalog.json")
    payload = json.loads((tmp_path / "catalog.json").read_text(encoding="utf-8"))

    assert tuple(sorted(payload)) == (
        "entries",
        "lineage",
        "schema_version",
        "source_revision",
        "source_tree_hash",
    )
    assert payload["schema_version"] == "tabu.catalog-index.v1"
    assert payload["source_revision"] is None
    assert payload["source_tree_hash"] == catalog.source_tree_hash


def test_catalog_source_revision_is_preserved_and_bound_during_check(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)
    base = build_catalog(tmp_path)
    revision = CatalogSourceRevision(
        repository_uri="https://github.com/1587causalai/TabU-lab.git",
        commit="a" * 40,
        catalog_source_tree_hash=base.source_tree_hash,
    )

    catalog = build_catalog(tmp_path, "catalog.json", source_revision=revision)

    assert catalog.source_revision is not None
    assert catalog.source_revision.repository_uri == ("https://github.com/1587causalai/TabU-lab")
    assert load_catalog(tmp_path / "catalog.json") == catalog
    assert check_catalog(tmp_path).ok

    _write_snapshot(tmp_path, contamination_boundary="changed train-only policy")
    report = check_catalog(tmp_path)
    assert not report.ok
    assert "does not bind source_tree_hash" in report.issues[0].message


def test_check_without_materialized_catalog_validates_sources(tmp_path: Path) -> None:
    _write_snapshot(tmp_path)

    report = check_catalog(tmp_path)

    assert report.ok
    assert report.catalog_hash is not None


def test_checked_in_catalog_schemas_match_runtime_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in PUBLIC_CATALOG_SCHEMA_MODELS:
        checked_in = json.loads((root / f"schemas/{name}.schema.json").read_text(encoding="utf-8"))
        assert checked_in == generate_catalog_schema(name)


def test_checked_in_f0_preregistrations_form_seven_explicit_revision_chains() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = build_catalog(root)
    expected_chains = {
        "tabuf": (
            "G000-tabuf-artificial-mask",
            "F0-001-tabuf-v1",
            "F0-008-tabuf-identifiable-v2",
        ),
        "tabu.unit_row": (
            "F0-002-tabu-unit-row-v1",
            "F0-009-tabu-unit-row-identifiable-v2",
        ),
        "tabu.unit_pair": (
            "F0-003-tabu-unit-pair-v1",
            "F0-010-tabu-unit-pair-identifiable-v2",
            "F0-020-tabu-unit-pair-local-linear-contract-v1",
        ),
        "tabu4graph": ("F0-006-tabu4graph-v1", "F0-011-tabu4graph-row-unit-v2"),
        "tabu4rec": (
            "F0-007-tabu4rec-v1",
            "F0-014-tabu4rec-axis-address-v2",
            "F0-021-tabu4rec-axis-address-wide-v1",
            "F0-022-tabu4rec-cell-global-support-v1",
        ),
        "tabul": (
            "F0-004-tabul-v1",
            "F0-012-tabul-predictor-address-v2",
            "F0-015-tabul-unit-linked-address-v3",
        ),
        "tabufl": (
            "F0-005-tabufl-v1",
            "F0-013-tabufl-independent-ledgers-v2",
            "F0-016-tabufl-independent-dynamics-v3",
            "F0-017-tabufl-balanced-joint-v4",
            "F0-018-tabufl-balanced-16f-v5",
        ),
    }
    experiments = {
        entry.object_id: ExperimentRecord.model_validate(entry.data)
        for entry in catalog.entries
        if entry.kind is CatalogObjectKind.EXPERIMENT
    }
    expected_f0_ids = {
        experiment_id for chain in expected_chains.values() for experiment_id in chain
    }
    actual_f0_ids = {
        experiment_id
        for experiment_id in experiments
        if experiment_id.startswith("F0-") or experiment_id == "G000-tabuf-artificial-mask"
    }
    assert actual_f0_ids == expected_f0_ids
    assert all(
        experiments[experiment_id].status is ExperimentStatus.DRAFT
        for experiment_id in expected_f0_ids
    )

    for contract_id, chain in expected_chains.items():
        for predecessor, successor in pairwise(chain):
            assert experiments[successor].contract_id == contract_id
            assert experiments[successor].supersedes_experiment_ids == (predecessor,)
            assert experiments[successor].revision_rationale is not None

    superseded = {
        edge.target.object_id
        for edge in catalog.lineage
        if edge.relation.value == "supersedes" and edge.target.kind is CatalogObjectKind.EXPERIMENT
    }
    active = actual_f0_ids - superseded
    assert active == {chain[-1] for chain in expected_chains.values()}


def test_builder_rejects_unknown_schema_in_a_canonical_source_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "datasets/unknown.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "schema_version": "tabu.catalog-unknown.v999",
                "object_id": "must-not-enter-catalog",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CatalogBuildError, match="unsupported catalog source schema_version"):
        build_catalog(tmp_path)


def test_builder_rejects_future_eval_result_schema_instead_of_guessing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "evaluations/results/future-result.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "schema_version": "tabu.eval-result.v999",
                "result_id": "future-result",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CatalogBuildError, match=r"tabu\.eval-result\.v999"):
        build_catalog(tmp_path)


def test_unissued_eval_result_cannot_enter_catalog(tmp_path: Path) -> None:
    adapter = AdapterSpec(
        adapter_id="train-only-mean",
        adapter_version="1.0.0",
        kind="baseline",
        fit_iterations=0,
        device_class="single_device",
        deterministic=True,
        baseline_family="mean",
    )
    producer = EvalProducerBinding(
        provenance="unissued_baseline",
        publication_eligible=False,
    )
    failure = EvaluationFailure(
        category=FailureCategory.INFRASTRUCTURE,
        code="unissued-pilot",
        public_detail="train-validation pilot only",
    )
    counts = {"targets": 0, "scored": 0, "abstained": 0}
    failure_counts = {FailureCategory.INFRASTRUCTURE: 1}
    claim_boundary = "unissued baseline pilot; not public evaluation evidence"
    result_id = _result_id_from_components(
        suite_id="table-completion-micro-v0",
        suite_version="0.1.0",
        suite_hash="a" * 64,
        scenario_id="adult-completion",
        task=TaskKind.TABLE_COMPLETION,
        adapter=adapter,
        producer=producer,
        seed=1729,
        source_sha256="b" * 64,
        split_sha256="c" * 64,
        recipe_sha256="d" * 64,
        budget_hash="e" * 64,
        status=EvaluationStatus.FAILED,
        raw_predictions=(),
        topology_checks=(),
        per_example=(),
        metrics={},
        counts=counts,
        failure_counts=failure_counts,
        coverage=0.0,
        failure=failure,
        claim_boundary=claim_boundary,
    )
    result = EvalResult(
        result_id=result_id,
        status=EvaluationStatus.FAILED,
        suite_id="table-completion-micro-v0",
        suite_version="0.1.0",
        suite_hash="a" * 64,
        scenario_id="adult-completion",
        task=TaskKind.TABLE_COMPLETION,
        adapter=adapter,
        producer=producer,
        seed=1729,
        source_sha256="b" * 64,
        split_sha256="c" * 64,
        recipe_sha256="d" * 64,
        budget_hash="e" * 64,
        counts=counts,
        failure_counts=failure_counts,
        coverage=0.0,
        failure=failure,
        claim_boundary=claim_boundary,
    )
    source = tmp_path / "evaluations/results/unissued-result.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(result.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(CatalogBuildError, match="immutable producer receipt"):
        build_catalog(tmp_path)


def _materialize_fit_receipt(
    root: Path,
    *,
    issuance_status: str,
) -> tuple[Path, str, FormalAuthorizationContext | None]:
    inputs, spec = _formal_inputs(root)
    inputs = dict(inputs)
    authorization_context = inputs.get("formal_authorization")
    assert isinstance(authorization_context, FormalAuthorizationContext)
    if issuance_status == "local_unissued":
        resolved_configs = dict(inputs["resolved_configs"])
        code = copy.deepcopy(resolved_configs["code"])
        source_identity = copy.deepcopy(code["source_identity"])
        source_identity.update(
            issuance_status="local_unissued",
            reviewed=False,
            reasons=("formal_issuance_not_requested",),
        )
        code["source_identity"] = source_identity
        resolved_configs["code"] = code
        previous = inputs["run_identity"]
        assert isinstance(previous, RunIdentity)
        identity = RunIdentity.create(
            spec_hash=previous.spec_hash,
            code_hash=canonical_hash(code),
            data_hash=previous.data_hash,
            split_hash=previous.split_hash,
            compiler_hash=previous.compiler_hash,
            semantic_config_hash=previous.semantic_config_hash,
            execution_config_hash=previous.execution_config_hash,
            training_config_hash=previous.training_config_hash,
            seeds=dict(previous.seeds),
        )
        metadata = copy.deepcopy(inputs["metadata"])
        metadata.update(
            code_hash=identity.code_hash,
            issuance_status="local_unissued",
            source_identity_hash=canonical_hash(source_identity),
        )
        inputs.update(
            resolved_configs=resolved_configs,
            run_identity=identity,
            metadata=metadata,
        )
        inputs.pop("formal_authorization", None)
        authorization_context = None
    elif issuance_status != "formal":
        raise ValueError(f"unsupported test issuance status: {issuance_status}")

    identity = inputs["run_identity"]
    assert isinstance(identity, RunIdentity)
    run_id = identity.run_id
    attempt_id = str(inputs["attempt_id"])
    catalog_root = (
        authorization_context.repository
        if issuance_status == "formal" and authorization_context is not None
        else root
    )
    attempt = catalog_root / "runs" / run_id / attempt_id
    write_target = (
        catalog_root / ".local-runs" / run_id / attempt_id
        if issuance_status == "formal"
        else attempt
    )
    written = write_fit_attempt_artifacts(write_target, **inputs)  # type: ignore[arg-type]

    if issuance_status == "formal":
        assert authorization_context is not None
        attempt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(write_target, attempt)
        record_path = catalog_root / "experiments/records" / f"{spec.experiment_id}.json"
        record = ExperimentRecord.model_validate_json(record_path.read_bytes())
        succeeded = record.model_copy(
            update={
                "status": ExperimentStatus.SUCCEEDED,
                "status_history": (
                    *record.status_history,
                    StatusEvent(status=ExperimentStatus.RUNNING.value),
                    StatusEvent(
                        status=ExperimentStatus.SUCCEEDED.value,
                        evidence_hashes=(written.receipt_hash,),
                    ),
                ),
                "run_ids": (run_id,),
            }
        )
        record_path.write_text(canonical_json(succeeded) + "\n", encoding="utf-8")
        return attempt, run_id, authorization_context

    preregistration = (
        root / "experiments" / "fit-first" / "F0" / spec.experiment_id / "preregistration.yaml"
    )
    preregistration.parent.mkdir(parents=True, exist_ok=True)
    preregistration.write_text(str(inputs["preregistration_text"]), encoding="utf-8")
    model = root / "specs/models/tabuf.yaml"
    model.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    shutil.copyfile(repository / "specs/models/tabuf.yaml", model)
    legacy_preregistration = root / "experiments/G000-tabuf-artificial-mask/preregistration.yaml"
    legacy_preregistration.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        repository / "experiments/G000-tabuf-artificial-mask/preregistration.yaml",
        legacy_preregistration,
    )
    return attempt, run_id, authorization_context


def test_real_formal_receipt_envelope_projects_run_attempt_and_artifact(
    tmp_path: Path,
) -> None:
    attempt, run_id, authorization = _materialize_fit_receipt(tmp_path, issuance_status="formal")
    assert authorization is not None
    catalog_root = authorization.repository

    catalog = build_catalog(catalog_root)

    receipt_entries = catalog.collections()[CatalogObjectKind.RECEIPT.value]
    artifact_entries = catalog.collections()[CatalogObjectKind.MODEL_ARTIFACT.value]
    run = catalog.show(run_id)
    assert len(receipt_entries) == 1
    assert (
        receipt_entries[0].source_path
        == attempt.joinpath("receipt.json").relative_to(catalog_root).as_posix()
    )
    assert len(artifact_entries) == 1
    assert artifact_entries[0].data["producer_run_id"] == run_id
    assert (
        artifact_entries[0].data["checkpoint"]["uri"].endswith("/checkpoint/checkpoint.safetensors")
    )
    assert run.data["artifact_ids"] == [artifact_entries[0].object_id]


def test_public_eval_result_requires_independent_eval_receipt_before_dataset_snapshot(
    tmp_path: Path,
) -> None:
    attempt, run_id, authorization = _materialize_fit_receipt(tmp_path, issuance_status="formal")
    assert authorization is not None
    catalog_root = authorization.repository
    receipt = read_receipt(attempt / "receipt.json")
    suite = load_suite("table-completion-micro-v0")
    scenario = next(
        item for item in suite.scenarios if item.scenario_id == "adult-v2-feature-completion-micro"
    )
    suite_path = catalog_root / "evaluations/suites/table-completion-micro-v0.yaml"
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(
        yaml.safe_dump(suite.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )

    source_hash = "a" * 64
    split_hash = "b" * 64
    recipe_hash = "c" * 64
    result_truth_hash = "d" * 64
    snapshot = DatasetSnapshotSpec(
        dataset_snapshot_id="adult-completion-canonical-snapshot",
        dataset_id=scenario.dataset.dataset_id,
        source_uri=scenario.dataset.source_uri,
        source_sha256=source_hash,
        content_sha256="e" * 64,
        license_id=scenario.dataset.license_id,
        split_manifest_sha256=split_hash,
        fit_partition="train",
        adapter=DatasetAdapter(adapter_id="adult-eval", adapter_version="1.0.0"),
        episode_recipe_hashes=(recipe_hash,),
        evaluation_scenario_id=scenario.scenario_id,
        truth_sidecar_sha256="f" * 64,
        mask_boundary="post-split artificial masking; truth remains evaluator-side",
        contamination_boundary="all transforms are fitted on train only",
    )
    snapshot_path = catalog_root / "datasets/adult-completion-canonical-snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    producer = EvalProducerBinding(
        provenance="receipted_run",
        run_id=run_id,
        receipt_sha256=receipt.receipt_hash,
        receipt_pointer=(attempt / "receipt.json").relative_to(catalog_root).as_posix(),
        publication_eligible=True,
    )
    adapter = AdapterSpec(
        adapter_id="tabuf-checkpoint",
        adapter_version="1.0.0",
        kind="model",
        fit_iterations=1,
        device_class="single_device",
        deterministic=True,
        contract_id="tabuf",
        artifact_id=f"{run_id}.{attempt.name}.checkpoint",
    )
    failure = EvaluationFailure(
        category=FailureCategory.MODEL,
        code="model-output-invalid",
        public_detail="model output failed the frozen evaluator contract",
    )
    counts = {"targets": 0, "scored": 0, "abstained": 0}
    failure_counts = {FailureCategory.MODEL: 1}
    claim_boundary = "failed micro evaluation only; no benchmark claim"
    result_id = _result_id_from_components(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        adapter=adapter,
        producer=producer,
        seed=suite.budget.model_seeds[0],
        source_sha256=source_hash,
        split_sha256=split_hash,
        recipe_sha256=recipe_hash,
        budget_hash=suite.budget.content_hash,
        truth_sidecar_sha256=result_truth_hash,
        status=EvaluationStatus.FAILED,
        raw_predictions=(),
        topology_checks=(),
        per_example=(),
        metrics={},
        counts=counts,
        failure_counts=failure_counts,
        coverage=0.0,
        failure=failure,
        claim_boundary=claim_boundary,
    )
    result = EvalResult(
        result_id=result_id,
        status=EvaluationStatus.FAILED,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        adapter=adapter,
        producer=producer,
        seed=suite.budget.model_seeds[0],
        source_sha256=source_hash,
        split_sha256=split_hash,
        recipe_sha256=recipe_hash,
        budget_hash=suite.budget.content_hash,
        truth_sidecar_sha256=result_truth_hash,
        counts=counts,
        failure_counts=failure_counts,
        coverage=0.0,
        failure=failure,
        claim_boundary=claim_boundary,
    )
    result_path = catalog_root / "evaluations/results/model-failure.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(CatalogBuildError, match="own formal evaluation receipt"):
        build_catalog(catalog_root)

    corrected = snapshot.model_copy(update={"truth_sidecar_sha256": result_truth_hash})
    snapshot_path.write_text(
        json.dumps(corrected.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    # Dataset correction cannot bypass the earlier execution-evidence gate.
    # The downstream lineage assertion returns when the replayable formal eval
    # authorization writer is implemented.
    with pytest.raises(CatalogBuildError, match="own formal evaluation receipt"):
        build_catalog(catalog_root)


def test_runs_verify_checks_typed_receipt_hash_not_envelope_file_hash(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attempt, run_id, authorization = _materialize_fit_receipt(tmp_path, issuance_status="formal")
    assert authorization is not None
    catalog_root = authorization.repository
    catalog_path = catalog_root / "catalog-current.json"
    catalog = build_catalog(catalog_root, catalog_path.name)
    pointer = catalog.show(run_id).data["receipt"]

    assert (
        main(
            [
                "runs",
                "verify",
                run_id,
                "--catalog",
                str(catalog_path),
                "--receipt-file",
                str(attempt / "receipt.json"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["content_verified"] is True
    assert payload["receipt_hash"] == pointer["sha256"]


def test_real_receipt_projection_rejects_checksum_tamper(tmp_path: Path) -> None:
    attempt, _, authorization = _materialize_fit_receipt(tmp_path, issuance_status="formal")
    assert authorization is not None
    checkpoint = attempt / "checkpoint/checkpoint.safetensors"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")

    with pytest.raises(CatalogBuildError, match="checksum manifest is invalid or has drift"):
        build_catalog(authorization.repository)


def test_formal_authorization_replays_multiple_git_generations(tmp_path: Path) -> None:
    attempt, _, authorization = _materialize_fit_receipt(
        tmp_path,
        issuance_status="formal",
    )
    assert authorization is not None
    repository = authorization.repository
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "record first formal receipt sources"),
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
    previous_revision = load_catalog(repository / "catalog.json").source_revision
    assert previous_revision is not None
    generation_two = build_catalog(
        repository,
        "catalog.json",
        source_revision=CatalogSourceRevision(
            repository_uri=previous_revision.repository_uri,
            commit=revision_commit,
            catalog_source_tree_hash=draft.source_tree_hash,
        ),
    )
    subprocess.run(("git", "-C", repository, "add", "catalog.json"), check=True)
    subprocess.run(
        ("git", "-C", repository, "commit", "-m", "freeze second catalog"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ("git", "-C", repository, "push", "origin", "main"),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", repository, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    receipt = read_receipt(attempt / "receipt.json")
    generation_one = FormalAuthorizationSummary.model_validate(
        receipt.metadata["formal_authorization"]
    )
    generation_two_summary = generation_one.model_copy(
        update={
            "canonical_commit": commit,
            "catalog_hash": generation_two.catalog_hash,
            "catalog_source_tree_hash": generation_two.source_tree_hash,
            "experiment_status": ExperimentStatus.SUCCEEDED.value,
        }
    )
    code = json.loads((attempt / "resolved-configs/code.json").read_text(encoding="utf-8"))
    source_identity = SourceIdentity.model_validate(code["source_identity"])
    preregistration = (attempt / "preregistration.yaml").read_text(encoding="utf-8")
    replay = FormalAuthorizationReplaySession(repository)

    verified = replay.verify(
        generation_two_summary,
        preregistration_text=preregistration,
        live_source_identity=source_identity,
    )
    cached = replay.verify(
        generation_two_summary,
        preregistration_text=preregistration,
        live_source_identity=source_identity,
    )

    assert verified.summary == generation_two_summary
    assert cached is verified


def test_local_unissued_success_receipt_does_not_project_model_artifact(
    tmp_path: Path,
) -> None:
    _, run_id, _ = _materialize_fit_receipt(tmp_path, issuance_status="local_unissued")

    catalog = build_catalog(tmp_path)

    assert catalog.collections()[CatalogObjectKind.MODEL_ARTIFACT.value] == ()
    assert catalog.show(run_id).data["artifact_ids"] == []
    receipt = catalog.collections()[CatalogObjectKind.RECEIPT.value][0]
    assert receipt.data["metadata"]["issuance_status"] == "local_unissued"


def test_public_catalog_rejects_formal_receipt_without_live_remote_authority(
    tmp_path: Path,
) -> None:
    _, _, authorization = _materialize_fit_receipt(tmp_path, issuance_status="formal")
    assert authorization is not None
    subprocess.run(
        ("git", "-C", authorization.repository, "remote", "remove", "origin"),
        check=True,
        capture_output=True,
    )

    with pytest.raises(
        CatalogBuildError,
        match="complete artifact and authorization replay",
    ):
        build_catalog(authorization.repository)
