from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import tabu_lab.evaluation.formal_receipt as formal_receipt_module
import tabu_lab.evaluation.foundry.runner as foundry_runner_module
from tabu_lab.catalog import (
    CatalogEntry,
    CatalogIndex,
    CatalogObjectKind,
    DatasetAdapter,
    DatasetAuthorityStatus,
    DatasetSnapshotSpec,
    EvidencePointer,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    build_catalog,
)
from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.formal_receipt import (
    FormalEvaluationReceiptError,
    issue_formal_evaluation_receipt,
)
from tabu_lab.evaluation.foundry import (
    AdapterSpec,
    ComparisonReport,
    EvalProducerBinding,
    EvalResult,
    EvaluationFailure,
    EvaluationStatus,
    EvaluatorSourceAuthorization,
    FailureCategory,
    TaskKind,
    bind_evaluation_receipt,
    compare_results,
    load_suite,
)
from tabu_lab.evaluation.foundry.contracts import (
    _bind_evaluation_receipt,
    _result_id_from_components,
)
from tabu_lab.evidence import EnvironmentDisclosure, SourceIdentity
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    FormalAuthorizationSummary,
    VerifiedFormalAuthorization,
    verify_committed_evidence_pointer,
)

HASHES = tuple(character * 64 for character in "abcdef123456789")
STARTED_AT = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)


def _environment(*, operating_system: str = "Darwin") -> EnvironmentDisclosure:
    return EnvironmentDisclosure(
        environment_hash=HASHES[6],
        host_class="cpu-host",
        operating_system=operating_system,
        device="cpu",
        architecture="arm64",
        accelerator=None,
        python_version="3.11.14",
    )


def _source_identity(*, formal: bool) -> SourceIdentity:
    if not formal:
        return SourceIdentity(
            source_kind="local",
            issuance_status="local_unissued",
            reasons=("formal_evaluation_source_identity_not_provided",),
        )
    return SourceIdentity(
        source_kind="git",
        issuance_status="formal",
        reviewed=True,
        repository_uri="https://github.com/wehub-community/tabu-lab.git",
        repository_subdirectory=".",
        commit="1" * 40,
        remote_ref="refs/remotes/origin/main",
        git_tree_oid="2" * 40,
        source_tree_hash=HASHES[7],
        preregistration_blob_hash=HASHES[8],
        lock_hash=HASHES[9],
    )


def _failed_baseline_result(
    *,
    suite_backed: bool = False,
    bind_truth_sidecar: bool = False,
    baseline_index: int = 0,
    seed: int | None = None,
    fit_iterations: int = 0,
    deterministic: bool = True,
    claim_boundary: str | None = None,
) -> EvalResult:
    if suite_backed:
        suite = load_suite("table-completion-micro-v0")
        scenario = suite.scenarios[0]
        baseline = scenario.baselines[baseline_index]
        suite_id = suite.suite_id
        suite_version = suite.suite_version
        suite_hash = suite.suite_hash
        scenario_id = scenario.scenario_id
        task = scenario.task
        seed = suite.budget.model_seeds[0] if seed is None else seed
        budget_hash = suite.budget.content_hash
        adapter_id = baseline.baseline_id
        baseline_family = baseline.family
        claim_boundary = claim_boundary or suite.claim_boundary
    else:
        suite_id = "test-suite"
        suite_version = "0.1.0"
        suite_hash = HASHES[0]
        scenario_id = "test-scenario"
        task = TaskKind.TABLE_COMPLETION
        seed = 1729 if seed is None else seed
        budget_hash = HASHES[4]
        adapter_id = "train-only-mean"
        baseline_family = "mean"
        claim_boundary = claim_boundary or "failed micro evaluation only; no benchmark claim"
    adapter = AdapterSpec(
        adapter_id=adapter_id,
        adapter_version="1.0.0",
        kind="baseline",
        fit_iterations=fit_iterations,
        device_class="single_device",
        deterministic=deterministic,
        baseline_family=baseline_family,
    )
    producer = EvalProducerBinding(
        provenance="unissued_baseline",
        publication_eligible=False,
    )
    failure = EvaluationFailure(
        category=FailureCategory.INFRASTRUCTURE,
        code="worker-exited",
        public_detail="isolated evaluator worker exited before returning predictions",
    )
    values = {
        "suite_id": suite_id,
        "suite_version": suite_version,
        "suite_hash": suite_hash,
        "scenario_id": scenario_id,
        "task": task,
        "adapter": adapter,
        "producer": producer,
        "seed": seed,
        "source_sha256": HASHES[1],
        "split_sha256": HASHES[2],
        "recipe_sha256": HASHES[3],
        "budget_hash": budget_hash,
        "truth_sidecar_sha256": HASHES[5] if bind_truth_sidecar else None,
        "status": EvaluationStatus.FAILED,
        "raw_predictions": (),
        "topology_checks": (),
        "per_example": (),
        "metrics": {},
        "counts": {"targets": 0, "scored": 0, "abstained": 0},
        "failure_counts": {FailureCategory.INFRASTRUCTURE: 1},
        "coverage": 0.0,
        "failure": failure,
        "claim_boundary": claim_boundary,
    }
    return EvalResult(
        result_id=_result_id_from_components(**values),
        **values,
    )


def _formal_baseline_fixture(tmp_path: Path):
    suite = load_suite("table-completion-micro-v0")
    scenario = suite.scenarios[0]
    result = _failed_baseline_result(
        suite_backed=True,
        bind_truth_sidecar=True,
    )
    prepared_snapshot = DatasetSnapshotSpec(
        schema_version="tabu.dataset-snapshot.v3",
        dataset_snapshot_id="formal-eval-dataset-snapshot",
        dataset_id=scenario.dataset.dataset_id,
        source_uri=scenario.dataset.source_uri,
        source_sha256=result.source_sha256,
        content_sha256=HASHES[10],
        license_id=scenario.dataset.license_id,
        split_manifest_sha256=result.split_sha256,
        fit_partition="train",
        adapter=DatasetAdapter(
            adapter_id="offline-eval-prepared-snapshot",
            adapter_version="1.0.0",
        ),
        episode_recipe_hashes=(result.recipe_sha256,),
        evaluation_scenario_id=result.scenario_id,
        truth_sidecar_sha256=result.truth_sidecar_sha256,
        request_sha256=HASHES[11],
        authority_sha256=HASHES[12],
        authority_status=DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED,
        mask_boundary="post-split masking with evaluator-side truth",
        contamination_boundary="split before train-only preparation",
    )
    reviewed_payload = prepared_snapshot.model_dump(mode="python")
    reviewed_payload.update(
        authority_status=DatasetAuthorityStatus.REVIEWED,
        review_ids=("review-formal-eval-dataset",),
    )
    reviewed_snapshot = DatasetSnapshotSpec.model_validate(reviewed_payload)
    review = ReviewRecord(
        review_id="review-formal-eval-dataset",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.DATASET_SNAPSHOT,
                object_id=reviewed_snapshot.dataset_snapshot_id,
                evidence_sha256=prepared_snapshot.content_hash,
            ),
        ),
        developer_identity="dataset-developer",
        reviewer_identity="independent-dataset-reviewer",
        decision=ReviewDecision.APPROVED,
        report=EvidencePointer(
            uri="reviews/formal-eval-dataset-report.json",
            sha256=HASHES[13],
        ),
    )
    sources = (
        (
            tmp_path / "evaluations/suites/table-completion-micro-v0.yaml",
            yaml.safe_dump(suite.model_dump(mode="json"), sort_keys=False),
        ),
        (
            tmp_path / "datasets/formal-eval-dataset-snapshot.json",
            json.dumps(reviewed_snapshot.model_dump(mode="json"), sort_keys=True),
        ),
        (
            tmp_path / "reviews/review-formal-eval-dataset.json",
            json.dumps(review.model_dump(mode="json"), sort_keys=True),
        ),
    )
    for path, text in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    catalog = build_catalog(tmp_path)
    source_identity = _source_identity(formal=True)
    summary = FormalAuthorizationSummary(
        canonical_commit="1" * 40,
        catalog_hash=catalog.catalog_hash,
        catalog_source_tree_hash=catalog.source_tree_hash,
        experiment_id="evaluator-source-carrier",
        experiment_status="runnable",
        preregistration_sha256=HASHES[2],
        source_identity_sha256=canonical_hash(source_identity),
        review_ids=("review-evaluator-source",),
        review_report_sha256s=(HASHES[3],),
        gong_approval_sha256s=(HASHES[4],),
    )
    context = FormalAuthorizationContext(
        repository=tmp_path / "source-authority-repository",
        catalog=tmp_path / "source-authority-repository/catalog.json",
        experiment_id=summary.experiment_id,
    )
    return (
        result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        context,
    )


def test_failed_baseline_gets_independent_local_evaluation_receipt() -> None:
    result = _failed_baseline_result()
    bound = bind_evaluation_receipt(
        result,
        environment=_environment(),
        source_identity=_source_identity(formal=False),
        started_at=STARTED_AT,
        completed_at=STARTED_AT + timedelta(seconds=2),
    )

    receipt = bound.execution_receipt
    assert receipt is not None
    assert receipt.status is EvaluationStatus.FAILED
    assert receipt.failure_category is FailureCategory.INFRASTRUCTURE
    assert receipt.result_content_sha256 == result.result_content_hash
    assert receipt.adapter_sha256 == result.adapter.content_hash
    assert receipt.producer_sha256 == result.producer.content_hash
    assert receipt.issuance_status == "local_unissued"
    assert not receipt.publication_eligible
    assert bound.content_hash != bound.result_content_hash


def test_baseline_receipt_does_not_require_a_training_producer() -> None:
    result = _failed_baseline_result()
    bound = bind_evaluation_receipt(
        result,
        environment=_environment(),
        source_identity=_source_identity(formal=False),
        started_at=STARTED_AT,
        completed_at=STARTED_AT + timedelta(seconds=2),
    )

    assert result.producer.run_id is None
    assert bound.execution_receipt is not None
    assert not bound.execution_receipt.publication_eligible


def test_self_declared_formal_evaluation_receipt_is_forbidden() -> None:
    with pytest.raises(ValueError, match="self-declared formal evaluation source"):
        bind_evaluation_receipt(
            _failed_baseline_result(),
            environment=_environment(),
            source_identity=_source_identity(formal=True),
            started_at=STARTED_AT,
            completed_at=STARTED_AT + timedelta(seconds=2),
        )


def test_execution_receipt_tamper_and_private_path_fail_closed() -> None:
    result = _failed_baseline_result()
    bound = bind_evaluation_receipt(
        result,
        environment=_environment(),
        source_identity=_source_identity(formal=False),
        started_at=STARTED_AT,
        completed_at=STARTED_AT + timedelta(seconds=2),
    )
    payload = bound.model_dump(mode="python")
    payload["execution_receipt"]["result_content_sha256"] = HASHES[10]
    with pytest.raises(ValidationError, match=r"receipt_id|exact result"):
        EvalResult.model_validate(payload)

    with pytest.raises(ValidationError, match="absolute local path"):
        bind_evaluation_receipt(
            result,
            environment=_environment(operating_system="Darwin /private/var/build"),
            source_identity=_source_identity(formal=False),
            started_at=STARTED_AT,
            completed_at=STARTED_AT + timedelta(seconds=2),
        )

    with pytest.raises(ValidationError, match="host identity or secret"):
        bind_evaluation_receipt(
            result,
            environment=_environment(
                operating_system="Authorization: Bearer abcdefghijklmnopqrstuvwxyz"
            ),
            source_identity=_source_identity(formal=False),
            started_at=STARTED_AT,
            completed_at=STARTED_AT + timedelta(seconds=2),
        )


def test_catalog_rejects_local_baseline_even_with_its_eval_receipt(tmp_path: Path) -> None:
    suite = load_suite("table-completion-micro-v0")
    suite_path = tmp_path / "evaluations/suites/table-completion-micro-v0.yaml"
    suite_path.parent.mkdir(parents=True)
    suite_path.write_text(
        yaml.safe_dump(suite.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    result = bind_evaluation_receipt(
        _failed_baseline_result(suite_backed=True),
        environment=_environment(),
        source_identity=_source_identity(formal=False),
        started_at=STARTED_AT,
        completed_at=STARTED_AT + timedelta(seconds=2),
    )
    result_path = tmp_path / "evaluations/results/formal-baseline-failure.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(result.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="own formal evaluation receipt"):
        build_catalog(tmp_path)


def test_git_backed_formal_baseline_receipt_binds_all_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)
    monkeypatch.setattr(
        formal_receipt_module,
        "verify_formal_authorization",
        lambda *args, **kwargs: VerifiedFormalAuthorization(
            summary=summary,
            source_identity=source_identity,
            catalog=catalog,
        ),
    )
    monkeypatch.setattr(
        formal_receipt_module,
        "verify_committed_evidence_pointer",
        lambda *args, **kwargs: kwargs["sha256"],
    )

    bound = issue_formal_evaluation_receipt(
        result,
        environment=_environment(),
        live_source_identity=source_identity,
        started_at=STARTED_AT,
        completed_at=STARTED_AT + timedelta(seconds=2),
        source_authorization_context=context,
        preregistration_text="schema_version: source-authority-fixture\n",
        catalog=catalog,
        dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
        prepared_snapshot=prepared_snapshot,
    )

    receipt = bound.execution_receipt
    assert receipt is not None
    assert receipt.issuance_status == "formal"
    assert receipt.publication_eligible
    assert receipt.dataset_snapshot_sha256 == reviewed_snapshot.content_hash
    assert receipt.dataset_request_sha256 == prepared_snapshot.request_sha256
    assert receipt.dataset_authority_sha256 == prepared_snapshot.authority_sha256
    assert receipt.baseline_spec_sha256 == receipt.producer_sha256
    assert receipt.source_authorization is not None
    assert receipt.source_authorization.purpose == "evaluator_source"


def test_formal_eval_rejects_catalog_swapped_after_source_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        result,
        authorized_catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)
    unrelated = DatasetSnapshotSpec(
        dataset_snapshot_id="unrelated-dataset-snapshot",
        dataset_id="unrelated-dataset",
        source_uri="https://example.test/unrelated.csv",
        source_sha256=HASHES[0],
        content_sha256=HASHES[1],
        license_id="CC-BY-4.0",
        split_manifest_sha256=HASHES[2],
        fit_partition="train",
        adapter=DatasetAdapter(
            adapter_id="unrelated-dataset",
            adapter_version="1.0.0",
        ),
        mask_boundary="no evaluator masking",
        contamination_boundary="train-only preprocessing",
    )
    unrelated_path = tmp_path / "datasets/unrelated-dataset-snapshot.json"
    unrelated_path.write_text(
        json.dumps(unrelated.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    swapped_catalog = build_catalog(tmp_path)
    assert swapped_catalog.catalog_hash != authorized_catalog.catalog_hash
    monkeypatch.setattr(
        formal_receipt_module,
        "verify_formal_authorization",
        lambda *args, **kwargs: VerifiedFormalAuthorization(
            summary=summary,
            source_identity=source_identity,
            catalog=authorized_catalog,
        ),
    )

    with pytest.raises(
        FormalEvaluationReceiptError,
        match="evaluation catalog differs",
    ):
        issue_formal_evaluation_receipt(
            result,
            environment=_environment(),
            live_source_identity=source_identity,
            started_at=STARTED_AT,
            source_authorization_context=context,
            preregistration_text="schema_version: source-authority-fixture\n",
            catalog=swapped_catalog,
            dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
            prepared_snapshot=prepared_snapshot,
        )


def test_formal_eval_rejects_missing_committed_dataset_review_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)
    monkeypatch.setattr(
        formal_receipt_module,
        "verify_formal_authorization",
        lambda *args, **kwargs: VerifiedFormalAuthorization(
            summary=summary,
            source_identity=source_identity,
            catalog=catalog,
        ),
    )

    with pytest.raises(
        FormalEvaluationReceiptError,
        match="not an exact committed evidence blob",
    ):
        issue_formal_evaluation_receipt(
            result,
            environment=_environment(),
            live_source_identity=source_identity,
            started_at=STARTED_AT,
            source_authorization_context=context,
            preregistration_text="schema_version: source-authority-fixture\n",
            catalog=catalog,
            dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
            prepared_snapshot=prepared_snapshot,
        )


@pytest.mark.parametrize(
    ("result_overrides", "message"),
    (
        (
            {"claim_boundary": "foundation model and benchmark evidence"},
            "claim boundary differs",
        ),
        ({"fit_iterations": 999}, "fit iterations exceed"),
        ({"deterministic": False}, "deterministic execution contract"),
    ),
)
def test_formal_eval_rechecks_claim_and_adapter_budget_against_suite(
    tmp_path: Path,
    result_overrides: dict[str, object],
    message: str,
) -> None:
    (
        _result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        _summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)
    result = _failed_baseline_result(
        suite_backed=True,
        bind_truth_sidecar=True,
        **result_overrides,
    )

    with pytest.raises(FormalEvaluationReceiptError, match=message):
        issue_formal_evaluation_receipt(
            result,
            environment=_environment(),
            live_source_identity=source_identity,
            started_at=STARTED_AT,
            source_authorization_context=context,
            preregistration_text="schema_version: source-authority-fixture\n",
            catalog=catalog,
            dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
            prepared_snapshot=prepared_snapshot,
        )


def test_committed_evidence_pointer_uses_exact_blob_and_canonical_mapping_hash(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "authority-repository"
    repository.mkdir()
    subprocess.run(("git", "init", str(repository)), check=True, capture_output=True)
    report = repository / "reviews/dataset-report.json"
    report.parent.mkdir(parents=True)
    report_payload = {"decision": "approved", "reviewer": "independent"}
    report.write_text(
        json.dumps(report_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        ("git", "-C", str(repository), "add", "reviews/dataset-report.json"),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=TabU Test",
            "-c",
            "user.email=tabu-test@example.invalid",
            "commit",
            "-m",
            "add dataset review report",
        ),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = canonical_hash(report_payload)

    assert (
        verify_committed_evidence_pointer(
            repository,
            commit,
            uri="reviews/dataset-report.json",
            sha256=digest,
            label="dataset review report",
        )
        == digest
    )
    with pytest.raises(FormalAuthorizationError, match="digest differs"):
        verify_committed_evidence_pointer(
            repository,
            commit,
            uri="reviews/dataset-report.json",
            sha256=HASHES[0],
            label="dataset review report",
        )
    with pytest.raises(FormalAuthorizationError, match="missing"):
        verify_committed_evidence_pointer(
            repository,
            commit,
            uri="reviews/uncommitted-report.json",
            sha256=digest,
            label="dataset review report",
        )


def test_comparison_publication_eligibility_uses_formal_eval_receipts_for_baselines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        _context,
    ) = _formal_baseline_fixture(tmp_path)
    summary_payload = summary.model_dump(mode="python")
    authorization_schema_version = summary_payload.pop("schema_version")
    source_authorization = EvaluatorSourceAuthorization(
        authorization_schema_version=authorization_schema_version,
        **summary_payload,
    )
    suite = load_suite("table-completion-micro-v0")
    scenario = suite.scenarios[0]
    results = []
    for baseline_index, baseline in enumerate(scenario.baselines):
        for seed in suite.budget.model_seeds:
            result = _failed_baseline_result(
                suite_backed=True,
                bind_truth_sidecar=True,
                baseline_index=baseline_index,
                seed=seed,
            )
            results.append(
                _bind_evaluation_receipt(
                    result,
                    environment=_environment(),
                    source_identity=source_identity,
                    source_authorization=source_authorization,
                    producer_sha256=baseline.content_hash,
                    baseline_spec_sha256=baseline.content_hash,
                    dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
                    dataset_snapshot_sha256=reviewed_snapshot.content_hash,
                    dataset_request_sha256=prepared_snapshot.request_sha256,
                    dataset_authority_sha256=prepared_snapshot.authority_sha256,
                    dataset_authority_status="reviewed",
                    started_at=STARTED_AT,
                    completed_at=STARTED_AT + timedelta(seconds=2),
                )
            )
    assert all(not result.producer.publication_eligible for result in results)
    monkeypatch.setattr(
        foundry_runner_module,
        "verify_result_against_prepared",
        lambda suite, *, result, prepared: result,
    )

    comparison = compare_results(
        suite,
        results,
        prepared={scenario.scenario_id: object()},
    )

    assert comparison.publication_eligible

    def _rebind_second_adapter(
        *,
        alternate_source_identity: SourceIdentity,
        alternate_source_authorization: EvaluatorSourceAuthorization,
        alternate_dataset_authority_sha256: str | None = None,
    ) -> list[EvalResult]:
        rebound: list[EvalResult] = []
        second_adapter_id = scenario.baselines[1].baseline_id
        for result in results:
            if result.adapter.adapter_id != second_adapter_id:
                rebound.append(result)
                continue
            receipt = result.execution_receipt
            assert receipt is not None
            unbound = EvalResult.model_validate(
                result.model_dump(mode="python", exclude={"execution_receipt"})
            )
            rebound.append(
                _bind_evaluation_receipt(
                    unbound,
                    environment=receipt.environment,
                    source_identity=alternate_source_identity,
                    source_authorization=alternate_source_authorization,
                    producer_sha256=receipt.producer_sha256,
                    baseline_spec_sha256=receipt.baseline_spec_sha256,
                    dataset_snapshot_id=receipt.dataset_snapshot_id,
                    dataset_snapshot_sha256=receipt.dataset_snapshot_sha256,
                    dataset_request_sha256=receipt.dataset_request_sha256,
                    dataset_authority_sha256=(
                        alternate_dataset_authority_sha256
                        or receipt.dataset_authority_sha256
                    ),
                    dataset_authority_status=receipt.dataset_authority_status,
                    started_at=receipt.started_at,
                    completed_at=receipt.completed_at,
                )
            )
        return rebound

    alternate_authorization = EvaluatorSourceAuthorization.model_validate(
        source_authorization.model_copy(update={"catalog_hash": HASHES[0]})
    )
    authority_mismatch_results = _rebind_second_adapter(
        alternate_source_identity=source_identity,
        alternate_source_authorization=alternate_authorization,
    )
    authority_mismatch = compare_results(
        suite,
        authority_mismatch_results,
        prepared={scenario.scenario_id: object()},
    )
    assert not authority_mismatch.publication_eligible

    alternate_source_identity = SourceIdentity.model_validate(
        source_identity.model_copy(update={"commit": "3" * 40})
    )
    alternate_identity_authorization = EvaluatorSourceAuthorization.model_validate(
        source_authorization.model_copy(
            update={
                "canonical_commit": "3" * 40,
                "source_identity_sha256": canonical_hash(alternate_source_identity),
            }
        )
    )
    identity_mismatch = compare_results(
        suite,
        _rebind_second_adapter(
            alternate_source_identity=alternate_source_identity,
            alternate_source_authorization=alternate_identity_authorization,
        ),
        prepared={scenario.scenario_id: object()},
    )
    assert not identity_mismatch.publication_eligible

    dataset_authority_mismatch = compare_results(
        suite,
        _rebind_second_adapter(
            alternate_source_identity=source_identity,
            alternate_source_authorization=source_authorization,
            alternate_dataset_authority_sha256=HASHES[0],
        ),
        prepared={scenario.scenario_id: object()},
    )
    assert not dataset_authority_mismatch.publication_eligible

    def _catalog_with_comparison(
        compared_results: list[EvalResult],
        report: ComparisonReport,
    ) -> CatalogIndex:
        additions: list[CatalogEntry] = []
        for result in compared_results:
            payload = result.model_dump(mode="json")
            digest = canonical_hash(payload)
            additions.append(
                CatalogEntry(
                    kind=CatalogObjectKind.EVAL_RESULT,
                    object_id=result.result_id,
                    object_schema_version=result.schema_version,
                    object_hash=digest,
                    source_hash=digest,
                    source_path=f"evaluations/results/{result.result_id}.json",
                    status=result.status.value,
                    data=payload,
                )
            )
        comparison_payload = report.model_dump(mode="json")
        comparison_digest = canonical_hash(comparison_payload)
        additions.append(
            CatalogEntry(
                kind=CatalogObjectKind.EVAL_COMPARISON,
                object_id=report.comparison_id,
                object_schema_version=report.schema_version,
                object_hash=comparison_digest,
                source_hash=comparison_digest,
                source_path=f"evaluations/comparisons/{report.comparison_id}.json",
                status="publication_eligible",
                data=comparison_payload,
            )
        )
        entries = (*catalog.entries, *additions)
        source_tree_hash = canonical_hash(
            {
                "schema": "tabu.catalog-source-tree.v1",
                "sources": [
                    {
                        "kind": entry.kind.value,
                        "object_id": entry.object_id,
                        "source_hash": entry.source_hash,
                        "source_path": entry.source_path,
                    }
                    for entry in entries
                ],
                "lineage": [
                    edge.model_dump(mode="json") for edge in catalog.lineage
                ],
            }
        )
        return CatalogIndex(
            source_tree_hash=source_tree_hash,
            entries=entries,
            lineage=catalog.lineage,
        )

    rebuilt = _catalog_with_comparison(results, comparison)

    assert rebuilt.show(comparison.comparison_id).kind is CatalogObjectKind.EVAL_COMPARISON

    forged_public = authority_mismatch.model_copy(update={"publication_eligible": True})
    with pytest.raises(ValueError, match="publication status"):
        _catalog_with_comparison(authority_mismatch_results, forged_public)


def test_formal_eval_rejects_prepared_dataset_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)
    monkeypatch.setattr(
        formal_receipt_module,
        "verify_formal_authorization",
        lambda *args, **kwargs: VerifiedFormalAuthorization(
            summary=summary,
            source_identity=source_identity,
            catalog=catalog,
        ),
    )
    tampered = DatasetSnapshotSpec.model_validate(
        prepared_snapshot.model_dump(mode="python")
        | {"mask_boundary": "tampered preparation boundary"}
    )

    with pytest.raises(
        FormalEvaluationReceiptError,
        match="does not approve these exact prepared bytes",
    ):
        issue_formal_evaluation_receipt(
            result,
            environment=_environment(),
            live_source_identity=source_identity,
            started_at=STARTED_AT,
            source_authorization_context=context,
            preregistration_text="schema_version: source-authority-fixture\n",
            catalog=catalog,
            dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
            prepared_snapshot=tampered,
        )


def test_formal_eval_fails_closed_when_source_remote_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        result,
        catalog,
        prepared_snapshot,
        reviewed_snapshot,
        source_identity,
        _summary,
        context,
    ) = _formal_baseline_fixture(tmp_path)

    def unavailable(*args, **kwargs):
        raise FormalAuthorizationError("public remote unavailable")

    monkeypatch.setattr(
        formal_receipt_module,
        "verify_formal_authorization",
        unavailable,
    )
    with pytest.raises(
        FormalEvaluationReceiptError,
        match="source authorization replay failed",
    ):
        issue_formal_evaluation_receipt(
            result,
            environment=_environment(),
            live_source_identity=source_identity,
            started_at=STARTED_AT,
            source_authorization_context=context,
            preregistration_text="schema_version: source-authority-fixture\n",
            catalog=catalog,
            dataset_snapshot_id=reviewed_snapshot.dataset_snapshot_id,
            prepared_snapshot=prepared_snapshot,
        )
