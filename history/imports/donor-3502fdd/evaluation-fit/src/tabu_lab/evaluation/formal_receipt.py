"""Git-backed issuance and replay for formal Evaluation Foundry receipts.

The evaluator source, evaluation protocol, dataset authority, and evaluated
subject are deliberately independent authorities:

* ``FormalAuthorizationContext`` closes the evaluator implementation source;
* the exact cataloged ``EvalSuiteSpec`` hash closes the protocol;
* a reviewed ``DatasetSnapshotSpec`` closes data and split authority;
* a model producer receipt/artifact or frozen baseline spec closes the subject.

This module is imported lazily by the CLI and catalog builder so the low-level
Foundry contracts remain free of catalog import cycles.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from tabu_lab.catalog import (
    CatalogEntry,
    CatalogIndex,
    CatalogObjectKind,
    DatasetAuthorityStatus,
    DatasetSnapshotSpec,
    ExperimentRecord,
    ModelArtifact,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    RunRecord,
)
from tabu_lab.evaluation.foundry.contracts import (
    AdapterKind,
    EvalResult,
    EvalSuiteSpec,
    EvaluatorSourceAuthorization,
    ProducerProvenance,
    ScenarioSpec,
    _bind_evaluation_receipt,
)
from tabu_lab.evidence import EnvironmentDisclosure, SourceIdentity
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    FormalAuthorizationReplaySession,
    FormalAuthorizationSummary,
    verify_committed_evidence_pointer,
    verify_formal_authorization,
)


class FormalEvaluationReceiptError(ValueError):
    """A formal evaluation authority chain failed closed."""


def _entry(
    catalog: CatalogIndex,
    object_id: str,
    kind: CatalogObjectKind,
) -> CatalogEntry:
    try:
        entry = catalog.show(object_id)
    except KeyError as exc:
        raise FormalEvaluationReceiptError(
            f"formal evaluation references missing catalog object {object_id!r}"
        ) from exc
    if entry.kind is not kind:
        raise FormalEvaluationReceiptError(
            f"formal evaluation object {object_id!r} is {entry.kind.value}, not {kind.value}"
        )
    return entry


def _source_authorization(
    summary: FormalAuthorizationSummary,
) -> EvaluatorSourceAuthorization:
    payload = summary.model_dump(mode="python")
    schema_version = payload.pop("schema_version")
    return EvaluatorSourceAuthorization(
        authorization_schema_version=schema_version,
        **payload,
    )


def _validate_suite_binding(result: EvalResult, catalog: CatalogIndex) -> ScenarioSpec:
    suite = EvalSuiteSpec.model_validate(
        _entry(catalog, result.suite_id, CatalogObjectKind.EVAL_SUITE).data
    )
    if (
        suite.suite_hash != result.suite_hash
        or suite.suite_version != result.suite_version
        or suite.budget.content_hash != result.budget_hash
        or result.seed not in suite.budget.model_seeds
    ):
        raise FormalEvaluationReceiptError(
            "evaluation result differs from its exact cataloged suite/version/budget"
        )
    scenarios = tuple(
        scenario for scenario in suite.scenarios if scenario.scenario_id == result.scenario_id
    )
    if len(scenarios) != 1 or scenarios[0].task is not result.task:
        raise FormalEvaluationReceiptError(
            "evaluation result scenario is not uniquely frozen by its suite"
        )
    scenario = scenarios[0]
    if result.claim_boundary != suite.claim_boundary:
        raise FormalEvaluationReceiptError(
            "evaluation result claim boundary differs from the frozen suite"
        )
    if result.adapter.fit_iterations > suite.budget.max_fit_iterations:
        raise FormalEvaluationReceiptError(
            "evaluation adapter fit iterations exceed the frozen suite budget"
        )
    if result.adapter.device_class != suite.budget.device_class:
        raise FormalEvaluationReceiptError(
            "evaluation adapter device class differs from the frozen suite budget"
        )
    if suite.budget.deterministic and not result.adapter.deterministic:
        raise FormalEvaluationReceiptError(
            "evaluation adapter violates the frozen deterministic execution contract"
        )
    if (
        result.adapter.kind is AdapterKind.MODEL
        and result.adapter.contract_id not in scenario.applicable_contracts
    ):
        raise FormalEvaluationReceiptError(
            "evaluation model contract is not applicable to the frozen scenario"
        )
    return scenario


def _validate_dataset_binding(
    *,
    result: EvalResult,
    catalog: CatalogIndex,
    dataset_snapshot_id: str,
    prepared_snapshot: DatasetSnapshotSpec,
    scenario: ScenarioSpec,
) -> DatasetSnapshotSpec:
    snapshot = DatasetSnapshotSpec.model_validate(
        _entry(
            catalog,
            dataset_snapshot_id,
            CatalogObjectKind.DATASET_SNAPSHOT,
        ).data
    )
    if (
        snapshot.authority_status is not DatasetAuthorityStatus.REVIEWED
        or not snapshot.publication_eligible
        or snapshot.request_sha256 is None
        or snapshot.authority_sha256 is None
    ):
        raise FormalEvaluationReceiptError(
            "formal evaluation requires a reviewed dataset authority snapshot"
        )
    prepared = DatasetSnapshotSpec.model_validate(
        prepared_snapshot.model_dump(mode="python")
    )
    if prepared.authority_status is not DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED:
        raise FormalEvaluationReceiptError(
            "private prepared data must derive its self-consistent unreviewed snapshot"
        )
    if snapshot.authority_review_subject_sha256 != prepared.content_hash:
        raise FormalEvaluationReceiptError(
            "reviewed dataset snapshot does not approve these exact prepared bytes and authority"
        )
    dataset = scenario.dataset
    if (
        snapshot.dataset_id != dataset.dataset_id
        or snapshot.source_uri != dataset.source_uri
        or snapshot.license_id != dataset.license_id
        or snapshot.evaluation_scenario_id != result.scenario_id
        or snapshot.source_sha256 != result.source_sha256
        or snapshot.split_manifest_sha256 != result.split_sha256
        or snapshot.episode_recipe_hashes != (result.recipe_sha256,)
        or snapshot.truth_sidecar_sha256 != result.truth_sidecar_sha256
    ):
        raise FormalEvaluationReceiptError(
            "evaluation result does not bind the exact reviewed dataset snapshot"
        )
    return snapshot


def _validate_dataset_receipt_binding(
    *,
    result: EvalResult,
    catalog: CatalogIndex,
    scenario: ScenarioSpec,
) -> DatasetSnapshotSpec:
    """Replay the exact reviewed snapshot bound by a serialized receipt."""

    receipt = result.execution_receipt
    if receipt is None or receipt.dataset_snapshot_id is None:
        raise FormalEvaluationReceiptError(
            "formal evaluation receipt lacks a reviewed dataset snapshot binding"
        )
    snapshot = DatasetSnapshotSpec.model_validate(
        _entry(
            catalog,
            receipt.dataset_snapshot_id,
            CatalogObjectKind.DATASET_SNAPSHOT,
        ).data
    )
    if (
        snapshot.authority_status is not DatasetAuthorityStatus.REVIEWED
        or not snapshot.publication_eligible
        or snapshot.content_hash != receipt.dataset_snapshot_sha256
        or snapshot.request_sha256 != receipt.dataset_request_sha256
        or snapshot.authority_sha256 != receipt.dataset_authority_sha256
        or receipt.dataset_authority_status != "reviewed"
    ):
        raise FormalEvaluationReceiptError(
            "formal evaluation receipt differs from its exact reviewed dataset authority"
        )
    dataset = scenario.dataset
    if (
        snapshot.dataset_id != dataset.dataset_id
        or snapshot.source_uri != dataset.source_uri
        or snapshot.license_id != dataset.license_id
        or snapshot.evaluation_scenario_id != result.scenario_id
        or snapshot.source_sha256 != result.source_sha256
        or snapshot.split_manifest_sha256 != result.split_sha256
        or snapshot.episode_recipe_hashes != (result.recipe_sha256,)
        or snapshot.truth_sidecar_sha256 != result.truth_sidecar_sha256
    ):
        raise FormalEvaluationReceiptError(
            "formal evaluation receipt dataset does not match the frozen scenario/result"
        )
    return snapshot


def _verify_dataset_review_reports(
    *,
    catalog: CatalogIndex,
    snapshot: DatasetSnapshotSpec,
    repository: Path,
    commit: str,
) -> None:
    """Bind dataset promotion to the exact review report blobs in Git history."""

    subject_sha256 = snapshot.authority_review_subject_sha256
    if subject_sha256 is None or not snapshot.review_ids:
        raise FormalEvaluationReceiptError(
            "reviewed dataset snapshot lacks a replayable review subject"
        )
    expected_subject = ObjectRef(
        kind=CatalogObjectKind.DATASET_SNAPSHOT,
        object_id=snapshot.dataset_snapshot_id,
        evidence_sha256=subject_sha256,
    )
    for review_id in snapshot.review_ids:
        review = ReviewRecord.model_validate(
            _entry(catalog, review_id, CatalogObjectKind.REVIEW).data
        )
        if (
            review.decision is not ReviewDecision.APPROVED
            or expected_subject not in review.subjects
            or review.developer_identity.strip().casefold()
            == review.reviewer_identity.strip().casefold()
        ):
            raise FormalEvaluationReceiptError(
                "dataset authority review is not approved, independent, and hash-bound"
            )
        try:
            verify_committed_evidence_pointer(
                repository,
                commit,
                uri=review.report.uri,
                sha256=review.report.sha256,
                label=f"dataset review report {review.review_id}",
            )
        except FormalAuthorizationError as exc:
            raise FormalEvaluationReceiptError(
                "dataset authority review report is not an exact committed evidence blob"
            ) from exc


def _producer_sha256(
    *,
    result: EvalResult,
    catalog: CatalogIndex,
    scenario: ScenarioSpec,
) -> tuple[str, str | None]:
    if result.adapter.kind is AdapterKind.BASELINE:
        if result.producer.provenance is not ProducerProvenance.UNISSUED_BASELINE:
            raise FormalEvaluationReceiptError(
                "formal baseline result must remain evaluator-owned and unissued"
            )
        baselines = tuple(
            baseline
            for baseline in scenario.baselines
            if baseline.baseline_id == result.adapter.adapter_id
        )
        if (
            len(baselines) != 1
            or baselines[0].family != result.adapter.baseline_family
        ):
            raise FormalEvaluationReceiptError(
                "formal baseline result differs from the exact frozen baseline spec"
            )
        digest = baselines[0].content_hash
        return digest, digest

    if (
        result.producer.provenance is not ProducerProvenance.RECEIPTED_RUN
        or not result.producer.publication_eligible
        or result.adapter.artifact_id is None
        or result.producer.run_id is None
        or result.producer.receipt_pointer is None
        or result.producer.receipt_sha256 is None
    ):
        raise FormalEvaluationReceiptError(
            "formal model result requires a receipted producer run and artifact"
        )
    artifact = ModelArtifact.model_validate(
        _entry(
            catalog,
            result.adapter.artifact_id,
            CatalogObjectKind.MODEL_ARTIFACT,
        ).data
    )
    run = RunRecord.model_validate(
        _entry(catalog, result.producer.run_id, CatalogObjectKind.RUN).data
    )
    if (
        artifact.producer_run_id != run.run_id
        or artifact.producer_receipt.uri != result.producer.receipt_pointer
        or artifact.producer_receipt.sha256 != result.producer.receipt_sha256
        or run.receipt != artifact.producer_receipt
        or artifact.contract_id != result.adapter.contract_id
    ):
        raise FormalEvaluationReceiptError(
            "model evaluation producer receipt/artifact lineage is not exact"
        )
    # Catalog validation has already replayed the underlying formal successful
    # fit receipt; this explicit equality prevents swapping another run pointer.
    if run.receipt is None:
        raise FormalEvaluationReceiptError("model evaluation producer run has no receipt")
    return result.producer.content_hash, None


def issue_formal_evaluation_receipt(
    result: EvalResult,
    *,
    environment: EnvironmentDisclosure,
    live_source_identity: SourceIdentity,
    started_at: datetime,
    source_authorization_context: FormalAuthorizationContext,
    preregistration_text: str,
    catalog: CatalogIndex,
    dataset_snapshot_id: str,
    prepared_snapshot: DatasetSnapshotSpec,
    completed_at: datetime | None = None,
) -> EvalResult:
    """Issue one formal receipt only after every independent authority closes."""

    result = EvalResult.model_validate(result.model_dump(mode="python"))
    if live_source_identity.source_kind != "git" or (
        live_source_identity.issuance_status != "formal"
    ):
        raise FormalEvaluationReceiptError(
            "formal evaluation currently requires a live reviewed Git source identity; "
            "distribution issuance remains fail-closed"
        )
    catalog = CatalogIndex.model_validate(catalog.model_dump(mode="python"))
    scenario = _validate_suite_binding(result, catalog)
    snapshot = _validate_dataset_binding(
        result=result,
        catalog=catalog,
        dataset_snapshot_id=dataset_snapshot_id,
        prepared_snapshot=prepared_snapshot,
        scenario=scenario,
    )
    producer_sha256, baseline_spec_sha256 = _producer_sha256(
        result=result,
        catalog=catalog,
        scenario=scenario,
    )
    try:
        verified = verify_formal_authorization(
            source_authorization_context,
            preregistration_text=preregistration_text,
            live_source_identity=live_source_identity,
        )
    except (FormalAuthorizationError, TypeError, ValueError) as exc:
        raise FormalEvaluationReceiptError(
            "evaluator source authorization replay failed"
        ) from exc
    if (
        catalog.catalog_hash != verified.summary.catalog_hash
        or catalog.source_tree_hash != verified.summary.catalog_source_tree_hash
        or catalog != verified.catalog
    ):
        raise FormalEvaluationReceiptError(
            "evaluation catalog differs from the evaluator source authorization catalog"
        )
    authorized_snapshot = DatasetSnapshotSpec.model_validate(
        _entry(
            verified.catalog,
            snapshot.dataset_snapshot_id,
            CatalogObjectKind.DATASET_SNAPSHOT,
        ).data
    )
    if authorized_snapshot != snapshot:
        raise FormalEvaluationReceiptError(
            "reviewed dataset snapshot differs from the authorized canonical catalog"
        )
    _verify_dataset_review_reports(
        catalog=verified.catalog,
        snapshot=authorized_snapshot,
        repository=source_authorization_context.repository,
        commit=verified.summary.canonical_commit,
    )
    return _bind_evaluation_receipt(
        result,
        environment=environment,
        source_identity=verified.source_identity,
        source_authorization=_source_authorization(verified.summary),
        producer_sha256=producer_sha256,
        baseline_spec_sha256=baseline_spec_sha256,
        dataset_snapshot_id=snapshot.dataset_snapshot_id,
        dataset_snapshot_sha256=snapshot.content_hash,
        dataset_request_sha256=snapshot.request_sha256,
        dataset_authority_sha256=snapshot.authority_sha256,
        dataset_authority_status="reviewed",
        started_at=started_at,
        completed_at=completed_at,
    )


def _safe_preregistration_text(repository: Path, pointer: str) -> str:
    normalized = pointer.replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts:
        raise FormalEvaluationReceiptError(
            "evaluator source preregistration pointer escapes the repository"
        )
    target = (repository / relative).resolve()
    try:
        target.relative_to(repository)
    except ValueError as exc:
        raise FormalEvaluationReceiptError(
            "evaluator source preregistration pointer escapes the repository"
        ) from exc
    if not target.is_file():
        raise FormalEvaluationReceiptError(
            "evaluator source preregistration is not retrievable"
        )
    return target.read_text(encoding="utf-8")


def replay_formal_evaluation_source_authorization(
    result: EvalResult,
    *,
    repository: Path,
    entries: Iterable[CatalogEntry],
    replay: FormalAuthorizationReplaySession,
) -> None:
    """Replay a serialized formal evaluator-source authorization during build."""

    receipt = result.execution_receipt
    if receipt is None or receipt.issuance_status != "formal":
        raise FormalEvaluationReceiptError(
            "catalog replay requires a formal evaluation execution receipt"
        )
    authorization = receipt.source_authorization
    if authorization is None:
        raise FormalEvaluationReceiptError(
            "formal evaluation receipt lacks evaluator-source authorization"
        )
    experiments = tuple(
        ExperimentRecord.model_validate(entry.data)
        for entry in entries
        if entry.kind is CatalogObjectKind.EXPERIMENT
        and entry.object_id == authorization.experiment_id
    )
    if len(experiments) != 1 or experiments[0].preregistration is None:
        raise FormalEvaluationReceiptError(
            "evaluator-source authorization experiment is not uniquely cataloged"
        )
    preregistration_text = _safe_preregistration_text(
        repository,
        experiments[0].preregistration.uri,
    )
    expected = FormalAuthorizationSummary.model_validate(
        authorization.formal_authorization_payload
    )
    verified = replay.verify(
        expected,
        preregistration_text=preregistration_text,
        live_source_identity=receipt.source_identity,
    )
    if verified.summary != expected or verified.source_identity != receipt.source_identity:
        raise FormalEvaluationReceiptError(
            "formal evaluator-source replay returned different authority"
        )
    scenario = _validate_suite_binding(result, verified.catalog)
    snapshot = _validate_dataset_receipt_binding(
        result=result,
        catalog=verified.catalog,
        scenario=scenario,
    )
    producer_sha256, baseline_spec_sha256 = _producer_sha256(
        result=result,
        catalog=verified.catalog,
        scenario=scenario,
    )
    if (
        receipt.producer_sha256 != producer_sha256
        or receipt.baseline_spec_sha256 != baseline_spec_sha256
    ):
        raise FormalEvaluationReceiptError(
            "formal evaluation receipt producer differs from the authorization catalog"
        )
    _verify_dataset_review_reports(
        catalog=verified.catalog,
        snapshot=snapshot,
        repository=replay.repository,
        commit=verified.summary.canonical_commit,
    )


__all__ = [
    "FormalEvaluationReceiptError",
    "issue_formal_evaluation_receipt",
    "replay_formal_evaluation_source_authorization",
]
