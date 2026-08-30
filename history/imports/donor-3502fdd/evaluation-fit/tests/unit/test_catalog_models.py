from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from tabu_lab.catalog import (
    ArtifactStatusEvent,
    CatalogEntry,
    CatalogIndex,
    CatalogObjectKind,
    ClaimMaturity,
    ClaimRecord,
    ClaimStatus,
    DatasetAdapter,
    DatasetAuthorityStatus,
    DatasetSnapshotSpec,
    EvidencePointer,
    ExperimentRecord,
    ExperimentStatus,
    FailureCategory,
    LineageEdge,
    LineageRelation,
    ModelArtifact,
    ModelArtifactStatus,
    ObjectRef,
    ReviewDecision,
    ReviewRecord,
    RunAttemptRecord,
    RunAttemptStatus,
    RunRecord,
    RunStatus,
    StatusEvent,
)
from tabu_lab.contracts.canonical import canonical_hash
from tabu_lab.evaluation.foundry import (
    AdapterSpec,
    EvalProducerBinding,
    EvalResult,
    EvaluationFailure,
    EvaluationStatus,
    EvaluatorSourceAuthorization,
    load_suite,
)
from tabu_lab.evaluation.foundry import FailureCategory as EvalFailureCategory
from tabu_lab.evaluation.foundry.contracts import (
    _bind_evaluation_receipt,
    _result_id_from_components,
)
from tabu_lab.evidence import (
    ArtifactRef,
    EnvironmentDisclosure,
    Receipt,
    ReceiptStatus,
    SourceIdentity,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _pointer(name: str, digest: str = HASH_A) -> EvidencePointer:
    return EvidencePointer(uri=f"git://tabu-lab/{name}", sha256=digest)


def _entry(
    kind: CatalogObjectKind,
    object_id: str,
    data: dict[str, Any],
    *,
    status: str | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        kind=kind,
        object_id=object_id,
        object_schema_version=str(data["schema_version"]),
        object_hash=canonical_hash(data),
        source_hash=canonical_hash({"source": object_id, "data": data}),
        source_path=f"fixtures/{object_id}.json",
        status=status,
        data=data,
    )


def _index(
    entries: tuple[CatalogEntry, ...], edges: tuple[LineageEdge, ...] = ()
) -> CatalogIndex:
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
            "lineage": [edge.model_dump(mode="json") for edge in edges],
        }
    )
    return CatalogIndex(
        source_tree_hash=source_tree_hash,
        entries=entries,
        lineage=edges,
    )


def _evaluation_snapshot(
    *,
    status: DatasetAuthorityStatus = DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED,
    review_ids: tuple[str, ...] = (),
    mask_boundary: str = "test truth remains evaluator-side",
) -> DatasetSnapshotSpec:
    return DatasetSnapshotSpec(
        schema_version="tabu.dataset-snapshot.v3",
        dataset_snapshot_id="evaluation-snapshot-a",
        dataset_id="dataset-a",
        source_uri="https://example.org/dataset-a.csv",
        source_sha256=HASH_A,
        content_sha256=HASH_B,
        license_id="CC-BY-4.0",
        split_manifest_sha256=HASH_C,
        fit_partition="train",
        adapter=DatasetAdapter(
            adapter_id="offline-eval-prepared-snapshot",
            adapter_version="1.0.0",
        ),
        episode_recipe_hashes=("d" * 64,),
        evaluation_scenario_id="scenario-a",
        truth_sidecar_sha256="e" * 64,
        request_sha256="f" * 64,
        authority_sha256="1" * 64,
        authority_status=status,
        review_ids=review_ids,
        mask_boundary=mask_boundary,
        contamination_boundary="split-before-prepare; preprocessing fits train only",
    )


def _authority_review(
    *,
    subject_sha256: str | None,
    decision: ReviewDecision = ReviewDecision.APPROVED,
) -> ReviewRecord:
    return ReviewRecord(
        review_id="dataset-authority-review-a",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.DATASET_SNAPSHOT,
                object_id="evaluation-snapshot-a",
                evidence_sha256=subject_sha256,
            ),
        ),
        developer_identity="dataset-developer",
        reviewer_identity="independent-dataset-reviewer",
        decision=decision,
        report=_pointer("dataset-authority-review-report", HASH_C),
    )


def test_dataset_snapshot_v2_remains_backward_compatible_without_authority_lineage() -> None:
    snapshot = DatasetSnapshotSpec(
        dataset_snapshot_id="legacy-snapshot-a",
        dataset_id="legacy-dataset-a",
        source_uri="https://example.org/legacy.csv",
        source_sha256=HASH_A,
        content_sha256=HASH_B,
        license_id="CC-BY-4.0",
        split_manifest_sha256=HASH_C,
        fit_partition="train",
        adapter=DatasetAdapter(adapter_id="legacy-adapter", adapter_version="1.0.0"),
        mask_boundary="legacy mask boundary",
        contamination_boundary="legacy contamination boundary",
    )

    payload = snapshot.model_dump(mode="json")
    assert snapshot.schema_version == "tabu.dataset-snapshot.v2"
    assert "authority_status" not in payload
    assert "authority_sha256" not in payload
    assert "request_sha256" not in payload
    assert "review_ids" not in payload
    assert not snapshot.publication_eligible


def test_reviewed_dataset_authority_requires_review_ids() -> None:
    with pytest.raises(ValidationError, match="requires at least one review id"):
        _evaluation_snapshot(status=DatasetAuthorityStatus.REVIEWED)


def test_catalog_accepts_only_exact_reviewed_dataset_authority_subject() -> None:
    unreviewed = _evaluation_snapshot()
    subject_sha256 = unreviewed.authority_review_subject_sha256
    assert subject_sha256 == unreviewed.content_hash
    reviewed = _evaluation_snapshot(
        status=DatasetAuthorityStatus.REVIEWED,
        review_ids=("dataset-authority-review-a",),
    )
    review = _authority_review(subject_sha256=subject_sha256)

    catalog = _index(
        (
            _entry(
                CatalogObjectKind.DATASET_SNAPSHOT,
                reviewed.dataset_snapshot_id,
                reviewed.model_dump(mode="json"),
            ),
            _entry(
                CatalogObjectKind.REVIEW,
                review.review_id,
                review.model_dump(mode="json"),
                status=review.decision.value,
            ),
        )
    )

    assert reviewed.publication_eligible
    assert any(
        entry.object_id == reviewed.dataset_snapshot_id for entry in catalog.entries
    )


def test_old_dataset_authority_review_cannot_promote_changed_snapshot_content() -> None:
    original = _evaluation_snapshot()
    review = _authority_review(
        subject_sha256=original.authority_review_subject_sha256,
    )
    changed = _evaluation_snapshot(
        status=DatasetAuthorityStatus.REVIEWED,
        review_ids=(review.review_id,),
        mask_boundary="changed after the authority review",
    )

    with pytest.raises(ValidationError, match=r"does not bind.*evidence hash"):
        _index(
            (
                _entry(
                    CatalogObjectKind.DATASET_SNAPSHOT,
                    changed.dataset_snapshot_id,
                    changed.model_dump(mode="json"),
                ),
                _entry(
                    CatalogObjectKind.REVIEW,
                    review.review_id,
                    review.model_dump(mode="json"),
                    status=review.decision.value,
                ),
            )
        )


def test_dataset_authority_review_without_exact_evidence_hash_cannot_promote() -> None:
    reviewed = _evaluation_snapshot(
        status=DatasetAuthorityStatus.REVIEWED,
        review_ids=("dataset-authority-review-a",),
    )
    review = _authority_review(subject_sha256=None)

    with pytest.raises(ValidationError, match=r"does not bind.*evidence hash"):
        _index(
            (
                _entry(
                    CatalogObjectKind.DATASET_SNAPSHOT,
                    reviewed.dataset_snapshot_id,
                    reviewed.model_dump(mode="json"),
                ),
                _entry(
                    CatalogObjectKind.REVIEW,
                    review.review_id,
                    review.model_dump(mode="json"),
                    status=review.decision.value,
                ),
            )
        )


def test_experiment_cannot_skip_reviewed_preregistration_gate() -> None:
    with pytest.raises(ValidationError, match="illegal status transition"):
        ExperimentRecord(
            experiment_id="F0-test",
            contract_id="tabuf",
            hypothesis="bounded fit",
            claim_boundary="F0 only",
            status=ExperimentStatus.RUNNABLE,
            status_history=(
                StatusEvent(status="draft"),
                StatusEvent(status="runnable", evidence_hashes=(HASH_C,)),
            ),
            preregistration=_pointer("prereg", HASH_A),
            preregistration_review=_pointer("review", HASH_B),
            review_ids=("review-a",),
            source_identity=_pointer("source", HASH_C),
        )


def test_preregistered_experiment_binds_preregistration_and_review_hashes() -> None:
    with pytest.raises(ValidationError, match="bind preregistration and review hashes"):
        ExperimentRecord(
            experiment_id="F0-test",
            contract_id="tabuf",
            hypothesis="bounded fit",
            claim_boundary="F0 only",
            status=ExperimentStatus.PREREGISTERED,
            status_history=(
                StatusEvent(status="draft"),
                StatusEvent(status="preregistered", evidence_hashes=(HASH_A,)),
            ),
            preregistration=_pointer("prereg", HASH_A),
            preregistration_review=_pointer("review", HASH_B),
        )


@pytest.mark.parametrize(
    ("uri", "message"),
    (
        ("/Users/researcher/checkpoint.safetensors", "absolute local path"),
        ("/mnt/research/checkpoint.safetensors", "absolute local path"),
        (r"C:\\research\\checkpoint.safetensors", "absolute local path"),
        ("~/private/checkpoint.safetensors", "absolute local path"),
        ("file:///tmp/checkpoint.safetensors", "local file URI"),
    ),
)
def test_public_catalog_pointer_rejects_private_local_path(uri: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        EvidencePointer(uri=uri, sha256=HASH_A)


def test_public_catalog_rejects_embedded_arbitrary_absolute_path() -> None:
    with pytest.raises(ValidationError, match="absolute local paths"):
        StatusEvent(status="draft", note="debug cache: /srv/tabu/run-1")


@pytest.mark.parametrize("field_name", ("database_password", "training_hostname"))
def test_public_catalog_rejects_nested_sensitive_key_shapes(field_name: str) -> None:
    with pytest.raises(ValidationError, match="sensitive field"):
        CatalogEntry(
            kind=CatalogObjectKind.DATASET_SNAPSHOT,
            object_id="sensitive-fixture",
            object_schema_version="test.v1",
            object_hash=HASH_A,
            source_hash=HASH_B,
            source_path="fixtures/sensitive.json",
            data={"schema_version": "test.v1", field_name: "redacted"},
        )


def test_claim_maturity_does_not_promote_without_evidence() -> None:
    with pytest.raises(ValidationError, match="requires linked evidence"):
        ClaimRecord(
            claim_id="claim-foundation",
            statement="foundation model",
            boundary="not established",
            maturity=ClaimMaturity.FOUNDATION_MODEL,
            status=ClaimStatus.PROPOSED,
        )


def test_catalog_rejects_duplicate_global_id() -> None:
    first = _entry(
        CatalogObjectKind.MODEL_CONTRACT,
        "duplicate",
        {"schema_version": "test.model.v1"},
    )
    second = _entry(
        CatalogObjectKind.DATASET_SNAPSHOT,
        "duplicate",
        {"schema_version": "test.dataset.v1"},
    )
    with pytest.raises(ValidationError, match="duplicate catalog object ids"):
        _index((first, second))


def test_catalog_rejects_dangling_lineage_reference() -> None:
    experiment = _entry(
        CatalogObjectKind.EXPERIMENT,
        "experiment-a",
        {"schema_version": "test.experiment.v1"},
    )
    edge = LineageEdge(
        source=ObjectRef(kind=CatalogObjectKind.EXPERIMENT, object_id="experiment-a"),
        relation=LineageRelation.IMPLEMENTS,
        target=ObjectRef(kind=CatalogObjectKind.MODEL_CONTRACT, object_id="missing-model"),
    )
    with pytest.raises(ValidationError, match="dangling lineage reference"):
        _index((experiment,), (edge,))


def test_catalog_rejects_implements_hash_drift_from_current_model_spec() -> None:
    model = _entry(
        CatalogObjectKind.MODEL_CONTRACT,
        "tabuf",
        {"schema_version": "test.model.v1", "contract_version": "9.9.9"},
    )
    experiment = _draft_experiment_entry("experiment-a", "tabuf")
    edge = LineageEdge(
        source=ObjectRef(kind=CatalogObjectKind.EXPERIMENT, object_id="experiment-a"),
        relation=LineageRelation.IMPLEMENTS,
        target=ObjectRef(kind=CatalogObjectKind.MODEL_CONTRACT, object_id="tabuf"),
        evidence_hash=HASH_A,
    )
    assert model.object_hash != HASH_A

    with pytest.raises(ValidationError, match="differs from its exact ModelSpec"):
        _index((model, experiment), (edge,))


def _draft_experiment_entry(
    experiment_id: str,
    contract_id: str,
    *,
    supersedes: tuple[str, ...] = (),
) -> CatalogEntry:
    preregistration = _pointer(f"{experiment_id}-preregistration", HASH_A)
    record = ExperimentRecord(
        experiment_id=experiment_id,
        contract_id=contract_id,
        hypothesis="bounded revision fixture",
        claim_boundary="test fixture only",
        status=ExperimentStatus.DRAFT,
        status_history=(StatusEvent(status="draft"),),
        preregistration=preregistration,
        supersedes_experiment_ids=supersedes,
        revision_rationale=("typed direct revision" if supersedes else None),
    )
    return _entry(
        CatalogObjectKind.EXPERIMENT,
        experiment_id,
        record.model_dump(mode="json"),
        status=record.status.value,
    )


def _supersedes_edge(successor_id: str, predecessor_id: str) -> LineageEdge:
    return LineageEdge(
        source=ObjectRef(kind=CatalogObjectKind.EXPERIMENT, object_id=successor_id),
        relation=LineageRelation.SUPERSEDES,
        target=ObjectRef(kind=CatalogObjectKind.EXPERIMENT, object_id=predecessor_id),
        evidence_hash=HASH_A,
    )


def test_catalog_rejects_cross_contract_experiment_supersession() -> None:
    predecessor = _draft_experiment_entry("experiment-old", "tabuf")
    successor = _draft_experiment_entry(
        "experiment-new",
        "tabul",
        supersedes=("experiment-old",),
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1"},
        ),
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabul",
            {"schema_version": "test.model.v1"},
        ),
        predecessor,
        successor,
    )

    with pytest.raises(ValidationError, match="same contract"):
        _index(entries, (_supersedes_edge("experiment-new", "experiment-old"),))


def test_catalog_rejects_supersedes_declaration_edge_drift() -> None:
    predecessor = _draft_experiment_entry("experiment-old", "tabuf")
    successor = _draft_experiment_entry(
        "experiment-new",
        "tabuf",
        supersedes=("experiment-old",),
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1"},
        ),
        predecessor,
        successor,
    )

    with pytest.raises(ValidationError, match="declarations and lineage edges"):
        _index(entries)


def test_catalog_rejects_ambiguous_direct_superseding_successors() -> None:
    predecessor = _draft_experiment_entry("experiment-old", "tabuf")
    successor_a = _draft_experiment_entry(
        "experiment-new-a",
        "tabuf",
        supersedes=("experiment-old",),
    )
    successor_b = _draft_experiment_entry(
        "experiment-new-b",
        "tabuf",
        supersedes=("experiment-old",),
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1"},
        ),
        predecessor,
        successor_a,
        successor_b,
    )
    edges = (
        _supersedes_edge("experiment-new-a", "experiment-old"),
        _supersedes_edge("experiment-new-b", "experiment-old"),
    )

    with pytest.raises(ValidationError, match="only one direct superseding successor"):
        _index(entries, edges)


def test_catalog_rejects_resume_cycle() -> None:
    run_a = RunRecord(
        run_id="run-a",
        experiment_id="experiment-a",
        status=RunStatus.PLANNED,
        status_history=(StatusEvent(status="planned"),),
    )
    run_b = RunRecord(
        run_id="run-b",
        experiment_id="experiment-a",
        status=RunStatus.PLANNED,
        status_history=(StatusEvent(status="planned"),),
    )
    entries = (
        _entry(
            CatalogObjectKind.RUN,
            run_a.run_id,
            run_a.model_dump(mode="json"),
        ),
        _entry(
            CatalogObjectKind.RUN,
            run_b.run_id,
            run_b.model_dump(mode="json"),
        ),
    )
    edges = (
        LineageEdge(
            source=ObjectRef(kind=CatalogObjectKind.RUN, object_id="run-a"),
            relation=LineageRelation.RESUMES_FROM,
            target=ObjectRef(kind=CatalogObjectKind.RUN, object_id="run-b"),
        ),
        LineageEdge(
            source=ObjectRef(kind=CatalogObjectKind.RUN, object_id="run-b"),
            relation=LineageRelation.RESUMES_FROM,
            target=ObjectRef(kind=CatalogObjectKind.RUN, object_id="run-a"),
        ),
    )
    with pytest.raises(ValidationError, match="contains a cycle"):
        _index(entries, edges)


def test_nonterminal_run_cannot_attach_terminal_receipt() -> None:
    with pytest.raises(ValidationError, match="nonterminal runs cannot attach"):
        RunRecord(
            run_id="run-pending",
            experiment_id="experiment-a",
            status=RunStatus.RUNNING,
            status_history=(StatusEvent(status="planned"), StatusEvent(status="running")),
            receipt=_pointer("receipt", HASH_A),
        )


def test_terminal_experiment_requires_matching_terminal_run() -> None:
    run = RunRecord(
        run_id="run-still-running",
        experiment_id="experiment-terminal",
        status=RunStatus.RUNNING,
        status_history=(StatusEvent(status="planned"), StatusEvent(status="running")),
    )
    preregistration = _pointer("preregistration", HASH_A)
    report = _pointer("preregistration-review", HASH_B)
    source_identity = _pointer("source-identity", HASH_C)
    review = ReviewRecord(
        review_id="experiment-terminal-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id="experiment-terminal",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=report,
    )
    experiment = ExperimentRecord(
        experiment_id="experiment-terminal",
        contract_id="tabuf",
        hypothesis="terminal state binding",
        claim_boundary="test only",
        status=ExperimentStatus.SUCCEEDED,
        status_history=(
            StatusEvent(status="draft"),
            StatusEvent(
                status="preregistered",
                evidence_hashes=(preregistration.sha256, report.sha256),
            ),
            StatusEvent(status="runnable", evidence_hashes=(source_identity.sha256,)),
            StatusEvent(status="running"),
            StatusEvent(status="succeeded", evidence_hashes=(HASH_A,)),
        ),
        preregistration=preregistration,
        preregistration_review=report,
        source_identity=source_identity,
        run_ids=(run.run_id,),
        review_ids=(review.review_id,),
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1"},
        ),
        _entry(
            CatalogObjectKind.EXPERIMENT,
            experiment.experiment_id,
            experiment.model_dump(mode="json"),
            status=experiment.status.value,
        ),
        _entry(
            CatalogObjectKind.REVIEW,
            review.review_id,
            review.model_dump(mode="json"),
            status=review.decision.value,
        ),
        _entry(
            CatalogObjectKind.RUN,
            run.run_id,
            run.model_dump(mode="json"),
            status=run.status.value,
        ),
    )

    with pytest.raises(ValidationError, match="not supported by a matching cataloged run"):
        _index(entries)


def test_only_successful_receipted_run_can_produce_model_artifact() -> None:
    run_id = f"run-{'1' * 64}"
    receipt_record = Receipt(
        receipt_id="receipt-failed",
        run_id=run_id,
        run_identity_hash=HASH_A,
        run_bundle_hash=HASH_B,
        status=ReceiptStatus.FAILED,
        error="typed model failure",
        metadata={"issuance_status": "local_unissued"},
    )
    receipt_entry = _entry(
        CatalogObjectKind.RECEIPT,
        receipt_record.receipt_id,
        receipt_record.model_dump(mode="json"),
        status=receipt_record.status.value,
    )
    receipt = EvidencePointer(
        uri=receipt_entry.source_path,
        sha256=receipt_record.receipt_hash,
    )
    failed_run = RunRecord(
        run_id=run_id,
        experiment_id="experiment-a",
        status=RunStatus.FAILED,
            status_history=(
                StatusEvent(status="planned"),
                StatusEvent(status="running"),
                StatusEvent(
                    status="failed",
                    evidence_hashes=(receipt_record.receipt_hash,),
                ),
            ),
        receipt=receipt,
        attempt_ids=("attempt-failed",),
        failure_category=FailureCategory.MODEL,
    )
    failed_attempt = RunAttemptRecord(
        attempt_id="attempt-failed",
        run_id=run_id,
        status=RunAttemptStatus.FAILED,
        receipt=receipt,
        failure_category=FailureCategory.MODEL,
    )
    artifact = ModelArtifact(
        artifact_id="artifact-a",
        contract_id="tabuf",
        contract_version="0.1.0",
        producer_run_id=failed_run.run_id,
        producer_receipt=receipt,
        checkpoint=_pointer("checkpoint", HASH_B),
        checkpoint_format="safetensors",
        checkpoint_schema_version="tabu.training-checkpoint.v3",
        model_state_schema_version="tabu.model-state.v1",
        model_spec=EvidencePointer(
            uri="fixtures/tabuf.json",
            sha256=canonical_hash(
                {"schema_version": "test.model.v1", "contract_version": "0.1.0"}
            ),
        ),
        semantic_config=_pointer("semantic-config", HASH_A),
        compiler_manifest=_pointer("compiler-manifest", HASH_B),
        license_id="Apache-2.0",
        status_history=(
            ArtifactStatusEvent(status="produced", evidence=receipt),
        ),
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1", "contract_version": "0.1.0"},
        ),
        _entry(
            CatalogObjectKind.EXPERIMENT,
            "experiment-a",
            ExperimentRecord(
                experiment_id="experiment-a",
                contract_id="tabuf",
                hypothesis="bounded fit",
                claim_boundary="F0 only",
                status=ExperimentStatus.DRAFT,
                status_history=(StatusEvent(status="draft"),),
            ).model_dump(mode="json"),
            status=ExperimentStatus.DRAFT.value,
        ),
        receipt_entry,
        _entry(
            CatalogObjectKind.RUN_ATTEMPT,
            failed_attempt.attempt_id,
            failed_attempt.model_dump(mode="json"),
            status=failed_attempt.status.value,
        ),
        _entry(
            CatalogObjectKind.RUN,
            failed_run.run_id,
            failed_run.model_dump(mode="json"),
            status=failed_run.status.value,
        ),
        _entry(
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact.artifact_id,
            artifact.model_dump(mode="json"),
            status=artifact.status.value,
        ),
    )
    edge = LineageEdge(
        source=ObjectRef(kind=CatalogObjectKind.RUN, object_id=failed_run.run_id),
        relation=LineageRelation.PRODUCED,
        target=ObjectRef(kind=CatalogObjectKind.MODEL_ARTIFACT, object_id=artifact.artifact_id),
    )
    with pytest.raises(ValidationError, match="successful receipted run"):
        _index(entries, (edge,))


def test_catalog_entry_rejects_hash_drift() -> None:
    with pytest.raises(ValidationError, match="object_hash does not match"):
        CatalogEntry(
            kind=CatalogObjectKind.MODEL_CONTRACT,
            object_id="model-a",
            object_schema_version="test.v1",
            object_hash=HASH_A,
            source_hash=HASH_B,
            source_path="specs/model-a.yaml",
            data={"schema_version": "test.v1"},
        )


def _released_artifact_entries(
    review: ReviewRecord | None,
) -> tuple[CatalogEntry, ...]:
    run_id = f"run-{'4' * 64}"
    artifact_id = "artifact-release-candidate"
    receipt = Receipt(
        receipt_id="receipt-release-candidate",
        run_id=run_id,
        run_identity_hash=HASH_A,
        run_bundle_hash=HASH_B,
        status=ReceiptStatus.SUCCEEDED,
        artifacts=(
            ArtifactRef(
                artifact_id="checkpoint",
                kind="checkpoint",
                uri="checkpoint/checkpoint.safetensors",
                sha256=HASH_C,
                size_bytes=1,
            ),
            ArtifactRef(
                artifact_id="semantic-config",
                kind="fit_artifact",
                uri="resolved-configs/semantic.json",
                sha256=HASH_A,
                size_bytes=1,
            ),
            ArtifactRef(
                artifact_id="compiler-manifest",
                kind="fit_artifact",
                uri="compiler-manifest.json",
                sha256=HASH_B,
                size_bytes=1,
            ),
        ),
        metadata={"issuance_status": "formal"},
    )
    receipt_entry = _entry(
        CatalogObjectKind.RECEIPT,
        receipt.receipt_id,
        receipt.model_dump(mode="json"),
        status=receipt.status.value,
    )
    receipt_pointer = EvidencePointer(
        uri=receipt_entry.source_path,
        sha256=receipt.receipt_hash,
    )
    release_report = _pointer("release-report", HASH_B)
    run = RunRecord(
        run_id=run_id,
        experiment_id="experiment-release",
        status=RunStatus.SUCCEEDED,
        status_history=(
            StatusEvent(status="planned"),
            StatusEvent(status="running"),
            StatusEvent(status="succeeded", evidence_hashes=(receipt.receipt_hash,)),
        ),
        attempt_ids=("attempt-release-candidate",),
        receipt=receipt_pointer,
        artifact_ids=(artifact_id,),
    )
    attempt = RunAttemptRecord(
        attempt_id="attempt-release-candidate",
        run_id=run_id,
        status=RunAttemptStatus.SUCCEEDED,
        receipt=receipt_pointer,
    )
    artifact = ModelArtifact(
        artifact_id=artifact_id,
        contract_id="tabuf",
        contract_version="0.1.0",
        producer_run_id=run_id,
        producer_receipt=receipt_pointer,
        checkpoint=EvidencePointer(
            uri="fixtures/checkpoint/checkpoint.safetensors",
            sha256=HASH_C,
        ),
        checkpoint_format="safetensors",
        checkpoint_schema_version="tabu.training-checkpoint.v3",
        model_state_schema_version="tabu.model-state.v1",
        model_spec=EvidencePointer(
            uri="fixtures/tabuf.json",
            sha256=canonical_hash(
                {"schema_version": "test.model.v1", "contract_version": "0.1.0"}
            ),
        ),
        semantic_config=EvidencePointer(
            uri="fixtures/resolved-configs/semantic.json",
            sha256=HASH_A,
        ),
        compiler_manifest=EvidencePointer(
            uri="fixtures/compiler-manifest.json",
            sha256=HASH_B,
        ),
        license_id="Apache-2.0",
        status=ModelArtifactStatus.RELEASED,
        status_history=(
            ArtifactStatusEvent(status="produced", evidence=receipt_pointer),
            ArtifactStatusEvent(status="verified", evidence=receipt_pointer),
            ArtifactStatusEvent(status="released", evidence=release_report),
        ),
        review_ids=("release-review",),
    )
    preregistration = _pointer("release-preregistration", HASH_A)
    preregistration_review = _pointer("release-preregistration-review", HASH_B)
    source_identity = _pointer("release-source-identity", HASH_C)
    experiment_review = ReviewRecord(
        review_id="experiment-release-preregistration-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.EXPERIMENT,
                object_id="experiment-release",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=preregistration_review,
    )
    entries = (
        _entry(
            CatalogObjectKind.MODEL_CONTRACT,
            "tabuf",
            {"schema_version": "test.model.v1", "contract_version": "0.1.0"},
        ),
        _entry(
            CatalogObjectKind.EXPERIMENT,
            "experiment-release",
            ExperimentRecord(
                experiment_id="experiment-release",
                contract_id="tabuf",
                hypothesis="bounded release fixture",
                claim_boundary="test fixture only",
                status=ExperimentStatus.SUCCEEDED,
                status_history=(
                    StatusEvent(status="draft"),
                    StatusEvent(
                        status="preregistered",
                        evidence_hashes=(
                            preregistration.sha256,
                            preregistration_review.sha256,
                        ),
                    ),
                    StatusEvent(
                        status="runnable",
                        evidence_hashes=(source_identity.sha256,),
                    ),
                    StatusEvent(status="running"),
                    StatusEvent(status="succeeded", evidence_hashes=(receipt.receipt_hash,)),
                ),
                preregistration=preregistration,
                preregistration_review=preregistration_review,
                source_identity=source_identity,
                run_ids=(run_id,),
                review_ids=(experiment_review.review_id,),
            ).model_dump(mode="json"),
            status=ExperimentStatus.SUCCEEDED.value,
        ),
        _entry(
            CatalogObjectKind.REVIEW,
            experiment_review.review_id,
            experiment_review.model_dump(mode="json"),
            status=experiment_review.decision.value,
        ),
        receipt_entry,
        _entry(
            CatalogObjectKind.RUN_ATTEMPT,
            attempt.attempt_id,
            attempt.model_dump(mode="json"),
            status=attempt.status.value,
        ),
        _entry(
            CatalogObjectKind.RUN,
            run.run_id,
            run.model_dump(mode="json"),
            status=run.status.value,
        ),
        _entry(
            CatalogObjectKind.MODEL_ARTIFACT,
            artifact.artifact_id,
            artifact.model_dump(mode="json"),
            status=artifact.status.value,
        ),
    )
    if review is not None:
        entries += (
            _entry(
                CatalogObjectKind.REVIEW,
                review.review_id,
                review.model_dump(mode="json"),
                status=review.decision.value,
            ),
        )
    return entries


def test_released_artifact_requires_cataloged_review() -> None:
    with pytest.raises(ValidationError, match="references missing review"):
        _index(_released_artifact_entries(review=None))


@pytest.mark.parametrize("binding", ("semantic_config", "compiler_manifest"))
def test_artifact_metadata_pointer_digest_must_match_producer_receipt(binding: str) -> None:
    review = ReviewRecord(
        review_id="release-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id="artifact-release-candidate",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=_pointer("release-report", HASH_B),
        gong_approval=_pointer("gong-release-approval", HASH_C),
    )
    entries = list(_released_artifact_entries(review=review))
    artifact_index = next(
        index
        for index, entry in enumerate(entries)
        if entry.kind is CatalogObjectKind.MODEL_ARTIFACT
    )
    artifact = ModelArtifact.model_validate(entries[artifact_index].data)
    pointer = getattr(artifact, binding)
    tampered_pointer = pointer.model_copy(
        update={"sha256": HASH_B if pointer.sha256 != HASH_B else HASH_A}
    )
    tampered = artifact.model_copy(update={binding: tampered_pointer})
    entries[artifact_index] = _entry(
        CatalogObjectKind.MODEL_ARTIFACT,
        artifact.artifact_id,
        tampered.model_dump(mode="json"),
        status=tampered.status.value,
    )

    with pytest.raises(ValidationError, match=r"artifact .* digest differs"):
        _index(tuple(entries))


def test_public_model_eval_rejects_artifact_from_a_different_valid_producer_run() -> None:
    release_review = ReviewRecord(
        review_id="release-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id="artifact-release-candidate",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=_pointer("release-report", HASH_B),
        gong_approval=_pointer("gong-release-approval", HASH_C),
    )
    entries = list(_released_artifact_entries(review=release_review))
    experiment_index = next(
        index
        for index, entry in enumerate(entries)
        if entry.kind is CatalogObjectKind.EXPERIMENT
    )
    artifact_index = next(
        index
        for index, entry in enumerate(entries)
        if entry.kind is CatalogObjectKind.MODEL_ARTIFACT
    )
    experiment = ExperimentRecord.model_validate(entries[experiment_index].data)
    artifact = ModelArtifact.model_validate(entries[artifact_index].data)

    second_run_id = f"run-{'5' * 64}"
    second_receipt = Receipt(
        receipt_id="receipt-second-valid-producer",
        run_id=second_run_id,
        run_identity_hash="4" * 64,
        run_bundle_hash="5" * 64,
        status=ReceiptStatus.SUCCEEDED,
        artifacts=(),
        metadata={"issuance_status": "formal"},
    )
    second_receipt_entry = _entry(
        CatalogObjectKind.RECEIPT,
        second_receipt.receipt_id,
        second_receipt.model_dump(mode="json"),
        status=second_receipt.status.value,
    )
    second_pointer = EvidencePointer(
        uri=second_receipt_entry.source_path,
        sha256=second_receipt.receipt_hash,
    )
    second_attempt = RunAttemptRecord(
        attempt_id="attempt-second-valid-producer",
        run_id=second_run_id,
        status=RunAttemptStatus.SUCCEEDED,
        receipt=second_pointer,
    )
    second_run = RunRecord(
        run_id=second_run_id,
        experiment_id=experiment.experiment_id,
        status=RunStatus.SUCCEEDED,
        status_history=(
            StatusEvent(status="planned"),
            StatusEvent(status="running"),
            StatusEvent(
                status="succeeded",
                evidence_hashes=(second_receipt.receipt_hash,),
            ),
        ),
        attempt_ids=(second_attempt.attempt_id,),
        receipt=second_pointer,
    )
    experiment = experiment.model_copy(
        update={"run_ids": (*experiment.run_ids, second_run_id)}
    )
    entries[experiment_index] = _entry(
        CatalogObjectKind.EXPERIMENT,
        experiment.experiment_id,
        experiment.model_dump(mode="json"),
        status=experiment.status.value,
    )

    suite = load_suite("table-completion-micro-v0")
    scenario = suite.scenarios[0]
    producer = EvalProducerBinding(
        provenance="receipted_run",
        run_id=second_run_id,
        receipt_sha256=second_pointer.sha256,
        receipt_pointer=second_pointer.uri,
        publication_eligible=True,
    )
    adapter = AdapterSpec(
        adapter_id="tabuf-catalog-mismatch",
        adapter_version="1.0.0",
        kind="model",
        fit_iterations=1,
        device_class="single_device",
        deterministic=True,
        contract_id="tabuf",
        artifact_id=artifact.artifact_id,
    )
    failure = EvaluationFailure(
        category=EvalFailureCategory.MODEL,
        code="bounded-model-failure",
        public_detail="model failed after its producer lineage was resolved",
    )
    result_values = {
        "suite_id": suite.suite_id,
        "suite_version": suite.suite_version,
        "suite_hash": suite.suite_hash,
        "scenario_id": scenario.scenario_id,
        "task": scenario.task,
        "adapter": adapter,
        "producer": producer,
        "seed": suite.budget.model_seeds[0],
        "source_sha256": "6" * 64,
        "split_sha256": "7" * 64,
        "recipe_sha256": "8" * 64,
        "budget_hash": suite.budget.content_hash,
        "truth_sidecar_sha256": "9" * 64,
        "status": EvaluationStatus.FAILED,
        "raw_predictions": (),
        "topology_checks": (),
        "per_example": (),
        "metrics": {},
        "counts": {"targets": 0, "scored": 0, "abstained": 0},
        "failure_counts": {EvalFailureCategory.MODEL: 1},
        "coverage": 0.0,
        "failure": failure,
        "claim_boundary": "catalog producer-lineage regression only",
    }
    result = EvalResult(
        result_id=_result_id_from_components(**result_values),
        **result_values,
    )
    source_identity = SourceIdentity(
        source_kind="git",
        issuance_status="formal",
        reviewed=True,
        repository_uri="https://github.com/wehub-community/tabu-lab.git",
        repository_subdirectory=".",
        commit="1" * 40,
        remote_ref="refs/remotes/origin/main",
        git_tree_oid="2" * 40,
        source_tree_hash="3" * 64,
        preregistration_blob_hash="4" * 64,
        lock_hash="5" * 64,
    )
    source_authorization = EvaluatorSourceAuthorization(
        canonical_commit="1" * 40,
        catalog_hash="a" * 64,
        catalog_source_tree_hash="b" * 64,
        experiment_id=experiment.experiment_id,
        experiment_status=experiment.status.value,
        preregistration_sha256="c" * 64,
        source_identity_sha256=canonical_hash(source_identity),
        review_ids=("source-review",),
        review_report_sha256s=("d" * 64,),
        gong_approval_sha256s=("e" * 64,),
    )
    unreviewed_snapshot = DatasetSnapshotSpec(
        schema_version="tabu.dataset-snapshot.v3",
        dataset_snapshot_id="model-eval-reviewed-snapshot",
        dataset_id=scenario.dataset.dataset_id,
        source_uri=scenario.dataset.source_uri,
        source_sha256=result.source_sha256,
        content_sha256="f" * 64,
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
        request_sha256="1" * 64,
        authority_sha256="2" * 64,
        authority_status=DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED,
        mask_boundary="post-split evaluator-side truth",
        contamination_boundary="train-only preprocessing",
    )
    snapshot = DatasetSnapshotSpec.model_validate(
        unreviewed_snapshot.model_dump(mode="python")
        | {
            "authority_status": DatasetAuthorityStatus.REVIEWED,
            "review_ids": ("review-model-eval-dataset",),
        }
    )
    result = _bind_evaluation_receipt(
        result,
        environment=EnvironmentDisclosure(
            environment_hash="3" * 64,
            host_class="cpu-host",
            operating_system="Linux",
            device="cpu",
            architecture="x86_64",
            python_version="3.11.14",
        ),
        source_identity=source_identity,
        source_authorization=source_authorization,
        producer_sha256=producer.content_hash,
        dataset_snapshot_id=snapshot.dataset_snapshot_id,
        dataset_snapshot_sha256=snapshot.content_hash,
        dataset_request_sha256=snapshot.request_sha256,
        dataset_authority_sha256=snapshot.authority_sha256,
        dataset_authority_status="reviewed",
        started_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC),
        completed_at=datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
        + timedelta(seconds=1),
    )
    artifact = artifact.model_copy(
        update={"evaluation_result_ids": (result.result_id,)}
    )
    entries[artifact_index] = _entry(
        CatalogObjectKind.MODEL_ARTIFACT,
        artifact.artifact_id,
        artifact.model_dump(mode="json"),
        status=artifact.status.value,
    )
    dataset_review = ReviewRecord(
        review_id="review-model-eval-dataset",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.DATASET_SNAPSHOT,
                object_id=snapshot.dataset_snapshot_id,
                evidence_sha256=unreviewed_snapshot.content_hash,
            ),
        ),
        developer_identity="dataset-developer",
        reviewer_identity="dataset-reviewer",
        decision=ReviewDecision.APPROVED,
        report=_pointer("model-eval-dataset-review", "3" * 64),
    )
    entries.extend(
        (
            second_receipt_entry,
            _entry(
                CatalogObjectKind.RUN_ATTEMPT,
                second_attempt.attempt_id,
                second_attempt.model_dump(mode="json"),
                status=second_attempt.status.value,
            ),
            _entry(
                CatalogObjectKind.RUN,
                second_run.run_id,
                second_run.model_dump(mode="json"),
                status=second_run.status.value,
            ),
            _entry(
                CatalogObjectKind.EVAL_SUITE,
                suite.suite_id,
                suite.model_dump(mode="json"),
                status=suite.status,
            ),
            _entry(
                CatalogObjectKind.DATASET_SNAPSHOT,
                snapshot.dataset_snapshot_id,
                snapshot.model_dump(mode="json"),
            ),
            _entry(
                CatalogObjectKind.REVIEW,
                dataset_review.review_id,
                dataset_review.model_dump(mode="json"),
                status=dataset_review.decision.value,
            ),
            _entry(
                CatalogObjectKind.EVAL_RESULT,
                result.result_id,
                result.model_dump(mode="json"),
                status=result.status.value,
            ),
        )
    )

    with pytest.raises(
        ValidationError,
        match="model artifact differs from its exact producer run receipt",
    ):
        _index(tuple(entries))


def test_verified_artifact_requires_cataloged_digest_binding_receipt() -> None:
    review = ReviewRecord(
        review_id="release-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id="artifact-release-candidate",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=_pointer("release-report", HASH_B),
        gong_approval=_pointer("gong-release-approval", HASH_C),
    )
    entries = list(_released_artifact_entries(review=review))
    artifact_index = next(
        index
        for index, entry in enumerate(entries)
        if entry.kind is CatalogObjectKind.MODEL_ARTIFACT
    )
    artifact = ModelArtifact.model_validate(entries[artifact_index].data)
    history = tuple(
        event.model_copy(update={"evidence": _pointer("forged-verification", HASH_C)})
        if event.status is ModelArtifactStatus.VERIFIED
        else event
        for event in artifact.status_history
    )
    tampered = artifact.model_copy(update={"status_history": history})
    entries[artifact_index] = _entry(
        CatalogObjectKind.MODEL_ARTIFACT,
        artifact.artifact_id,
        tampered.model_dump(mode="json"),
        status=tampered.status.value,
    )

    with pytest.raises(ValidationError, match="verification evidence is not a cataloged"):
        _index(tuple(entries))


@pytest.mark.parametrize(
    ("review", "message"),
    (
        (
            ReviewRecord(
                review_id="release-review",
                subjects=(
                    ObjectRef(
                        kind=CatalogObjectKind.RUN,
                        object_id=f"run-{'4' * 64}",
                    ),
                ),
                developer_identity="developer-a",
                reviewer_identity="reviewer-b",
                decision=ReviewDecision.APPROVED,
                report=_pointer("release-report", HASH_B),
                gong_approval=_pointer("gong-release-approval", HASH_C),
            ),
            "review does not name the promoted subject",
        ),
        (
            ReviewRecord(
                review_id="release-review",
                subjects=(
                    ObjectRef(
                        kind=CatalogObjectKind.MODEL_ARTIFACT,
                        object_id="artifact-release-candidate",
                    ),
                ),
                developer_identity="developer-a",
                reviewer_identity="reviewer-b",
                decision=ReviewDecision.CHANGES_REQUESTED,
                report=_pointer("release-report", HASH_B),
                gong_approval=_pointer("gong-release-approval", HASH_C),
            ),
            "requires an approved review",
        ),
        (
            ReviewRecord(
                review_id="release-review",
                subjects=(
                    ObjectRef(
                        kind=CatalogObjectKind.MODEL_ARTIFACT,
                        object_id="artifact-release-candidate",
                    ),
                ),
                developer_identity="developer-a",
                reviewer_identity="reviewer-b",
                decision=ReviewDecision.APPROVED,
                report=_pointer("release-report", HASH_B),
            ),
            "release review lacks gong approval",
        ),
    ),
)
def test_released_artifact_rejects_wrong_or_incomplete_review(
    review: ReviewRecord,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _index(_released_artifact_entries(review=review))


def _accepted_foundation_claim_entries(
    *,
    include_review: bool = True,
    claim_gong_hash: str = HASH_C,
    receipt_pointer_hash: str | None = None,
    issuance_status: str = "formal",
    receipt_status: ReceiptStatus = ReceiptStatus.SUCCEEDED,
    include_receipt_evidence: bool = True,
    include_review_evidence: bool = True,
) -> tuple[CatalogEntry, ...]:
    receipt = Receipt(
        receipt_id="receipt-claim-foundation",
        run_id=f"run-{'5' * 64}",
        run_identity_hash=HASH_A,
        run_bundle_hash=HASH_B,
        status=receipt_status,
        error="failed evidence" if receipt_status is ReceiptStatus.FAILED else None,
        metadata={"issuance_status": issuance_status},
    )
    receipt_entry = _entry(
        CatalogObjectKind.RECEIPT,
        receipt.receipt_id,
        receipt.model_dump(mode="json"),
        status=receipt.status.value,
    )
    receipt_pointer = EvidencePointer(
        uri=receipt_entry.source_path,
        sha256=receipt_pointer_hash or receipt.receipt_hash,
    )
    review = ReviewRecord(
        review_id="claim-foundation-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.CLAIM,
                object_id="claim-foundation-accepted",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=_pointer("claim-foundation-review", HASH_B),
        gong_approval=_pointer("claim-foundation-gong", HASH_C),
    )
    evidence: list[ObjectRef] = []
    if include_receipt_evidence:
        evidence.append(
            ObjectRef(kind=CatalogObjectKind.RECEIPT, object_id=receipt.receipt_id)
        )
    if include_review_evidence:
        evidence.append(ObjectRef(kind=CatalogObjectKind.REVIEW, object_id=review.review_id))
    claim = ClaimRecord(
        claim_id="claim-foundation-accepted",
        statement="foundation model claim fixture",
        boundary="test closure only; not a real public claim",
        maturity=ClaimMaturity.FOUNDATION_MODEL,
        status=ClaimStatus.ACCEPTED,
        evidence=tuple(evidence),
        receipt=receipt_pointer,
        review_ids=(review.review_id,),
        gong_approval=_pointer("claim-foundation-gong", claim_gong_hash),
    )
    entries = (
        receipt_entry,
        _entry(
            CatalogObjectKind.CLAIM,
            claim.claim_id,
            claim.model_dump(mode="json"),
            status=claim.status.value,
        ),
    )
    if include_review:
        entries += (
            _entry(
                CatalogObjectKind.REVIEW,
                review.review_id,
                review.model_dump(mode="json"),
                status=review.decision.value,
            ),
        )
    return entries


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"include_review": False}, "references missing review"),
        ({"claim_gong_hash": HASH_A}, "gong approval does not match"),
        ({"receipt_pointer_hash": HASH_C}, "receipt pointer is not cataloged"),
    ),
)
def test_accepted_high_maturity_claim_requires_closed_review_receipt_and_approval(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _index(_accepted_foundation_claim_entries(**kwargs))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {"include_receipt_evidence": False},
            "claim receipt must be explicitly included in claim evidence",
        ),
        (
            {"include_review_evidence": False},
            "promoted claim must explicitly include its review in evidence",
        ),
        (
            {"issuance_status": "local_unissued"},
            "successful formal receipt",
        ),
        (
            {"receipt_status": ReceiptStatus.FAILED},
            "successful formal receipt",
        ),
    ),
)
def test_accepted_high_maturity_claim_rejects_unissued_or_unlinked_evidence(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _index(_accepted_foundation_claim_entries(**kwargs))


def test_bare_receipt_cannot_promote_high_maturity_claim() -> None:
    entries = _accepted_foundation_claim_entries()
    edges = (
        LineageEdge(
            source=ObjectRef(
                kind=CatalogObjectKind.RECEIPT,
                object_id="receipt-claim-foundation",
            ),
            relation=LineageRelation.SUPPORTS,
            target=ObjectRef(
                kind=CatalogObjectKind.CLAIM,
                object_id="claim-foundation-accepted",
            ),
        ),
        LineageEdge(
            source=ObjectRef(
                kind=CatalogObjectKind.REVIEW,
                object_id="claim-foundation-review",
            ),
            relation=LineageRelation.SUPPORTS,
            target=ObjectRef(
                kind=CatalogObjectKind.CLAIM,
                object_id="claim-foundation-accepted",
            ),
        ),
    )

    with pytest.raises(ValidationError, match="select exactly one cataloged run"):
        _index(entries, edges)


def _supported_model_claim_entries(
    *, maturity: ClaimMaturity
) -> tuple[CatalogEntry, ...]:
    release_review = ReviewRecord(
        review_id="release-review",
        subjects=(
            ObjectRef(
                kind=CatalogObjectKind.MODEL_ARTIFACT,
                object_id="artifact-release-candidate",
            ),
        ),
        developer_identity="developer-a",
        reviewer_identity="reviewer-b",
        decision=ReviewDecision.APPROVED,
        report=_pointer("release-report", HASH_B),
        gong_approval=_pointer("gong-release-approval", HASH_C),
    )
    entries = list(_released_artifact_entries(review=release_review))
    run = RunRecord.model_validate(
        next(entry.data for entry in entries if entry.kind is CatalogObjectKind.RUN)
    )
    attempt = RunAttemptRecord.model_validate(
        next(entry.data for entry in entries if entry.kind is CatalogObjectKind.RUN_ATTEMPT)
    )
    artifact = ModelArtifact.model_validate(
        next(entry.data for entry in entries if entry.kind is CatalogObjectKind.MODEL_ARTIFACT)
    )
    receipt_entry = next(
        entry for entry in entries if entry.kind is CatalogObjectKind.RECEIPT
    )
    experiment = ExperimentRecord.model_validate(
        next(entry.data for entry in entries if entry.kind is CatalogObjectKind.EXPERIMENT)
    )
    assert run.receipt is not None
    claim_id = "claim-supported-model"
    claim_review = ReviewRecord(
        review_id="claim-supported-model-review",
        subjects=(ObjectRef(kind=CatalogObjectKind.CLAIM, object_id=claim_id),),
        developer_identity="developer-a",
        reviewer_identity="reviewer-c",
        decision=ReviewDecision.APPROVED,
        report=_pointer("claim-supported-model-review", HASH_B),
        gong_approval=_pointer("claim-supported-model-gong", HASH_C),
    )
    claim = ClaimRecord(
        claim_id=claim_id,
        statement="bounded supported model fixture",
        boundary="catalog closure test only",
        maturity=maturity,
        status=ClaimStatus.ACCEPTED,
        evidence=(
            ObjectRef(
                kind=CatalogObjectKind.RECEIPT,
                object_id=receipt_entry.object_id,
            ),
            ObjectRef(kind=CatalogObjectKind.EXPERIMENT, object_id=experiment.experiment_id),
            ObjectRef(kind=CatalogObjectKind.RUN, object_id=run.run_id),
            ObjectRef(kind=CatalogObjectKind.RUN_ATTEMPT, object_id=attempt.attempt_id),
            ObjectRef(kind=CatalogObjectKind.MODEL_ARTIFACT, object_id=artifact.artifact_id),
            ObjectRef(kind=CatalogObjectKind.REVIEW, object_id=claim_review.review_id),
        ),
        receipt=run.receipt,
        review_ids=(claim_review.review_id,),
        gong_approval=claim_review.gong_approval,
    )
    entries.extend(
        (
            _entry(
                CatalogObjectKind.CLAIM,
                claim.claim_id,
                claim.model_dump(mode="json"),
                status=claim.status.value,
            ),
            _entry(
                CatalogObjectKind.REVIEW,
                claim_review.review_id,
                claim_review.model_dump(mode="json"),
                status=claim_review.decision.value,
            ),
        )
    )
    return tuple(entries)


def test_supported_model_claim_requires_closed_run_and_artifact_lineage() -> None:
    catalog = _index(_supported_model_claim_entries(maturity=ClaimMaturity.SUPPORTED_MODEL))

    assert catalog.show("claim-supported-model").status == ClaimStatus.ACCEPTED.value


def test_foundation_claim_also_requires_formal_comparison_evidence() -> None:
    with pytest.raises(ValidationError, match="formal comparison evidence"):
        _index(_supported_model_claim_entries(maturity=ClaimMaturity.FOUNDATION_MODEL))
