from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabu_lab.adapters import (
    EvalSnapshotRegistrationError,
    dataset_snapshot_from_prepared,
    write_dataset_snapshot_manifest,
)
from tabu_lab.catalog import CatalogObjectKind, DatasetAuthorityStatus, build_catalog
from tabu_lab.evaluation.foundry import (
    DatasetSnapshotBinding,
    EvalSuiteSpec,
    PreparationContract,
    PreparedExample,
    PreparedScenario,
    SourceMaterial,
    TargetKind,
    load_suite,
)


def _micro_suite() -> EvalSuiteSpec:
    payload = load_suite("table-supervised-micro-v0").model_dump(mode="python")
    payload["scenarios"] = [payload["scenarios"][1]]
    payload["scenarios"][0]["selection"]["partition_limits"] = {
        "train": 3,
        "validation": 1,
        "test": 2,
    }
    return EvalSuiteSpec.model_validate(payload)


def _prepared(*, dataset_id: str = "sklearn-diabetes") -> PreparedScenario:
    scenario = _micro_suite().scenarios[0]
    train = tuple(
        PreparedExample(
            example_id=f"train-{index}",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": float(index)},
            target=float(index * 2),
        )
        for index in range(3)
    )
    validation = (
        PreparedExample(
            example_id="validation-0",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 3.0},
            target=6.0,
        ),
    )
    test = tuple(
        PreparedExample(
            example_id=f"test-{index}",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": float(index + 4)},
            target=float((index + 4) * 2),
        )
        for index in range(2)
    )
    preparation = PreparationContract(
        preprocessing={
            "fit_partition": "train",
            "implementation_sha256": "1" * 64,
            "fitted_state_sha256": "2" * 64,
        },
        selection=scenario.selection.model_dump(mode="python"),
        mask={"kind": "none"},
    )
    source = SourceMaterial.from_bytes(
        dataset_id=dataset_id,
        content=f"{dataset_id}:offline-source".encode(),
        media_type="application/json",
    )
    binding = DatasetSnapshotBinding(
        dataset_id=dataset_id,
        source_sha256=source.raw_sha256,
        split_sha256=PreparedScenario.split_sha256_for(
            train=train,
            validation=validation,
            test=test,
        ),
        recipe_sha256=PreparedScenario.recipe_sha256_for(preparation=preparation),
        truth_sidecar_sha256=PreparedScenario.truth_sidecar_sha256_for(test=test),
        partition_counts={"train": 3, "validation": 1, "test": 2},
    )
    return PreparedScenario(
        scenario_id=scenario.scenario_id,
        binding=binding,
        source_material=source,
        preparation=preparation,
        train=train,
        validation=validation,
        test=test,
    )


def test_offline_prepared_scenario_registers_as_catalog_snapshot(tmp_path: Path) -> None:
    suite = _micro_suite()
    prepared = _prepared()

    snapshot = dataset_snapshot_from_prepared(
        suite=suite,
        scenario_id=prepared.scenario_id,
        prepared=prepared,
        request_sha256="a" * 64,
        authority_sha256="b" * 64,
    )

    assert snapshot.schema_version == "tabu.dataset-snapshot.v3"
    assert snapshot.source_sha256 == prepared.source_material.raw_sha256
    assert snapshot.content_sha256 == prepared.content_hash
    assert snapshot.split_manifest_sha256 == prepared.binding.split_sha256
    assert snapshot.episode_recipe_hashes == (prepared.binding.recipe_sha256,)
    assert snapshot.evaluation_scenario_id == prepared.scenario_id
    assert snapshot.truth_sidecar_sha256 == prepared.binding.truth_sidecar_sha256
    assert snapshot.request_sha256 == "a" * 64
    assert snapshot.authority_sha256 == "b" * 64
    assert snapshot.authority_status is DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED
    assert snapshot.review_ids == ()
    assert not snapshot.publication_eligible
    destination = tmp_path / "datasets" / f"{snapshot.dataset_snapshot_id}.json"
    write_dataset_snapshot_manifest(snapshot, destination)
    first = destination.read_bytes()
    write_dataset_snapshot_manifest(snapshot, destination)
    assert destination.read_bytes() == first

    catalog = build_catalog(tmp_path)
    entry = catalog.show(snapshot.dataset_snapshot_id)
    assert entry.kind is CatalogObjectKind.DATASET_SNAPSHOT
    assert entry.data == snapshot.model_dump(mode="json")


def test_snapshot_registration_rejects_suite_dataset_mismatch() -> None:
    suite = _micro_suite()
    prepared = _prepared(dataset_id="different-dataset")

    with pytest.raises(
        EvalSnapshotRegistrationError,
        match="prepared_dataset_id_mismatch",
    ):
        dataset_snapshot_from_prepared(
            suite=suite,
            scenario_id=prepared.scenario_id,
            prepared=prepared,
            request_sha256="a" * 64,
            authority_sha256="b" * 64,
        )


def test_snapshot_manifest_refuses_different_content_at_same_path(tmp_path: Path) -> None:
    suite = _micro_suite()
    snapshot = dataset_snapshot_from_prepared(
        suite=suite,
        scenario_id=suite.scenarios[0].scenario_id,
        prepared=_prepared(),
        request_sha256="a" * 64,
        authority_sha256="b" * 64,
    )
    destination = tmp_path / "snapshot.json"
    write_dataset_snapshot_manifest(snapshot, destination)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    payload["license_id"] = "different-license"
    destination.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(FileExistsError, match="different content"):
        write_dataset_snapshot_manifest(snapshot, destination)
