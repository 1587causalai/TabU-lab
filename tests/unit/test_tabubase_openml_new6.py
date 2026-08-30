from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from tabu_lab.experiments.tabubase_openml_new6 import (
    OPENML_NEW6_BY_ID,
    OPENML_NEW6_PANEL_ID,
    OPENML_NEW6_SPECS,
    OpenMLNew6Spec,
    build_fetched_openml_new6_panel_manifest,
    fetch_openml_new6_dataset,
    load_openml_new6_panel_manifest,
    preregistered_openml_new6_manifest,
)


def _details(spec: OpenMLNew6Spec) -> dict[str, str]:
    return {
        "id": str(spec.data_id),
        "name": spec.openml_name,
        "version": str(spec.version),
        "md5_checksum": spec.upstream_md5,
        "licence": spec.license,
        "status": "active",
        "default_target_attribute": "fixture_target",
    }


def _classification_bunch(spec: OpenMLNew6Spec) -> SimpleNamespace:
    assert spec.classes is not None
    labels = np.asarray(
        [f"label_{index % spec.classes}" for index in range(spec.rows)],
        dtype=object,
    )
    return SimpleNamespace(
        data=np.arange(spec.rows * spec.predictors, dtype=np.float32).reshape(
            spec.rows, spec.predictors
        ),
        target=labels,
        details=_details(spec),
    )


def _regression_bunch(spec: OpenMLNew6Spec) -> SimpleNamespace:
    return SimpleNamespace(
        data=np.linspace(
            -1.0,
            1.0,
            num=spec.rows * spec.predictors,
            dtype=np.float32,
        ).reshape(spec.rows, spec.predictors),
        target=np.linspace(0.0, 1.0, num=spec.rows, dtype=np.float32),
        details=_details(spec),
    )


def test_preregistered_manifest_is_complete_and_no_network() -> None:
    manifest = preregistered_openml_new6_manifest()
    assert manifest["panel_id"] == OPENML_NEW6_PANEL_ID
    assert manifest["registration_status"] == "candidate_preregistered"
    assert manifest["execution_status"] == "not_run"
    assert len(manifest["manifest_sha256"]) == 64
    assert [item["data_id"] for item in manifest["datasets"]] == [
        1462,
        36,
        44,
        43919,
        44959,
        44970,
    ]


def test_checked_in_panel_manifest_is_strictly_bound_to_registry() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
    manifest = load_openml_new6_panel_manifest(path)

    assert manifest.path == path.resolve()
    assert manifest.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert tuple(manifest.dataset_ids) == tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
    assert len(manifest.canonical_payload_sha256) == 64
    assert manifest.context_policy == "low_shot_grid"


def test_checked_in_full_context_panel_manifest_freezes_all_train_and_query_rows() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml"
    manifest = load_openml_new6_panel_manifest(path)

    assert manifest.path == path.resolve()
    assert manifest.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest.context_policy == "full_train"
    evaluation = manifest.payload["evaluation_design"]
    assert evaluation["context_rows"] == "all_train_partition_rows"
    assert evaluation["query_policy"] == "all_heldout_rows"
    assert evaluation["query_limit"] is None
    assert evaluation["regression_primary_metrics"] == [
        "scaled_rmse",
        "scaled_mae",
        "r2",
    ]


def test_full_context_panel_manifest_rejects_context_truncation(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/transfer-base-v2/real-full-context-frozen-icl-openml-new6.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["evaluation_design"]["context_rows"] = 32
    drifted = tmp_path / "drifted-full-context.yaml"
    drifted.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="context_rows"):
        load_openml_new6_panel_manifest(drifted)


def test_panel_manifest_rejects_a_pin_drift(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
    payload = yaml.safe_load(source.read_text())
    payload["datasets"][0]["upstream_md5"] = "0" * 32
    drifted = tmp_path / "drifted.yaml"
    drifted.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="upstream_md5"):
        load_openml_new6_panel_manifest(drifted)


@pytest.mark.parametrize(
    ("section", "field", "bad_value", "message"),
    [
        ("status", "execution", "complete", "status.execution"),
        ("model", "contract_version", "9.9.9", "model.contract_version"),
        ("tokenizer", "codebook_size", 99, "tokenizer.codebook_size"),
        ("evaluation_design", "query_limit", 128, "evaluation_design.query_limit"),
        ("frozen_protocol", "optimizer_created", True, "optimizer_created"),
    ],
)
def test_panel_manifest_rejects_protocol_drift(
    tmp_path: Path,
    section: str,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
    payload = yaml.safe_load(source.read_text())
    payload[section][field] = bad_value
    drifted = tmp_path / f"drifted-{section}-{field}.yaml"
    drifted.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        load_openml_new6_panel_manifest(drifted)


def test_mocked_classification_fetch_is_pinned_and_encodes_labels() -> None:
    spec = OPENML_NEW6_BY_ID["banknote_authentication"]
    bunch = _classification_bunch(spec)
    bunch.target = np.asarray(
        ["zeta" if index % 2 == 0 else "alpha" for index in range(spec.rows)],
        dtype=object,
    )
    calls: list[dict[str, Any]] = []

    def fake_fetcher(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return bunch

    fetched = fetch_openml_new6_dataset(spec.dataset_id, fetcher=fake_fetcher)

    assert calls == [
        {
            "data_id": 1462,
            "target_column": "default-target",
            "as_frame": False,
            "parser": "liac-arff",
            "cache": True,
        }
    ]
    assert fetched.dataset.features.dtype == np.float32
    assert fetched.dataset.response.dtype == np.int64
    assert fetched.source_manifest["materialized"]["label_mapping"] == {
        "alpha": 0,
        "zeta": 1,
    }
    assert fetched.source_manifest["fetch"]["scikit_learn_version"]


def test_mocked_classification_fetch_has_deterministic_string_label_mapping() -> None:
    spec = OPENML_NEW6_BY_ID["banknote_authentication"]
    bunch = _classification_bunch(spec)
    bunch.target = np.asarray(
        ["zeta" if index % 2 == 0 else "alpha" for index in range(spec.rows)],
        dtype=object,
    )
    fetched = fetch_openml_new6_dataset(spec.dataset_id, fetcher=lambda **_kwargs: bunch)

    assert fetched.source_manifest["materialized"]["label_mapping"] == {
        "alpha": 0,
        "zeta": 1,
    }
    assert fetched.dataset.response[:4].tolist() == [1, 0, 1, 0]
    assert fetched.source_manifest["source"]["resolved_target_column"] == "fixture_target"
    assert len(fetched.source_manifest["materialized"]["array_sha256"]) == 64
    assert len(fetched.source_manifest_sha256) == 64


def test_mocked_regression_fetch_validates_and_records_cache_policy() -> None:
    spec = OPENML_NEW6_BY_ID["airfoil_self_noise"]
    bunch = _regression_bunch(spec)
    fetched = fetch_openml_new6_dataset(
        spec.dataset_id,
        fetcher=lambda **_kwargs: bunch,
        cache=False,
    )

    assert fetched.dataset.features.shape == (1503, 5)
    assert fetched.dataset.response.shape == (1503,)
    assert fetched.dataset.response.dtype == np.float32
    assert fetched.source_manifest["fetch"]["cache"] is False
    assert fetched.source_manifest["materialized"]["label_mapping"] is None


def test_mocked_fetch_routes_an_explicit_existing_data_home(tmp_path: Path) -> None:
    spec = OPENML_NEW6_BY_ID["banknote_authentication"]
    bunch = _classification_bunch(spec)
    calls: list[dict[str, Any]] = []

    fetched = fetch_openml_new6_dataset(
        spec.dataset_id,
        fetcher=lambda **kwargs: calls.append(kwargs) or bunch,
        data_home=tmp_path,
    )

    assert calls[0]["data_home"] == str(tmp_path.resolve())
    assert fetched.source_manifest["fetch"]["data_home"] == str(tmp_path.resolve())


def test_mocked_fetch_rejects_a_missing_explicit_data_home(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data_home"):
        fetch_openml_new6_dataset(
            "banknote_authentication",
            fetcher=lambda **_kwargs: _classification_bunch(
                OPENML_NEW6_BY_ID["banknote_authentication"]
            ),
            data_home=tmp_path / "absent",
        )


def test_complete_mocked_panel_manifest_binds_every_source_receipt() -> None:
    fetched = []
    for spec in OPENML_NEW6_SPECS:
        bunch = (
            _classification_bunch(spec)
            if spec.task == "classification"
            else _regression_bunch(spec)
        )
        fetched.append(
            fetch_openml_new6_dataset(
                spec.dataset_id,
                fetcher=lambda bunch=bunch, **_kwargs: bunch,
            )
        )
    manifest = build_fetched_openml_new6_panel_manifest(fetched)

    assert manifest["execution_status"] == "materialized_not_evaluated"
    assert [item["dataset_id"] for item in manifest["datasets"]] == [
        spec.dataset_id for spec in OPENML_NEW6_SPECS
    ]
    assert all(len(item["source_manifest_sha256"]) == 64 for item in manifest["datasets"])
    assert len(manifest["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="exactly"):
        build_fetched_openml_new6_panel_manifest(fetched[:-1])


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("id", "999", "data_id drift"),
        ("version", "99", "version drift"),
        ("md5_checksum", "0" * 32, "md5 drift"),
        ("licence", "unknown", "license drift"),
        ("status", "deactivated", "not active"),
    ],
)
def test_mocked_fetch_fails_closed_on_metadata_drift(
    field: str, bad_value: str, message: str
) -> None:
    spec = OPENML_NEW6_BY_ID["banknote_authentication"]
    bunch = _classification_bunch(spec)
    bunch.details[field] = bad_value

    with pytest.raises(RuntimeError, match=message):
        fetch_openml_new6_dataset(spec.dataset_id, fetcher=lambda **_kwargs: bunch)


def test_mocked_fetch_fails_closed_on_shape_and_non_finite_values() -> None:
    spec = OPENML_NEW6_BY_ID["airfoil_self_noise"]
    shape_drift = _regression_bunch(spec)
    shape_drift.data = shape_drift.data[:-1]
    with pytest.raises(RuntimeError, match="shape drift"):
        fetch_openml_new6_dataset(spec.dataset_id, fetcher=lambda **_kwargs: shape_drift)

    non_finite = _regression_bunch(spec)
    non_finite.data[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="missing/non-finite predictors"):
        fetch_openml_new6_dataset(spec.dataset_id, fetcher=lambda **_kwargs: non_finite)

    missing_target = _regression_bunch(spec)
    missing_target.target[0] = np.nan
    with pytest.raises(RuntimeError, match="missing/non-finite target"):
        fetch_openml_new6_dataset(spec.dataset_id, fetcher=lambda **_kwargs: missing_target)


def test_prereg_yaml_matches_registry_and_freezes_optimizer_hash_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
    payload = yaml.safe_load(path.read_text())

    assert payload["panel_id"] == OPENML_NEW6_PANEL_ID
    assert payload["status"] == {
        "registration": "candidate_preregistered",
        "execution": "not_run",
        "data_materialization": "not_run",
        "empirical_claim": "none",
    }
    yaml_by_id = {item["dataset_id"]: item for item in payload["datasets"]}
    assert set(yaml_by_id) == set(OPENML_NEW6_BY_ID)
    for spec in OPENML_NEW6_SPECS:
        item = yaml_by_id[spec.dataset_id]
        assert item["data_id"] == spec.data_id
        assert item["version"] == spec.version
        assert item["upstream_md5"] == spec.upstream_md5
        assert item["license"] == spec.license
        assert item["task"] == spec.task
        assert item["shape"] == {"rows": spec.rows, "predictors": spec.predictors}
    protocol = payload["frozen_protocol"]
    assert protocol["optimizer_created"] is False
    assert protocol["parameter_hash_gate"]["timing"].startswith("immediately adjacent")
    assert protocol["parameter_hash_gate"]["aggregate_gate"].endswith("must be true")
