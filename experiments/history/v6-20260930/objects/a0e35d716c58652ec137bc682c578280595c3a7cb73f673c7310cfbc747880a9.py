"""Pinned cached OpenML ARFF inputs for an exploratory real-data panel.

The panel is intentionally separate from the six-dataset ``new6`` adapter.
It consumes already materialized, numeric-only ARFF files from an explicit
cache snapshot.  No network fetch or model evaluation happens in this module;
the loader only validates source identity, shape, finiteness, and target
resolution before returning a :class:`RealDataset` carrier.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .tabubase_real_benchmark import RealDataset

CACHED_OPENML_PANEL_ID = "tabubase-real-full-context-cached-openml-regression-8-v1"
CACHED_OPENML_PANEL_SCHEMA = "tabu.tabubase-cached-openml-panel.v1"
CACHED_OPENML_SOURCE_SCHEMA = "tabu.tabubase-cached-openml-source.v1"
CACHED_OPENML_PREREG_SCHEMA = "tabu.transfer-base-real-full-context-cached-openml.v1"

TaskKind = Literal["regression"]


@dataclass(frozen=True, slots=True)
class CachedOpenMLSpec:
    dataset_id: str
    openml_name: str
    data_id: int
    task_id: int
    task_sha256: str
    target_column: str
    rows: int
    predictors: int
    file_sha256: str
    data_sha256: str
    license: str = "Public"
    task: TaskKind = "regression"


CACHED_OPENML_SPECS = (
    CachedOpenMLSpec(
        dataset_id="white_wine",
        openml_name="white_wine",
        data_id=44971,
        task_id=361249,
        task_sha256="b5fe9cc6b1cbfe122b1fd597a9ec0c3071bc5fe750f024c72db45b615b8ee55a",
        target_column="quality",
        rows=4_898,
        predictors=11,
        file_sha256="bf6ded4123efe6be1ed2998736794fdbac3a65aca6d08cd349bd750393dc58ba",
        data_sha256="bf6ded4123efe6be1ed2998736794fdbac3a65aca6d08cd349bd750393dc58ba",
    ),
    CachedOpenMLSpec(
        dataset_id="red_wine",
        openml_name="red_wine",
        data_id=44972,
        task_id=361250,
        task_sha256="83fca5b50eef4115acbf190dbf69f814bf39cb531da57e479ceb3785446ed9f6",
        target_column="quality",
        rows=1_599,
        predictors=11,
        file_sha256="48f5b09fb01b33ac1d6905c916d3c93f578c63b6ff4cc87db2b06e05839b484d",
        data_sha256="48f5b09fb01b33ac1d6905c916d3c93f578c63b6ff4cc87db2b06e05839b484d",
    ),
    CachedOpenMLSpec(
        dataset_id="cpu_activity",
        openml_name="cpu_activity",
        data_id=44978,
        task_id=361256,
        task_sha256="b1c30c7e05e33edb539023dafd731711607b8e5e96279a588979e0ec1bbc7c9b",
        target_column="usr",
        rows=8_192,
        predictors=21,
        file_sha256="ab02dc78a837a939da07fb9d95269b563d4504cb849ede1c4b7b8c02e713ed89",
        data_sha256="ab02dc78a837a939da07fb9d95269b563d4504cb849ede1c4b7b8c02e713ed89",
    ),
    CachedOpenMLSpec(
        dataset_id="kin8nm",
        openml_name="kin8nm",
        data_id=44980,
        task_id=361258,
        task_sha256="54942ec7994eba3fbd49812aecd720cc0d8b2e5486b8d7c24c88d3c16f9cfc14",
        target_column="y",
        rows=8_192,
        predictors=8,
        file_sha256="81b4b681d4134670f04434e3815037c5ee9e2ea54551ebe2d8b7344dbc7b4210",
        data_sha256="81b4b681d4134670f04434e3815037c5ee9e2ea54551ebe2d8b7344dbc7b4210",
    ),
    CachedOpenMLSpec(
        dataset_id="pumadyn32nh",
        openml_name="pumadyn32nh",
        data_id=44981,
        task_id=361259,
        task_sha256="c528a76d405dfd9b4764f56b45b83be5f05f77b31902913698ca4873fd7599c3",
        target_column="thetadd6",
        rows=8_192,
        predictors=32,
        file_sha256="0756f4da4500b61e2025862e2b550ee83fc8684616b1a77cebf6dccd5e13d705",
        data_sha256="0756f4da4500b61e2025862e2b550ee83fc8684616b1a77cebf6dccd5e13d705",
    ),
    CachedOpenMLSpec(
        dataset_id="energy_efficiency",
        openml_name="energy_efficiency",
        data_id=44960,
        task_id=361617,
        task_sha256="ff4890578ca20d455e3fd30670771cdb0c659f781b0817aa313f0cc80bbf7692",
        target_column="heating_load",
        rows=768,
        predictors=9,
        file_sha256="7d52e49858ff8785d6d4cfe36a4202ffcb0c83df084f1582e666e17a6fa69b24",
        data_sha256="7d52e49858ff8785d6d4cfe36a4202ffcb0c83df084f1582e666e17a6fa69b24",
    ),
    CachedOpenMLSpec(
        dataset_id="cars",
        openml_name="cars",
        data_id=44994,
        task_id=361622,
        task_sha256="b0b70ad95d9c4a54e356c7e209100a18589ca89e64ca52a84ccc6c1daf60552b",
        target_column="Price",
        rows=804,
        predictors=17,
        file_sha256="98a42023fa9701a1266dcea95b0808807f33b1f346d24b5503d89048656ae48c",
        data_sha256="98a42023fa9701a1266dcea95b0808807f33b1f346d24b5503d89048656ae48c",
    ),
    CachedOpenMLSpec(
        dataset_id="space_ga",
        openml_name="space_ga",
        data_id=45402,
        task_id=361623,
        task_sha256="addedb9730e52637a9ed8dfae8400b2a722fbd493d68bd38b84299cb45beb7e8",
        target_column="ln_votes_pop",
        rows=3_107,
        predictors=6,
        file_sha256="417f4852334046bd927e71fbbe51557071fc3679cb053fe504ec13dfa0ba4291",
        data_sha256="417f4852334046bd927e71fbbe51557071fc3679cb053fe504ec13dfa0ba4291",
    ),
)
CACHED_OPENML_BY_ID = {spec.dataset_id: spec for spec in CACHED_OPENML_SPECS}


@dataclass(frozen=True, slots=True)
class CachedOpenMLPanelManifest:
    path: Path
    file_sha256: str
    canonical_payload_sha256: str
    payload: dict[str, Any]
    dataset_ids: tuple[str, ...]
    context_policy: str


@dataclass(frozen=True, slots=True)
class FetchedCachedOpenMLDataset:
    spec: CachedOpenMLSpec
    dataset: RealDataset
    source_manifest: dict[str, Any]
    source_manifest_sha256: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm_name(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    return text.strip()


def _require_equal(*, field: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise RuntimeError(
            f"cached OpenML panel manifest drift at {field}: expected "
            f"{expected!r}, got {observed!r}"
        )


def load_cached_openml_panel_manifest(path: Path) -> CachedOpenMLPanelManifest:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing cached OpenML panel manifest: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("cached OpenML panel manifest must be a YAML mapping")
    _require_equal(
        field="schema_version",
        expected=CACHED_OPENML_PREREG_SCHEMA,
        observed=payload.get("schema_version"),
    )
    _require_equal(
        field="panel_id", expected=CACHED_OPENML_PANEL_ID, observed=payload.get("panel_id")
    )
    status = payload.get("status")
    if not isinstance(status, Mapping):
        raise RuntimeError("cached OpenML panel manifest status must be a mapping")
    for key, expected in {
        "registration": "candidate_preregistered",
        "execution": "not_run",
        "data_materialization": "pinned_cache_available_not_evaluated",
        "empirical_claim": "none",
    }.items():
        _require_equal(field=f"status.{key}", expected=expected, observed=status.get(key))
    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("cached OpenML panel model must be a mapping")
    for key, expected in {
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
    }.items():
        _require_equal(field=f"model.{key}", expected=expected, observed=model.get(key))
    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise RuntimeError("cached OpenML panel tokenizer must be a mapping")
    for key, expected in {
        "version": "cell-tokenizer.v2",
        "nominal_plan": "source_scoped_frozen_codebook.v2",
        "codebook_size": 100,
        "codebook_seed": 1729,
    }.items():
        _require_equal(field=f"tokenizer.{key}", expected=expected, observed=tokenizer.get(key))
    source_contract = payload.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError("cached OpenML panel source_contract must be a mapping")
    for key, expected in {
        "provider": "OpenML cached ARFF",
        "api": "scipy.io.arff.loadarff",
        "identity_key": "data_id",
        "parser": "liac-arff",
        "cache": True,
    }.items():
        _require_equal(
            field=f"source_contract.{key}", expected=expected, observed=source_contract.get(key)
        )
    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list):
        raise RuntimeError("cached OpenML panel datasets must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in raw_datasets:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("dataset_id"), str):
            raise RuntimeError("cached OpenML panel dataset entries are invalid")
        dataset_id = str(entry["dataset_id"])
        if dataset_id in by_id:
            raise RuntimeError(f"duplicate cached OpenML dataset: {dataset_id}")
        by_id[dataset_id] = entry
    expected_ids = tuple(spec.dataset_id for spec in CACHED_OPENML_SPECS)
    if set(by_id) != set(expected_ids):
        raise RuntimeError(f"cached OpenML panel must contain exactly {expected_ids!r}")
    for spec in CACHED_OPENML_SPECS:
        entry = by_id[spec.dataset_id]
        expected = {
            "openml_name": spec.openml_name,
            "data_id": spec.data_id,
            "task_id": spec.task_id,
            "task_sha256": spec.task_sha256,
            "target_column": spec.target_column,
            "task": spec.task,
            "shape": {"rows": spec.rows, "predictors": spec.predictors},
            "file_sha256": spec.file_sha256,
            "data_sha256": spec.data_sha256,
        }
        for key, value in expected.items():
            _require_equal(
                field=f"datasets.{spec.dataset_id}.{key}", expected=value, observed=entry.get(key)
            )
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RuntimeError(f"datasets.{spec.dataset_id}.path must be a non-empty string")
    evaluation = payload.get("evaluation_design")
    if not isinstance(evaluation, Mapping):
        raise RuntimeError("cached OpenML panel evaluation_design must be a mapping")
    for key, expected in {
        "checkpoint_seeds": [1729, 2718, 31415],
        "split_seeds": [1729, 2718, 31415],
        "context_policy": "full_train",
        "train_fraction": 0.7,
        "context_rows": "all_train_partition_rows",
        "query_policy": "all_heldout_rows",
        "query_limit": None,
        "query_readout_chunk_rows": 64,
    }.items():
        _require_equal(
            field=f"evaluation_design.{key}", expected=expected, observed=evaluation.get(key)
        )
    frozen = payload.get("frozen_protocol")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("cached OpenML panel frozen_protocol must be a mapping")
    for key, expected in {
        "arms": ["pretrained_frozen", "random_init_frozen", "pretrained_shuffled"],
        "one_independent_model_instance_per_arm": True,
        "requires_grad": False,
        "eval_mode": True,
        "inference_mode": True,
        "optimizer_created": False,
    }.items():
        _require_equal(field=f"frozen_protocol.{key}", expected=expected, observed=frozen.get(key))
    return CachedOpenMLPanelManifest(
        path=resolved,
        file_sha256=_file_sha256(resolved),
        canonical_payload_sha256=_canonical_sha256(payload),
        payload=payload,
        dataset_ids=expected_ids,
        context_policy="full_train",
    )


def _materialized_array_sha256(features: np.ndarray, response: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (features, response):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(contiguous.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def fetch_cached_openml_dataset(
    dataset_id: str,
    *,
    panel_manifest: CachedOpenMLPanelManifest,
) -> FetchedCachedOpenMLDataset:
    try:
        spec = CACHED_OPENML_BY_ID[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown cached OpenML dataset: {dataset_id}") from exc
    entry = next(
        item for item in panel_manifest.payload["datasets"] if item["dataset_id"] == dataset_id
    )
    path = Path(str(entry["path"])).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"cached OpenML ARFF is missing: {path}")
    observed_file_sha256 = _file_sha256(path)
    if observed_file_sha256 != spec.file_sha256:
        raise RuntimeError(
            f"cached OpenML file hash drift for {dataset_id}: expected "
            f"{spec.file_sha256}, got {observed_file_sha256}"
        )
    try:
        from scipy.io import arff

        raw, metadata = arff.loadarff(str(path))
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("cached OpenML loading requires scipy") from exc
    names = list(metadata.names())
    normalized_names = [_norm_name(name) for name in names]
    target_norm = _norm_name(spec.target_column)
    matches = [
        index
        for index, name in enumerate(normalized_names)
        if name.casefold() == target_norm.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"cached OpenML target column is not uniquely resolved for {dataset_id}")
    target_index = matches[0]
    types = list(metadata.types())
    if any(str(kind).casefold() not in {"numeric", "real", "integer"} for kind in types):
        raise RuntimeError(f"cached OpenML dataset {dataset_id} contains a non-numeric column")
    try:
        response = np.asarray(raw[names[target_index]], dtype=np.float32)
        feature_columns = [
            np.asarray(raw[name], dtype=np.float32)
            for index, name in enumerate(names)
            if index != target_index
        ]
        features = np.column_stack(feature_columns).astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cached OpenML dataset {dataset_id} is not numeric") from exc
    if features.shape != (spec.rows, spec.predictors) or response.shape != (spec.rows,):
        raise RuntimeError(
            f"cached OpenML shape drift for {dataset_id}: expected "
            f"{(spec.rows, spec.predictors)}, {spec.rows}; got "
            f"{features.shape}, {response.shape}"
        )
    if not np.isfinite(features).all() or not np.isfinite(response).all():
        raise RuntimeError(f"cached OpenML dataset {dataset_id} contains missing/non-finite values")
    dataset = RealDataset(
        dataset_id=spec.dataset_id,
        task=spec.task,
        features=features,
        response=response,
        source=f"OpenML cached ARFF data_id={spec.data_id} task_id={spec.task_id} parser=liac-arff",
    )
    source_manifest: dict[str, Any] = {
        "schema_version": CACHED_OPENML_SOURCE_SCHEMA,
        "panel_id": CACHED_OPENML_PANEL_ID,
        "dataset_id": spec.dataset_id,
        "source": {
            "provider": "OpenML cached ARFF",
            "data_id": spec.data_id,
            "name": spec.openml_name,
            "task_id": spec.task_id,
            "task_sha256": spec.task_sha256,
            "target_column": spec.target_column,
            "resolved_target_column": names[target_index],
            "license": spec.license,
            "status": "active_snapshot",
        },
        "fetch": {
            "api": "scipy.io.arff.loadarff",
            "parser": "liac-arff",
            "cache": True,
            "path": str(path),
        },
        "validation": {
            "expected_shape": [spec.rows, spec.predictors],
            "observed_shape": list(features.shape),
            "observed_missing_or_non_finite_values": 0,
            "numeric_columns_only": True,
        },
        "materialized": {
            "file_sha256": observed_file_sha256,
            "features_dtype": str(features.dtype),
            "response_dtype": str(response.dtype),
            "array_sha256": _materialized_array_sha256(features, response),
            "real_dataset_content_sha256": dataset.content_hash,
        },
    }
    return FetchedCachedOpenMLDataset(
        spec=spec,
        dataset=dataset,
        source_manifest=source_manifest,
        source_manifest_sha256=_canonical_sha256(source_manifest),
    )


def is_cached_openml_panel_manifest(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == CACHED_OPENML_PREREG_SCHEMA
    )


def cached_openml_panel_manifest_identity(panel: CachedOpenMLPanelManifest) -> dict[str, Any]:
    return {
        "panel_id": CACHED_OPENML_PANEL_ID,
        "schema_version": panel.payload["schema_version"],
        "file_sha256": panel.file_sha256,
        "canonical_payload_sha256": panel.canonical_payload_sha256,
        "registered_dataset_ids": list(panel.dataset_ids),
    }


def build_cached_openml_materialization_manifest(
    panel: CachedOpenMLPanelManifest,
    fetched: Sequence[FetchedCachedOpenMLDataset],
) -> dict[str, Any]:
    if {item.spec.dataset_id for item in fetched} != set(panel.dataset_ids):
        raise ValueError("cached OpenML materialization must cover the complete panel")
    body: dict[str, Any] = {
        "schema_version": "tabu.tabubase-cached-openml-evaluation-materialization.v1",
        "panel_id": CACHED_OPENML_PANEL_ID,
        "panel_manifest_file_sha256": panel.file_sha256,
        "panel_manifest_canonical_payload_sha256": panel.canonical_payload_sha256,
        "datasets": [
            {
                "dataset_id": item.spec.dataset_id,
                "source_manifest_sha256": item.source_manifest_sha256,
                "materialized_array_sha256": item.source_manifest["materialized"]["array_sha256"],
                "file_sha256": item.source_manifest["materialized"]["file_sha256"],
            }
            for item in fetched
        ],
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}
