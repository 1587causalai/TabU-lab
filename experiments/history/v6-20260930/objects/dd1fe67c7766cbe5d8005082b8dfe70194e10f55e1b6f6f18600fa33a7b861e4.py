"""Pinned OpenML inputs for the TabUBase real frozen-ICL ``new6`` panel.

This module owns data acquisition and validation only.  It deliberately does
not import or invoke the frozen-ICL evaluator, create an optimizer, or run an
experiment.  The OpenML ``data_id`` identifies one immutable dataset version;
the returned metadata and materialized arrays are still checked fail-closed
against the preregistered identity before they can enter an evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from .tabubase_real_benchmark import RealDataset

OPENML_NEW6_PANEL_ID = "tabubase-real-frozen-icl-openml-numeric-nomissing-new6-v1"
OPENML_NEW6_MANIFEST_SCHEMA = "tabu.tabubase-openml-new6-panel-manifest.v1"
OPENML_NEW6_SOURCE_SCHEMA = "tabu.tabubase-openml-source-manifest.v1"
OPENML_NEW6_PREREG_SCHEMA = "tabu.transfer-base-real-frozen-icl-openml-new6.v1"
OPENML_NEW6_FULL_CONTEXT_PREREG_SCHEMA = (
    "tabu.transfer-base-real-full-context-frozen-icl-openml-new6.v1"
)

TaskKind = Literal["classification", "regression"]
FetchOpenML = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class OpenMLNew6Spec:
    """One immutable OpenML dataset identity and its expected materialization."""

    dataset_id: str
    openml_name: str
    data_id: int
    version: int
    task: TaskKind
    rows: int
    predictors: int
    classes: int | None
    upstream_md5: str
    license: str
    target_column: str = "default-target"
    expected_missing_values: int = 0


OPENML_NEW6_SPECS = (
    OpenMLNew6Spec(
        dataset_id="banknote_authentication",
        openml_name="banknote-authentication",
        data_id=1462,
        version=1,
        task="classification",
        rows=1_372,
        predictors=4,
        classes=2,
        upstream_md5="baa2dc5b745775a943ebeb9c276401f8",
        license="Public",
    ),
    OpenMLNew6Spec(
        dataset_id="segment",
        openml_name="segment",
        data_id=36,
        version=1,
        task="classification",
        rows=2_310,
        predictors=19,
        classes=7,
        upstream_md5="037621812203bfd85c8dab1d7f16ebc6",
        license="Public",
    ),
    OpenMLNew6Spec(
        dataset_id="spambase",
        openml_name="spambase",
        data_id=44,
        version=1,
        task="classification",
        rows=4_601,
        predictors=57,
        classes=2,
        upstream_md5="d9ace01aeac3461e326a8e1b2d53fd84",
        license="Public",
    ),
    OpenMLNew6Spec(
        dataset_id="airfoil_self_noise",
        openml_name="airfoil_self_noise",
        data_id=43_919,
        version=1,
        task="regression",
        rows=1_503,
        predictors=5,
        classes=None,
        upstream_md5="79f7daecfdeb6457ba37fd8982966048",
        license="CC BY 4.0",
    ),
    OpenMLNew6Spec(
        dataset_id="concrete_compressive_strength",
        openml_name="concrete_compressive_strength",
        data_id=44_959,
        version=7,
        task="regression",
        rows=1_030,
        predictors=8,
        classes=None,
        upstream_md5="1906eae71bd8b8142d079a4b966549ac",
        license="CC BY 4.0",
    ),
    OpenMLNew6Spec(
        dataset_id="qsar_fish_toxicity",
        openml_name="QSAR_fish_toxicity",
        data_id=44_970,
        version=7,
        task="regression",
        rows=908,
        predictors=6,
        classes=None,
        upstream_md5="4900e250afcfee9d523aba87895260fe",
        license="CC BY 4.0",
    ),
)
OPENML_NEW6_BY_ID = {spec.dataset_id: spec for spec in OPENML_NEW6_SPECS}


@dataclass(frozen=True, slots=True)
class FetchedOpenMLDataset:
    """A validated array carrier plus its source-provenance receipt."""

    spec: OpenMLNew6Spec
    dataset: RealDataset
    source_manifest: dict[str, Any]
    source_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class OpenMLNew6PanelManifest:
    """A locally read and strictly validated new6 preregistration carrier."""

    path: Path
    file_sha256: str
    canonical_payload_sha256: str
    payload: dict[str, Any]
    dataset_ids: tuple[str, ...]
    context_policy: str


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(*, field: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise RuntimeError(
            f"OpenML new6 panel manifest drift at {field}: expected {expected!r}, got {observed!r}"
        )


def load_openml_new6_panel_manifest(path: Path) -> OpenMLNew6PanelManifest:
    """Read the checked-in panel manifest and reject any identity drift."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing OpenML new6 panel manifest: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("OpenML new6 panel manifest must be a YAML mapping")

    schema_version = payload.get("schema_version")
    if schema_version not in {
        OPENML_NEW6_PREREG_SCHEMA,
        OPENML_NEW6_FULL_CONTEXT_PREREG_SCHEMA,
    }:
        raise RuntimeError(
            "OpenML new6 panel manifest drift at schema_version: expected one of "
            f"{OPENML_NEW6_PREREG_SCHEMA!r} or "
            f"{OPENML_NEW6_FULL_CONTEXT_PREREG_SCHEMA!r}, got {schema_version!r}"
        )
    _require_equal(
        field="panel_id", expected=OPENML_NEW6_PANEL_ID, observed=payload.get("panel_id")
    )
    status = payload.get("status")
    if not isinstance(status, Mapping):
        raise RuntimeError("OpenML new6 panel manifest status must be a mapping")
    _require_equal(
        field="status.registration",
        expected="candidate_preregistered",
        observed=status.get("registration"),
    )
    _require_equal(field="status.execution", expected="not_run", observed=status.get("execution"))
    _require_equal(
        field="status.data_materialization",
        expected=(
            "not_run"
            if schema_version == OPENML_NEW6_PREREG_SCHEMA
            else "pinned_cache_available_not_evaluated"
        ),
        observed=status.get("data_materialization"),
    )
    _require_equal(
        field="status.empirical_claim", expected="none", observed=status.get("empirical_claim")
    )

    model = payload.get("model")
    if not isinstance(model, Mapping):
        raise RuntimeError("OpenML new6 panel manifest model must be a mapping")
    for key, expected in {
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
    }.items():
        _require_equal(field=f"model.{key}", expected=expected, observed=model.get(key))

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise RuntimeError("OpenML new6 panel manifest tokenizer must be a mapping")
    for key, expected in {
        "version": "cell-tokenizer.v2",
        "nominal_plan": "source_scoped_frozen_codebook.v2",
        "codebook_size": 100,
        "codebook_seed": 1729,
    }.items():
        _require_equal(field=f"tokenizer.{key}", expected=expected, observed=tokenizer.get(key))

    source_contract = payload.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise RuntimeError("OpenML new6 panel manifest source_contract must be a mapping")
    required_source_contract = {
        "provider": "OpenML",
        "api": "sklearn.datasets.fetch_openml",
        "identity_key": "data_id",
        "target_column": "default-target",
        "as_frame": False,
        "parser": "liac-arff",
        "cache": True,
    }
    for key, expected in required_source_contract.items():
        _require_equal(
            field=f"source_contract.{key}", expected=expected, observed=source_contract.get(key)
        )

    raw_datasets = payload.get("datasets")
    if not isinstance(raw_datasets, list):
        raise RuntimeError("OpenML new6 panel manifest datasets must be a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for entry in raw_datasets:
        if not isinstance(entry, Mapping):
            raise RuntimeError("OpenML new6 panel manifest dataset entries must be mappings")
        dataset_id = entry.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise RuntimeError("OpenML new6 panel manifest dataset_id must be a non-empty string")
        if dataset_id in by_id:
            raise RuntimeError(f"duplicate OpenML new6 dataset entry: {dataset_id}")
        by_id[dataset_id] = entry

    expected_ids = tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
    if set(by_id) != set(expected_ids):
        raise RuntimeError(f"OpenML new6 panel must contain exactly {expected_ids!r}")
    for spec in OPENML_NEW6_SPECS:
        entry = by_id[spec.dataset_id]
        expected_fields = {
            "openml_name": spec.openml_name,
            "data_id": spec.data_id,
            "version": spec.version,
            "upstream_md5": spec.upstream_md5,
            "license": spec.license,
            "task": spec.task,
            "shape": {"rows": spec.rows, "predictors": spec.predictors},
            "classes": spec.classes,
            "missing_values": spec.expected_missing_values,
        }
        for key, expected in expected_fields.items():
            _require_equal(
                field=f"datasets.{spec.dataset_id}.{key}",
                expected=expected,
                observed=entry.get(key),
            )

    evaluation = payload.get("evaluation_design")
    if not isinstance(evaluation, Mapping):
        raise RuntimeError("OpenML new6 panel manifest evaluation_design must be a mapping")
    if schema_version == OPENML_NEW6_PREREG_SCHEMA:
        context_policy = "low_shot_grid"
        expected_evaluation = {
            "checkpoint_seeds": [1729, 2718, 31415],
            "split_seeds": [1729, 2718, 31415],
            "context_sizes": [0, 1, 2, 4, 8, 16, 32],
            "query_limit": 256,
            "query_chunk_rows": 64,
            "classification_primary_curve": "exclude K below the declared class count",
            "regression_primary_curve": "K in [1, 2, 4, 8, 16, 32]",
        }
    else:
        context_policy = "full_train"
        expected_evaluation = {
            "checkpoint_seeds": [1729, 2718, 31415],
            "split_seeds": [1729, 2718, 31415],
            "context_policy": "full_train",
            "train_fraction": 0.7,
            "context_rows": "all_train_partition_rows",
            "query_policy": "all_heldout_rows",
            "query_limit": None,
            "query_chunk_rows": 64,
            "classification_primary_metrics": ["normalized_nll", "accuracy"],
            "regression_primary_metrics": ["scaled_rmse", "scaled_mae", "r2"],
        }
    for key, expected in expected_evaluation.items():
        _require_equal(
            field=f"evaluation_design.{key}",
            expected=expected,
            observed=evaluation.get(key),
        )

    frozen = payload.get("frozen_protocol")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("OpenML new6 panel manifest frozen_protocol must be a mapping")
    for key, expected in {
        "arms": ["pretrained_frozen", "random_init_frozen", "pretrained_shuffled"],
        "one_independent_model_instance_per_arm": True,
        "requires_grad": False,
        "eval_mode": True,
        "inference_mode": True,
        "optimizer_created": False,
    }.items():
        _require_equal(field=f"frozen_protocol.{key}", expected=expected, observed=frozen.get(key))
    return OpenMLNew6PanelManifest(
        path=resolved,
        file_sha256=_file_sha256(resolved),
        canonical_payload_sha256=_canonical_sha256(payload),
        payload=payload,
        dataset_ids=expected_ids,
        context_policy=context_policy,
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


def _require_detail(details: Mapping[str, Any], key: str) -> Any:
    value = details.get(key)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"OpenML metadata is missing required field {key!r}")
    return value


def _validate_openml_details(spec: OpenMLNew6Spec, details: Mapping[str, Any]) -> str:
    try:
        data_id = int(_require_detail(details, "id"))
        version = int(_require_detail(details, "version"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("OpenML id/version metadata is not integral") from exc
    if data_id != spec.data_id:
        raise RuntimeError(f"OpenML data_id drift: expected {spec.data_id}, got {data_id}")
    if version != spec.version:
        raise RuntimeError(f"OpenML version drift: expected {spec.version}, got {version}")

    name = str(_require_detail(details, "name"))
    if name.casefold() != spec.openml_name.casefold():
        raise RuntimeError(f"OpenML name drift: expected {spec.openml_name!r}, got {name!r}")
    upstream_md5 = str(_require_detail(details, "md5_checksum")).casefold()
    if upstream_md5 != spec.upstream_md5.casefold():
        raise RuntimeError(
            f"OpenML md5 drift for {spec.dataset_id}: expected {spec.upstream_md5}, "
            f"got {upstream_md5}"
        )
    license_name = " ".join(str(_require_detail(details, "licence")).split())
    if license_name.casefold() != spec.license.casefold():
        raise RuntimeError(
            f"OpenML license drift for {spec.dataset_id}: expected {spec.license!r}, "
            f"got {license_name!r}"
        )
    status = str(_require_detail(details, "status"))
    if status.casefold() != "active":
        raise RuntimeError(f"OpenML dataset {spec.data_id} is not active: {status!r}")
    return str(_require_detail(details, "default_target_attribute"))


def _normalized_class_labels(raw_response: Any) -> tuple[np.ndarray, dict[str, int]]:
    raw = np.asarray(raw_response)
    if raw.ndim != 1:
        raise RuntimeError("OpenML classification target must be one-dimensional")
    labels: list[str] = []
    for value in raw.tolist():
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            raise RuntimeError("OpenML classification target contains missing values")
        label = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        if not label.strip() or label == "?":
            raise RuntimeError("OpenML classification target contains missing values")
        labels.append(label)
    unique_labels = sorted(set(labels))
    mapping = {label: index for index, label in enumerate(unique_labels)}
    encoded = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    return encoded, mapping


def preregistered_openml_new6_manifest() -> dict[str, Any]:
    """Return the static, no-network preregistration identity for the panel."""

    body: dict[str, Any] = {
        "schema_version": OPENML_NEW6_MANIFEST_SCHEMA,
        "panel_id": OPENML_NEW6_PANEL_ID,
        "registration_status": "candidate_preregistered",
        "execution_status": "not_run",
        "fetch_contract": {
            "provider": "OpenML",
            "api": "sklearn.datasets.fetch_openml",
            "identity_key": "data_id",
            "target_column": "default-target",
            "as_frame": False,
            "parser": "liac-arff",
            "cache": True,
        },
        "datasets": [asdict(spec) for spec in OPENML_NEW6_SPECS],
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


def fetch_openml_new6_dataset(
    dataset_id: str,
    *,
    fetcher: FetchOpenML | None = None,
    cache: bool = True,
    data_home: Path | None = None,
    sklearn_version: str | None = None,
) -> FetchedOpenMLDataset:
    """Fetch and fail-closed validate one preregistered OpenML dataset.

    ``fetcher`` exists for deterministic tests and controlled runtimes.  When
    omitted, scikit-learn is imported lazily so the core package keeps OpenML
    access as an explicit optional operation.
    """

    try:
        spec = OPENML_NEW6_BY_ID[dataset_id]
    except KeyError as exc:
        raise ValueError(f"unknown OpenML new6 dataset: {dataset_id}") from exc
    injected_fetcher = fetcher is not None
    if not injected_fetcher:
        try:
            import sklearn
            from sklearn.datasets import fetch_openml as sklearn_fetch_openml
        except ImportError as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("OpenML new6 loading requires scikit-learn") from exc
        fetcher = sklearn_fetch_openml
        resolved_sklearn_version = sklearn.__version__
    else:
        if sklearn_version is not None:
            resolved_sklearn_version = sklearn_version
        else:
            try:
                resolved_sklearn_version = importlib.metadata.version("scikit-learn")
            except importlib.metadata.PackageNotFoundError:
                resolved_sklearn_version = "unavailable-custom-fetcher"

    if not resolved_sklearn_version.strip():
        raise RuntimeError("scikit-learn version provenance must be non-empty")

    fetch_kwargs: dict[str, Any] = {
        "data_id": spec.data_id,
        "target_column": spec.target_column,
        "as_frame": False,
        "parser": "liac-arff",
        "cache": cache,
    }
    resolved_data_home: str | None = None
    if data_home is not None:
        resolved_path = data_home.expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"OpenML data_home is not a directory: {resolved_path}")
        resolved_data_home = str(resolved_path)
        fetch_kwargs["data_home"] = resolved_data_home
    bunch = fetcher(
        **fetch_kwargs,
    )
    details = getattr(bunch, "details", None)
    if not isinstance(details, Mapping):
        raise RuntimeError("fetch_openml result is missing a metadata details mapping")
    resolved_target = _validate_openml_details(spec, details)

    try:
        features = np.asarray(bunch.data, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OpenML dataset {dataset_id} has non-numeric predictors") from exc
    if features.shape != (spec.rows, spec.predictors):
        raise RuntimeError(
            f"OpenML shape drift for {dataset_id}: expected "
            f"{(spec.rows, spec.predictors)}, got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise RuntimeError(f"OpenML dataset {dataset_id} contains missing/non-finite predictors")

    label_mapping: dict[str, int] | None
    if spec.task == "classification":
        response, label_mapping = _normalized_class_labels(bunch.target)
        if len(response) != spec.rows:
            raise RuntimeError(f"OpenML target row count drift for {dataset_id}")
        if spec.classes is None or len(label_mapping) != spec.classes:
            raise RuntimeError(
                f"OpenML class-count drift for {dataset_id}: expected {spec.classes}, "
                f"got {len(label_mapping)}"
            )
    else:
        try:
            response = np.asarray(bunch.target, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenML dataset {dataset_id} has a non-numeric target") from exc
        label_mapping = None
        if response.shape != (spec.rows,):
            raise RuntimeError(f"OpenML target shape drift for {dataset_id}: got {response.shape}")
        if not np.isfinite(response).all():
            raise RuntimeError(f"OpenML dataset {dataset_id} has a missing/non-finite target")

    source = (
        f"OpenML data_id={spec.data_id} version={spec.version} "
        f"md5={spec.upstream_md5} parser=liac-arff"
    )
    dataset = RealDataset(
        dataset_id=spec.dataset_id,
        task=spec.task,
        features=features,
        response=response,
        source=source,
    )
    source_manifest: dict[str, Any] = {
        "schema_version": OPENML_NEW6_SOURCE_SCHEMA,
        "panel_id": OPENML_NEW6_PANEL_ID,
        "dataset_id": spec.dataset_id,
        "source": {
            "provider": "OpenML",
            "data_id": spec.data_id,
            "name": spec.openml_name,
            "version": spec.version,
            "license": spec.license,
            "upstream_md5": spec.upstream_md5,
            "requested_target_column": spec.target_column,
            "resolved_target_column": resolved_target,
            "status": "active",
        },
        "fetch": {
            "api": "sklearn.datasets.fetch_openml",
            "fetcher_mode": "injected_callable" if injected_fetcher else "sklearn_native",
            "scikit_learn_version": resolved_sklearn_version,
            "as_frame": False,
            "parser": "liac-arff",
            "cache": cache,
            "data_home": resolved_data_home,
        },
        "validation": {
            "expected_shape": [spec.rows, spec.predictors],
            "observed_shape": list(features.shape),
            "expected_missing_values": spec.expected_missing_values,
            "observed_missing_or_non_finite_values": 0,
            "expected_classes": spec.classes,
            "observed_classes": len(label_mapping) if label_mapping is not None else None,
        },
        "materialized": {
            "features_dtype": str(features.dtype),
            "response_dtype": str(response.dtype),
            "label_mapping": label_mapping,
            "array_sha256": _materialized_array_sha256(features, response),
            "real_dataset_content_sha256": dataset.content_hash,
        },
    }
    return FetchedOpenMLDataset(
        spec=spec,
        dataset=dataset,
        source_manifest=source_manifest,
        source_manifest_sha256=_canonical_sha256(source_manifest),
    )


def build_fetched_openml_new6_panel_manifest(
    fetched: Sequence[FetchedOpenMLDataset],
) -> dict[str, Any]:
    """Bind a complete validated ``new6`` materialization into one manifest."""

    by_id = {item.spec.dataset_id: item for item in fetched}
    if len(by_id) != len(fetched):
        raise ValueError("fetched OpenML new6 datasets must be unique")
    expected_ids = tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
    if set(by_id) != set(expected_ids):
        raise ValueError(f"fetched panel must contain exactly {expected_ids!r}")
    body: dict[str, Any] = {
        "schema_version": OPENML_NEW6_MANIFEST_SCHEMA,
        "panel_id": OPENML_NEW6_PANEL_ID,
        "registration_status": "candidate_preregistered",
        "execution_status": "materialized_not_evaluated",
        "datasets": [
            {
                "dataset_id": dataset_id,
                "source_manifest_sha256": by_id[dataset_id].source_manifest_sha256,
                "source_manifest": by_id[dataset_id].source_manifest,
            }
            for dataset_id in expected_ids
        ],
    }
    return {**body, "manifest_sha256": _canonical_sha256(body)}


__all__ = [
    "OPENML_NEW6_BY_ID",
    "OPENML_NEW6_FULL_CONTEXT_PREREG_SCHEMA",
    "OPENML_NEW6_MANIFEST_SCHEMA",
    "OPENML_NEW6_PANEL_ID",
    "OPENML_NEW6_PREREG_SCHEMA",
    "OPENML_NEW6_SOURCE_SCHEMA",
    "OPENML_NEW6_SPECS",
    "FetchOpenML",
    "FetchedOpenMLDataset",
    "OpenMLNew6PanelManifest",
    "OpenMLNew6Spec",
    "build_fetched_openml_new6_panel_manifest",
    "fetch_openml_new6_dataset",
    "load_openml_new6_panel_manifest",
    "preregistered_openml_new6_manifest",
]
