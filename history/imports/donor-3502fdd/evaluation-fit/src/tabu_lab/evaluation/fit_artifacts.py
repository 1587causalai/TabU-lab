"""Immutable artifact and receipt bundles for fit-first experiment attempts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import yaml

from tabu_lab.contracts import (
    EvaluationBundle,
    PredictionBundle,
    canonical_hash,
    canonical_json,
    require_sha256,
    to_canonical_data,
)
from tabu_lab.evidence import (
    ArtifactRef,
    EnvironmentDisclosure,
    Receipt,
    ReceiptStatus,
    RunBundle,
    RunIdentity,
    read_receipt,
    write_receipt,
)
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    FormalAuthorizationReplaySession,
    FormalAuthorizationSummary,
    verify_formal_authorization,
)
from tabu_lab.evidence.public_safety import (
    contains_absolute_local_path,
    contains_private_identity_or_secret,
    is_sensitive_public_key,
)
from tabu_lab.evidence.source_identity import SourceIdentity
from tabu_lab.experiments import (
    DynamicsBlockKind,
    FitEvaluationBundle,
    FitExperimentSpec,
    FitStage,
    ModelSemanticConfig,
)
from tabu_lab.experiments.corpus_manifest import (
    CORPUS_COMPILER_BINDING_SCHEMA,
    corpus_compiler_episode_recipe_hashes,
    validate_corpus_compiler_binding_manifest,
)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_INFRASTRUCTURE_ARTIFACTS = {
    "artifacts.sha256",
    "receipt.json",
    "run_bundle.json",
    "run_manifest.json",
}
_REQUIRED_PAYLOAD_FILES = {
    "baselines.json",
    "compiler-manifest.json",
    "dataset-manifest.json",
    "environment.json",
    "evaluation.json",
    "feasibility.json",
    "forward-traces.json",
    "metrics.jsonl",
    "preregistration.yaml",
    "split-manifest.json",
    "verdict.md",
}
_REQUIRED_RUNTIME_FAILURE_FILES = {
    "baselines.json",
    "compiler-manifest.json",
    "dataset-manifest.json",
    "environment.json",
    "failure.json",
    "feasibility.json",
    "preregistration.yaml",
    "split-manifest.json",
    "verdict.md",
}
_FORMAL_RESOLVED_CONFIGS = frozenset(
    {"code", "experiment", "semantic", "training", "execution", "seeds"}
)
_FORMAL_EXECUTION_RUNTIME_FIELDS = frozenset(
    {"resolved_device", "environment_hash", "host_class", "torch_version", "python_version"}
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "schema_version",
        "host_class",
        "os_family",
        "architecture",
        "python_version",
        "torch_version",
        "device",
        "accelerator",
        "cuda_version",
        "cudnn_version",
        "deterministic_algorithms",
        "dependency_lock_hash",
        "container_image_hash",
    }
)
@dataclass(frozen=True, slots=True)
class FitAttemptArtifacts:
    directory: Path
    receipt: Path
    checksums: Path
    run_bundle: Path
    checkpoint: Path | None
    receipt_hash: str


def _write_json(path: Path, payload: Any) -> None:
    canonical = to_canonical_data(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(canonical, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write canonical newline-delimited JSON, one mapping per physical line."""

    if isinstance(rows, (str, bytes, bytearray, Mapping)):
        raise TypeError("metrics must be a sequence of JSON object rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError(f"metrics row {index} must be a mapping")
            canonical = to_canonical_data(row)
            if not isinstance(canonical, dict):  # defensive: Mapping canonicalizes to dict
                raise TypeError(f"metrics row {index} must canonicalize to a JSON object")
            target.write(
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _normalize_metrics(
    metrics: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], FitEvaluationBundle | None]:
    """Preserve the existing metrics API while projecting it to canonical JSONL.

    The fit runner supplies ``{"history": ..., "summary": ...}``; history
    records are emitted first and the summary is the final JSONL row.  A typed
    ``FitEvaluationBundle`` embedded in that summary is revalidated here.  A
    plain mapping remains a backwards-compatible single metrics row.
    """

    fit_evaluation: FitEvaluationBundle | None = None
    if isinstance(metrics, Mapping):
        if "history" in metrics or "summary" in metrics:
            if "history" not in metrics or "summary" not in metrics:
                raise ValueError("structured metrics require both history and summary")
            history = metrics["history"]
            summary = metrics["summary"]
            if isinstance(history, (str, bytes, bytearray, Mapping)) or not isinstance(
                history, Sequence
            ):
                raise TypeError("metrics history must be a sequence of mapping rows")
            if not isinstance(summary, Mapping):
                raise TypeError("metrics summary must be a mapping")
            raw_fit_evaluation = summary.get("fit_evaluation")
            if raw_fit_evaluation is None:
                raise ValueError("structured metrics summary requires fit_evaluation")
            try:
                fit_evaluation = (
                    raw_fit_evaluation
                    if isinstance(raw_fit_evaluation, FitEvaluationBundle)
                    else FitEvaluationBundle.model_validate(raw_fit_evaluation)
                )
            except ValueError as exc:
                raise ValueError("metrics summary has an invalid fit_evaluation") from exc
            normalized_summary = dict(summary)
            normalized_summary["fit_evaluation"] = fit_evaluation
            rows = (*tuple(history), normalized_summary)
        else:
            rows = (metrics,)
    else:
        if isinstance(metrics, (str, bytes, bytearray)) or not isinstance(metrics, Sequence):
            raise TypeError("metrics must be a mapping or sequence of mapping rows")
        rows = tuple(metrics)
    if not all(isinstance(row, Mapping) for row in rows):
        raise TypeError("every metrics row must be a mapping")
    return tuple(rows), fit_evaluation


def _read_canonical_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc
    expected = (
        json.dumps(
            to_canonical_data(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if raw != expected:
        raise ValueError(f"non-canonical JSON artifact: {path.name}")
    return payload


def _read_compact_canonical_json(path: Path) -> Any:
    """Read runner-authored compact canonical JSON used by failure receipts."""

    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc
    if raw != canonical_json(payload) + "\n":
        raise ValueError(f"non-canonical JSON artifact: {path.name}")
    return payload


def _read_canonical_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("invalid metrics.jsonl encoding") from exc
    if raw and not raw.endswith("\n"):
        raise ValueError("metrics.jsonl must end with a newline")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        if not line:
            raise ValueError("metrics.jsonl cannot contain blank lines")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid metrics.jsonl row {index}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"metrics.jsonl row {index} must be a JSON object")
        if line != canonical_json(payload):
            raise ValueError(f"metrics.jsonl row {index} is not canonical JSON")
        rows.append(payload)
    return tuple(rows)


def _parse_formal_preregistration(preregistration_text: str) -> FitExperimentSpec:
    if not isinstance(preregistration_text, str):
        raise TypeError("preregistration_text must be a string")
    try:
        payload = yaml.safe_load(preregistration_text)
    except yaml.YAMLError as exc:
        raise ValueError("formal preregistration is not valid safe YAML") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("formal preregistration must contain exactly one mapping")
    try:
        return FitExperimentSpec.model_validate(payload)
    except ValueError as exc:
        raise ValueError("formal preregistration is not a valid FitExperimentSpec") from exc


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed mapping")
    return value


def _validate_code_manifest(
    code: Any,
    *,
    expected_hash: str,
    preregistration_text: str,
    expected_issuance_status: str,
) -> SourceIdentity:
    manifest = _require_mapping(code, name="resolved code manifest")
    if set(manifest) != {
        "schema_version",
        "mode",
        "root_label",
        "files",
        "source_identity",
    }:
        raise ValueError("resolved code manifest has an unexpected shape")
    if manifest.get("schema_version") != "tabu.source-tree.v3":
        raise ValueError("resolved code manifest has an unsupported schema")
    if manifest.get("mode") not in {"repository", "installed_package"}:
        raise ValueError("resolved code manifest has an invalid mode")
    files = manifest.get("files")
    if (
        isinstance(files, (str, bytes, bytearray, Mapping))
        or not isinstance(files, Sequence)
        or not files
    ):
        raise ValueError("resolved code manifest requires source file records")
    seen_paths: set[str] = set()
    for record in files:
        item = _require_mapping(record, name="resolved source file record")
        if set(item) != {"path", "sha256", "size"}:
            raise ValueError("resolved source file record has an unexpected shape")
        relative = item.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ValueError("resolved source file path is not portable")
        pure = PurePosixPath(relative)
        if (
            pure.is_absolute()
            or pure.as_posix() != relative
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ValueError("resolved source file path is unsafe")
        if relative in seen_paths:
            raise ValueError("resolved code manifest contains duplicate paths")
        seen_paths.add(relative)
        source_hash = item.get("sha256")
        if not isinstance(source_hash, str):
            raise ValueError("resolved source file has an invalid sha256")
        try:
            require_sha256(source_hash, field_name="source file sha256")
        except ValueError as exc:
            raise ValueError("resolved source file has an invalid sha256") from exc
        if type(item.get("size")) is not int or item["size"] < 0:
            raise ValueError("resolved source file size must be a non-negative integer")
    try:
        source_identity = SourceIdentity.model_validate(manifest.get("source_identity"))
    except ValueError as exc:
        raise ValueError("resolved code manifest has an invalid SourceIdentity") from exc
    if source_identity.issuance_status != expected_issuance_status:
        raise ValueError("SourceIdentity and attempt issuance status differ")
    expected_tree_hash = canonical_hash(
        {
            "schema_version": "tabu.source-tree-preimage.v1",
            "mode": manifest["mode"],
            "root_label": manifest["root_label"],
            "files": manifest["files"],
        }
    )
    if (
        source_identity.source_kind != "distribution"
        or source_identity.issuance_status == "formal"
    ) and source_identity.source_tree_hash != expected_tree_hash:
        raise ValueError("SourceIdentity source_tree_hash does not match source files")
    if source_identity.issuance_status == "formal" and source_identity.source_kind == "git":
        preregistration_hash = hashlib.sha256(preregistration_text.encode("utf-8")).hexdigest()
        if source_identity.preregistration_blob_hash != preregistration_hash:
            raise ValueError("formal SourceIdentity is not bound to the preregistration blob")
    if canonical_hash(manifest) != expected_hash:
        raise ValueError("resolved code manifest does not match RunIdentity code_hash")
    return source_identity


def _resolved_source_identity(resolved_configs: Mapping[str, Any]) -> SourceIdentity:
    code = _require_mapping(
        resolved_configs.get("code"),
        name="resolved code manifest",
    )
    try:
        return SourceIdentity.model_validate(code.get("source_identity"))
    except ValueError as exc:
        raise ValueError("resolved code manifest has an invalid SourceIdentity") from exc


def _verify_recorded_formal_authorization(
    *,
    issuance_status: str,
    bundle_metadata: Mapping[str, Any],
    receipt_metadata: Mapping[str, Any],
    preregistration_text: str,
    resolved_configs: Mapping[str, Any],
    context: FormalAuthorizationContext | None,
    replay: FormalAuthorizationReplaySession | None,
) -> None:
    bundle_payload = bundle_metadata.get("formal_authorization")
    receipt_payload = receipt_metadata.get("formal_authorization")
    if issuance_status == "local_unissued":
        if (
            bundle_payload is not None
            or receipt_payload is not None
            or context is not None
            or replay is not None
        ):
            raise ValueError("local_unissued evidence cannot carry formal authorization")
        return
    if bundle_payload is None or receipt_payload is None:
        raise ValueError("formal evidence is missing its authorization summary")
    try:
        summary = FormalAuthorizationSummary.model_validate(bundle_payload)
        receipt_summary = FormalAuthorizationSummary.model_validate(receipt_payload)
    except ValueError as exc:
        raise ValueError("formal evidence has an invalid authorization summary") from exc
    if receipt_summary != summary:
        raise ValueError("receipt and RunBundle authorization summaries differ")
    if context is not None and replay is not None:
        raise ValueError("formal evidence cannot use two authorization replay contexts")
    if context is None and replay is None:
        raise ValueError(
            "formal evidence verification requires a replayable authorization context"
        )
    source_identity = _resolved_source_identity(resolved_configs)
    try:
        if context is not None:
            verify_formal_authorization(
                context,
                preregistration_text=preregistration_text,
                live_source_identity=source_identity,
                expected_summary=summary,
            )
        else:
            assert replay is not None
            replay.verify(
                summary,
                preregistration_text=preregistration_text,
                live_source_identity=source_identity,
            )
    except FormalAuthorizationError as exc:
        raise ValueError("formal authorization canonical replay failed") from exc


def _validate_compiler_manifest(
    compiler_manifest: Any,
    *,
    spec: FitExperimentSpec,
    expected_hash: str,
) -> None:
    manifest = _require_mapping(compiler_manifest, name="compiler manifest")
    schema = manifest.get("schema")
    if schema == CORPUS_COMPILER_BINDING_SCHEMA:
        if spec.stage is FitStage.F0:
            raise ValueError("F0 attempts require the historical single-episode compiler schema")
        validate_corpus_compiler_binding_manifest(
            manifest,
            expected_hash=expected_hash,
            contract_id=spec.contract_id,
            dataset_hash=spec.dataset.dataset_hash,
            typed_split_hash=spec.split.content_hash,
            typed_split_kind=spec.split.kind.value,
            fit_partition=spec.split.fit_partition,
            episode_schedule=spec.episode_schedule,
        )
        return
    if spec.stage is not FitStage.F0:
        raise ValueError("S1/R1 attempts require the multi-episode corpus compiler schema")
    expected_fields = {
        "schema",
        "typed_split_hash",
        "typed_split_kind",
        "fit_partition",
        "compiler_provenance",
        "compiler_provenance_hash",
        "numeric_normalizer",
        "projection",
    }
    if set(manifest) != expected_fields:
        raise ValueError("compiler manifest has an unexpected shape")
    if schema != "tabu.fit-compiler-binding.v1":
        raise ValueError("compiler manifest has an unsupported schema")
    if manifest.get("typed_split_hash") != spec.split.content_hash:
        raise ValueError("compiler manifest is not bound to the typed split")
    if manifest.get("typed_split_kind") != spec.split.kind.value:
        raise ValueError("compiler manifest split kind differs from preregistration")
    if manifest.get("fit_partition") != spec.split.fit_partition:
        raise ValueError("compiler manifest fit partition differs from preregistration")
    expected_projection = (
        "observed_interactions_to_full_matrix_row_carrier"
        if spec.contract_id == "tabu4rec"
        else "nodes_to_graph_row_carrier"
        if spec.contract_id == "tabu4graph"
        else "rows_to_tabular_row_carrier"
    )
    if manifest.get("projection") != expected_projection:
        raise ValueError("compiler manifest projection differs from the contract")

    provenance = _require_mapping(manifest.get("compiler_provenance"), name="compiler provenance")
    provenance_fields = {
        "dataset_hash",
        "split_manifest_hash",
        "source_view_hash",
        "fit_view_hash",
        "recipe_hash",
        "graph_topology_hash",
        "numeric_normalizer_hash",
    }
    if set(provenance) != provenance_fields:
        raise ValueError("compiler provenance has an unexpected shape")
    for field_name in provenance_fields - {"graph_topology_hash"}:
        field_hash = provenance.get(field_name)
        if not isinstance(field_hash, str):
            raise ValueError(f"compiler provenance has an invalid {field_name}")
        try:
            require_sha256(field_hash, field_name=field_name)
        except ValueError as exc:
            raise ValueError(f"compiler provenance has an invalid {field_name}") from exc
    graph_hash = provenance.get("graph_topology_hash")
    if graph_hash is not None:
        if not isinstance(graph_hash, str):
            raise ValueError("compiler provenance has an invalid graph_topology_hash")
        try:
            require_sha256(graph_hash, field_name="graph_topology_hash")
        except ValueError as exc:
            raise ValueError("compiler provenance has an invalid graph_topology_hash") from exc
    if provenance.get("dataset_hash") != spec.dataset.dataset_hash:
        raise ValueError("compiler provenance is not bound to the dataset")
    if spec.episode_schedule.recipe_hashes and provenance.get("recipe_hash") not in (
        spec.episode_schedule.recipe_hashes
    ):
        raise ValueError("compiler provenance recipe is not preregistered")
    normalizer = _require_mapping(
        manifest.get("numeric_normalizer"),
        name="numeric normalizer",
    )
    normalizer_fields = {
        "schema",
        "fit_view_hash",
        "split_definition_hash",
        "config_hash",
        "fit_value_mask_hash",
        "feature_names",
        "feature_kinds",
        "counts",
        "means",
        "scales",
        "epsilon",
        "shared_numeric_groups",
        "artifact_hash",
    }
    if set(normalizer) != normalizer_fields:
        raise ValueError("numeric normalizer manifest has an unexpected shape")
    if normalizer.get("schema") != "tabu.numeric-normalizer-binding.v1":
        raise ValueError("numeric normalizer manifest has an unsupported schema")
    for field_name in (
        "fit_view_hash",
        "split_definition_hash",
        "config_hash",
        "fit_value_mask_hash",
        "artifact_hash",
    ):
        value = normalizer.get(field_name)
        if not isinstance(value, str):
            raise ValueError(f"numeric normalizer has an invalid {field_name}")
        try:
            require_sha256(value, field_name=field_name)
        except ValueError as exc:
            raise ValueError(f"numeric normalizer has an invalid {field_name}") from exc
    if normalizer.get("fit_view_hash") != provenance.get("fit_view_hash"):
        raise ValueError("numeric normalizer is not bound to compiler fit_view_hash")
    if normalizer.get("artifact_hash") != provenance.get("numeric_normalizer_hash"):
        raise ValueError("compiler provenance is not bound to numeric normalizer")
    raw_feature_names = normalizer.get("feature_names")
    raw_feature_kinds = normalizer.get("feature_kinds")
    if (
        isinstance(raw_feature_names, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_feature_names, Sequence)
        or not raw_feature_names
    ):
        raise ValueError("numeric normalizer feature_names are invalid")
    feature_names = list(raw_feature_names)
    if (
        any(not isinstance(name, str) or not name for name in feature_names)
        or len(feature_names) != len(set(feature_names))
    ):
        raise ValueError("numeric normalizer feature_names are invalid")
    if (
        isinstance(raw_feature_kinds, (str, bytes, bytearray, Mapping))
        or not isinstance(raw_feature_kinds, Sequence)
    ):
        raise ValueError("numeric normalizer feature_kinds are invalid")
    feature_kinds = list(raw_feature_kinds)
    if len(feature_kinds) != len(feature_names) or any(
        kind not in {"numeric", "categorical", "ordinal"} for kind in feature_kinds
    ):
        raise ValueError("numeric normalizer feature_kinds are invalid")
    epsilon = normalizer.get("epsilon")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or epsilon <= 0:
        raise ValueError("numeric normalizer epsilon must be positive")
    groups = normalizer.get("shared_numeric_groups")
    if isinstance(groups, (str, bytes, bytearray, Mapping)) or not isinstance(
        groups, Sequence
    ):
        raise ValueError("numeric normalizer shared groups are invalid")
    normalized_groups: list[list[str]] = []
    grouped_names: list[str] = []
    for group in groups:
        if (
            isinstance(group, (str, bytes, bytearray, Mapping))
            or not isinstance(group, Sequence)
            or len(group) < 2
            or any(not isinstance(name, str) or name not in feature_names for name in group)
            or len(group) != len(set(group))
        ):
            raise ValueError("numeric normalizer shared groups are invalid")
        if any(feature_kinds[feature_names.index(name)] != "numeric" for name in group):
            raise ValueError("numeric normalizer groups may contain numeric features only")
        normalized_group = list(group)
        normalized_groups.append(normalized_group)
        grouped_names.extend(normalized_group)
    if len(grouped_names) != len(set(grouped_names)):
        raise ValueError("numeric normalizer shared groups must be disjoint")
    expected_config_hash = canonical_hash(
        {
            "kind": "numeric_normalizer",
            "epsilon": float(epsilon),
            "shared_numeric_groups": normalized_groups,
        }
    )
    if normalizer.get("config_hash") != expected_config_hash:
        raise ValueError("numeric normalizer config hash does not match its preimage")
    statistics_preimage = {
        "schema": "tabu.fitted-statistics.v2",
        "fit_view_hash": normalizer["fit_view_hash"],
        "split_definition_hash": normalizer["split_definition_hash"],
        "config_hash": normalizer["config_hash"],
        "fit_value_mask_hash": normalizer["fit_value_mask_hash"],
        "feature_names": feature_names,
        "feature_kinds": feature_kinds,
        "counts": normalizer["counts"],
        "means": normalizer["means"],
        "scales": normalizer["scales"],
    }
    if canonical_hash(statistics_preimage) != normalizer.get("artifact_hash"):
        raise ValueError("numeric normalizer artifact hash does not match statistics")
    expected_provenance_hash = canonical_hash(
        {"schema": "tabu.compilation-provenance.v2", **dict(provenance)}
    )
    if manifest.get("compiler_provenance_hash") != expected_provenance_hash:
        raise ValueError("compiler provenance hash does not match its preimage")
    if canonical_hash(manifest) != expected_hash:
        raise ValueError("compiler manifest does not match RunIdentity compiler_hash")


def _validate_formal_fit_bindings(
    *,
    preregistration_text: str,
    resolved_configs: Mapping[str, Any],
    dataset_manifest: Any,
    split_manifest: Any,
    compiler_manifest: Any,
    run_identity: RunIdentity,
    model_id: str,
    dataset_id: str,
    fit_partition: str,
    metadata: Mapping[str, Any],
    environment_payload: Mapping[str, Any],
    episode_recipe_hashes: Sequence[str] | None = None,
) -> FitExperimentSpec:
    """Recompute every formal fit identity preimage and bind it across files."""

    spec = _parse_formal_preregistration(preregistration_text)
    if set(resolved_configs) != _FORMAL_RESOLVED_CONFIGS:
        raise ValueError(
            "formal resolved configs must contain exactly code, experiment, semantic, "
            "training, execution, and seeds"
        )
    resolved_experiment_payload = _require_mapping(
        to_canonical_data(resolved_configs["experiment"]),
        name="resolved experiment",
    )
    try:
        resolved_experiment = FitExperimentSpec.model_validate(resolved_experiment_payload)
    except ValueError as exc:
        raise ValueError("resolved experiment is not a valid FitExperimentSpec") from exc
    if resolved_experiment != spec:
        raise ValueError("resolved experiment does not match preregistration")
    # Hash the immutable resolved experiment artifact itself.  Re-materializing
    # an older preregistration through a newer Pydantic model can add a new
    # defaulted field and must not invalidate the historical receipt preimage.
    if run_identity.spec_hash != canonical_hash(resolved_experiment_payload):
        raise ValueError("preregistration does not match RunIdentity spec_hash")

    resolved_semantic_payload = _require_mapping(
        to_canonical_data(resolved_configs["semantic"]),
        name="resolved semantic config",
    )
    try:
        resolved_semantic = type(spec.semantic).model_validate(resolved_semantic_payload)
    except ValueError as exc:
        raise ValueError("resolved semantic config is invalid") from exc
    if resolved_semantic != spec.semantic:
        raise ValueError("resolved semantic config does not match preregistration")
    experiment_semantic = _require_mapping(
        resolved_experiment_payload.get("semantic"),
        name="resolved experiment semantic config",
    )
    if to_canonical_data(resolved_semantic_payload) != to_canonical_data(experiment_semantic):
        raise ValueError("resolved semantic config does not match resolved experiment")
    if canonical_hash(resolved_semantic_payload) != run_identity.semantic_config_hash:
        raise ValueError("resolved semantic config does not match RunIdentity")

    training = _require_mapping(
        to_canonical_data(resolved_configs["training"]),
        name="resolved training config",
    )
    expected_training = _require_mapping(
        resolved_experiment_payload.get("training"),
        name="resolved experiment training config",
    )
    if to_canonical_data(training) != to_canonical_data(expected_training):
        raise ValueError("resolved training config does not match preregistration")
    if canonical_hash(training) != run_identity.training_config_hash:
        raise ValueError("resolved training config does not match RunIdentity")

    execution = _require_mapping(
        to_canonical_data(resolved_configs["execution"]),
        name="resolved execution config",
    )
    preregistered_execution = _require_mapping(
        resolved_experiment_payload.get("execution"),
        name="resolved experiment execution config",
    )
    if set(execution) != set(preregistered_execution) | _FORMAL_EXECUTION_RUNTIME_FIELDS:
        raise ValueError("resolved execution config has an unexpected shape")
    for name, expected in preregistered_execution.items():
        if execution.get(name) != expected:
            raise ValueError("resolved execution config does not match preregistration")
    expected_device = (
        f"cuda:{spec.execution.device_index}"
        if spec.execution.device.value == "cuda"
        else spec.execution.device.value
    )
    _validate_environment_payload(environment_payload)
    if (
        environment_payload.get("device") != expected_device
        or environment_payload.get("deterministic_algorithms")
        is not spec.execution.deterministic_algorithms
    ):
        raise ValueError("captured environment does not match preregistered execution")
    expected_runtime = {
        "resolved_device": expected_device,
        "environment_hash": canonical_hash(environment_payload),
        "host_class": environment_payload.get("host_class"),
        "torch_version": environment_payload.get("torch_version"),
        "python_version": environment_payload.get("python_version"),
    }
    if any(execution.get(name) != expected for name, expected in expected_runtime.items()):
        raise ValueError("resolved execution config does not match captured environment")
    if canonical_hash(execution) != run_identity.execution_config_hash:
        raise ValueError("resolved execution config does not match RunIdentity")

    model_seed = metadata.get("model_seed")
    if type(model_seed) is not int or model_seed not in spec.seeds.model_seeds:
        raise ValueError("formal metadata model_seed is not preregistered")
    expected_seeds = {
        "episode": spec.seeds.data_seed,
        "model_init": model_seed,
        "numpy": model_seed,
        "python": model_seed,
        "sampler": spec.seeds.episode_order_seed,
        "torch_cpu": model_seed,
    }
    if spec.execution.device.value == "cuda":
        expected_seeds["torch_cuda"] = model_seed
    elif spec.execution.device.value == "mps":
        expected_seeds["torch_mps"] = model_seed
    seeds = _require_mapping(resolved_configs["seeds"], name="resolved seed map")
    if dict(seeds) != expected_seeds or dict(run_identity.seeds) != expected_seeds:
        raise ValueError("resolved seeds do not match preregistration and RunIdentity")

    issuance_status = metadata.get("issuance_status")
    if issuance_status not in {"formal", "local_unissued"}:
        raise ValueError("fit metadata requires an explicit issuance_status")
    source_identity = _validate_code_manifest(
        resolved_configs["code"],
        expected_hash=run_identity.code_hash,
        preregistration_text=preregistration_text,
        expected_issuance_status=issuance_status,
    )
    if metadata.get("source_identity_hash") != canonical_hash(source_identity):
        raise ValueError("fit metadata source_identity_hash does not match SourceIdentity")
    if metadata.get("code_hash") != run_identity.code_hash:
        raise ValueError("formal metadata code_hash does not match RunIdentity")
    if metadata.get("experiment_id") != spec.experiment_id:
        raise ValueError("formal metadata experiment_id does not match preregistration")
    if metadata.get("stage") != spec.stage.value:
        raise ValueError("formal metadata stage does not match preregistration")
    expected_variant_role = (
        "canonical"
        if spec.semantic.dynamics.block_kind is DynamicsBlockKind.OMAB
        else "non_o_ablation"
    )
    for key, expected in (
        ("block_kind", spec.semantic.dynamics.block_kind.value),
        ("variant_role", expected_variant_role),
        ("numeric_terminal", spec.semantic.numeric_terminal.value),
    ):
        # Historical receipts may predate these metadata keys; new artifacts
        # emitted from a resolved semantic config always carry them.
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"formal metadata {key} does not match preregistration")
    if spec.contract_id == "tabu.cell.base":
        expected_reference = {
            key: getattr(value, "value", value)
            for key, value in spec.semantic.reference.model_dump(mode="python").items()
            if key != "backend"
        }
        expected_base_metadata = {
            "profile_id": spec.semantic.profile_id,
            "tokenizer_version": "cell-tokenizer.v1",
            "label_broadcast": spec.semantic.profile_id == "supervised.label_broadcast.v1",
            "label_broadcast_tau": 1.0e-6,
            "reference_config": {
                **expected_reference,
                "block_kind": spec.semantic.dynamics.block_kind.value,
            },
            "terminal": spec.semantic.numeric_terminal.value,
            "bandwidth": spec.semantic.reference.routing_bandwidth,
        }
        for key, expected in expected_base_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"formal metadata {key} does not match TabUBase identity")
        variant_ref = metadata.get("variant_ref")
        if not isinstance(variant_ref, Mapping):
            raise ValueError("formal TabUBase metadata requires a variant_ref mapping")
        expected_variant = {
            "contract_id": spec.contract_id,
            "contract_version": spec.contract_version,
            "profile_id": spec.semantic.profile_id,
            "model_spec_hash": spec.model_spec_hash,
            "source_identity": spec.dataset.source_sha256,
            "semantic_config_hash": spec.semantic.content_hash,
        }
        if dict(variant_ref) != expected_variant:
            raise ValueError("formal TabUBase variant_ref does not match preregistration")
        if not isinstance(metadata.get("variant_hash"), str) or len(metadata["variant_hash"]) != 64:
            raise ValueError("formal TabUBase metadata requires a variant_hash")

    dataset = _require_mapping(dataset_manifest, name="dataset manifest")
    expected_dataset_fields = {
        "schema",
        "dataset",
        "dataset_id",
        "dataset_hash",
        "feature_specs",
        "row_ids",
        "metadata",
    }
    if set(dataset) != expected_dataset_fields:
        raise ValueError("dataset manifest has an unexpected shape")
    if dataset.get("schema") != "tabu.fit-dataset-manifest.v1":
        raise ValueError("dataset manifest has an unsupported schema")
    try:
        resolved_dataset = type(spec.dataset).model_validate(dataset.get("dataset"))
    except ValueError as exc:
        raise ValueError("dataset manifest embeds an invalid dataset spec") from exc
    if resolved_dataset != spec.dataset:
        raise ValueError("dataset manifest does not match preregistration")
    if (
        dataset.get("dataset_id") != spec.dataset.dataset_id
        or dataset.get("dataset_hash") != spec.dataset.dataset_hash
        or dataset_id != spec.dataset.dataset_id
        or run_identity.data_hash != spec.dataset.dataset_hash
    ):
        raise ValueError("dataset manifest does not match RunIdentity and run dataset")

    try:
        resolved_split = type(spec.split).model_validate(split_manifest)
    except ValueError as exc:
        raise ValueError("split manifest is not the preregistered typed split") from exc
    if resolved_split != spec.split:
        raise ValueError("split manifest does not match preregistration")
    if canonical_hash(resolved_split) != run_identity.split_hash:
        raise ValueError("split manifest does not match RunIdentity split_hash")
    if fit_partition != spec.split.fit_partition:
        raise ValueError("run fit partition does not match preregistration")
    _validate_compiler_manifest(
        compiler_manifest,
        spec=spec,
        expected_hash=run_identity.compiler_hash,
    )
    if spec.stage is not FitStage.F0 and episode_recipe_hashes is not None:
        expected_recipe_hashes = corpus_compiler_episode_recipe_hashes(compiler_manifest)
        if tuple(episode_recipe_hashes) != expected_recipe_hashes:
            raise ValueError(
                "RunBundle episode recipe hashes differ from the corpus schedule realization"
            )
    if model_id != spec.contract_id:
        raise ValueError("run model_id does not match preregistered contract_id")
    return spec


def _safe_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must match [A-Za-z0-9][A-Za-z0-9_.-]*")
    return value


def _safe_artifact_path(root: Path, relative: str) -> Path:
    """Resolve one portable POSIX artifact URI without traversal or symlinks."""

    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("artifact paths must be non-empty portable POSIX paths")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative:
        raise ValueError(f"unsafe or non-canonical artifact path: {relative}")
    if any(part in {"", ".", ".."} or not _SAFE_IDENTIFIER.fullmatch(part) for part in pure.parts):
        raise ValueError(f"unsafe or non-portable artifact path: {relative}")
    candidate = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlinked fit artifact is forbidden: {relative}")
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"artifact path escapes or is missing: {relative}") from exc
    if not candidate.is_file():
        raise ValueError(f"fit artifact is not a regular file: {relative}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_ref(root: Path, path: Path, *, kind: str) -> ArtifactRef:
    relative = path.relative_to(root).as_posix()
    path = _safe_artifact_path(root, relative)
    return ArtifactRef(
        artifact_id=relative.replace("/", "--"),
        kind=kind,
        uri=relative,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        media_type=(
            "application/json"
            if path.suffix == ".json"
            else "application/x-ndjson"
            if path.suffix == ".jsonl"
            else "application/yaml"
            if path.suffix in {".yaml", ".yml"}
            else "text/markdown"
            if path.suffix == ".md"
            else "application/octet-stream"
        ),
    )


def _dependency_lock_hash() -> str | None:
    repository = Path(__file__).resolve().parents[3]
    lock = repository / "uv.lock"
    return _sha256(lock) if lock.is_file() else None


def _validate_environment_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != _ENVIRONMENT_FIELDS:
        raise ValueError("environment disclosure has an unexpected public shape")
    if payload.get("schema_version") != "tabu.fit-environment.v2":
        raise ValueError("environment disclosure has an unsupported schema")
    if payload.get("host_class") not in {"cpu-host", "cuda-host", "mps-host"}:
        raise ValueError("environment disclosure has an invalid generalized host class")
    for name in ("os_family", "architecture", "python_version", "torch_version", "device"):
        if not isinstance(payload.get(name), str) or not payload[name]:
            raise ValueError(f"environment disclosure has an invalid {name}")
    for name in ("dependency_lock_hash", "container_image_hash"):
        value = payload.get(name)
        if value is not None:
            require_sha256(value, field_name=name)
    if type(payload.get("deterministic_algorithms")) is not bool:
        raise ValueError("environment deterministic_algorithms must be boolean")


def assert_public_payload_safe(payload: Any, *, location: str = "public evidence") -> None:
    """Reject common secret and private-machine shapes from formal evidence."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if is_sensitive_public_key(key):
                raise ValueError(f"{location} contains forbidden private field {key!r}")
            assert_public_payload_safe(value, location=f"{location}.{key}")
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for index, value in enumerate(payload):
            assert_public_payload_safe(value, location=f"{location}[{index}]")
        return
    if isinstance(payload, str):
        if contains_absolute_local_path(payload):
            raise ValueError(f"{location} contains a private absolute path")
        if contains_private_identity_or_secret(payload):
            raise ValueError(f"{location} contains a private identity or secret")


def assert_public_artifact_tree_safe(root: Path, *, location: str) -> None:
    """Scan every public text payload, not only the receipt summary objects.

    Checkpoints remain binary and are covered by their digest.  Every JSON,
    JSONL, YAML, and Markdown file is decoded and traversed so a private path or
    secret-shaped field cannot hide in metrics, traces, baselines, feasibility,
    verdicts, or a newly added formal artifact.
    """

    supported = frozenset({".json", ".jsonl", ".yaml", ".yml", ".md"})
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix.lower() == ".jsonl":
                payload = tuple(
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                )
            elif path.suffix.lower() in {".yaml", ".yml"}:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            else:
                payload = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ValueError(
                f"{location} contains an unreadable text artifact: {relative}"
            ) from exc
        assert_public_payload_safe(payload, location=f"{location}.{relative}")


def capture_environment(device: str) -> tuple[EnvironmentDisclosure, dict[str, Any]]:
    """Capture the hash preimage and validated environment disclosure."""

    resolved = torch.device(device)
    accelerator: str | None = None
    if resolved.type == "cuda" and torch.cuda.is_available():
        accelerator = torch.cuda.get_device_name(resolved)
    elif resolved.type == "mps" and torch.backends.mps.is_available():
        accelerator = "Apple Metal Performance Shaders"
    host_class = {
        "cpu": "cpu-host",
        "cuda": "cuda-host",
        "mps": "mps-host",
    }.get(resolved.type, "cpu-host")
    container_image_hash = os.environ.get("TABU_LAB_CONTAINER_IMAGE_HASH")
    if container_image_hash is not None:
        require_sha256(container_image_hash, field_name="TABU_LAB_CONTAINER_IMAGE_HASH")
    payload: dict[str, Any] = {
        "schema_version": "tabu.fit-environment.v2",
        "host_class": host_class,
        "os_family": platform.system(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "device": str(resolved),
        "accelerator": accelerator,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dependency_lock_hash": _dependency_lock_hash(),
        "container_image_hash": container_image_hash,
    }
    _validate_environment_payload(payload)
    assert_public_payload_safe(payload, location="environment disclosure")
    disclosure = EnvironmentDisclosure(
        environment_hash=canonical_hash(payload),
        host_class=host_class,
        operating_system=platform.system(),
        device=str(resolved),
        architecture=platform.machine(),
        accelerator=accelerator,
        python_version=platform.python_version(),
    )
    return disclosure, payload


def _publish_directory_create_once(staging: Path, destination: Path) -> None:
    """Serialize cooperative publishers and never intentionally replace a target.

    Python does not expose POSIX ``renameat2(RENAME_NOREPLACE)`` portably.  An
    exclusive same-parent lock closes the race between this package's writers;
    the final existence check also fails closed if any external writer created
    the destination while the bundle was staged.  ``os.rename`` is atomic
    within the parent filesystem and refuses an existing destination on
    Windows.  On POSIX, the lock prevents our writers from exercising rename's
    otherwise replacement-capable behavior.
    """

    lock = destination.parent / f".{destination.name}.publish.lock"
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                lock,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"another publisher owns the fit attempt path: {destination}"
            ) from exc
        if os.path.lexists(destination):
            raise FileExistsError(f"immutable fit attempt already exists: {destination}")
        os.rename(staging, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                lock.unlink()


def write_fit_attempt_artifacts(
    destination: str | os.PathLike[str],
    *,
    attempt_id: str,
    run_identity: RunIdentity,
    model_id: str,
    dataset_id: str,
    fit_partition: str,
    preregistration_text: str,
    resolved_configs: Mapping[str, Any],
    dataset_manifest: Any,
    split_manifest: Any,
    compiler_manifest: Any,
    feasibility: Any,
    metrics: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    evaluation: EvaluationBundle,
    predictions: Sequence[PredictionBundle],
    episode_recipe_hashes: Sequence[str] = (),
    baselines: Mapping[str, Any],
    verdict: str,
    status: ReceiptStatus,
    command: Sequence[str],
    checkpoint_writer: Callable[[Path], Path] | None = None,
    error: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    formal_authorization: FormalAuthorizationContext | None = None,
) -> FitAttemptArtifacts:
    """Atomically write one immutable success or failure attempt.

    The output directory is create-once.  Receipt and checksum verification is
    performed before the temporary directory is atomically published.  Recipe
    identities are supplied by the experiment/compiler boundary rather than
    inferred from model-facing prediction metadata; the empty default preserves
    compatibility with historical single-episode F0 callers.
    """

    attempt_id = _safe_identifier(attempt_id, field_name="attempt_id")
    if not isinstance(run_identity, RunIdentity):
        raise TypeError("run_identity must be a validated RunIdentity")
    if not isinstance(evaluation, EvaluationBundle):
        raise TypeError("evaluation must be a validated EvaluationBundle")
    prediction_items = tuple(predictions)
    if not prediction_items or not all(
        isinstance(item, PredictionBundle) for item in prediction_items
    ):
        raise TypeError("predictions must be a non-empty PredictionBundle sequence")
    if any(item.model_id != model_id for item in prediction_items):
        raise ValueError("every prediction model_id must match the run model_id")
    if isinstance(episode_recipe_hashes, (str, bytes, bytearray, Mapping)):
        raise TypeError("episode_recipe_hashes must be a sequence of SHA-256 strings")
    recipe_hashes = tuple(
        require_sha256(value, field_name="episode_recipe_hash")
        for value in episode_recipe_hashes
    )
    if len(recipe_hashes) != len(set(recipe_hashes)):
        raise ValueError("episode_recipe_hashes must be unique")
    if not isinstance(status, ReceiptStatus):
        raise TypeError("status must be a typed ReceiptStatus")
    if status not in {ReceiptStatus.SUCCEEDED, ReceiptStatus.FAILED}:
        raise ValueError("fit attempt receipts must be succeeded or failed")
    if status is ReceiptStatus.SUCCEEDED and checkpoint_writer is None:
        raise ValueError("succeeded fit attempts require a checkpoint_writer")
    if status is ReceiptStatus.FAILED and not error:
        raise ValueError("failed fit attempts require an error boundary")
    if status is not ReceiptStatus.FAILED and error is not None:
        raise ValueError("error is only valid for failed fit attempts")
    if isinstance(command, (str, bytes, bytearray)) or any(
        not isinstance(item, str) or not item for item in command
    ):
        raise TypeError("command must be a sequence of non-empty strings")
    resolved_command = tuple(command)
    metric_rows, fit_evaluation = _normalize_metrics(metrics)
    if not verdict.strip():
        raise ValueError("verdict cannot be empty")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    run_metadata = dict(metadata or {})
    issuance_status = run_metadata.setdefault("issuance_status", "local_unissued")
    if issuance_status not in {"formal", "local_unissued"}:
        raise ValueError("issuance_status must be formal or local_unissued")
    if issuance_status == "local_unissued" and (
        formal_authorization is not None or "formal_authorization" in run_metadata
    ):
        raise ValueError("local_unissued attempts cannot carry formal authorization")
    if issuance_status == "formal" and "experiment_id" not in run_metadata:
        raise ValueError("formal attempts require complete experiment bindings")

    # Bind the selected dynamics variant into both RunBundle and Receipt even
    # when a lower-level caller did not duplicate it in ``metadata``.  Legacy
    # semantic payloads are parsed as OMAB but retain their original hash and
    # payload shape through ModelSemanticConfig's compatibility serializer.
    semantic_payload = resolved_configs.get("semantic")
    try:
        semantic_config = (
            semantic_payload
            if isinstance(semantic_payload, ModelSemanticConfig)
            else ModelSemanticConfig.model_validate(semantic_payload)
        )
    except (TypeError, ValueError):
        semantic_config = None
    if semantic_config is not None:
        block_kind = semantic_config.dynamics.block_kind
        numeric_terminal = semantic_config.numeric_terminal
        if "block_kind" in run_metadata and run_metadata["block_kind"] != block_kind.value:
            raise ValueError("fit metadata block_kind differs from semantic config")
        if (
            "numeric_terminal" in run_metadata
            and run_metadata["numeric_terminal"] != numeric_terminal.value
        ):
            raise ValueError("fit metadata numeric_terminal differs from semantic config")
        expected_variant_role = (
            "canonical" if block_kind is DynamicsBlockKind.OMAB else "non_o_ablation"
        )
        if (
            "variant_role" in run_metadata
            and run_metadata["variant_role"] != expected_variant_role
        ):
            raise ValueError("fit metadata variant_role differs from semantic config")
        run_metadata.setdefault("block_kind", block_kind.value)
        run_metadata.setdefault("variant_role", expected_variant_role)
        run_metadata.setdefault("numeric_terminal", numeric_terminal.value)

    output = Path(destination)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable fit attempt already exists: {output}")
    execution_config = resolved_configs.get("execution", {})
    if not isinstance(execution_config, Mapping):
        raise TypeError("resolved execution config must be a mapping")
    execution_device = str(execution_config.get("device", "cpu"))
    device_index = execution_config.get("device_index")
    if execution_device == "cuda" and device_index is not None:
        if type(device_index) is not int or device_index < 0:
            raise ValueError("CUDA device_index must be a non-negative integer")
        execution_device = f"cuda:{device_index}"
    environment, environment_payload = capture_environment(execution_device)
    formal_spec: FitExperimentSpec | None = None
    if "experiment_id" in run_metadata:
        formal_spec = _validate_formal_fit_bindings(
            preregistration_text=preregistration_text,
            resolved_configs=resolved_configs,
            dataset_manifest=dataset_manifest,
            split_manifest=split_manifest,
            compiler_manifest=compiler_manifest,
            run_identity=run_identity,
            model_id=model_id,
            dataset_id=dataset_id,
            fit_partition=fit_partition,
            metadata=run_metadata,
            environment_payload=environment_payload,
            episode_recipe_hashes=recipe_hashes,
        )
        declared_contract_version = run_metadata.setdefault(
            "contract_version", formal_spec.contract_version
        )
        if declared_contract_version != formal_spec.contract_version:
            raise ValueError("fit metadata contract_version differs from preregistration")
        checkpoint_license_id = run_metadata.setdefault(
            "checkpoint_license_id", "Apache-2.0"
        )
        if not isinstance(checkpoint_license_id, str) or not checkpoint_license_id.strip():
            raise ValueError("fit metadata checkpoint_license_id must be non-empty")
        if fit_evaluation is None:
            raise ValueError("formal fit attempts require a typed FitEvaluationBundle")
        if (
            fit_evaluation.experiment_id != formal_spec.experiment_id
            or fit_evaluation.stage is not formal_spec.stage
            or fit_evaluation.model_seed != run_metadata.get("model_seed")
        ):
            raise ValueError("FitEvaluationBundle does not match formal fit metadata")
        if issuance_status == "formal":
            if formal_authorization is None:
                raise ValueError(
                    "formal attempts require replayable canonical authorization context"
                )
            source_identity = _resolved_source_identity(resolved_configs)
            try:
                verified_authorization = verify_formal_authorization(
                    formal_authorization,
                    preregistration_text=preregistration_text,
                    live_source_identity=source_identity,
                    expected_summary=run_metadata.get("formal_authorization"),
                )
            except FormalAuthorizationError as exc:
                raise ValueError("formal authorization replay failed") from exc
            run_metadata["formal_authorization"] = (
                verified_authorization.summary.model_dump(mode="json")
            )
            assert_public_payload_safe(
                {
                    "command": resolved_command,
                    "metadata": run_metadata,
                    "resolved_configs": resolved_configs,
                    "dataset_manifest": dataset_manifest,
                    "split_manifest": split_manifest,
                    "compiler_manifest": compiler_manifest,
                    "environment": environment_payload,
                },
                location="formal fit evidence",
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    checkpoint_path: Path | None = None
    try:
        preregistration = temporary / "preregistration.yaml"
        preregistration.write_text(preregistration_text, encoding="utf-8")
        if not preregistration_text.endswith("\n"):
            with preregistration.open("a", encoding="utf-8") as target:
                target.write("\n")

        resolved_directory = temporary / "resolved-configs"
        resolved_directory.mkdir(parents=True, exist_ok=False)
        for name, payload in sorted(resolved_configs.items()):
            name = _safe_identifier(name, field_name="resolved config name")
            _write_json(resolved_directory / f"{name}.json", payload)
        _write_json(temporary / "dataset-manifest.json", dataset_manifest)
        _write_json(temporary / "split-manifest.json", split_manifest)
        _write_json(temporary / "compiler-manifest.json", compiler_manifest)
        _write_json(temporary / "feasibility.json", feasibility)
        _write_jsonl(temporary / "metrics.jsonl", metric_rows)
        _write_json(temporary / "evaluation.json", evaluation)
        _write_json(
            temporary / "forward-traces.json",
            {
                "schema": "tabu.fit-forward-traces.v1",
                "predictions": tuple(
                    {
                        "episode_id": prediction.episode_id,
                        "prediction_hash": prediction.prediction_hash,
                        "trace": prediction.trace,
                    }
                    for prediction in prediction_items
                ),
            },
        )
        _write_json(temporary / "baselines.json", baselines)
        _write_json(temporary / "environment.json", environment_payload)
        (temporary / "verdict.md").write_text(verdict.rstrip() + "\n", encoding="utf-8")

        if checkpoint_writer is not None:
            checkpoint_path = temporary / "checkpoint" / "checkpoint.safetensors"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            written = checkpoint_writer(checkpoint_path)
            if Path(written) != checkpoint_path or not checkpoint_path.is_file():
                raise ValueError("checkpoint_writer did not publish the requested checkpoint")

        payload_paths = tuple(
            path
            for path in sorted(
                temporary.rglob("*"),
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
            if path.is_file() and path.name not in {"receipt.json", "artifacts.sha256"}
        )
        artifact_refs = tuple(
            _artifact_ref(
                temporary,
                path,
                kind=(
                    "checkpoint"
                    if path.suffix == ".safetensors"
                    else "forward_trace"
                    if path.name == "forward-traces.json"
                    else "fit_artifact"
                ),
            )
            for path in payload_paths
        )
        run_bundle = RunBundle(
            identity=run_identity,
            model_id=model_id,
            dataset_id=dataset_id,
            fit_partition=fit_partition,
            environment=environment,
            episode_recipe_hashes=recipe_hashes,
            artifacts=artifact_refs,
            metadata={
                **run_metadata,
                "attempt_id": attempt_id,
                "claim_boundary": "fit_attempt_only_no_accepted_claim",
            },
        )
        run_bundle_path = temporary / "run_bundle.json"
        _write_json(run_bundle_path, run_bundle)
        _write_json(
            temporary / "run_manifest.json",
            {
                "schema": "tabu.fit-run-manifest.v1",
                "attempt_id": attempt_id,
                "run_id": run_identity.run_id,
                "run_identity": run_identity,
                "run_bundle_hash": run_bundle.run_bundle_hash,
                "evaluation_hash": evaluation.evaluation_hash,
                "fit_evaluation_hash": (
                    None if fit_evaluation is None else fit_evaluation.evaluation_hash
                ),
                "prediction_hashes": tuple(item.prediction_hash for item in prediction_items),
                "status": status,
                "issuance_status": issuance_status,
                "source_identity_hash": run_metadata.get("source_identity_hash"),
                "claim_boundary": "fit_attempt_only_no_accepted_claim",
            },
        )
        completed_at = datetime.now(UTC)
        receipt_metadata: dict[str, Any] = {
            "attempt_id": attempt_id,
            "verdict": "failed" if status is ReceiptStatus.FAILED else "completed",
            "issuance_status": issuance_status,
            "source_identity_hash": run_metadata.get("source_identity_hash"),
        }
        for key in (
            "block_kind",
            "variant_role",
            "numeric_terminal",
            "profile_id",
            "tokenizer_version",
            "label_broadcast",
            "label_broadcast_tau",
            "reference_config",
            "terminal",
            "ll_ridge",
            "bandwidth",
            "variant_ref",
            "variant_hash",
        ):
            if key in run_metadata:
                receipt_metadata[key] = run_metadata[key]
        if "formal_authorization" in run_metadata:
            receipt_metadata["formal_authorization"] = run_metadata[
                "formal_authorization"
            ]
        receipt = Receipt.from_run_bundle(
            run_bundle,
            receipt_id=f"receipt-{attempt_id}",
            status=status,
            created_at=completed_at,
            completed_at=completed_at,
            command=resolved_command,
            evaluation_hashes=(evaluation.evaluation_hash,),
            artifacts=artifact_refs,
            error=error,
            metadata=receipt_metadata,
        )
        receipt_path = temporary / "receipt.json"
        write_receipt(receipt_path, receipt)

        if issuance_status == "formal":
            assert_public_artifact_tree_safe(
                temporary,
                location="formal fit evidence",
            )

        checksum_paths = tuple(
            path
            for path in sorted(
                temporary.rglob("*"),
                key=lambda item: item.relative_to(temporary).as_posix(),
            )
            if path.is_file() and path.name != "artifacts.sha256"
        )
        checksums_path = temporary / "artifacts.sha256"
        checksums_path.write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        verified_receipt = verify_fit_attempt_artifacts(
            temporary,
            formal_authorization=formal_authorization,
        )
        if verified_receipt.receipt_hash != receipt.receipt_hash:
            raise ValueError("pre-publish receipt verification returned a different hash")
        checkpoint_relative = (
            None if checkpoint_path is None else checkpoint_path.relative_to(temporary)
        )
        _publish_directory_create_once(temporary, output)
        result = FitAttemptArtifacts(
            directory=output,
            receipt=output / "receipt.json",
            checksums=output / "artifacts.sha256",
            run_bundle=output / "run_bundle.json",
            checkpoint=(None if checkpoint_relative is None else output / checkpoint_relative),
            receipt_hash=receipt.receipt_hash,
        )
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _verify_runtime_failure_artifacts(
    root: Path,
    *,
    actual: set[str],
    listed: Mapping[str, str],
    run_manifest: Mapping[str, Any],
    formal_authorization: FormalAuthorizationContext | None,
    formal_authorization_replay: FormalAuthorizationReplaySession | None,
) -> Receipt:
    """Verify a failed attempt without inventing predictions or fit metrics."""

    required = _REQUIRED_RUNTIME_FAILURE_FILES | {
        "receipt.json",
        "run_bundle.json",
        "run_manifest.json",
    }
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"runtime failure attempt is missing required artifacts: {missing}")
    forbidden = _REQUIRED_PAYLOAD_FILES - _REQUIRED_RUNTIME_FAILURE_FILES
    unexpected = sorted(forbidden & actual)
    if unexpected:
        raise ValueError(
            "runtime failure attempt cannot fabricate successful-fit payloads: "
            f"{unexpected}"
        )

    receipt = read_receipt(root / "receipt.json")
    try:
        bundle = RunBundle.model_validate(
            _read_compact_canonical_json(root / "run_bundle.json")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime failure attempt has an invalid RunBundle") from exc
    environment_payload = _read_compact_canonical_json(root / "environment.json")
    if not isinstance(environment_payload, Mapping):
        raise ValueError("environment.json must contain one object")
    _validate_environment_payload(environment_payload)
    if canonical_hash(environment_payload) != bundle.environment.environment_hash:
        raise ValueError("environment payload does not match RunBundle environment_hash")

    try:
        preregistration_text = (root / "preregistration.yaml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("runtime failure preregistration is not valid UTF-8") from exc
    resolved_configs: dict[str, Any] = {}
    for path in (root / "resolved-configs").iterdir():
        if not path.is_file() or path.suffix != ".json":
            raise ValueError("runtime failure resolved-configs may contain only JSON files")
        name = _safe_identifier(path.stem, field_name="resolved config name")
        if name in resolved_configs:
            raise ValueError("duplicate runtime failure resolved config name")
        resolved_configs[name] = _read_compact_canonical_json(path)
    spec = _validate_formal_fit_bindings(
        preregistration_text=preregistration_text,
        resolved_configs=resolved_configs,
        dataset_manifest=_read_compact_canonical_json(root / "dataset-manifest.json"),
        split_manifest=_read_compact_canonical_json(root / "split-manifest.json"),
        compiler_manifest=_read_compact_canonical_json(root / "compiler-manifest.json"),
        run_identity=bundle.identity,
        model_id=bundle.model_id,
        dataset_id=bundle.dataset_id,
        fit_partition=bundle.fit_partition,
        metadata=bundle.metadata,
        environment_payload=environment_payload,
        episode_recipe_hashes=bundle.episode_recipe_hashes,
    )

    if receipt.status is not ReceiptStatus.FAILED or not receipt.error:
        raise ValueError("runtime failure receipt must have failed status and an error boundary")
    if receipt.evaluation_hashes:
        raise ValueError("runtime failure receipt cannot claim evaluation hashes")
    if (
        receipt.run_id != bundle.run_id
        or receipt.run_identity_hash != bundle.identity.identity_hash
        or receipt.run_bundle_hash != bundle.run_bundle_hash
        or receipt.artifacts != bundle.artifacts
    ):
        raise ValueError("runtime failure receipt and RunBundle bindings differ")
    issuance_status = bundle.metadata.get("issuance_status")
    if issuance_status not in {"formal", "local_unissued"}:
        raise ValueError("RunBundle has an invalid issuance_status")
    if receipt.metadata.get("issuance_status") != issuance_status:
        raise ValueError("receipt and RunBundle issuance statuses differ")
    if receipt.metadata.get("source_identity_hash") != bundle.metadata.get(
        "source_identity_hash"
    ):
        raise ValueError("receipt and RunBundle SourceIdentity hashes differ")
    for key in (
        "block_kind",
        "variant_role",
        "numeric_terminal",
        "profile_id",
        "tokenizer_version",
        "label_broadcast",
        "label_broadcast_tau",
        "reference_config",
        "terminal",
        "ll_ridge",
        "bandwidth",
        "variant_ref",
        "variant_hash",
    ):
        if receipt.metadata.get(key) != bundle.metadata.get(key):
            raise ValueError(f"receipt and RunBundle {key} metadata differ")
    _verify_recorded_formal_authorization(
        issuance_status=issuance_status,
        bundle_metadata=bundle.metadata,
        receipt_metadata=receipt.metadata,
        preregistration_text=preregistration_text,
        resolved_configs=resolved_configs,
        context=formal_authorization,
        replay=formal_authorization_replay,
    )

    attempt_id = run_manifest.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise ValueError("runtime failure manifest attempt_id must be a string")
    _safe_identifier(attempt_id, field_name="runtime failure attempt_id")
    try:
        manifest_identity = RunIdentity.model_validate(run_manifest.get("run_identity"))
    except ValueError as exc:
        raise ValueError("runtime failure manifest has an invalid RunIdentity") from exc
    if manifest_identity != bundle.identity:
        raise ValueError("runtime failure manifest and bundle RunIdentity differ")
    expected_manifest_fields = {
        "run_id": bundle.run_id,
        "run_bundle_hash": bundle.run_bundle_hash,
        "status": ReceiptStatus.FAILED.value,
        "failure_phase": bundle.metadata.get("failure_phase"),
        "failure_code": bundle.metadata.get("failure_code"),
        "issuance_status": issuance_status,
        "source_identity_hash": bundle.metadata.get("source_identity_hash"),
        "claim_boundary": "runtime_failure_only_no_fit_result_no_accepted_claim",
    }
    for name, expected in expected_manifest_fields.items():
        if run_manifest.get(name) != expected:
            raise ValueError(f"runtime failure manifest {name} does not match typed evidence")
    if (
        receipt.metadata.get("attempt_id") != attempt_id
        or bundle.metadata.get("attempt_id") != attempt_id
        or receipt.metadata.get("failure_phase") != bundle.metadata.get("failure_phase")
        or receipt.metadata.get("failure_code") != bundle.metadata.get("failure_code")
        or receipt.metadata.get("bundle_schema") != "tabu.fit-runtime-failure.v1"
        or bundle.metadata.get("experiment_id") != spec.experiment_id
        or bundle.metadata.get("contract_version") != spec.contract_version
        or bundle.metadata.get("claim_boundary")
        != "runtime_failure_only_no_fit_result_no_accepted_claim"
    ):
        raise ValueError("runtime failure metadata differs across evidence objects")

    failure = _read_compact_canonical_json(root / "failure.json")
    if not isinstance(failure, Mapping):
        raise ValueError("failure.json must contain one object")
    expected_failure_fields = {
        "attempt_id": attempt_id,
        "run_id": bundle.run_id,
        "experiment_id": spec.experiment_id,
        "contract_id": spec.contract_id,
        "stage": spec.stage.value,
        "model_seed": bundle.metadata.get("model_seed"),
        "phase": bundle.metadata.get("failure_phase"),
        "code": bundle.metadata.get("failure_code"),
        "claim_boundary": "runtime_failure_only_no_fit_result_no_accepted_claim",
    }
    for name, expected in expected_failure_fields.items():
        if failure.get(name) != expected:
            raise ValueError(f"failure.json {name} does not match typed evidence")

    artifact_uris = {artifact.uri for artifact in bundle.artifacts}
    expected_artifact_uris = actual - _INFRASTRUCTURE_ARTIFACTS
    if artifact_uris != expected_artifact_uris:
        raise ValueError("RunBundle artifacts do not exactly cover failure payload files")
    for artifact in bundle.artifacts:
        path = _safe_artifact_path(root, artifact.uri)
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"run bundle artifact integrity failure: {artifact.uri}")
        if listed.get(artifact.uri) != artifact.sha256:
            raise ValueError(f"checksum and RunBundle digest differ: {artifact.uri}")
    if any(artifact.kind == "checkpoint" for artifact in bundle.artifacts):
        raise ValueError("runtime failure attempts cannot register a checkpoint")
    if issuance_status == "formal":
        assert_public_artifact_tree_safe(root, location="formal runtime failure evidence")
    return receipt


def verify_fit_attempt_artifacts(
    directory: str | os.PathLike[str],
    *,
    formal_authorization: FormalAuthorizationContext | None = None,
    formal_authorization_replay: FormalAuthorizationReplaySession | None = None,
) -> Receipt:
    """Verify exact coverage, schemas, hashes, and all cross-artifact bindings."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("fit attempt must be a real directory")
    descendants = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("fit attempts cannot contain symlinked artifacts or directories")
    try:
        checksums = _safe_artifact_path(root, "artifacts.sha256")
    except ValueError:
        raise ValueError("fit attempt is missing artifacts.sha256") from None
    if not (root / "resolved-configs").is_dir():
        raise ValueError("fit attempt is missing resolved-configs")
    listed: dict[str, str] = {}
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator != "  " or not relative:
            raise ValueError("invalid artifacts.sha256 line")
        try:
            normalized_digest = require_sha256(digest)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid artifacts.sha256 digest") from exc
        if digest != normalized_digest:
            raise ValueError("artifacts.sha256 digests must be lowercase canonical SHA-256")
        if relative in listed:
            raise ValueError("duplicate checksum path")
        path = _safe_artifact_path(root, relative)
        if _sha256(path) != digest:
            raise ValueError(f"fit artifact checksum mismatch: {relative}")
        listed[relative] = digest
    actual: set[str] = set()
    for path in descendants:
        if not path.is_file() or path.name == "artifacts.sha256":
            continue
        relative = path.relative_to(root).as_posix()
        _safe_artifact_path(root, relative)
        actual.add(relative)
    if set(listed) != actual:
        raise ValueError("artifacts.sha256 does not exactly cover the attempt")
    try:
        raw_run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid JSON artifact: run_manifest.json") from exc
    if not isinstance(raw_run_manifest, dict):
        raise ValueError("run_manifest.json must contain one object")
    manifest_schema = raw_run_manifest.get("schema")
    if manifest_schema == "tabu.fit-runtime-failure-manifest.v1":
        run_manifest = _read_compact_canonical_json(root / "run_manifest.json")
        assert isinstance(run_manifest, dict)
        return _verify_runtime_failure_artifacts(
            root,
            actual=actual,
            listed=listed,
            run_manifest=run_manifest,
            formal_authorization=formal_authorization,
            formal_authorization_replay=formal_authorization_replay,
        )
    if manifest_schema != "tabu.fit-run-manifest.v1":
        raise ValueError("unsupported fit run manifest schema")
    run_manifest = _read_canonical_json(root / "run_manifest.json")
    assert isinstance(run_manifest, dict)

    required = _REQUIRED_PAYLOAD_FILES | {
        "receipt.json",
        "run_bundle.json",
        "run_manifest.json",
    }
    missing = sorted(required - actual)
    if missing:
        raise ValueError(f"fit attempt is missing required artifacts: {missing}")

    receipt = read_receipt(root / "receipt.json")
    try:
        bundle = RunBundle.model_validate(_read_canonical_json(root / "run_bundle.json"))
        evaluation_payload = _read_canonical_json(root / "evaluation.json")
        if not isinstance(evaluation_payload, dict):
            raise ValueError("evaluation.json must contain one object")
        evaluation = EvaluationBundle(**evaluation_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("fit attempt contains an invalid typed evidence artifact") from exc
    environment_payload = _read_canonical_json(root / "environment.json")
    if not isinstance(environment_payload, Mapping):
        raise ValueError("environment.json must contain one object")
    _validate_environment_payload(environment_payload)
    if canonical_hash(environment_payload) != bundle.environment.environment_hash:
        raise ValueError("environment payload does not match RunBundle environment_hash")
    metric_rows = _read_canonical_jsonl(root / "metrics.jsonl")
    formal_spec: FitExperimentSpec | None = None
    if "experiment_id" in bundle.metadata:
        try:
            preregistration_text = (root / "preregistration.yaml").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError("formal preregistration is not valid UTF-8") from exc
        resolved_directory = root / "resolved-configs"
        resolved_configs: dict[str, Any] = {}
        for path in resolved_directory.iterdir():
            if not path.is_file() or path.suffix != ".json":
                raise ValueError("formal resolved-configs may contain only JSON files")
            name = _safe_identifier(path.stem, field_name="formal resolved config name")
            if name in resolved_configs:
                raise ValueError("duplicate formal resolved config name")
            resolved_configs[name] = _read_canonical_json(path)
        formal_spec = _validate_formal_fit_bindings(
            preregistration_text=preregistration_text,
            resolved_configs=resolved_configs,
            dataset_manifest=_read_canonical_json(root / "dataset-manifest.json"),
            split_manifest=_read_canonical_json(root / "split-manifest.json"),
            compiler_manifest=_read_canonical_json(root / "compiler-manifest.json"),
            run_identity=bundle.identity,
            model_id=bundle.model_id,
            dataset_id=bundle.dataset_id,
            fit_partition=bundle.fit_partition,
            metadata=bundle.metadata,
            environment_payload=environment_payload,
            episode_recipe_hashes=bundle.episode_recipe_hashes,
        )

    if receipt.run_id != bundle.run_id:
        raise ValueError("receipt and run bundle ids differ")
    if receipt.run_identity_hash != bundle.identity.identity_hash:
        raise ValueError("receipt and run identity hashes differ")
    if receipt.run_bundle_hash != bundle.run_bundle_hash:
        raise ValueError("receipt and run bundle hashes differ")
    if receipt.artifacts != bundle.artifacts:
        raise ValueError("receipt and run bundle artifact references differ")
    if receipt.evaluation_hashes != (evaluation.evaluation_hash,):
        raise ValueError("receipt evaluation hash does not match evaluation.json")
    issuance_status = bundle.metadata.get("issuance_status")
    if issuance_status not in {"formal", "local_unissued"}:
        raise ValueError("RunBundle has an invalid issuance_status")
    if receipt.metadata.get("issuance_status") != issuance_status:
        raise ValueError("receipt and RunBundle issuance statuses differ")
    if receipt.metadata.get("source_identity_hash") != bundle.metadata.get(
        "source_identity_hash"
    ):
        raise ValueError("receipt and RunBundle SourceIdentity hashes differ")
    for key in (
        "block_kind",
        "variant_role",
        "numeric_terminal",
        "profile_id",
        "tokenizer_version",
        "label_broadcast",
        "label_broadcast_tau",
        "reference_config",
        "terminal",
        "ll_ridge",
        "bandwidth",
        "variant_ref",
        "variant_hash",
    ):
        if receipt.metadata.get(key) != bundle.metadata.get(key):
            raise ValueError(f"receipt and RunBundle {key} metadata differ")
    if issuance_status == "formal" and formal_spec is None:
        raise ValueError("formal evidence is missing complete experiment bindings")
    if formal_spec is not None:
        _verify_recorded_formal_authorization(
            issuance_status=issuance_status,
            bundle_metadata=bundle.metadata,
            receipt_metadata=receipt.metadata,
            preregistration_text=preregistration_text,
            resolved_configs=resolved_configs,
            context=formal_authorization,
            replay=formal_authorization_replay,
        )
    elif formal_authorization is not None or formal_authorization_replay is not None:
        raise ValueError("authorization context is invalid for a non-formal fit artifact")
    if issuance_status == "formal":
        assert_public_artifact_tree_safe(root, location="formal fit evidence")

    attempt_id = run_manifest.get("attempt_id")
    if not isinstance(attempt_id, str):
        raise ValueError("run manifest attempt_id must be a string")
    _safe_identifier(attempt_id, field_name="run manifest attempt_id")
    if (
        receipt.metadata.get("attempt_id") != attempt_id
        or bundle.metadata.get("attempt_id") != attempt_id
    ):
        raise ValueError("attempt_id differs across receipt, bundle, and manifest")
    try:
        manifest_identity = RunIdentity.model_validate(run_manifest.get("run_identity"))
    except ValueError as exc:
        raise ValueError("run manifest contains an invalid RunIdentity") from exc
    if manifest_identity != bundle.identity:
        raise ValueError("run manifest and bundle RunIdentity differ")
    expected_manifest_fields = {
        "run_id": bundle.run_id,
        "run_bundle_hash": bundle.run_bundle_hash,
        "evaluation_hash": evaluation.evaluation_hash,
        "status": receipt.status.value,
        "issuance_status": issuance_status,
        "source_identity_hash": bundle.metadata.get("source_identity_hash"),
        "claim_boundary": "fit_attempt_only_no_accepted_claim",
    }
    for name, expected in expected_manifest_fields.items():
        if run_manifest.get(name) != expected:
            raise ValueError(f"run manifest {name} does not match typed evidence")
    if bundle.metadata.get("claim_boundary") != "fit_attempt_only_no_accepted_claim":
        raise ValueError("RunBundle claim boundary is not the frozen fit-only boundary")

    summaries = tuple(row for row in metric_rows if row.get("record_type") == "summary")
    if len(summaries) > 1:
        raise ValueError("metrics.jsonl contains multiple summary rows")
    if formal_spec is not None and len(summaries) != 1:
        raise ValueError("formal metrics.jsonl requires exactly one summary row")
    fit_evaluation: FitEvaluationBundle | None = None
    raw_fit_evaluation: Mapping[str, Any] | None = None
    if summaries:
        raw_value = summaries[0].get("fit_evaluation")
        if raw_value is None:
            raise ValueError("metrics summary is missing fit_evaluation")
        raw_fit_evaluation = _require_mapping(
            raw_value,
            name="metrics fit_evaluation",
        )
        try:
            fit_evaluation = FitEvaluationBundle.model_validate(raw_fit_evaluation)
        except ValueError as exc:
            raise ValueError("metrics summary has an invalid FitEvaluationBundle") from exc
        if (
            evaluation.counts.get("targets") != fit_evaluation.targets
            or evaluation.counts.get("scored_targets") != fit_evaluation.scored_targets
        ):
            raise ValueError("fit and standard evaluation target counts differ")
        if evaluation.metrics.get("coverage") != fit_evaluation.coverage:
            raise ValueError("fit and standard evaluation coverage differ")
        if formal_spec is not None and (
            fit_evaluation.experiment_id != formal_spec.experiment_id
            or fit_evaluation.stage is not formal_spec.stage
            or fit_evaluation.model_seed != bundle.metadata.get("model_seed")
        ):
            raise ValueError("FitEvaluationBundle does not match formal fit metadata")
    # The immutable JSONL object is the hash preimage.  Re-validating an old
    # object with a newer schema may materialize new optional defaults; those
    # defaults must not rewrite the historical evaluation identity.
    expected_fit_hash = (
        None if raw_fit_evaluation is None else canonical_hash(raw_fit_evaluation)
    )
    if run_manifest.get("fit_evaluation_hash") != expected_fit_hash:
        raise ValueError("run manifest fit_evaluation_hash does not match metrics.jsonl")

    traces_payload = _read_canonical_json(root / "forward-traces.json")
    if not isinstance(traces_payload, dict) or traces_payload.get("schema") != (
        "tabu.fit-forward-traces.v1"
    ):
        raise ValueError("invalid forward-traces artifact")
    trace_items = traces_payload.get("predictions")
    if not isinstance(trace_items, list) or not trace_items:
        raise ValueError("forward-traces artifact requires prediction records")
    try:
        prediction_hashes = tuple(
            require_sha256(item["prediction_hash"], field_name="prediction_hash")
            for item in trace_items
            if isinstance(item, dict)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("forward-traces contains an invalid prediction hash") from exc
    if len(prediction_hashes) != len(trace_items):
        raise ValueError("forward-traces prediction records must be objects")
    if tuple(run_manifest.get("prediction_hashes", ())) != prediction_hashes:
        raise ValueError("run manifest prediction hashes differ from forward traces")

    artifact_uris = {artifact.uri for artifact in bundle.artifacts}
    expected_artifact_uris = actual - _INFRASTRUCTURE_ARTIFACTS
    if artifact_uris != expected_artifact_uris:
        raise ValueError("RunBundle artifacts do not exactly cover payload files")
    for artifact in bundle.artifacts:
        if artifact.uri in _INFRASTRUCTURE_ARTIFACTS:
            raise ValueError("receipt infrastructure cannot recursively reference itself")
        path = _safe_artifact_path(root, artifact.uri)
        if path.stat().st_size != artifact.size_bytes or _sha256(path) != artifact.sha256:
            raise ValueError(f"run bundle artifact integrity failure: {artifact.uri}")
        if listed.get(artifact.uri) != artifact.sha256:
            raise ValueError(f"checksum and RunBundle digest differ: {artifact.uri}")
    checkpoint = root / "checkpoint" / "checkpoint.safetensors"
    if receipt.status is ReceiptStatus.SUCCEEDED and not checkpoint.is_file():
        raise ValueError("succeeded fit attempt is missing its checkpoint")
    return receipt


__all__ = [
    "FitAttemptArtifacts",
    "assert_public_artifact_tree_safe",
    "assert_public_payload_safe",
    "capture_environment",
    "verify_fit_attempt_artifacts",
    "write_fit_attempt_artifacts",
]
