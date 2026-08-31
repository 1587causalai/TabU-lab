"""Full-train-context frozen transfer for the Axis-C TabUR query-row model.

This is the main table-foundation-model-style real-data diagnostic.  It is
intentionally separate from the low-shot ``K`` panel: one real train/test
split is created, all labeled train rows enter one evidence episode, and every
held-out row is scored as a query.  Classical baselines fit exactly that same
train partition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from tabu_lab.contracts import FeatureKind, FeatureRole
from tabu_lab.models.types import DenseModelInput
from tabu_lab.primitives import RoutingOutput, masked_rbf_weights
from tabu_lab.primitives.routing import _local_linear_values

from .query_row_transfer_common import (
    QUERY_BASE_MODEL_SPEC_HASH,
)
from .tabubase_openml_new6 import OPENML_NEW6_SPECS
from .tabubase_real_metrics import classification_metrics, regression_metrics
from .tabubase_scale import ROOT_SEEDS

QUERY_OPENML_FULL_PANEL_SCHEMA = "tabu.query-row.openml-new6-full-context-panel.v1"
QUERY_OPENML_FULL_RESULT_SCHEMA = "tabu.query-row.openml-full-context-result.v1"
QUERY_OPENML_FULL_PANEL_ID = "tabur-query-row-openml-new6-full-context-2026-08-31"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(field: str, expected: Any, observed: Any) -> None:
    if observed != expected:
        raise RuntimeError(
            f"query full-context OpenML panel drift at {field}: "
            f"expected {expected!r}, got {observed!r}"
        )


def load_query_openml_full_context_panel_manifest(path: Path) -> dict[str, Any]:
    """Validate the full-context query panel without importing Axis-B identity."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing query full-context panel manifest: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("query full-context panel manifest must be a mapping")
    _require_equal("schema_version", QUERY_OPENML_FULL_PANEL_SCHEMA, payload.get("schema_version"))
    _require_equal("panel_id", QUERY_OPENML_FULL_PANEL_ID, payload.get("panel_id"))
    status = payload.get("status")
    if not isinstance(status, dict):
        raise RuntimeError("query full-context panel status must be a mapping")
    for key, expected in {
        "registration": "candidate_preregistered",
        "execution": "not_run",
        "data_materialization": "not_run",
        "empirical_claim": "none",
    }.items():
        _require_equal(f"status.{key}", expected, status.get(key))
    model = payload.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("query full-context panel model must be a mapping")
    for key, expected in {
        "contract_id": "tabu.query.row",
        "contract_version": "0.1.0",
        "model_spec_hash": QUERY_BASE_MODEL_SPEC_HASH,
        "profile_id": "supervised.label_broadcast.v1",
        "row_token_count": 4,
        "synthetic_generator_id": "tabur.supervised-query-row-diverse-v2",
    }.items():
        _require_equal(f"model.{key}", expected, model.get(key))
    source = payload.get("source_contract")
    if not isinstance(source, dict):
        raise RuntimeError("query full-context source_contract must be a mapping")
    for key, expected in {
        "provider": "OpenML",
        "api": "sklearn.datasets.fetch_openml",
        "identity_key": "data_id",
        "target_column": "default-target",
        "as_frame": False,
        "parser": "liac-arff",
        "cache": True,
    }.items():
        _require_equal(f"source_contract.{key}", expected, source.get(key))
    entries = payload.get("datasets")
    if not isinstance(entries, list):
        raise RuntimeError("query full-context datasets must be a list")
    by_id = {entry.get("dataset_id"): entry for entry in entries if isinstance(entry, dict)}
    expected_ids = tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
    if set(by_id) != set(expected_ids):
        raise RuntimeError(f"query full-context panel must contain exactly {expected_ids!r}")
    for spec in OPENML_NEW6_SPECS:
        entry = by_id[spec.dataset_id]
        for key, expected in {
            "openml_name": spec.openml_name,
            "data_id": spec.data_id,
            "version": spec.version,
            "upstream_md5": spec.upstream_md5,
            "license": spec.license,
            "task": spec.task,
            "rows": spec.rows,
            "predictors": spec.predictors,
            "classes": spec.classes,
        }.items():
            _require_equal(f"datasets.{spec.dataset_id}.{key}", expected, entry.get(key))
    evaluation = payload.get("evaluation_design")
    if not isinstance(evaluation, dict):
        raise RuntimeError("query full-context evaluation_design must be a mapping")
    for key, expected in {
        "checkpoint_seeds": list(ROOT_SEEDS),
        "split_seeds": list(ROOT_SEEDS),
        "context_policy": "full_train",
        "split_fraction": 0.7,
        "query_policy": "all_heldout_rows",
        "query_chunk_rows": 64,
    }.items():
        _require_equal(f"evaluation_design.{key}", expected, evaluation.get(key))
    return {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "canonical_payload_sha256": hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest(),
        "payload": payload,
        "dataset_ids": expected_ids,
    }


def _forward_query_row_no_trace(
    model: torch.nn.Module,
    evidence: Any,
    *,
    device: torch.device,
) -> Any:
    """Use the public EvidenceEpisode boundary, suppressing quadratic trace ledgers."""

    if not hasattr(model, "_forward_dense"):
        raise TypeError("query-row model does not expose its dense implementation boundary")
    dense = DenseModelInput.from_any(evidence).to(device)
    return model._forward_dense(dense, emit_trace=False)


def _forward_query_response_only(
    model: torch.nn.Module,
    evidence: Any,
    *,
    context_rows: int,
    classes: int | None,
    query_chunk_rows: int,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run full evidence dynamics, then materialize only response-query routing.

    The ordinary dense terminal constructs a same-column routing ledger for
    every row and feature.  That is quadratic in the number of rows, while a
    supervised transfer score needs only the response column and held-out
    rows.  This adapter preserves the complete train+test episode through the
    tokenizer, label broadcast, dynamics and geometry; it bounds memory only
    for the final response readout.
    """

    if context_rows < 1 or query_chunk_rows < 1:
        raise ValueError("context_rows and query_chunk_rows must be positive")
    encoded = model._encode_dense_queries(
        DenseModelInput.from_any(evidence).to(device),
        emit_trace=False,
    )
    resolved, _, _, cells, numeric_scale_state = encoded
    if context_rows >= cells.shape[1]:
        raise ValueError("full-context readout requires context and query rows")
    response_features = [
        index
        for index, spec in enumerate(resolved.feature_specs)
        if FeatureRole(getattr(spec.role, "value", spec.role)) is FeatureRole.RESPONSE
    ]
    if len(response_features) != 1:
        raise RuntimeError("query response readout requires exactly one response feature")
    response_feature = int(response_features[0])
    spec = resolved.feature_specs[response_feature]
    response_kind = FeatureKind(getattr(spec.kind, "value", spec.kind))
    declared_classes = len(spec.domain) if response_kind is not FeatureKind.NUMERIC else 0
    if classes is None and response_kind is not FeatureKind.NUMERIC:
        raise RuntimeError("categorical response requires a declared class count")
    if classes is not None and (
        response_kind is FeatureKind.NUMERIC or declared_classes != classes
    ):
        raise RuntimeError("response feature kind/domain does not match the split")
    expected_query_targets = torch.zeros_like(resolved.query_target_mask)
    expected_query_targets[:, context_rows:, response_feature] = True
    if not torch.equal(resolved.query_target_mask, expected_query_targets):
        raise RuntimeError("query targets must be exactly the held-out response cells")
    support_mask = resolved.visible_mask[:, :context_rows, response_feature]
    if not bool(support_mask.all()):
        raise RuntimeError("all train response labels must be visible context support")
    if bool(resolved.visible_mask[:, context_rows:, response_feature].any()):
        raise RuntimeError("held-out response truth entered the model evidence")

    coordinates = model.geometry(cells)
    context_coordinates = coordinates[:, :context_rows, response_feature, :].to(torch.float32)
    query_coordinates = coordinates[:, context_rows:, response_feature, :].to(torch.float32)
    if response_kind is FeatureKind.NUMERIC:
        support_values = numeric_scale_state.standardized_values[
            :, :context_rows, response_feature
        ].to(torch.float32)
        numeric_chunks: list[torch.Tensor] = []
    else:
        support_values = resolved.values[:, :context_rows, response_feature].to(torch.float32)
        probability_chunks: list[torch.Tensor] = []
    for offset in range(0, query_coordinates.shape[1], query_chunk_rows):
        query = query_coordinates[:, offset : offset + query_chunk_rows]
        difference = query.unsqueeze(2) - context_coordinates.unsqueeze(1)
        squared_distance = difference.square().sum(dim=-1)
        allowed = support_mask.unsqueeze(1).expand_as(squared_distance)
        routing = masked_rbf_weights(
            squared_distance,
            allowed,
            bandwidth=model.terminal.terminal.bandwidth.to(dtype=torch.float32),
        )
        if response_kind is FeatureKind.NUMERIC:
            terminal_kind = model.terminal.numeric_terminal
            if terminal_kind == "local_linear":
                expanded = RoutingOutput(
                    weights=routing.weights.unsqueeze(2),
                    log_weights=routing.log_weights.unsqueeze(2),
                    support_mask=routing.support_mask.unsqueeze(2),
                    support_available=routing.support_available.unsqueeze(2),
                    support_count=routing.support_count.unsqueeze(2),
                )
                expanded_values = (
                    support_values.unsqueeze(1).unsqueeze(2).expand(-1, query.shape[1], 1, -1)
                )
                values = _local_linear_values(
                    expanded,
                    difference.unsqueeze(2),
                    expanded_values,
                    ridge=float(model.terminal.ll_ridge),
                ).squeeze(2)
            elif terminal_kind == "nadaraya_watson":
                values = torch.einsum("bqs,bs->bq", routing.weights, support_values)
            else:  # pragma: no cover - QueryTerminalAdapter currently fixes this set
                raise RuntimeError(f"unsupported query numeric terminal: {terminal_kind}")
            numeric_chunks.append(values)
        else:
            assert classes is not None
            labels = support_values.to(torch.long)
            if bool(((labels < 0) | (labels >= classes)).any()):
                raise RuntimeError("context response label is outside the declared domain")
            membership = torch.nn.functional.one_hot(labels, num_classes=classes).to(
                routing.weights.dtype
            )
            probability_chunks.append(torch.einsum("bqs,bsc->bqc", routing.weights, membership))
    if response_kind is FeatureKind.NUMERIC:
        standardized = torch.cat(numeric_chunks, dim=1)
        response_scale = numeric_scale_state.scale[:, 0, response_feature].view(-1, 1)
        response_mean = numeric_scale_state.mean[:, 0, response_feature].view(-1, 1)
        raw = standardized * response_scale + response_mean
        return None, raw.detach().cpu().numpy().astype(np.float64)
    assert classes is not None
    probabilities = torch.cat(probability_chunks, dim=1).detach().cpu().numpy().astype(np.float64)
    probabilities = np.clip(probabilities, 1.0e-12, None)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities, None


def _tabur_full_metrics(
    model: torch.nn.Module,
    evidence: Any,
    split: Any,
    *,
    context_rows: int,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    truth = split.response[split.query_indices]
    with torch.inference_mode():
        probabilities, predicted = _forward_query_response_only(
            model,
            evidence,
            context_rows=context_rows,
            classes=split.classes,
            query_chunk_rows=64,
            device=device,
        )
    if split.dataset.task == "classification":
        assert split.classes is not None and probabilities is not None
        if probabilities.ndim == 3 and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        if probabilities.shape != (len(truth), split.classes):
            raise RuntimeError("full-context TabUR classification query shape mismatch")
        return (
            classification_metrics(truth, probabilities, classes=split.classes),
            torch.from_numpy(probabilities),
        )
    assert predicted is not None
    if predicted.ndim == 2 and predicted.shape[0] == 1:
        predicted = predicted[0]
    if predicted.shape != truth.shape:
        raise RuntimeError("full-context TabUR regression query shape mismatch")
    return regression_metrics(truth, predicted, target_scale=split.target_scale), torch.from_numpy(
        predicted
    )


def _mean_metric_records(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows or any(row.get("metrics") is None for row in rows):
        return {}
    metric_names = tuple(rows[0]["metrics"])
    return {
        name: float(np.mean([float(row["metrics"][name]) for row in rows])) for name in metric_names
    }


def _summary(
    baseline_records: list[dict[str, Any]],
    frozen_records: list[dict[str, Any]],
    *,
    dataset_ids: tuple[str, ...],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset_id in dataset_ids:
        base = [row for row in baseline_records if row["dataset_id"] == dataset_id]
        frozen = [row for row in frozen_records if row["dataset_id"] == dataset_id]
        task = str(base[0]["task"])
        output[dataset_id] = {
            "task": task,
            "train_rows_mean": float(np.mean([row["train_rows_total"] for row in base])),
            "test_rows_mean": float(np.mean([row["query_rows"] for row in base])),
            "baselines": {
                estimator: _mean_metric_records(
                    [row for row in base if row["estimator"] == estimator]
                )
                for estimator in ("linear", "mlp", "xgboost")
            },
            "frozen": {
                arm: _mean_metric_records([row for row in frozen if row["arm"] == arm])
                for arm in ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled")
            },
        }
    for task, primary in (
        ("classification", "normalized_nll"),
        ("regression", "scaled_rmse"),
    ):
        selected = [
            output[dataset_id] for dataset_id in dataset_ids if output[dataset_id]["task"] == task
        ]
        macro: dict[str, Any] = {"dataset_count": len(selected), "primary_metric": primary}
        for arm in ("pretrained_frozen", "random_init_frozen", "pretrained_shuffled"):
            values = [item["frozen"][arm].get(primary) for item in selected]
            values = [float(value) for value in values if value is not None]
            macro[arm] = {"dataset_macro_mean_primary": float(np.mean(values)) if values else None}
        for estimator in ("linear", "mlp", "xgboost"):
            values = [item["baselines"][estimator].get(primary) for item in selected]
            values = [float(value) for value in values if value is not None]
            macro[estimator] = {
                "dataset_macro_mean_primary": float(np.mean(values)) if values else None
            }
        output[f"_{task}_macro"] = macro
    return output


def run_query_row_openml_full_context(
    *,
    panel_manifest: Path,
    checkpoint_paths: tuple[Path, ...],
    dataset_ids: tuple[str, ...] | None = None,
    checkpoint_seeds: tuple[int, ...] = ROOT_SEEDS,
    split_seeds: tuple[int, ...] = ROOT_SEEDS,
    device: str | torch.device = "cuda",
    openml_data_home: Path | None = None,
) -> dict[str, Any]:
    """Refuse the frozen 0.1 panel under the current 0.2 readout runtime."""

    del dataset_ids, device, openml_data_home
    if not checkpoint_paths:
        raise ValueError("at least one TabUR checkpoint is required")
    if tuple(checkpoint_seeds) != ROOT_SEEDS or tuple(split_seeds) != ROOT_SEEDS:
        raise ValueError("full-context OpenML panel requires the preregistered three seeds")
    load_query_openml_full_context_panel_manifest(panel_manifest)
    for checkpoint_path in checkpoint_paths:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"missing TabUR checkpoint: {checkpoint_path}")
    raise RuntimeError(
        "the supplied full-context preregistration is frozen at "
        "tabu.query.row@0.1.0 and is superseded by the 0.2.0 symmetric-readout contract; "
        "this historical runner intentionally performs no data access or tensor loading. "
        "A new 0.2.0 preregistration is required before the anchored model may be evaluated."
    )


__all__ = [
    "QUERY_OPENML_FULL_PANEL_ID",
    "QUERY_OPENML_FULL_PANEL_SCHEMA",
    "QUERY_OPENML_FULL_RESULT_SCHEMA",
    "load_query_openml_full_context_panel_manifest",
    "run_query_row_openml_full_context",
]
