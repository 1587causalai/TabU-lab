"""Optimizer-free TabUBase ICL evaluation on real datasets.

This is intentionally separate from :mod:`tabubase_real_benchmark`: the latter
fine-tunes every neural arm, while this module never creates an optimizer and
requires byte-identical parameter hashes before and after evaluation.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    OriginState,
    TruthSidecar,
    canonical_hash,
    origin_code,
)
from tabu_lab.models.components import CellTokenizer

from .tabubase_icl import FROZEN_ARMS, K_GRID, paired_world_bootstrap
from .tabubase_openml_cached import (
    CACHED_OPENML_BY_ID,
    CACHED_OPENML_PANEL_ID,
    build_cached_openml_materialization_manifest,
    fetch_cached_openml_dataset,
    is_cached_openml_panel_manifest,
    load_cached_openml_panel_manifest,
)
from .tabubase_openml_new6 import (
    OPENML_NEW6_BY_ID,
    OPENML_NEW6_PANEL_ID,
    FetchOpenML,
    fetch_openml_new6_dataset,
    load_openml_new6_panel_manifest,
)
from .tabubase_real_benchmark import RealDataset, _source_tree_hash, load_real_dataset
from .tabubase_real_metrics import classification_metrics, regression_metrics
from .tabubase_response_readout import query_response_readout
from .tabubase_scale import (
    ROOT_SEEDS,
    _sha256_file,
    _state_hash,
    build_tabubase_scale_model,
    load_pretrain_checkpoint,
)

DEFAULT_REAL_ICL_DATASETS = (
    "iris",
    "wine",
    "breast_cancer",
    "digits",
    "diabetes",
    "california_housing",
)
MAX_PREDICTORS = 63  # SCALE_MODEL_CONFIG.max_features minus the response column.
FULL_CONTEXT_POLICY = "full_train"
LOW_SHOT_CONTEXT_POLICY = "low_shot_grid"
RealContextPolicy = Literal["full_train", "low_shot_grid"]
REAL_FULL_CONTEXT_SCHEMA = "tabu.transfer-base-real-full-context-frozen-icl-local-unissued.v1"
REAL_LOW_SHOT_SCHEMA = "tabu.transfer-base-real-frozen-icl-local-unissued.v1"


@dataclass(frozen=True, slots=True)
class RealIclConfig:
    checkpoint_root: Path
    output_path: Path
    dataset_ids: tuple[str, ...] = DEFAULT_REAL_ICL_DATASETS
    checkpoint_seeds: tuple[int, ...] = ROOT_SEEDS
    split_seeds: tuple[int, ...] = ROOT_SEEDS
    context_policy: RealContextPolicy = FULL_CONTEXT_POLICY
    query_limit: int | None = None
    query_chunk_rows: int = 64
    bootstrap_replicates: int = 2_000
    nominal_codebook_size: int = 100
    nominal_codebook_seed: int = 1729
    checkpoint_run_suffix: str = "-icl-kcurriculum-v1"
    panel_manifest_path: Path | None = None
    openml_cache: bool = True
    openml_data_home: Path | None = None

    def validate(self) -> RealIclConfig:
        if not self.dataset_ids or len(set(self.dataset_ids)) != len(self.dataset_ids):
            raise ValueError("real ICL datasets must be non-empty and unique")
        if not self.checkpoint_seeds or len(set(self.checkpoint_seeds)) != len(
            self.checkpoint_seeds
        ):
            raise ValueError("checkpoint seeds must be non-empty and unique")
        if not self.split_seeds or len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split seeds must be non-empty and unique")
        if self.context_policy not in {FULL_CONTEXT_POLICY, LOW_SHOT_CONTEXT_POLICY}:
            raise ValueError("real ICL context policy must be full_train or low_shot_grid")
        if self.query_limit is not None and self.query_limit < 1:
            raise ValueError("query limit must be positive when provided")
        if self.query_chunk_rows < 1:
            raise ValueError("query chunk rows must be positive")
        if self.context_policy == LOW_SHOT_CONTEXT_POLICY and self.query_limit is None:
            raise ValueError("low-shot real ICL requires an explicit query limit")
        if self.context_policy == FULL_CONTEXT_POLICY and self.query_limit is not None:
            raise ValueError("full-context real ICL must evaluate every held-out query row")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap replicates must be at least 100")
        if self.nominal_codebook_size != 100:
            raise ValueError("real ICL v1 is bound to the 100-code tokenizer v2 pool")
        if self.checkpoint_run_suffix and (
            not self.checkpoint_run_suffix.startswith("-") or "/" in self.checkpoint_run_suffix
        ):
            raise ValueError("checkpoint suffix must be empty or a safe leading-dash suffix")
        new6_ids = set(OPENML_NEW6_BY_ID)
        cached_ids = set(CACHED_OPENML_BY_ID)
        if self.panel_manifest_path is None and (new6_ids | cached_ids).intersection(
            self.dataset_ids
        ):
            raise ValueError("pinned OpenML datasets require an explicit panel manifest")
        if self.panel_manifest_path is not None:
            cached_panel = (
                load_cached_openml_panel_manifest(self.panel_manifest_path)
                if is_cached_openml_panel_manifest(self.panel_manifest_path)
                else None
            )
            if cached_panel is not None:
                allowed_ids = cached_ids
                panel = cached_panel
                panel_name = "cached OpenML"
            else:
                allowed_ids = new6_ids
                panel = load_openml_new6_panel_manifest(self.panel_manifest_path)
                panel_name = "OpenML new6"
            unsupported = set(self.dataset_ids) - allowed_ids
            if unsupported:
                raise ValueError(
                    f"an {panel_name} panel manifest cannot be mixed with other datasets: "
                    f"{sorted(unsupported)!r}"
                )
            if not self.openml_cache:
                raise ValueError(f"the preregistered {panel_name} panel requires cache=true")
            if (
                cached_panel is None
                and self.openml_data_home is not None
                and not self.openml_data_home.is_dir()
            ):
                raise ValueError("OpenML new6 data_home must be an existing directory")
            if not set(self.checkpoint_seeds).issubset(set(ROOT_SEEDS)):
                raise ValueError(
                    f"{panel_name} checkpoint seeds must come from the preregistered set"
                )
            if self.split_seeds != ROOT_SEEDS:
                raise ValueError(f"{panel_name} requires the preregistered three split seeds")
            if panel.context_policy != self.context_policy:
                raise ValueError(
                    f"{panel_name} runtime context policy differs from its panel manifest"
                )
            expected_query_limit = 256 if panel.context_policy == LOW_SHOT_CONTEXT_POLICY else None
            if self.query_limit != expected_query_limit or self.query_chunk_rows != 64:
                raise ValueError(
                    f"{panel_name} runtime query policy differs from its panel manifest: "
                    f"expected query_limit={expected_query_limit}, query_chunk_rows=64"
                )
        return self


@dataclass(frozen=True, slots=True)
class PreparedRealIclSplit:
    dataset: RealDataset
    split_seed: int
    features: np.ndarray
    response: np.ndarray
    train_indices: np.ndarray
    query_indices: np.ndarray
    context_order: np.ndarray
    feature_indices: np.ndarray
    classes: int | None
    target_scale: float


def real_icl_split_manifest(split: PreparedRealIclSplit) -> dict[str, Any]:
    """Return the row/feature identity shared by frozen and baseline evaluators."""

    return {
        "dataset_id": split.dataset.dataset_id,
        "dataset_sha256": split.dataset.content_hash,
        "split_seed": split.split_seed,
        "train_rows": len(split.train_indices),
        "query_rows": len(split.query_indices),
        "train_indices_sha256": canonical_hash(split.train_indices.tolist()),
        "context_order_sha256": canonical_hash(split.context_order.tolist()),
        "query_indices_sha256": canonical_hash(split.query_indices.tolist()),
        "feature_indices_sha256": canonical_hash(split.feature_indices.tolist()),
        "feature_indices": split.feature_indices.tolist(),
        "target_scale": split.target_scale,
    }


def _stratified_partition(
    response: np.ndarray, *, seed: int, train_fraction: float = 0.7
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train: list[int] = []
    query: list[int] = []
    for label in np.unique(response):
        indices = rng.permutation(np.flatnonzero(response == label))
        cut = min(max(1, round(train_fraction * len(indices))), len(indices) - 1)
        train.extend(indices[:cut].tolist())
        query.extend(indices[cut:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(query), dtype=np.int64)


def _nested_class_order(indices: np.ndarray, response: np.ndarray, *, seed: int) -> np.ndarray:
    """Return a deterministic nested order that covers each class as soon as possible."""

    rng = np.random.default_rng(seed)
    labels = np.unique(response[indices])
    queues = {
        int(label): rng.permutation(indices[response[indices] == label]).tolist()
        for label in labels
    }
    ordered: list[int] = []
    while any(queues.values()):
        for label in labels:
            queue = queues[int(label)]
            if queue:
                ordered.append(queue.pop())
    return np.asarray(ordered, dtype=np.int64)


def prepare_real_icl_split(
    dataset: RealDataset,
    *,
    split_seed: int,
    query_limit: int | None = None,
) -> PreparedRealIclSplit:
    """Split before feature selection and preserve every train row as context evidence."""

    if dataset.task == "classification":
        train, query = _stratified_partition(dataset.response, seed=split_seed)
        labels = np.unique(dataset.response[train])
        mapping = {int(label): index for index, label in enumerate(labels.tolist())}
        response = np.asarray([mapping[int(value)] for value in dataset.response], dtype=np.int64)
        context_order = _nested_class_order(train, response, seed=split_seed + 17)
        classes: int | None = len(labels)
        if query_limit is not None and query_limit < classes:
            raise ValueError("classification query limit must cover every class")
        if query_limit is not None and len(query) > query_limit:
            query = _nested_class_order(query, response, seed=split_seed + 23)[:query_limit]
        target_scale = 1.0
    else:
        rng = np.random.default_rng(split_seed)
        permutation = rng.permutation(len(dataset.response))
        cut = min(max(32, round(0.7 * len(permutation))), len(permutation) - 1)
        train, query = permutation[:cut], permutation[cut:]
        if query_limit is not None and len(query) > query_limit:
            query = query[:query_limit]
        context_order = rng.permutation(train)
        response = np.asarray(dataset.response, dtype=np.float32)
        classes = None
        target_scale = max(float(response[train].std()), 1.0e-6)
    # The frozen checkpoint supports 64 total columns. Selection is fitted only
    # on the train partition and is label-free.
    if dataset.features.shape[1] > MAX_PREDICTORS:
        variances = dataset.features[train].var(axis=0)
        feature_indices = np.sort(np.argsort(-variances, kind="stable")[:MAX_PREDICTORS]).astype(
            np.int64
        )
    else:
        feature_indices = np.arange(dataset.features.shape[1], dtype=np.int64)
    features = np.asarray(dataset.features[:, feature_indices], dtype=np.float32)
    return PreparedRealIclSplit(
        dataset=dataset,
        split_seed=split_seed,
        features=features,
        response=response,
        train_indices=np.asarray(train, dtype=np.int64),
        query_indices=np.asarray(query, dtype=np.int64),
        context_order=np.asarray(context_order, dtype=np.int64),
        feature_indices=feature_indices,
        classes=classes,
        target_scale=target_scale,
    )


def build_real_icl_episode(
    split: PreparedRealIclSplit,
    *,
    context_size: int,
    query_indices: np.ndarray,
    shuffled_context: bool,
    context_policy: RealContextPolicy = FULL_CONTEXT_POLICY,
) -> tuple[EvidenceEpisode, TruthSidecar]:
    if context_size < 0 or context_size > len(split.context_order):
        raise ValueError("context size is outside the available train partition")
    if context_policy == FULL_CONTEXT_POLICY and context_size != len(split.train_indices):
        raise ValueError("full-context ICL must expose every train-partition row")
    if context_policy == LOW_SHOT_CONTEXT_POLICY and context_size not in K_GRID:
        raise ValueError("low-shot context size is outside the frozen diagnostic grid")
    context = split.context_order[:context_size]
    rows = np.concatenate((context, np.asarray(query_indices, dtype=np.int64)))
    predictors = torch.from_numpy(split.features[rows])
    truth_response = torch.as_tensor(split.response[rows], dtype=torch.float32)
    shown_response = truth_response.clone()
    if shuffled_context and context_size > 1:
        rng = np.random.default_rng(split.split_seed * 1_000_003 + context_size)
        shown_response[:context_size] = shown_response[:context_size][
            torch.as_tensor(rng.permutation(context_size), dtype=torch.int64)
        ]
    shown_response[context_size:] = 0.0
    values = torch.cat((predictors, shown_response.unsqueeze(1)), dim=1)
    feature_specs = tuple(
        FeatureSpec(name=f"feature_{int(source_index)}") for source_index in split.feature_indices
    )
    if split.dataset.task == "classification":
        assert split.classes is not None
        response_spec = FeatureSpec(
            name="response",
            kind=FeatureKind.CATEGORICAL,
            domain=tuple(f"class_{index}" for index in range(split.classes)),
            codebook_id=f"{split.dataset.dataset_id}-response-v1",
            role=FeatureRole.RESPONSE,
        )
    else:
        response_spec = FeatureSpec(name="response", role=FeatureRole.RESPONSE)
    feature_specs = (*feature_specs, response_spec)
    roles = torch.full(
        values.shape,
        int(ForwardRole.RECEIVER | ForwardRole.SOURCE),
        dtype=torch.int64,
    )
    origins = torch.full(values.shape, origin_code(OriginState.OBSERVED), dtype=torch.int64)
    roles[context_size:, -1] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    origins[context_size:, -1] = origin_code(OriginState.QUERY)
    suffix = "shuffled" if shuffled_context else "normal"
    episode_id = (
        f"real-icl-{split.dataset.dataset_id}-s{split.split_seed}-k{context_size}-"
        f"{suffix}-{int(query_indices[0]) if len(query_indices) else 0}"
    )
    row_ids = tuple(f"{split.dataset.dataset_id}-row-{int(index)}" for index in rows)
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id=split.dataset.dataset_id,
        source_partition="train_context",
        fit_partition="none_frozen_icl",
        row_ids=row_ids,
        feature_names=tuple(spec.name for spec in feature_specs),
        feature_specs=feature_specs,
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "statistics_scope": "visible_context_only",
            "split_seed": split.split_seed,
            "context_size": context_size,
            "context_policy": context_policy,
            "train_rows_total": len(split.train_indices),
            "query_rows_total": len(split.query_indices),
            "context_shuffled": shuffled_context,
        },
    )
    truth_values = torch.zeros_like(values)
    truth_values[context_size:, -1] = truth_response[context_size:]
    truth = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": (
                    "tabubase-real-full-context-frozen-icl-episode.v1"
                    if context_policy == FULL_CONTEXT_POLICY
                    else "tabubase-real-low-shot-frozen-icl-episode.v1"
                ),
                "dataset_sha256": split.dataset.content_hash,
                "split_seed": split.split_seed,
                "context_policy": context_policy,
                "context_indices": context.tolist(),
                "query_indices": query_indices.tolist(),
                "shuffled_context": shuffled_context,
            }
        ),
        row_ids=row_ids,
        feature_names=evidence.feature_names,
        target_values=truth_values,
        target_mask=evidence.target_mask,
    )
    return evidence, truth


def _evaluate(
    model: torch.nn.Module,
    split: PreparedRealIclSplit,
    *,
    context_size: int,
    shuffled_context: bool,
    context_policy: RealContextPolicy,
    query_chunk_rows: int,
    device: torch.device,
) -> dict[str, float]:
    truth = split.response[split.query_indices]
    if context_size == 0:
        if split.dataset.task == "classification":
            assert split.classes is not None
            probabilities = np.full(
                (len(truth), split.classes), 1.0 / split.classes, dtype=np.float64
            )
        else:
            predicted = np.zeros(len(truth), dtype=np.float64)
    elif context_policy == FULL_CONTEXT_POLICY:
        # All held-out predictor rows belong to one transductive episode.  Only
        # the query-target readout axis is chunked after shared dynamics; the
        # chunk size therefore cannot change the evidence set or model state.
        evidence, _ = build_real_icl_episode(
            split,
            context_size=context_size,
            query_indices=split.query_indices,
            shuffled_context=shuffled_context,
            context_policy=context_policy,
        )
        probabilities, predicted = _forward_full_context_response(
            model,
            evidence,
            context_size=context_size,
            classes=split.classes,
            query_readout_chunk_rows=query_chunk_rows,
            device=device,
        )
    else:
        chunks: list[np.ndarray] = []
        model.eval()
        with torch.inference_mode():
            for offset in range(0, len(split.query_indices), query_chunk_rows):
                query = split.query_indices[offset : offset + query_chunk_rows]
                evidence, _ = build_real_icl_episode(
                    split,
                    context_size=context_size,
                    query_indices=query,
                    shuffled_context=shuffled_context,
                    context_policy=context_policy,
                )
                prediction = model._forward_dense(evidence.to(device), emit_trace=False)
                if split.dataset.task == "classification":
                    values = prediction.entries["distribution"].values
                    if values is None:
                        raise RuntimeError("real ICL classification returned no distribution")
                    assert split.classes is not None
                    chunk = values[context_size:, -1, : split.classes].detach().cpu().numpy()
                else:
                    values = prediction.entries["numeric"].values
                    if values is None:
                        raise RuntimeError("real ICL regression returned no numeric prediction")
                    chunk = values[context_size:, -1].detach().cpu().numpy()
                chunks.append(chunk)
        if split.dataset.task == "classification":
            probabilities = np.concatenate(chunks).astype(np.float64)
            probabilities = np.clip(probabilities, 1.0e-8, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        else:
            predicted = np.concatenate(chunks).astype(np.float64)

    if split.dataset.task == "classification":
        assert split.classes is not None
        assert probabilities is not None
        return classification_metrics(truth, probabilities, classes=split.classes)
    assert predicted is not None
    return regression_metrics(truth, predicted, target_scale=split.target_scale)


def _forward_full_context_response(
    model: torch.nn.Module,
    evidence: EvidenceEpisode,
    *,
    context_size: int,
    classes: int | None,
    query_readout_chunk_rows: int,
    device: torch.device,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Evaluate response targets against every context label without ``N x N`` routing.

    The complete context and complete query predictor set pass through the
    ordinary TabUBase symbolizer, tokenizer, label broadcast, and dynamics in
    one episode.  Only Step 4 is specialized: response-query coordinates are
    routed against response-context coordinates in bounded query chunks.  On a
    dense-supportable episode this is numerically equivalent to selecting the
    response-query slice of ``_forward_dense``.
    """

    model.eval()
    with torch.inference_mode():
        result = query_response_readout(
            model,
            evidence.to(device),
            context_rows=context_size,
            query_readout_chunk_rows=query_readout_chunk_rows,
        )
        if classes is not None:
            if (
                result.probabilities is None
                or result.response_kind is FeatureKind.NUMERIC
                or result.probabilities.shape[-1] != classes
            ):
                raise RuntimeError("full-context classification domain mismatch")
            probabilities = result.probabilities[0].detach().cpu().numpy()
            probabilities = np.clip(probabilities.astype(np.float64), 1.0e-8, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            return probabilities, None
        if result.numeric_values is None or result.response_kind is not FeatureKind.NUMERIC:
            raise RuntimeError("full-context regression response kind mismatch")
        predicted = result.numeric_values[0].detach().cpu().numpy()
        return None, predicted.astype(np.float64)


def _checkpoint_path(config: RealIclConfig, seed: int) -> Path:
    run_name = (
        f"tabubase-pt-s1-seed-{seed}-nominal-codebook-v2-"
        f"b{config.nominal_codebook_size}-s{config.nominal_codebook_seed}"
        f"{config.checkpoint_run_suffix}"
    )
    return config.checkpoint_root / run_name / "checkpoint-20000.safetensors"


def _source_commit() -> str | None:
    explicit = os.environ.get("TABU_SOURCE_COMMIT")
    if explicit:
        return explicit
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _curve_area(values: Sequence[float], context_sizes: Sequence[int]) -> float:
    if len(values) != len(context_sizes) or not values:
        raise ValueError("curve area inputs must be non-empty and aligned")
    if len(values) == 1:
        return float(values[0])
    x = np.log2(np.asarray(context_sizes, dtype=np.float64))
    return float(np.trapezoid(np.asarray(values, dtype=np.float64), x=x) / (x[-1] - x[0]))


def _summarize_low_shot(records: list[dict[str, Any]], *, replicates: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for dataset_id in sorted({str(row["dataset_id"]) for row in records}):
        rows = [row for row in records if row["dataset_id"] == dataset_id]
        task = str(rows[0]["task"])
        classes = rows[0]["classes"]
        metric = "normalized_nll" if task == "classification" else "scaled_rmse"
        eligible = [k for k in K_GRID if k >= int(classes)] if classes else [k for k in K_GRID if k]
        curves = {
            arm: {
                str(k): float(
                    np.mean(
                        [
                            row["metrics"][metric]
                            for row in rows
                            if row["arm"] == arm and row["context_size"] == k
                        ]
                    )
                )
                for k in K_GRID
            }
            for arm in FROZEN_ARMS
        }
        replicate_keys = sorted(
            {(int(row["checkpoint_seed"]), int(row["split_seed"])) for row in rows}
        )
        areas: dict[str, list[float]] = {arm: [] for arm in FROZEN_ARMS}
        endpoints: dict[str, list[float]] = {arm: [] for arm in FROZEN_ARMS}
        for checkpoint_seed, split_seed in replicate_keys:
            selected = [
                row
                for row in rows
                if row["checkpoint_seed"] == checkpoint_seed and row["split_seed"] == split_seed
            ]
            for arm in FROZEN_ARMS:
                by_k = {
                    int(row["context_size"]): float(row["metrics"][metric])
                    for row in selected
                    if row["arm"] == arm
                }
                areas[arm].append(_curve_area([by_k[k] for k in eligible], eligible))
                endpoints[arm].append(by_k[32])
        random_gain = [
            random - pretrained
            for random, pretrained in zip(
                areas["random_init_frozen"], areas["pretrained_frozen"], strict=True
            )
        ]
        shuffled_gain = [
            shuffled - pretrained
            for shuffled, pretrained in zip(
                areas["pretrained_shuffled"], areas["pretrained_frozen"], strict=True
            )
        ]
        endpoint_gain = [
            random - pretrained
            for random, pretrained in zip(
                endpoints["random_init_frozen"], endpoints["pretrained_frozen"], strict=True
            )
        ]
        random_ci = paired_world_bootstrap(
            random_gain, replicates=replicates, seed=1729 + len(summary)
        )
        shuffled_ci = paired_world_bootstrap(
            shuffled_gain, replicates=replicates, seed=2718 + len(summary)
        )
        summary[dataset_id] = {
            "task": task,
            "classes": classes,
            "primary_metric": metric,
            "primary_context_sizes": eligible,
            "mean_curve": curves,
            "pretrained_vs_random_primary_aulc_gain": {
                "mean": random_ci[0],
                "lower_95": random_ci[1],
                "upper_95": random_ci[2],
                "paired_wins": sum(value > 0.0 for value in random_gain),
                "paired_total": len(random_gain),
            },
            "normal_vs_shuffled_primary_aulc_gain": {
                "mean": shuffled_ci[0],
                "lower_95": shuffled_ci[1],
                "upper_95": shuffled_ci[2],
                "paired_wins": sum(value > 0.0 for value in shuffled_gain),
                "paired_total": len(shuffled_gain),
            },
            "pretrained_vs_random_k32_gain": {
                "mean": float(np.mean(endpoint_gain)),
                "paired_wins": sum(value > 0.0 for value in endpoint_gain),
                "paired_total": len(endpoint_gain),
            },
        }
    dataset_summaries = tuple(summary.values())
    for task in ("classification", "regression"):
        task_rows = [value for value in dataset_summaries if value["task"] == task]
        if not task_rows:
            summary[f"_{task}_macro"] = {"dataset_count": 0, "status": "not_run"}
            continue
        summary[f"_{task}_macro"] = {
            "dataset_count": len(task_rows),
            "mean_pretrained_vs_random_primary_aulc_gain": float(
                np.mean(
                    [row["pretrained_vs_random_primary_aulc_gain"]["mean"] for row in task_rows]
                )
            ),
            "datasets_with_positive_mean_gain": sum(
                row["pretrained_vs_random_primary_aulc_gain"]["mean"] > 0.0 for row in task_rows
            ),
            "datasets_with_positive_k32_mean_gain": sum(
                row["pretrained_vs_random_k32_gain"]["mean"] > 0.0 for row in task_rows
            ),
        }
    return summary


def _summarize_full_context(records: list[dict[str, Any]], *, replicates: int) -> dict[str, Any]:
    """Summarize one dataset-specific all-train-row context point per split."""

    summary: dict[str, Any] = {}
    for dataset_index, dataset_id in enumerate(sorted({str(row["dataset_id"]) for row in records})):
        rows = [row for row in records if row["dataset_id"] == dataset_id]
        task = str(rows[0]["task"])
        classes = rows[0]["classes"]
        primary = "normalized_nll" if task == "classification" else "scaled_rmse"
        replicate_keys = sorted(
            {(int(row["checkpoint_seed"]), int(row["split_seed"])) for row in rows}
        )
        random_gains: list[float] = []
        shuffled_gains: list[float] = []
        accuracy_random_deltas: list[float] = []
        accuracy_shuffled_deltas: list[float] = []
        r2_random_deltas: list[float] = []
        r2_shuffled_deltas: list[float] = []
        paired_units: list[dict[str, Any]] = []
        for checkpoint_seed, split_seed in replicate_keys:
            selected = {
                str(row["arm"]): row
                for row in rows
                if row["checkpoint_seed"] == checkpoint_seed and row["split_seed"] == split_seed
            }
            pretrained = selected["pretrained_frozen"]
            random = selected["random_init_frozen"]
            shuffled = selected["pretrained_shuffled"]
            pretrained_primary = float(pretrained["metrics"][primary])
            random_primary = float(random["metrics"][primary])
            shuffled_primary = float(shuffled["metrics"][primary])
            random_gain = random_primary - pretrained_primary
            shuffled_gain = shuffled_primary - pretrained_primary
            random_gains.append(random_gain)
            shuffled_gains.append(shuffled_gain)
            unit: dict[str, Any] = {
                "checkpoint_seed": checkpoint_seed,
                "split_seed": split_seed,
                "context_rows": int(pretrained["context_size"]),
                "query_rows": int(pretrained["query_rows"]),
                "pretrained_primary": pretrained_primary,
                "random_primary": random_primary,
                "shuffled_primary": shuffled_primary,
                "pretrained_vs_random_primary_loss_gain": random_gain,
                "normal_vs_shuffled_primary_loss_gain": shuffled_gain,
            }
            if task == "classification":
                pretrained_accuracy = float(pretrained["metrics"]["accuracy"])
                random_accuracy = float(random["metrics"]["accuracy"])
                shuffled_accuracy = float(shuffled["metrics"]["accuracy"])
                accuracy_random_delta = pretrained_accuracy - random_accuracy
                accuracy_shuffled_delta = pretrained_accuracy - shuffled_accuracy
                accuracy_random_deltas.append(accuracy_random_delta)
                accuracy_shuffled_deltas.append(accuracy_shuffled_delta)
                unit.update(
                    {
                        "pretrained_accuracy": pretrained_accuracy,
                        "random_accuracy": random_accuracy,
                        "shuffled_accuracy": shuffled_accuracy,
                        "pretrained_vs_random_accuracy_delta": accuracy_random_delta,
                        "normal_vs_shuffled_accuracy_delta": accuracy_shuffled_delta,
                    }
                )
            else:
                pretrained_r2 = float(pretrained["metrics"]["r2"])
                random_r2 = float(random["metrics"]["r2"])
                shuffled_r2 = float(shuffled["metrics"]["r2"])
                r2_random_delta = pretrained_r2 - random_r2
                r2_shuffled_delta = pretrained_r2 - shuffled_r2
                r2_random_deltas.append(r2_random_delta)
                r2_shuffled_deltas.append(r2_shuffled_delta)
                unit.update(
                    {
                        "pretrained_scaled_mae": float(pretrained["metrics"]["scaled_mae"]),
                        "pretrained_r2": pretrained_r2,
                        "random_r2": random_r2,
                        "shuffled_r2": shuffled_r2,
                        "pretrained_vs_random_r2_delta": r2_random_delta,
                        "normal_vs_shuffled_r2_delta": r2_shuffled_delta,
                    }
                )
            paired_units.append(unit)

        dataset_summary: dict[str, Any] = {
            "task": task,
            "classes": classes,
            "evaluation_scope": "all_train_partition_rows_as_context",
            "primary_metric": primary,
            "paired_units": paired_units,
            "pretrained_vs_random_primary_loss_gain": dict(
                zip(
                    ("mean", "lower_95", "upper_95"),
                    paired_world_bootstrap(
                        random_gains,
                        replicates=replicates,
                        seed=1729 + dataset_index,
                    ),
                    strict=True,
                )
            )
            | {
                "paired_wins": sum(value > 0.0 for value in random_gains),
                "paired_total": len(random_gains),
            },
            "normal_vs_shuffled_primary_loss_gain": dict(
                zip(
                    ("mean", "lower_95", "upper_95"),
                    paired_world_bootstrap(
                        shuffled_gains,
                        replicates=replicates,
                        seed=2718 + dataset_index,
                    ),
                    strict=True,
                )
            )
            | {
                "paired_wins": sum(value > 0.0 for value in shuffled_gains),
                "paired_total": len(shuffled_gains),
            },
        }
        if task == "classification":
            dataset_summary["pretrained_vs_random_accuracy_delta"] = {
                "mean": float(np.mean(accuracy_random_deltas)),
                "paired_wins": sum(value > 0.0 for value in accuracy_random_deltas),
                "paired_total": len(accuracy_random_deltas),
            }
            dataset_summary["normal_vs_shuffled_accuracy_delta"] = {
                "mean": float(np.mean(accuracy_shuffled_deltas)),
                "paired_wins": sum(value > 0.0 for value in accuracy_shuffled_deltas),
                "paired_total": len(accuracy_shuffled_deltas),
            }
        else:
            dataset_summary["pretrained_vs_random_r2_delta"] = {
                "mean": float(np.mean(r2_random_deltas)),
                "paired_wins": sum(value > 0.0 for value in r2_random_deltas),
                "paired_total": len(r2_random_deltas),
            }
            dataset_summary["normal_vs_shuffled_r2_delta"] = {
                "mean": float(np.mean(r2_shuffled_deltas)),
                "paired_wins": sum(value > 0.0 for value in r2_shuffled_deltas),
                "paired_total": len(r2_shuffled_deltas),
            }
        summary[dataset_id] = dataset_summary
    return summary


def _summarize(
    records: list[dict[str, Any]], *, context_policy: RealContextPolicy, replicates: int
) -> dict[str, Any]:
    if context_policy == FULL_CONTEXT_POLICY:
        return _summarize_full_context(records, replicates=replicates)
    return _summarize_low_shot(records, replicates=replicates)


def _run_real_frozen_arm(
    model: torch.nn.Module,
    *,
    arm: str,
    checkpoint_seed: int,
    dataset_ids: Sequence[str],
    split_seeds: Sequence[int],
    splits: dict[tuple[str, int], PreparedRealIclSplit],
    context_policy: RealContextPolicy,
    shuffled_context: bool,
    query_chunk_rows: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, str | bool]]:
    """Evaluate one real-data arm with state hashes adjacent to the full invocation."""

    model.requires_grad_(False)
    model.eval()
    before = _state_hash(model)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for dataset_id in dataset_ids:
            for split_seed in split_seeds:
                split = splits[(dataset_id, split_seed)]
                context_sizes = (
                    (len(split.train_indices),) if context_policy == FULL_CONTEXT_POLICY else K_GRID
                )
                for context_size in context_sizes:
                    metrics = _evaluate(
                        model,
                        split,
                        context_size=context_size,
                        shuffled_context=shuffled_context,
                        context_policy=context_policy,
                        query_chunk_rows=query_chunk_rows,
                        device=device,
                    )
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "dataset_sha256": split.dataset.content_hash,
                            "dataset_source": split.dataset.source,
                            "task": split.dataset.task,
                            "classes": split.classes,
                            "checkpoint_seed": checkpoint_seed,
                            "split_seed": split_seed,
                            "arm": arm,
                            "context_size": context_size,
                            "context_policy": context_policy,
                            "train_rows_total": len(split.train_indices),
                            "full_context": context_size == len(split.train_indices),
                            "query_rows": len(split.query_indices),
                            "predictor_count": split.features.shape[1],
                            "selected_feature_indices": split.feature_indices.tolist(),
                            "split_manifest": real_icl_split_manifest(split),
                            "context_class_count": (
                                len(np.unique(split.response[split.context_order[:context_size]]))
                                if split.dataset.task == "classification" and context_size
                                else 0
                            ),
                            "metrics": metrics,
                        }
                    )
    after = _state_hash(model)
    return records, {"before": before, "after": after, "unchanged": before == after}


def _load_real_icl_panel(
    config: RealIclConfig,
    *,
    openml_fetcher: FetchOpenML | None,
    openml_sklearn_version: str | None,
) -> tuple[dict[str, RealDataset], dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Load old6, strict new6, or the exploratory cached OpenML panel."""

    if config.panel_manifest_path is None:
        datasets = {dataset_id: load_real_dataset(dataset_id) for dataset_id in config.dataset_ids}
        provenance = {
            dataset_id: {
                "provider": "sklearn_builtin_or_explicit_fetch",
                "source": dataset.source,
                "real_dataset_content_sha256": dataset.content_hash,
            }
            for dataset_id, dataset in datasets.items()
        }
        return datasets, None, provenance

    if is_cached_openml_panel_manifest(config.panel_manifest_path):
        panel = load_cached_openml_panel_manifest(config.panel_manifest_path)
        missing = set(config.dataset_ids) - set(panel.dataset_ids)
        if missing:
            raise RuntimeError(
                f"requested datasets are absent from the cached OpenML panel: {missing!r}"
            )
        fetched = [
            fetch_cached_openml_dataset(dataset_id, panel_manifest=panel)
            for dataset_id in config.dataset_ids
        ]
        datasets = {item.spec.dataset_id: item.dataset for item in fetched}
        provenance = {
            item.spec.dataset_id: {
                "source_manifest_sha256": item.source_manifest_sha256,
                "source_manifest": item.source_manifest,
            }
            for item in fetched
        }
        materialization_body = build_cached_openml_materialization_manifest(panel, fetched)
        panel_receipt = {
            "path": str(panel.path),
            "panel_id": CACHED_OPENML_PANEL_ID,
            "schema_version": panel.payload["schema_version"],
            "context_policy": panel.context_policy,
            "file_sha256": panel.file_sha256,
            "canonical_payload_sha256": panel.canonical_payload_sha256,
            "registered_dataset_ids": list(panel.dataset_ids),
            "evaluated_dataset_ids": list(config.dataset_ids),
            "materialization_manifest_sha256": materialization_body["manifest_sha256"],
            "materialization_manifest": materialization_body,
        }
        return datasets, panel_receipt, provenance

    panel = load_openml_new6_panel_manifest(config.panel_manifest_path)
    missing = set(config.dataset_ids) - set(panel.dataset_ids)
    if missing:  # Defensive: config validation and manifest validation should catch this first.
        raise RuntimeError(f"requested datasets are absent from the OpenML new6 panel: {missing!r}")
    fetched = [
        fetch_openml_new6_dataset(
            dataset_id,
            fetcher=openml_fetcher,
            cache=config.openml_cache,
            data_home=config.openml_data_home,
            sklearn_version=openml_sklearn_version,
        )
        for dataset_id in config.dataset_ids
    ]
    datasets = {item.spec.dataset_id: item.dataset for item in fetched}
    provenance = {
        item.spec.dataset_id: {
            "source_manifest_sha256": item.source_manifest_sha256,
            "source_manifest": item.source_manifest,
        }
        for item in fetched
    }
    materialization_body: dict[str, Any] = {
        "schema_version": "tabu.tabubase-openml-new6-evaluation-materialization.v1",
        "panel_id": OPENML_NEW6_PANEL_ID,
        "panel_manifest_file_sha256": panel.file_sha256,
        "panel_manifest_canonical_payload_sha256": panel.canonical_payload_sha256,
        "evaluated_dataset_ids": list(config.dataset_ids),
        "datasets": [
            {
                "dataset_id": dataset_id,
                "source_manifest_sha256": provenance[dataset_id]["source_manifest_sha256"],
                "materialized_array_sha256": provenance[dataset_id]["source_manifest"][
                    "materialized"
                ]["array_sha256"],
            }
            for dataset_id in config.dataset_ids
        ],
    }
    panel_receipt = {
        "path": str(panel.path),
        "panel_id": OPENML_NEW6_PANEL_ID,
        "schema_version": panel.payload["schema_version"],
        "context_policy": panel.context_policy,
        "file_sha256": panel.file_sha256,
        "canonical_payload_sha256": panel.canonical_payload_sha256,
        "registered_dataset_ids": list(panel.dataset_ids),
        "evaluated_dataset_ids": list(config.dataset_ids),
        "materialization_manifest_sha256": canonical_hash(materialization_body),
        "materialization_manifest": materialization_body,
    }
    return datasets, panel_receipt, provenance


def run_real_frozen_icl(
    config: RealIclConfig,
    *,
    device: torch.device,
    openml_fetcher: FetchOpenML | None = None,
    openml_sklearn_version: str | None = None,
) -> dict[str, Any]:
    """Run the real-data ICL panel without constructing any optimizer."""

    config.validate()
    started = time.monotonic()
    datasets, panel_manifest, dataset_provenance = _load_real_icl_panel(
        config,
        openml_fetcher=openml_fetcher,
        openml_sklearn_version=openml_sklearn_version,
    )
    splits = {
        (dataset_id, split_seed): prepare_real_icl_split(
            dataset, split_seed=split_seed, query_limit=config.query_limit
        )
        for dataset_id, dataset in datasets.items()
        for split_seed in config.split_seeds
    }
    records: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    parameter_hashes: dict[str, dict[str, str | bool]] = {}
    per_arm_parameter_hashes: dict[str, dict[str, dict[str, str | bool]]] = {}
    for checkpoint_seed in config.checkpoint_seeds:
        checkpoint = _checkpoint_path(config, checkpoint_seed)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen ICL checkpoint: {checkpoint}")
        models = {
            arm: build_tabubase_scale_model(
                seed=checkpoint_seed,
                device=device,
                nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
                nominal_codebook_size=config.nominal_codebook_size,
                nominal_codebook_seed=config.nominal_codebook_seed,
            )
            for arm in FROZEN_ARMS
        }
        load_pretrain_checkpoint(models["pretrained_frozen"], checkpoint)
        load_pretrain_checkpoint(models["pretrained_shuffled"], checkpoint)
        checkpoint_arm_hashes: dict[str, dict[str, str | bool]] = {}
        for arm, shuffled_context in (
            ("pretrained_frozen", False),
            ("random_init_frozen", False),
            ("pretrained_shuffled", True),
        ):
            arm_records, checkpoint_arm_hashes[arm] = _run_real_frozen_arm(
                models[arm],
                arm=arm,
                checkpoint_seed=checkpoint_seed,
                dataset_ids=config.dataset_ids,
                split_seeds=config.split_seeds,
                splits=splits,
                context_policy=config.context_policy,
                shuffled_context=shuffled_context,
                query_chunk_rows=config.query_chunk_rows,
                device=device,
            )
            records.extend(arm_records)
        per_arm_parameter_hashes[str(checkpoint_seed)] = checkpoint_arm_hashes
        pretrained_hashes = checkpoint_arm_hashes["pretrained_frozen"]
        random_hashes = checkpoint_arm_hashes["random_init_frozen"]
        shuffled_hashes = checkpoint_arm_hashes["pretrained_shuffled"]
        parameter_hashes[str(checkpoint_seed)] = {
            "pretrained_before": str(pretrained_hashes["before"]),
            "pretrained_after": str(pretrained_hashes["after"]),
            "pretrained_unchanged": bool(pretrained_hashes["unchanged"]),
            "random_before": str(random_hashes["before"]),
            "random_after": str(random_hashes["after"]),
            "random_unchanged": bool(random_hashes["unchanged"]),
            "pretrained_shuffled_before": str(shuffled_hashes["before"]),
            "pretrained_shuffled_after": str(shuffled_hashes["after"]),
            "pretrained_shuffled_unchanged": bool(shuffled_hashes["unchanged"]),
        }
        checkpoints.append(
            {
                "seed": checkpoint_seed,
                "path": str(checkpoint),
                "sha256": _sha256_file(checkpoint),
            }
        )
    summary = _summarize(
        records,
        context_policy=config.context_policy,
        replicates=config.bootstrap_replicates,
    )
    all_frozen_arm_parameter_hashes_unchanged = all(
        bool(arm_hashes["unchanged"])
        for checkpoint_hashes in per_arm_parameter_hashes.values()
        for arm_hashes in checkpoint_hashes.values()
    )
    receipt: dict[str, Any] = {
        "schema_version": (
            REAL_FULL_CONTEXT_SCHEMA
            if config.context_policy == FULL_CONTEXT_POLICY
            else REAL_LOW_SHOT_SCHEMA
        ),
        "status": "local_unissued",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": "supervised.label_broadcast.v1",
        "tokenizer_version": "cell-tokenizer.v2",
        "nominal_codebook_size": config.nominal_codebook_size,
        "nominal_codebook_seed": config.nominal_codebook_seed,
        "datasets": list(config.dataset_ids),
        "dataset_hashes": {key: value.content_hash for key, value in datasets.items()},
        "panel_manifest": panel_manifest,
        "openml_data_home": (
            str(config.openml_data_home.expanduser().resolve())
            if config.openml_data_home is not None
            else None
        ),
        "dataset_provenance": dataset_provenance,
        "checkpoint_seeds": list(config.checkpoint_seeds),
        "split_seeds": list(config.split_seeds),
        "context_policy": config.context_policy,
        "context_sizes": (None if config.context_policy == FULL_CONTEXT_POLICY else list(K_GRID)),
        "context_rows_by_dataset_split": {
            dataset_id: {
                str(split_seed): len(splits[(dataset_id, split_seed)].train_indices)
                for split_seed in config.split_seeds
            }
            for dataset_id in config.dataset_ids
        },
        "split_manifests": {
            dataset_id: {
                str(split_seed): real_icl_split_manifest(splits[(dataset_id, split_seed)])
                for split_seed in config.split_seeds
            }
            for dataset_id in config.dataset_ids
        },
        "query_limit": config.query_limit,
        "query_policy": (
            "all_heldout_rows" if config.query_limit is None else "deterministic_limited_subset"
        ),
        "query_chunk_rows": config.query_chunk_rows,
        "query_chunk_semantics": (
            "response_readout_only_after_one_full_transductive_evidence_episode"
            if config.context_policy == FULL_CONTEXT_POLICY
            else "separate_low_shot_query_evidence_chunks"
        ),
        "query_readout_chunk_rows": config.query_chunk_rows,
        "query_evidence_policy": (
            "all_heldout_predictors_single_transductive_episode"
            if config.context_policy == FULL_CONTEXT_POLICY
            else "historical_deterministic_query_chunks"
        ),
        "arms": list(FROZEN_ARMS),
        "optimizer_created": False,
        "frozen_arm_optimizer_created": False,
        "per_arm_parameter_hashes": per_arm_parameter_hashes,
        "all_frozen_arm_parameter_hashes_unchanged": (all_frozen_arm_parameter_hashes_unchanged),
        "parameter_hashes": parameter_hashes,
        "all_parameter_hashes_unchanged": all_frozen_arm_parameter_hashes_unchanged,
        "checkpoints": checkpoints,
        "summary": summary,
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "runtime_backend": os.environ.get("WEHUB_RUNTIME_BACKEND"),
            "runtime_image": os.environ.get("WEHUB_RUNTIME_IMAGE"),
        },
        "git_commit": _source_commit(),
        "source_tree_sha256": _source_tree_hash(),
        "source_status": "local_unissued_mirrored_worktree_not_clean_commit",
        "claim_boundary": (
            "optimizer-free real-data full-context ICL diagnostic; every train-partition row is "
            "visible as labeled context and every held-out row is query-only; sklearn/OpenML "
            "snapshots and local split receipts are not reviewed formal data authority; no "
            "fine-tuning, SOTA, or foundation-model claim"
            if config.context_policy == FULL_CONTEXT_POLICY
            else "optimizer-free real-data low-shot ICL diagnostic only; K<=32 results do not "
            "measure all-train-row context performance; no fine-tuning, SOTA, or "
            "foundation-model claim"
        ),
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt | {
        "result_path": str(config.output_path),
        "result_sha256": _sha256_file(config.output_path),
    }


__all__ = [
    "DEFAULT_REAL_ICL_DATASETS",
    "FULL_CONTEXT_POLICY",
    "LOW_SHOT_CONTEXT_POLICY",
    "REAL_FULL_CONTEXT_SCHEMA",
    "REAL_LOW_SHOT_SCHEMA",
    "PreparedRealIclSplit",
    "RealIclConfig",
    "build_real_icl_episode",
    "prepare_real_icl_split",
    "real_icl_split_manifest",
    "run_real_frozen_icl",
]
