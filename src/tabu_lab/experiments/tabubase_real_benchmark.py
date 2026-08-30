"""Paired real-data transfer benchmark for scaled TabUBase checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
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

from .tabubase_response_readout import query_response_readout
from .tabubase_scale import (
    ROOT_SEEDS,
    _train_one,
    build_tabubase_scale_model,
    load_pretrain_checkpoint,
)

TaskKind = Literal["classification", "regression"]
QUERY_READOUT_SEMANTICS = "response_readout_only_after_one_full_transductive_evidence_episode"


def _source_tree_hash() -> str:
    root = Path(__file__).resolve().parents[3]
    paths = sorted((root / "src" / "tabu_lab").rglob("*.py"))
    paths += sorted((root / "specs" / "models").rglob("*.yaml"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RealDataset:
    dataset_id: str
    task: TaskKind
    features: np.ndarray
    response: np.ndarray
    source: str

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.dataset_id.encode())
        digest.update(self.task.encode())
        digest.update(np.ascontiguousarray(self.features).tobytes())
        digest.update(np.ascontiguousarray(self.response).tobytes())
        return digest.hexdigest()


def load_real_dataset(dataset_id: str, *, panel_manifest: Path | None = None) -> RealDataset:
    """Load one numeric real dataset.

    The historical sklearn panel remains the default.  An explicit cached
    OpenML panel manifest opts into the pinned ARFF loader; keeping this
    switch explicit prevents an accidental network fetch or source drift.
    """

    if panel_manifest is not None:
        from .tabubase_openml_cached import (
            CACHED_OPENML_BY_ID,
            fetch_cached_openml_dataset,
            is_cached_openml_panel_manifest,
            load_cached_openml_panel_manifest,
        )

        if not is_cached_openml_panel_manifest(panel_manifest):
            raise ValueError(
                "real benchmark panel_manifest must be the checked-in cached OpenML panel"
            )
        panel = load_cached_openml_panel_manifest(panel_manifest)
        if dataset_id not in CACHED_OPENML_BY_ID or dataset_id not in panel.dataset_ids:
            raise ValueError(f"dataset {dataset_id!r} is not in the cached OpenML panel")
        return fetch_cached_openml_dataset(
            dataset_id,
            panel_manifest=panel,
        ).dataset

    try:
        from sklearn import datasets
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("real benchmark requires scikit-learn") from exc
    loaders: dict[str, tuple[Any, TaskKind, str]] = {
        "iris": (datasets.load_iris, "classification", "sklearn.datasets.load_iris"),
        "wine": (datasets.load_wine, "classification", "sklearn.datasets.load_wine"),
        "breast_cancer": (
            datasets.load_breast_cancer,
            "classification",
            "sklearn.datasets.load_breast_cancer",
        ),
        "digits": (datasets.load_digits, "classification", "sklearn.datasets.load_digits"),
        "diabetes": (datasets.load_diabetes, "regression", "sklearn.datasets.load_diabetes"),
        "california_housing": (
            datasets.fetch_california_housing,
            "regression",
            "sklearn.datasets.fetch_california_housing",
        ),
    }
    if dataset_id not in loaders:
        raise ValueError(f"unknown real benchmark dataset: {dataset_id}")
    loader, task, source = loaders[dataset_id]
    bunch = loader()
    features = np.asarray(bunch.data, dtype=np.float32)
    response_dtype = np.int64 if task == "classification" else np.float32
    response = np.asarray(bunch.target, dtype=response_dtype)
    if features.ndim != 2 or response.ndim != 1 or len(features) != len(response):
        raise RuntimeError(f"dataset {dataset_id} has an unsupported shape")
    if not np.isfinite(features).all() or not np.isfinite(response).all():
        raise RuntimeError(f"dataset {dataset_id} contains non-finite values")
    return RealDataset(dataset_id, task, features, response, source)


def _split_indices(dataset: RealDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260829)
    strata = tuple(np.unique(dataset.response)) if dataset.task == "classification" else (None,)
    partitions: list[list[int]] = [[], [], []]
    for stratum in strata:
        indices = (
            np.flatnonzero(dataset.response == stratum)
            if stratum is not None
            else np.arange(len(dataset.response))
        )
        indices = rng.permutation(indices)
        train_end = round(0.6 * len(indices))
        validation_end = train_end + round(0.2 * len(indices))
        partitions[0].extend(indices[:train_end].tolist())
        partitions[1].extend(indices[train_end:validation_end].tolist())
        partitions[2].extend(indices[validation_end:].tolist())
    return tuple(np.sort(np.asarray(partition, dtype=np.int64)) for partition in partitions)  # type: ignore[return-value]


def _label_subset(
    dataset: RealDataset,
    train_indices: np.ndarray,
    *,
    budget: int,
    seed: int,
) -> np.ndarray:
    if budget >= len(train_indices):
        return train_indices.copy()
    rng = np.random.default_rng(seed)
    if dataset.task == "regression":
        return np.sort(rng.choice(train_indices, size=budget, replace=False))
    classes = np.unique(dataset.response[train_indices])
    selected: list[int] = []
    queues = {
        int(label): rng.permutation(
            train_indices[dataset.response[train_indices] == label]
        ).tolist()
        for label in classes
    }
    while len(selected) < budget:
        for label in classes:
            queue = queues[int(label)]
            if queue and len(selected) < budget:
                selected.append(queue.pop())
    return np.sort(np.asarray(selected, dtype=np.int64))


def _stratified_sample(
    indices: np.ndarray,
    response: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
    reserve_per_class: int = 0,
) -> np.ndarray:
    """Sample a class-covered subset while reserving support for later roles."""

    indices = np.asarray(indices, dtype=np.int64)
    if size < 1 or size > len(indices):
        raise ValueError(f"sample size {size} is invalid for {len(indices)} rows")
    classes = np.unique(response[indices])
    if size < len(classes):
        raise ValueError(f"sample size {size} cannot cover all {len(classes)} response classes")
    queues: dict[int, list[int]] = {}
    for label in classes:
        class_indices = indices[response[indices] == label]
        available = len(class_indices) - reserve_per_class
        if available < 1:
            raise ValueError(
                f"class {int(label)} cannot be sampled while reserving {reserve_per_class} row(s)"
            )
        shuffled = rng.permutation(class_indices).tolist()
        queues[int(label)] = shuffled[:available]
    if sum(len(queue) for queue in queues.values()) < size:
        raise ValueError(f"cannot sample {size} rows while reserving {reserve_per_class} per class")

    # Seed every class before filling the rest round-robin. This makes class
    # coverage a construction invariant, not a high-probability accident.
    selected = [queues[int(label)].pop() for label in classes]
    while len(selected) < size:
        progressed = False
        for label in classes:
            queue = queues[int(label)]
            if queue and len(selected) < size:
                selected.append(queue.pop())
                progressed = True
        if not progressed:  # pragma: no cover - guarded by the capacity check
            raise RuntimeError("stratified sampler exhausted before reaching target size")
    return np.sort(np.asarray(selected, dtype=np.int64))


@dataclass(frozen=True, slots=True)
class PreparedRealTask:
    dataset: RealDataset
    features: np.ndarray
    response: np.ndarray
    response_mean: float
    response_scale: float
    train_indices: np.ndarray
    label_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def prepare_real_task(
    dataset: RealDataset,
    *,
    budget: int,
    seed: int,
    test_limit: int | None = 512,
) -> PreparedRealTask:
    train_indices, validation_indices, test_indices = _split_indices(dataset)
    label_indices = _label_subset(dataset, train_indices, budget=budget, seed=seed)
    mean = dataset.features[label_indices].mean(axis=0, keepdims=True)
    scale = dataset.features[label_indices].std(axis=0, keepdims=True)
    scale = np.maximum(scale, 1.0e-6)
    features = (dataset.features - mean) / scale
    if dataset.task == "regression":
        response_mean = float(dataset.response[label_indices].mean())
        response_scale = float(dataset.response[label_indices].std())
        response_scale = max(response_scale, 1.0e-6)
        response = (dataset.response - response_mean) / response_scale
    else:
        classes = np.unique(dataset.response[label_indices])
        if not np.array_equal(classes, np.arange(len(classes))):
            mapping = {int(value): index for index, value in enumerate(classes.tolist())}
            response = np.asarray(
                [mapping[int(value)] for value in dataset.response], dtype=np.int64
            )
        else:
            response = dataset.response.copy()
        response_mean, response_scale = 0.0, 1.0
    # All arms use the same bounded test projection to keep the dense episode
    # execution cost and metric row universe identical.
    if test_limit is not None and test_limit < 1:
        raise ValueError("test_limit must be positive or None")
    if test_limit is not None and len(test_indices) > test_limit:
        rng = np.random.default_rng(20260831)
        test_indices = np.sort(rng.choice(test_indices, size=test_limit, replace=False))
    return PreparedRealTask(
        dataset=dataset,
        features=np.asarray(features, dtype=np.float32),
        response=np.asarray(response),
        response_mean=response_mean,
        response_scale=response_scale,
        train_indices=train_indices,
        label_indices=label_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def _real_episode(
    task: PreparedRealTask,
    *,
    context_indices: np.ndarray,
    query_indices: np.ndarray,
    episode_id: str,
) -> tuple[EvidenceEpisode, TruthSidecar]:
    rows = np.concatenate((context_indices, query_indices))
    predictors = torch.from_numpy(task.features[rows])
    truth_response = torch.as_tensor(task.response[rows], dtype=torch.float32)
    query_start = len(context_indices)
    forward_response = truth_response.clone()
    forward_response[query_start:] = 0.0
    forward_values = torch.cat((predictors, forward_response.unsqueeze(1)), dim=1)
    feature_specs = tuple(
        FeatureSpec(name=f"feature_{index}") for index in range(predictors.shape[1])
    )
    if task.dataset.task == "classification":
        classes = int(np.max(task.response[task.train_indices])) + 1
        response_spec = FeatureSpec(
            name="response",
            kind=FeatureKind.CATEGORICAL,
            domain=tuple(f"class_{index}" for index in range(classes)),
            codebook_id=f"{task.dataset.dataset_id}-response-v1",
            role=FeatureRole.RESPONSE,
        )
    else:
        response_spec = FeatureSpec(name="response", role=FeatureRole.RESPONSE)
    feature_specs = (*feature_specs, response_spec)
    roles = torch.full(
        forward_values.shape,
        int(ForwardRole.RECEIVER | ForwardRole.SOURCE),
        dtype=torch.int64,
    )
    origins = torch.full(
        forward_values.shape,
        origin_code(OriginState.OBSERVED),
        dtype=torch.int64,
    )
    roles[query_start:, -1] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    origins[query_start:, -1] = origin_code(OriginState.QUERY)
    row_ids = tuple(f"{task.dataset.dataset_id}-row-{int(index)}" for index in rows)
    evidence = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id=task.dataset.dataset_id,
        source_partition="train",
        fit_partition="train",
        row_ids=row_ids,
        feature_names=tuple(spec.name for spec in feature_specs),
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={"statistics_scope": "label_subset_only"},
    )
    truth_values = torch.zeros_like(forward_values)
    truth_values[query_start:, -1] = truth_response[query_start:]
    truth = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabubase-real-episode.v1",
                "dataset_id": task.dataset.dataset_id,
                "context_indices": context_indices.tolist(),
                "query_indices": query_indices.tolist(),
            }
        ),
        row_ids=row_ids,
        feature_names=evidence.feature_names,
        target_values=truth_values,
        target_mask=evidence.target_mask,
    )
    return evidence, truth


def _sample_training_episode(
    task: PreparedRealTask,
    *,
    seed: int,
    update: int,
) -> tuple[EvidenceEpisode, TruthSidecar]:
    context, query = training_episode_indices(task, seed=seed, update=update)
    return _real_episode(
        task,
        context_indices=context,
        query_indices=query,
        episode_id=f"{task.dataset.dataset_id}-train-{seed}-{update:04d}",
    )


def training_episode_indices(
    task: PreparedRealTask,
    *,
    seed: int,
    update: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct disjoint context/query row sets for one fine-tuning episode."""

    rng = np.random.default_rng(seed * 1_000_003 + update)
    query_size = min(32, max(1, len(task.label_indices) // 4))
    context_size = min(64, len(task.label_indices) - query_size)
    if task.dataset.task == "classification":
        classes = np.unique(task.response[task.label_indices])
        if query_size < len(classes) or context_size < len(classes):
            raise ValueError(
                "classification episode budgets must cover every response class in "
                "both context and query"
            )
        # Query sampling reserves at least one row per class for context. The
        # second stratified draw then guarantees context support independently.
        query = _stratified_sample(
            task.label_indices,
            task.response,
            size=query_size,
            rng=rng,
            reserve_per_class=1,
        )
        remaining = np.setdiff1d(task.label_indices, query, assume_unique=True)
        context = _stratified_sample(
            remaining,
            task.response,
            size=context_size,
            rng=rng,
        )
    else:
        permutation = rng.permutation(task.label_indices)
        query = np.sort(permutation[:query_size])
        context = np.sort(permutation[query_size : query_size + context_size])
    if np.intersect1d(context, query).size:
        raise RuntimeError("training context and query rows must be disjoint")
    return context, query


def evaluation_context_indices(task: PreparedRealTask) -> np.ndarray:
    """Use the complete labeled budget as inference support, with coverage checks."""

    context = task.label_indices.copy()
    if task.dataset.task == "classification":
        expected = np.unique(task.response[task.train_indices])
        observed = np.unique(task.response[context])
        if not np.array_equal(observed, expected):
            missing = np.setdiff1d(expected, observed).tolist()
            raise ValueError(
                f"evaluation context lacks response-class support for classes {missing}"
            )
    return context


def fine_tune_tabubase(
    task: PreparedRealTask,
    *,
    seed: int,
    device: torch.device,
    updates: int,
    learning_rate: float,
    checkpoint: Path | None,
    nominal_tokenizer: str = "episode_random_sphere",
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model = build_tabubase_scale_model(
        seed=seed,
        device=device,
        nominal_tokenizer=nominal_tokenizer,
        nominal_codebook_size=nominal_codebook_size,
        nominal_codebook_seed=nominal_codebook_seed,
    )
    arm = "pretrained" if checkpoint is not None else "scratch"
    if checkpoint is not None:
        load_pretrain_checkpoint(model, checkpoint)
    initial_hash = hashlib.sha256(
        b"".join(value.detach().cpu().numpy().tobytes() for value in model.state_dict().values())
    ).hexdigest()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    losses: list[float] = []
    started = time.monotonic()
    for update in range(updates):
        evidence, truth = _sample_training_episode(task, seed=seed, update=update)
        loss_value = _train_one(
            model,
            optimizer,
            evidence,
            truth,
            device=device,
            gradient_clip_norm=1.0,
        )
        if update == 0 or (update + 1) % 50 == 0:
            losses.append(loss_value)
    return model, {
        "arm": arm,
        "initial_parameter_sha256": initial_hash,
        "updates": updates,
        "learning_rate": learning_rate,
        "loss_history": losses,
        "elapsed_seconds": time.monotonic() - started,
    }


def evaluate_tabubase(
    model: torch.nn.Module,
    task: PreparedRealTask,
    *,
    device: torch.device,
    query_readout_chunk_rows: int = 64,
) -> dict[str, float]:
    return evaluate_tabubase_on_indices(
        model,
        task,
        device=device,
        query_indices=task.test_indices,
        query_partition="test",
        query_readout_chunk_rows=query_readout_chunk_rows,
    )


def evaluate_tabubase_on_indices(
    model: torch.nn.Module,
    task: PreparedRealTask,
    *,
    device: torch.device,
    query_indices: np.ndarray,
    query_partition: Literal["validation", "test"],
    query_readout_chunk_rows: int = 64,
) -> dict[str, float]:
    """Evaluate one partition in one transductive episode with bounded readout only."""

    context = evaluation_context_indices(task)
    if len(query_indices) < 1:
        raise ValueError("query_indices must contain at least one row")
    evidence, _ = _real_episode(
        task,
        context_indices=context,
        query_indices=query_indices,
        episode_id=f"{task.dataset.dataset_id}-{query_partition}",
    )
    model.eval()
    with torch.no_grad():
        readout = query_response_readout(
            model,
            evidence.to(device),
            context_rows=len(context),
            query_readout_chunk_rows=query_readout_chunk_rows,
        )
    if task.dataset.task == "classification":
        if readout.probabilities is None:
            raise RuntimeError("TabUBase omitted a classification distribution")
        predicted = readout.probabilities[0].detach().cpu().numpy()
    else:
        if readout.numeric_values is None:
            raise RuntimeError("TabUBase omitted a numeric prediction")
        predicted = readout.numeric_values[0].detach().cpu().numpy()
    truth = task.response[query_indices]
    if task.dataset.task == "classification":
        from sklearn.metrics import accuracy_score, log_loss

        probabilities = predicted / np.maximum(predicted.sum(axis=1, keepdims=True), 1.0e-12)
        return {
            "accuracy": float(accuracy_score(truth, probabilities.argmax(axis=1))),
            "log_loss": float(
                log_loss(truth, probabilities, labels=np.arange(probabilities.shape[1]))
            ),
        }
    return _regression_metrics(task, query_indices, predicted)


def _regression_metrics(
    task: PreparedRealTask,
    query_indices: np.ndarray,
    predicted_scaled: np.ndarray,
) -> dict[str, float]:
    """Return common raw and train-scale regression metrics for one test set."""

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    predicted_scaled = np.asarray(predicted_scaled, dtype=np.float64)
    truth_scaled = np.asarray(task.response[query_indices], dtype=np.float64)
    truth_raw = np.asarray(task.dataset.response[query_indices], dtype=np.float64)
    predicted_raw = (
        predicted_scaled * float(task.response_scale) + float(task.response_mean)
    )
    scaled_rmse = float(math.sqrt(mean_squared_error(truth_scaled, predicted_scaled)))
    scaled_mae = float(mean_absolute_error(truth_scaled, predicted_scaled))
    return {
        "rmse": float(math.sqrt(mean_squared_error(truth_raw, predicted_raw))),
        "mae": float(mean_absolute_error(truth_raw, predicted_raw)),
        "scaled_rmse": scaled_rmse,
        "scaled_mae": scaled_mae,
        "nrmse": scaled_rmse,
        "r2": float(r2_score(truth_raw, predicted_raw)),
    }


def temperature_scale_probabilities(
    probabilities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Soften or sharpen probabilities without changing their class ranking."""

    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("classification probabilities must be a rank-2 class matrix")
    values = values / np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)
    logits = np.log(np.clip(values, 1.0e-12, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def evaluate_temperature_calibration(
    model: torch.nn.Module,
    task: PreparedRealTask,
    *,
    device: torch.device,
    temperatures: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    query_readout_chunk_rows: int = 64,
) -> dict[str, Any]:
    """Select one temperature on validation and evaluate it once on test."""

    if task.dataset.task != "classification":
        raise ValueError("temperature calibration is classification-only")
    from sklearn.metrics import accuracy_score, log_loss

    def probabilities(indices: np.ndarray, partition: Literal["validation", "test"]) -> np.ndarray:
        context = evaluation_context_indices(task)
        if len(indices) < 1:
            raise ValueError(f"{partition} indices must contain at least one row")
        evidence, _ = _real_episode(
            task,
            context_indices=context,
            query_indices=indices,
            episode_id=f"{task.dataset.dataset_id}-{partition}-cal",
        )
        model.eval()
        with torch.no_grad():
            readout = query_response_readout(
                model,
                evidence.to(device),
                context_rows=len(context),
                query_readout_chunk_rows=query_readout_chunk_rows,
            )
        if readout.probabilities is None:
            raise RuntimeError("TabUBase omitted a classification distribution")
        values = readout.probabilities[0].cpu().numpy()
        return values / np.maximum(values.sum(axis=1, keepdims=True), 1.0e-12)

    validation_probabilities = probabilities(task.validation_indices, "validation")
    validation_truth = task.response[task.validation_indices]
    candidates = []
    for temperature in temperatures:
        scaled = temperature_scale_probabilities(validation_probabilities, temperature)
        candidates.append(
            {
                "temperature": temperature,
                "validation_log_loss": float(
                    log_loss(
                        validation_truth,
                        scaled,
                        labels=np.arange(scaled.shape[1]),
                    )
                ),
            }
        )
    selected = min(candidates, key=lambda item: (item["validation_log_loss"], item["temperature"]))
    test_probabilities = temperature_scale_probabilities(
        probabilities(task.test_indices, "test"),
        float(selected["temperature"]),
    )
    test_truth = task.response[task.test_indices]
    return {
        "selection_partition": "validation",
        "temperatures": list(temperatures),
        "candidates": candidates,
        "selected_temperature": selected["temperature"],
        "selected_validation_log_loss": selected["validation_log_loss"],
        "test_metrics": {
            "accuracy": float(accuracy_score(test_truth, test_probabilities.argmax(axis=1))),
            "log_loss": float(
                log_loss(
                    test_truth,
                    test_probabilities,
                    labels=np.arange(test_probabilities.shape[1]),
                )
            ),
        },
    }


def evaluate_classical_baselines(
    task: PreparedRealTask,
    *,
    seed: int,
) -> dict[str, dict[str, float]]:
    try:
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, log_loss
        from sklearn.neural_network import MLPClassifier, MLPRegressor
    except ImportError as exc:  # pragma: no cover - optional runtime dependencies
        raise RuntimeError("baseline evaluation requires scikit-learn and xgboost") from exc
    x_train = task.features[task.label_indices]
    y_train = task.response[task.label_indices]
    x_test = task.features[task.test_indices]
    y_test = task.response[task.test_indices]
    results: dict[str, dict[str, float]] = {}
    if task.dataset.task == "classification":
        models = {
            "xgboost": xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=8,
                random_state=seed,
                eval_metric="logloss",
            ),
            "mlp": MLPClassifier(
                hidden_layer_sizes=(64, 64),
                alpha=1.0e-4,
                batch_size=min(64, len(y_train)),
                learning_rate_init=1.0e-3,
                max_iter=500,
                random_state=seed,
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)
            results[name] = {
                "accuracy": float(accuracy_score(y_test, probabilities.argmax(axis=1))),
                "log_loss": float(
                    log_loss(
                        y_test,
                        probabilities,
                        labels=np.arange(probabilities.shape[1]),
                    )
                ),
            }
    else:
        models = {
            "xgboost": xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=8,
                random_state=seed,
                objective="reg:squarederror",
            ),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 64),
                alpha=1.0e-4,
                batch_size=min(64, len(y_train)),
                learning_rate_init=1.0e-3,
                max_iter=500,
                random_state=seed,
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            results[name] = _regression_metrics(task, task.test_indices, predicted)
    return results


def run_real_benchmark(
    *,
    dataset_ids: tuple[str, ...],
    checkpoint_root: Path,
    output_path: Path,
    device: torch.device,
    budget: int = 128,
    updates: int = 400,
    learning_rate: float = 3.0e-4,
    checkpoint_phase: Literal["PT-S1", "PT-S2"] = "PT-S1",
    seeds: tuple[int, ...] = ROOT_SEEDS,
    nominal_tokenizer: str = "episode_random_sphere",
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
    temperature_calibration: bool = False,
    checkpoint_run_suffix: str = "",
    panel_manifest: Path | None = None,
    test_limit: int | None = 512,
    query_readout_chunk_rows: int = 64,
) -> dict[str, Any]:
    import sklearn
    import xgboost

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset = load_real_dataset(dataset_id, panel_manifest=panel_manifest)
        for seed in seeds:
            task = prepare_real_task(
                dataset,
                budget=budget,
                seed=seed,
                test_limit=test_limit,
            )
            checkpoint_update = 20_000 if checkpoint_phase == "PT-S1" else 200_000
            run_name = f"tabubase-{checkpoint_phase.lower()}-seed-{seed}"
            if nominal_tokenizer == "source_scoped_frozen_codebook.v2":
                run_name += (
                    f"-nominal-codebook-v2-b{nominal_codebook_size}-s{nominal_codebook_seed}"
                )
            if checkpoint_run_suffix:
                if not checkpoint_run_suffix.startswith("-") or "/" in checkpoint_run_suffix:
                    raise ValueError("checkpoint_run_suffix must be a safe leading-dash suffix")
                run_name += checkpoint_run_suffix
            checkpoint = (
                checkpoint_root / run_name / f"checkpoint-{checkpoint_update:05d}.safetensors"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(f"missing paired PT-S1 checkpoint: {checkpoint}")
            baselines = evaluate_classical_baselines(task, seed=seed)
            arms: dict[str, dict[str, Any]] = {}
            for arm, source in (("pretrained", checkpoint), ("scratch", None)):
                model, training = fine_tune_tabubase(
                    task,
                    seed=seed,
                    device=device,
                    updates=updates,
                    learning_rate=learning_rate,
                    checkpoint=source,
                    nominal_tokenizer=nominal_tokenizer,
                    nominal_codebook_size=nominal_codebook_size,
                    nominal_codebook_seed=nominal_codebook_seed,
                )
                arm_result: dict[str, Any] = {
                    "training": training,
                    "metrics": evaluate_tabubase(
                        model,
                        task,
                        device=device,
                        query_readout_chunk_rows=query_readout_chunk_rows,
                    ),
                }
                if temperature_calibration and dataset.task == "classification":
                    arm_result["temperature_calibration"] = evaluate_temperature_calibration(
                        model,
                        task,
                        device=device,
                        query_readout_chunk_rows=query_readout_chunk_rows,
                    )
                arms[arm] = arm_result
            results.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_source": dataset.source,
                    "dataset_sha256": dataset.content_hash,
                    "task": dataset.task,
                    "seed": seed,
                    "label_budget": len(task.label_indices),
                    "evaluation_context_rows": len(evaluation_context_indices(task)),
                    "evaluation_context_class_counts": (
                        {
                            str(int(label)): int(
                                np.sum(task.response[evaluation_context_indices(task)] == label)
                            )
                            for label in np.unique(task.response[task.train_indices])
                        }
                        if dataset.task == "classification"
                        else None
                    ),
                    "test_rows": len(task.test_indices),
                    "evaluation_estimand": QUERY_READOUT_SEMANTICS,
                    "query_readout_chunk_rows": query_readout_chunk_rows,
                    "evaluation_context_indices_sha256": canonical_hash(
                        evaluation_context_indices(task).tolist()
                    ),
                    "test_indices_sha256": canonical_hash(task.test_indices.tolist()),
                    "split_sha256": canonical_hash(
                        {
                            "schema": "tabubase-real-split.v1",
                            "train": task.train_indices.tolist(),
                            "validation": task.validation_indices.tolist(),
                            "test": task.test_indices.tolist(),
                        }
                    ),
                    "label_subset_sha256": canonical_hash(
                        {
                            "schema": "tabubase-real-label-subset.v1",
                            "seed": seed,
                            "rows": task.label_indices.tolist(),
                        }
                    ),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "tabubase": arms,
                    "baselines": baselines,
                }
            )
    summaries: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        rows = [row for row in results if row["dataset_id"] == dataset_id]
        task_kind = rows[0]["task"]
        primary = "log_loss" if task_kind == "classification" else "nrmse"

        def values(source_rows: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
            output: list[float] = []
            for row in source_rows:
                value: Any = row
                for key in path:
                    value = value[key]
                output.append(float(value))
            return output

        pretrained = values(rows, ("tabubase", "pretrained", "metrics", primary))
        scratch = values(rows, ("tabubase", "scratch", "metrics", primary))
        xgboost_values = values(rows, ("baselines", "xgboost", primary))
        mlp = values(rows, ("baselines", "mlp", primary))
        summaries.append(
            {
                "dataset_id": dataset_id,
                "task": task_kind,
                "primary_metric": primary,
                "lower_is_better": True,
                "means": {
                    "tabubase_pretrained": float(np.mean(pretrained)),
                    "tabubase_scratch": float(np.mean(scratch)),
                    "xgboost": float(np.mean(xgboost_values)),
                    "mlp": float(np.mean(mlp)),
                },
                "pretrained_seed_wins": {
                    "vs_scratch": sum(
                        left < right for left, right in zip(pretrained, scratch, strict=True)
                    ),
                    "vs_xgboost": sum(
                        left < right for left, right in zip(pretrained, xgboost_values, strict=True)
                    ),
                    "vs_mlp": sum(
                        left < right for left, right in zip(pretrained, mlp, strict=True)
                    ),
                    "denominator": len(rows),
                },
            }
        )
    payload = {
        "schema_version": "tabu.transfer-base-real-panel-local-unissued.v2",
        "status": "local_unissued",
        "datasets": list(dataset_ids),
        "seeds": list(seeds),
        "checkpoint_phase": checkpoint_phase,
        "label_budget": budget,
        "fine_tune_updates": updates,
        "learning_rate": learning_rate,
        "nominal_tokenizer": nominal_tokenizer,
        "nominal_codebook_size": nominal_codebook_size,
        "nominal_codebook_seed": nominal_codebook_seed,
        "temperature_calibration": temperature_calibration,
        "checkpoint_run_suffix": checkpoint_run_suffix,
        "panel_manifest": str(panel_manifest) if panel_manifest is not None else None,
        "test_limit": test_limit,
        "evaluation_estimand": QUERY_READOUT_SEMANTICS,
        "query_readout_chunk_rows": query_readout_chunk_rows,
        "results": results,
        "summaries": summaries,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "hostname": platform.node(),
            "physical_hostname": os.environ.get("WEHUB_PHYSICAL_HOST") or platform.node(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "device": str(device),
            "runtime_backend": os.environ.get("WEHUB_RUNTIME_BACKEND"),
            "runtime_image": os.environ.get("WEHUB_RUNTIME_IMAGE"),
            "extra_site_packages": os.environ.get("TABU_EXTRA_SITE_PACKAGES"),
        },
        "source_tree_sha256": _source_tree_hash(),
        "claim_boundary": "paired exploratory panel; no SOTA or accepted benchmark claim",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "QUERY_READOUT_SEMANTICS",
    "PreparedRealTask",
    "RealDataset",
    "evaluate_classical_baselines",
    "evaluate_tabubase",
    "evaluate_tabubase_on_indices",
    "evaluate_temperature_calibration",
    "evaluation_context_indices",
    "fine_tune_tabubase",
    "load_real_dataset",
    "prepare_real_task",
    "run_real_benchmark",
    "temperature_scale_probabilities",
    "training_episode_indices",
]
