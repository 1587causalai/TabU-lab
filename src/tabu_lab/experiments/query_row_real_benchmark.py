"""Stage-4 scratch-only real-data diagnostic for the TabUR row family.

This runner intentionally accepts no checkpoint.  By default it trains a fresh
TabUR instance with the canonical real train/test split and evaluates every
held-out row; finite context/query limits are explicit bounded diagnostics.
Results are local, unissued diagnostics, not a benchmark receipt or transfer
claim.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch

from tabu_lab.contracts import TruthSidecar
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig
from tabu_lab.training.objective import Objective

from .tabubase_real_benchmark import (
    PreparedRealTask,
    _real_episode,
    evaluation_context_indices,
    load_real_dataset,
    prepare_real_task,
    training_episode_indices,
)
from .tabubase_scale import resolve_device
from .query_row_real_coordinates import (
    numeric_raw_prediction_from_public,
    query_row_real_regression_loss,
    task_scale_to_raw,
)

TaskKind = Literal["classification", "regression"]


@dataclass(frozen=True, slots=True)
class QueryRowRealDatasetResult:
    dataset_id: str
    task: TaskKind
    status: str
    model_metrics: dict[str, float]
    baseline_metrics: dict[str, dict[str, float]]
    label_budget: int | None
    updates: int
    test_rows: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRowRealBenchmarkResult:
    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    model_spec_hash: str
    row_token_count: int
    device: str
    seed: int
    datasets: tuple[QueryRowRealDatasetResult, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["datasets"] = [item.as_dict() for item in self.datasets]
        return payload


def _train_one(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    evidence: Any,
    truth: TruthSidecar,
    device: torch.device,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model._forward_dense(evidence.to(device), emit_trace=False)
    if evidence.feature_specs[-1].kind.value == "numeric":
        loss = query_row_real_regression_loss(prediction, truth.to(device)).total
    else:
        loss = Objective()(prediction, truth.to(device)).total
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite TabUR real scratch loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().item())


def _model_prediction(
    model: torch.nn.Module,
    task: PreparedRealTask,
) -> tuple[np.ndarray, np.ndarray]:
    context = evaluation_context_indices(task)
    query = task.test_indices
    evidence, _ = _real_episode(
        task,
        context_indices=context,
        query_indices=query,
        episode_id=f"{task.dataset.dataset_id}-tabur-row-scratch-test",
    )
    model.eval()
    with torch.no_grad():
        prediction = model(evidence)
    query_start = len(context)
    if task.dataset.task == "classification":
        distribution = prediction["distribution"]
        if distribution.ndim == 3:
            probabilities = distribution[query_start:, -1, :].detach().cpu().numpy()
        elif distribution.ndim == 4:
            probabilities = distribution[0, query_start:, -1, :].detach().cpu().numpy()
        else:
            raise RuntimeError("unexpected TabUR classification distribution shape")
        return probabilities, task.response[query]
    values = numeric_raw_prediction_from_public(prediction)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    predicted_task_scale = values[0, query_start:, -1].detach().cpu().numpy()
    predicted_raw = task_scale_to_raw(
        predicted_task_scale,
        response_mean=task.response_mean,
        response_scale=task.response_scale,
    )
    truth_raw = task.dataset.response[query]
    return predicted_raw, truth_raw


def _metrics(
    task: PreparedRealTask,
    predicted: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float]:
    if task.dataset.task == "classification":
        from sklearn.metrics import accuracy_score, log_loss

        probabilities = np.asarray(predicted, dtype=np.float64)
        probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
        return {
            "accuracy": float(accuracy_score(truth, probabilities.argmax(axis=1))),
            "log_loss": float(
                log_loss(truth, probabilities, labels=np.arange(probabilities.shape[1]))
            ),
        }
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    return {
        "rmse": float(math.sqrt(mean_squared_error(truth, predicted))),
        "mae": float(mean_absolute_error(truth, predicted)),
    }


def _baselines(task: PreparedRealTask) -> dict[str, dict[str, float]]:
    truth = task.dataset.response[task.test_indices]
    if task.dataset.task == "classification":
        classes = np.arange(int(np.max(task.response[task.train_indices])) + 1)
        counts = np.bincount(
            task.response[task.label_indices].astype(np.int64),
            minlength=len(classes),
        )
        majority = int(np.argmax(counts))
        majority_probabilities = np.full((len(truth), len(classes)), 1e-6, dtype=np.float64)
        majority_probabilities[:, majority] = 1.0 - 1e-6 * (len(classes) - 1)
        uniform_probabilities = np.full(
            (len(truth), len(classes)),
            1.0 / len(classes),
            dtype=np.float64,
        )
        return {
            "majority": _metrics(task, majority_probabilities, truth),
            "uniform": _metrics(task, uniform_probabilities, truth),
        }
    train_mean = float(task.dataset.response[task.label_indices].mean())
    return {
        "train_mean": _metrics(
            task,
            np.full(len(truth), train_mean, dtype=np.float64),
            truth,
        )
    }


def run_query_row_real_scratch_benchmark(
    *,
    dataset_ids: tuple[str, ...] = ("iris", "wine", "diabetes"),
    label_budget: int | None = None,
    updates: int = 20,
    learning_rate: float = 3.0e-4,
    test_limit: int | None = None,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 1729,
) -> QueryRowRealBenchmarkResult:
    """Run Stage-4 without loading or accepting any checkpoint."""

    if not dataset_ids:
        raise ValueError("dataset_ids must not be empty")
    if label_budget is not None and label_budget <= 0:
        raise ValueError("label_budget must be positive or None")
    if updates <= 0:
        raise ValueError("updates must be positive")
    if test_limit is not None and test_limit <= 0:
        raise ValueError("test_limit must be positive or None")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    torch.manual_seed(seed)
    records: list[QueryRowRealDatasetResult] = []
    model_spec_hash = ""
    for offset, dataset_id in enumerate(dataset_ids):
        task = prepare_real_task(
            load_real_dataset(dataset_id),
            budget=label_budget,
            seed=seed + offset,
            test_limit=test_limit,
        )
        model = build_model(
            "tabu.query.row",
            config=ReferenceConfig(
                d_model=8,
                n_heads=2,
                d_ff=16,
                n_blocks=1,
                inducing_slots=2,
                matched_slots=row_token_count,
                max_features=256,
            ),
            profile="supervised.label_broadcast.v1",
            row_token_count=row_token_count,
        ).to(resolved_device)
        model_spec_hash = model.model_spec_hash
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
        for update in range(updates):
            context, query = training_episode_indices(task, seed=seed + offset, update=update)
            evidence, truth = _real_episode(
                task,
                context_indices=context,
                query_indices=query,
                episode_id=f"{dataset_id}-tabur-row-scratch-train-{update:04d}",
            )
            _train_one(model, optimizer, evidence, truth, resolved_device)
        predicted, truth_values = _model_prediction(model, task)
        model_metrics = _metrics(task, predicted, truth_values)
        baseline_metrics = _baselines(task)
        records.append(
            QueryRowRealDatasetResult(
                dataset_id=dataset_id,
                task=task.dataset.task,
                status=(
                    "pass"
                    if all(math.isfinite(value) for value in model_metrics.values())
                    else "kill"
                ),
                model_metrics=model_metrics,
                baseline_metrics=baseline_metrics,
                label_budget=label_budget,
                updates=updates,
                test_rows=len(task.test_indices),
                seed=seed + offset,
            )
        )
    status = "pass" if all(item.status == "pass" for item in records) else "kill"
    return QueryRowRealBenchmarkResult(
        status=status,
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR scratch-only real-data diagnostic on the canonical full train/test split "
            "by default; finite label/test limits are explicit bounded overrides. No checkpoint transfer, "
            "frozen ICL, fine-tuning lift, benchmark, or accepted claim"
        ),
        model_id="tabu.query.row",
        contract_version="0.1.0",
        model_spec_hash=model_spec_hash,
        row_token_count=row_token_count,
        device=str(resolved_device),
        seed=seed,
        datasets=tuple(records),
    )


__all__ = [
    "QueryRowRealBenchmarkResult",
    "QueryRowRealDatasetResult",
    "run_query_row_real_scratch_benchmark",
]
