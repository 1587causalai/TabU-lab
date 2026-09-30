"""Gate R3 corrected real-regression adaptation diagnosis.

This runner keeps the existing supervised synthetic prior fixed, evaluates only
on real validation rows, and records exact same-init scratch controls. It is a
local-unissued diagnostic; it does not create a capability claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabu_lab.experiments.query_row_finetune_lift import _parameter_sha256
from tabu_lab.experiments.query_row_identity import query_row_result_identity
from tabu_lab.experiments.query_row_pretraining import (
    read_query_row_pretrain_checkpoint_identity,
)
from tabu_lab.experiments.query_row_real_benchmark import _train_one
from tabu_lab.experiments.query_row_real_coordinates import task_scale_to_raw
from tabu_lab.experiments.query_row_transfer_common import (
    build_model_from_checkpoint,
    build_random_model,
)
from tabu_lab.experiments.tabubase_real_benchmark import (
    _real_episode,
    evaluation_context_indices,
    load_real_dataset,
    prepare_real_task,
    training_episode_indices,
)
from tabu_lab.experiments.tabubase_scale import resolve_device

DEFAULT_R3_DATASETS = ("diabetes", "cpu_activity", "kin8nm", "pumadyn32nh", "white_wine")
DEFAULT_R3_UPDATES = (0, 20, 100, 400)
DEFAULT_R3_SEEDS = (1729, 2718, 31415)


def _schedule_hash(task: Any, *, seed: int, updates: int) -> str:
    digest = hashlib.sha256()
    for update in range(updates):
        context, query = training_episode_indices(task, seed=seed, update=update)
        digest.update(context.tobytes())
        digest.update(query.tobytes())
    return digest.hexdigest()


def _validation_prediction(
    model: torch.nn.Module,
    task: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    context = evaluation_context_indices(task)
    query = task.validation_indices
    evidence, _ = _real_episode(
        task,
        context_indices=context,
        query_indices=query,
        episode_id=f"{task.dataset.dataset_id}-r3-validation",
    )
    model.eval()
    with torch.no_grad():
        prediction = model(evidence.to(device))
    values = prediction["numeric_raw_prediction"]
    if values.ndim == 3:
        values = values[0]
    predicted_task_scale = values[len(context) :, -1].detach().cpu().numpy()
    predicted_raw = task_scale_to_raw(
        predicted_task_scale,
        response_mean=task.response_mean,
        response_scale=task.response_scale,
    )
    truth_raw = task.dataset.response[query]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "rmse": float(math.sqrt(mean_squared_error(truth_raw, predicted_raw))),
        "mae": float(mean_absolute_error(truth_raw, predicted_raw)),
        "scaled_rmse": float(
            math.sqrt(mean_squared_error(task.response[query], predicted_task_scale))
        ),
        "r2": float(r2_score(truth_raw, predicted_raw)),
    }


def _validation_baselines(task: Any, *, seed: int) -> dict[str, dict[str, float]]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor

    x_train = task.features[task.label_indices]
    y_train = task.response[task.label_indices]
    x_validation = task.features[task.validation_indices]
    truth_raw = task.dataset.response[task.validation_indices]

    def metrics(predicted_task_scale: np.ndarray) -> dict[str, float]:
        predicted_raw = task_scale_to_raw(
            predicted_task_scale,
            response_mean=task.response_mean,
            response_scale=task.response_scale,
        )
        return {
            "rmse": float(math.sqrt(mean_squared_error(truth_raw, predicted_raw))),
            "mae": float(mean_absolute_error(truth_raw, predicted_raw)),
            "scaled_rmse": float(
                math.sqrt(
                    mean_squared_error(task.response[task.validation_indices], predicted_task_scale)
                )
            ),
            "r2": float(r2_score(truth_raw, predicted_raw)),
        }

    results = {"train_mean": metrics(np.full(len(x_validation), float(y_train.mean())))}
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        alpha=1.0e-4,
        batch_size=min(64, len(y_train)),
        learning_rate_init=1.0e-3,
        max_iter=500,
        random_state=seed,
    )
    mlp.fit(x_train, y_train)
    results["mlp"] = metrics(mlp.predict(x_validation))
    try:
        import xgboost as xgb
    except ImportError:
        results["xgboost"] = {"status": "unavailable"}  # type: ignore[assignment]
    else:
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=8,
            random_state=seed,
            objective="reg:squarederror",
        )
        xgb_model.fit(x_train, y_train)
        results["xgboost"] = metrics(xgb_model.predict(x_validation))
    return results


def run_query_row_r3_diagnosis(
    *,
    checkpoint: Path,
    output: Path | None = None,
    panel_manifest: Path | None = None,
    dataset_ids: tuple[str, ...] = DEFAULT_R3_DATASETS,
    seeds: tuple[int, ...] = DEFAULT_R3_SEEDS,
    update_checkpoints: tuple[int, ...] = DEFAULT_R3_UPDATES,
    label_budget: int | None = None,
    test_limit: int | None = None,
    row_token_count: int = 4,
    learning_rate: float = 3.0e-4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run R3 against one strict, profile-bound supervised checkpoint."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"R3 checkpoint does not exist: {checkpoint}")
    if not dataset_ids or not seeds or not update_checkpoints:
        raise ValueError("R3 datasets, seeds and update checkpoints must not be empty")
    if tuple(sorted(set(update_checkpoints))) != update_checkpoints or update_checkpoints[0] != 0:
        raise ValueError("R3 update checkpoints must be sorted, unique and start at zero")
    if label_budget is not None and label_budget <= 0:
        raise ValueError("R3 label_budget must be positive or None")
    if learning_rate <= 0.0:
        raise ValueError("R3 learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    checkpoint_identity = read_query_row_pretrain_checkpoint_identity(checkpoint)
    model_identity = checkpoint_identity["model_identity"]
    result_identity = query_row_result_identity(model_identity)
    if int(model_identity["row_token_count"]) != row_token_count:
        raise ValueError("R3 row_token_count must match the checkpoint identity")
    records: list[dict[str, Any]] = []
    for seed in seeds:
        for dataset_id in dataset_ids:
            dataset = (
                load_real_dataset(dataset_id)
                if dataset_id == "diabetes"
                else load_real_dataset(dataset_id, panel_manifest=panel_manifest)
            )
            task = prepare_real_task(
                dataset,
                budget=label_budget,
                seed=seed,
                test_limit=test_limit,
            )
            pretrained, loaded_identity = build_model_from_checkpoint(
                checkpoint,
                device=resolved_device,
            )
            if loaded_identity != checkpoint_identity:
                raise RuntimeError("R3 checkpoint identity changed during reconstruction")
            scratch = build_random_model(
                checkpoint_identity,
                seed=seed,
                device=resolved_device,
            )
            pretrained.requires_grad_(True)
            scratch.requires_grad_(True)
            theta0 = {
                name: value.detach().cpu().clone() for name, value in scratch.state_dict().items()
            }
            scratch.load_state_dict(theta0)
            scratch_initial_hash = _parameter_sha256(scratch)
            pretrain_initial_hash = scratch_initial_hash
            pretrained_initial_hash = _parameter_sha256(pretrained)
            scratch_optimizer = torch.optim.AdamW(
                scratch.parameters(), lr=learning_rate, weight_decay=1.0e-4
            )
            pretrained_optimizer = torch.optim.AdamW(
                pretrained.parameters(), lr=learning_rate, weight_decay=1.0e-4
            )
            schedule_hash = _schedule_hash(
                task,
                seed=seed,
                updates=max(update_checkpoints),
            )
            baseline_metrics = _validation_baselines(task, seed=seed)
            current_updates = 0
            for target_updates in update_checkpoints:
                for update in range(current_updates, target_updates):
                    context, query = training_episode_indices(task, seed=seed, update=update)
                    evidence, truth = _real_episode(
                        task,
                        context_indices=context,
                        query_indices=query,
                        episode_id=f"{dataset_id}-r3-{seed}-{update:04d}",
                    )
                    _train_one(scratch, scratch_optimizer, evidence, truth, resolved_device)
                    _train_one(pretrained, pretrained_optimizer, evidence, truth, resolved_device)
                current_updates = target_updates
                scratch_metrics = _validation_prediction(scratch, task, device=resolved_device)
                pretrained_metrics = _validation_prediction(
                    pretrained, task, device=resolved_device
                )
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "updates": target_updates,
                        "label_budget": label_budget,
                        "scratch_metrics": scratch_metrics,
                        "pretrained_metrics": pretrained_metrics,
                        "gain_scratch_minus_pretrained": scratch_metrics["rmse"]
                        - pretrained_metrics["rmse"],
                        "baseline_metrics": baseline_metrics,
                        "scratch_initial_parameter_sha256": scratch_initial_hash,
                        "pretrain_initial_parameter_sha256": pretrain_initial_hash,
                        "pretrained_initial_parameter_sha256": pretrained_initial_hash,
                        "scratch_episode_schedule_sha256": schedule_hash,
                        "pretrained_episode_schedule_sha256": schedule_hash,
                        "exact_same_init": scratch_initial_hash == pretrain_initial_hash,
                    }
                )
    result = {
        **result_identity,
        "schema_version": "tabu.query-row.r3-diagnosis.v2",
        "status": "pass" if records else "kill",
        "execution_status": "succeeded" if records else "killed",
        "capability_gate": "not_applicable",
        "evidence_status": "local_unissued",
        "claim_boundary": (
            "R3 corrected validation-only adaptation diagnosis; "
            "no accepted transfer claim"
        ),
        "checkpoint": str(checkpoint),
        "panel_manifest": str(panel_manifest) if panel_manifest is not None else None,
        "datasets": list(dataset_ids),
        "seeds": list(seeds),
        "update_checkpoints": list(update_checkpoints),
        "label_budget": label_budget,
        "device": str(resolved_device),
        "records": records,
    }
    if output is not None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = [
    "DEFAULT_R3_DATASETS",
    "DEFAULT_R3_SEEDS",
    "DEFAULT_R3_UPDATES",
    "run_query_row_r3_diagnosis",
]
