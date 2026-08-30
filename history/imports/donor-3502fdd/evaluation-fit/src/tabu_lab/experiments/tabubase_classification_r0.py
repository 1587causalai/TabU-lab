"""Validation-only R0 schedule selection for TabUBase classification transfer."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabu_lab.training import Objective

from .tabubase_real_benchmark import (
    _sample_training_episode,
    _source_tree_hash,
    evaluate_tabubase_on_indices,
    load_real_dataset,
    prepare_real_task,
)
from .tabubase_scale import (
    ROOT_SEEDS,
    build_tabubase_scale_model,
    load_pretrain_checkpoint,
)

R0_LEARNING_RATES = (1.0e-4, 3.0e-4, 1.0e-3)
R0_CHECKPOINT_UPDATES = (400, 1_200)


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_path(checkpoint_root: Path, seed: int) -> Path:
    path = checkpoint_root / f"tabubase-pt-s1-seed-{seed}" / "checkpoint-20000.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing paired PT-S1 checkpoint: {path}")
    return path


def run_classification_r0_validation(
    *,
    dataset_ids: tuple[str, ...],
    checkpoint_root: Path,
    output_path: Path,
    device: torch.device,
    budget: int = 128,
) -> dict[str, Any]:
    """Run the R0 grid without evaluating any test row."""

    started = time.monotonic()
    records: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset = load_real_dataset(dataset_id)
        if dataset.task != "classification":
            raise ValueError(f"R0 classification grid received {dataset.task}: {dataset_id}")
        for seed in ROOT_SEEDS:
            task = prepare_real_task(dataset, budget=budget, seed=seed)
            checkpoint = _checkpoint_path(checkpoint_root, seed)
            for arm in ("pretrained", "scratch"):
                for learning_rate in R0_LEARNING_RATES:
                    model = build_tabubase_scale_model(seed=seed, device=device)
                    if arm == "pretrained":
                        load_pretrain_checkpoint(model, checkpoint)
                    initial_hash = _state_hash(model)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=learning_rate,
                        weight_decay=1.0e-4,
                    )
                    losses: list[float] = []
                    for update in range(max(R0_CHECKPOINT_UPDATES)):
                        evidence, truth = _sample_training_episode(
                            task,
                            seed=seed,
                            update=update,
                        )
                        model.train()
                        optimizer.zero_grad(set_to_none=True)
                        prediction = model(evidence.to(device))
                        loss = Objective()(prediction, truth.to(device)).total
                        if not bool(torch.isfinite(loss)):
                            raise RuntimeError(
                                f"non-finite R0 loss for {dataset_id}/{seed}/{arm}/"
                                f"{learning_rate}"
                            )
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        if update == 0 or (update + 1) % 100 == 0:
                            losses.append(float(loss.detach().cpu()))
                        completed = update + 1
                        if completed in R0_CHECKPOINT_UPDATES:
                            validation = evaluate_tabubase_on_indices(
                                model,
                                task,
                                device=device,
                                query_indices=task.validation_indices,
                                query_partition="validation",
                            )
                            records.append(
                                {
                                    "dataset_id": dataset_id,
                                    "dataset_sha256": dataset.content_hash,
                                    "seed": seed,
                                    "arm": arm,
                                    "learning_rate": learning_rate,
                                    "updates": completed,
                                    "validation_metrics": validation,
                                    "label_budget": len(task.label_indices),
                                    "validation_rows": len(task.validation_indices),
                                    "initial_parameter_sha256": initial_hash,
                                    "parameter_sha256": _state_hash(model),
                                    "checkpoint_sha256": (
                                        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                        if arm == "pretrained"
                                        else None
                                    ),
                                    "loss_history": losses.copy(),
                                }
                            )
    payload = {
        "schema_version": "tabu.transfer-base-classification-r0-validation.v1",
        "status": "local_unissued",
        "selection_partition": "validation",
        "test_evaluations": 0,
        "datasets": list(dataset_ids),
        "seeds": list(ROOT_SEEDS),
        "arms": ["pretrained", "scratch"],
        "learning_rates": list(R0_LEARNING_RATES),
        "checkpoint_updates": list(R0_CHECKPOINT_UPDATES),
        "selection_objective": (
            "dataset-macro mean validation log_loss across paired seeds and arms"
        ),
        "records": records,
        "elapsed_seconds": time.monotonic() - started,
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "source_tree_sha256": _source_tree_hash(),
        "claim_boundary": (
            "validation-only exploratory schedule search; no test or accepted claim"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def select_global_r0_schedule(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one schedule by dataset-macro validation log loss."""

    if not payloads:
        raise ValueError("cannot select an R0 schedule from no payloads")
    for payload in payloads:
        if payload.get("selection_partition") != "validation":
            raise ValueError("R0 selection accepts validation-only payloads")
        if payload.get("test_evaluations") != 0:
            raise ValueError("R0 selection rejects payloads that evaluated test rows")
    source_hashes = {str(payload["source_tree_sha256"]) for payload in payloads}
    if len(source_hashes) != 1:
        raise ValueError("R0 payloads must bind one exact source tree")
    records = [record for payload in payloads for record in payload["records"]]
    datasets = sorted({str(record["dataset_id"]) for record in records})
    candidates = sorted(
        {(float(record["learning_rate"]), int(record["updates"])) for record in records}
    )
    expected_per_dataset = len(ROOT_SEEDS) * 2
    scores: list[dict[str, Any]] = []
    for learning_rate, updates in candidates:
        dataset_means: dict[str, float] = {}
        for dataset_id in datasets:
            values = [
                float(record["validation_metrics"]["log_loss"])
                for record in records
                if record["dataset_id"] == dataset_id
                and float(record["learning_rate"]) == learning_rate
                and int(record["updates"]) == updates
            ]
            if len(values) != expected_per_dataset:
                raise ValueError(
                    f"incomplete R0 candidate {learning_rate}/{updates} for "
                    f"{dataset_id}: {len(values)} != {expected_per_dataset}"
                )
            dataset_means[dataset_id] = float(np.mean(values))
        scores.append(
            {
                "learning_rate": learning_rate,
                "updates": updates,
                "dataset_mean_validation_log_loss": dataset_means,
                "macro_mean_validation_log_loss": float(np.mean(list(dataset_means.values()))),
            }
        )
    if not scores:
        raise ValueError("cannot select an R0 schedule from no records")
    selected = min(
        scores,
        key=lambda item: (
            item["macro_mean_validation_log_loss"],
            item["updates"],
            item["learning_rate"],
        ),
    )
    return {
        "schema_version": "tabu.transfer-base-classification-r0-selection.v1",
        "status": "local_unissued",
        "selection_partition": "validation",
        "test_evaluations": 0,
        "source_tree_sha256": source_hashes.pop(),
        "datasets": datasets,
        "candidate_scores": scores,
        "selected": selected,
    }


__all__ = [
    "R0_CHECKPOINT_UPDATES",
    "R0_LEARNING_RATES",
    "run_classification_r0_validation",
    "select_global_r0_schedule",
]
