"""Stage-6 paired synthetic-pretrained versus scratch fine-tuning diagnostic."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch

from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig
from tabu_lab.training.objective import Objective

from .query_row_real_benchmark import _metrics, _model_prediction
from .query_row_supervised_synthetic import (
    make_query_row_supervised_synthetic_episode,
    supervised_synthetic_episode_loss,
)
from .tabubase_real_benchmark import (
    _real_episode,
    load_real_dataset,
    prepare_real_task,
    training_episode_indices,
)
from .tabubase_scale import resolve_device

TaskKind = Literal["classification", "regression"]


def _new_row_model(
    *,
    seed: int,
    row_token_count: int,
    device: torch.device,
) -> torch.nn.Module:
    torch.manual_seed(seed)
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
    )
    return model.to(device)


@dataclass(frozen=True, slots=True)
class QueryRowFinetuneLiftRecord:
    dataset_id: str
    task: TaskKind
    scratch_loss: float
    pretrained_loss: float
    gain_scratch_minus_pretrained: float
    scratch_metrics: dict[str, float]
    pretrained_metrics: dict[str, float]
    updates: int
    label_budget: int
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryRowFinetuneLiftResult:
    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    profile_id: str
    model_spec_hash: str
    row_token_count: int
    device: str
    seed: int
    pretrain_steps: int
    pretrain_worlds: int
    pretrain_final_loss: float
    records: tuple[QueryRowFinetuneLiftRecord, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.as_dict() for record in self.records]
        return payload


def _pretrain_supervised_profile(
    model: torch.nn.Module,
    *,
    seed: int,
    steps: int,
    worlds: int,
    rows: int,
    row_token_count: int,
    learning_rate: float,
) -> float:
    episodes = tuple(
        make_query_row_supervised_synthetic_episode(
            seed=seed + index,
            rows=rows,
            context_rows=max(2, rows // 2),
            row_token_count=row_token_count,
            world_id=f"stage6-pretrain-{index}",
            world_family=(
                "row_latent_linear"
                if index % 3 == 0
                else "row_latent_periodic"
                if index % 3 == 1
                else "row_latent_polynomial"
            ),
        )
        for index in range(worlds)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    final_loss = float("nan")
    for step in range(steps):
        loss = supervised_synthetic_episode_loss(model, episodes[step % len(episodes)])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite TabUR Stage-6 synthetic pretraining loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    return final_loss


def _train_real_arm(
    model: torch.nn.Module,
    task: Any,
    *,
    seed: int,
    updates: int,
    learning_rate: float,
    device: torch.device,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    for update in range(updates):
        context, query = training_episode_indices(task, seed=seed, update=update)
        evidence, truth = _real_episode(
            task,
            context_indices=context,
            query_indices=query,
            episode_id=f"{task.dataset.dataset_id}-stage6-{seed}-{update:04d}",
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model._forward_dense(evidence.to(device), emit_trace=False)
        loss = Objective()(prediction, truth.to(device)).total
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite Stage-6 fine-tuning loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def _evaluation_loss(model: torch.nn.Module, task: Any) -> tuple[float, dict[str, float]]:
    predictions, truth = _model_prediction(model, task)
    metrics = _metrics(task, predictions, truth)
    return (
        float(metrics["log_loss"] if task.dataset.task == "classification" else metrics["rmse"]),
        metrics,
    )


def run_query_row_finetune_lift(
    *,
    dataset_ids: tuple[str, ...] = ("iris", "diabetes"),
    label_budget: int = 64,
    updates: int = 20,
    pretrain_steps: int = 20,
    pretrain_worlds: int = 4,
    learning_rate: float = 3.0e-4,
    pretrain_learning_rate: float = 1.0e-2,
    test_limit: int = 64,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 1729,
) -> QueryRowFinetuneLiftResult:
    """Run paired fine-tuning with a profile-compatible synthetic checkpoint."""

    if not dataset_ids:
        raise ValueError("dataset_ids must not be empty")
    if label_budget <= 0 or updates <= 0 or pretrain_steps <= 0 or pretrain_worlds <= 0:
        raise ValueError("budgets and step counts must be positive")
    if learning_rate <= 0.0 or pretrain_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if test_limit <= 0 or row_token_count <= 0:
        raise ValueError("test_limit and row_token_count must be positive")
    resolved_device = resolve_device(str(device))

    pretrained = _new_row_model(
        seed=seed,
        row_token_count=row_token_count,
        device=resolved_device,
    )
    pretrain_final_loss = _pretrain_supervised_profile(
        pretrained,
        seed=seed + 10_000,
        steps=pretrain_steps,
        worlds=pretrain_worlds,
        rows=24,
        row_token_count=row_token_count,
        learning_rate=pretrain_learning_rate,
    )
    records: list[QueryRowFinetuneLiftRecord] = []
    model_spec_hash = pretrained.model_spec_hash
    # Keep a detached CPU snapshot as the transfer boundary.  Loading an MPS
    # state_dict directly into a second MPS module is not stable across
    # repeated runs on current PyTorch builds.
    pretrained_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in pretrained.state_dict().items()
    }
    for offset, dataset_id in enumerate(dataset_ids):
        task_seed = seed + offset
        task = prepare_real_task(
            load_real_dataset(dataset_id),
            budget=label_budget,
            seed=task_seed,
            test_limit=test_limit,
        )
        # Keep exact profile, architecture, split, schedule and optimizer the
        # same; only the initial parameter state differs.
        pretrained_arm = _new_row_model(
            seed=seed,
            row_token_count=row_token_count,
            device=resolved_device,
        )
        pretrained_arm.load_state_dict(pretrained_state)
        scratch_arm = _new_row_model(
            seed=seed + 1000 + offset,
            row_token_count=row_token_count,
            device=resolved_device,
        )
        _train_real_arm(
            pretrained_arm,
            task,
            seed=task_seed,
            updates=updates,
            device=resolved_device,
            learning_rate=learning_rate,
        )
        _train_real_arm(
            scratch_arm,
            task,
            seed=task_seed,
            updates=updates,
            device=resolved_device,
            learning_rate=learning_rate,
        )
        pretrained_loss, pretrained_metrics = _evaluation_loss(pretrained_arm, task)
        scratch_loss, scratch_metrics = _evaluation_loss(scratch_arm, task)
        gain = scratch_loss - pretrained_loss
        if not all(
            math.isfinite(value)
            for value in (*pretrained_metrics.values(), *scratch_metrics.values(), gain)
        ):
            raise RuntimeError(f"non-finite Stage-6 evaluation for {dataset_id}")
        records.append(
            QueryRowFinetuneLiftRecord(
                dataset_id=dataset_id,
                task=task.dataset.task,
                scratch_loss=scratch_loss,
                pretrained_loss=pretrained_loss,
                gain_scratch_minus_pretrained=gain,
                scratch_metrics=scratch_metrics,
                pretrained_metrics=pretrained_metrics,
                updates=updates,
                label_budget=label_budget,
                seed=task_seed,
            )
        )
    return QueryRowFinetuneLiftResult(
        status="pass" if records else "kill",
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR paired bounded fine-tuning lift diagnostic with profile-compatible "
            "synthetic pretraining; no formal receipt, benchmark, accepted transfer, "
            "or causal capability claim"
        ),
        model_id=pretrained.model_id,
        contract_version=pretrained.contract_version,
        profile_id=pretrained.profile.value,
        model_spec_hash=model_spec_hash,
        row_token_count=row_token_count,
        device=str(resolved_device),
        seed=seed,
        pretrain_steps=pretrain_steps,
        pretrain_worlds=pretrain_worlds,
        pretrain_final_loss=pretrain_final_loss,
        records=tuple(records),
    )


__all__ = [
    "QueryRowFinetuneLiftRecord",
    "QueryRowFinetuneLiftResult",
    "run_query_row_finetune_lift",
]
