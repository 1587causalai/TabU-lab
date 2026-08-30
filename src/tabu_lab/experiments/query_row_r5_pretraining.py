"""Gate R5 bounded v2 pretraining and frozen synthetic validation.

The runner screens the declared AdamW learning rates on a small validation
pilot, then trains only the B0/B1 rungs.  Every output is a new local-unissued
diagnostic and refuses to overwrite an existing checkpoint or result.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tabu_lab.contracts import canonical_hash
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_pretraining import save_query_row_pretrain_checkpoint
from .query_row_supervised_synthetic_v2 import (
    build_query_row_supervised_synthetic_v2_plan,
    make_query_row_supervised_synthetic_v2_episode,
    supervised_synthetic_v2_episode_loss,
)
from .tabubase_scale import resolve_device

R5_LEARNING_RATES = (1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2)
R5_RUNG_SPECS = {
    "B0": {"worlds": 512, "updates": 1500},
    "B1": {"worlds": 2048, "updates": 6000},
}


def _model(*, seed: int, row_token_count: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_model(
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
    ).to(device)


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _plan_hash(plan: tuple[dict[str, Any], ...]) -> str:
    return canonical_hash(plan)


def _episode(root_seed: int, spec: dict[str, Any]) -> Any:
    return make_query_row_supervised_synthetic_v2_episode(
        root_seed=root_seed,
        world_id=spec["world_id"],
        partition=spec["partition"],
        width=spec["width"],
        family=spec["family"],
        predictor_regime=spec["predictor_regime"],
        noise_level=spec["noise_level"],
        context_rows=spec["context_rows"],
    )


def _mean_loss(
    model: torch.nn.Module,
    *,
    root_seed: int,
    plan: tuple[dict[str, Any], ...],
    limit: int | None = None,
) -> float:
    selected = plan if limit is None else plan[:limit]
    values = []
    model.eval()
    with torch.no_grad():
        for spec in selected:
            values.append(float(supervised_synthetic_v2_episode_loss(model, _episode(root_seed, spec))))
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise RuntimeError("non-finite v2 synthetic validation loss")
    return value


def _label_shuffled(episode: Any) -> Any:
    values = episode.sidecar.target_values.clone()
    target = episode.sidecar.target_mask
    if int(target.sum()) > 1:
        values[target] = values[target].roll(1)
    return replace(episode, sidecar=replace(episode.sidecar, target_values=values))


def _context_row_shuffled(episode: Any) -> Any:
    context_rows = episode.context_rows
    permutation = torch.randperm(context_rows, generator=torch.Generator().manual_seed(17))
    row_order = torch.cat((permutation, torch.arange(context_rows, episode.evidence.forward_values.shape[0])))
    evidence = replace(
        episode.evidence,
        row_ids=tuple(episode.evidence.row_ids[int(index)] for index in row_order),
        forward_values=episode.evidence.forward_values[row_order],
        origin_states=episode.evidence.origin_states[row_order],
        forward_roles=episode.evidence.forward_roles[row_order],
    )
    sidecar = replace(
        episode.sidecar,
        row_ids=evidence.row_ids,
        target_values=episode.sidecar.target_values[row_order],
        target_mask=episode.sidecar.target_mask[row_order],
    )
    return replace(episode, evidence=evidence, sidecar=sidecar)


def _frozen_controls(
    model: torch.nn.Module,
    *,
    root_seed: int,
    plan: tuple[dict[str, Any], ...],
    row_token_count: int,
    device: torch.device,
    limit: int,
    same_init_state: dict[str, Tensor],
) -> dict[str, Any]:
    random_init = _model(seed=root_seed + 991, row_token_count=row_token_count, device=device)
    random_init.load_state_dict(same_init_state)
    arms: dict[str, tuple[torch.nn.Module, str]] = {
        "pretrained_frozen": (model, "original"),
        "same_init_random_frozen": (random_init, "original"),
        "pretrained_label_shuffled": (model, "label_shuffled"),
        "pretrained_context_row_shuffled": (model, "context_row_shuffled"),
    }
    losses: dict[str, list[float]] = {name: [] for name in arms}
    hashes: dict[str, dict[str, Any]] = {}
    for name, (arm_model, transform) in arms.items():
        before = _state_hash(arm_model)
        arm_model.eval()
        with torch.no_grad():
            for spec in plan[:limit]:
                episode = _episode(root_seed, spec)
                if transform == "label_shuffled":
                    episode = _label_shuffled(episode)
                elif transform == "context_row_shuffled":
                    episode = _context_row_shuffled(episode)
                losses[name].append(float(supervised_synthetic_v2_episode_loss(arm_model, episode)))
        after = _state_hash(arm_model)
        hashes[name] = {
            "parameter_hash_before": before,
            "parameter_hash_after": after,
            "parameter_hash_unchanged": before == after,
            "optimizer_created": False,
            "requires_grad_update_attempted": False,
        }
    aggregates = {name: sum(values) / len(values) for name, values in losses.items()}
    return {
        "aggregate_loss": aggregates,
        "per_world_loss": losses,
        "hash_controls": hashes,
        "promotion_gate": {
            "pretrained_lower_than_random": aggregates["pretrained_frozen"]
            < aggregates["same_init_random_frozen"],
            "pretrained_lower_than_label_shuffled": aggregates["pretrained_frozen"]
            < aggregates["pretrained_label_shuffled"],
            "all_frozen_hashes_unchanged": all(
                item["parameter_hash_unchanged"] for item in hashes.values()
            ),
        },
    }


def _train_rung(
    *,
    root_seed: int,
    rung: str,
    learning_rate: float,
    row_token_count: int,
    device: torch.device,
    output_root: Path,
    validation_worlds: int,
) -> dict[str, Any]:
    spec = R5_RUNG_SPECS[rung]
    train_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed, worlds=spec["worlds"], partition="train"
    )
    validation_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed + 500_000, worlds=validation_worlds, partition="validation"
    )
    model = _model(seed=root_seed, row_token_count=row_token_count, device=device)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    initial_validation = _mean_loss(
        model, root_seed=root_seed + 500_000, plan=validation_plan
    )
    model.train()
    losses: list[float] = []
    for update in range(spec["updates"]):
        episode = _episode(root_seed, train_plan[update % len(train_plan)])
        loss = supervised_synthetic_v2_episode_loss(model, episode)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite R5 {rung} loss at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 0 or (update + 1) % max(1, spec["updates"] // 10) == 0:
            losses.append(float(loss.detach().item()))
    final_validation = _mean_loss(
        model, root_seed=root_seed + 500_000, plan=validation_plan
    )
    controls = _frozen_controls(
        model,
        root_seed=root_seed + 700_000,
        plan=validation_plan,
        row_token_count=row_token_count,
        device=device,
        limit=validation_worlds,
        same_init_state=initial_state,
    )
    checkpoint = output_root / f"tabur-v2-{rung.lower()}-seed{root_seed}.safetensors"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {checkpoint}")
    checkpoint_info = save_query_row_pretrain_checkpoint(
        model,
        checkpoint,
        metadata={
            "generator_id": "tabur.supervised-query-row-diverse-v2",
            "generator_schema": "tabu.query-row.supervised-synthetic-v2.recipe.v1",
            "root_seed": root_seed,
            "rung": rung,
            "worlds": spec["worlds"],
            "updates": spec["updates"],
            "learning_rate": learning_rate,
            "weight_decay": 1.0e-4,
            "row_token_count": row_token_count,
            "train_plan_hash": _plan_hash(train_plan),
            "validation_plan_hash": _plan_hash(validation_plan),
        },
    )
    return {
        "rung": rung,
        "root_seed": root_seed,
        "worlds": spec["worlds"],
        "updates": spec["updates"],
        "learning_rate": learning_rate,
        "weight_decay": 1.0e-4,
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "loss_trace": losses,
        "checkpoint": checkpoint_info,
        "synthetic_frozen_controls": controls,
        "status": "passed" if math.isfinite(final_validation) else "failed",
    }


def _pilot(
    *,
    root_seed: int,
    learning_rates: tuple[float, ...],
    row_token_count: int,
    device: torch.device,
    pilot_worlds: int,
    pilot_updates: int,
) -> tuple[float, tuple[dict[str, Any], ...]]:
    train_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed, worlds=pilot_worlds, partition="train"
    )
    validation_plan = build_query_row_supervised_synthetic_v2_plan(
        root_seed=root_seed + 500_000, worlds=max(16, pilot_worlds // 2), partition="validation"
    )
    records: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        model = _model(seed=root_seed, row_token_count=row_token_count, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
        for update in range(pilot_updates):
            loss = supervised_synthetic_v2_episode_loss(
                model, _episode(root_seed, train_plan[update % len(train_plan)])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation_loss = _mean_loss(
            model, root_seed=root_seed + 500_000, plan=validation_plan
        )
        records.append({"learning_rate": learning_rate, "validation_loss": validation_loss})
    selected = min(records, key=lambda item: (item["validation_loss"], item["learning_rate"]))
    return float(selected["learning_rate"]), tuple(records)


def run_query_row_r5_bounded_pretraining(
    *,
    output_root: Path,
    seeds: tuple[int, ...] = (1729, 2718, 31415),
    rungs: tuple[str, ...] = ("B0", "B1"),
    learning_rates: tuple[float, ...] = R5_LEARNING_RATES,
    row_token_count: int = 4,
    device: str | torch.device = "cuda",
    pilot_worlds: int = 64,
    pilot_updates: int = 100,
    validation_worlds: int = 48,
) -> dict[str, Any]:
    """Screen AdamW briefly, then run the requested B0/B1 rungs."""

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to use non-empty output root: {output_root}")
    if not seeds or not rungs:
        raise ValueError("R5 seeds and rungs must not be empty")
    if any(rung not in R5_RUNG_SPECS for rung in rungs):
        raise ValueError("R5 only supports B0 and B1 in this runner")
    if any(rate <= 0.0 for rate in learning_rates):
        raise ValueError("R5 learning rates must be positive")
    resolved_device = resolve_device(str(device))
    output_root.mkdir(parents=True, exist_ok=True)
    selected_lr, pilot = _pilot(
        root_seed=seeds[0],
        learning_rates=learning_rates,
        row_token_count=row_token_count,
        device=resolved_device,
        pilot_worlds=pilot_worlds,
        pilot_updates=pilot_updates,
    )
    rung_records = []
    for rung in rungs:
        for seed in seeds:
            rung_records.append(
                _train_rung(
                    root_seed=seed,
                    rung=rung,
                    learning_rate=selected_lr,
                    row_token_count=row_token_count,
                    device=resolved_device,
                    output_root=output_root,
                    validation_worlds=validation_worlds,
                )
            )
    result = {
        "schema_version": "tabu.query-row.r5-bounded-pretraining.v1",
        "generator_id": "tabur.supervised-query-row-diverse-v2",
        "device": str(resolved_device),
        "seeds": list(seeds),
        "rungs": list(rungs),
        "pilot": {"learning_rates": list(learning_rates), "records": list(pilot), "selected_learning_rate": selected_lr},
        "records": rung_records,
        "status": "passed" if all(item["status"] == "passed" for item in rung_records) else "failed",
        "evidence_status": "local_unissued",
        "claim_boundary": "R5 bounded v2 pretraining and synthetic frozen controls; no accepted capability claim",
    }
    result_path = output_root / "r5-bounded-pretraining.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {result_path}")
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


__all__ = ["R5_LEARNING_RATES", "R5_RUNG_SPECS", "run_query_row_r5_bounded_pretraining"]
