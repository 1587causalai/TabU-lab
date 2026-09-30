"""Stage-5 frozen-ICL diagnostic for TabUR.

The harness pretrains one fresh TabUR model on synthetic episodes, then compares
three frozen arms on held-out synthetic episodes.  No optimizer is constructed
after pretraining; every arm records an adjacent parameter hash check.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_identity import (
    query_row_result_identity,
    require_query_row_readout_identity,
)
from .query_row_pretraining import (
    load_query_row_pretrain_checkpoint,
    read_query_row_pretrain_checkpoint_identity,
)
from .query_row_synthetic_fit import (
    QueryRowSyntheticEpisode,
    _episode_loss,
    make_query_row_synthetic_episode,
)
from .query_row_transfer_common import _reference_config_from_identity
from .tabubase_scale import resolve_device


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _row_model(
    *,
    seed: int,
    row_token_count: int,
    device: torch.device,
    max_features: int = 4,
    config: ReferenceConfig | None = None,
    profile: str = "completion.artificial_mask.v1",
    row_readout_mode: str = "anchored",
    anchored_gamma_initial: float = 1.0e-2,
) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_model(
        "tabu.query.row",
        config=config
        or ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=row_token_count,
            max_features=max_features,
        ),
        profile=profile,
        row_token_count=row_token_count,
        row_readout_mode=row_readout_mode,
        anchored_gamma_initial=anchored_gamma_initial,
    ).to(device)


def _shuffled_sidecar(episode: QueryRowSyntheticEpisode) -> QueryRowSyntheticEpisode:
    values = episode.sidecar.target_values.clone()
    mask = episode.sidecar.target_mask
    target_indices = mask.nonzero(as_tuple=False)
    if len(target_indices) > 1:
        target_values = values[mask]
        values[mask] = target_values.roll(1)
    from dataclasses import replace

    return replace(
        episode,
        sidecar=replace(episode.sidecar, target_values=values),
    )


@dataclass(frozen=True, slots=True)
class QueryRowFrozenICLRecord:
    world_id: str
    context_rows: int
    arm: str
    mse: float
    target_count: int
    parameter_hash_before: str
    parameter_hash_after: str
    optimizer_created: bool

    @property
    def parameter_hash_unchanged(self) -> bool:
        return self.parameter_hash_before == self.parameter_hash_after

    def as_dict(self) -> dict[str, Any]:
        return asdict(self) | {"parameter_hash_unchanged": self.parameter_hash_unchanged}


@dataclass(frozen=True, slots=True)
class QueryRowFrozenICLResult:
    schema_version: str
    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    model_spec_hash: str
    variant_hash: str
    row_token_count: int
    row_readout_mode: str
    row_readout_identity: dict[str, Any]
    device: str
    seed: int
    eval_worlds: int
    records: tuple[QueryRowFrozenICLRecord, ...]
    checkpoint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["records"] = [record.as_dict() for record in self.records]
        return payload


def _evaluate_frozen(
    model: torch.nn.Module,
    episode: QueryRowSyntheticEpisode,
) -> float:
    model.eval()
    with torch.no_grad():
        loss = _episode_loss(model, episode)
    return float(loss.item())


def run_query_row_frozen_icl(
    *,
    seed: int = 1729,
    pretrain_steps: int = 20,
    pretrain_worlds: int = 4,
    context_rows: tuple[int, ...] = (2, 4, 8),
    rows: int = 16,
    row_token_count: int = 4,
    learning_rate: float = 1.0e-2,
    device: str | torch.device = "cpu",
    checkpoint: Path | None = None,
    eval_worlds: int = 1,
) -> QueryRowFrozenICLResult:
    """Run frozen synthetic ICL with explicit no-optimizer controls."""

    if pretrain_steps <= 0 or pretrain_worlds <= 0 or rows < 3 or eval_worlds <= 0:
        raise ValueError("pretrain_steps, pretrain_worlds and rows must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    if not context_rows or any(size < 1 or size >= rows for size in context_rows):
        raise ValueError("context_rows must be non-empty and smaller than rows")

    checkpointed = checkpoint is not None
    if checkpointed:
        assert checkpoint is not None
        checkpoint_identity = read_query_row_pretrain_checkpoint_identity(checkpoint)
        checkpoint_model_identity = checkpoint_identity["model_identity"]
        checkpoint_readout = require_query_row_readout_identity(checkpoint_model_identity)
        checkpoint_token_count = int(checkpoint_model_identity["row_token_count"])
        if row_token_count != checkpoint_token_count:
            raise ValueError(
                "row_token_count must match the checkpoint identity; "
                "checkpoint reconstruction never applies a caller default"
            )
        pretrained = _row_model(
            seed=seed,
            row_token_count=checkpoint_token_count,
            device=resolved_device,
            config=_reference_config_from_identity(checkpoint_model_identity),
            profile=str(checkpoint_model_identity["profile_id"]),
            row_readout_mode=str(checkpoint_readout["mode"]),
            anchored_gamma_initial=float(
                checkpoint_readout["anchored_gamma_initial"]
            ),
        )
        load_query_row_pretrain_checkpoint(pretrained, checkpoint)
    else:
        pretrained = _row_model(
            seed=seed,
            row_token_count=row_token_count,
            device=resolved_device,
            max_features=4,
            row_readout_mode="anchored",
            anchored_gamma_initial=1.0e-2,
        )
        train_episodes = tuple(
            make_query_row_synthetic_episode(
                seed=seed + 10 + index,
                rows=rows,
                row_token_count=row_token_count,
                world_id=f"pretrain-world-{index}",
                world_family="row_latent_linear"
                if index % 2 == 0
                else "row_latent_polynomial",
            )
            for index in range(pretrain_worlds)
        )
        optimizer = torch.optim.Adam(pretrained.parameters(), lr=learning_rate)
        pretrained.train()
        for step in range(pretrain_steps):
            loss = _episode_loss(pretrained, train_episodes[step % len(train_episodes)])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    records: list[QueryRowFrozenICLRecord] = []
    pretrained_identity = pretrained.checkpoint_identity()
    result_identity = query_row_result_identity(pretrained_identity)
    readout_identity = result_identity["row_readout_identity"]
    reference_config = _reference_config_from_identity(pretrained_identity)
    profile_id = str(pretrained_identity["profile_id"])
    families = ("row_latent_periodic", "row_latent_linear", "row_latent_polynomial")
    for world_index in range(eval_worlds):
        family = families[world_index % len(families)]
        for context_size in context_rows:
            episode = make_query_row_synthetic_episode(
                seed=seed + 100 + world_index * 101,
                rows=rows,
                row_token_count=row_token_count,
                context_rows=context_size,
                world_id=f"heldout-context-{world_index}-{context_size}",
                world_family=family,
            )
            random_init = _row_model(
                seed=seed + 1000 + world_index * 101 + context_size,
                row_token_count=row_token_count,
                device=resolved_device,
                config=reference_config,
                profile=profile_id,
                row_readout_mode=str(readout_identity["mode"]),
                anchored_gamma_initial=float(
                    readout_identity["anchored_gamma_initial"]
                ),
            )
            shuffled = _shuffled_sidecar(episode)
            arms = (
                ("pretrained_frozen", pretrained, episode),
                ("random_init_frozen", random_init, episode),
                ("pretrained_shuffled", pretrained, shuffled),
            )
            for arm, model, arm_episode in arms:
                before = _state_hash(model)
                value = _evaluate_frozen(model, arm_episode)
                after = _state_hash(model)
                records.append(
                    QueryRowFrozenICLRecord(
                        world_id=episode.world_id,
                        context_rows=context_size,
                        arm=arm,
                        mse=value,
                        target_count=arm_episode.sidecar.target_count,
                        parameter_hash_before=before,
                        parameter_hash_after=after,
                        optimizer_created=False,
                    )
                )

    status = "pass" if all(
        record.parameter_hash_unchanged
        and not record.optimizer_created
        and math.isfinite(record.mse)
        for record in records
    ) else "kill"
    return QueryRowFrozenICLResult(
        schema_version="tabu.query-row.frozen-icl-result.v2",
        status=status,
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR frozen synthetic ICL diagnostic only; no real-data transfer, "
            "fine-tuning lift, benchmark, or accepted claim"
        ),
        model_id=result_identity["model_id"],
        contract_version=result_identity["contract_version"],
        model_spec_hash=result_identity["model_spec_hash"],
        variant_hash=result_identity["variant_hash"],
        row_token_count=row_token_count,
        row_readout_mode=str(readout_identity["mode"]),
        row_readout_identity=readout_identity,
        device=str(resolved_device),
        seed=seed,
        eval_worlds=eval_worlds,
        records=tuple(records),
        checkpoint=str(checkpoint) if checkpoint is not None else None,
    )


__all__ = [
    "QueryRowFrozenICLRecord",
    "QueryRowFrozenICLResult",
    "run_query_row_frozen_icl",
]
