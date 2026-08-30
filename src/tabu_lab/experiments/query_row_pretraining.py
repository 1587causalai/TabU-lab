"""Explicit, bounded TabUR synthetic pretraining and checkpoint boundary."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from tabu_lab.contracts import canonical_hash
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

from .query_row_supervised_synthetic import (
    make_query_row_supervised_synthetic_episode,
    supervised_synthetic_episode_loss,
)
from .query_row_synthetic_fit import _episode_loss, make_query_row_synthetic_episode
from .tabubase_scale import resolve_device

_PRETRAIN_GENERATOR_IDS = {
    "completion.artificial_mask.v1": "tabur.query-row-latent-mixture.v1",
    "supervised.label_broadcast.v1": "tabur.supervised-query-row-latent-mixture.v1",
}


def _generator_id(profile: str) -> str:
    try:
        return _PRETRAIN_GENERATOR_IDS[profile]
    except KeyError as exc:
        raise ValueError(f"unsupported TabUR synthetic pretraining profile: {profile!r}") from exc


def _config(row_token_count: int) -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=row_token_count,
        max_features=256,
    )


def _new_model(*, seed: int, row_token_count: int, profile: str, device: torch.device) -> Any:
    torch.manual_seed(seed)
    return build_model(
        "tabu.query.row",
        config=_config(row_token_count),
        profile=profile,
        row_token_count=row_token_count,
    ).to(device)


def _episode_set(
    *,
    seed: int,
    worlds: int,
    rows: int,
    row_token_count: int,
    profile: str,
) -> tuple[Any, ...]:
    episodes: list[Any] = []
    for index in range(worlds):
        family = (
            "row_latent_linear"
            if index % 3 == 0
            else "row_latent_periodic"
            if index % 3 == 1
            else "row_latent_polynomial"
        )
        if profile == "completion.artificial_mask.v1":
            episode = make_query_row_synthetic_episode(
                seed=seed + index,
                rows=rows,
                row_token_count=row_token_count,
                world_id=f"pretrain-world-{index}",
                world_family=family,
            )
        elif profile == "supervised.label_broadcast.v1":
            episode = make_query_row_supervised_synthetic_episode(
                seed=seed + index,
                rows=rows,
                context_rows=max(2, rows // 2),
                row_token_count=row_token_count,
                world_id=f"pretrain-world-{index}",
                world_family=family,
            )
        else:
            raise ValueError(f"unsupported TabUR synthetic pretraining profile: {profile!r}")
        episodes.append(episode)
    return tuple(episodes)


def _loss(model: Any, episode: Any, profile: str) -> torch.Tensor:
    if profile == "completion.artificial_mask.v1":
        return _episode_loss(model, episode)
    return supervised_synthetic_episode_loss(model, episode)


@dataclass(frozen=True, slots=True)
class QueryRowPretrainingResult:
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
    rows: int
    worlds: int
    steps: int
    learning_rate: float
    loss_aggregation: str
    generator_id: str
    initial_loss: float
    final_loss: float
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None
    identity: str | None = None
    identity_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_query_row_pretrain_checkpoint(
    model: Any,
    path: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Save model tensors plus an embedded/sidecar identity before any resume."""

    if path.suffix != ".safetensors":
        raise ValueError("TabUR pretraining checkpoints must use .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_identity = dict(model.checkpoint_identity())
    identity = {
        "schema": "tabu.query-row-pretraining-checkpoint.v1",
        "model_identity": model_identity,
        "metadata": metadata,
    }
    identity["identity_hash"] = canonical_hash(identity)
    tensors = {
        f"model.{name}": value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    save_file(tensors, str(path), metadata={"identity": json.dumps(identity, sort_keys=True)})
    identity_path = path.with_suffix(".identity.json")
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "identity": str(identity_path),
        "identity_sha256": _sha256(identity_path),
    }


def load_query_row_pretrain_checkpoint(model: Any, path: Path) -> None:
    """Validate query contract identity and only then load tensors."""

    identity_path = path.with_suffix(".identity.json")
    if not identity_path.is_file():
        raise FileNotFoundError(f"TabUR checkpoint identity sidecar is required: {identity_path}")
    sidecar = json.loads(identity_path.read_text(encoding="utf-8"))
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        embedded = json.loads((checkpoint.metadata() or {}).get("identity", "{}"))
    if sidecar != embedded:
        raise ValueError("TabUR checkpoint sidecar identity does not match embedded identity")
    if sidecar.get("identity_hash") != canonical_hash(
        {key: value for key, value in sidecar.items() if key != "identity_hash"}
    ):
        raise ValueError("TabUR checkpoint identity hash is invalid")
    model.validate_checkpoint_identity(sidecar["model_identity"])
    tensors = load_file(str(path), device="cpu")
    model.load_state_dict(
        {name.removeprefix("model."): value for name, value in tensors.items()},
        strict=True,
    )


def train_query_row_synthetic_pretraining_model(
    *,
    profile: str = "completion.artificial_mask.v1",
    seed: int = 1729,
    rows: int = 32,
    worlds: int = 16,
    steps: int = 100,
    learning_rate: float = 1.0e-2,
    row_token_count: int = 4,
    device: str | torch.device = "cpu",
    output: Path | None = None,
) -> tuple[Any, QueryRowPretrainingResult]:
    """Train a fresh TabUR model on independent synthetic worlds.

    The model is returned alongside its diagnostic result so a caller can run
    a held-out frozen-ICL gate without serializing and reloading between the
    training and evaluation boundaries.
    """

    if rows < 3 or worlds <= 0 or steps <= 0 or row_token_count <= 0:
        raise ValueError("rows, worlds, steps and row_token_count must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    resolved_device = resolve_device(str(device))
    model = _new_model(
        seed=seed,
        row_token_count=row_token_count,
        profile=profile,
        device=resolved_device,
    )
    episodes = _episode_set(
        seed=seed + 10_000,
        worlds=worlds,
        rows=rows,
        row_token_count=row_token_count,
        profile=profile,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_loss = sum(
            float(_loss(model, episode, profile).item()) for episode in episodes
        ) / len(episodes)
    for step in range(steps):
        loss = _loss(model, episodes[step % len(episodes)], profile)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite TabUR synthetic pretraining loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        final_loss = sum(
            float(_loss(model, episode, profile).item()) for episode in episodes
        ) / len(episodes)
    generator_id = _generator_id(profile)
    checkpoint_info: dict[str, str] = {}
    if output is not None:
        metadata = {
            "profile_id": profile,
            "generator_id": generator_id,
            "seed": seed,
            "rows": rows,
            "worlds": worlds,
            "steps": steps,
            "learning_rate": learning_rate,
            "row_token_count": row_token_count,
            "device_class": resolved_device.type,
            "world_schedule": ("linear", "periodic", "polynomial"),
        }
        checkpoint_info = save_query_row_pretrain_checkpoint(model, output, metadata=metadata)
    if not math.isfinite(final_loss):
        raise RuntimeError("TabUR synthetic pretraining did not complete")
    result = QueryRowPretrainingResult(
        status="pass" if final_loss < initial_loss else "diagnostic_complete_no_decrease",
        evidence_status="local_unissued",
        claim_boundary=(
            "TabUR bounded synthetic pretraining diagnostic; checkpoint is profile- and "
            "contract-bound, not a foundation-model or accepted capability claim"
        ),
        model_id=model.model_id,
        contract_version=model.contract_version,
        profile_id=profile,
        model_spec_hash=model.model_spec_hash,
        row_token_count=row_token_count,
        device=str(resolved_device),
        seed=seed,
        rows=rows,
        worlds=worlds,
        steps=steps,
        learning_rate=learning_rate,
        loss_aggregation="mean_over_training_worlds",
        generator_id=generator_id,
        initial_loss=initial_loss,
        final_loss=final_loss,
        **checkpoint_info,
    )
    return model, result


def run_query_row_synthetic_pretraining(
    **kwargs: Any,
) -> QueryRowPretrainingResult:
    """Train TabUR synthetic worlds and return only the diagnostic result."""

    _, result = train_query_row_synthetic_pretraining_model(**kwargs)
    return result


__all__ = [
    "QueryRowPretrainingResult",
    "load_query_row_pretrain_checkpoint",
    "run_query_row_synthetic_pretraining",
    "save_query_row_pretrain_checkpoint",
    "train_query_row_synthetic_pretraining_model",
]
