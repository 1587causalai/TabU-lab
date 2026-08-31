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

from .query_row_identity import (
    query_row_result_identity,
    require_query_row_readout_identity,
)
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

_CHECKPOINT_SCHEMA = "tabu.query-row-pretraining-checkpoint.v2"


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


def _new_model(
    *,
    seed: int,
    row_token_count: int,
    profile: str,
    row_readout_mode: str,
    anchored_gamma_initial: float,
    device: torch.device,
) -> Any:
    torch.manual_seed(seed)
    return build_model(
        "tabu.query.row",
        config=_config(row_token_count),
        profile=profile,
        row_token_count=row_token_count,
        row_readout_mode=row_readout_mode,
        anchored_gamma_initial=anchored_gamma_initial,
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
    variant_hash: str
    row_token_count: int
    row_readout_mode: str
    row_readout_identity: dict[str, Any]
    checkpoint_kind: str
    training_resume_supported: bool
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


def _validate_checkpoint_envelope(
    identity: Any,
    *,
    source: str,
) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"TabUR checkpoint {source} identity must be an object")
    if identity.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError(
            f"TabUR checkpoint {source} schema must be {_CHECKPOINT_SCHEMA}; "
            "legacy v1 checkpoints are not migrated"
        )
    expected_hash = canonical_hash(
        {key: value for key, value in identity.items() if key != "identity_hash"}
    )
    if identity.get("identity_hash") != expected_hash:
        raise ValueError(f"TabUR checkpoint {source} identity hash is invalid")
    model_identity = identity.get("model_identity")
    if not isinstance(model_identity, dict):
        raise ValueError(f"TabUR checkpoint {source} model_identity is required")
    readout = require_query_row_readout_identity(model_identity)
    result_identity = query_row_result_identity(model_identity)
    metadata = identity.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"TabUR checkpoint {source} metadata is required")
    if metadata.get("row_readout_mode") != readout["mode"]:
        raise ValueError(
            f"TabUR checkpoint {source} metadata.row_readout_mode is required "
            "and must match model_identity"
        )
    if canonical_hash(metadata.get("row_readout_identity")) != canonical_hash(readout):
        raise ValueError(
            f"TabUR checkpoint {source} metadata.row_readout_identity is required "
            "and must match model_identity"
        )
    if metadata.get("variant_hash") != result_identity["variant_hash"]:
        raise ValueError(
            f"TabUR checkpoint {source} metadata.variant_hash is required "
            "and must match model_identity"
        )
    if metadata.get("checkpoint_kind") != "weights_only_transfer_snapshot":
        raise ValueError(
            f"TabUR checkpoint {source} metadata.checkpoint_kind must identify a "
            "weights-only transfer snapshot"
        )
    if metadata.get("training_resume_supported") is not False:
        raise ValueError(
            f"TabUR checkpoint {source} metadata.training_resume_supported must be false"
        )
    return identity


def read_query_row_pretrain_checkpoint_identity(path: Path) -> dict[str, Any]:
    """Read and validate the v2 sidecar before reconstructing a model."""

    identity_path = path.with_suffix(".identity.json")
    if not identity_path.is_file():
        raise FileNotFoundError(f"TabUR checkpoint identity sidecar is required: {identity_path}")
    try:
        sidecar = json.loads(identity_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("TabUR checkpoint sidecar identity is not valid JSON") from exc
    return _validate_checkpoint_envelope(sidecar, source="sidecar")


def save_query_row_pretrain_checkpoint(
    model: Any,
    path: Path,
    *,
    metadata: dict[str, Any],
) -> dict[str, str]:
    """Save an identity-bound weights snapshot for transfer/evaluation only.

    This artifact intentionally omits optimizer, update cursor, and RNG state;
    it is not a deterministic training-resume checkpoint.
    """

    if path.suffix != ".safetensors":
        raise ValueError("TabUR pretraining checkpoints must use .safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_identity = dict(model.checkpoint_identity())
    readout_identity = require_query_row_readout_identity(model_identity)
    result_identity = query_row_result_identity(model_identity)
    checkpoint_metadata = dict(metadata)
    supplied_mode = checkpoint_metadata.get("row_readout_mode", readout_identity["mode"])
    if supplied_mode != readout_identity["mode"]:
        raise ValueError("checkpoint metadata row_readout_mode conflicts with model identity")
    supplied_identity = checkpoint_metadata.get("row_readout_identity", readout_identity)
    if canonical_hash(supplied_identity) != canonical_hash(readout_identity):
        raise ValueError("checkpoint metadata row_readout_identity conflicts with model identity")
    supplied_kind = checkpoint_metadata.get(
        "checkpoint_kind", "weights_only_transfer_snapshot"
    )
    if supplied_kind != "weights_only_transfer_snapshot":
        raise ValueError("checkpoint metadata checkpoint_kind conflicts with artifact scope")
    supplied_resume = checkpoint_metadata.get("training_resume_supported", False)
    if supplied_resume is not False:
        raise ValueError(
            "checkpoint metadata training_resume_supported conflicts with artifact scope"
        )
    checkpoint_metadata.update(
        {
            "row_readout_mode": readout_identity["mode"],
            "row_readout_identity": readout_identity,
            "variant_hash": result_identity["variant_hash"],
            "checkpoint_kind": "weights_only_transfer_snapshot",
            "training_resume_supported": False,
        }
    )
    identity = {
        "schema": _CHECKPOINT_SCHEMA,
        "model_identity": model_identity,
        "metadata": checkpoint_metadata,
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

    sidecar = read_query_row_pretrain_checkpoint_identity(path)
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        embedded_raw = (checkpoint.metadata() or {}).get("identity")
    if embedded_raw is None:
        raise ValueError("TabUR checkpoint embedded identity is required")
    try:
        embedded = json.loads(embedded_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("TabUR checkpoint embedded identity is not valid JSON") from exc
    _validate_checkpoint_envelope(embedded, source="embedded")
    if sidecar != embedded:
        raise ValueError("TabUR checkpoint sidecar identity does not match embedded identity")
    checkpoint_readout = require_query_row_readout_identity(sidecar["model_identity"])
    model_readout = require_query_row_readout_identity(model.checkpoint_identity())
    if canonical_hash(checkpoint_readout) != canonical_hash(model_readout):
        raise ValueError("TabUR checkpoint row_readout does not match the target model")
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
    row_readout_mode: str = "anchored",
    anchored_gamma_initial: float = 1.0e-2,
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
        row_readout_mode=row_readout_mode,
        anchored_gamma_initial=anchored_gamma_initial,
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
    result_identity = query_row_result_identity(model.checkpoint_identity())
    row_readout_identity = result_identity["row_readout_identity"]
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
            "row_readout_mode": row_readout_identity["mode"],
            "row_readout_identity": row_readout_identity,
            "variant_hash": result_identity["variant_hash"],
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
            "contract-bound weights only (not a training-resume state), not a "
            "foundation-model or accepted capability claim"
        ),
        model_id=model.model_id,
        contract_version=model.contract_version,
        profile_id=profile,
        model_spec_hash=model.model_spec_hash,
        variant_hash=result_identity["variant_hash"],
        row_token_count=row_token_count,
        row_readout_mode=str(row_readout_identity["mode"]),
        row_readout_identity=row_readout_identity,
        checkpoint_kind="weights_only_transfer_snapshot",
        training_resume_supported=False,
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
    "read_query_row_pretrain_checkpoint_identity",
    "run_query_row_synthetic_pretraining",
    "save_query_row_pretrain_checkpoint",
    "train_query_row_synthetic_pretraining_model",
]
