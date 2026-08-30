"""Executable local-unissued scale and transfer helpers for TabUBase 0.2.0.

This module deliberately sits beside the frozen ``transfer-base-v1`` schema
loaders.  The schemas describe the evidence contract; the functions below
materialize deterministic supervised episodes and checkpoints without
promoting an exploratory run into a formal receipt.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from safetensors.torch import load_file, save_file

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
from tabu_lab.models import ReferenceConfig, TabUCellBaseProfile, build_model
from tabu_lab.models.components import CellTokenizer
from tabu_lab.training import Objective

from .tabubase_expanded_synthetic import (
    CONTEXT_CANDIDATE_INITIAL_ROWS,
    EXPANDED_SYNTHETIC_GENERATOR_VERSION,
    FROZEN_CONTEXT_ROWS_SCHEDULE,
    LONG_CONTEXT_CANDIDATE_ROWS,
    LONG_CONTEXT_ROWS_SCHEDULE,
    build_expanded_synthetic_episode,
    expanded_eligible_context_rows,
    expanded_training_context_rows,
)
from .tabubase_response_readout import query_response_objective_loss

SCALE_MODEL_CONFIG = ReferenceConfig(
    d_model=32,
    n_heads=4,
    d_ff=64,
    n_blocks=2,
    inducing_slots=4,
    matched_slots=4,
    max_features=64,
    dropout=0.0,
)
ROOT_SEEDS = (1729, 2718, 31415)
TRAIN_FAMILIES = ("sparse_scm", "tree_threshold", "latent_factor")
TARGET_TYPES = ("numeric", "binary", "ordinal", "categorical")
LEGACY_SYNTHETIC_GENERATOR_VERSION = "tabubase.synthetic-prior.v1"
SYNTHETIC_GENERATOR_VERSIONS = (
    LEGACY_SYNTHETIC_GENERATOR_VERSION,
    EXPANDED_SYNTHETIC_GENERATOR_VERSION,
)
DENSE_TRAINING_FORWARD_MODE = "fast_no_trace"
QUERY_RESPONSE_TRAINING_FORWARD_MODE = "query_response_only_v1"
TRAINING_FORWARD_MODES = (
    DENSE_TRAINING_FORWARD_MODE,
    QUERY_RESPONSE_TRAINING_FORWARD_MODE,
)
LONG_CONTEXT_PRETRAINING_PROTOCOL_ID = "tabubase.expanded-synthetic-long-context.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[3]
    candidates = sorted((root / "src" / "tabu_lab").rglob("*.py"))
    candidates += sorted((root / "specs" / "models").rglob("*.yaml"))
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit_or_none() -> str | None:
    declared = os.environ.get("TABU_SOURCE_COMMIT")
    if declared:
        return declared
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return completed.stdout.strip() or None


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _stable_seed(root_seed: int, namespace: str) -> int:
    payload = f"tabubase-scale-v1|{root_seed}|{namespace}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _standardize_from_context(values: torch.Tensor, context_rows: int) -> torch.Tensor:
    if context_rows == 0:
        # K=0 ICL is an explicit no-context condition.  There are no legal
        # context statistics to estimate, so preserve generator-space values
        # rather than silently borrowing a population or query statistic.
        return values
    evidence = values[:context_rows]
    mean = evidence.mean(dim=0, keepdim=True)
    scale = evidence.std(dim=0, unbiased=False, keepdim=True).clamp_min(1.0e-5)
    return (values - mean) / scale


def _feature_specs(response_kind: FeatureKind, response_classes: int) -> tuple[FeatureSpec, ...]:
    ordinal_domain = ("low", "middle", "high")
    nominal_domain = ("n0", "n1", "n2", "n3")
    predictors = tuple(FeatureSpec(name=f"numeric_{index}") for index in range(4))
    predictors += tuple(
        FeatureSpec(
            name=f"ordinal_{index}",
            kind=FeatureKind.ORDINAL,
            domain=ordinal_domain,
            codebook_id=f"tabubase-scale-ordinal-{index}-v1",
        )
        for index in range(2)
    )
    predictors += tuple(
        FeatureSpec(
            name=f"nominal_{index}",
            kind=FeatureKind.CATEGORICAL,
            domain=nominal_domain,
            codebook_id=f"tabubase-scale-nominal-{index}-v1",
        )
        for index in range(2)
    )
    if response_kind is FeatureKind.NUMERIC:
        response = FeatureSpec(name="response", role=FeatureRole.RESPONSE)
    else:
        response = FeatureSpec(
            name="response",
            kind=response_kind,
            domain=tuple(f"class_{index}" for index in range(response_classes)),
            codebook_id=f"tabubase-scale-response-{response_classes}-v1",
            role=FeatureRole.RESPONSE,
        )
    return (*predictors, response)


def build_synthetic_episode(
    *,
    root_seed: int,
    world_index: int,
    partition: Literal["train", "validation", "test"] = "train",
    context_rows: int = 64,
    query_rows: int = 64,
) -> tuple[EvidenceEpisode, TruthSidecar, dict[str, Any]]:
    """Build one mixed-type world using context-only numerical statistics."""

    if root_seed < 0 or world_index < 0 or context_rows < 0 or query_rows <= 0:
        raise ValueError("synthetic episode indices and row counts must be non-negative")
    family = TRAIN_FAMILIES[world_index % len(TRAIN_FAMILIES)]
    target_type = TARGET_TYPES[(world_index // len(TRAIN_FAMILIES)) % len(TARGET_TYPES)]
    generator = torch.Generator(device="cpu").manual_seed(
        _stable_seed(root_seed, f"{partition}:{world_index}:{family}:{target_type}")
    )
    rows = context_rows + query_rows
    latent = torch.randn((rows, 4), generator=generator, dtype=torch.float32)
    numeric = torch.stack(
        (
            latent[:, 0],
            latent[:, 1],
            latent[:, 2] + 0.25 * latent[:, 0],
            latent[:, 3] + 0.2 * latent[:, 1].square(),
        ),
        dim=1,
    )
    numeric = _standardize_from_context(numeric, context_rows)
    ordinal = torch.stack(
        (
            torch.bucketize(latent[:, 0].contiguous(), torch.tensor((-0.5, 0.5))),
            torch.bucketize(
                (latent[:, 1] + 0.2 * latent[:, 2]).contiguous(),
                torch.tensor((-0.5, 0.5)),
            ),
        ),
        dim=1,
    ).to(torch.float32)
    nominal = torch.stack(
        (
            (latent[:, 2] > 0).to(torch.int64) + 2 * (latent[:, 3] > 0).to(torch.int64),
            torch.remainder(
                torch.bucketize(
                    (latent[:, 0] - latent[:, 1]).contiguous(),
                    torch.tensor((-0.7, 0.0, 0.7)),
                ),
                4,
            ),
        ),
        dim=1,
    ).to(torch.float32)
    predictors = torch.cat((numeric, ordinal, nominal), dim=1)

    if family == "sparse_scm":
        score = 0.9 * numeric[:, 0] - 0.45 * numeric[:, 1].square()
        score = score + 0.35 * torch.tanh(numeric[:, 2]) + 0.2 * (nominal[:, 0] == 3)
    elif family == "tree_threshold":
        score = 0.8 * (numeric[:, 0] > 0).to(torch.float32)
        score = score - 0.6 * (numeric[:, 1] > 0.5).to(torch.float32)
        score = score + 0.35 * ordinal[:, 0] + 0.15 * numeric[:, 3]
    else:
        factor = 0.7 * numeric[:, 0] + 0.5 * numeric[:, 1]
        score = factor + 0.35 * torch.sin(numeric[:, 2] + factor)
        score = score + 0.15 * nominal[:, 1]

    response_kind = FeatureKind.NUMERIC
    response_classes = 0
    if target_type == "numeric":
        response = _standardize_from_context(score.unsqueeze(1), context_rows).squeeze(1)
    elif target_type == "binary":
        response_kind = FeatureKind.CATEGORICAL
        response_classes = 2
        response = (score > 0.0).to(torch.float32)
    elif target_type == "ordinal":
        response_kind = FeatureKind.ORDINAL
        response_classes = 4
        response = torch.bucketize(score.contiguous(), torch.tensor((-0.75, 0.0, 0.75))).to(
            torch.float32
        )
    else:
        response_kind = FeatureKind.CATEGORICAL
        response_classes = 4
        response = torch.remainder(
            torch.bucketize(score.contiguous(), torch.tensor((-0.75, 0.0, 0.75)))
            + nominal[:, 0].to(torch.int64),
            4,
        ).to(torch.float32)

    values = torch.cat((predictors, response.unsqueeze(1)), dim=1)
    feature_specs = _feature_specs(response_kind, response_classes)
    roles = torch.full(
        values.shape,
        int(ForwardRole.RECEIVER | ForwardRole.SOURCE),
        dtype=torch.int64,
    )
    origins = torch.full(
        values.shape,
        origin_code(OriginState.OBSERVED),
        dtype=torch.int64,
    )
    roles[context_rows:, -1] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    origins[context_rows:, -1] = origin_code(OriginState.QUERY)
    forward_values = values.clone()
    forward_values[context_rows:, -1] = 0.0
    episode_id = f"tabubase-scale-{partition}-{root_seed}-{world_index:06d}"
    row_ids = tuple(f"{episode_id}-row-{index:03d}" for index in range(rows))
    episode = EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="tabubase-synthetic-prior-v1",
        source_partition=partition,
        fit_partition="train",
        row_ids=row_ids,
        feature_names=tuple(spec.name for spec in feature_specs),
        feature_specs=feature_specs,
        forward_values=forward_values,
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "generator_family": family,
            "response_family": target_type,
            "statistics_scope": "context_only",
            "world_index": world_index,
        },
    )
    truth_values = torch.zeros_like(values)
    truth_values[context_rows:, -1] = response[context_rows:]
    truth = TruthSidecar(
        episode_id=episode_id,
        recipe_hash=canonical_hash(
            {
                "schema": "tabubase-scale-episode-recipe.v1",
                "root_seed": root_seed,
                "world_index": world_index,
                "partition": partition,
                "context_rows": context_rows,
                "query_rows": query_rows,
            }
        ),
        row_ids=row_ids,
        feature_names=episode.feature_names,
        target_values=truth_values,
        target_mask=episode.target_mask,
    )
    return (
        episode,
        truth,
        {
            "family": family,
            "target_type": target_type,
            "world_index": world_index,
        },
    )


def _build_pretraining_episode(
    *,
    generator_version: str,
    root_seed: int,
    world_index: int,
    partition: Literal["train", "validation", "test"] = "train",
    context_rows: int = 64,
    query_rows: int = 64,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
) -> tuple[EvidenceEpisode, TruthSidecar, dict[str, Any]]:
    if generator_version == LEGACY_SYNTHETIC_GENERATOR_VERSION:
        return build_synthetic_episode(
            root_seed=root_seed,
            world_index=world_index,
            partition=partition,
            context_rows=context_rows,
            query_rows=query_rows,
        )
    if generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION:
        return build_expanded_synthetic_episode(
            root_seed=root_seed,
            world_index=world_index,
            partition=partition,
            context_rows=context_rows,
            query_rows=query_rows,
            context_candidate_rows=context_candidate_rows,
        )
    raise ValueError(f"unknown synthetic generator version: {generator_version}")


def _training_context_rows(
    *,
    generator_version: str,
    world_index: int,
    update_index: int,
    context_rows_schedule: tuple[int, ...],
) -> int:
    if generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION:
        return expanded_training_context_rows(
            world_index=world_index,
            context_rows_schedule=context_rows_schedule,
        )
    return context_rows_schedule[update_index % len(context_rows_schedule)]


def _validation_episode_count(
    *,
    generator_version: str,
    worlds: int,
    context_rows_schedule: tuple[int, ...],
) -> int:
    if generator_version != EXPANDED_SYNTHETIC_GENERATOR_VERSION:
        return worlds
    return sum(
        len(
            expanded_eligible_context_rows(
                world_index=1_000_000 + index,
                context_rows_schedule=context_rows_schedule,
            )
        )
        for index in range(worlds)
    )


def build_tabubase_scale_model(
    *,
    seed: int,
    device: torch.device,
    nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
) -> torch.nn.Module:
    torch.manual_seed(_stable_seed(seed, "model-init"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(_stable_seed(seed, "model-init"))
    return build_model(
        "tabu.cell.base",
        config=SCALE_MODEL_CONFIG,
        profile=TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1,
        numeric_terminal="local_linear",
        nominal_tokenizer=nominal_tokenizer,
        nominal_codebook_size=nominal_codebook_size,
        nominal_codebook_seed=nominal_codebook_seed,
    ).to(device)


def pretrain_run_id(config: PretrainRunConfig) -> str:
    """Return a collision-free run identity while preserving historical v1 paths."""

    base = f"tabubase-{config.phase.lower()}-seed-{config.seed}"
    if config.nominal_tokenizer == CellTokenizer.EPISODE_RANDOM_SPHERE_V1:
        resolved = base
    else:
        resolved = (
            f"{base}-nominal-codebook-v2-b{config.nominal_codebook_size}"
            f"-s{config.nominal_codebook_seed}"
        )
    if config.context_rows_schedule != (64,):
        resolved += "-icl-kcurriculum-v1"
    if config.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION:
        resolved += "-expanded-synthetic-v4-val192-support-k-v1"
    if config.training_forward_mode == QUERY_RESPONSE_TRAINING_FORWARD_MODE:
        resolved += "-long-context-v1-k512-query-response-v1"
    return resolved


def save_pretrain_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    identity: dict[str, Any],
) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {f"model.{name}": value.detach().cpu() for name, value in model.state_dict().items()}
    save_file(tensors, str(path), metadata={"identity": json.dumps(identity, sort_keys=True)})
    identity_path = path.with_suffix(".identity.json")
    identity_path.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "identity": str(identity_path),
        "identity_sha256": _sha256_file(identity_path),
    }


def load_pretrain_checkpoint(model: torch.nn.Module, path: Path) -> None:
    identity_path = path.with_suffix(".identity.json")
    if identity_path.is_file():
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        model_identity = identity.get("model_identity")
        validate_identity = getattr(model, "validate_checkpoint_identity", None)
        if model_identity is not None and validate_identity is not None:
            validate_identity(model_identity)
    tensors = load_file(str(path), device="cpu")
    state = {name.removeprefix("model."): value for name, value in tensors.items()}
    model.load_state_dict(state, strict=True)


def _train_one(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    episode: EvidenceEpisode,
    truth: TruthSidecar,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    context_rows: int | None = None,
    training_forward_mode: str = DENSE_TRAINING_FORWARD_MODE,
    query_readout_chunk_rows: int = 64,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Training does not need the truth-free receipt trace on every update.
    # The public ``model(EvidenceEpisode)`` boundary remains unchanged; this
    # private implementation path avoids per-step SHA/CPU routing diagnostics
    # that otherwise force a CUDA synchronization.  The long-context protocol
    # uses the same shared dynamics but projects/routes only query response
    # cells; its dense equivalence is a separate preregistered gate.
    dense_input = episode.to(device)
    if training_forward_mode == QUERY_RESPONSE_TRAINING_FORWARD_MODE:
        if context_rows is None:
            declared_context_rows = episode.metadata.get("context_rows")
            if type(declared_context_rows) is not int:
                raise ValueError("query-response training requires declared context rows")
            context_rows = declared_context_rows
        loss = query_response_objective_loss(
            model,
            dense_input,
            truth.to(device),
            context_rows=context_rows,
            query_readout_chunk_rows=query_readout_chunk_rows,
        )
    elif training_forward_mode == DENSE_TRAINING_FORWARD_MODE:
        forward_dense = getattr(model, "_forward_dense", None)
        if forward_dense is None:
            prediction = model(dense_input)
        else:
            prediction = forward_dense(dense_input, emit_trace=False)
        loss = Objective()(prediction, truth.to(device)).total
    else:
        raise ValueError(f"unknown training forward mode: {training_forward_mode}")
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite pretraining loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach().cpu())


class SyntheticEpisodePrefetcher:
    """Overlap deterministic CPU episode compilation with model execution."""

    def __init__(
        self,
        *,
        root_seed: int,
        worlds: int,
        first_update: int,
        last_update: int,
        workers: int,
        queue_depth: int,
        context_rows_schedule: tuple[int, ...] = (64,),
        generator_version: str = LEGACY_SYNTHETIC_GENERATOR_VERSION,
        context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
    ) -> None:
        if workers < 1 or queue_depth < 1:
            raise ValueError("prefetch workers and queue depth must be positive")
        if first_update < 1 or last_update < first_update:
            raise ValueError("prefetch update range is invalid")
        self.root_seed = root_seed
        self.worlds = worlds
        self.next_update = first_update
        self.last_update = last_update
        self.context_rows_schedule = context_rows_schedule
        if generator_version not in SYNTHETIC_GENERATOR_VERSIONS:
            raise ValueError("unknown synthetic generator version")
        self.generator_version = generator_version
        self.context_candidate_rows = context_candidate_rows
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.futures: deque[Future[tuple[EvidenceEpisode, TruthSidecar, dict[str, Any]]]] = deque()
        for _ in range(min(queue_depth, last_update - first_update + 1)):
            self._submit_next()

    def _submit_next(self) -> None:
        if self.next_update > self.last_update:
            return
        update = self.next_update
        self.next_update += 1
        world_index = ((update - 1) * 7919 + self.root_seed) % self.worlds
        context_rows = _training_context_rows(
            generator_version=self.generator_version,
            world_index=world_index,
            update_index=update - 1,
            context_rows_schedule=self.context_rows_schedule,
        )
        self.futures.append(
            self.executor.submit(
                _build_pretraining_episode,
                generator_version=self.generator_version,
                root_seed=self.root_seed,
                world_index=world_index,
                context_rows=context_rows,
                context_candidate_rows=self.context_candidate_rows,
            )
        )

    def next(self) -> tuple[EvidenceEpisode, TruthSidecar, dict[str, Any]]:
        if not self.futures:
            raise RuntimeError("synthetic episode prefetch queue is exhausted")
        result = self.futures.popleft().result()
        self._submit_next()
        return result

    def close(self) -> None:
        self.executor.shutdown(wait=True)

    def __enter__(self) -> SyntheticEpisodePrefetcher:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def evaluate_synthetic(
    model: torch.nn.Module,
    *,
    root_seed: int,
    device: torch.device,
    worlds: int = 12,
    context_rows_schedule: tuple[int, ...] = (64,),
    generator_version: str = LEGACY_SYNTHETIC_GENERATOR_VERSION,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
    training_forward_mode: str = DENSE_TRAINING_FORWARD_MODE,
    query_readout_chunk_rows: int = 64,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for index in range(worlds):
            validation_contexts = (
                expanded_eligible_context_rows(
                    world_index=1_000_000 + index,
                    context_rows_schedule=context_rows_schedule,
                )
                if generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION
                else (context_rows_schedule[index % len(context_rows_schedule)],)
            )
            for context_rows in validation_contexts:
                episode, truth, _ = _build_pretraining_episode(
                    generator_version=generator_version,
                    root_seed=root_seed,
                    world_index=1_000_000 + index,
                    partition="validation",
                    context_rows=context_rows,
                    context_candidate_rows=context_candidate_rows,
                )
                dense_input = episode.to(device)
                if training_forward_mode == QUERY_RESPONSE_TRAINING_FORWARD_MODE:
                    loss = query_response_objective_loss(
                        model,
                        dense_input,
                        truth.to(device),
                        context_rows=context_rows,
                        query_readout_chunk_rows=query_readout_chunk_rows,
                    )
                elif training_forward_mode == DENSE_TRAINING_FORWARD_MODE:
                    prediction = model(dense_input)
                    loss = Objective()(prediction, truth.to(device)).total
                else:
                    raise ValueError(
                        f"unknown training forward mode: {training_forward_mode}"
                    )
                losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _exact_resume_probe(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    root_seed: int,
    world_index: int,
    device: torch.device,
    context_rows: int = 64,
    generator_version: str = LEGACY_SYNTHETIC_GENERATOR_VERSION,
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS,
    training_forward_mode: str = DENSE_TRAINING_FORWARD_MODE,
    query_readout_chunk_rows: int = 64,
) -> bool:
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    episode, truth, _ = _build_pretraining_episode(
        generator_version=generator_version,
        root_seed=root_seed,
        world_index=world_index,
        context_rows=context_rows,
        context_candidate_rows=context_candidate_rows,
    )
    expected_loss = _train_one(
        model,
        optimizer,
        episode,
        truth,
        device=device,
        gradient_clip_norm=1.0,
        context_rows=context_rows,
        training_forward_mode=training_forward_mode,
        query_readout_chunk_rows=query_readout_chunk_rows,
    )
    expected_hash = _state_hash(model)
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    actual_loss = _train_one(
        model,
        optimizer,
        episode,
        truth,
        device=device,
        gradient_clip_norm=1.0,
        context_rows=context_rows,
        training_forward_mode=training_forward_mode,
        query_readout_chunk_rows=query_readout_chunk_rows,
    )
    return expected_loss == actual_loss and expected_hash == _state_hash(model)


@dataclass(frozen=True, slots=True)
class PretrainRunConfig:
    phase: Literal["PT-S0", "PT-S1", "PT-S2"]
    worlds: int
    updates: int
    seed: int
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    checkpoint_updates: tuple[int, ...] = (0, 2_000)
    validation_worlds: int = 12
    prefetch_workers: int = 0
    prefetch_queue_depth: int = 8
    nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1
    nominal_codebook_size: int = 100
    nominal_codebook_seed: int = 1729
    context_rows_schedule: tuple[int, ...] = (64,)
    generator_version: str = LEGACY_SYNTHETIC_GENERATOR_VERSION
    context_candidate_rows: int = CONTEXT_CANDIDATE_INITIAL_ROWS
    training_forward_mode: str = DENSE_TRAINING_FORWARD_MODE
    query_readout_chunk_rows: int = 64

    def validate(self) -> PretrainRunConfig:
        expected = {
            "PT-S0": (2_048, 2_000),
            "PT-S1": (20_000, 20_000),
            "PT-S2": (200_000, 200_000),
        }[self.phase]
        if (self.worlds, self.updates) != expected:
            raise ValueError(
                f"{self.phase} is frozen to {expected[0]} worlds/{expected[1]} updates"
            )
        if self.seed not in ROOT_SEEDS:
            raise ValueError("pretraining seed is outside the frozen root seed set")
        if self.phase == "PT-S0" and self.seed != ROOT_SEEDS[0]:
            raise ValueError("PT-S0 is frozen to seed 1729")
        if self.checkpoint_updates[0] != 0 or self.checkpoint_updates[-1] != self.updates:
            raise ValueError("checkpoint ladder must begin at zero and end at the update budget")
        if self.prefetch_workers < 0 or self.prefetch_queue_depth < 1:
            raise ValueError("prefetch_workers must be non-negative and queue depth positive")
        if self.validation_worlds < 1:
            raise ValueError("validation_worlds must be positive")
        if self.nominal_tokenizer not in {
            CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
            CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
        }:
            raise ValueError("unknown nominal tokenizer plan")
        if self.nominal_codebook_size < 2 or self.nominal_codebook_seed < 0:
            raise ValueError("nominal codebook size must exceed one and seed be non-negative")
        if not self.context_rows_schedule or any(
            type(value) is not int or value < 2 for value in self.context_rows_schedule
        ):
            raise ValueError("pretraining context schedule requires integer K values >= 2")
        if len(set(self.context_rows_schedule)) != len(self.context_rows_schedule):
            raise ValueError("pretraining context schedule cannot repeat K values")
        if self.generator_version not in SYNTHETIC_GENERATOR_VERSIONS:
            raise ValueError("unknown synthetic generator version")
        if self.training_forward_mode not in TRAINING_FORWARD_MODES:
            raise ValueError("unknown pretraining forward mode")
        if (
            type(self.context_candidate_rows) is not int
            or self.context_candidate_rows < CONTEXT_CANDIDATE_INITIAL_ROWS
        ):
            raise ValueError("context candidate rows must be an integer >= 64")
        if (
            type(self.query_readout_chunk_rows) is not int
            or self.query_readout_chunk_rows < 1
        ):
            raise ValueError("query readout chunk rows must be a positive integer")
        if max(self.context_rows_schedule) > self.context_candidate_rows:
            raise ValueError("context schedule exceeds the frozen context candidate bank")
        if self.generator_version == LEGACY_SYNTHETIC_GENERATOR_VERSION and (
            self.context_candidate_rows != CONTEXT_CANDIDATE_INITIAL_ROWS
            or self.training_forward_mode != DENSE_TRAINING_FORWARD_MODE
        ):
            raise ValueError("legacy synthetic prior only supports the dense 64-row protocol")
        if self.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION:
            if self.phase == "PT-S2":
                raise ValueError("expanded synthetic v4 has no preregistered PT-S2 phase")
            if self.nominal_tokenizer != CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2:
                raise ValueError("expanded synthetic v4 requires the frozen source codebook")
            if (self.nominal_codebook_size, self.nominal_codebook_seed) != (100, 1729):
                raise ValueError("expanded synthetic v4 is frozen to codebook B=100 seed=1729")
            current_protocol = (
                self.context_rows_schedule == FROZEN_CONTEXT_ROWS_SCHEDULE
                and self.context_candidate_rows == CONTEXT_CANDIDATE_INITIAL_ROWS
                and self.training_forward_mode == DENSE_TRAINING_FORWARD_MODE
            )
            long_context_protocol = (
                self.context_rows_schedule == LONG_CONTEXT_ROWS_SCHEDULE
                and self.context_candidate_rows == LONG_CONTEXT_CANDIDATE_ROWS
                and self.training_forward_mode == QUERY_RESPONSE_TRAINING_FORWARD_MODE
                and self.query_readout_chunk_rows == 64
            )
            if not (current_protocol or long_context_protocol):
                raise ValueError(
                    "expanded synthetic v4 requires either the frozen K<=64 dense "
                    "protocol or the preregistered K<=512 query-response protocol"
                )
            if self.validation_worlds != 192:
                raise ValueError("expanded synthetic v4 requires 192 validation worlds")
            if (self.learning_rate, self.weight_decay) != (3.0e-4, 1.0e-4):
                raise ValueError("expanded synthetic v4 changed the frozen optimizer settings")
        return self


def run_pretraining(
    config: PretrainRunConfig,
    *,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute one deterministic local-unissued pretraining seed."""

    config.validate()
    run_id = pretrain_run_id(config)
    output_dir = output_root / run_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_tabubase_scale_model(
        seed=config.seed,
        device=device,
        nominal_tokenizer=config.nominal_tokenizer,
        nominal_codebook_size=config.nominal_codebook_size,
        nominal_codebook_seed=config.nominal_codebook_seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    model_identity = model.checkpoint_identity()
    pretraining_protocol_id = (
        LONG_CONTEXT_PRETRAINING_PROTOCOL_ID
        if config.training_forward_mode == QUERY_RESPONSE_TRAINING_FORWARD_MODE
        else config.generator_version
    )
    identity = {
        "schema_version": "tabu.transfer-base-local-unissued-checkpoint.v1",
        "contract_id": "tabu.cell.base",
        "contract_version": "0.2.0",
        "profile_id": TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1.value,
        "tokenizer_version": model_identity["tokenizer_version"],
        "model_identity": model_identity,
        "phase": config.phase,
        "seed": config.seed,
        "worlds": config.worlds,
        "updates": config.updates,
        "training_forward_mode": config.training_forward_mode,
        "pretraining_protocol_id": pretraining_protocol_id,
        "context_candidate_rows": config.context_candidate_rows,
        "query_readout_chunk_rows": config.query_readout_chunk_rows,
        "prefetch_workers": config.prefetch_workers,
        "prefetch_queue_depth": config.prefetch_queue_depth,
        "context_rows_schedule": config.context_rows_schedule,
        "generator_version": config.generator_version,
        "validation_worlds": config.validation_worlds,
        "model_config": {
            name: getattr(value, "value", value)
            for name, value in SCALE_MODEL_CONFIG.__dict__.items()
        },
        "source_tree_sha256": source_tree_sha256(),
        "source_status": "local_unissued",
    }
    if model_identity["tokenizer_version"] == "cell-tokenizer.v2":
        identity.update(
            {
                "nominal_tokenizer": config.nominal_tokenizer,
                "nominal_codebook_size": config.nominal_codebook_size,
                "nominal_codebook_seed": config.nominal_codebook_seed,
                "nominal_codebook_hash": model_identity["nominal_codebook_hash"],
            }
        )
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint_records.append(
        {
            "update": 0,
            **save_pretrain_checkpoint(
                model,
                output_dir / "checkpoint-00000.safetensors",
                identity=identity | {"update": 0},
            ),
        }
    )
    initial_validation = evaluate_synthetic(
        model,
        root_seed=config.seed,
        device=device,
        worlds=config.validation_worlds,
        context_rows_schedule=config.context_rows_schedule,
        generator_version=config.generator_version,
        context_candidate_rows=config.context_candidate_rows,
        training_forward_mode=config.training_forward_mode,
        query_readout_chunk_rows=config.query_readout_chunk_rows,
    )
    started = time.monotonic()
    losses: list[dict[str, float | int]] = []
    exact_resume = False
    finite = True
    checkpoint_updates = set(config.checkpoint_updates[1:])
    prefetcher = (
        SyntheticEpisodePrefetcher(
            root_seed=config.seed,
            worlds=config.worlds,
            first_update=1,
            last_update=config.updates,
            workers=config.prefetch_workers,
            queue_depth=config.prefetch_queue_depth,
            context_rows_schedule=config.context_rows_schedule,
            generator_version=config.generator_version,
            context_candidate_rows=config.context_candidate_rows,
        )
        if config.prefetch_workers
        else None
    )
    try:
        for update in range(1, config.updates + 1):
            world_index = ((update - 1) * 7919 + config.seed) % config.worlds
            context_rows = _training_context_rows(
                generator_version=config.generator_version,
                world_index=world_index,
                update_index=update - 1,
                context_rows_schedule=config.context_rows_schedule,
            )
            if prefetcher is not None:
                episode, truth, _ = prefetcher.next()
            else:
                episode, truth, _ = _build_pretraining_episode(
                    generator_version=config.generator_version,
                    root_seed=config.seed,
                    world_index=world_index,
                    context_rows=context_rows,
                    context_candidate_rows=config.context_candidate_rows,
                )
            if update == min(10, config.updates):
                exact_resume = _exact_resume_probe(
                    model,
                    optimizer,
                    root_seed=config.seed,
                    world_index=world_index,
                    device=device,
                    context_rows=context_rows,
                    generator_version=config.generator_version,
                    context_candidate_rows=config.context_candidate_rows,
                    training_forward_mode=config.training_forward_mode,
                    query_readout_chunk_rows=config.query_readout_chunk_rows,
                )
                loss = math.nan
            else:
                loss = _train_one(
                    model,
                    optimizer,
                    episode,
                    truth,
                    device=device,
                    gradient_clip_norm=1.0,
                    context_rows=context_rows,
                    training_forward_mode=config.training_forward_mode,
                    query_readout_chunk_rows=config.query_readout_chunk_rows,
                )
            if not math.isnan(loss):
                finite = finite and math.isfinite(loss)
            if update == 1 or update % 100 == 0:
                losses.append(
                    {
                        "update": update,
                        "loss": loss,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                if update == 1 or update % 1_000 == 0:
                    print(
                        f"{config.phase} seed={config.seed} update={update}/{config.updates} "
                        f"elapsed={time.monotonic() - started:.1f}s loss={loss:.6f}",
                        flush=True,
                    )
            if update in checkpoint_updates:
                checkpoint_records.append(
                    {
                        "update": update,
                        **save_pretrain_checkpoint(
                            model,
                            output_dir / f"checkpoint-{update:05d}.safetensors",
                            identity=identity | {"update": update},
                        ),
                    }
                )
    finally:
        if prefetcher is not None:
            prefetcher.close()
    final_validation = evaluate_synthetic(
        model,
        root_seed=config.seed,
        device=device,
        worlds=config.validation_worlds,
        context_rows_schedule=config.context_rows_schedule,
        generator_version=config.generator_version,
        context_candidate_rows=config.context_candidate_rows,
        training_forward_mode=config.training_forward_mode,
        query_readout_chunk_rows=config.query_readout_chunk_rows,
    )
    validation_improvement = final_validation < initial_validation
    gates = {
        "finite": finite,
        "exact_resume": exact_resume,
        "validation_improvement": validation_improvement,
    }
    receipt = {
        "schema_version": "tabu.transfer-base-local-unissued-pretrain-result.v1",
        "status": "local_unissued",
        "run_id": run_id,
        "phase": config.phase,
        "seed": config.seed,
        "worlds": config.worlds,
        "updates": config.updates,
        "batch_semantics": "64 query rows per synthetic world episode",
        "context_rows_schedule": list(config.context_rows_schedule),
        "context_candidate_rows": config.context_candidate_rows,
        "generator_version": config.generator_version,
        "pretraining_protocol_id": pretraining_protocol_id,
        "training_forward_mode": config.training_forward_mode,
        "query_readout_chunk_rows": config.query_readout_chunk_rows,
        "validation_worlds": config.validation_worlds,
        "validation_context_policy": (
            "every_world_at_every_support_realizable_K"
            if config.generator_version == EXPANDED_SYNTHETIC_GENERATOR_VERSION
            else "one_K_per_world_cyclic"
        ),
        "validation_episodes": _validation_episode_count(
            generator_version=config.generator_version,
            worlds=config.validation_worlds,
            context_rows_schedule=config.context_rows_schedule,
        ),
        "source_tree_sha256": identity["source_tree_sha256"],
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "prefetch_workers": config.prefetch_workers,
        "prefetch_queue_depth": config.prefetch_queue_depth,
        "tokenizer_version": model_identity["tokenizer_version"],
        "nominal_tokenizer": config.nominal_tokenizer,
        "nominal_codebook_size": config.nominal_codebook_size,
        "nominal_codebook_seed": config.nominal_codebook_seed,
        "nominal_codebook_hash": model_identity.get("nominal_codebook_hash"),
        "initial_validation_loss": initial_validation,
        "final_validation_loss": final_validation,
        "gates": gates,
        "passed": all(gates.values()),
        "loss_history": losses,
        "checkpoints": checkpoint_records,
        "final_model_state_sha256": _state_hash(model),
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
        "git_commit": _git_commit_or_none(),
        "claim_boundary": (
            "exploratory mechanism evidence only; not a formal receipt or foundation-model claim"
        ),
        "exact_resume_probe_scope": (
            "single_step_in_memory_model_optimizer_and_rng_restore; not persisted checkpoint resume"
        ),
    }
    receipt_path = output_dir / "result.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt | {"result_path": str(receipt_path), "result_sha256": _sha256_file(receipt_path)}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


__all__ = [
    "CONTEXT_CANDIDATE_INITIAL_ROWS",
    "DENSE_TRAINING_FORWARD_MODE",
    "EXPANDED_SYNTHETIC_GENERATOR_VERSION",
    "LEGACY_SYNTHETIC_GENERATOR_VERSION",
    "LONG_CONTEXT_CANDIDATE_ROWS",
    "LONG_CONTEXT_PRETRAINING_PROTOCOL_ID",
    "LONG_CONTEXT_ROWS_SCHEDULE",
    "QUERY_RESPONSE_TRAINING_FORWARD_MODE",
    "ROOT_SEEDS",
    "SCALE_MODEL_CONFIG",
    "SYNTHETIC_GENERATOR_VERSIONS",
    "TRAINING_FORWARD_MODES",
    "PretrainRunConfig",
    "build_synthetic_episode",
    "build_tabubase_scale_model",
    "evaluate_synthetic",
    "load_pretrain_checkpoint",
    "pretrain_run_id",
    "resolve_device",
    "run_pretraining",
    "save_pretrain_checkpoint",
    "source_tree_sha256",
]
