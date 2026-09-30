"""Stage-3 synthetic-data fitting gate for the TabUBase anchor.

The generator is deliberately small and fixed: two visible numeric causes
produce one numeric response through a noisy linear law. The response truth is
kept in this module's sidecar and never placed in ``DenseModelInput.values``
at masked target cells.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from tabu_lab.models import build_model
from tabu_lab.models.types import DenseModelInput, ReferenceConfig


@dataclass(frozen=True)
class SyntheticWorldBatch:
    """A truth-free model carrier plus its evaluation-only response sidecar."""

    model_input: DenseModelInput
    response_truth: Tensor
    target_mask: Tensor
    generator_id: str = "tabubase.linear-world.v1"


@dataclass(frozen=True)
class SyntheticFitResult:
    """JSON-friendly local result; this is not a formal immutable receipt."""

    status: str
    evidence_status: str
    claim_boundary: str
    model_id: str
    contract_version: str
    profile_id: str
    model_spec_hash: str
    generator_id: str
    seed: int
    steps: int
    learning_rate: float
    initial_train_loss: float
    final_train_loss: float
    initial_validation_loss: float
    final_validation_loss: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_linear_world_batch(
    *,
    seed: int,
    batch_size: int = 8,
    rows: int = 12,
    mask_ratio: float = 0.35,
    noise_scale: float = 0.1,
) -> SyntheticWorldBatch:
    """Create one deterministic train/validation world."""

    if batch_size <= 0 or rows <= 0:
        raise ValueError("batch_size and rows must be positive")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0, 1)")
    if noise_scale < 0.0:
        raise ValueError("noise_scale must be non-negative")
    generator = torch.Generator().manual_seed(seed)
    causes = torch.randn(batch_size, rows, 2, generator=generator)
    response = (
        0.8 * causes[:, :, 0]
        - 0.4 * causes[:, :, 1]
        + noise_scale * torch.randn(batch_size, rows, generator=generator)
    )
    full_values = torch.cat((causes, response.unsqueeze(-1)), dim=-1)
    visible = torch.ones_like(full_values, dtype=torch.bool)
    response_targets = torch.rand(batch_size, rows, generator=generator) < mask_ratio
    visible[:, :, 2] = ~response_targets
    target = torch.zeros_like(visible)
    target[:, :, 2] = response_targets
    forward_values = full_values.masked_fill(~visible, 0.0)
    model_input = DenseModelInput(
        forward_values,
        visible,
        target,
        torch.zeros_like(target),
        episode_id=f"synthetic-linear-{seed}",
    )
    return SyntheticWorldBatch(model_input, response, response_targets)


def _masked_numeric_loss(model: Any, batch: SyntheticWorldBatch) -> Tensor:
    prediction = model._forward_dense(batch.model_input)
    scale = prediction.auxiliaries["numeric_context_scale"][:, :, 2]
    mean = prediction.auxiliaries["numeric_context_mean"][:, :, 2]
    standardized_truth = (batch.response_truth - mean) / scale
    predicted = prediction.outputs["numeric"][:, :, 2]
    mask = batch.target_mask
    return ((predicted[mask] - standardized_truth[mask]) ** 2).mean()


def run_synthetic_fit(
    *,
    seed: int = 1729,
    steps: int = 80,
    learning_rate: float = 1.0e-2,
    config: ReferenceConfig | None = None,
) -> SyntheticFitResult:
    """Fit a fresh TabUBase instance on one fixed synthetic world."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    torch.manual_seed(seed)
    train = make_linear_world_batch(seed=seed + 1)
    validation = make_linear_world_batch(seed=seed + 2)
    model = build_model(
        "tabu.cell.base",
        config=config
        or ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
            max_features=3,
        ),
        profile="completion.artificial_mask.v1",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with torch.no_grad():
        initial_validation = float(_masked_numeric_loss(model, validation).item())
    initial_train: float | None = None
    final_train = float("nan")
    for _ in range(steps):
        loss = _masked_numeric_loss(model, train)
        if initial_train is None:
            initial_train = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_train = float(loss.item())
    model.eval()
    with torch.no_grad():
        final_validation = float(_masked_numeric_loss(model, validation).item())
    status = (
        "pass"
        if torch.isfinite(torch.tensor(final_train)) and final_train < initial_train
        else "kill"
    )
    return SyntheticFitResult(
        status=status,
        evidence_status="local_unissued",
        claim_boundary=(
            "synthetic linear-world fitting only; no real-data, ICL, or "
            "foundation-model claim"
        ),
        model_id=model.model_id,
        contract_version=model.contract_version,
        profile_id=model.profile.value,
        model_spec_hash=model.model_spec_hash,
        generator_id=train.generator_id,
        seed=seed,
        steps=steps,
        learning_rate=learning_rate,
        initial_train_loss=float(initial_train),
        final_train_loss=final_train,
        initial_validation_loss=initial_validation,
        final_validation_loss=final_validation,
    )


__all__ = [
    "SyntheticFitResult",
    "SyntheticWorldBatch",
    "make_linear_world_batch",
    "run_synthetic_fit",
]
