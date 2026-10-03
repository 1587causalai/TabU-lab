"""Scorer-only coder-space losses; the only V7 code that reads Query truth."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..restoration._validation import finite
from ..restoration.contracts import TruthSidecar
from .config import V7Config, round_weights
from .model import V7Episode, V7Output


@dataclass(frozen=True)
class V7Score:
    loss: Tensor  # one output loss by default; geometric sum for explicit K > 1
    round_losses: Tensor  # [K], L_t on each written state
    initial_loss: Tensor  # L_0 on the initial Query state; diagnostic, not trained
    weights: tuple[float, ...]


def reference_values(episode: V7Episode, truth: TruthSidecar) -> Tensor:
    return truth.values[episode.target][episode.query_rows]


def state_loss(state: Tensor, reference: Tensor, chi: float) -> Tensor:
    """``(chi/p) ||C - C_ref||^2`` averaged over Query rows; ``J_r = {tau}``."""
    return (chi / state.shape[-1]) * (state - reference).square().sum(-1).mean()


def score_rounds(
    output: V7Output, episode: V7Episode, reference: Tensor, config: V7Config
) -> V7Score:
    """Score every written state against one fixed reference and denominator.

    The default K=1 has weight 1 for every rho and is exactly the main
    design's single-output loss. Multi-round weights apply only to controls.

    A hidden nominal target category without a visible code raises the
    ``no-answer-code`` protocol status; training admission must exclude it
    before the episode is built, not by dropping cells here.
    """
    column = episode.codec.columns[episode.target]
    codes = column.encode(reference).to(output.initial)
    chi = config.chi_numeric if column.kind == "numeric" else config.chi_discrete
    losses = torch.stack([state_loss(state, codes, chi) for state in output.states])
    weights = round_weights(len(output.states), config.round_loss_rho)
    loss = (losses * losses.new_tensor(weights)).sum()
    finite(loss, "V7 weighted round loss")
    return V7Score(loss, losses, state_loss(output.initial, codes, chi).detach(), weights)


__all__ = ["V7Score", "reference_values", "score_rounds", "state_loss"]
