"""Scorer-only truth preflight, typed/state-aware loss reduction, batch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import RestorationInput, RestorationRequest, TruthSidecar, validate_values
from .losses import encoding_mse
from .model import RestorationModel, RestorationOutput


@dataclass(frozen=True)
class LossConfig:
    # None is canonical type-internal mean. Explicit G/Q/Z/B weights opt in to
    # sum_s gamma_s * mean_{type,state=s}; it does not change mask sampling.
    state_weights: tuple[float, float, float, float] | None = None

    def __post_init__(self):
        if self.state_weights is not None:
            w = torch.tensor(self.state_weights, dtype=torch.float64)
            if w.shape != (4,) or not bool(torch.isfinite(w).all()) or bool((w < 0).any()):
                raise ValueError("state weights must be four finite nonnegative values")
            if not bool((w > 0).any()):
                raise ValueError("at least one state weight must be positive")


@dataclass(frozen=True)
class EpisodeScore:
    loss: Tensor
    per_target: Tensor
    by_state: dict[str, dict[str, float | int]]
    output: RestorationOutput


def _preflight(inputs, request, truth, facts):
    request.validate(inputs)
    targets = request.targets
    if not len(targets):
        raise ValueError("training requires nonempty targets")
    if (
        truth.states.shape != inputs.visible.shape
        or truth.states.dtype != torch.long
        or truth.states.device != targets.device
        or len(truth.values) != len(inputs.schema)
        or bool(((truth.states < -1) | (truth.states > 3)).any())
    ):
        raise ValueError("invalid truth sidecar shape, device, or state")
    expected_visible = (truth.states == 0) | (truth.states == 3)
    if not torch.equal(inputs.visible, expected_visible) or not torch.equal(
        inputs.query, truth.states == 1
    ):
        raise ValueError("truth state and input roles disagree")
    if bool((truth.states[targets[:, 0], targets[:, 1]] < 0).any()):
        raise ValueError("requested training target has no original truth")
    observed = (truth.states >= 0).nonzero()
    # Arbitrary request subsets are an inference feature, not a way to omit
    # unscorable truth, unsupported columns, or difficult damaged targets.
    if len(targets) != len(observed):
        raise ValueError("invalid-episode: training targets must cover all original observations")
    encoded = {}
    for a, schema in enumerate(inputs.schema):
        values = truth.values[a]
        if values.shape != (len(inputs.visible),) or values.device != targets.device:
            raise ValueError("truth columns must align with input rows and device")
        positions = (targets[:, 1] == a).nonzero().flatten()
        if not len(positions):
            continue
        if len(facts[a].rows) < 2:
            raise ValueError("invalid-episode: every supervised column needs two visible supports")
        selected = values[targets[positions, 0]]
        validate_values(schema, selected)
        # Missing clean class invalidates the WHOLE episode, before neural forward.
        encoded[a] = facts[a].answers.encode_targets(selected)
    return encoded


def score_episode(
    model: RestorationModel,
    inputs: RestorationInput,
    request: RestorationRequest,
    truth: TruthSidecar,
    loss_config: LossConfig | None = None,
) -> EpisodeScore:
    loss_config = loss_config or LossConfig()
    facts = model.encoder.prepare(inputs)
    encoded_truth = _preflight(inputs, request, truth, facts)
    output = model._forward_prepared(inputs, request, facts)
    targets = request.targets
    per_target = output.carriers.new_zeros(len(targets), dtype=torch.float64)
    numeric = torch.tensor(
        [inputs.schema[a].kind == "numeric" for a in targets[:, 1].tolist()],
        dtype=torch.bool,
        device=targets.device,
    )
    for col in output.columns:
        per_target = per_target.index_copy(
            0, col.target_indices, encoding_mse(col.result.encoding, encoded_truth[col.column])
        )
    states = truth.states[targets[:, 0], targets[:, 1]]
    loss = per_target.sum() * 0
    for branch in (numeric, ~numeric):
        if loss_config.state_weights is None:
            if bool(branch.any()):
                loss = loss + per_target[branch].mean()
        else:
            for state, weight in enumerate(loss_config.state_weights):
                selected = branch & (states == state)
                if bool(selected.any()):
                    loss = loss + weight * per_target[selected].mean()
    by_state = {}
    for state, name in enumerate(("retained", "query", "null", "corrupted")):
        selected = states == state
        by_state[name] = {"count": int(selected.sum())}
        if bool(selected.any()):
            by_state[name]["encoding_mse"] = float(per_target[selected].detach().mean())
    return EpisodeScore(loss, per_target, by_state, output)


def batch_loss(model: RestorationModel, episodes, loss_config: LossConfig | None = None):
    """Equal episode weight; no silent skipping/retry of invalid episodes."""
    scores = [score_episode(model, *episode, loss_config=loss_config) for episode in episodes]
    if not scores:
        raise ValueError("batch must contain at least one episode")
    return torch.stack([score.loss for score in scores]).mean(), scores
