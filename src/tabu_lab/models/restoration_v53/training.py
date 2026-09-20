"""Scorer-only truth and the V5.3 chi_numeric=128 loss normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..restoration._prepared import TensorVersions
from ..restoration._validation import finite, positive
from ..restoration.losses import encoding_mse
from ..restoration.training import LossConfig as LegacyLossConfig
from ..restoration.training import _masked_mean, _preflight
from .encoding import ANSWER_WIDTH
from .model import PreparedV53, V53Model, V53Output


@dataclass(frozen=True)
class V53LossConfig:
    discrete_weight: float = 1.0
    # Query coefficient 1; optional retained reconstruction is explicit.
    # None preserves the historical mixed-state mean for controlled comparisons.
    state_weights: tuple[float, float, float, float] | None = (0.0, 1.0, 0.0, 0.0)

    def __post_init__(self):
        positive(self.discrete_weight, "discrete_weight")
        LegacyLossConfig(self.state_weights)


@dataclass(frozen=True)
class PreparedV53Episode:
    visible: PreparedV53
    encoded_truth: dict[int, Tensor]
    numeric: Tensor
    states: Tensor
    versions: TensorVersions


@dataclass(frozen=True)
class V53Score:
    loss: Tensor
    per_target: Tensor
    output: V53Output


@torch.no_grad()
def prepare_episode(model: V53Model, inputs, request, truth) -> PreparedV53Episode:
    if not bool(inputs.query.any()):
        raise ValueError("no-valid-episode: V5.3 training requires nonempty Query")
    visible = model.prepare(inputs, request)
    # Reuse the established full-observation/state/codebook preflight. Its
    # custom-codec path invokes affine encode_targets, with truth confined here.
    encoded = _preflight(visible.inputs, visible.request, truth, visible.facts, visible.positions)
    for a, schema in enumerate(visible.inputs.schema):
        if (model.config.codec_version != "legacy_v53" and schema.kind == "numeric"
                and len(visible.positions[a])):
            values = visible.inputs.values[a][visible.facts[a].rows]
            if len(values.unique()) < 2:
                raise ValueError("no-valid-episode: supervised numeric column needs two distinct "
                                 "visible values")
    targets = visible.request.targets
    numeric = torch.tensor(
        [s.kind == "numeric" for s in visible.inputs.schema],
        dtype=torch.bool, device=targets.device,
    )[targets[:, 1]]
    states = truth.states[targets[:, 0], targets[:, 1]].detach().clone()
    return PreparedV53Episode(visible, encoded, numeric, states,
                              TensorVersions(encoded, numeric, states))


def score_prepared_episode(
    model: V53Model, prepared: PreparedV53Episode,
    loss_config: V53LossConfig | None = None, *, decode: bool = False,
) -> V53Score:
    config = loss_config or V53LossConfig()
    prepared.versions.validate()
    output = model.forward_prepared(prepared.visible, decode=decode)
    losses = output.carriers.new_zeros(len(output.request.targets), dtype=torch.float64)
    for column in output.columns:
        per_cell = encoding_mse(column.result.encoding, prepared.encoded_truth[column.column])
        if prepared.visible.inputs.schema[column.column].kind == "numeric":
            per_cell = ANSWER_WIDTH * per_cell
        losses = losses.index_copy(0, column.target_indices, per_cell)
    terms = []
    for weight, mask in ((1.0, prepared.numeric), (config.discrete_weight, ~prepared.numeric)):
        if config.state_weights is None:
            terms.append(weight * _masked_mean(losses, mask))
        else:
            for state, state_weight in enumerate(config.state_weights):
                if state_weight:
                    terms.append(weight * state_weight * _masked_mean(
                        losses, mask & (prepared.states == state)
                    ))
    loss = sum(terms, start=losses.new_zeros(()))
    finite(loss, "V5.3 episode loss")
    return V53Score(loss, losses, output)


def score_episode(model: V53Model, inputs, request, truth,
                  loss_config: V53LossConfig | None = None) -> V53Score:
    return score_prepared_episode(
        model, prepare_episode(model, inputs, request, truth), loss_config, decode=True
    )
