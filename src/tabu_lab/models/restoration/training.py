"""Scorer-only truth preflight, typed/state-aware loss reduction, batch training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._dtype import solve_dtype
from ._packing import column_positions
from ._prepared import TensorVersions
from ._validation import finite
from .answers import NumericAnswers
from .contracts import RestorationInput, RestorationRequest, TruthSidecar, validate_values
from .losses import encoding_mse
from .model import PreparedRestoration, RestorationModel, RestorationOutput, prepare_readout


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
    by_state: dict[str, dict[str, float | int]] | None
    output: RestorationOutput


def _preflight(inputs, request, truth, facts, positions_by_column=None):
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
    numeric_groups = {}
    if positions_by_column is None:
        positions_by_column = column_positions(targets[:, 1], len(inputs.schema))
    for a, schema in enumerate(inputs.schema):
        values = truth.values[a]
        if values.shape != (len(inputs.visible),) or values.device != targets.device:
            raise ValueError("truth columns must align with input rows and device")
        positions = positions_by_column[a]
        if not len(positions):
            continue
        if len(facts[a].rows) < 2:
            raise ValueError("invalid-episode: every supervised column needs two visible supports")
        selected = values[targets[positions, 0]]
        if schema.kind == "numeric" and type(facts[a].answers) is NumericAnswers:
            if not selected.is_floating_point():
                raise ValueError("numeric values must be finite floating values")
            numeric_groups.setdefault(len(positions), []).append((a, selected))
            continue
        validate_values(schema, selected)
        # Missing clean class invalidates the WHOLE episode, before neural forward.
        encoded[a] = facts[a].answers.encode_targets(selected)
    for group in numeric_groups.values():
        values = torch.stack([values for _, values in group])
        if not bool(torch.isfinite(values).all()):
            raise ValueError("numeric values must be finite floating values")
        codes = NumericAnswers.encode_targets_batch([facts[a].answers for a, _ in group], values)
        encoded.update((a, codes[i]) for i, (a, _) in enumerate(group))
    return encoded


@dataclass(frozen=True)
class LossLayout:
    groups: tuple  # (column indices, concatenated scorer-only truth encodings)
    indices: Tensor
    types: Tensor
    state_masks: Tensor
    counts: Tensor


@dataclass(frozen=True)
class PreparedEpisode:
    """Scorer-owned plan; only visible reaches model.forward_prepared."""

    visible: PreparedRestoration
    scoring: LossLayout
    versions: TensorVersions


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean after scaling each term, so a finite mean cannot overflow in sum."""
    if not bool(mask.any()):
        return values.new_zeros(())
    count = mask.sum()
    return (values[mask] / count).sum()


def _prepare_loss(inputs, request, truth, facts, layout):
    encoded_truth = _preflight(inputs, request, truth, facts, layout.positions)
    targets = request.targets
    numeric = torch.tensor(
        [schema.kind == "numeric" for schema in inputs.schema],
        dtype=torch.bool,
        device=targets.device,
    )[targets[:, 1]]
    groups = {}
    for a, _ in layout.active:
        groups.setdefault(facts[a].answers.encoded.shape[1], []).append(a)
    indices, fixed = [], []
    for columns in groups.values():
        indices.extend(layout.positions[a] for a in columns)
        fixed.append((tuple(columns), torch.cat([encoded_truth[a] for a in columns])))
    states = truth.states[targets[:, 0], targets[:, 1]]
    state_masks = states[None] == torch.arange(4, device=states.device)[:, None]
    return LossLayout(
        tuple(fixed),
        torch.cat(indices),
        torch.stack((numeric, ~numeric)),
        state_masks,
        state_masks.sum(-1),
    )


@torch.no_grad()
def prepare_episode(model, inputs, request, truth) -> PreparedEpisode:
    """Validate and snapshot fixed data once; truth encoding stays in the scorer."""
    visible = model.prepare(inputs, request)
    scoring = _prepare_loss(visible.inputs, visible.request, truth, visible.facts, visible.layout)
    return PreparedEpisode(visible, scoring, TensorVersions(scoring))


def _score_output(output, layout, loss_config, report):
    targets = output.request.targets
    columns = {col.column: col for col in output.columns}
    losses = [
        encoding_mse(torch.cat([columns[a].result.encoding for a in group]), encoded_truth)
        for group, encoded_truth in layout.groups
    ]
    per_target = output.carriers.new_zeros(len(targets), dtype=solve_dtype(output.carriers))
    per_target = per_target.index_copy(0, layout.indices, torch.cat(losses))
    types, state_masks = layout.types, layout.state_masks
    if loss_config.state_weights is None:
        loss = sum(
            (_masked_mean(per_target, type_mask) for type_mask in types),
            start=per_target.new_zeros(()),
        )
    else:
        terms = []
        for type_mask in types:
            for state, weight in enumerate(loss_config.state_weights):
                if weight:
                    terms.append(
                        per_target.new_tensor(weight)
                        * _masked_mean(per_target, type_mask & state_masks[state])
                    )
        loss = sum(terms, start=per_target.new_zeros(()))
    finite(loss, "episode loss")
    by_state = None
    if report:
        counts = layout.counts
        by_state = {}
        for state, name in enumerate(("retained", "query", "null", "corrupted")):
            count = counts[state]
            by_state[name] = {"count": int(count)}
            if bool(count):
                mean = _masked_mean(per_target.detach(), state_masks[state])
                finite(mean, f"{name} encoding MSE")
                by_state[name]["encoding_mse"] = float(mean)
    return EpisodeScore(loss, per_target, by_state, output)


def score_prepared_episode(
    model: RestorationModel,
    prepared: PreparedEpisode,
    loss_config: LossConfig | None = None,
    *,
    decode: bool = False,
    report: bool = False,
) -> EpisodeScore:
    """Replay a fixed episode. Decoding and Python reports are explicit opt-ins.

    Invalid episodes still fail during preparation, before any learned forward.
    Learned finite/Cholesky checks remain active on every replay. A skipped
    decoder does not certify finiteness in original units; evaluation still does.
    """
    prepared.versions.validate()
    output = model.forward_prepared(prepared.visible, decode=decode)
    return _score_output(output, prepared.scoring, loss_config or LossConfig(), report)


def score_episode(
    model: RestorationModel,
    inputs: RestorationInput,
    request: RestorationRequest,
    truth: TruthSidecar,
    loss_config: LossConfig | None = None,
) -> EpisodeScore:
    request.validate(inputs)
    facts = model.encoder.prepare(inputs)
    layout = prepare_readout(request, facts)
    scoring = _prepare_loss(inputs, request, truth, facts, layout)
    output = model._forward_prepared(inputs, request, facts, layout=layout)
    return _score_output(output, scoring, loss_config or LossConfig(), True)


def batch_loss(model: RestorationModel, episodes, loss_config: LossConfig | None = None):
    """Equal episode weight; no silent skipping/retry of invalid episodes."""
    scores = [score_episode(model, *episode, loss_config=loss_config) for episode in episodes]
    if not scores:
        raise ValueError("batch must contain at least one episode")
    return torch.stack([score.loss for score in scores]).mean(), scores
