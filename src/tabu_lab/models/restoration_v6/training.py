"""Scorer-only supervised losses for broadcast Query rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..restoration._dtype import solve_dtype
from ..restoration._validation import finite
from ..restoration.contracts import RestorationRequest
from ..restoration_v53.training import (
    V53LossConfig,
    prepare_training_episode,
    score_prepared_episode,
)
from .model import V6Model, task_target_column


@dataclass(frozen=True)
class V6Score:
    loss: torch.Tensor
    per_cell: torch.Tensor
    output: object
    query_rows: int
    scored_cells: int


def score_training_episode(
    model: V6Model, inputs, truth, *, request: RestorationRequest | None = None,
    loss_config: V53LossConfig | None = None,
) -> V6Score:
    """Default to V5.5 Query-target square loss; optionally score all columns.

    ``request`` can carry the original training episode's full observation
    addresses. The target-only path retains its Query cells and V5.5 scorer;
    when omitted, the full observed request is reconstructed from scorer-only
    truth states for the V5.5 preflight. Neither path
    passes Query truth to ``model.forward_prepared``.
    """
    target = task_target_column(inputs)
    query_rows = inputs.query[:, target]
    if model.supervision == "target_only":
        config = loss_config or V53LossConfig()
        if tuple(config.state_weights or ()) != (0.0, 1.0, 0.0, 0.0):
            raise ValueError("target_only requires the original Query-only loss weights")
        target_request = request if request is not None else RestorationRequest(
            (truth.states >= 0).nonzero()
        )
        prepared = prepare_training_episode(model, inputs, target_request, truth, config)
        selected = prepared.visible.request.targets
        if (len(selected) != int(query_rows.sum())
                or not bool((selected[:, 1] == target).all())
                or not bool(query_rows[selected[:, 0]].all())):
            raise ValueError("target_only must score every Query target cell exactly once")
        score = score_prepared_episode(model, prepared, loss_config=config)
        return V6Score(score.loss, score.per_target, score.output,
                       int(query_rows.sum()), len(selected))

    if model.supervision != "joint_all":
        raise ValueError("unknown V6 supervision mode")
    if loss_config is not None or request is not None:
        raise ValueError("joint_all uses its own Query-row request and loss")
    valid = query_rows[:, None] & (truth.states >= 0)
    request = RestorationRequest(valid.nonzero())
    if not len(request.targets):
        raise ValueError("no-valid-episode: V6 needs supervised Query-row cells")
    prepared = model.prepare(inputs, request)
    support = inputs.visible[:, target]
    if int(support.sum()) < 2:
        raise ValueError("no-valid-episode: target needs two visible support labels")
    if (inputs.schema[target].kind == "numeric"
            and len(inputs.values[target][support].unique()) < 2):
        raise ValueError("no-valid-episode: numeric target needs diverse supports")
    for fact in prepared.facts:
        if int(inputs.visible[fact.rows, target].sum()) < 2:
            raise ValueError("no-valid-episode: each scored column needs two complete supports")

    output = model.forward_prepared(prepared, decode=False)
    rows = request.targets[:, 0]
    column_count = valid.sum(1)
    dtype = solve_dtype(output.carriers)
    per_cell = output.carriers.new_zeros(len(rows), dtype=dtype)
    target_truth = output.facts[target].answers.encode_targets(
        truth.values[target][rows]
    )
    for column in output.columns:
        positions = column.target_indices
        a = column.column
        values = truth.values[a][rows[positions]]
        own_truth = output.facts[a].answers.encode_targets(values)
        combined_truth = own_truth + target_truth[positions]
        residual = column.result.encoding - combined_truth
        squared = residual.square().sum(-1)
        per_cell = per_cell.index_copy(0, positions, squared)
    chi = 128.0 if inputs.schema[target].kind == "numeric" else 1.0
    per_cell = (chi / 128.0) * per_cell
    per_row = per_cell.new_zeros(len(inputs.visible)).index_add(0, rows, per_cell)
    loss = (per_row[query_rows] / column_count[query_rows]).mean()
    finite(loss, "V6 all-column Query-row squared loss")
    return V6Score(loss, per_cell, output, int(query_rows.sum()), len(request.targets))
