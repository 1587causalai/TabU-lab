"""Optional same-column visible reconstruction; never changes the Query mask.

Auxiliary outputs are in-sample LL predictions, not clamped facts or held-out
estimates. The fixed plan is sampled without truth and reused across rounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .codec import V7ProtocolError


@dataclass(frozen=True)
class AuxiliaryColumn:
    column: int
    rows: Tensor
    query_positions: Tensor
    weight: float  # n_query / (n_query + n_aux)


@dataclass(frozen=True)
class AuxiliaryPlan:
    columns: tuple[AuxiliaryColumn, ...]
    seed: int
    scale: float
    per_query: float | None
    min_group_weight: float

    def as_dict(self):
        return dict(
            sampling=(
                "all_visible_per_column"
                if self.per_query is None
                else "uniform_without_replacement_per_column"
            ),
            count_rule=(
                "n_support"
                if self.per_query is None
                else "min(n_support, ceil(auxiliary_per_query * n_query))"
            ),
            reconstruction="in_sample_LL_at_visible_rows; no_writeback",
            seed=self.seed,
            global_scale=self.scale,
            per_query=self.per_query,
            min_group_weight=self.min_group_weight,
            columns=[
                dict(
                    column=c.column,
                    rows=c.rows.detach().cpu().tolist(),
                    n_query=len(c.query_positions),
                    n_aux=len(c.rows),
                    w_bal=c.weight,
                )
                for c in self.columns
            ],
        )


def prepare_auxiliary_reconstruction(episode, config, *, seed: int):
    """Attach a deterministic visible-address plan to a single or joint episode.

    Query-only is an exact no-op. An impossible/extreme ratio is rejected before
    forward, never repaired by dropping Query cells or clipping the weight.
    """
    if config.loss_mode == "query_only":
        if episode.auxiliary is not None:
            raise ValueError("auxiliary plan requires balanced_reconstruction")
        return episode
    if type(seed) is not int:
        raise ValueError("auxiliary seed must be an integer")
    if episode.auxiliary is not None:
        raise ValueError("auxiliary plan already fixed")
    generator = torch.Generator().manual_seed(seed)
    if hasattr(episode, "columns"):
        parts = [(p.column, p.supports, p.positions) for p in episode.columns]
    else:
        parts = [
            (
                episode.target,
                episode.support_rows,
                torch.arange(len(episode.query_rows), device=episode.query_rows.device),
            )
        ]
    columns = []
    for a, supports, positions in parts:
        nq = len(positions)
        na = (
            len(supports)
            if config.auxiliary_per_query is None
            else min(len(supports), math.ceil(config.auxiliary_per_query * nq))
        )
        if not nq or not na:
            raise V7ProtocolError("no-valid-episode", "balanced reconstruction needs both groups")
        w = nq / (nq + na)
        if min(w, 1 - w) < config.auxiliary_min_group_weight:
            raise V7ProtocolError("no-valid-episode", f"column {a}: extreme auxiliary ratio")
        draw = torch.randperm(len(supports), generator=generator)[:na].to(supports.device)
        rows = supports[draw].sort().values
        if not bool(episode.inputs.visible[rows, a].all()):
            raise ValueError("auxiliary addresses must be visible")
        columns.append(AuxiliaryColumn(a, rows, positions, w))
    return replace(
        episode,
        auxiliary=AuxiliaryPlan(
            tuple(columns),
            seed,
            config.balanced_loss_scale,
            config.auxiliary_per_query,
            config.auxiliary_min_group_weight,
        ),
    )


def balanced_round_losses(output, episode, query_reference, config):
    """Document formula: column sum of two weighted sums, no extra averaging."""
    plan = episode.auxiliary
    if plan is None or len(output.auxiliary_states) != len(output.states):
        raise ValueError("balanced reconstruction requires a fixed plan and auxiliary predictions")
    validate_auxiliary_plan(episode, config)
    if (plan.scale, plan.per_query, plan.min_group_weight) != (
        config.balanced_loss_scale,
        config.auxiliary_per_query,
        config.auxiliary_min_group_weight,
    ):
        raise ValueError("auxiliary plan differs from loss config")
    values = []
    for state, aux in zip(output.states, output.auxiliary_states, strict=True):
        total = state.new_zeros(())
        for part in plan.columns:
            a = part.column
            chi = (
                config.chi_numeric
                if episode.codec.columns[a].kind == "numeric"
                else config.chi_discrete
            )
            q = (
                (state[part.query_positions] - query_reference.to(state)[part.query_positions])
                .square()
                .sum()
            )
            prediction = aux[a]
            reference = episode.observed[part.rows, a].to(prediction)
            if prediction.shape != reference.shape:
                raise ValueError("auxiliary prediction shape differs from reference")
            v = (prediction - reference).square().sum()
            total = total + (chi / state.shape[-1]) * ((1 - part.weight) * q + part.weight * v)
        values.append(total * plan.scale)
    return torch.stack(values)


def validate_auxiliary_plan(episode, config):
    """Reject malformed plans rather than silently changing the scoring domain."""
    plan = episode.auxiliary
    if plan is None:
        return
    if config.loss_mode != "balanced_reconstruction":
        raise ValueError("auxiliary plan requires balanced_reconstruction")
    _qr, qc = episode.inputs.query.nonzero(as_tuple=True)
    expected = set(qc.tolist())
    if len(plan.columns) != len(expected) or {p.column for p in plan.columns} != expected:
        raise ValueError("auxiliary plan must cover every queried column exactly once")
    for part in plan.columns:
        rows = part.rows
        positions = (qc == part.column).nonzero(as_tuple=True)[0]
        if not torch.equal(part.query_positions, positions):
            raise ValueError("auxiliary Query addresses differ from episode")
        if (
            rows.ndim != 1
            or rows.dtype != torch.long
            or not len(rows)
            or len(rows.unique()) != len(rows)
            or bool((rows < 0).any())
            or bool((rows >= len(episode.inputs.visible)).any())
        ):
            raise ValueError("invalid auxiliary rows")
        if not bool(episode.inputs.visible[rows, part.column].all()):
            raise ValueError("auxiliary addresses must be visible")
        w = len(positions) / (len(positions) + len(rows))
        if part.weight != w or min(w, 1 - w) < config.auxiliary_min_group_weight:
            raise ValueError("invalid auxiliary balance weight")
