"""Differentiable query-response-only readout for supervised TabUBase episodes.

The ordinary dense terminal emits a same-column routing ledger for every cell.
Supervised pretraining and real frozen ICL score only query response cells, so
materializing the other ledgers is quadratic work with no objective effect.
This module keeps the ordinary symbolizer, tokenizer, label broadcast, and
dynamics path, then evaluates only response-query coordinates against visible
response-context supports.  Truth remains an objective-side input and is never
passed to :func:`query_response_readout`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from tabu_lab.contracts import FeatureKind, FeatureRole, TruthSidecar
from tabu_lab.primitives import RoutingOutput, masked_rbf_weights
from tabu_lab.primitives.routing import _local_linear_values


@dataclass(frozen=True, slots=True)
class QueryResponseReadout:
    """Tensor result for all query cells in the single response column."""

    response_feature: int
    response_kind: FeatureKind
    context_rows: int
    query_rows: int
    numeric_values: Tensor | None
    probabilities: Tensor | None
    log_probabilities: Tensor | None
    class_support_available: Tensor | None
    support_available: Tensor


def _response_feature_index(resolved: Any) -> int:
    response_features = [
        index
        for index, spec in enumerate(resolved.feature_specs)
        if FeatureRole(getattr(spec.role, "value", spec.role)) is FeatureRole.RESPONSE
    ]
    if not response_features:
        response_features = (
            resolved.query_target_mask.any(dim=(0, 1)).nonzero(as_tuple=False).flatten().tolist()
        )
    if len(response_features) != 1:
        raise ValueError("query-response-only readout requires exactly one response column")
    return int(response_features[0])


def _response_kind_and_classes(resolved: Any, response_feature: int) -> tuple[FeatureKind, int]:
    if not resolved.feature_specs:
        return FeatureKind.NUMERIC, 0
    spec = resolved.feature_specs[response_feature]
    kind = FeatureKind(getattr(spec.kind, "value", spec.kind))
    classes = len(spec.domain) if kind is not FeatureKind.NUMERIC else 0
    if kind is not FeatureKind.NUMERIC and classes < 2:
        raise ValueError("categorical response readout requires a declared domain")
    return kind, classes


def query_response_readout(
    model: torch.nn.Module,
    evidence: Any,
    *,
    context_rows: int,
    query_readout_chunk_rows: int = 64,
) -> QueryResponseReadout:
    """Run shared TabUBase dynamics once and read only query response cells.

    The returned tensors retain autograd history.  Chunking applies only to
    query response routing after the complete evidence episode has passed
    through shared dynamics.
    """

    if type(context_rows) is not int or context_rows < 1:
        raise ValueError("query-response-only readout requires positive context rows")
    if type(query_readout_chunk_rows) is not int or query_readout_chunk_rows < 1:
        raise ValueError("query readout chunk rows must be a positive integer")
    encode = getattr(model, "_encode_dense_cells", None)
    readout = getattr(model, "readout", None)
    projection = getattr(readout, "projection", None)
    terminal = getattr(readout, "terminal", None)
    if encode is None or projection is None or terminal is None:
        raise TypeError("query-response-only readout requires a TabUBase cell readout")

    encoded = encode(evidence, emit_trace=False)
    if len(encoded) == 5:
        resolved, _, _, cells, numeric_scale_state = encoded
        terminal_values = numeric_scale_state.standardized_values
    else:
        resolved, _, _, cells = encoded
        terminal_values = resolved.values
    if context_rows >= cells.shape[1]:
        raise ValueError("query-response-only readout requires context and query rows")
    response_feature = _response_feature_index(resolved)
    response_kind, classes = _response_kind_and_classes(resolved, response_feature)
    query_rows = cells.shape[1] - context_rows

    expected_query_targets = torch.zeros_like(resolved.query_target_mask)
    expected_query_targets[:, context_rows:, response_feature] = True
    if not torch.equal(resolved.query_target_mask, expected_query_targets):
        raise ValueError("query targets must be every post-context response cell and nothing else")
    if bool((resolved.target_mask & ~resolved.query_target_mask).any()):
        raise ValueError("query-response-only readout does not accept non-query targets")

    support_mask = resolved.visible_mask[:, :context_rows, response_feature]
    if not bool(support_mask.all()):
        raise RuntimeError("all context response labels must be visible supports")
    if bool(resolved.visible_mask[:, context_rows:, response_feature].any()):
        raise RuntimeError("query response truth entered the model evidence")

    work_cells = cells[:, :, response_feature].to(dtype=torch.float32)
    coordinates = F.linear(
        work_cells,
        projection.weight.to(dtype=torch.float32),
    )
    context_coordinates = coordinates[:, :context_rows]
    query_coordinates = coordinates[:, context_rows:]
    support_values = terminal_values[:, :context_rows, response_feature].to(torch.float32)

    numeric_chunks: list[Tensor] = []
    probability_chunks: list[Tensor] = []
    log_probability_chunks: list[Tensor] = []
    class_support_chunks: list[Tensor] = []
    support_chunks: list[Tensor] = []
    for offset in range(0, query_rows, query_readout_chunk_rows):
        query = query_coordinates[:, offset : offset + query_readout_chunk_rows]
        difference = query.unsqueeze(2) - context_coordinates.unsqueeze(1)
        squared_distance = difference.square().sum(dim=-1)
        allowed = support_mask.unsqueeze(1).expand_as(squared_distance)
        routing = masked_rbf_weights(
            squared_distance,
            allowed,
            bandwidth=terminal.bandwidth.to(dtype=torch.float32),
        )
        support_chunks.append(routing.support_available)

        if response_kind is not FeatureKind.NUMERIC:
            labels = support_values.to(torch.long)
            if bool(((labels < 0) | (labels >= classes)).any()):
                raise RuntimeError("context response label is outside the declared domain")
            membership = F.one_hot(labels, num_classes=classes).to(routing.weights.dtype)
            probability_chunks.append(torch.einsum("bqs,bsc->bqc", routing.weights, membership))
            class_support = routing.support_mask.unsqueeze(-1) & membership.unsqueeze(1).to(
                torch.bool
            )
            finite_floor = -torch.finfo(routing.log_weights.dtype).max
            class_log_mass = torch.logsumexp(
                torch.where(
                    class_support,
                    routing.log_weights.unsqueeze(-1),
                    torch.full_like(routing.log_weights.unsqueeze(-1), finite_floor),
                ),
                dim=2,
            )
            class_available = class_support.any(dim=2)
            log_probability_chunks.append(
                torch.where(
                    class_available,
                    class_log_mass,
                    torch.full_like(class_log_mass, finite_floor),
                )
            )
            class_support_chunks.append(class_available)
            continue

        terminal_kind = getattr(readout, "numeric_terminal", None)
        if terminal_kind == "local_linear":
            expanded_routing = RoutingOutput(
                weights=routing.weights.unsqueeze(2),
                log_weights=routing.log_weights.unsqueeze(2),
                support_mask=routing.support_mask.unsqueeze(2),
                support_available=routing.support_available.unsqueeze(2),
                support_count=routing.support_count.unsqueeze(2),
            )
            expanded_values = support_values.unsqueeze(1).unsqueeze(2).expand(
                -1,
                query.shape[1],
                1,
                -1,
            )
            values = _local_linear_values(
                expanded_routing,
                difference.unsqueeze(2),
                expanded_values,
                ridge=float(terminal.ridge),
            ).squeeze(2)
        elif terminal_kind == "nadaraya_watson":
            values = torch.einsum("bqs,bs->bq", routing.weights, support_values)
        else:  # pragma: no cover - current TabUBase contract fixes local-linear
            raise RuntimeError("unsupported TabUBase numeric terminal")
        numeric_chunks.append(values)

    support_available = torch.cat(support_chunks, dim=1)
    if response_kind is FeatureKind.NUMERIC:
        return QueryResponseReadout(
            response_feature=response_feature,
            response_kind=response_kind,
            context_rows=context_rows,
            query_rows=query_rows,
            numeric_values=torch.cat(numeric_chunks, dim=1),
            probabilities=None,
            log_probabilities=None,
            class_support_available=None,
            support_available=support_available,
        )
    return QueryResponseReadout(
        response_feature=response_feature,
        response_kind=response_kind,
        context_rows=context_rows,
        query_rows=query_rows,
        numeric_values=None,
        probabilities=torch.cat(probability_chunks, dim=1),
        log_probabilities=torch.cat(log_probability_chunks, dim=1),
        class_support_available=torch.cat(class_support_chunks, dim=1),
        support_available=support_available,
    )


def query_response_objective_loss(
    model: torch.nn.Module,
    evidence: Any,
    truth: TruthSidecar,
    *,
    context_rows: int,
    query_readout_chunk_rows: int = 64,
    categorical_epsilon: float = 1.0e-8,
) -> Tensor:
    """Return the dense-objective-equivalent supervised response loss."""

    if not 0.0 < categorical_epsilon < 1.0:
        raise ValueError("categorical epsilon must be in (0, 1)")
    result = query_response_readout(
        model,
        evidence,
        context_rows=context_rows,
        query_readout_chunk_rows=query_readout_chunk_rows,
    )
    if truth.episode_id != getattr(evidence, "episode_id", truth.episode_id):
        raise ValueError("evidence and TruthSidecar episode ids must match")
    truth_values = truth.target_values.to(device=result.support_available.device)
    truth_mask = truth.target_mask.to(device=result.support_available.device)
    if truth_values.ndim == 2:
        truth_values = truth_values.unsqueeze(0)
        truth_mask = truth_mask.unsqueeze(0)
    expected_truth = torch.zeros_like(truth_mask)
    expected_truth[:, context_rows:, result.response_feature] = True
    if not torch.equal(truth_mask, expected_truth):
        raise ValueError("TruthSidecar must contain exactly the query response targets")
    if not bool(result.support_available.all()):
        raise RuntimeError("query response target lacks context support")
    response_truth = truth_values[:, context_rows:, result.response_feature]

    if result.response_kind is FeatureKind.NUMERIC:
        if result.numeric_values is None:
            raise RuntimeError("numeric query response readout returned no values")
        error = result.numeric_values - response_truth.to(result.numeric_values.dtype)
        return error.square().mean()

    if result.log_probabilities is None or result.class_support_available is None:
        raise RuntimeError("categorical query response readout returned no distribution")
    rounded = response_truth.round()
    if not bool(torch.isclose(response_truth, rounded).all()):
        raise ValueError("categorical response truth must use integer domain codes")
    labels = rounded.to(torch.long)
    if bool(((labels < 0) | (labels >= result.log_probabilities.shape[-1])).any()):
        raise ValueError("categorical response truth is outside the declared domain")
    selected_log_probability = result.log_probabilities.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    selected_support = result.class_support_available.gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    epsilon_nll = -math.log(categorical_epsilon)
    selected_nll = torch.where(
        selected_support,
        (-selected_log_probability).clamp_min(0.0),
        selected_log_probability.new_full(selected_log_probability.shape, epsilon_nll),
    )
    return selected_nll.mean()


__all__ = [
    "QueryResponseReadout",
    "query_response_objective_loss",
    "query_response_readout",
]
