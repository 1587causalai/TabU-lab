"""Opt-in phi -> closed row 64->64 -> existing lift; 64-wide LL response.

Whole-row inverse precedes per-Cell phi inverse. Null increments are masked;
observations are clamped in code coordinates by composing each recovery round.
No duplicate-code embedding, mean readback, or new loss is introduced.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
from ..restoration._validation import finite
from .dual_stream import RowDualStreamEncoder, DualStreamPart, DualStreamRound, episode_parts
from .readout import column_shared_ll, evaluate_fitted_ll

class RowReversible64Encoder(RowDualStreamEncoder):
    """G on [N,M,64], split rather than duplicate; deterministic additive inverse."""
    def __init__(self, config, *, strict_finite_content):
        super().__init__(config, 32, strict_finite_content=strict_finite_content)
        for attention, ffn in zip(self.attention, self.ffn):
            nn.init.zeros_(attention.out.weight)
            nn.init.zeros_(ffn.ff[2].weight)

    def forward(self, state: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        if state.shape[-1] != 64:
            raise ValueError("row_reversible64 requires width 64")
        result = self.forward_state(state, active, sources)
        finite(result, "row_reversible64 encoding")
        return result

    def inverse(self, state: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        result = torch.cat(super().inverse(state, active, sources), dim=-1)
        finite(result, "row_reversible64 inverse")
        return result

    def readback(self, state, active, sources):
        return self.inverse(state, active, sources)

def reversible64_round(
    model,
    block,
    observed: Tensor,
    current: Tensor,
    query_rows: Tensor,
    query_cols: Tensor,
    parts: tuple[DualStreamPart, ...],
    visible: Tensor,
    query: Tensor,
) -> DualStreamRound:
    """One recovery round; every column reads the same ``Z`` and ``h`` snapshot."""
    config = model.config
    n, m = visible.shape
    dtype = current.dtype
    active = visible | query
    sources = active if config.query_source else visible
    table = observed.index_put((query_rows, query_cols), current)
    rows, cols = active.nonzero(as_tuple=True)
    embedded = torch.zeros_like(table).index_put((rows, cols), block.phi(table[rows, cols]))
    encoded = block.row_encoder(embedded, active, sources)
    cells = encoded.new_zeros(n, m, block.lift.out_features)
    cells = cells.index_put((rows, cols), block.lift(encoded[rows, cols]))
    units = block.unit_seed.expand(n, -1)[:, None]
    features = torch.cat((block.feature_seed.expand(m, -1), cells.new_zeros(1, cells.shape[2])))
    h = torch.cat((torch.cat((cells, units), dim=1), features[None]), dim=0)
    finite(h, "row-reversible64 compiled carriers")
    h = model.run_backbone(block, h, visible, query)  # exactly once for all columns
    predicted = encoded.new_zeros(len(query_rows), encoded.shape[2])
    slopes, auxiliary_predictions = {}, {}
    for part in parts:
        a = part.column
        responses = encoded[part.supports, a]  # this round's snapshot, gradient kept
        recovery = column_shared_ll(
            h[:n, m],
            part.supports,
            responses,
            h[:n, a],
            part.rows,
            ridge=config.ridge,
            bandwidth=config.bandwidth,
            center_chunk_size=config.center_chunk_size,
        )
        finite(recovery.recovered, "row-reversible64 LL prediction")
        predicted = predicted.index_put((part.positions,), recovery.recovered.to(dtype))
        slopes[a] = recovery.slope
        if part.auxiliary_rows is not None:
            auxiliary_predictions[a] = evaluate_fitted_ll(
                h[:n, m],
                part.supports,
                responses,
                h[:n, a],
                part.auxiliary_rows,
                slope=recovery.slope,
                bandwidth=config.bandwidth,
            ).to(dtype)
    # Simultaneous assembly: no column's write can affect another column's
    # readout in this round; non-Query Cells keep this round's encoding.
    assembled = encoded.index_put((query_rows, query_cols), predicted)
    recovered = block.row_encoder.inverse(assembled, active, sources)
    state = torch.zeros_like(current)
    for part in parts:
        codes = block.phi.inverse(recovered[part.rows, part.column])
        state = state.index_put((part.positions,), codes)
    finite(state, "row-reversible64 Query writeback")
    auxiliary = {}
    if auxiliary_predictions:
        # Independent assembly: from the same Z, replace every auxiliary visible
        # address by its LL prediction (Query addresses keep Z), invert the
        # whole row and extract the auxiliary addresses. It never feeds the
        # Query writeback, and the observed codes stay clamped.
        planned = [p for p in parts if p.column in auxiliary_predictions]
        rows = torch.cat([p.auxiliary_rows for p in planned])
        cols = torch.cat([torch.full_like(p.auxiliary_rows, p.column) for p in planned])
        values = torch.cat([auxiliary_predictions[p.column] for p in planned])
        auxiliary_assembled = encoded.index_put((rows, cols), values)
        decoded = block.row_encoder.inverse(auxiliary_assembled, active, sources)
        for part in parts:
            if part.column in auxiliary_predictions:
                codes = block.phi.inverse(decoded[part.auxiliary_rows, part.column])
                finite(codes, "auxiliary reconstructed codes")
                auxiliary[part.column] = codes
    return DualStreamRound(state, encoded, predicted, assembled, slopes, auxiliary)


def forward_reversible64(model, episode, *, decode: bool = True):
    from .auxiliary import validate_auxiliary_plan
    from .joint import JointEpisode, JointOutput
    from .model import V7Output

    config = model.config
    if episode.codec.dim != config.code_dim or episode.codec.family != config.codec:
        raise ValueError("codec differs from model")
    if episode.codec.numeric_preprocessing != config.numeric_preprocessing:
        raise ValueError("episode numeric preprocessing differs from model config")
    validate_auxiliary_plan(episode, config)
    if episode.auxiliary is not None and config.loss_mode != "balanced_reconstruction":
        raise ValueError("auxiliary plan requires balanced_reconstruction")
    joint = isinstance(episode, JointEpisode)
    parts, query_rows, query_cols, donor_rows = episode_parts(episode)
    visible, query = episode.inputs.visible, episode.inputs.query
    dtype = model.rounds[0].unit_seed.dtype
    observed = episode.observed.to(dtype)
    if config.query_init == "donor":
        current = observed[donor_rows, query_cols]
    else:
        current = torch.zeros_like(observed[query_rows, query_cols])
        for part in parts:
            seed = model.query_seed(episode.inputs.schema[part.column].kind).to(observed)
            current = current.index_put((part.positions,), seed.expand(len(part.rows), -1))
    initial = current
    states, slopes, auxiliary_states = [], [], []
    for t in range(config.rounds):
        block = model.round_module(t)
        result = reversible64_round(
            model, block, observed, current, query_rows, query_cols, parts, visible, query
        )
        current = result.state  # simultaneous Query-only writeback, no detach
        states.append(current)
        if joint:
            slopes.append(result.slopes)
        else:
            slopes.append(result.slopes[parts[0].column])
        if episode.auxiliary is not None:
            auxiliary_states.append(result.auxiliary)
    if joint:
        decoded = (
            {
                p.column: episode.codec.columns[p.column].decode(current[p.positions].detach())
                for p in parts
            }
            if decode
            else None
        )
        return JointOutput(initial, tuple(states), tuple(slopes), decoded, tuple(auxiliary_states))
    decoded = episode.codec.columns[parts[0].column].decode(current.detach()) if decode else None
    return V7Output(initial, tuple(states), tuple(slopes), decoded, tuple(auxiliary_states))
