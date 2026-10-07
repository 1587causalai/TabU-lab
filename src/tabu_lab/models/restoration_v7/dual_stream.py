"""Row-context dual-stream value embedding: an explicit experimental V7 branch.

Selected only by ``V7Config(value_encoder="row_dual_stream", dual_stream=...)``.
It replaces the per-Cell ``phi`` and ``W_up`` of a recovery round:

* encode: ``u0 = v0 = C`` per row, then per block ``u = u + A_l(v)``,
  ``v = v + F_l(u)``; ``Z = [u4; v4]`` (128) enters token dynamics directly.
  ``A_l`` is OAttention among the Cells of ONE sample, ``F_l`` a token-wise OFFN.
* LL: each queried column reads its true visible supports' 128-wide ``Z``
  responses from the SAME round snapshot (no detach, no cross-round cache).
* readback (``mean``, a named candidate, not a validated rule): all Query
  addresses receive their column LL predictions simultaneously, every other
  Cell keeps this round's ``Z``; the whole row is inverted from block 4 to 1
  (``v -= F_l(u)``, then ``u -= A_l(v)``) and the two 64-wide branches are
  averaged. Only Query codes are written back; observed codes stay fixed.

``G`` (the coupling on the full 128-wide row state) is the bijection;
``E(C) = G(C, C)`` is an embedding of the 64-wide code, not a bijection onto
the 128-wide space. Static role masks are identical in forward and inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..restoration._validation import finite
from ..restoration.backbone import OFFN, OAttention
from .config import DualStreamConfig
from .readout import column_shared_ll, evaluate_fitted_ll


def _initialize_operator(module: nn.Module) -> None:
    """Same Xavier initialization as V7 OMAB projections; presence stays identity."""
    for name in ("q", "k", "v", "out"):
        if hasattr(module, name):
            nn.init.xavier_uniform_(getattr(module, name).weight)
    if hasattr(module, "ff"):
        nn.init.xavier_uniform_(module.ff[0].weight)
        nn.init.xavier_uniform_(module.ff[2].weight)


class RowDualStreamEncoder(nn.Module):
    """``G`` on ``[N, M, 2p]`` row states; independent parameters per block."""

    def __init__(self, config: DualStreamConfig, code_dim: int, *, strict_finite_content: bool):
        super().__init__()
        self.config = config
        self.code_dim = code_dim
        operator = config.operator_config(code_dim)
        self.attention = nn.ModuleList()
        self.ffn = nn.ModuleList()
        for _ in range(config.blocks):
            attention = OAttention(operator, strict_finite_content=strict_finite_content)
            ffn = OFFN(operator)
            _initialize_operator(attention)
            _initialize_operator(ffn)
            self.attention.append(attention)
            self.ffn.append(ffn)

    @staticmethod
    def _check_masks(state: Tensor, active: Tensor, sources: Tensor) -> None:
        if state.ndim != 3 or active.shape != state.shape[:2] or sources.shape != active.shape:
            raise ValueError("dual-stream state must be [N,M,d] with [N,M] masks")
        if active.dtype != torch.bool or sources.dtype != torch.bool:
            raise ValueError("dual-stream masks must be boolean")
        if bool((sources & ~active).any()):
            raise ValueError("dual-stream sources must be active Cells")

    def attention_update(self, block: int, v: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        """``A_l(v)``: each row is one batch element; Cells of one sample only."""
        update, _ = self.attention[block].attention_update(v, v, sources)
        # Static receiver mask: a Null Cell never changes in either direction.
        return torch.where(active[..., None], update, torch.zeros_like(update))

    def ffn_update(self, block: int, u: Tensor, active: Tensor) -> Tensor:
        """``F_l(u)``: token-wise; no Cell mixing."""
        update = self.ffn[block].local_update(u)
        return torch.where(active[..., None], update, torch.zeros_like(update))

    def couple(self, u: Tensor, v: Tensor, active: Tensor, sources: Tensor):
        self._check_masks(u, active, sources)
        for block in range(self.config.blocks):
            u = u + self.attention_update(block, v, active, sources)
            v = v + self.ffn_update(block, u, active)
        return u, v

    def uncouple(self, u: Tensor, v: Tensor, active: Tensor, sources: Tensor):
        self._check_masks(u, active, sources)
        for block in reversed(range(self.config.blocks)):
            v = v - self.ffn_update(block, u, active)
            u = u - self.attention_update(block, v, active, sources)
        return u, v

    def forward_state(self, state: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        """``G`` on an arbitrary 128-wide row state (the two branches may differ)."""
        u, v = state.split(self.code_dim, dim=-1)
        return torch.cat(self.couple(u, v, active, sources), dim=-1)

    def inverse(self, state: Tensor, active: Tensor, sources: Tensor) -> tuple[Tensor, Tensor]:
        """``G^{-1}``; returns the two 64-wide branches ``(u0, v0)``."""
        if state.shape[-1] != 2 * self.code_dim:
            raise ValueError("dual-stream state width must be 2 * code_dim")
        u, v = state.split(self.code_dim, dim=-1)
        return self.uncouple(u, v, active, sources)

    def forward(self, codes: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        """``E(C) = G(C, C)``; Null Cells must enter as exact zero codes."""
        if codes.shape[-1] != self.code_dim:
            raise ValueError("dual-stream codes must have width code_dim")
        self._check_masks(codes, active, sources)
        if bool(codes[~active].ne(0).any()):
            raise ValueError("dual-stream Null Cells must enter as exact zero codes")
        state = torch.cat(self.couple(codes, codes, active, sources), dim=-1)
        finite(state, "dual-stream encoding")
        return state

    def readback(self, state: Tensor, active: Tensor, sources: Tensor) -> Tensor:
        """Explicit ``mean`` readback ``(u0 + v0) / 2`` of ``G^{-1}(state)``."""
        u, v = self.inverse(state, active, sources)
        codes = (u + v) / 2
        finite(codes, "dual-stream readback")
        return codes


@dataclass(frozen=True)
class DualStreamPart:
    column: int
    supports: Tensor
    rows: Tensor
    positions: Tensor  # addresses in the flattened Query state
    auxiliary_rows: Tensor | None = None


@dataclass(frozen=True)
class DualStreamRound:
    state: Tensor  # [|Q|, p] written Query codes
    encoded: Tensor  # Z, [N, M, 2p]
    predicted: Tensor  # [|Q|, 2p] simultaneous column LL predictions
    assembled: Tensor  # Z with every Query address replaced by its prediction
    slopes: dict[int, Tensor]
    auxiliary: dict[int, Tensor]  # visible reconstructions [n_aux, p] per column


def dual_stream_round(
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
    sources = visible | query  # query_source=True is enforced by the config
    table = observed.index_put((query_rows, query_cols), current)
    encoded = block.encoder(table, active, sources)
    units = block.unit_seed.expand(n, -1)[:, None]
    features = torch.cat((block.feature_seed.expand(m, -1), encoded.new_zeros(1, encoded.shape[2])))
    h = torch.cat((torch.cat((encoded, units), dim=1), features[None]), dim=0)
    finite(h, "dual-stream compiled carriers")
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
        finite(recovery.recovered, "dual-stream LL prediction")
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
    state = block.encoder.readback(assembled, active, sources)[query_rows, query_cols]
    finite(state, "dual-stream Query writeback")
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
        decoded = block.encoder.readback(auxiliary_assembled, active, sources)
        for part in parts:
            if part.column in auxiliary_predictions:
                codes = decoded[part.auxiliary_rows, part.column]
                finite(codes, "auxiliary reconstructed codes")
                auxiliary[part.column] = codes
    return DualStreamRound(state, encoded, predicted, assembled, slopes, auxiliary)


def episode_parts(episode):
    """Single-column episodes are the one-part case of the joint readback."""
    from .joint import JointEpisode

    plan = episode.auxiliary
    auxiliary = {} if plan is None else {p.column: p.rows for p in plan.columns}
    if isinstance(episode, JointEpisode):
        parts = tuple(
            DualStreamPart(p.column, p.supports, p.rows, p.positions, auxiliary.get(p.column))
            for p in episode.columns
        )
        return parts, episode.query_rows, episode.query_cols, episode.donor_rows
    positions = torch.arange(len(episode.query_rows), device=episode.query_rows.device)
    part = DualStreamPart(
        episode.target,
        episode.support_rows,
        episode.query_rows,
        positions,
        auxiliary.get(episode.target),
    )
    query_cols = torch.full_like(episode.query_rows, episode.target)
    return (part,), episode.query_rows, query_cols, episode.donor_rows


def forward_dual_stream(model, episode, *, decode: bool = True):
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
        result = dual_stream_round(
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


__all__ = [
    "DualStreamPart",
    "DualStreamRound",
    "RowDualStreamEncoder",
    "dual_stream_round",
    "episode_parts",
    "forward_dual_stream",
]
