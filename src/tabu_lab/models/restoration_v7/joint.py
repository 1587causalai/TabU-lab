"""Joint cyclic recovery, promoted from the audited 2026-10-03 experiment.

Query cells may occupy different row sets in different columns. One backbone
snapshot per round feeds all column readouts, followed by simultaneous writes.
Truth is consumed only by the scorer. Parameter names are those of V7Model.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..restoration._validation import finite
from ..restoration.contracts import RestorationInput, TruthSidecar
from .auxiliary import AuxiliaryPlan, balanced_round_losses, validate_auxiliary_plan
from .codec import G64Codec, V7ProtocolError, build_value_codec, null_constant_inputs
from .config import round_weights
from .readout import column_shared_ll, evaluate_fitted_ll
from .training import V7Score
from .training import state_loss as single_state_loss


@dataclass(frozen=True)
class ColumnQuery:
    column: int
    supports: Tensor
    rows: Tensor
    positions: Tensor  # addresses in the row-major flattened Query state
    donors: Tensor


@dataclass(frozen=True)
class JointEpisode:
    inputs: RestorationInput
    codec: G64Codec
    columns: tuple[ColumnQuery, ...]
    query_rows: Tensor
    query_cols: Tensor
    donor_rows: Tensor
    observed: Tensor
    cell_weights: Tensor  # 1 / (number of Query rows * Query cells in this row)
    auxiliary: AuxiliaryPlan | None = None


@dataclass(frozen=True)
class JointOutput:
    initial: Tensor
    states: tuple[Tensor, ...]
    slopes: tuple[dict[int, Tensor], ...]
    decoded: dict[int, Tensor] | None = None
    auxiliary_states: tuple[dict[int, Tensor], ...] = ()


def prepare_joint_episode(inputs, *, donor_seed, config, admission="training"):
    if admission not in ("training", "inference"):
        raise ValueError("invalid admission")
    if type(donor_seed) is not int:
        raise ValueError("donor seed must be an integer")
    inputs = null_constant_inputs(inputs, config.numeric_preprocessing)
    qr, qc = inputs.query.nonzero(as_tuple=True)
    if not len(qr):
        raise ValueError("no Query cells")
    cols = inputs.query.any(0).nonzero(as_tuple=True)[0].tolist()
    generator = torch.Generator().manual_seed(donor_seed)
    parts = []
    donors = torch.zeros_like(qr)
    for a in cols:
        supports = inputs.visible[:, a].nonzero(as_tuple=True)[0]
        positions = (qc == a).nonzero(as_tuple=True)[0]
        rows = qr[positions]
        if not len(supports):
            raise V7ProtocolError("no-support", f"column {a}")
        if admission == "training":
            if len(supports) < 2:
                raise V7ProtocolError("no-valid-episode", f"column {a}: fewer than two supports")
            if inputs.schema[a].kind == "numeric" and len(inputs.values[a][supports].unique()) < 2:
                raise V7ProtocolError("no-valid-episode", f"column {a}: no support diversity")
        draw = torch.randint(len(supports), (len(rows),), generator=generator)
        chosen = supports[draw.to(supports.device)]
        donors = donors.index_put((positions,), chosen)
        parts.append(ColumnQuery(a, supports, rows, positions, chosen))
    # Every Query was hidden in RestorationInput before any codec is built.
    codec = build_value_codec(
        inputs,
        codec=config.codec,
        dim=config.code_dim,
        epsilon=config.epsilon,
        numeric_preprocessing=config.numeric_preprocessing,
    )
    observed = codec.encode_observed(inputs)
    counts = inputs.query.sum(1)
    weights = 1 / (int((counts > 0).sum()) * counts[qr].to(observed.dtype))
    return JointEpisode(inputs, codec, tuple(parts), qr, qc, donors, observed, weights)


def forward_joint(model, episode, *, decode=True):
    config = model.config
    validate_auxiliary_plan(episode, config)
    if episode.codec.dim != config.code_dim or episode.codec.family != config.codec:
        raise ValueError("codec differs from model")
    if episode.codec.numeric_preprocessing != config.numeric_preprocessing:
        raise ValueError("episode numeric preprocessing differs from model config")
    visible, query = episode.inputs.visible, episode.inputs.query
    n, m = visible.shape
    dtype = model.rounds[0].unit_seed.dtype
    observed = episode.observed.to(dtype)
    active_rows, active_cols = (visible | query).nonzero(as_tuple=True)
    if config.query_init == "donor":
        current = observed[episode.donor_rows, episode.query_cols]
    else:
        current = torch.zeros_like(observed[episode.query_rows, episode.query_cols])
        for part in episode.columns:
            seed = model.query_seed(episode.inputs.schema[part.column].kind).to(observed)
            current = current.index_put((part.positions,), seed.expand(len(part.rows), -1))
    initial = current
    states, slopes = [], []
    auxiliary_states = []
    auxiliary_columns = (
        {} if episode.auxiliary is None else {p.column: p for p in episode.auxiliary.columns}
    )
    if auxiliary_columns and config.loss_mode != "balanced_reconstruction":
        raise ValueError("auxiliary plan requires balanced_reconstruction")
    for t in range(config.rounds):
        block = model.round_module(t)
        table = observed.index_put((episode.query_rows, episode.query_cols), current)
        embedded = block.phi(table[active_rows, active_cols])
        embeddings = torch.zeros_like(table).index_put((active_rows, active_cols), embedded)
        cells = torch.zeros(n, m, block.lift.out_features, dtype=dtype, device=table.device)
        cells = cells.index_put((active_rows, active_cols), block.lift(embedded))
        units = block.unit_seed.expand(n, -1)[:, None]
        features = torch.cat((block.feature_seed.expand(m, -1), cells.new_zeros(1, cells.shape[2])))
        h = torch.cat((torch.cat((cells, units), dim=1), features[None]), dim=0)
        finite(h, "joint V7 compiled carriers")
        h = model.run_backbone(block, h, visible, query)  # exactly once for all columns
        next_state = torch.zeros_like(current)
        round_slopes = {}
        round_auxiliary = {}
        # All columns read the same h and embeddings snapshot. No per-column
        # write can affect another column's readout within this round.
        for part in episode.columns:
            a = part.column
            recovery = column_shared_ll(
                h[:n, m],
                part.supports,
                embeddings[part.supports, a],
                h[:n, a],
                part.rows,
                ridge=config.ridge,
                bandwidth=config.bandwidth,
                center_chunk_size=config.center_chunk_size,
            )
            predicted = block.phi.inverse(recovery.recovered.to(dtype))
            finite(predicted, "joint V7 Query writeback")
            next_state = next_state.index_put((part.positions,), predicted)
            round_slopes[a] = recovery.slope
            if a in auxiliary_columns:
                aux = evaluate_fitted_ll(
                    h[:n, m],
                    part.supports,
                    embeddings[part.supports, a],
                    h[:n, a],
                    auxiliary_columns[a].rows,
                    slope=recovery.slope,
                    bandwidth=config.bandwidth,
                )
                aux_codes = block.phi.inverse(aux.to(dtype))
                finite(aux_codes, "auxiliary reconstructed codes")
                round_auxiliary[a] = aux_codes
        current = next_state  # simultaneous writeback, no detach
        states.append(current)
        slopes.append(round_slopes)
        if episode.auxiliary is not None:
            auxiliary_states.append(round_auxiliary)
    decoded = (
        {
            part.column: episode.codec.columns[part.column].decode(current[part.positions].detach())
            for part in episode.columns
        }
        if decode
        else None
    )
    return JointOutput(initial, tuple(states), tuple(slopes), decoded, tuple(auxiliary_states))


def reference_codes(episode, truth: TruthSidecar, config):
    ref = torch.zeros_like(episode.observed[episode.query_rows, episode.query_cols])
    chi = ref.new_zeros(len(ref))
    for part in episode.columns:
        a = part.column
        column = episode.codec.columns[a]
        codes = column.encode(truth.values[a][part.rows]).to(ref)
        ref = ref.index_put((part.positions,), codes)
        chi[part.positions] = (
            config.chi_numeric if column.kind == "numeric" else config.chi_discrete
        )
    return ref, chi


def joint_state_loss(state, reference, chi, cell_weights):
    errors = (state - reference.to(state)).square().sum(-1) * chi.to(state) / state.shape[-1]
    return (errors * cell_weights.to(state)).sum()


def score_joint(output, episode, truth, config):
    reference, chi = reference_codes(episode, truth, config)

    def score(state):
        if len(episode.columns) == 1:
            # Preserve the frozen one-column reduction order, including FP32
            # rounding, in this degenerate interface case.
            a = episode.columns[0].column
            coefficient = (
                config.chi_numeric
                if episode.inputs.schema[a].kind == "numeric"
                else config.chi_discrete
            )
            return single_state_loss(state, reference.to(state), coefficient)
        return joint_state_loss(state, reference, chi, episode.cell_weights)

    losses = torch.stack([score(s) for s in output.states])
    if config.loss_mode == "balanced_reconstruction":
        losses = balanced_round_losses(output, episode, reference, config)
    weights = round_weights(len(output.states), config.round_loss_rho)
    loss = (losses * losses.new_tensor(weights)).sum()
    finite(loss, "joint coder loss")
    initial = score(output.initial).detach()
    return V7Score(loss, losses, initial, weights)
