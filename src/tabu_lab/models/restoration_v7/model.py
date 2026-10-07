"""V7 recovery with single-column and joint-column episodes.

Each round uses one shared backbone snapshot and true visible supports for LL.
Observed codes stay fixed; recovered Query codes remain on the gradient path.
Historical seed/fixed-source defaults are preserved in V7Config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...primitives.coupling import CouplingValueMap
from ..restoration._validation import finite
from ..restoration.backbone import OMAB, AxialBackbone
from ..restoration.contracts import RestorationInput
from .auxiliary import AuxiliaryPlan, validate_auxiliary_plan
from .codec import G64Codec, V7ProtocolError, build_value_codec, null_constant_inputs
from .config import V7Config
from .dual_stream import RowDualStreamEncoder
from .readout import column_shared_ll, evaluate_fitted_ll

if TYPE_CHECKING:
    from .joint import JointEpisode, JointOutput


@dataclass(frozen=True)
class V7Episode:
    """Forward-visible package: inputs, fixed codec, roles and the donor draw."""

    inputs: RestorationInput
    codec: G64Codec
    target: int
    support_rows: Tensor
    query_rows: Tensor
    donor_rows: Tensor  # aligned with query_rows; drawn once from support_rows
    observed: Tensor  # [N,M,p] visible codes, exact zero elsewhere
    auxiliary: AuxiliaryPlan | None = None


@dataclass(frozen=True)
class V7Output:
    initial: Tensor  # [|R_Q|,p] typed seed, or donor codes in the explicit control
    states: tuple[Tensor, ...]  # K written Query states, the loss inputs
    slopes: tuple[Tensor, ...]
    decoded: Tensor | None  # decode of the final state only
    auxiliary_states: tuple[dict[int, Tensor], ...] = ()


def target_column(inputs: RestorationInput) -> int:
    columns = inputs.query.any(0).nonzero(as_tuple=True)[0]
    if len(columns) != 1:
        raise ValueError("V7 requires Query cells in exactly one target column")
    return int(columns[0])


def prepare_episode(
    inputs: RestorationInput,
    *,
    donor_seed: int,
    code_dim: int = 64,
    epsilon: float = 1e-6,
    codec: str = "C64/8",
    admission: str = "training",
    numeric_preprocessing: str = "legacy",
) -> V7Episode:
    """Fix codec, roles and one donor per Query from visible support labels.

    Training requires ``|S_tau| >= 2`` and, for a numeric target, two distinct
    support values. Inference accepts a single support; an empty support set
    is the ``no-support`` protocol status.
    """
    if admission not in ("training", "inference"):
        raise ValueError("admission must be training or inference")
    if type(donor_seed) is not int:
        raise ValueError("donor_seed must be an explicit integer")
    inputs = null_constant_inputs(inputs, numeric_preprocessing)
    target = target_column(inputs)
    support_rows = inputs.visible[:, target].nonzero(as_tuple=True)[0]
    query_rows = inputs.query[:, target].nonzero(as_tuple=True)[0]
    if not len(support_rows):
        raise V7ProtocolError("no-support", "target column has no visible support")
    if admission == "training":
        if len(support_rows) < 2:
            raise V7ProtocolError("no-valid-episode", "training needs two target supports")
        labels = inputs.values[target][support_rows]
        if inputs.schema[target].kind == "numeric" and len(labels.unique()) < 2:
            raise V7ProtocolError("no-valid-episode", "numeric target needs diverse supports")
    codec = build_value_codec(
        inputs,
        codec=codec,
        dim=code_dim,
        epsilon=epsilon,
        numeric_preprocessing=numeric_preprocessing,
    )
    generator = torch.Generator().manual_seed(donor_seed)
    draws = torch.randint(len(support_rows), (len(query_rows),), generator=generator)
    donor_rows = support_rows[draws.to(support_rows.device)]
    observed = codec.encode_observed(inputs)
    return V7Episode(inputs, codec, target, support_rows, query_rows, donor_rows, observed)


def compose_state(observed: Tensor, query_rows: Tensor, target: int, state: Tensor) -> Tensor:
    """Out-of-place table with the Query target cells replaced by ``state``."""
    columns = torch.full_like(query_rows, target)
    return observed.index_put((query_rows, columns), state)


def _unit_vector(width: int) -> Tensor:
    value = torch.randn(width)
    return value / value.norm()


def _query_seed(code_dim: int) -> Tensor:
    """Same scale as the inherited cell Query seed: ``randn / sqrt(dim)``."""
    return torch.randn(code_dim) / math.sqrt(code_dim)


def _initialize_omab(module: OMAB) -> None:
    for linear in (module.q, module.k, module.v, module.out, module.ff[0], module.ff[2]):
        nn.init.xavier_uniform_(linear.weight)


def _initialize_backbone(backbone: AxialBackbone) -> None:
    for module in backbone.modules():
        if isinstance(module, OMAB):
            _initialize_omab(module)
    for layer in backbone.layers:
        with torch.no_grad():
            for slot_seed in (layer.slot_seed, layer.row_slot_seed):
                if slot_seed is not None:
                    seeds = torch.randn_like(slot_seed)
                    slot_seed.copy_(seeds / seeds.norm(dim=-1, keepdim=True))


class IdentityValueMap(nn.Module):
    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class InheritedV6Dynamics(nn.Module):
    """Serializable optional Axial + Unit stack; default V7 remains unchanged."""

    def __init__(self, config, unit_layers, *, unit_source_policy="legacy_cell_sources"):
        super().__init__()
        if unit_source_policy not in ("observed", "legacy_cell_sources"):
            raise ValueError("unknown Unit source policy")
        self.unit_source_policy = unit_source_policy
        self.axial = AxialBackbone(config)
        self.unit_blocks = nn.ModuleList(OMAB(config) for _ in range(unit_layers))

    def forward(self, h, visible, query):
        h = self.axial(h, visible, query)
        n, m = visible.shape
        units = h[:n, m]
        # The caller passes Cell source eligibility as ``visible``. In cyclic
        # runs that includes Query, whose estimates are not real Unit evidence.
        # Query and observed roles are disjoint, so removing Query recovers the
        # true observed mask. Historical checkpoints retain their original graph.
        unit_sources = visible & ~query if self.unit_source_policy == "observed" else visible
        for block in self.unit_blocks:
            units = block(units, units, unit_sources.any(-1))
        rows = torch.arange(n, device=h.device)
        return h.index_put((rows, torch.full_like(rows, m)), units)


class V7Round(nn.Module):
    """One full representation-recovery module: ``phi``, ``W_up``, seeds, backbone."""

    def __init__(self, config: V7Config):
        super().__init__()
        width = config.backbone.width
        if config.value_encoder == "row_dual_stream":
            self._init_row_dual_stream(config)
            return
        self.phi = CouplingValueMap(
            config.code_dim,
            n_blocks=config.coupling_blocks,
            hidden=config.coupling_hidden,
            alpha=config.coupling_alpha,
            scale=config.coupling_scale,
            bias=config.coupling_bias,
        )
        if config.value_map == "identity":
            self.phi = IdentityValueMap()
        self.lift = nn.Linear(config.code_dim, width, bias=False)
        nn.init.orthogonal_(self.lift.weight)
        self.unit_seed = nn.Parameter(_unit_vector(width))
        self.feature_seed = nn.Parameter(_unit_vector(width))
        # V7 retains its original construction order, including the discarded
        # initialized axial stack: those random draws determine subsequent
        # weights and must remain reproducible from the same seed.
        if not config.unit_layers or config.model_version == "v7":
            self.backbone = AxialBackbone(config.backbone)
            _initialize_backbone(self.backbone)
        if config.unit_layers:
            self.backbone = InheritedV6Dynamics(
                config.backbone,
                config.unit_layers,
                unit_source_policy=config.unit_source_policy,
            )
            if config.model_version == "v7.3":
                _initialize_backbone(self.backbone.axial)
                for module in self.backbone.unit_blocks:
                    _initialize_omab(module)
        # Keep shared V5/V6/V7 operator defaults frozen. Only the declared new
        # protocol changes the attention's numerical-failure boundary.
        if config.model_version == "v7.3":
            for module in self.backbone.modules():
                if isinstance(module, OMAB):
                    module.strict_finite_content = True

        if config.value_encoder == "row_reversible64":
            from .reversible64 import RowReversible64Encoder
            self.row_encoder = RowReversible64Encoder(
                config.row_reversible64, strict_finite_content=config.model_version == "v7.3"
            )

    def _init_row_dual_stream(self, config: V7Config) -> None:
        """Experimental branch: row encoder ``G`` replaces ``phi`` and ``lift``.

        Unit/Feature seeds and token dynamics keep their names and roles, so
        compatible donor tensors map one-to-one. No entry lift, no target
        broadcast, no joint/role embedding.
        """
        width = config.backbone.width
        strict = config.model_version == "v7.3"
        self.encoder = RowDualStreamEncoder(
            config.dual_stream, config.code_dim, strict_finite_content=strict
        )
        self.unit_seed = nn.Parameter(_unit_vector(width))
        self.feature_seed = nn.Parameter(_unit_vector(width))
        if config.unit_layers:
            self.backbone = InheritedV6Dynamics(
                config.backbone,
                config.unit_layers,
                unit_source_policy=config.unit_source_policy,
            )
            _initialize_backbone(self.backbone.axial)
            for module in self.backbone.unit_blocks:
                _initialize_omab(module)
        else:
            self.backbone = AxialBackbone(config.backbone)
            _initialize_backbone(self.backbone)
        if strict:
            for module in self.backbone.modules():
                if isinstance(module, OMAB):
                    module.strict_finite_content = True


class V7Model(nn.Module):
    def __init__(self, config: V7Config | None = None):
        super().__init__()
        self.config = config or V7Config()
        count = 1 if self.config.share_rounds else self.config.rounds
        self.rounds = nn.ModuleList(V7Round(self.config) for _ in range(count))
        dim = self.config.code_dim
        self.query_seed_numeric = nn.Parameter(_query_seed(dim))
        self.query_seed_nominal = nn.Parameter(_query_seed(dim))
        self.query_seed_ordinal = nn.Parameter(_query_seed(dim))

    def query_seed(self, kind: str) -> nn.Parameter:
        if kind == "numeric":
            return self.query_seed_numeric
        if kind == "nominal":
            return self.query_seed_nominal
        if kind == "ordinal":
            return self.query_seed_ordinal
        raise ValueError("query seed kind must be numeric, nominal, or ordinal")

    def round_module(self, t: int) -> V7Round:
        return self.rounds[0 if self.config.share_rounds else t]

    def run_backbone(self, block: V7Round, h: Tensor, visible: Tensor, query: Tensor):
        sources = visible | query if self.config.query_source else visible
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(block.backbone, h, sources, query, use_reentrant=False)
        return block.backbone(h, sources, query)

    def forward(
        self, episode: V7Episode | JointEpisode, *, decode: bool = True
    ) -> V7Output | JointOutput:
        from .joint import JointEpisode, forward_joint

        if self.config.value_encoder == "row_reversible64":
            from .reversible64 import forward_reversible64
            return forward_reversible64(self, episode, decode=decode)
        if self.config.value_encoder == "row_dual_stream":
            from .dual_stream import forward_dual_stream

            return forward_dual_stream(self, episode, decode=decode)
        if isinstance(episode, JointEpisode):
            return forward_joint(self, episode, decode=decode)
        config = self.config
        if episode.codec.dim != config.code_dim:
            raise ValueError("episode codec dimension differs from the model code_dim")
        if episode.codec.family != config.codec:
            raise ValueError("episode codec family differs from the model codec")
        if episode.codec.numeric_preprocessing != config.numeric_preprocessing:
            raise ValueError("episode numeric preprocessing differs from model config")
        inputs, target = episode.inputs, episode.target
        validate_auxiliary_plan(episode, config)
        visible, query = inputs.visible, inputs.query
        n, m = visible.shape
        dtype = self.rounds[0].unit_seed.dtype
        observed = episode.observed.to(dtype)
        active_rows, active_cols = (visible | query).nonzero(as_tuple=True)
        support_address = torch.full_like(episode.support_rows, target)
        if config.query_init == "donor":
            current = observed[episode.donor_rows, target]
        else:
            seed = self.query_seed(inputs.schema[target].kind).to(
                dtype=dtype, device=observed.device
            )
            current = seed.expand(len(episode.query_rows), -1).clone()
        initial = current
        states, slopes = [], []
        auxiliary_states = []
        for t in range(config.rounds):
            block = self.round_module(t)
            table = compose_state(observed, episode.query_rows, target, current)
            embedded = block.phi(table[active_rows, active_cols])
            embeddings = torch.zeros_like(table).index_put((active_rows, active_cols), embedded)
            cells = torch.zeros(n, m, block.lift.out_features, dtype=dtype, device=table.device)
            cells = cells.index_put((active_rows, active_cols), block.lift(embedded))
            units = block.unit_seed.expand(n, -1)[:, None]
            features = torch.cat(
                (block.feature_seed.expand(m, -1), cells.new_zeros(1, cells.shape[2]))
            )
            h = torch.cat((torch.cat((cells, units), dim=1), features[None]), dim=0)
            finite(h, "V7 compiled carriers")
            h = self.run_backbone(block, h, visible, query)
            responses = embeddings[episode.support_rows, support_address]
            recovery = column_shared_ll(
                h[:n, m],
                episode.support_rows,
                responses,
                h[:n, target],
                episode.query_rows,
                ridge=config.ridge,
                bandwidth=config.bandwidth,
                center_chunk_size=config.center_chunk_size,
            )
            finite(recovery.recovered, "V7 recovered embedding")
            current = block.phi.inverse(recovery.recovered.to(dtype))
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError("inverse-failed: nonfinite Query state after phi inverse")
            states.append(current)
            slopes.append(recovery.slope)
            if episode.auxiliary is not None:
                if config.loss_mode != "balanced_reconstruction":
                    raise ValueError("auxiliary plan requires balanced_reconstruction")
                (part,) = episode.auxiliary.columns
                if part.column != target:
                    raise ValueError("auxiliary column differs from target")
                aux = evaluate_fitted_ll(
                    h[:n, m],
                    episode.support_rows,
                    responses,
                    h[:n, target],
                    part.rows,
                    slope=recovery.slope,
                    bandwidth=config.bandwidth,
                )
                predicted = block.phi.inverse(aux.to(dtype))
                finite(predicted, "auxiliary reconstructed codes")
                auxiliary_states.append({target: predicted})
        decoded = episode.codec.columns[target].decode(current.detach()) if decode else None
        return V7Output(initial, tuple(states), tuple(slopes), decoded, tuple(auxiliary_states))


__all__ = [
    "V7Episode",
    "V7Model",
    "V7Output",
    "V7Round",
    "compose_state",
    "prepare_episode",
    "target_column",
]
