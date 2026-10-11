"""First executable completion branch, separate from existing V7 defaults."""

from dataclasses import dataclass, replace

import torch
from torch import nn

from ..restoration._validation import finite
from ..restoration.contracts import RestorationInput, TruthSidecar
from ..restoration_v7.codec import build_value_codec, null_constant_inputs
from ..restoration_v7.config import V7Config
from ..restoration_v7.model import V7Round
from ..restoration_v7.readout import column_shared_ll
from .cell_dynamics import cell_only_dynamics
from .encoder import TableEncoder
from .readout import PositiveLinearReadout, SignedLinearReadout


@dataclass(frozen=True)
class V8Config:
    # Preserve the full inherited spec for reproducibility, but V8 owns its
    # query seed, round count, coupling encoder and scorer. See README.
    inherited: V7Config
    rounds: int = 1
    readout: str = "gaussian_ll"

    def __post_init__(self):
        if not isinstance(self.inherited, V7Config) or self.inherited.code_dim != 64:
            raise ValueError("V8 requires an explicit 64-code V7/V7.3 architecture spec")
        if self.inherited.backbone.width != 128:
            raise ValueError("first V8 branch requires 128-wide Token Dynamics")
        if type(self.rounds) is not int or self.rounds < 1:
            raise ValueError("V8 rounds must be a positive integer")

        if self.readout not in (
            "gaussian_ll",
            "positive_linear_attention",
            "signed_linear_attention",
        ):
            raise ValueError("unknown V8 readout")

    def as_dict(self):
        result = {"inherited": self.inherited.as_dict(), "rounds": self.rounds}
        if self.readout != "gaussian_ll":
            result["readout"] = self.readout
        return result

    @classmethod
    def from_dict(cls, value):
        if not {"inherited", "rounds"} <= set(value) or set(value) - {
            "inherited",
            "rounds",
            "readout",
        }:
            raise ValueError("unknown V8 configuration fields")
        return cls(
            V7Config.from_dict(value["inherited"]),
            value["rounds"],
            value.get("readout", "gaussian_ll"),
        )


@dataclass(frozen=True)
class CompletionEpisode:
    inputs: RestorationInput
    codec: object
    observed: torch.Tensor


@dataclass(frozen=True)
class CompletionOutput:
    initial: torch.Tensor  # Query codes in row-major mask order
    states: tuple[torch.Tensor, ...]
    completed_codes: torch.Tensor  # visible clamped, Query predicted, Null zero
    decoded: dict[int, torch.Tensor] | None


def prepare_completion(inputs: RestorationInput, config: V8Config):
    spec = config.inherited
    inputs = null_constant_inputs(inputs, spec.numeric_preprocessing)
    if not bool(inputs.query.any()):
        raise ValueError("completion requires at least one Query cell")
    for a in inputs.query.any(0).nonzero().flatten().tolist():
        if not bool(inputs.visible[:, a].any()):
            raise ValueError(f"no-support: completion column {a}")
    codec = build_value_codec(
        inputs,
        codec=spec.codec,
        dim=64,
        epsilon=spec.epsilon,
        numeric_preprocessing=spec.numeric_preprocessing,
    )
    return CompletionEpisode(inputs, codec, codec.encode_observed(inputs))


class V8CompletionModel(nn.Module):
    def __init__(self, config: V8Config):
        super().__init__()
        self.config = config
        # Construct the exact inherited backbone, including optional Unit
        # blocks and strict-finite behavior, without copying its readback path.
        base = V7Round(
            replace(
                config.inherited, value_encoder="phi_lift", dual_stream=None, row_reversible64=None
            )
        )
        self.encoder = TableEncoder(base.phi)
        self.lift = base.lift
        self.unit_seed = base.unit_seed
        self.feature_seed = base.feature_seed
        self.backbone = base.backbone
        self.query_seed = nn.Parameter(torch.randn(64) / 8)
        if config.readout == "positive_linear_attention":
            self.attention_readout = PositiveLinearReadout()
        elif config.readout == "signed_linear_attention":
            self.attention_readout = SignedLinearReadout()
        self.migration_receipt = None
        from .precision import configure_precision

        configure_precision(
            self, mode="fast_fp32" if self.query_seed.dtype == torch.float32 else "reference"
        )

    def _apply(self, fn, recurse=True):
        """Explicit parameter dtype changes also select matching computation."""
        previous_dtype = self.query_seed.dtype
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "execution_precision"):
            from .precision import configure_precision

            mode = self.execution_precision["mode"]
            if self.query_seed.dtype == torch.float64:
                mode = "reference"
            elif previous_dtype != self.query_seed.dtype:
                mode = "fast_fp32"
            configure_precision(self, mode=mode, reason=self.execution_precision.get("reason"))
        return result

    def forward(self, episode: CompletionEpisode, *, decode=True):
        spec = self.config.inherited
        if (episode.codec.dim, episode.codec.family, episode.codec.numeric_preprocessing) != (
            64,
            spec.codec,
            spec.numeric_preprocessing,
        ):
            raise ValueError("completion episode codec differs from model")
        visible, query = episode.inputs.visible, episode.inputs.query
        if visible.device != self.query_seed.device:
            raise ValueError("completion inputs and model must share a device")
        valid = visible | query
        observed = episode.observed.to(self.query_seed)
        current = self.query_seed.expand(int(query.sum()), -1)
        initial = current
        states = []
        n, m = visible.shape
        for _ in range(self.config.rounds):
            codes = observed.clone()
            codes[query] = current
            embeddings = self.encoder(codes, valid)
            cells = self.lift(embeddings)
            sources = valid if spec.query_source else visible
            if self.config.readout != "gaussian_ll":
                h = cell_only_dynamics(self.backbone, cells, sources, query)
            else:
                units = self.unit_seed.expand(n, 1, -1)
                features = torch.cat((self.feature_seed.expand(m, -1), cells.new_zeros(1, 128)))
                h = torch.cat((torch.cat((cells, units), 1), features[None]), 0)
                h = self.backbone(h, sources, query)
            finite(h, "V8 inherited Token Dynamics")
            # Fit every column from the SAME latent snapshot. Only after all
            # predictions exist is the assembled WHOLE table inverted once.
            assembled = embeddings.clone()
            for a in query.any(0).nonzero().flatten().tolist():
                active_rows = valid[:, a].nonzero().flatten()
                supports = visible[:, a].nonzero().flatten()
                rows = query[:, a].nonzero().flatten()
                if self.config.readout != "gaussian_ll":
                    assembled[rows, a] = self.attention_readout(
                        h[rows, a], h[supports, a], embeddings[supports, a]
                    ).to(assembled)
                    continue
                local_supports = visible[active_rows, a].nonzero().flatten()
                local_queries = query[active_rows, a].nonzero().flatten()
                recovery = column_shared_ll(
                    h[active_rows, m],
                    local_supports,
                    embeddings[supports, a],
                    h[active_rows, a],
                    local_queries,
                    ridge=spec.ridge,
                    bandwidth=spec.bandwidth,
                    center_chunk_size=spec.center_chunk_size,
                    compute_dtype=self._ll_compute_dtype,
                )
                assembled[rows, a] = recovery.recovered.to(assembled)
            candidate = self.encoder.inverse(assembled, valid)
            current = candidate[query]
            finite(current, "V8 Query writeback")
            states.append(current)
        completed = observed.clone()
        completed[query] = current
        decoded = (
            None
            if not decode
            else {
                a: episode.codec.columns[a].decode(completed[query[:, a], a].detach())
                for a in query.any(0).nonzero().flatten().tolist()
            }
        )
        return CompletionOutput(initial, tuple(states), completed, decoded)


def completion_loss(
    output: CompletionOutput, episode: CompletionEpisode, truth: TruthSidecar, *, score_mask=None
):
    """Frozen visible-fitted codec; hidden truth is consumed only HERE.

    First reference objective: terminal-round mean squared error per held-out
    cell and across 64 coordinates (no type multipliers). Natural missing Query
    can be completed but never scored. An explicit scoring subset is allowed.
    """
    query = episode.inputs.query
    if truth.states.shape != query.shape or truth.states.device != query.device:
        raise ValueError("truth state must align with completion inputs")
    eligible = query & (truth.states == 1)
    if score_mask is None:
        score_mask = eligible
    if (
        score_mask.shape != query.shape
        or score_mask.dtype != torch.bool
        or score_mask.device != query.device
        or bool((score_mask & ~eligible).any())
    ):
        raise ValueError("score mask must select held-out observed Query cells")
    if not bool(score_mask.any()):
        raise ValueError("loss requires at least one held-out observed Query")
    reference = torch.zeros_like(episode.observed)
    for a in score_mask.any(0).nonzero().flatten().tolist():
        reference[score_mask[:, a], a] = (
            episode.codec.columns[a].encode(truth.values[a][score_mask[:, a]]).to(reference)
        )
    reference = reference[score_mask].detach()
    predicted = output.states[-1][score_mask[query]]
    loss = (predicted - reference.to(predicted)).square().mean()
    finite(loss, "V8 completion code loss")
    return loss
