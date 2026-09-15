"""Five-step reference model, with truth-free forward and explicit readout mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from ._packing import column_positions, padded_stack
from ._validation import positive
from .answers import NumericAnswers
from .backbone import AxialBackbone, BackboneConfig
from .contracts import RestorationInput, RestorationRequest
from .encoding import ColumnFacts, EncoderConfig, ValueEncoder
from .readout import EncodedRestoration, RestorationReadout, unit_kernel_logits


@dataclass(frozen=True)
class RestorationConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    readout: str = "ll"
    bandwidth: float = 1.0
    ridge: float = 1e-3

    def __post_init__(self):
        if self.encoder.width != self.backbone.width:
            raise ValueError("encoder and backbone widths must match")
        RestorationReadout(self.readout, self.ridge)
        positive(self.bandwidth, "bandwidth")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config):
        config = dict(config)
        config["encoder"] = EncoderConfig(**config["encoder"])
        config["backbone"] = BackboneConfig(**config["backbone"])
        return cls(**config)


@dataclass(frozen=True)
class ColumnPrediction:
    column: int
    target_indices: Tensor  # positions in original request, not table row addresses
    result: EncodedRestoration
    decoded: Tensor | None


@dataclass(frozen=True)
class RestorationOutput:
    request: RestorationRequest
    columns: tuple[ColumnPrediction, ...]
    facts: tuple[ColumnFacts, ...]
    carriers: Tensor


class RestorationModel(nn.Module):
    """No masks, truth labels, or clean-table statistics are accepted by forward.

    The only masks accepted are the two visible/query role masks inside input.
    Optional alternative encoders/backbones can be injected as nn.Modules with
    the same interfaces; the built-in configuration round-trip describes built-ins.
    """

    def __init__(self, config: RestorationConfig | None = None):
        super().__init__()
        self.config = config or RestorationConfig()
        self.encoder = ValueEncoder(self.config.encoder)
        self.backbone = AxialBackbone(self.config.backbone)
        self.readout = RestorationReadout(self.config.readout, self.config.ridge)

    def forward(self, inputs: RestorationInput, request: RestorationRequest) -> RestorationOutput:
        request.validate(inputs)
        return self._forward_prepared(inputs, request, self.encoder.prepare(inputs))

    def _forward_prepared(self, inputs, request, facts):
        h = self.backbone(self.encoder(inputs, facts), inputs.visible, inputs.query)
        n, m = inputs.visible.shape
        targets = request.targets
        # Compute geometry once per unique requested Unit, reusable by every column.
        rows, inverse = targets[:, 0].unique(sorted=True, return_inverse=True)
        shared = unit_kernel_logits(h[rows, m], h[:n, m], bandwidth=self.config.bandwidth)
        # Columns are independent given the shared kernel and final carriers:
        # pack them into padded stacks and solve every column in one batched
        # readout call instead of a Python loop. no-support columns are not
        # solved at all; they report status exactly like the per-column path.
        active, unsupported = [], []
        for a, positions in enumerate(column_positions(targets[:, 1], m)):
            if not len(positions):
                continue
            (active if len(facts[a].rows) else unsupported).append((a, positions))
        columns = [
            ColumnPrediction(a, positions, EncodedRestoration("no-support", 0), None)
            for a, positions in unsupported
        ]
        if active:
            ll = self.config.readout == "ll"
            max_targets = max(len(positions) for _, positions in active)
            max_supports = max(len(facts[a].rows) for a, _ in active)
            max_p = max(facts[a].answers.encoded.shape[1] for a, _ in active)
            device = h.device
            column_ids = torch.tensor([a for a, _ in active], device=device)
            packed_positions = padded_stack([p for _, p in active], (max_targets,))
            packed_rows = padded_stack([facts[a].rows for a, _ in active], (max_supports,))
            target_lengths = torch.tensor([len(p) for _, p in active], device=device)
            support_lengths = torch.tensor([len(facts[a].rows) for a, _ in active], device=device)
            target_mask = torch.arange(max_targets, device=device)[None] < target_lengths[:, None]
            support_mask = (
                torch.arange(max_supports, device=device)[None] < support_lengths[:, None]
            )
            logits = shared[inverse[packed_positions][..., None], packed_rows[:, None, :]]
            logits = torch.where(target_mask[..., None] & support_mask[:, None, :], logits, 0)
            answers = padded_stack(
                [facts[a].answers.encoded for a, _ in active], (max_supports, max_p)
            )
            support_cells = target_cells = None
            if ll:
                support_cells = torch.where(
                    support_mask[..., None], h[packed_rows, column_ids[:, None]], 0
                )
                target_cells = torch.where(
                    target_mask[..., None],
                    h[targets[packed_positions, 0], column_ids[:, None]], 0,
                )
            encoded, log_weights, coefficients = self.readout.batched(
                logits,
                support_mask,
                target_mask,
                answers,
                support_cells=support_cells,
                target_cells=target_cells,
            )
            # Standard numeric codecs share inverse scaling. Keep custom codecs'
            # decode methods as extension points and exclude padded targets.
            numeric_slots = [i for i, (a, _) in enumerate(active)
                             if type(facts[a].answers) is NumericAnswers]
            numeric_decoded = {}
            if numeric_slots:
                if any(facts[active[i][0]].answers.encoded.shape[1] != 1 for i in numeric_slots):
                    raise ValueError("numeric predictions require one answer coordinate")
                codes = torch.where(
                    target_mask[numeric_slots, :, None], encoded[numeric_slots, :, :1], 0
                )
                decoded = NumericAnswers.decode_batch(
                    [facts[active[i][0]].answers for i in numeric_slots], codes
                )
                numeric_decoded = dict(zip(numeric_slots, decoded.unbind(0), strict=True))
            for i, (a, positions) in enumerate(active):
                fact = facts[a]
                t_a, n_a, p_a = len(positions), len(fact.rows), fact.answers.encoded.shape[1]
                result = EncodedRestoration(
                    "ok",
                    n_a,
                    encoded[i, :t_a, :p_a],
                    log_weights[i, :t_a, :n_a],
                    coefficients[i, :t_a, :n_a],
                )
                decoded = (numeric_decoded[i][:t_a] if i in numeric_decoded
                           else fact.answers.decode(result.encoding))
                columns.append(ColumnPrediction(a, positions, result, decoded))
        columns.sort(key=lambda prediction: prediction.column)
        return RestorationOutput(request, tuple(columns), facts, h)
