"""Five-step reference model, with truth-free forward and explicit readout mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from ._packing import column_positions, padded_stack
from ._prepared import TensorVersions
from ._validation import positive
from .answers import NumericAnswers
from .backbone import AxialBackbone, BackboneConfig
from .contracts import RestorationInput, RestorationRequest
from .encoding import ColumnFacts, EncoderConfig, EncodingLayout, ValueEncoder, prepare_features
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


@dataclass(frozen=True)
class ReadoutLayout:
    positions: tuple[Tensor, ...]
    unit_rows: Tensor
    active: tuple
    unsupported: tuple
    column_ids: Tensor | None = None
    packed_rows: Tensor | None = None
    target_rows: Tensor | None = None
    unit_indices: Tensor | None = None
    target_mask: Tensor | None = None
    support_mask: Tensor | None = None
    answers: Tensor | None = None
    numeric_slots: tuple[int, ...] = ()


def prepare_readout(request, facts):
    targets = request.targets
    rows, inverse = targets[:, 0].unique(sorted=True, return_inverse=True)
    positions = tuple(column_positions(targets[:, 1], len(facts)))
    active, unsupported = [], []
    for a, selected in enumerate(positions):
        if len(selected):
            (active if len(facts[a].rows) else unsupported).append((a, selected))
    if not active:
        return ReadoutLayout(positions, rows, (), tuple(unsupported))
    max_targets = max(len(p) for _, p in active)
    max_supports = max(len(facts[a].rows) for a, _ in active)
    max_p = max(facts[a].answers.encoded.shape[1] for a, _ in active)
    device = targets.device
    column_ids = torch.tensor([a for a, _ in active], device=device)
    packed_positions = padded_stack([p for _, p in active], (max_targets,))
    packed_rows = padded_stack([facts[a].rows for a, _ in active], (max_supports,))
    target_lengths = torch.tensor([len(p) for _, p in active], device=device)
    support_lengths = torch.tensor([len(facts[a].rows) for a, _ in active], device=device)
    target_mask = torch.arange(max_targets, device=device)[None] < target_lengths[:, None]
    support_mask = torch.arange(max_supports, device=device)[None] < support_lengths[:, None]
    answers = padded_stack([facts[a].answers.encoded for a, _ in active], (max_supports, max_p))
    numeric_slots = tuple(i for i, (a, _) in enumerate(active)
                          if type(facts[a].answers) is NumericAnswers)
    if any(facts[active[i][0]].answers.encoded.shape[1] != 1 for i in numeric_slots):
        raise ValueError("numeric predictions require one answer coordinate")
    return ReadoutLayout(
        positions, rows, tuple(active), tuple(unsupported), column_ids, packed_rows,
        targets[packed_positions, 0], inverse[packed_positions], target_mask,
        support_mask, answers, numeric_slots,
    )


@dataclass(frozen=True)
class PreparedRestoration:
    """Visible-only snapshot. Never contains scorer truth or learned carriers."""

    inputs: RestorationInput
    request: RestorationRequest
    facts: tuple[ColumnFacts, ...]
    layout: ReadoutLayout
    features: EncodingLayout
    encoder_id: int
    encoder_config: EncoderConfig
    versions: TensorVersions = field(repr=False, compare=False)


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

    @torch.no_grad()
    def prepare(self, inputs: RestorationInput, request: RestorationRequest) -> PreparedRestoration:
        """Own one fixed input/request snapshot for repeated trainable forwards.

        The built-in encoder has parameter-independent preparation. Custom
        encoders retain the ordinary forward API; they must define their own
        preparation lifetime rather than silently reusing learned artifacts.
        """
        if type(self.encoder) is not ValueEncoder:
            raise ValueError("prepared execution requires the built-in ValueEncoder")
        owned = RestorationInput(inputs.schema, inputs.values, inputs.visible, inputs.query,
                                 inputs.code_seed)
        request = RestorationRequest(request.targets.detach().clone())
        request.validate(owned)
        facts = self.encoder.prepare(owned)
        layout = prepare_readout(request, facts)
        features = prepare_features(owned, facts)
        versions = TensorVersions(owned, request, facts, layout, features)
        return PreparedRestoration(owned, request, facts, layout, features, id(self.encoder),
                                   self.encoder.config, versions)

    def forward_prepared(self, prepared: PreparedRestoration, *, decode=True) -> RestorationOutput:
        """Replay fixed facts/indices; recompute every learned operation and LL system."""
        if (prepared.encoder_id != id(self.encoder)
                or prepared.encoder_config != self.encoder.config):
            raise ValueError("prepared encoder changed; prepare a new snapshot")
        prepared.versions.validate()
        return self._forward_prepared(
            prepared.inputs, prepared.request, prepared.facts,
            layout=prepared.layout, features=prepared.features, decode=decode,
        )

    def forward(self, inputs: RestorationInput, request: RestorationRequest) -> RestorationOutput:
        request.validate(inputs)
        return self._forward_prepared(inputs, request, self.encoder.prepare(inputs))

    def _forward_prepared(self, inputs, request, facts, *, layout=None, features=None, decode=True):
        initial = (self.encoder(inputs, facts) if features is None
                   else self.encoder.forward_prepared(inputs, features))
        h = self.backbone(initial, inputs.visible, inputs.query)
        n, m = inputs.visible.shape
        layout = layout or prepare_readout(request, facts)
        shared = unit_kernel_logits(h[layout.unit_rows, m], h[:n, m],
                                    bandwidth=self.config.bandwidth)
        active = layout.active
        columns = [
            ColumnPrediction(a, positions, EncodedRestoration("no-support", 0), None)
            for a, positions in layout.unsupported
        ]
        if active:
            ll = self.config.readout == "ll"
            column_ids, packed_rows = layout.column_ids, layout.packed_rows
            target_mask, support_mask = layout.target_mask, layout.support_mask
            logits = shared[layout.unit_indices[..., None], packed_rows[:, None, :]]
            logits = torch.where(target_mask[..., None] & support_mask[:, None, :], logits, 0)
            answers = layout.answers
            support_cells = target_cells = None
            if ll:
                support_cells = torch.where(
                    support_mask[..., None], h[packed_rows, column_ids[:, None]], 0
                )
                target_cells = torch.where(
                    target_mask[..., None],
                    h[layout.target_rows, column_ids[:, None]], 0,
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
            numeric_slots = list(layout.numeric_slots) if decode else []
            numeric_decoded = {}
            if numeric_slots:
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
                decoded = None
                if decode:
                    decoded = (numeric_decoded[i][:t_a] if i in numeric_decoded
                               else fact.answers.decode(result.encoding))
                columns.append(ColumnPrediction(a, positions, result, decoded))
        columns.sort(key=lambda prediction: prediction.column)
        return RestorationOutput(request, tuple(columns), facts, h)
