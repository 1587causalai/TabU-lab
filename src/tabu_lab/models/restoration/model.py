"""Five-step reference model, with truth-free forward and explicit readout mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from torch import Tensor, nn

from ._validation import positive
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
        columns = []
        for a in range(m):
            positions = (targets[:, 1] == a).nonzero().flatten()
            if not len(positions):
                continue
            fact = facts[a]
            result = self.readout(
                shared[inverse[positions]],
                fact.rows,
                fact.answers.encoded,
                support_cells=h[fact.rows, a],
                target_cells=h[targets[positions, 0], a],
            )
            decoded = None if result.encoding is None else fact.answers.decode(result.encoding)
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return RestorationOutput(request, tuple(columns), facts, h)
