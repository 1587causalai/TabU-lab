"""Five-step reference model, with truth-free forward and explicit readout mode."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
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
        # Columns are independent given the shared kernel and final carriers:
        # pack them into padded stacks and solve every column in one batched
        # readout call instead of a Python loop. no-support columns are not
        # solved at all; they report status exactly like the per-column path.
        active, unsupported = [], []
        for a in range(m):
            positions = (targets[:, 1] == a).nonzero().flatten()
            if not len(positions):
                continue
            (active if len(facts[a].rows) else unsupported).append((a, positions))
        columns = [
            ColumnPrediction(a, positions, EncodedRestoration("no-support", 0), None)
            for a, positions in unsupported
        ]
        if active:
            ll = self.config.readout == "ll"
            width = h.shape[-1]
            n_cols = len(active)
            max_targets = max(len(positions) for _, positions in active)
            max_supports = max(len(facts[a].rows) for a, _ in active)
            max_p = max(facts[a].answers.encoded.shape[1] for a, _ in active)
            device, f64 = h.device, torch.float64
            logits = torch.zeros(n_cols, max_targets, max_supports, dtype=f64, device=device)
            support_mask = torch.zeros(n_cols, max_supports, dtype=torch.bool, device=device)
            target_mask = torch.zeros(n_cols, max_targets, dtype=torch.bool, device=device)
            answers = torch.zeros(n_cols, max_supports, max_p, dtype=f64, device=device)
            support_cells = target_cells = None
            if ll:
                support_cells = torch.zeros(
                    n_cols, max_supports, width, dtype=f64, device=device
                )
                target_cells = torch.zeros(
                    n_cols, max_targets, width, dtype=f64, device=device
                )
            for i, (a, positions) in enumerate(active):
                fact = facts[a]
                t_a, n_a, p_a = len(positions), len(fact.rows), fact.answers.encoded.shape[1]
                target_mask[i, :t_a] = True
                support_mask[i, :n_a] = True
                logits[i, :t_a, :n_a] = shared[inverse[positions]][:, fact.rows].to(f64)
                answers[i, :n_a, :p_a] = fact.answers.encoded.to(f64)
                if ll:
                    support_cells[i, :n_a] = h[fact.rows, a].to(f64)
                    target_cells[i, :t_a] = h[targets[positions, 0], a].to(f64)
            encoded, log_weights, coefficients = self.readout.batched(
                logits,
                support_mask,
                target_mask,
                answers,
                support_cells=support_cells,
                target_cells=target_cells,
            )
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
                decoded = fact.answers.decode(result.encoding)
                columns.append(ColumnPrediction(a, positions, result, decoded))
        columns.sort(key=lambda prediction: prediction.column)
        return RestorationOutput(request, tuple(columns), facts, h)
