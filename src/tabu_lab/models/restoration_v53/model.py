"""V5.3 reference path, separate from historical restoration checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import Tensor, nn

from ..restoration._packing import column_positions
from ..restoration._prepared import TensorVersions
from ..restoration._validation import finite, positive
from ..restoration.backbone import OMAB, AxialBackbone, BackboneConfig
from ..restoration.contracts import RestorationInput, RestorationRequest
from ..restoration.encoding import EncodingLayout, prepare_features
from ..restoration.model import ColumnPrediction
from ..restoration.readout import EncodedRestoration
from .encoding import ANSWER_WIDTH, AffineValueEncoder, V53ColumnFacts
from .readout import FeatureSlopeProvider, evaluate_column, shared_slope


@dataclass(frozen=True)
class V53Config:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    epsilon: float = 1e-6
    bandwidth: float = 1.0
    ridge: float = 1e-3
    unit_layers: int = 0
    regression_width: int | None = None  # None is exact identity; explicit width enables P_R.
    center_chunk_size: int = 32
    slope_source: str = "shared_ll"
    codec_version: str = "unit_gaussian_v1"
    numeric_scaling: str = "zscore"

    def __post_init__(self):
        if self.backbone.width < ANSWER_WIDTH:
            raise ValueError("V5.3 carrier width must be at least 128")
        for name in ("epsilon", "bandwidth", "ridge"):
            positive(getattr(self, name), name)
        if type(self.unit_layers) is not int or self.unit_layers < 0:
            raise ValueError("unit_layers must be a nonnegative integer")
        if self.regression_width is not None and (
            type(self.regression_width) is not int or self.regression_width < 1
        ):
            raise ValueError("regression_width must be a positive integer or None")
        if type(self.center_chunk_size) is not int or self.center_chunk_size < 1:
            raise ValueError("center_chunk_size must be a positive integer")
        if self.slope_source not in ("shared_ll", "feature"):
            raise ValueError("slope_source must be shared_ll or feature")
        if self.codec_version not in ("unit_gaussian_v1", "legacy_v53"):
            raise ValueError("unknown V5.3 codec version")
        if self.numeric_scaling not in ("zscore", "median_half_iqr"):
            raise ValueError("unknown numeric scaling")

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if "codec_version" not in values or "numeric_scaling" not in values:
            raise ValueError(
                "V5.3 config lacks codec identity; historical configs require explicit "
                "codec_version='legacy_v53', numeric_scaling='median_half_iqr'"
            )
        values["backbone"] = BackboneConfig(**values["backbone"])
        return cls(**values)


@dataclass(frozen=True)
class PreparedV53:
    """Owned visible-only snapshot, reusable across parameter updates."""

    inputs: RestorationInput
    request: RestorationRequest
    facts: tuple[V53ColumnFacts, ...]
    features: EncodingLayout
    positions: tuple[Tensor, ...]
    encoder_id: int
    config: V53Config
    versions: TensorVersions = field(repr=False, compare=False)


@dataclass(frozen=True)
class V53Output:
    request: RestorationRequest
    columns: tuple[ColumnPrediction, ...]
    facts: tuple[V53ColumnFacts, ...]
    carriers: Tensor
    units: Tensor
    slopes: dict[int, Tensor]


class V53Model(nn.Module):
    def __init__(
        self, config: V53Config | None = None, *, feature_slope: FeatureSlopeProvider | None = None
    ):
        super().__init__()
        self.config = config or V53Config()
        config = self.config
        if (config.slope_source == "feature") != (feature_slope is not None):
            raise ValueError("feature slope mode requires an explicit FeatureSlopeProvider")
        if feature_slope is not None and not isinstance(feature_slope, FeatureSlopeProvider):
            raise TypeError("feature_slope must implement FeatureSlopeProvider")
        self.encoder = AffineValueEncoder(
            config.backbone.width, config.epsilon, codec_version=config.codec_version,
            numeric_scaling=config.numeric_scaling,
        )
        self.register_buffer("_codec_signature", torch.tensor([
            {"legacy_v53": 0, "unit_gaussian_v1": 1}[config.codec_version],
            {"median_half_iqr": 0, "zscore": 1}[config.numeric_scaling],
        ], dtype=torch.long))
        self.backbone = AxialBackbone(config.backbone)
        self.unit_blocks = nn.ModuleList(OMAB(config.backbone) for _ in range(config.unit_layers))
        self.regression = (
            nn.Identity() if config.regression_width is None
            else nn.Linear(config.backbone.width, config.regression_width, bias=False)
        )
        self.feature_slope = feature_slope

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        key = prefix + "_codec_signature"
        signature = state_dict.get(key)
        if signature is None:
            if self.config.codec_version == "legacy_v53" and (
                self.config.numeric_scaling == "median_half_iqr"
            ):
                # Explicit legacy construction is the only unversioned migration.
                state_dict[key] = self._codec_signature.detach().clone()
            else:
                error_msgs.append("checkpoint lacks codec identity; select the explicit legacy "
                                  "codec or perform a documented weights-only conversion")
        elif not torch.equal(signature.cpu(), self._codec_signature.cpu()):
            error_msgs.append("checkpoint codec identity does not match the V5.3 model config")
            state_dict[key] = self._codec_signature.detach().clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
        )

    @torch.inference_mode(False)
    @torch.no_grad()
    def prepare(self, inputs: RestorationInput, request: RestorationRequest) -> PreparedV53:
        # Even public forward under inference_mode needs owned ordinary tensors:
        # TensorVersions protects their reuse, and inference tensors lack _version.
        # The caller's inference/gradient context resumes after preparation.
        owned = RestorationInput(
            inputs.schema, inputs.values, inputs.visible, inputs.query, inputs.code_seed
        )
        request = RestorationRequest(request.targets.detach().clone())
        request.validate(owned)
        facts = self.encoder.prepare(owned)
        features = prepare_features(owned, facts)
        positions = tuple(column_positions(request.targets[:, 1], len(facts)))
        versions = TensorVersions(owned, request, facts, features, positions)
        return PreparedV53(
            owned, request, facts, features, positions, id(self.encoder), self.config, versions
        )

    def forward(self, inputs: RestorationInput, request: RestorationRequest) -> V53Output:
        return self.forward_prepared(self.prepare(inputs, request))

    def forward_prepared(self, prepared: PreparedV53, *, decode: bool = True) -> V53Output:
        if prepared.encoder_id != id(self.encoder) or prepared.config != self.config:
            raise ValueError("prepared model changed; prepare a new snapshot")
        prepared.versions.validate()
        inputs, request, facts = prepared.inputs, prepared.request, prepared.facts
        h = self.encoder.forward_prepared(inputs, prepared.features)
        h = self.backbone(h, inputs.visible, inputs.query)
        n, m = inputs.visible.shape
        units = h[:n, m]
        # LU=0 is a literal identity path: no normalization, projection or FFN.
        eligible = inputs.visible.any(-1)
        for block in self.unit_blocks:
            units = block(units, units, eligible)
        columns, slopes = [], {}
        for a, positions in enumerate(prepared.positions):
            if not len(positions):
                continue
            fact = facts[a]
            if not len(fact.rows):
                columns.append(ColumnPrediction(a, positions, EncodedRestoration("no-support", 0),
                                                None))
                continue
            cells = self.regression(h[:n, a])
            finite(cells, "regression Cell features")
            if self.feature_slope is None:
                slope = shared_slope(
                    units, fact.rows, cells[fact.rows], fact.answers.encoded,
                    ridge=self.config.ridge, bandwidth=self.config.bandwidth,
                    center_chunk_size=self.config.center_chunk_size,
                )
            else:
                slope = self.feature_slope(h[n, a])
                if (slope.shape != (ANSWER_WIDTH, cells.shape[1])
                        or not slope.is_floating_point() or slope.device != h.device):
                    raise ValueError("Feature slope must have floating [128,d_R] shape on device")
                slope = slope.double()
                finite(slope, "Feature slope")
            slopes[a] = slope
            result = evaluate_column(
                units, cells, fact.rows, fact.answers.encoded, request.targets[positions, 0],
                slope, bandwidth=self.config.bandwidth, chunk_size=self.config.center_chunk_size,
            )
            decoded = fact.answers.decode(result.encoding) if decode else None
            columns.append(ColumnPrediction(a, positions, result, decoded))
        return V53Output(request, tuple(columns), facts, h, units, slopes)
