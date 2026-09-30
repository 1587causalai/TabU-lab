"""Visible facts and trainable input maps have separate lifetimes and widths."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ._validation import finite, positive
from .answers import CategoricalAnswers, NumericAnswers
from .contracts import RestorationInput


@dataclass(frozen=True)
class EncoderConfig:
    width: int = 128
    category_map: str = "identity128"
    mlp_hidden: int = 128
    epsilon: float = 1e-6

    def __post_init__(self):
        if type(self.width) is not int or self.width < 128:
            raise ValueError("carrier width must be at least 128")
        if self.category_map not in ("identity128", "rotary32", "mlp32", "mlp256"):
            raise ValueError("unknown category input map")
        if type(self.mlp_hidden) is not int or self.mlp_hidden < 1:
            raise ValueError("MLP hidden width must be positive")
        # Shared default scale floor in each column's ORIGINAL units; a column
        # rescaling x' = gamma*x + beta must carry epsilon' = gamma*epsilon.
        positive(self.epsilon, "epsilon")

    @property
    def answer_width(self):
        return {"identity128": 128, "rotary32": 32, "mlp32": 32, "mlp256": 256}[self.category_map]


@dataclass(frozen=True)
class ColumnFacts:
    rows: Tensor
    answers: NumericAnswers | CategoricalAnswers
    input_coordinates: Tensor  # scalar numeric coordinate or raw category code
    rank: Tensor | None = None  # ordinal declared-order rank r(x) in [0,1]; else None


def visible_codes(labels: Tensor, *, width: int, seed: int, key: str) -> tuple[Tensor, Tensor]:
    """Stable per-column CPU generator, no dependence on process RNG or row order."""
    classes = labels.unique(sorted=True)
    if len(classes) > math.comb(width, 8):
        raise ValueError("visible classes exceed the distinct constant-weight code capacity")
    identity = json.dumps([seed, key, classes.cpu().tolist()], ensure_ascii=True).encode()
    local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    gen = torch.Generator(device="cpu").manual_seed(local_seed)
    codes = torch.zeros(len(classes), width, dtype=torch.float64)
    seen = set()
    for i in range(len(classes)):
        while True:
            positions = tuple(sorted(torch.randperm(width, generator=gen)[:8].tolist()))
            if positions not in seen:
                break
        seen.add(positions)
        codes[i, list(positions)] = 1
    return classes, codes.to(labels.device)


@dataclass(frozen=True)
class EncodingLayout:
    groups: tuple  # (kind, fixed coordinates, optional ordinal rank)
    addresses: Tensor
    active: Tensor


def prepare_features(inputs, facts):
    n, m = inputs.visible.shape
    groups = {}
    for a, schema in enumerate(inputs.schema):
        groups.setdefault(schema.kind, []).append(a)
    fixed, addresses = [], []
    for kind, columns in groups.items():
        coord = torch.cat([facts[a].input_coordinates for a in columns])
        rank = torch.cat([facts[a].rank for a in columns]) if kind == "ordinal" else None
        fixed.append((kind, coord, rank))
        addresses.extend(facts[a].rows * (m + 1) + a for a in columns)
    active = torch.zeros(n + 1, m + 1, dtype=torch.bool, device=inputs.visible.device)
    active[:n, :m] = inputs.visible | inputs.query
    active[:n, m] = True
    active[n, :m] = True
    return EncodingLayout(tuple(fixed), torch.cat(addresses), active)


class ValueEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.frequencies = nn.Parameter(torch.randn(64))
        self.projection = nn.Linear(128, config.width, bias=False)
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(config.width, 128), mode="reduced")
            self.projection.weight.copy_(q / 8)
        self.cell_seed = nn.Parameter(torch.randn(config.width) / math.sqrt(config.width))
        self.unit_seed = nn.Parameter(torch.randn(config.width) / math.sqrt(config.width))
        self.feature_seed = nn.Parameter(torch.randn(config.width) / math.sqrt(config.width))
        if config.category_map == "rotary32":
            self.angles = nn.Parameter(torch.randn(4, 16))
        else:
            self.register_parameter("angles", None)
        if config.category_map.startswith("mlp"):
            self.category_mlp = nn.Sequential(
                nn.Linear(config.answer_width, config.mlp_hidden),
                nn.GELU(),
                nn.Linear(config.mlp_hidden, 128),
            )
        else:
            self.category_mlp = nn.Identity()

    def prepare(self, inputs: RestorationInput) -> tuple[ColumnFacts, ...]:
        """Nonlearned artifacts; reads only stored visible values, never target truth."""
        counts = inputs.visible.sum(0).tolist()
        rows_by_column = inputs.visible.T.nonzero()[:, 1].split(counts)
        numeric_groups = {}
        for a, schema in enumerate(inputs.schema):
            if schema.kind == "numeric":
                numeric_groups.setdefault(counts[a], []).append(a)
        numeric_codecs = {}
        for columns in numeric_groups.values():
            values = torch.stack([inputs.values[a][rows_by_column[a]] for a in columns])
            codecs = NumericAnswers.from_visible_batch(values, epsilon=self.config.epsilon)
            numeric_codecs.update(zip(columns, codecs, strict=True))
        facts = []
        for a, schema in enumerate(inputs.schema):
            rows = rows_by_column[a]
            if schema.kind == "numeric":
                # Input coordinate IS the answer encoding: one shared robust
                # coordinate (median/half-IQR) per the v2 design, not a second
                # preprocessing scale.
                codec = numeric_codecs[a]
                coordinates = codec.encoded
                rank = None
            else:
                values = inputs.values[a][rows]
                # Alternative ordinal lifts are design-open. Nominal alternatives
                # coexist with the declared default ordinal 128/8 + rank lift.
                width = 128 if schema.kind == "ordinal" else self.config.answer_width
                classes, codes = visible_codes(
                    values, width=width, seed=inputs.code_seed, key=schema.key
                )
                codec = CategoricalAnswers.from_visible(
                    values, classes, codes, domain_size=schema.domain_size
                )
                coordinates = codec.encoded
                if schema.kind == "ordinal":
                    # rank_a(x) counts categories strictly below x under the
                    # DECLARED order ≺_a (ColumnSchema.order), never the label
                    # index. Identity order is an explicit caller convention.
                    positions = torch.tensor(
                        schema.rank_positions(), dtype=torch.float64, device=values.device
                    )
                    rank = positions[values] / max(schema.domain_size - 1, 1)
                    finite(rank, "ordinal rank coordinates")
                else:
                    rank = None
            # The codec checked all coordinate values, batched for numeric columns.
            facts.append(ColumnFacts(rows, codec, coordinates, rank))
        return tuple(facts)

    def category_features(self, codes: Tensor) -> Tensor:
        if self.angles is None:
            return self.category_mlp(codes)
        pairs = codes.reshape(-1, 16, 2)
        x, y = pairs[..., 0][:, None], pairs[..., 1][:, None]
        cos, sin = self.angles.cos(), self.angles.sin()
        return torch.stack((x * cos - y * sin, x * sin + y * cos), -1).flatten(1)

    def forward(self, inputs: RestorationInput, facts: tuple[ColumnFacts, ...]) -> Tensor:
        return self.forward_prepared(inputs, prepare_features(inputs, facts))

    def forward_prepared(self, inputs: RestorationInput, layout: EncodingLayout) -> Tensor:
        n, m = inputs.visible.shape
        weight = self.projection.weight
        if inputs.visible.device != weight.device:
            raise ValueError("model and input must share a device")
        h = weight.new_zeros(n + 1, m + 1, self.config.width)
        h[:n, :m] = torch.where(inputs.query[..., None], self.cell_seed, 0)
        h[:n, m] = self.unit_seed
        h[n, :m] = self.feature_seed
        feature_parts = []
        for kind, coordinates, fixed_rank in layout.groups:
            coord = coordinates.to(weight)
            if kind == "numeric":
                phase = coord * self.frequencies
                features = torch.cat((phase.sin(), phase.cos()), -1)
            elif kind == "ordinal":
                # b_a(x) + r_a(x)·1: rank comes from prepare, derived from the
                # declared order — not from raw label indices in inputs.values.
                rank = fixed_rank.to(weight)
                features = coord + rank[:, None]
            else:
                features = self.category_features(coord)
            feature_parts.append(features)
        h = h.flatten(0, 1).index_copy(
            0, layout.addresses, self.projection(torch.cat(feature_parts))
        ).reshape(n + 1, m + 1, self.config.width)
        finite(h, "initial carriers")
        if bool((h[layout.active].square().sum(-1) == 0).any()):
            raise FloatingPointError("non-Null input carrier collapsed to zero")
        return h
