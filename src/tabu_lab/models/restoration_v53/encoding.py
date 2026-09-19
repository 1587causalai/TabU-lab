"""V5.3 episode-local affine value spaces; no learned or hidden codec state."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..restoration._validation import finite, matrix, positive
from ..restoration.answers import CategoricalAnswers, NumericAnswers
from ..restoration.contracts import RestorationInput
from ..restoration.encoding import EncodingLayout, prepare_features, visible_codes

ANSWER_WIDTH = 128


@dataclass(frozen=True)
class AffineNumericAnswers:
    scalar: NumericAnswers
    origin: Tensor
    direction: Tensor
    encoded: Tensor

    @classmethod
    def from_visible(cls, values: Tensor, *, epsilon: float, seed: int, key: str):
        scalar = NumericAnswers.from_visible(values, epsilon=epsilon)
        identity = json.dumps(["v53-affine", seed, key], ensure_ascii=True).encode()
        local_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
        generator = torch.Generator(device="cpu").manual_seed(local_seed)
        # Independent unit vectors, deliberately NOT orthogonalized. The local
        # generator consumes no process RNG and is stable under column reordering.
        basis = torch.randn(2, ANSWER_WIDTH, generator=generator, dtype=torch.float64)
        basis = (basis / torch.linalg.vector_norm(basis, dim=-1, keepdim=True)).to(values.device)
        origin, direction = basis.unbind()
        encoded = origin + scalar.encoded * direction
        finite(encoded, "affine visible encoding")
        return cls(scalar, origin, direction, encoded)

    def encode_targets(self, values: Tensor) -> Tensor:
        """Scorer-only truth encoding with this episode's visible statistics."""
        encoded = self.origin + self.scalar.encode_targets(values) * self.direction
        finite(encoded, "affine truth encoding")
        return encoded

    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "affine prediction")
        if encoded.shape[1] != ANSWER_WIDTH or encoded.device != self.origin.device:
            raise ValueError("affine predictions need 128 coordinates on the codec device")
        z = (encoded.double() - self.origin) @ self.direction
        return self.scalar.decode(z[:, None])


@dataclass(frozen=True)
class V53ColumnFacts:
    rows: Tensor
    answers: AffineNumericAnswers | CategoricalAnswers
    input_coordinates: Tensor
    rank: Tensor | None = None


class AffineValueEncoder(nn.Module):
    """Shared bias-free W_enc; same lift for numeric input and answer."""

    def __init__(self, width: int = 128, epsilon: float = 1e-6):
        super().__init__()
        if type(width) is not int or width < ANSWER_WIDTH:
            raise ValueError("carrier width must be at least 128")
        positive(epsilon, "epsilon")
        self.width, self.epsilon = width, epsilon
        self.projection = nn.Linear(ANSWER_WIDTH, width, bias=False)
        with torch.no_grad():
            q, _ = torch.linalg.qr(torch.randn(width, ANSWER_WIDTH), mode="reduced")
            self.projection.weight.copy_(q / 8)
        self.cell_seed = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.unit_seed = nn.Parameter(torch.randn(width) / math.sqrt(width))
        self.feature_seed = nn.Parameter(torch.randn(width) / math.sqrt(width))

    @torch.no_grad()
    def prepare(self, inputs: RestorationInput) -> tuple[V53ColumnFacts, ...]:
        facts = []
        for a, schema in enumerate(inputs.schema):
            rows = inputs.visible[:, a].nonzero(as_tuple=True)[0]
            values = inputs.values[a][rows]
            rank = None
            if schema.kind == "numeric":
                codec = AffineNumericAnswers.from_visible(
                    values, epsilon=self.epsilon, seed=inputs.code_seed, key=schema.key
                )
            else:
                classes, codes = visible_codes(
                    values, width=ANSWER_WIDTH, seed=inputs.code_seed, key=schema.key
                )
                codec = CategoricalAnswers.from_visible(
                    values, classes, codes, domain_size=schema.domain_size
                )
                if schema.kind == "ordinal":
                    positions = torch.tensor(
                        schema.rank_positions(), dtype=torch.float64, device=values.device
                    )
                    rank = positions[values] / max(schema.domain_size - 1, 1)
            facts.append(V53ColumnFacts(rows, codec, codec.encoded, rank))
        return tuple(facts)

    def forward(self, inputs: RestorationInput, facts: tuple[V53ColumnFacts, ...]) -> Tensor:
        return self.forward_prepared(inputs, prepare_features(inputs, facts))

    def forward_prepared(self, inputs: RestorationInput, layout: EncodingLayout) -> Tensor:
        n, m = inputs.visible.shape
        weight = self.projection.weight
        if inputs.visible.device != weight.device:
            raise ValueError("model and input must share a device")
        h = weight.new_zeros(n + 1, m + 1, self.width)
        h[:n, :m] = torch.where(inputs.query[..., None], self.cell_seed, 0)
        h[:n, m] = self.unit_seed
        h[n, :m] = self.feature_seed
        lifts = []
        for kind, coordinates, fixed_rank in layout.groups:
            lift = coordinates.to(weight)
            if kind == "ordinal":
                lift = lift + fixed_rank.to(weight)[:, None]
            lifts.append(lift)
        h = h.flatten(0, 1).index_copy(
            0, layout.addresses, self.projection(torch.cat(lifts))
        ).reshape(n + 1, m + 1, self.width)
        finite(h, "V5.3 initial carriers")
        if bool((h[layout.active].square().sum(-1) == 0).any()):
            raise FloatingPointError("non-Null input carrier collapsed to zero")
        return h
