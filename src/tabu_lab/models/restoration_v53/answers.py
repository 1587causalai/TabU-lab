"""Fixed z-score and typed Gaussian/constant-weight answer spaces for V5.3."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..restoration._validation import finite, matrix, positive
from ..restoration.contracts import ColumnSchema, validate_values
from .codec_versions import DEFAULT_CODEC_VERSION

ANSWER_WIDTH = 128


def unit_gaussians(identity: list, count: int, device: torch.device) -> Tensor:
    """CPU realization independent of process RNG and traversal order."""
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
    vectors = torch.randn(count, ANSWER_WIDTH, generator=generator, dtype=torch.float64)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    finite(vectors, "unit Gaussian realization")
    return vectors.to(device)


def constant_weight_vectors(
    identity: list, count: int, weight: int, device: torch.device,
) -> Tensor:
    """Uniform raw binary vectors, sampled without replacement using a local RNG."""
    if type(weight) is not int or not 0 < weight <= ANSWER_WIDTH:
        raise ValueError("constant weight must be in [1,128]")
    if type(count) is not int or not 0 <= count <= math.comb(ANSWER_WIDTH, weight):
        raise ValueError("requested vectors exceed constant-weight code capacity")
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=True).encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little"))
    vectors = torch.zeros(count, ANSWER_WIDTH, dtype=torch.float64)
    seen = set()
    for i in range(count):
        while True:
            positions = tuple(sorted(torch.randperm(ANSWER_WIDTH, generator=generator)[:weight]
                                     .tolist()))
            if positions not in seen:
                break
        seen.add(positions)
        vectors[i, list(positions)] = 1
    return vectors.to(device)


@dataclass(frozen=True)
class ZScoreAnswers:
    """Population statistics from the current visible supports, never truth."""

    encoded: Tensor
    mean: Tensor | None
    scale: Tensor | None

    @classmethod
    def from_visible(cls, values: Tensor, *, epsilon: float):
        positive(epsilon, "epsilon")
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("visible numeric answers must be a floating vector")
        finite(values, "numeric answers")
        values = values.detach().double()
        if not len(values):
            return cls(values[:, None], None, None)
        # Scale before reductions, but center in original units whenever that
        # is finite: large offsets should not erase small visible variation.
        mean = (values / len(values)).sum()
        centered = values - mean
        if bool(torch.isfinite(centered).all()):
            magnitude = centered.abs().amax().clamp_min(torch.finfo(torch.float64).tiny)
            std = (centered / magnitude).square().mean().sqrt() * magnitude
        else:
            magnitude = values.abs().amax()
            std = (values / magnitude - mean / magnitude).square().mean().sqrt() * magnitude
        scale = std.clamp_min(epsilon)
        finite(mean, "numeric mean")
        finite(scale, "numeric scale")
        if not bool(scale > 0):
            raise FloatingPointError("numerical-failure: numeric scale rounded to zero")
        # Divide first only when ordinary subtraction would overflow. Keeping
        # centered subtraction normally retains precision for offset columns.
        coordinates = torch.where(
            torch.isfinite(centered), centered / scale, values / scale - mean / scale
        )
        encoded = coordinates[:, None]
        finite(encoded, "numeric answer encoding")
        return cls(encoded, mean, scale)

    def encode_targets(self, values: Tensor) -> Tensor:
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("numeric target values must be a floating vector")
        finite(values, "numeric target values")
        if self.mean is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        if values.device != self.encoded.device:
            raise ValueError("numeric targets and visible answers must share a device")
        values = values.detach().double()
        centered = values - self.mean
        encoded = torch.where(
            torch.isfinite(centered), centered / self.scale,
            values / self.scale - self.mean / self.scale,
        )[:, None]
        finite(encoded, "numeric target encoding")
        return encoded

    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "predicted encoding")
        if encoded.shape[1] != 1 or encoded.device != self.encoded.device:
            raise ValueError("numeric predictions need one coordinate on the codec device")
        if self.mean is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        result = self.mean + self.scale * encoded[:, 0]
        finite(result, "numeric prediction")
        return result


@dataclass(frozen=True)
class GaussianNominalAnswers:
    encoded: Tensor
    labels: Tensor
    classes: Tensor
    codebook: Tensor
    domain_size: int

    @classmethod
    def from_visible(cls, labels: Tensor, *, schema: ColumnSchema, seed: int):
        if schema.kind != "nominal":
            raise ValueError("Gaussian nominal codec requires a nominal schema")
        validate_values(schema, labels)
        classes = labels.unique(sorted=True)
        # A class keeps its realization when another class becomes visible.
        codes = [unit_gaussians(["v53-nominal", seed, schema.key, c], 1, labels.device)[0]
                 for c in classes.cpu().tolist()]
        codebook = (torch.stack(codes) if codes else
                    torch.empty(0, ANSWER_WIDTH, dtype=torch.float64, device=labels.device))
        if len(classes) != len(codebook.unique(dim=0)):
            raise FloatingPointError("numerical-failure: duplicate Gaussian nominal codes")
        positions = torch.searchsorted(classes, labels)
        return cls(codebook[positions], labels.detach().clone(), classes, codebook,
                   schema.domain_size)

    def encode_targets(self, labels: Tensor) -> Tensor:
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("target class identities must be an int64 vector")
        if labels.device != self.classes.device:
            raise ValueError("target and visible class identities must share a device")
        if not len(self.classes):
            raise ValueError("no-support: nominal targets need a visible codebook")
        matches = labels[:, None] == self.classes[None, :]
        if not bool(matches.any(-1).all()):
            raise ValueError("no-answer-code: truth class has no visible identity code")
        return self.codebook.detach()[matches.long().argmax(-1)]

    @torch.no_grad()
    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "nominal prediction")
        if encoded.shape[1] != ANSWER_WIDTH or encoded.device != self.codebook.device:
            raise ValueError("nominal predictions need 128 coordinates on the codec device")
        if not len(self.classes):
            raise ValueError("no-support: nominal decoding needs visible evidence")
        # Equal norms make maximum dot product equivalent to nearest code;
        # argmax's first entry gives declared-domain order on exact ties.
        scores = encoded.double() @ self.codebook.T
        finite(scores, "nominal code comparison")
        return self.classes[scores.argmax(-1)]


@dataclass(frozen=True)
class ConstantWeightNominalAnswers(GaussianNominalAnswers):
    """Raw 128/8 identities; the visible class set fixes the codebook realization."""

    @classmethod
    def from_visible(cls, labels: Tensor, *, schema: ColumnSchema, seed: int):
        if schema.kind != "nominal":
            raise ValueError("constant-weight nominal codec requires a nominal schema")
        validate_values(schema, labels)
        classes = labels.unique(sorted=True)
        codes = constant_weight_vectors(
            ["v53-nominal-constant", seed, schema.key, classes.cpu().tolist()],
            len(classes), 8, labels.device,
        )
        return cls(codes[torch.searchsorted(classes, labels)], labels.detach().clone(),
                   classes, codes, schema.domain_size)


@dataclass(frozen=True)
class IdentityOrdinalAnswers:
    """Full schema identity codebook plus one shared normalized-rank direction."""

    encoded: Tensor
    identities: Tensor  # Domain-index order, including classes absent from supports.
    direction: Tensor
    rank_by_label: Tensor
    classes: Tensor  # Declared rank order determines exact-distance tie breaking.
    codebook: Tensor  # Domain-index order.

    @classmethod
    def from_visible(cls, labels: Tensor, *, schema: ColumnSchema, seed: int,
                     codec_version: str = DEFAULT_CODEC_VERSION):
        if schema.kind != "ordinal":
            raise ValueError("identity-rank codec requires an ordinal schema")
        validate_values(schema, labels)
        positions = torch.tensor(schema.rank_positions(), dtype=torch.float64, device=labels.device)
        ranks = positions / max(schema.domain_size - 1, 1)
        if codec_version == "unit_gaussian_v2":
            identities = torch.stack([
                unit_gaussians(["v53-ordinal-identity", seed, schema.key, c], 1, labels.device)[0]
                for c in range(schema.domain_size)
            ])
            direction = unit_gaussians(
                ["v53-ordinal-direction", seed, schema.key], 1, labels.device,
            )[0]
        elif codec_version == "constant_weight_v1":
            identities = constant_weight_vectors(
                ["v53-ordinal-identity-constant", seed, schema.key],
                schema.domain_size, 4, labels.device,
            )
            # Independent of identities: overlap or equality with one is legal.
            direction = constant_weight_vectors(
                ["v53-ordinal-direction-constant", seed, schema.key], 1, 4, labels.device,
            )[0]
        else:
            raise ValueError("identity-rank ordinal requires a current codec version")
        if len(identities.unique(dim=0)) != schema.domain_size:
            raise FloatingPointError("numerical-failure: duplicate ordinal identity codes")
        codebook = identities + ranks[:, None] * direction
        finite(codebook, "ordinal identity-rank codebook")
        return cls(codebook[labels], identities, direction, ranks, positions.argsort(), codebook)

    def encode_targets(self, labels: Tensor) -> Tensor:
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("ordinal targets must be an int64 vector")
        if labels.device != self.codebook.device:
            raise ValueError("ordinal targets and codec must share a device")
        if bool(((labels < 0) | (labels >= len(self.classes))).any()):
            raise ValueError("ordinal target outside the declared domain")
        if not len(self.encoded):
            raise ValueError("no-support: ordinal targets need visible evidence")
        return self.codebook.detach()[labels]

    @torch.no_grad()
    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "ordinal prediction")
        if encoded.shape[1] != ANSWER_WIDTH or encoded.device != self.codebook.device:
            raise ValueError("ordinal predictions need 128 coordinates on the codec device")
        if not len(self.encoded):
            raise ValueError("no-support: ordinal decoding needs visible evidence")
        candidates = self.codebook[self.classes]
        # Drop only the common ||prediction||^2. Candidate norms are NOT equal.
        distances = candidates.square().sum(-1) - 2 * (encoded.double() @ candidates.T)
        finite(distances, "ordinal nearest-code distances")
        return self.classes[distances.argmin(-1)]


@dataclass(frozen=True)
class AffineOrdinalAnswers:
    """Historical unit_gaussian_v1 shared-origin codec; retained for exact replay."""
    encoded: Tensor
    origin: Tensor
    direction: Tensor
    rank_by_label: Tensor
    classes: Tensor  # In increasing declared-rank order, not domain-index order.

    @classmethod
    def from_visible(cls, labels: Tensor, *, schema: ColumnSchema, seed: int):
        if schema.kind != "ordinal":
            raise ValueError("affine ordinal codec requires an ordinal schema")
        validate_values(schema, labels)
        positions = torch.tensor(schema.rank_positions(), dtype=torch.float64, device=labels.device)
        ranks = positions / max(schema.domain_size - 1, 1)
        origin, direction = unit_gaussians(["v53-ordinal", seed, schema.key], 2, labels.device)
        encoded = origin + ranks[labels, None] * direction
        return cls(encoded, origin, direction, ranks, positions.argsort())

    def encode_targets(self, labels: Tensor) -> Tensor:
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("ordinal targets must be an int64 vector")
        if labels.device != self.origin.device:
            raise ValueError("ordinal targets and codec must share a device")
        if bool(((labels < 0) | (labels >= len(self.classes))).any()):
            raise ValueError("ordinal target outside the declared domain")
        if not len(self.encoded):
            raise ValueError("no-support: ordinal targets need visible evidence")
        return self.origin + self.rank_by_label[labels, None] * self.direction

    @torch.no_grad()
    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "ordinal prediction")
        if encoded.shape[1] != ANSWER_WIDTH or encoded.device != self.origin.device:
            raise ValueError("ordinal predictions need 128 coordinates on the codec device")
        if not len(self.encoded):
            raise ValueError("no-support: ordinal decoding needs visible evidence")
        coordinate = (encoded.double() - self.origin) @ self.direction
        finite(coordinate, "ordinal projected rank")
        ranks = self.rank_by_label[self.classes]
        # right=False chooses the lower rank at an exact midpoint. Outside
        # [0,1] the nearest legal endpoint follows without clipping training.
        midpoints = (ranks[:-1] + ranks[1:]) / 2
        return self.classes[torch.bucketize(coordinate, midpoints, right=False)]
