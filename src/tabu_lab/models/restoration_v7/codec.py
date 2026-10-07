"""Versioned V7 codec. C64/8 is default; G64 and V6 bases are explicit options.

The G64 and C64/8 families use the same outer formulas: numeric ``q + z b``, nominal
``q + b_k``, ordinal ``q + b_k + r b``. Composed codes are not renormalised.
G64 draws each base as ``g / ||g||``. ``C64/8`` draws each base uniformly from
the raw weight-8 vectors in ``{0,1}^64`` and does not divide by ``sqrt(8)``.
V7.3 defaults to std floor .01 and continuous asinh coordinates;
standard_softlog_v1 remains an explicit whole-column-switch alternative;
legacy preserves historical numeric coordinates, including the exact V6 decoder.
Statistics and the nominal codebook come only from the visible payload of
``RestorationInput``; hidden Query truth never reaches the codec. Ordinal
columns always use their full declared domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch
from torch import Tensor

from ..restoration._dtype import solve_dtype
from ..restoration._validation import finite, positive
from ..restoration.contracts import ColumnSchema, RestorationInput


class V7ProtocolError(ValueError):
    """A declared protocol outcome such as ``no-support`` or ``no-answer-code``."""

    def __init__(self, status: str, detail: str = ""):
        super().__init__(f"{status}: {detail}" if detail else status)
        self.status = status


def unit_directions(count: int, dim: int, generator: torch.Generator) -> Tensor:
    """Independent ``g / ||g||`` directions in FP64 on CPU, without orthogonalisation."""
    draws = torch.randn(count, dim, generator=generator, dtype=torch.float64)
    norms = draws.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(draws).all()) or bool((norms == 0).any()):
        raise FloatingPointError("numerical-failure: degenerate Gaussian base direction")
    return draws / norms


def constant_weight_bank(
    count: int, dim: int, weight: int, generator: torch.Generator
) -> Tensor:
    """Uniform raw ``{0,1}`` vectors of one weight, distinct inside this bank.

    A vector is the first ``weight`` positions of a uniform permutation. Repeats
    inside the bank are rejected and redrawn. Separate calls may overlap,
    including exact equality. The returned tensor is FP64 on CPU.
    """
    if type(count) is not int or count < 0:
        raise ValueError("constant-weight count must be a non-negative integer")
    if type(dim) is not int or type(weight) is not int or not 0 < weight <= dim:
        raise ValueError("constant weight must be an integer in [1, dim]")
    if count > math.comb(dim, weight):
        raise ValueError("requested vectors exceed the distinct constant-weight code capacity")
    vectors = torch.zeros(count, dim, dtype=torch.float64)
    seen: set[tuple[int, ...]] = set()
    for i in range(count):
        while True:
            positions = tuple(
                sorted(torch.randperm(dim, generator=generator)[:weight].tolist())
            )
            if positions not in seen:
                break
        seen.add(positions)
        vectors[i, list(positions)] = 1
    return vectors


@dataclass(frozen=True)
class G64ColumnCodec:
    """Fixed per-column realization; codes for ``categories[i]`` are ``codes[i]``."""

    kind: str
    base: Tensor  # q [p]
    direction: Tensor | None  # b [p]; numeric and ordinal only
    categories: Tensor | None  # [K] declared-domain indices (visible for nominal)
    codes: Tensor | None  # [K,p] full candidate codes
    mean: float = 0.0
    scale: float = 1.0
    domain_size: int | None = None
    rank_positions: Tensor | None = None  # domain index -> declared ordinal rank
    legacy_answers: object | None = None  # exact V6 encoder/decoder, when selected
    softlog: bool = field(default=False, kw_only=True)  # immutable whole-column switch
    numeric_encoding: str = field(default="affine", kw_only=True)

    def __post_init__(self):
        if self.numeric_encoding not in ("affine", "asinh"):
            raise ValueError("unknown numeric_encoding")
        if self.numeric_encoding == "asinh" and (
            self.kind != "numeric" or self.softlog or self.legacy_answers is not None
        ):
            raise ValueError("asinh requires a numeric codec without softlog or legacy answers")
        if self.kind == "numeric" and (
            not math.isfinite(self.mean) or not math.isfinite(self.scale) or self.scale <= 0
        ):
            raise FloatingPointError("numerical-failure: invalid G64 numeric statistics")

    def encode(self, values: Tensor) -> Tensor:
        """Encode legal values; a discrete value without a code is ``no-answer-code``."""
        if self.legacy_answers is not None:
            if not bool(self.encodable(values).all()):
                raise V7ProtocolError("no-answer-code", "category has no visible code")
            return self.legacy_answers.encode_targets(values)
        if self.kind == "numeric":
            if self.numeric_encoding == "asinh":
                # Host FP64 preprocessing is shared by CPU, CUDA, and MPS.
                # The log form handles finite inputs whose standardized
                # difference overflows; its asymptotic error is sub-ULP there.
                work = values.cpu().double()
                raw = (work - self.mean) / self.scale
                magnitude = torch.maximum(work.abs(), work.new_tensor(abs(self.mean))).clamp_min(torch.finfo(torch.float64).tiny)
                difference = work / magnitude - self.mean / magnitude
                stable = difference.sign() * (difference.abs().log() + magnitude.log() - math.log(self.scale) + math.log(2.0))
                z = torch.where(torch.isfinite(raw), raw.asinh(), stable).to(self.base)
                encoded = self.base + z[:, None] * self.direction
                finite(encoded, "G64 asinh numeric codes")
                return encoded
            values = values.to(self.base.dtype)
            z = (values - self.mean) / self.scale
            # A finite standardized value may have an unrepresentable raw
            # difference, e.g. supports near both ends of the FP64 range.
            if not bool(torch.isfinite(z).all()):
                z = torch.where(torch.isfinite(z), z, values / self.scale - self.mean / self.scale)
            if self.softlog:
                # Compute preprocessing in host FP64 (also for MPS). The log
                # branch must survive a standardized value beyond float range.
                work = values.cpu().double()
                raw = (work - self.mean) / self.scale
                magnitude = torch.maximum(work.abs(), work.new_tensor(abs(self.mean))).clamp_min(torch.finfo(torch.float64).tiny)
                difference = work / magnitude - self.mean / magnitude
                log_ratio = difference.abs().log() + magnitude.log() - math.log(self.scale)
                stable = difference.sign() * torch.logaddexp(torch.zeros_like(log_ratio), log_ratio)
                normal = raw.sign() * raw.abs().log1p()
                z = torch.where(torch.isfinite(normal), normal, stable).to(self.base)
            encoded = self.base + z[:, None] * self.direction
            finite(encoded, "G64 numeric codes")
            return encoded
        index = self._lookup(values)
        if bool((index < 0).any()):
            raise V7ProtocolError("no-answer-code", "category has no visible code")
        return self.codes[index]

    def encodable(self, values: Tensor) -> Tensor:
        if self.kind == "numeric":
            return torch.ones(values.shape, dtype=torch.bool, device=values.device)
        return self._lookup(values) >= 0

    def _lookup(self, values: Tensor) -> Tensor:
        if values.dtype != torch.long:
            raise ValueError("discrete values must be int64 domain indices")
        table = torch.full((self.domain_size,), -1, dtype=torch.long, device=values.device)
        table[self.categories.to(values.device)] = torch.arange(
            len(self.categories), device=values.device
        )
        return table[values]

    def decode(self, codes: Tensor) -> Tensor:
        """Numeric: subtract ``q`` then project on ``b``; discrete: nearest full code."""
        if self.legacy_answers is not None:
            return self.legacy_answers.decode(codes)
        codes = codes.to(self.base.dtype)
        finite(codes, "G64 decoder input")
        if self.kind == "numeric":
            z = (codes - self.base) @ self.direction / self.direction.square().sum()
            if self.numeric_encoding == "asinh":
                work = z.cpu().double()
                absolute = work.abs()
                delta = work.sinh() * self.scale
                # sinh may overflow before multiplication by a small scale.
                stable = work.sign() * (absolute + math.log(self.scale) - math.log(2.0)).exp() * (-torch.expm1(-2 * absolute))
                delta = torch.where(torch.isfinite(delta), delta, stable)
                decoded = self.mean + delta
                if not bool(torch.isfinite(decoded).all()):
                    magnitude = max(abs(self.mean), self.scale, 1.0)
                    ratio = (absolute + math.log(self.scale) - math.log(magnitude) - math.log(2.0)).exp() * (-torch.expm1(-2 * absolute))
                    scaled = self.mean / magnitude + work.sign() * ratio
                    decoded = torch.where(torch.isfinite(decoded), decoded, scaled * magnitude)
                decoded = decoded.to(self.base)
                finite(decoded, "G64 asinh numeric decoded values")
                return decoded
            if self.softlog:
                work = z.cpu().double()
                delta = work.sign() * work.abs().expm1() * self.scale
                # Rescale inside exp when only expm1 overflows, e.g. scale=.01
                # and a finite raw value near 1e308.
                stable = work.sign() * ((work.abs() + math.log(self.scale)).exp() - self.scale)
                delta = torch.where(torch.isfinite(delta), delta, stable)
                decoded = self.mean + delta
                if not bool(torch.isfinite(decoded).all()):
                    # The displacement can exceed FP64 while its sum with a
                    # large, oppositely signed mean is still representable.
                    # Add in a scaled coordinate before restoring raw units.
                    magnitude = max(abs(self.mean), self.scale, 1.0)
                    ratio = (work.abs() + (math.log(self.scale) - math.log(magnitude))).exp()
                    scaled = self.mean / magnitude + work.sign() * (ratio - self.scale / magnitude)
                    decoded = torch.where(torch.isfinite(decoded), decoded, scaled * magnitude)
                decoded = decoded.to(self.base)
                finite(decoded, "G64 numeric decoded values")
                return decoded
            decoded = self.mean + self.scale * z
            if not bool(torch.isfinite(decoded).all()):
                # Do the cancellation before rescaling when only the raw
                # product overflows; genuinely unrepresentable answers fail.
                decoded = torch.where(
                    torch.isfinite(decoded), decoded, (z + self.mean / self.scale) * self.scale
                )
            finite(decoded, "G64 numeric decoded values")
            return decoded
        # Preserve the historical decoder's arithmetic and failure boundary.
        # The new probability API separately checks its distance domain.
        distances = (codes[:, None, :] - self.codes[None]).square().sum(-1)
        # argmin returns the first minimum: ties break in fixed candidate order.
        return self.categories[distances.argmin(-1)]

    def _class_distances(self, codes: Tensor) -> Tensor:
        if self.kind not in ("nominal", "ordinal"):
            raise ValueError("class probabilities require a nominal or ordinal codec")
        if self.categories is None or self.codes is None or not len(self.categories):
            raise V7ProtocolError("no-support", "empty class candidate domain")
        if codes.ndim != 2 or codes.shape[-1] != self.codes.shape[-1]:
            raise ValueError("class readout expects [N, code_dim] codes")
        codes = codes.to(self.codes)
        finite(codes, "V7 class readout codes")
        distances = (codes[:, None, :] - self.codes[None]).square().sum(-1)
        finite(distances, "V7 class code distances")
        return distances

    def log_probabilities(self, codes: Tensor, *, temperature: float = 1.0) -> Tensor:
        """Return ``[N,K]`` log probabilities in the fixed ``categories`` order.

        Both nominal and ordinal compare their complete candidate codes. This
        readout has no learned parameters. Finite positive temperatures are
        accepted, including subnormal values: subtract the minimum distance
        before division so every row retains at least one finite logit.
        A far candidate may have log probability ``-inf`` after floating-point
        overflow (probability zero); NaN and positive infinity are never valid.
        Scorers must check the selected true-class log probability explicitly.
        """
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("probability temperature must be finite and positive")
        distances = self._class_distances(codes)
        shifted = distances - distances.amin(dim=-1, keepdim=True)
        # A positive Python temperature can round to zero in FP32. Use host
        # FP64 for this rare branch, including when the codec resides on MPS.
        work = shifted
        if temperature < torch.finfo(shifted.dtype).tiny:
            work = shifted.cpu().double()
        result = (-work / temperature).log_softmax(dim=-1)
        result = result.to(dtype=distances.dtype).to(device=distances.device)
        if bool((torch.isnan(result) | torch.isposinf(result)).any()):
            raise FloatingPointError("numerical-failure: invalid V7 class log probabilities")
        return result

    def probabilities(self, codes: Tensor, *, temperature: float = 1.0) -> Tensor:
        """Return ``[N,K]`` probabilities; column ``j`` is ``categories[j]``."""
        return self.log_probabilities(codes, temperature=temperature).exp()


@dataclass(frozen=True)
class G64Codec:
    columns: tuple[G64ColumnCodec, ...]
    dim: int
    family: str = "G64"
    numeric_preprocessing: str = "legacy"

    def encode_observed(self, inputs: RestorationInput) -> Tensor:
        """``[N,M,p]`` codes at visible addresses, exact zero elsewhere."""
        n, m = inputs.visible.shape
        dtype = self.columns[0].base.dtype
        table = torch.zeros(n, m, self.dim, dtype=dtype, device=inputs.visible.device)
        for a, column in enumerate(self.columns):
            rows = inputs.visible[:, a].nonzero(as_tuple=True)[0]
            if len(rows):
                table[rows, a] = column.encode(inputs.values[a][rows])
        return table


def build_g64_codec(inputs: RestorationInput, *, dim: int = 64, epsilon: float = 1e-6, numeric_preprocessing: str = "legacy") -> G64Codec:
    """Sample one G64 realization from ``inputs.code_seed`` using visible payload only."""
    return _build_codec(inputs, family="G64", dim=dim, epsilon=epsilon, numeric_preprocessing=numeric_preprocessing)


def build_c64_codec(inputs: RestorationInput, *, dim: int = 64, epsilon: float = 1e-6, numeric_preprocessing: str = "legacy") -> G64Codec:
    """Sample one raw 64/8 constant-weight realization. ``dim`` must stay 64."""
    if type(dim) is not int or dim != 64:
        raise ValueError("C64/8 requires code dimension 64")
    return _build_codec(inputs, family="C64/8", dim=64, epsilon=epsilon, numeric_preprocessing=numeric_preprocessing)


def build_value_codec(
    inputs: RestorationInput, *, codec: str = "C64/8", dim: int = 64, epsilon: float = 1e-6,
    numeric_preprocessing: str = "legacy"
) -> G64Codec:
    """Dispatch the codec family; new episodes default to C64/8."""
    if codec == "constant_weight_composition_v2":
        if dim != 128:
            raise ValueError("constant_weight_composition_v2 requires code_dim 128")
        return build_v6_codec(inputs, epsilon=epsilon, numeric_preprocessing=numeric_preprocessing)
    if codec == "G64":
        return build_g64_codec(inputs, dim=dim, epsilon=epsilon, numeric_preprocessing=numeric_preprocessing)
    if codec == "C64/8":
        return build_c64_codec(inputs, dim=dim, epsilon=epsilon, numeric_preprocessing=numeric_preprocessing)
    raise ValueError("codec must be G64 or C64/8 or constant_weight_composition_v2")


def _build_codec(
    inputs: RestorationInput, *, family: str, dim: int, epsilon: float, numeric_preprocessing: str
) -> G64Codec:
    if numeric_preprocessing not in ("legacy", "standard_softlog_v1", "standard_asinh_v1"):
        raise ValueError("unknown numeric_preprocessing")
    epsilon = epsilon if numeric_preprocessing == "legacy" else 0.01
    positive(epsilon, "numeric scale floor")
    if type(dim) is not int or dim < 2:
        raise ValueError("code dimension must be an integer of at least 2")
    generator = torch.Generator().manual_seed(inputs.code_seed)
    device = inputs.visible.device
    dtype = solve_dtype(inputs.visible)
    columns = []
    for a, schema in enumerate(inputs.schema):
        visible = inputs.values[a][inputs.visible[:, a]]
        columns.append(
            _column_codec(schema, visible, dim, epsilon, generator, device, dtype, family)
        )
    result = G64Codec(tuple(columns), dim, family, numeric_preprocessing)
    if numeric_preprocessing == "standard_asinh_v1":
        return replace(result, columns=tuple(
            replace(column, numeric_encoding="asinh") if column.kind == "numeric" else column
            for column in result.columns
        ))
    return select_softlog(result, inputs)


def _draw_bases(
    family: str, count: int, dim: int, generator: torch.Generator, *, distinct: bool
) -> Tensor:
    if family == "G64":
        if distinct:
            return _distinct_directions(count, dim, generator)
        return unit_directions(count, dim, generator)
    if family == "C64/8":
        # Category banks pass distinct=True. Origin and direction are separate
        # one-vector draws, so they may repeat a category support.
        return constant_weight_bank(count, dim, 8, generator)
    raise ValueError("codec must be G64 or C64/8")


def _column_codec(
    schema: ColumnSchema,
    visible: Tensor,
    dim: int,
    epsilon: float,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    family: str,
) -> G64ColumnCodec:
    base = _draw_bases(family, 1, dim, generator, distinct=False)[0]
    if schema.kind == "numeric":
        direction = _draw_bases(family, 1, dim, generator, distinct=False)[0]
        mean, scale = 0.0, 1.0
        if len(visible):
            mean, scale = _numeric_statistics(visible, epsilon)
        return G64ColumnCodec(
            "numeric",
            base.to(device, dtype),
            direction.to(device, dtype),
            None,
            None,
            mean,
            scale,
        )
    if schema.kind == "nominal":
        categories = visible.detach().to("cpu").unique(sorted=True)
        direction = None
    else:
        categories = torch.arange(schema.domain_size)
        direction = _draw_bases(family, 1, dim, generator, distinct=False)[0]
    identity = _draw_bases(family, len(categories), dim, generator, distinct=True)
    codes = base + identity
    rank_positions = None
    if schema.kind == "ordinal":
        rank_positions = torch.tensor(schema.rank_positions(), dtype=torch.long)
        denominator = max(schema.domain_size - 1, 1)
        positions = rank_positions.to(torch.float64)
        codes = codes + (positions[categories] / denominator)[:, None] * direction
        direction = direction.to(device, dtype)
        rank_positions = rank_positions.to(device)
    finite(codes, "G64 category codes")
    return G64ColumnCodec(
        schema.kind,
        base.to(device, dtype),
        direction,
        categories.to(device),
        codes.to(device, dtype),
        domain_size=schema.domain_size,
        rank_positions=rank_positions,
    )


def _numeric_statistics(visible: Tensor, epsilon: float) -> tuple[float, float]:
    """Visible FP64 mean and population std without squaring raw extremes."""
    work = visible.detach().cpu().to(torch.float64)
    finite(work, "G64 numeric observations")
    mean = work.mean()
    magnitude = work.abs().amax()
    if not bool(torch.isfinite(mean)):
        # The mean can be representable even when its intermediate sum is not.
        mean = (work / magnitude).mean() * magnitude
    finite(mean, "G64 numeric mean")
    centered = work - mean
    spread = centered.abs().amax()
    if not bool(torch.isfinite(spread)):
        # A centered difference can also overflow although the std is finite.
        scaled = work / magnitude - mean / magnitude
        std = scaled.square().mean().sqrt() * magnitude
    elif float(spread) == 0.0:
        std = work.new_zeros(())
    else:
        std = (centered / spread).square().mean().sqrt() * spread
    finite(std, "G64 numeric population standard deviation")
    return float(mean), max(float(std), epsilon)


def _distinct_directions(count: int, dim: int, generator: torch.Generator) -> Tensor:
    directions = unit_directions(count, dim, generator)
    # Exact repeats have probability zero; redraw only if finite precision collides.
    while count > 1 and len(directions.unique(dim=0)) < count:
        directions = unit_directions(count, dim, generator)
    return directions


__all__ = [
    "G64Codec",
    "G64ColumnCodec",
    "V7ProtocolError",
    "build_c64_codec",
    "build_g64_codec",
    "build_value_codec",
    "constant_weight_bank",
    "unit_directions",
]


def build_v6_codec(inputs: RestorationInput, *, epsilon: float = 1e-6, numeric_preprocessing: str = "legacy") -> G64Codec:
    """Reuse V6 bases; legacy delegates its exact statistics/encoder/decoder.

    The new V7 numeric policy replaces only numeric coordinates in this wrapper;
    it never changes the original V6 codec implementation.
    """
    from ..restoration_v53.answers import CompositionNominalAnswers, CompositionOrdinalAnswers
    from ..restoration_v53.encoding import AffineNumericAnswers
    if numeric_preprocessing not in ("legacy", "standard_softlog_v1"):
        raise ValueError("unknown numeric_preprocessing")
    family = "constant_weight_composition_v2"
    columns = []
    for a, schema in enumerate(inputs.schema):
        values = inputs.values[a][inputs.visible[:, a]]
        common = dict(seed=inputs.code_seed, codec_version=family)
        if schema.kind == "numeric":
            old = AffineNumericAnswers.from_visible(values if numeric_preprocessing == "legacy" else values[:0], epsilon=epsilon,
                key=schema.key, scaling="zscore", **common)
            column = G64ColumnCodec("numeric", old.origin, old.direction, None, None,
                float(old.scalar.mean) if old.scalar.mean is not None else 0.,
                float(old.scalar.scale) if old.scalar.scale is not None else 1.,
                legacy_answers=old if numeric_preprocessing == "legacy" else None)
            if numeric_preprocessing != "legacy":
                mean, scale = _numeric_statistics(values, 0.01) if len(values) else (0.0, 1.0)
                column = replace(column, mean=mean, scale=scale)
        else:
            cls = (
                CompositionNominalAnswers if schema.kind == "nominal" else CompositionOrdinalAnswers
            )
            old = cls.from_visible(values, schema=schema, **common)
            categories = old.classes
            codes = old.codebook if schema.kind == "nominal" else old.codebook[categories]
            ranks = (torch.tensor(schema.rank_positions(), device=values.device)
                     if schema.kind == "ordinal" else None)
            column = G64ColumnCodec(schema.kind, old.origin, getattr(old, "direction", None),
                categories, codes, domain_size=schema.domain_size,
                rank_positions=ranks, legacy_answers=old)
        columns.append(column)
    return select_softlog(G64Codec(tuple(columns), 128, family, numeric_preprocessing), inputs)


def select_softlog(codec: G64Codec, visible_inputs: RestorationInput) -> G64Codec:
    """Keep S statistics/bases fixed; select switches from finite Visible V only.

    Training uses S=V. Support-only inference passes its fitted codec plus the
    complete episode V; callers must re-encode *all* observed cells with the
    returned codec. Query targets have no payload in RestorationInput.
    """
    if codec.numeric_preprocessing != "standard_softlog_v1":
        return codec
    columns = []
    for a, column in enumerate(codec.columns):
        if column.kind == "numeric":
            values = visible_inputs.values[a][visible_inputs.visible[:, a]].detach().cpu().double()
            z = (values - column.mean) / column.scale
            z = torch.where(torch.isfinite(z), z, values / column.scale - column.mean / column.scale)
            column = replace(column, softlog=bool((z.abs() > 1e6).any()))
        columns.append(column)
    return replace(codec, columns=tuple(columns))


def null_constant_inputs(inputs: RestorationInput, numeric_preprocessing: str) -> RestorationInput:
    """Strict constant non-target numeric columns become Null, keeping addresses.

    Every column containing a Query is a target and is protected. No IQR test;
    missing values are already outside Visible. Role masks, never zero codes,
    determine source eligibility.
    """
    if numeric_preprocessing == "legacy":
        return inputs
    visible = inputs.visible.clone()
    for a, schema in enumerate(inputs.schema):
        if schema.kind == "numeric" and not bool(inputs.query[:, a].any()):
            values = inputs.values[a][visible[:, a]]
            if len(values) and bool((values == values[0]).all()):
                visible[:, a] = False
    return RestorationInput(inputs.schema, inputs.values, visible, inputs.query, inputs.code_seed)
