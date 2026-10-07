"""Explicit V7 and V7.3 ModelSpecs with historical checkpoint compatibility.

New runs select ``V7Config.for_version()`` or a named version factory.
Query masking is a separate training policy (``MaskingSpec``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import cast

from ..restoration._validation import positive
from ..restoration.backbone import BackboneConfig

# Omission is distinct from every explicit value, including False, 1 and None.
# Resolve this marker before validation; it never enters serialized ModelSpec.
_VERSION_DEFAULT = object()


def _example_backbone() -> BackboneConfig:
    return BackboneConfig(
        width=128,
        layers=4,
        heads=4,
        ff_width=512,
        kind="inducing",
        slots=256,
        tau_presence=1.0,
        reference_mass=1.0,
        norm_eps=1e-6,
    )


@dataclass(frozen=True)
class DualStreamConfig:
    """Row-context reversible value encoder; an explicit experimental branch.

    ``u0 = v0 = C``; per block ``u += A_l(v)`` then ``v += F_l(u)``; ``Z=[u;v]``.
    ``A_l`` is row-local OAttention over the Cells of one sample and ``F_l`` a
    token-wise OFFN, both at width ``code_dim``. Engineering choices of this
    first implementation (not validated research conclusions): independent
    parameters per block, no dropout, and the ``mean`` readback
    ``(u0 + v0) / 2`` after whole-row inversion of the assembled LL state.
    """

    blocks: int = 4
    heads: int = 4
    ff_width: int = 128
    tau_presence: float = 1.0
    reference_mass: float = 1.0
    norm_eps: float = 1e-6
    block_parameters: str = "independent"
    readback: str = "mean"
    assembly: str = "query_ll_whole_row"

    def __post_init__(self):
        for name in ("blocks", "heads", "ff_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"dual_stream.{name} must be a positive integer")
        for name in ("tau_presence", "reference_mass", "norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"dual_stream.{name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"dual_stream.{name} must be finite")
            positive(value, f"dual_stream.{name}")
        if self.block_parameters != "independent":
            raise ValueError("dual_stream.block_parameters must be independent")
        if self.readback != "mean":
            raise ValueError("dual_stream.readback must be mean")
        if self.assembly != "query_ll_whole_row":
            raise ValueError("dual_stream.assembly must be query_ll_whole_row")

    def operator_config(self, code_dim: int) -> BackboneConfig:
        """Width-``code_dim`` operator settings for ``A_l`` and ``F_l``."""
        return BackboneConfig(
            width=code_dim,
            layers=self.blocks,
            heads=self.heads,
            ff_width=self.ff_width,
            kind="direct",
            slots=1,
            tau_presence=self.tau_presence,
            reference_mass=self.reference_mass,
            norm_eps=self.norm_eps,
        )

    def as_dict(self):
        return asdict(self)


VALUE_ENCODERS = ("phi_lift", "row_dual_stream")


def _version_defaults(version: str) -> dict:
    """Versioned semantics; preserve separate architecture choices on initialization."""
    if version == "v7":
        return dict(
            model_version="v7",
            query_init="seed",
            query_source=False,
            rounds=1,
            share_rounds=True,
            coupling_bias=True,
            numeric_preprocessing="legacy",
            unit_source_policy="legacy_cell_sources",
            unit_layers=0,
        )
    if version == "v7.3":
        return dict(
            model_version="v7.3",
            query_init="donor",
            query_source=True,
            rounds=4,
            share_rounds=True,
            coupling_bias=False,
            numeric_preprocessing="standard_asinh_v1",
            unit_source_policy="observed",
            unit_layers=0,
        )
    raise ValueError("model_version must be v7 or v7.3")


@dataclass(frozen=True)
class V7Config:
    # The unqualified constructor is the historical V7 protocol. New fields are
    # keyword-only so the pre-version constructor's positional arguments retain
    # their meanings. Direct v7.3 construction resolves the same defaults as v73().
    model_version: str = field(default="v7", kw_only=True)
    code_dim: int = 64
    codec: str = "C64/8"
    unit_layers: int = 0  # optional inherited V6 Unit stack
    unit_source_policy: str = field(default=cast(str, _VERSION_DEFAULT), kw_only=True)
    value_map: str = "coupling"  # identity is an explicit ablation
    query_init: str = cast(str, _VERSION_DEFAULT)
    query_source: bool = cast(bool, _VERSION_DEFAULT)
    gradient_checkpointing: bool = False
    backbone: BackboneConfig = field(default_factory=_example_backbone)
    rounds: int = cast(int, _VERSION_DEFAULT)
    share_rounds: bool = True
    coupling_blocks: int = 4
    coupling_hidden: tuple[int, ...] = (128, 128)
    coupling_alpha: float = 0.5
    coupling_bias: bool = field(default=cast(bool, _VERSION_DEFAULT), kw_only=True)
    numeric_preprocessing: str = field(default=cast(str, _VERSION_DEFAULT), kw_only=True)
    coupling_scale: bool = True
    bandwidth: float = 1.0
    ridge: float = 1e-3
    epsilon: float = 1e-6
    center_chunk_size: int = 32
    round_loss_rho: float = 0.5
    chi_numeric: float = 128.0
    chi_discrete: float = 1.0
    loss_mode: str = "query_only"
    auxiliary_per_query: float | None = None  # None: all visible cells in each queried column
    auxiliary_min_group_weight: float = 0.05
    balanced_loss_scale: float = 1.0
    # Explicit opt-in. "phi_lift" is the per-Cell phi + W_up path of every
    # historical checkpoint; omitted from as_dict() so its ModelSpec is unchanged.
    value_encoder: str = field(default="phi_lift", kw_only=True)
    dual_stream: DualStreamConfig | None = field(default=None, kw_only=True)

    def __post_init__(self):
        for name, value in _version_defaults(self.model_version).items():
            if getattr(self, name) is _VERSION_DEFAULT:
                object.__setattr__(self, name, value)
        if isinstance(self.backbone, dict):
            object.__setattr__(self, "backbone", BackboneConfig(**self.backbone))
        if isinstance(self.dual_stream, dict):
            object.__setattr__(self, "dual_stream", DualStreamConfig(**self.dual_stream))
        object.__setattr__(self, "coupling_hidden", tuple(self.coupling_hidden))
        for name in ("code_dim", "rounds", "coupling_blocks", "center_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.code_dim < 2:
            raise ValueError("code_dim must be at least 2")
        if type(self.unit_layers) is not int or self.unit_layers < 0:
            raise ValueError("unit_layers must be a nonnegative integer")
        if self.unit_source_policy not in ("observed", "legacy_cell_sources"):
            raise ValueError("unit_source_policy must be observed or legacy_cell_sources")
        if self.value_map not in ("coupling", "identity"):
            raise ValueError("value_map must be coupling or identity")
        if self.codec == "constant_weight_composition_v2" and self.code_dim != 128:
            raise ValueError("constant_weight_composition_v2 requires code_dim 128")
        if self.codec not in ("G64", "C64/8", "constant_weight_composition_v2"):
            raise ValueError("codec must be G64 or C64/8 or constant_weight_composition_v2")
        if self.query_init not in ("seed", "donor"):
            raise ValueError("query_init must be seed or donor")
        if self.numeric_preprocessing not in ("legacy", "standard_softlog_v1", "standard_asinh_v1"):
            raise ValueError("unknown numeric_preprocessing")
        if self.numeric_preprocessing == "standard_asinh_v1" and self.codec == "constant_weight_composition_v2":
            raise ValueError("standard_asinh_v1 requires C64/8 or G64")
        for name in (
            "coupling_bias",
            "coupling_scale",
            "query_source",
            "gradient_checkpointing",
            "share_rounds",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.codec == "C64/8" and self.code_dim != 64:
            raise ValueError("C64/8 requires code_dim 64")
        if self.backbone.width < self.code_dim:
            raise ValueError("token width must be at least code_dim for a column-orthogonal lift")
        if self.backbone.kind != "inducing":
            raise ValueError("V7 uses column inducing collect/read")
        for name in (
            "coupling_alpha",
            "bandwidth",
            "ridge",
            "epsilon",
            "chi_numeric",
            "chi_discrete",
        ):
            positive(getattr(self, name), name)
        if not 0.0 <= self.round_loss_rho <= 1.0:
            raise ValueError("round_loss_rho must lie in [0, 1]")
        if self.loss_mode not in ("query_only", "balanced_reconstruction"):
            raise ValueError("unknown loss_mode")
        for name in ("auxiliary_per_query", "balanced_loss_scale"):
            value = getattr(self, name)
            if name == "auxiliary_per_query" and value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.auxiliary_min_group_weight <= 0.5:
            raise ValueError("auxiliary_min_group_weight must lie in (0, 0.5]")
        self._validate_value_encoder()

    def _validate_value_encoder(self):
        if self.value_encoder not in VALUE_ENCODERS:
            raise ValueError("value_encoder must be phi_lift or row_dual_stream")
        if self.value_encoder == "phi_lift":
            if self.dual_stream is not None:
                raise ValueError("dual_stream settings require value_encoder row_dual_stream")
            return
        if not isinstance(self.dual_stream, DualStreamConfig):
            raise ValueError("row_dual_stream requires an explicit DualStreamConfig")
        if self.code_dim != 64:
            raise ValueError("row_dual_stream requires code_dim 64")
        if self.backbone.width != 2 * self.code_dim:
            # Z=[u;v] enters token dynamics directly: no entry lift exists.
            raise ValueError("row_dual_stream requires backbone width 2 * code_dim")
        if self.code_dim % self.dual_stream.heads:
            raise ValueError("dual_stream.heads must divide code_dim")
        if self.value_map != "coupling":
            # phi does not exist on this branch; its identity ablation is meaningless.
            raise ValueError("row_dual_stream has no phi; value_map must stay at its default")
        if not self.query_source:
            # Row encoding reads current Query estimates; LL responses must
            # depend on them. A fixed-source role contract would contradict that.
            raise ValueError("row_dual_stream requires query_source=True")

    def as_dict(self):
        values = asdict(self)
        if self.value_encoder == "phi_lift":
            # Historical ModelSpec dictionaries and their digests stay identical.
            del values["value_encoder"], values["dual_stream"]
        return values

    @classmethod
    def for_version(cls, version, **overrides):
        """Build a declared version; explicit overrides remain experiment choices."""
        if "model_version" in overrides and overrides["model_version"] != version:
            raise ValueError("model_version override conflicts with the selected version")
        return cls(**(_version_defaults(version) | overrides))

    @classmethod
    def legacy(cls, **overrides):
        """V7 defaults; load recorded checkpoints to preserve their actual variants."""
        return cls.for_version("v7", **overrides)

    @classmethod
    def v73(cls, **overrides):
        """V7.3 donor/Query-source shared K4, zero-preserving value map and no Unit stack."""
        return cls.for_version("v7.3", **overrides)

    @classmethod
    def cyclic(cls, **overrides):
        """Alias for the V7.3 starting point; historical loading is unaffected."""
        return cls.v73(**overrides)

    @classmethod
    def from_dict(cls, values):
        # Deserializing a pre-version checkpoint must reconstruct its old graph.
        values = dict(values)
        version = values.pop("model_version", "v7")
        return cls.for_version(version, **values)


def round_weights(rounds: int, rho: float) -> tuple[float, ...]:
    """``(1-rho) rho^(K-t) / (1-rho^K)``; the endpoints are explicit branches."""
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if rho == 1.0:
        return (1.0 / rounds,) * rounds
    if rho == 0.0:
        return (0.0,) * (rounds - 1) + (1.0,)
    total = 1.0 - rho**rounds
    return tuple((1.0 - rho) * rho ** (rounds - t) / total for t in range(1, rounds + 1))


__all__ = ["VALUE_ENCODERS", "DualStreamConfig", "V7Config", "round_weights"]
