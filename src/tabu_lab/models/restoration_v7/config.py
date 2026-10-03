"""V7 ModelSpec, with backward-compatible defaults for historical checkpoints.

Current cyclic runs explicitly select donor initialization, Query-as-source and
multiple rounds. Query masking is a separate training policy (``MaskingSpec``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..restoration._validation import positive
from ..restoration.backbone import BackboneConfig


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
class V7Config:
    code_dim: int = 64
    codec: str = "C64/8"
    unit_layers: int = 0  # optional inherited V6 Unit stack
    value_map: str = "coupling"  # identity is an explicit ablation
    query_init: str = "seed"  # "seed" (default) or the donor-code candidate
    query_source: bool = False  # cyclic runs include current Query states as sources
    gradient_checkpointing: bool = False
    backbone: BackboneConfig = field(default_factory=_example_backbone)
    rounds: int = 1  # historical default; current cyclic runs set K explicitly
    share_rounds: bool = True
    coupling_blocks: int = 4
    coupling_hidden: tuple[int, ...] = (128, 128)
    coupling_alpha: float = 0.5
    coupling_scale: bool = True
    bandwidth: float = 1.0
    ridge: float = 1e-3
    epsilon: float = 1e-6
    center_chunk_size: int = 32
    round_loss_rho: float = 0.5
    chi_numeric: float = 128.0
    chi_discrete: float = 1.0

    def __post_init__(self):
        if isinstance(self.backbone, dict):
            object.__setattr__(self, "backbone", BackboneConfig(**self.backbone))
        object.__setattr__(self, "coupling_hidden", tuple(self.coupling_hidden))
        for name in ("code_dim", "rounds", "coupling_blocks", "center_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.code_dim < 2:
            raise ValueError("code_dim must be at least 2")
        if type(self.unit_layers) is not int or self.unit_layers < 0:
            raise ValueError("unit_layers must be a nonnegative integer")
        if self.value_map not in ("coupling", "identity"):
            raise ValueError("value_map must be coupling or identity")
        if self.codec == "constant_weight_composition_v2" and self.code_dim != 128:
            raise ValueError("constant_weight_composition_v2 requires code_dim 128")
        if self.codec not in ("G64", "C64/8", "constant_weight_composition_v2"):
            raise ValueError("codec must be G64 or C64/8 or constant_weight_composition_v2")
        if self.query_init not in ("seed", "donor"):
            raise ValueError("query_init must be seed or donor")
        for name in ("query_source", "gradient_checkpointing", "share_rounds"):
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

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))


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


__all__ = ["V7Config", "round_weights"]
