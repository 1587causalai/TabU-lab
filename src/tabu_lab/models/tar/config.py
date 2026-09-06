"""The explicit TAR realization; no changes to legacy ReferenceConfig defaults."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class TARConfig:
    width: int = 384
    blocks: int = 15
    heads: int = 8
    ff_width: int = 768
    semantic_slots: int = 32
    inducing_slots: int = 128
    inducing_enabled: bool = True
    lambda_feature: float = 1.0
    lambda_unit: float = 1.0
    presence_tau: float = 1.0
    rms_epsilon: float = 1e-6
    encoder_scale_epsilon: float = 1e-6
    terminal_scale_floor: float = 1e-6
    match_bandwidth: float = 1.0
    ll_ridge: float = 0.01
    category_smoothing_per_class: float = 1e-6
    minimum_support: int = 2
    receiver_chunk_rows: int = 256
    collect_column_chunk: int = 4
    source_chunk: int = 2048
    terminal_query_chunk: int = 32
    initialization_seed: int = 20260906

    def __post_init__(self):
        for name in (
            "width",
            "blocks",
            "heads",
            "ff_width",
            "semantic_slots",
            "inducing_slots",
            "minimum_support",
            "receiver_chunk_rows",
            "source_chunk",
            "collect_column_chunk",
            "terminal_query_chunk",
        ):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.width < 4 or self.width % self.heads:
            raise ValueError("width must be >= 4 and divisible by heads")
        if self.minimum_support != 2:
            raise ValueError("the TAR contract requires minimum_support=2")
        if type(self.inducing_enabled) is not bool:
            raise ValueError("inducing_enabled must be boolean")
        if type(self.initialization_seed) is not int or self.initialization_seed < 0:
            raise ValueError("initialization_seed must be a nonnegative integer")
        for name in ("lambda_feature", "lambda_unit"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be finite in [0,1]")
        for name in (
            "presence_tau",
            "rms_epsilon",
            "encoder_scale_epsilon",
            "terminal_scale_floor",
            "match_bandwidth",
            "ll_ridge",
            "category_smoothing_per_class",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def fourier_frequencies(self):
        return self.width // 4

    @property
    def parameter_count(self):
        d, f, k, s, j = (
            self.width,
            self.ff_width,
            self.semantic_slots,
            self.inducing_slots,
            self.fourier_frequencies,
        )
        block = 4 * d * d + 2 * d * f + 6 * d + f
        return (
            self.blocks
            * (
                (3 if self.inducing_enabled else 2) * block
                + (s * d if self.inducing_enabled else 0)
            )
            + j
            + 2 * j * d
            + (3 * k + 3) * d
        )

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_design_defaults(cls, payload):
        names = {f.name for f in fields(cls)}
        values = {k: v for k, v in payload.items() if k in names}
        values.setdefault(
            "initialization_seed", payload.get("proposed_training_rng_seed", 20260906)
        )
        result = cls(**values)
        for k, expected in [
            ("fourier_frequencies", result.fourier_frequencies),
            ("dropout", 0),
            ("qassmax", False),
            ("feature_group_size", 1),
            ("estimators", 1),
        ]:
            if k in payload and payload[k] != expected:
                raise ValueError(f"unsupported design default: {k}")
        return result
