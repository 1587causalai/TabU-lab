"""Small, reproducible evaluation runners."""

from .synthetic_fit import (
    SyntheticFitResult,
    SyntheticWorldBatch,
    make_linear_world_batch,
    run_synthetic_fit,
)

__all__ = [
    "SyntheticFitResult",
    "SyntheticWorldBatch",
    "make_linear_world_batch",
    "run_synthetic_fit",
]
