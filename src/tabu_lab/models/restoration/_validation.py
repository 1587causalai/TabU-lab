"""Small checks for the FP64 reference components."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"numerical-failure: nonfinite {name}")


def matrix(value: Tensor, name: str) -> None:
    if value.ndim != 2 or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating matrix")
    finite(value, name)
