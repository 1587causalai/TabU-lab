"""Device-aware numeric dtype policy for restoration execution."""

from __future__ import annotations

import torch
from torch import Tensor


def solve_dtype(like: Tensor) -> torch.dtype:
    """Use FP32 on Apple MPS, reference FP64 elsewhere."""
    return torch.float32 if like.device.type == "mps" else torch.float64


def execution_dtype(device: str) -> torch.dtype:
    return torch.float32 if device == "mps" else torch.float64


__all__ = ["execution_dtype", "solve_dtype"]
