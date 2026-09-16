"""Solve-dtype policy for restoration numerics.

The restoration terminal upcasts logits, answer codecs, and LL systems to
float64 as a numerical reference on CPU/CUDA. Apple MPS physically lacks
float64, so the same computations run in float32 there — the model itself is
float32 on that device — keeping device parity without changing the CPU/CUDA
reference path.
"""

from __future__ import annotations

import torch
from torch import Tensor


def solve_dtype(like: Tensor) -> torch.dtype:
    """Return the working dtype for reference-grade solves on ``like``'s device."""
    return torch.float32 if like.device.type == "mps" else torch.float64


__all__ = ["solve_dtype"]
