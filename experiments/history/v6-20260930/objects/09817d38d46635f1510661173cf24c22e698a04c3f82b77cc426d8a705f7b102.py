"""Repository-wide floating-point execution policy."""

from __future__ import annotations

import torch

DEFAULT_FLOAT_DTYPE = torch.float32
DEFAULT_FLOAT_DTYPE_NAME = "float32"


__all__ = ["DEFAULT_FLOAT_DTYPE", "DEFAULT_FLOAT_DTYPE_NAME"]
