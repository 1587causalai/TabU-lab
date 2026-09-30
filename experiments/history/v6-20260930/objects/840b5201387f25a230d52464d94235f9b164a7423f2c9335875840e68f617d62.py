"""Per-target answer-code loss; target aggregation belongs to the scorer."""

from __future__ import annotations

import torch
from torch import Tensor

from ._validation import finite, matrix
from ._dtype import solve_dtype


def encoding_mse(predicted: Tensor, truth: Tensor) -> Tensor:
    """Return [target] MSE over p answer coordinates, with detached truth.

    Both inputs are [target, p] in the same fixed answer-code realization.
    Numeric answers use p=1; category answers use raw identity-code coordinates.
    Gradients flow through the prediction and its NW/LL paths, never hard decoding.
    """
    matrix(predicted, "predicted answer encoding")
    matrix(truth, "truth answer encoding")
    if predicted.shape != truth.shape or not predicted.shape[1]:
        raise ValueError("answer encodings must have identical [target, p] shapes with p > 0")
    if predicted.device != truth.device:
        raise ValueError("predicted and truth answer encodings must share a device")
    dtype = solve_dtype(predicted)
    losses = (predicted.to(dtype) - truth.detach().to(dtype)).square().mean(-1)
    finite(losses, "answer encoding MSE")
    return losses
