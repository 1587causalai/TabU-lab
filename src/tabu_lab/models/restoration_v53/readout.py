"""One column-shared slope, fitted over ALL current Unit centers (pi=1/N).

FP64 sufficient-statistic reference. Center chunks bound forward temporaries;
autograd still retains intermediates across chunks. This is not yet a bounded
training-memory implementation or a production throughput claim.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..restoration._validation import finite, matrix, positive
from ..restoration._dtype import solve_dtype
from ..restoration.readout import EncodedRestoration, unit_kernel_logits


class FeatureSlopeProvider(nn.Module):
    """Optional research seam: feature [d_f] -> slope [128,d_R].

    A caller supplies a trainable module, e.g. a linear map followed by reshape.
    This is a different estimator from the default ridge solution. It may leave
    the numeric affine line. Its architecture must be supplied again on reload;
    module parameters are included in the model state_dict.
    """

    def forward(self, feature: Tensor) -> Tensor:
        raise NotImplementedError("provide a Feature-to-slope module explicitly")


def normalized_weights(centers: Tensor, supports: Tensor, bandwidth: float) -> Tensor:
    logits = unit_kernel_logits(centers, supports, bandwidth=bandwidth)
    weights = logits.log_softmax(-1).exp()
    finite(weights, "V5.3 normalized Unit weights")
    return weights


def shared_slope(
    units: Tensor,
    support_rows: Tensor,
    support_cells: Tensor,
    answers: Tensor,
    *,
    ridge: float,
    bandwidth: float,
    center_chunk_size: int,
) -> Tensor:
    """Return B [p,d_R] with a SINGLE Cholesky factorization per column.

    Center each support population BEFORE multiplying, then average its weighted
    covariance and cross moment over all centers. Subtracting two uncentered
    second moments can silently erase small local variance even in FP64.
    Each center chunk creates [b,n,d_R] and [b,n,p] centered tensors, never an
    [N,d_R,d_R] covariance stack. This stable reference costs O(N*n*d_R*(d_R+p));
    autograd can retain the centered tensors across chunks until backward.
    """
    positive(ridge, "ridge")
    positive(bandwidth, "bandwidth")
    if type(center_chunk_size) is not int or center_chunk_size < 1:
        raise ValueError("center_chunk_size must be a positive integer")
    for value, name in ((units, "Units"), (support_cells, "support Cells"),
                        (answers, "visible answers")):
        matrix(value, name)
    if not len(units) or not len(answers):
        raise ValueError("shared slope requires centers and visible supports")
    if (support_rows.dtype != torch.long or support_rows.ndim != 1
            or len(support_rows) != len(answers) or len(answers) != len(support_cells)
            or len(support_rows.unique()) != len(support_rows)
            or bool(((support_rows < 0) | (support_rows >= len(units))).any())):
        raise ValueError("support rows must be distinct, in bounds and aligned")
    if any(t.device != units.device for t in (support_rows, support_cells, answers)):
        raise ValueError("readout tensors must share a device")
    dtype = solve_dtype(units)
    x = support_cells.to(dtype) - support_cells[:1].to(dtype)
    y = answers.detach().to(dtype) - answers[:1].detach().to(dtype)
    source_units = units[support_rows]
    covariance = x.new_zeros(x.shape[1], x.shape[1])
    cross = x.new_zeros(x.shape[1], y.shape[1])
    # The loop ALWAYS traverses all N centers, independent of requested targets.
    for centers in units.split(center_chunk_size):
        weights = normalized_weights(centers, source_units, bandwidth)
        centered_x = x[None] - (weights @ x)[:, None]
        centered_y = y[None] - (weights @ y)[:, None]
        weighted_x = weights[..., None] * centered_x
        weighted_y = weights[..., None] * centered_y
        flat_x = centered_x.flatten(0, 1)
        covariance = covariance + (flat_x.T @ weighted_x.flatten(0, 1)) / len(units)
        cross = cross + (flat_x.T @ weighted_y.flatten(0, 1)) / len(units)
    system = (covariance + covariance.T) / 2
    system = system + ridge * torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    finite(system, "column-shared ridge system")
    finite(cross, "column-shared cross moment")
    solve_system, solve_cross = system, cross
    if system.device.type == "mps":
        solve_system, solve_cross = system.cpu(), cross.cpu()
    factor, info = torch.linalg.cholesky_ex(solve_system)
    if bool((info != 0).any()):
        raise FloatingPointError("numerical-failure: column-shared LL Cholesky failed")
    slope = torch.cholesky_solve(solve_cross, factor).T.to(system.device)
    finite(slope, "column-shared slope")
    return slope


def evaluate_column(
    units: Tensor,
    cells: Tensor,
    support_rows: Tensor,
    answers: Tensor,
    target_rows: Tensor,
    slope: Tensor,
    *,
    bandwidth: float,
    chunk_size: int,
) -> EncodedRestoration:
    """Local means + shared slope adjustment; vector answers remain unprojected."""
    if not len(support_rows):
        return EncodedRestoration("no-support", 0)
    dtype = solve_dtype(units)
    source_cells, source_units = cells[support_rows].to(dtype), units[support_rows]
    # Evaluate in shifted coordinates to avoid subtracting large local means.
    origin = source_cells[:1]
    x = source_cells - origin
    answer_origin = answers[:1].detach().to(dtype)
    y = answers.detach().to(dtype) - answer_origin
    predictions = []
    for rows in target_rows.split(chunk_size):
        weights = normalized_weights(units[rows], source_units, bandwidth)
        delta = (cells[rows].to(dtype) - origin) - weights @ x
        encoded = answer_origin + weights @ y + delta @ slope.T
        finite(encoded, "V5.3 restored encoding")
        predictions.append(encoded)
    return EncodedRestoration("ok", len(support_rows), torch.cat(predictions))
