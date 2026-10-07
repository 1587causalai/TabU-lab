"""Column-shared local-linear recovery with learnable support responses.

Every one of the ``N`` Unit centers (``pi = 1/N``) weights the full support
set; the centered weighted moments of all centers give ONE ridge system per
column and round. Unlike the V5.x readout, the responses ``E = phi(c_obs)``
are not detached: the gradient reaches ``phi`` through the fit and through
the local means.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..restoration._dtype import solve_dtype
from ..restoration._validation import finite, matrix, positive
from ..restoration.readout import unit_kernel_logits


@dataclass(frozen=True)
class ColumnRecovery:
    recovered: Tensor  # [R,p] embeddings at the evaluation rows
    slope: Tensor  # B [p,d_R]
    eval_weights: Tensor  # [R,n] center weights of the evaluation rows


def support_weights(centers: Tensor, supports: Tensor, bandwidth: float) -> Tensor:
    """Row-normalised Gaussian Unit weights, via a stable log-softmax."""
    return unit_kernel_logits(centers, supports, bandwidth=bandwidth).log_softmax(-1).exp()


def column_shared_ll(
    units: Tensor,
    support_rows: Tensor,
    responses: Tensor,
    features: Tensor,
    eval_rows: Tensor,
    *,
    ridge: float,
    bandwidth: float,
    center_chunk_size: int = 32,
) -> ColumnRecovery:
    """Fit ``B`` over all centers, then read ``E_bar_r + B (g_r - g_bar_r)``.

    ``units`` [N,d_U] and ``features`` [N,d_R] are indexed by table row;
    ``responses`` [n,p] align with ``support_rows``. Moments are accumulated
    from centered tensors in shifted coordinates, never by subtracting two
    uncentered second moments.
    """
    positive(ridge, "ridge")
    positive(bandwidth, "bandwidth")
    for value, name in (
        (units, "Units"),
        (responses, "support responses"),
        (features, "LL features"),
    ):
        matrix(value, name)
    n = len(support_rows)
    if not n:
        raise ValueError("no-support: column-shared LL needs at least one support")
    if len(responses) != n or len(features) != len(units):
        raise ValueError("support responses and features must align with rows")
    dtype = solve_dtype(units)
    source_units = units[support_rows]
    origin_x = features[support_rows[:1]].to(dtype)
    origin_y = responses[:1].to(dtype)
    x = features[support_rows].to(dtype) - origin_x
    y = responses.to(dtype) - origin_y
    covariance = x.new_zeros(x.shape[1], x.shape[1])
    cross = x.new_zeros(x.shape[1], y.shape[1])
    for centers in units.split(center_chunk_size):
        weights = support_weights(centers, source_units, bandwidth)
        centered_x = x[None] - (weights @ x)[:, None]
        centered_y = y[None] - (weights @ y)[:, None]
        flat_x = centered_x.flatten(0, 1)
        covariance = covariance + flat_x.T @ (weights[..., None] * centered_x).flatten(0, 1)
        cross = cross + flat_x.T @ (weights[..., None] * centered_y).flatten(0, 1)
    covariance, cross = covariance / len(units), cross / len(units)
    system = (covariance + covariance.T) / 2
    system = system + ridge * torch.eye(x.shape[1], dtype=dtype, device=x.device)
    finite(system, "column-shared ridge system")
    finite(cross, "column-shared cross moment")
    factor, info = torch.linalg.cholesky_ex(system)
    if bool((info != 0).any()):
        raise FloatingPointError("solve-failed: column-shared LL Cholesky failed")
    solution = torch.cholesky_solve(cross, factor)  # T = B^T, [d_R,p]
    finite(solution, "column-shared slope")
    eval_weights = support_weights(units[eval_rows], source_units, bandwidth)
    delta = (features[eval_rows].to(dtype) - origin_x) - eval_weights @ x
    recovered = origin_y + eval_weights @ y + delta @ solution
    finite(recovered, "recovered embeddings")
    return ColumnRecovery(recovered, solution.T, eval_weights)


def evaluate_fitted_ll(units, support_rows, responses, features, eval_rows, *, slope, bandwidth):
    """Evaluate an already fitted slope at extra rows; preserves its gradient.

    Used for explicit visible reconstruction. This is an in-sample prediction,
    not a copy of the clamped observed code and not a leave-one-out estimate.
    """
    dtype = slope.dtype
    origin_x = features[support_rows[:1]].to(dtype)
    origin_y = responses[:1].to(dtype)
    x = features[support_rows].to(dtype) - origin_x
    y = responses.to(dtype) - origin_y
    weights = support_weights(units[eval_rows], units[support_rows], bandwidth)
    delta = (features[eval_rows].to(dtype) - origin_x) - weights @ x
    recovered = origin_y + weights @ y + delta @ slope.T
    finite(recovered, "auxiliary recovered embeddings")
    return recovered


__all__ = ["ColumnRecovery", "column_shared_ll", "evaluate_fitted_ll", "support_weights"]
