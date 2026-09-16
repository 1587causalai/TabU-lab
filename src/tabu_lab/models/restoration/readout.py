"""FP64 reference for the unified encoded NW/LL table-restoration terminal.

The matching kernel can be shared across columns. Each column selects all of its
visible supports before normalization. Truth, raw-table masks, input encoding,
backbone execution, and loss aggregation are outside this component's interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ._dtype import solve_dtype
from ._validation import finite, matrix, positive


def unit_kernel_logits(
    target_units: Tensor, source_units: Tensor, *, bandwidth: float, query_chunk_size: int = 32
) -> Tensor:
    """Return [requested row, source row] Gaussian logits, reusable across columns.

    Same-column Feature terms cancel analytically. Subtract each pair before
    scaling to preserve nearby distances in tables with a large coordinate span.
    Chunking bounds each temporary; autograd can retain all chunks until backward.
    This reference computes distances in FP64 and preserves both gradient paths.
    """
    positive(bandwidth, "match bandwidth")
    if type(query_chunk_size) is not int or query_chunk_size < 1:
        raise ValueError("query_chunk_size must be a positive integer")
    matrix(target_units, "target Units")
    matrix(source_units, "source Units")
    if target_units.shape[1] != source_units.shape[1] or not target_units.shape[1]:
        raise ValueError("target and source Units must share a nonzero width")
    if target_units.device != source_units.device:
        raise ValueError("target and source Units must share a device")
    dtype = solve_dtype(target_units)
    targets = target_units.to(dtype)
    sources = source_units.to(dtype)
    if not len(targets):
        return targets[:, :1] @ sources[:, :1].T
    blocks = []
    for chunk in targets.split(query_chunk_size):
        scaled_difference = (chunk[:, None, :] - sources[None, :, :]) / bandwidth
        blocks.append(-scaled_difference.square().sum(-1) / 2)
    # One stage check after concatenation; a nonfinite chunk cannot hide.
    logits = torch.cat(blocks, dim=0)
    finite(logits, "Unit kernel logits")
    return logits


@dataclass(frozen=True)
class EncodedRestoration:
    status: Literal["ok", "no-support"]
    support_count: int
    encoding: Tensor | None = None
    log_weights: Tensor | None = None
    coefficients: Tensor | None = None  # LL equivalent coefficients can be negative


@dataclass(frozen=True)
class RestorationReadout:
    """One explicit mode for all answer types; no learned head or type routing.

    ``ridge`` is fixed and recorded even for NW, where it is inactive. The caller
    supplies already selected visible answer encodings in ``support_rows`` order.
    Cell tensors are optional and ignored in NW, required by LL for every type.
    Small widths are allowed for algebra tests; the future encoder requires d>=128.
    """

    mode: Literal["nw", "ll"]
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        if self.mode not in ("nw", "ll"):
            raise ValueError("readout mode must be nw or ll")
        positive(self.ridge, "LL ridge")

    def __call__(
        self,
        shared_logits: Tensor,
        support_rows: Tensor,
        answers: Tensor,
        *,
        support_cells: Tensor | None = None,
        target_cells: Tensor | None = None,
    ) -> EncodedRestoration:
        matrix(shared_logits, "shared Unit logits")
        matrix(answers, "visible answer encodings")
        if support_rows.ndim != 1 or support_rows.dtype != torch.long:
            raise ValueError("support_rows must be an int64 vector")
        if support_rows.device != shared_logits.device or answers.device != shared_logits.device:
            raise ValueError("readout inputs must share a device")
        if len(support_rows) != len(answers) or not answers.shape[1]:
            raise ValueError("answers must align with visible support addresses")
        if len(support_rows.unique()) != len(support_rows) or bool(
            ((support_rows < 0) | (support_rows >= shared_logits.shape[1])).any()
        ):
            raise ValueError("support addresses must be distinct valid source rows")
        n = len(support_rows)
        if n == 0:
            return EncodedRestoration("no-support", 0)
        log_weights = shared_logits[:, support_rows].to(solve_dtype(shared_logits)).log_softmax(-1)
        finite(log_weights, "normalized geometry logits")
        weights = log_weights.exp()
        coefficients = weights
        if self.mode == "ll":
            if support_cells is None or target_cells is None:
                raise ValueError("LL requires target and support Cell content for every type")
            matrix(support_cells, "support Cells")
            matrix(target_cells, "target Cells")
            if (
                len(support_cells) != n
                or len(target_cells) != len(shared_logits)
                or support_cells.shape[1] != target_cells.shape[1]
                or not target_cells.shape[1]
            ):
                raise ValueError("Cell content must align with targets, supports, and width")
            if support_cells.device != weights.device or target_cells.device != weights.device:
                raise ValueError("Cell content and geometry must share a device")
            support = support_cells.to(solve_dtype(support_cells))
            targets = target_cells.to(solve_dtype(support_cells))
            # An arbitrary common translation is exact in the real contract.
            # Center relative to one actual support before taking weighted sums:
            # identical large coordinates then have exactly zero covariance.
            origin = support[0]
            support = support - origin
            targets = targets - origin
            mean = weights @ support
            centered = support[None, :, :] - mean[:, None, :]
            covariance = centered.transpose(-1, -2) @ (weights[..., None] * centered)
            covariance = (covariance + covariance.transpose(-1, -2)) / 2
            system = covariance + self.ridge * torch.eye(
                support.shape[1], dtype=weights.dtype, device=weights.device
            )
            finite(system, "LL system")
            chol, info = torch.linalg.cholesky_ex(system)
            if bool((info != 0).any()):
                raise FloatingPointError("numerical-failure: LL Cholesky failed; ridge unchanged")
            evaluation = torch.cholesky_solve((targets - mean)[..., None], chol).squeeze(-1)
            # Enforce the exact weighted-annihilation identity structurally:
            # sum_s w_s q_s = 0 holds in exact arithmetic, so project q onto
            # that subspace instead of relying on solve accuracy alone. The
            # projection is a no-op up to rounding in FP64; in FP32 it keeps
            # constant reproduction at softmax precision regardless of the
            # Cholesky solve's conditioning.
            q = (centered * evaluation[:, None, :]).sum(-1)
            q = q - (weights * q).sum(-1, keepdim=True)
            coefficients = weights * (1 + q)
            coefficient_sums = coefficients.sum(-1)
            # FP64 rounding keeps this at 1e-9; FP32 (MPS) rounding is looser.
            tolerance = 1e-9 if coefficients.dtype == torch.float64 else 1e-4
            if not torch.allclose(
                coefficient_sums,
                coefficients.new_ones(len(coefficients)),
                atol=tolerance,
                rtol=tolerance,
            ):
                deviation = (coefficient_sums - 1).abs().max().item()
                raise FloatingPointError(
                    "numerical-failure: LL coefficient sum lost constant reproduction "
                    f"(max |sum-1| = {deviation:.3e})"
                )
        # Answer bytes/statistics/codebook are fixed facts, not learned tensors.
        encoded = coefficients @ answers.detach().to(coefficients.dtype)
        finite(encoded, "restored answer encoding")
        finite(coefficients, "equivalent coefficients")
        return EncodedRestoration("ok", n, encoded, log_weights, coefficients)

    def batched(
        self,
        shared_logits: Tensor,
        support_mask: Tensor,
        target_mask: Tensor,
        answers: Tensor,
        *,
        support_cells: Tensor | None = None,
        target_cells: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Cross-column batched readout over padded stacks; same per-column math.

        ``shared_logits`` is [A, T, S]: Unit-kernel logits for A active columns,
        up to T targets and S supports each. ``support_mask`` [A, S] and
        ``target_mask`` [A, T] mark real entries; every active column must have
        at least one real support (no-support columns are reported by the caller,
        never solved). Padded supports receive -inf logits, hence exactly zero
        weight after the stable softmax — the same values as a per-column call
        with filtered lists. Padded target rows produce well-defined ignored
        rows. ``answers`` is [A, S, P], zero-padded on both trailing dims;
        extra answer coordinates are exactly zero and sliced off by the caller.
        Returns (encoded [A, T, P], log_weights [A, T, S], coefficients [A, T, S]).
        """
        if shared_logits.ndim != 3 or answers.ndim != 3:
            raise ValueError("batched readout expects [A,T,S] logits and [A,S,P] answers")
        if (
            support_mask.shape != shared_logits.shape[::2]
            or target_mask.shape != shared_logits.shape[:2]
            or support_mask.dtype is not torch.bool
            or target_mask.dtype is not torch.bool
        ):
            raise ValueError("batched readout masks have incompatible shape or dtype")
        if answers.shape[:2] != shared_logits.shape[::2] or answers.shape[-1] == 0:
            raise ValueError("batched readout answers have incompatible shape")
        for tensor in (support_mask, target_mask, answers):
            if tensor.device != shared_logits.device:
                raise ValueError("batched readout tensors must share a device")
        if not bool(target_mask.any()):
            empty_dtype = solve_dtype(shared_logits)
            return (
                answers.new_zeros(
                    (*shared_logits.shape[:2], answers.shape[-1]), dtype=empty_dtype
                ),
                shared_logits.new_full(shared_logits.shape, -torch.inf, dtype=empty_dtype),
                shared_logits.new_zeros(shared_logits.shape, dtype=empty_dtype),
            )
        if self.mode == "ll":
            if support_cells is None or target_cells is None:
                raise ValueError("LL requires target and support Cell content for every type")
            if (
                support_cells.ndim != 3
                or target_cells.ndim != 3
                or support_cells.shape[:2] != shared_logits.shape[::2]
                or target_cells.shape[:2] != shared_logits.shape[:2]
                or support_cells.shape[-1] != target_cells.shape[-1]
                or not target_cells.shape[-1]
            ):
                raise ValueError("Cell content must align with targets, supports, and width")
            for tensor in (support_cells, target_cells):
                if tensor.device != shared_logits.device:
                    raise ValueError("Cell content and geometry must share a device")
        # Pack only real target rows before the LL solve. Padding is a storage
        # concern, never an additional regression problem: a padded row has no
        # target Cell and must not create its own weights/covariance/Cholesky.
        real_columns, real_targets = target_mask.nonzero(as_tuple=True)
        compact_logits = shared_logits[real_columns, real_targets].unsqueeze(1)
        compact_support_mask = support_mask[real_columns]
        compact_answers = answers[real_columns]
        if self.mode == "ll":
            compact_support_cells = None if support_cells is None else support_cells[real_columns]
            compact_target_cells = (
                None
                if target_cells is None
                else target_cells[real_columns, real_targets].unsqueeze(1)
            )
        else:
            compact_support_cells = compact_target_cells = None
        compact_encoded, compact_log_weights, compact_coefficients = self._batched_dense(
            compact_logits,
            compact_support_mask,
            torch.ones((len(real_columns), 1), dtype=torch.bool, device=target_mask.device),
            compact_answers,
            support_cells=compact_support_cells,
            target_cells=compact_target_cells,
        )
        n_columns, n_targets = target_mask.shape
        flat_indices = real_columns * n_targets + real_targets
        encoded = compact_encoded.new_zeros(
            (n_columns * n_targets, compact_encoded.shape[-1])
        ).index_copy_(0, flat_indices, compact_encoded[:, 0])
        log_weights = compact_log_weights.new_full(
            (n_columns * n_targets, compact_log_weights.shape[-1]), -torch.inf
        ).index_copy_(0, flat_indices, compact_log_weights[:, 0])
        coefficients = compact_coefficients.new_zeros(
            (n_columns * n_targets, compact_coefficients.shape[-1])
        ).index_copy_(0, flat_indices, compact_coefficients[:, 0])
        return (
            encoded.reshape(n_columns, n_targets, -1),
            log_weights.reshape(n_columns, n_targets, -1),
            coefficients.reshape(n_columns, n_targets, -1),
        )

    def _batched_dense(
        self,
        shared_logits: Tensor,
        support_mask: Tensor,
        target_mask: Tensor,
        answers: Tensor,
        *,
        support_cells: Tensor | None = None,
        target_cells: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Solve a stack whose target rows are all real; padding is support-only."""
        if shared_logits.ndim != 3 or answers.ndim != 3:
            raise ValueError("batched readout expects [A,T,S] logits and [A,S,P] answers")
        if (
            support_mask.shape != shared_logits.shape[::2]
            or target_mask.shape != shared_logits.shape[:2]
            or support_mask.dtype != torch.bool
            or target_mask.dtype != torch.bool
        ):
            raise ValueError("batched masks must be aligned boolean [A,S] and [A,T]")
        if answers.shape[:2] != shared_logits.shape[::2] or not answers.shape[2]:
            raise ValueError("answers must align with the padded support stack")
        if not bool(support_mask.any(-1).all()):
            raise ValueError("every batched column needs at least one real support")
        for tensor in (shared_logits, support_mask, target_mask, answers):
            if tensor.device != shared_logits.device:
                raise ValueError("batched readout inputs must share a device")
        logits = torch.where(
            support_mask[:, None, :], shared_logits.to(solve_dtype(shared_logits)), -torch.inf
        )
        # Unit logits are finite-checked upstream; padded supports are -inf by
        # construction. A nonfinite real weight propagates into the LL system or
        # the restored encoding, whose stage checks below report it explicitly.
        log_weights = logits.log_softmax(-1)
        weights = log_weights.exp()
        coefficients = weights
        if self.mode == "ll":
            if support_cells is None or target_cells is None:
                raise ValueError("LL requires target and support Cell content for every type")
            if (
                support_cells.shape != (*shared_logits.shape[::2], support_cells.shape[-1])
                or target_cells.shape[:2] != shared_logits.shape[:2]
                or support_cells.shape[-1] != target_cells.shape[-1]
                or not target_cells.shape[-1]
            ):
                raise ValueError("Cell content must align with targets, supports, and width")
            for tensor in (support_cells, target_cells):
                if tensor.device != weights.device:
                    raise ValueError("Cell content and geometry must share a device")
            support = support_cells.to(solve_dtype(support_cells))
            targets = target_cells.to(solve_dtype(support_cells))
            # An arbitrary common translation is exact in the real contract.
            # Center relative to one actual support before taking weighted sums:
            # identical large coordinates then have exactly zero covariance.
            # Row zero is a real support in every batched column (checked above).
            origin = support[:, 0]
            support = support - origin[:, None]
            targets = targets - origin[:, None]
            mean = weights @ support
            centered = support[:, None, :, :] - mean[:, :, None, :]
            covariance = centered.transpose(-1, -2) @ (weights[..., None] * centered)
            covariance = (covariance + covariance.transpose(-1, -2)) / 2
            system = covariance + self.ridge * torch.eye(
                support.shape[-1], dtype=weights.dtype, device=weights.device
            )
            finite(system, "LL system")
            chol, info = torch.linalg.cholesky_ex(system)
            if bool((info != 0).any()):
                raise FloatingPointError("numerical-failure: LL Cholesky failed; ridge unchanged")
            evaluation = torch.cholesky_solve((targets - mean)[..., None], chol).squeeze(-1)
            # Same structural projection as the single-column path: enforce
            # sum_s w_s q_s = 0 by construction (see above).
            q = (centered * evaluation[:, :, None, :]).sum(-1)
            q = q - (weights * q).sum(-1, keepdim=True)
            coefficients = weights * (1 + q)
            coefficient_sums = coefficients.sum(-1)[target_mask]
            # Same FP64/FP32 tolerance split as the single-column path.
            tolerance = 1e-9 if coefficients.dtype == torch.float64 else 1e-4
            if not torch.allclose(
                coefficient_sums,
                coefficients.new_ones(int(target_mask.sum())),
                atol=tolerance,
                rtol=tolerance,
            ):
                deviation = (coefficient_sums - 1).abs().max().item()
                raise FloatingPointError(
                    "numerical-failure: LL coefficient sum lost constant reproduction "
                    f"(max |sum-1| = {deviation:.3e})"
                )
        # Answer bytes/statistics/codebook are fixed facts, not learned tensors.
        encoded = coefficients @ answers.detach().to(coefficients.dtype)
        finite(encoded, "restored answer encoding")
        finite(coefficients[target_mask], "equivalent coefficients")
        return encoded, log_weights, coefficients
