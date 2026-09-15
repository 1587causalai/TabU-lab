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
    targets = target_units.to(torch.float64)
    sources = source_units.to(torch.float64)
    if not len(targets):
        return targets[:, :1] @ sources[:, :1].T
    blocks = []
    for chunk in targets.split(query_chunk_size):
        scaled_difference = (chunk[:, None, :] - sources[None, :, :]) / bandwidth
        logits = -scaled_difference.square().sum(-1) / 2
        finite(logits, "Unit kernel logits")
        blocks.append(logits)
    return torch.cat(blocks, dim=0)


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
        log_weights = shared_logits[:, support_rows].to(torch.float64).log_softmax(-1)
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
            support = support_cells.to(torch.float64)
            targets = target_cells.to(torch.float64)
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
            coefficients = weights * (1 + (centered * evaluation[:, None, :]).sum(-1))
            if not torch.allclose(
                coefficients.sum(-1),
                coefficients.new_ones(len(coefficients)),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise FloatingPointError(
                    "numerical-failure: LL coefficient sum lost constant reproduction"
                )
        # Answer bytes/statistics/codebook are fixed facts, not learned tensors.
        encoded = coefficients @ answers.detach().to(torch.float64)
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
            support_mask[:, None, :], shared_logits.to(torch.float64), -torch.inf
        )
        log_weights = logits.log_softmax(-1)
        real = target_mask[:, :, None] & support_mask[:, None, :]
        if not bool(torch.isfinite(log_weights[real]).all()):
            raise FloatingPointError("numerical-failure: nonfinite normalized geometry logits")
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
            support = support_cells.to(torch.float64)
            targets = target_cells.to(torch.float64)
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
            coefficients = weights * (1 + (centered * evaluation[:, :, None, :]).sum(-1))
            if not torch.allclose(
                coefficients.sum(-1)[target_mask],
                coefficients.new_ones(int(target_mask.sum())),
                atol=1e-9,
                rtol=1e-9,
            ):
                raise FloatingPointError(
                    "numerical-failure: LL coefficient sum lost constant reproduction"
                )
        # Answer bytes/statistics/codebook are fixed facts, not learned tensors.
        encoded = coefficients @ answers.detach().to(torch.float64)
        finite(encoded, "restored answer encoding")
        finite(coefficients[target_mask], "equivalent coefficients")
        return encoded, log_weights, coefficients
