"""Visible-only answer codecs, separate from learned backbone input encoding.

The caller supplies the very same visible identity-code realization used by the
input compiler. Only the scorer calls ``encode_targets`` with truth; construction
and prediction use visible facts alone and never sample another codebook.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import finite, matrix, positive


def _quantile(sorted_values: Tensor, q: float) -> Tensor:
    """Fixed linear-interpolation quantile convention from the design document.

    With 1-based ranks, v = 1 + (n-1)q, j = floor(v), delta = v - j, and
    Q(q) = (1-delta) x_(j) + delta x_(min(j+1, n)).
    """
    n = sorted_values.numel()
    v = 1 + (n - 1) * q
    j = math.floor(v)
    delta = v - j
    lo = sorted_values[j - 1]
    hi = sorted_values[min(j + 1, n) - 1]
    return (1 - delta) * lo + delta * hi


@dataclass(frozen=True)
class NumericAnswers:
    """Median/half-IQR robust coordinates shared by input, answer, score, decode."""

    encoded: Tensor  # [support, 1]
    median: Tensor | None
    scale: Tensor | None

    @staticmethod
    def _batch_parameters(codecs, device):
        if any(c.median is None or c.scale is None for c in codecs):
            raise ValueError("no-support: numeric statistics are undefined")
        if any(c.encoded.device != device for c in codecs):
            raise ValueError("numeric values and visible answers must share a device")
        return torch.stack([c.median for c in codecs]), torch.stack([c.scale for c in codecs])

    @classmethod
    def encode_targets_batch(cls, codecs, values):
        """Scorer-only [column,target] truth encoding, with detached values."""
        if values.ndim != 2 or len(values) != len(codecs) or not values.is_floating_point():
            raise ValueError("numeric target values must have floating [column,target] shape")
        finite(values, "numeric target values")
        median, scale = cls._batch_parameters(codecs, values.device)
        # Codec statistics are finite by construction; the quotient stays finite.
        return ((values.detach().double() - median[:, None]) / scale[:, None])[..., None]

    @classmethod
    def decode_batch(cls, codecs, encoded):
        """Decode [column,target,1] predictions with one batched inverse scaling."""
        if (encoded.ndim != 3 or len(encoded) != len(codecs) or encoded.shape[-1] != 1
                or not encoded.is_floating_point()):
            raise ValueError("numeric predictions need [column,target,1] shape")
        finite(encoded, "predicted encoding")
        median, scale = cls._batch_parameters(codecs, encoded.device)
        # Scalar decode's 0-D statistics follow the prediction tensor dtype.
        median, scale = median.to(encoded), scale.to(encoded)
        # Finite predictions and finite codec statistics keep the decode finite.
        return median[:, None] + scale[:, None] * encoded[..., 0]

    @classmethod
    def from_visible_batch(cls, values: Tensor, *, epsilon: float) -> tuple[NumericAnswers, ...]:
        """Batch columns with equal support counts; preserve the scalar Type-7 codec."""
        positive(epsilon, "epsilon")
        if values.ndim != 2 or not values.is_floating_point():
            raise ValueError("batched numeric answers must have floating [column, support] shape")
        finite(values, "numeric answers")
        values = values.detach().to(torch.float64)
        n = values.shape[1]
        if not n:
            return tuple(cls(row[:, None], None, None) for row in values)
        ordered = values.sort(-1).values

        def quantile(q):
            v = 1 + (n - 1) * q
            j = math.floor(v)
            delta = v - j
            return (1 - delta) * ordered[:, j - 1] + delta * ordered[:, min(j + 1, n) - 1]

        median = quantile(0.5)
        scale = torch.clamp((quantile(0.75) - quantile(0.25)) / 2, min=epsilon)
        # Values are finite (checked above), so the scale is finite; only FP
        # underflow can round it to zero, which is reported explicitly here.
        if not bool((scale > 0).all()):
            raise FloatingPointError("numerical-failure: numeric scale rounded to zero")
        # A finite values/positive-scale quotient is finite; no separate check.
        encoded = ((values - median[:, None]) / scale[:, None])[..., None]
        return tuple(cls(encoded[i], median[i], scale[i]) for i in range(len(values)))

    @classmethod
    def from_visible(cls, values: Tensor, *, epsilon: float) -> NumericAnswers:
        """m = Q(1/2), s = max{(Q(3/4)-Q(1/4))/2, epsilon}; no std fallback."""
        positive(epsilon, "epsilon")
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("visible numeric answers must be a floating vector")
        finite(values, "numeric answers")
        values = values.detach().to(torch.float64)
        if not values.numel():
            return cls(values[:, None], None, None)
        ordered = values.sort().values
        median = _quantile(ordered, 0.5)
        half_iqr = (_quantile(ordered, 0.75) - _quantile(ordered, 0.25)) / 2
        scale = torch.clamp(half_iqr, min=epsilon)
        finite(scale, "numeric scale")
        if not bool(scale > 0):
            raise FloatingPointError("numerical-failure: numeric scale rounded to zero")
        encoded = ((values - median) / scale)[:, None]
        finite(encoded, "numeric answer encoding")
        return cls(encoded, median, scale)

    def encode_targets(self, values: Tensor) -> Tensor:
        """Scorer-only truth encoding using fixed visible statistics, with p=1."""
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("numeric target values must be a floating vector")
        finite(values, "numeric target values")
        if self.median is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        if values.device != self.encoded.device:
            raise ValueError("numeric targets and visible answers must share a device")
        encoded = ((values.detach().to(torch.float64) - self.median) / self.scale)[:, None]
        finite(encoded, "numeric target encoding")
        return encoded

    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "predicted encoding")
        if encoded.shape[1] != 1:
            raise ValueError("numeric predictions require one answer coordinate")
        if self.median is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        result = self.median + self.scale * encoded[:, 0]
        finite(result, "numeric prediction")
        return result


@dataclass(frozen=True)
class CategoricalAnswers:
    encoded: Tensor  # [support, p], unprojected identity codes; p >= 8
    labels: Tensor  # [support], declared-domain indices
    classes: Tensor  # [visible class], schema order, aligned with codebook
    codebook: Tensor  # [visible class, p], eight ones per row
    domain_size: int

    @classmethod
    def from_visible(
        cls, labels: Tensor, classes: Tensor, codebook: Tensor, *, domain_size: int
    ) -> CategoricalAnswers:
        if type(domain_size) is not int or domain_size < 1:
            raise ValueError("domain_size must be a positive declared integer")
        for vector in (labels, classes):
            if vector.ndim != 1 or vector.dtype != torch.long:
                raise ValueError("class identities must be int64 vectors")
            if bool(((vector < 0) | (vector >= domain_size)).any()):
                raise ValueError("class identity outside the declared domain")
        if labels.device != classes.device or codebook.device != labels.device:
            raise ValueError("codes and class identities must share a device")
        matrix(codebook, "visible codebook")
        if codebook.shape[0] != len(classes) or codebook.shape[1] < 8:
            raise ValueError("visible codebook must have one p-vector per class with p >= 8")
        if not torch.equal(classes.sort().values, labels.unique(sorted=True)):
            raise ValueError("codebook classes must equal the visible class set exactly")
        if bool(((codebook != 0) & (codebook != 1)).any()) or bool((codebook.sum(-1) != 8).any()):
            raise ValueError("identity codes must be binary with exactly eight ones")
        if len(classes) and len(codebook.unique(dim=0)) != len(classes):
            raise ValueError("different visible classes must have different identity codes")
        # Declared-domain indices encode schema order, regardless of caller order.
        order = classes.argsort()
        classes = classes.detach()[order].clone()
        codes = codebook.detach().to(torch.float64)[order].clone()
        if len(labels):
            positions = (labels[:, None] == classes[None, :]).long().argmax(-1)
            encoded = codes[positions]
        else:
            encoded = codes.new_empty((0, codes.shape[1]))
        return cls(encoded, labels.detach().clone(), classes.detach().clone(), codes, domain_size)

    def encode_targets(self, labels: Tensor) -> Tensor:
        """Scorer-only lookup; every truth class must have a visible answer code."""
        if labels.ndim != 1 or labels.dtype != torch.long:
            raise ValueError("target class identities must be an int64 vector")
        if labels.device != self.classes.device:
            raise ValueError("target and visible class identities must share a device")
        if not len(self.classes):
            raise ValueError("no-support: categorical targets need a visible codebook")
        matches = labels[:, None] == self.classes[None, :]
        if not bool(matches.any(-1).all()):
            raise ValueError("no-answer-code: truth class has no visible identity code")
        return self.codebook.detach()[matches.long().argmax(-1)]

    @torch.no_grad()
    def decode(self, encoded: Tensor) -> Tensor:
        """Nearest visible class; exact distance ties use schema class order.

        Hard decoding is prediction-only. Training uses the restored encoding
        directly in ``encoding_mse``. Nominal and ordinal share this decoder.
        Equal code norms allow candidate-versus-incumbent dot products. Subtract
        codes before accumulating, without an unrelated shared reference score.

        Vectorized two-pass with exact scan semantics. Pass one scores every
        class directly and takes the first schema-order index attaining the
        maximum. Pass two verifies the winner against every class with the
        pairwise subtraction the reference scan uses: a class whose shared
        coordinates cancel exactly can beat a false direct-score tie, and an
        earlier class tying the winner would keep it under scan order. Rows
        failing verification — or producing nonfinite intermediates — fall back
        to the literal pairwise scan, which is also the explicit nonfinite
        reporting path. One host sync replaces the scan's per-candidate syncs.
        """
        matrix(encoded, "predicted encoding")
        if encoded.shape[1] != self.codebook.shape[1]:
            raise ValueError("categorical predictions must match the answer code width")
        if encoded.device != self.codebook.device:
            raise ValueError("decoder tensors must share a device")
        if not len(self.classes):
            raise ValueError("no-support: categorical decoding needs visible evidence")
        prediction = encoded.to(torch.float64)
        count = len(self.classes)
        order = torch.arange(count, device=prediction.device)
        winner_chunks, fallback_chunks = [], []
        for rows in prediction.split(256):
            scores = rows @ self.codebook.T
            maximum = scores.amax(-1, keepdim=True)
            # First schema-order maximum; out-of-range only on nonfinite rows.
            winner = torch.where(scores == maximum, order, count).amin(-1)
            safe_winner = winner.clamp(max=count - 1)
            # ||e-b||^2 - ||e-best||^2 = -2 e.(b-best), since every code has norm^2=8.
            # Pairwise subtraction cancels shared coordinates exactly, so this
            # resolves class differences that direct full-code products erase.
            difference = self.codebook[None, :, :] - self.codebook[safe_winner][:, None, :]
            advantage = (rows[:, None, :] * difference).sum(-1)
            fallback = (
                (advantage > 0).any(-1)
                | ((advantage >= 0) & (order < safe_winner[:, None])).any(-1)
                | ~torch.isfinite(scores).all(-1)
                | ~torch.isfinite(advantage).all(-1)
            )
            winner_chunks.append(safe_winner)
            fallback_chunks.append(fallback)
        winner = torch.cat(winner_chunks)
        fallback = torch.cat(fallback_chunks)
        if bool(fallback.any()):
            winner = winner.clone()
            for row in fallback.nonzero(as_tuple=False)[:, 0].tolist():
                winner[row] = self._decode_scan(prediction[row : row + 1])[0]
        return self.classes[winner]

    def _decode_scan(self, prediction: Tensor) -> Tensor:
        """Literal candidate-versus-incumbent scan; fallback and test reference.

        A fixed-reference score can still erase differences between two other
        classes. Compare each challenger directly to the current best instead.
        """
        best = torch.zeros(len(prediction), dtype=torch.long, device=prediction.device)
        for candidate in range(1, len(self.classes)):
            difference = self.codebook[candidate] - self.codebook[best]
            advantage = (prediction * difference).sum(-1)
            finite(advantage, "categorical pairwise code comparison")
            best = torch.where(advantage > 0, candidate, best)
        return best
