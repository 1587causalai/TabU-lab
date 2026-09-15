"""Visible-only answer codecs, separate from learned backbone input encoding.

The caller supplies the very same visible identity-code realization used by the
input compiler. Only the scorer calls ``encode_targets`` with truth; construction
and prediction use visible facts alone and never sample another codebook.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ._validation import finite, matrix, positive


@dataclass(frozen=True)
class NumericAnswers:
    encoded: Tensor  # [support, 1]
    center: Tensor | None  # visible Type-7 median
    scale: Tensor | None

    @classmethod
    def from_visible(cls, values: Tensor, *, scale_floor: float) -> NumericAnswers:
        """Fix visible-only median/half-IQR coordinates in FP64, without clipping."""
        positive(scale_floor, "scale_floor")
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("visible numeric answers must be a floating vector")
        finite(values, "numeric answers")
        values = values.detach().to(torch.float64)
        if not values.numel():
            return cls(values[:, None], None, None)
        # Type 7: linearly interpolate sorted values at zero-based index (n - 1) q.
        q25, center, q75 = torch.quantile(
            values, values.new_tensor([0.25, 0.5, 0.75]), interpolation="linear"
        )
        centered = values - center
        scale = ((q75 - q25) / 2).clamp_min(scale_floor)
        finite(scale, "numeric scale")
        if not bool(scale > 0):
            raise FloatingPointError("numerical-failure: numeric scale rounded to zero")
        encoded = (centered / scale)[:, None]
        finite(encoded, "numeric answer encoding")
        return cls(encoded, center, scale)

    def encode_targets(self, values: Tensor) -> Tensor:
        """Scorer-only truth encoding using fixed visible statistics, with p=1."""
        if values.ndim != 1 or not values.is_floating_point():
            raise ValueError("numeric target values must be a floating vector")
        finite(values, "numeric target values")
        if self.center is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        if values.device != self.encoded.device:
            raise ValueError("numeric targets and visible answers must share a device")
        encoded = ((values.detach().to(torch.float64) - self.center) / self.scale)[:, None]
        finite(encoded, "numeric target encoding")
        return encoded

    def decode(self, encoded: Tensor) -> Tensor:
        matrix(encoded, "predicted encoding")
        if encoded.shape[1] != 1:
            raise ValueError("numeric predictions require one answer coordinate")
        if self.center is None or self.scale is None:
            raise ValueError("no-support: numeric statistics are undefined")
        result = self.center + self.scale * encoded[:, 0]
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
        """
        matrix(encoded, "predicted encoding")
        if encoded.shape[1] != self.codebook.shape[1]:
            raise ValueError("categorical predictions must match the answer code width")
        if encoded.device != self.codebook.device:
            raise ValueError("decoder tensors must share a device")
        if not len(self.classes):
            raise ValueError("no-support: categorical decoding needs visible evidence")
        # ||e-b||^2 - ||e-best||^2 = -2 e.(b-best), since every code has norm^2=8.
        # A fixed-reference score can still erase differences between two other
        # classes. Compare each challenger directly to the current best instead.
        prediction = encoded.to(torch.float64)
        best = torch.zeros(len(encoded), dtype=torch.long, device=encoded.device)
        for candidate in range(1, len(self.classes)):
            difference = self.codebook[candidate] - self.codebook[best]
            advantage = (prediction * difference).sum(-1)
            finite(advantage, "categorical pairwise code comparison")
            best = torch.where(advantage > 0, candidate, best)
        return self.classes[best]
