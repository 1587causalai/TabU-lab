"""V5.3 input geometry, independent of the historical backbone defaults."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..restoration._validation import finite
from ..restoration.backbone import BackboneConfig as HistoricalBackboneConfig

INPUT_PROJECTIONS = {"isometric_qr": 1, "legacy_scaled": 0}


@dataclass(frozen=True)
class BackboneConfig(HistoricalBackboneConfig):
    # V5.3 default: unit-norm presence is at the half-participation point.
    # Smaller thresholds remain explicit experiment variants.
    tau_presence: float = 1.0


class QRIsometry(nn.Module):
    """Positive-diagonal thin QR; the effective weight is always an isometry.

    Optimizers update an unconstrained full-rank coordinate matrix. The public
    weight, including after loading or a plain torch optimizer step, is Q with
    the QR sign convention fixed. No optimizer-specific retraction is needed.
    """

    def forward(self, raw: Tensor) -> Tensor:
        finite(raw, "input projection coordinates")
        # MPS currently has no ``aten::linalg_qr.out`` implementation.  Keep
        # the geometry definition identical by running only this small,
        # deterministic factorization on CPU and copying the differentiable
        # result back to MPS.  ``Tensor.to`` retains the autograd path, so the
        # unconstrained coordinates still receive gradients from the MPS
        # training step.  Other devices stay on their native QR path.
        qr_input = raw.to("cpu") if raw.device.type == "mps" else raw
        q, r = torch.linalg.qr(qr_input, mode="reduced")
        if raw.device.type == "mps":
            q, r = q.to(raw.device), r.to(raw.device)
        diagonal = r.diagonal()
        floor = torch.finfo(raw.dtype).eps * torch.linalg.vector_norm(raw)
        if bool((diagonal.abs() <= floor).any()):
            raise FloatingPointError("input projection coordinates are numerically rank deficient")
        weight = q * diagonal.sign().unsqueeze(0)
        finite(weight, "isometric input projection")
        return weight
