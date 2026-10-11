"""Scoped V8 precision: new exploration defaults to FP32 throughout model math."""

import torch

from ..restoration.backbone import OFFN, OMAB, OAttention
from .model import V8CompletionModel


def configure_precision(model: V8CompletionModel, *, mode="fast_fp32", reason=None) -> dict:
    """Configure attention, presence and Gaussian LL without converting weights.

    ``reference`` replays historical device precision. The mixed mode requires
    a recorded reason for an explicitly qualified LL exception. Numeric codec
    fitting and deterministic codebook generation keep their stability policy.
    """
    if not isinstance(model, V8CompletionModel):
        raise TypeError("precision override requires a V8 completion model")
    if mode not in {"fast_fp32", "fp32_attention_fp64_ll", "reference"}:
        raise ValueError("unsupported precision mode")
    if mode != "reference" and any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("fast FP32 modes require FP32 model parameters; no implicit conversion")
    if mode == "fp32_attention_fp64_ll" and not reason:
        raise ValueError("FP64 LL exception requires a concrete reason")
    dtype = None if mode == "reference" else torch.float32
    count = 0
    for module in model.modules():
        if isinstance(module, OMAB | OAttention | OFFN):
            module._accumulation_dtype_override = dtype
            count += 1
    model._ll_compute_dtype = torch.float32 if mode == "fast_fp32" else None
    receipt = {
        "policy_version": 2,
        "mode": mode,
        "attention_and_presence": "device_reference" if dtype is None else "float32",
        "gaussian_ll": "float32" if mode == "fast_fp32" else "device_reference",
        "codec": "stable_reference_preprocessing_and_codebook",
        "reason": reason,
        "modules": count,
        "weights_changed": False,
    }
    model.execution_precision = receipt
    return dict(receipt)
