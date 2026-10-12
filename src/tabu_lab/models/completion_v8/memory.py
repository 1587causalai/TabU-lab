"""Explicit bounded batches and activation recomputation for wide V8 tables.

The qualified V8/V8.1 experiment overlays used these same two mechanisms.
V8.2 estimates linear summaries instead of quadratic logits for its row blocks.
No receiver/source axis is truncated or treated as an independent context.
"""

import types

import torch
from torch.utils.checkpoint import checkpoint

from ..restoration.backbone import OMAB
from .row_attention import LINEAR_MODES, PresenceLinearOMAB


def _bounded(self, receivers, sources, eligible):
    original = type(self)._batched_with_presence
    if receivers.ndim != 3 or sources.ndim != 3 or (eligible is not None and eligible.ndim != 2):
        return original(self, receivers, sources, eligible)
    itemsize = torch.empty((), dtype=self.accumulation_dtype(receivers)).element_size()
    if isinstance(self, PresenceLinearOMAB):
        width, heads = self.config.width, self.config.heads
        # A conservative workspace estimate; not a total allocator/gradient cap.
        elements = 8 * (receivers.shape[1] + sources.shape[1]) * width
        elements += 4 * heads * (width // heads) ** 2
    else:
        # Preserve the prior qualified overlay's partition for inherited OMABs.
        elements = self.config.heads * receivers.shape[1] * (sources.shape[1] + 1)
    block = max(1, self._bounded_bytes // max(1, elements * itemsize))
    if receivers.shape[0] <= block:
        return original(self, receivers, sources, eligible)
    outputs, presence = [], []
    for start in range(0, receivers.shape[0], block):
        args = (
            receivers[start : start + block],
            sources[start : start + block],
            None if eligible is None else eligible[start : start + block],
        )

        def op(a, b, c):
            return original(self, a, b, c)

        if torch.is_grad_enabled() and self._checkpoint_chunks:
            out, mass = checkpoint(op, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            out, mass = op(*args)
        outputs.append(out)
        presence.append(mass)
    return torch.cat(outputs), torch.cat(presence)


def _recompute(module, method):
    original = getattr(module, method)

    def wrapped(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return checkpoint(original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)

    setattr(module, method, types.MethodType(wrapped, module))


def configure_memory(model, *, attention_mib=128, checkpoint_activations=True):
    """Opt in once; same math, bounded independent batches, full-table context."""
    from .model import V8CompletionModel

    if not isinstance(model, V8CompletionModel):
        raise TypeError("memory policy requires a V8 completion model")
    if type(attention_mib) is not int or attention_mib < 1:
        raise ValueError("attention_mib must be a positive integer")
    if type(checkpoint_activations) is not bool:
        raise ValueError("checkpoint_activations must be boolean")
    if hasattr(model, "execution_memory") or any(
        isinstance(module, OMAB) and "_batched_with_presence" in vars(module)
        for module in model.modules()
    ):
        raise ValueError("memory policy already installed; start from an unwrapped model")
    count = 0
    for module in model.modules():
        if isinstance(module, OMAB):
            module._bounded_bytes = attention_mib * 2**20
            module._checkpoint_chunks = checkpoint_activations
            module._batched_with_presence = types.MethodType(_bounded, module)
            count += 1
    axial = getattr(model.backbone, "axial", model.backbone)
    if checkpoint_activations:
        _recompute(model.encoder, "forward")
        _recompute(model.encoder, "inverse")
        for layer in axial.layers:
            _recompute(layer, "forward")
        for block in getattr(model.backbone, "unit_blocks", []):
            _recompute(block, "forward")
    model.execution_memory = {
        "policy_version": 1,
        "attention_mib": attention_mib,
        "checkpoint_activations": checkpoint_activations,
        "partition_axis": "independent_batch_only",
        "linear_row_workspace": model.config.row_attention in LINEAR_MODES,
        "omab_modules": count,
        "no_axis_truncation": True,
    }
    return dict(model.execution_memory)
