"""Column grouping with one host count transfer, preserving within-column order."""

import torch


def column_positions(columns, n_columns):
    counts = torch.bincount(columns, minlength=n_columns).tolist()
    return torch.argsort(columns, stable=True).split(counts)


def padded_stack(tensors, shape):
    """Stack equal shapes directly; otherwise pad only ragged metadata/answers."""
    if all(tuple(t.shape) == tuple(shape) for t in tensors):
        return torch.stack(tensors)
    padded = []
    for tensor in tensors:
        padding = tuple(v for old, new in reversed(list(zip(tensor.shape, shape, strict=True)))
                        for v in (0, new - old))
        padded.append(torch.nn.functional.pad(tensor, padding))
    return torch.stack(padded)
