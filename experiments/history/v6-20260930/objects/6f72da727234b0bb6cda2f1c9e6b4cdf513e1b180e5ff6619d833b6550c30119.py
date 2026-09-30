"""Episode-local 8-of-128 candidates; no overlap filtering or learned codebook."""

from __future__ import annotations

import hashlib
import math

import torch


def constant_weight_codebook(classes, *, seed, column_id, unit_norm=False):
    """Return codes in sorted class order without consuming the caller's RNG.

    Only visible classes belong here. Rejection checks exact support duplicates;
    the seed and stable column ID scope each book to its episode and column.
    """
    classes = tuple(classes)
    if classes != tuple(sorted(set(classes))):
        raise ValueError("classes must be sorted and unique")
    if len(classes) > math.comb(128, 8):
        raise ValueError("8-of-128 codebook capacity exceeded")
    key = f"{seed}:{column_id}:constant-weight-128-8"
    digest = hashlib.sha256(key.encode()).digest()
    gen = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    book = torch.zeros(len(classes), 128, dtype=torch.float64)
    used = set()
    for row in range(len(classes)):
        while True:
            support = tuple(sorted(torch.randperm(128, generator=gen)[:8].tolist()))
            if support not in used:
                break
        used.add(support)
        book[row, list(support)] = 1
    return book / math.sqrt(8) if unit_norm else book


def validate_constant_weight(book, *, unit_norm):
    """Reject wrong scales, weights and duplicate codes supplied for replay."""
    active = book != 0
    value = 1 / math.sqrt(8) if unit_norm else 1.0
    expected = active.to(book.dtype) * value
    if (
        not bool(torch.isfinite(book).all())
        or not bool((active.sum(-1) == 8).all())
        or not torch.allclose(book, expected, atol=1e-7, rtol=1e-6)
        or len(torch.unique(active, dim=0)) != len(book)
    ):
        raise ValueError("invalid-input: expected unique 8-of-128 codes at configured scale")
