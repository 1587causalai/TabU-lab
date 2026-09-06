"""Reproducible all-row train/test splits and mandatory coverage validation."""

from __future__ import annotations

import math
import random


def full_train_test_split(values, *, seed, test_fraction=0.2, stratified=False):
    if not 0 < test_fraction < 1 or len(values) < 3:
        raise ValueError("invalid full-data split request")
    rng = random.Random(seed)
    count = math.ceil(len(values) * test_fraction)
    if stratified:
        groups = {}
        for index, row in enumerate(values):
            groups.setdefault(row[-1], []).append(index)
        labels = sorted(groups)
        quotas = {label: math.floor(len(groups[label]) * test_fraction) for label in labels}
        remainders = sorted(
            labels, key=lambda label: (-(len(groups[label]) * test_fraction - quotas[label]), label)
        )
        for label in remainders[: count - sum(quotas.values())]:
            quotas[label] += 1
        test = []
        for label in labels:
            indices = groups[label][:]
            rng.shuffle(indices)
            test.extend(indices[: quotas[label]])
    else:
        indices = list(range(len(values)))
        rng.shuffle(indices)
        test = indices[:count]
    held = set(test)
    return dict(train=[i for i in range(len(values)) if i not in held], test=sorted(test))


def validate_full_dataset(dataset, expected_rows):
    """Reject partial pools, duplicates and overlap before model allocation."""
    n = len(dataset["values"])
    if n != expected_rows:
        raise ValueError("dataset row count differs from preregistered full dataset")
    parts = dataset["splits"]
    if set(parts) != {"train", "test"}:
        raise ValueError("full-data protocol requires explicit train/test splits")
    train, test = parts["train"], parts["test"]
    if len(train) < 3 or not test:
        raise ValueError("empty or insufficient train/test split")
    ids = train + test
    if any(type(i) is not int or not 0 <= i < n for i in ids):
        raise ValueError("invalid row ID")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate or overlapping train/test row IDs")
    if set(ids) != set(range(n)):
        raise ValueError("full-data coverage requires every dataset row exactly once")
    return dict(total_rows=n, train_rows=len(train), test_rows=len(test), unused_rows=0)
