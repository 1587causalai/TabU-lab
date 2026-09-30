"""Composable observation/missingness mechanisms for synthetic SCM tables."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from tabu_lab.contracts import canonical_hash

SCM_MISSINGNESS_COMPONENT_ID = "tabur.scm-missingness.v1"
SCM_MISSINGNESS_RATE_RANGE = (0.01, 0.35)


class SCMMissingnessFamily(StrEnum):
    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"
    BLOCK = "block"
    CENSORING = "censoring"


SCM_MISSINGNESS_FAMILIES = tuple(SCMMissingnessFamily)
SCM_MISSINGNESS_FAMILY_PROBABILITIES = (0.20, 0.35, 0.25, 0.10, 0.10)


def _seed(root_seed: int, *parts: object) -> int:
    encoded = "|".join((str(root_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


@dataclass(frozen=True, slots=True)
class SCMMissingnessManifest:
    family: SCMMissingnessFamily | str
    rate: float
    mechanism_seed: int
    component_id: str = SCM_MISSINGNESS_COMPONENT_ID

    def __post_init__(self) -> None:
        family = SCMMissingnessFamily(self.family)
        if not 0.0 < float(self.rate) < 1.0:
            raise ValueError("SCM missingness rate must be in (0, 1)")
        if isinstance(self.mechanism_seed, bool) or int(self.mechanism_seed) < 0:
            raise ValueError("SCM missingness mechanism_seed must be non-negative")
        if self.component_id != SCM_MISSINGNESS_COMPONENT_ID:
            raise ValueError("unknown SCM missingness component identity")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "rate", float(self.rate))
        object.__setattr__(self, "mechanism_seed", int(self.mechanism_seed))

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabur.scm-missingness-manifest.v1",
                "component_id": self.component_id,
                "family": self.family.value,
                "rate": self.rate,
                "mechanism_seed": self.mechanism_seed,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "family": self.family.value,
            "rate": self.rate,
            "mechanism_seed": self.mechanism_seed,
            "manifest_hash": self.manifest_hash,
        }


@dataclass(frozen=True, slots=True)
class SCMMissingnessResult:
    raw_values: np.ndarray
    missing_mask: np.ndarray
    manifest: SCMMissingnessManifest

    @property
    def missing_count(self) -> int:
        return int(self.missing_mask.sum())

    @property
    def actual_rate(self) -> float:
        return float(self.missing_mask.mean())


def sample_scm_missingness_manifest(
    *,
    root_seed: int,
    world_id: str,
    partition: str,
    family: SCMMissingnessFamily | str | None = None,
    rate: float | None = None,
) -> SCMMissingnessManifest:
    """Sample one deterministic missingness component manifest for a world."""

    rng = np.random.default_rng(_seed(root_seed, partition, world_id, "missingness-plan"))
    resolved_family = (
        SCMMissingnessFamily(family)
        if family is not None
        else SCMMissingnessFamily(
            str(
                rng.choice(
                    tuple(item.value for item in SCM_MISSINGNESS_FAMILIES),
                    p=SCM_MISSINGNESS_FAMILY_PROBABILITIES,
                )
            )
        )
    )
    if rate is None:
        lower, upper = SCM_MISSINGNESS_RATE_RANGE
        resolved_rate = float(np.exp(rng.uniform(np.log(lower), np.log(upper))))
    else:
        resolved_rate = float(rate)
    return SCMMissingnessManifest(
        family=resolved_family,
        rate=resolved_rate,
        mechanism_seed=_seed(root_seed, partition, world_id, "missingness-mechanism"),
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return centered / np.maximum(scale, 1.0e-8)


def _bernoulli_from_score(
    rng: np.random.Generator, *, score: np.ndarray, rate: float
) -> np.ndarray:
    centered = score - score.mean()
    scale = max(float(centered.std()), 1.0e-8)
    normalized = centered / scale
    lower, upper = -30.0, 30.0
    for _ in range(48):
        intercept = 0.5 * (lower + upper)
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(intercept + normalized, -30.0, 30.0))
        )
        if float(probability.mean()) < rate:
            lower = intercept
        else:
            upper = intercept
    logits = 0.5 * (lower + upper) + normalized
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    return rng.random(score.shape[0]) < probability


def apply_scm_missingness(
    complete_values: np.ndarray,
    manifest: SCMMissingnessManifest,
    *,
    eligible_columns: Sequence[int] | None = None,
    driver_columns: Sequence[int] | None = None,
) -> SCMMissingnessResult:
    """Apply a manifest to a complete table and return raw NaNs plus a mask.

    The input is never mutated.  ``eligible_columns`` controls which columns
    may become missing; ``driver_columns`` controls the observed variables
    available to MAR mechanisms.  MAR protects its sampled driver columns
    from missingness so the mechanism depends only on observed variables.
    """

    values = np.asarray(complete_values, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError("SCM missingness requires a finite, non-empty rank-2 table")
    rows, columns = values.shape
    eligible = tuple(range(columns)) if eligible_columns is None else tuple(eligible_columns)
    drivers = tuple(range(columns)) if driver_columns is None else tuple(driver_columns)
    if not eligible or any(index < 0 or index >= columns for index in eligible):
        raise ValueError("eligible_columns must identify at least one table column")
    if not drivers or any(index < 0 or index >= columns for index in drivers):
        raise ValueError("driver_columns must identify at least one table column")
    rng = np.random.default_rng(manifest.mechanism_seed)
    mask = np.zeros((rows, columns), dtype=bool)
    rate = manifest.rate

    if manifest.family is SCMMissingnessFamily.MCAR:
        mask[:, eligible] = rng.random((rows, len(eligible))) < rate
    elif manifest.family is SCMMissingnessFamily.MAR:
        distinct = tuple(dict.fromkeys(drivers))
        if len(distinct) < 2:
            raise ValueError("MAR missingness requires at least two driver columns")
        protected_count = min(2, len(distinct) - 1)
        protected = tuple(
            int(item)
            for item in np.atleast_1d(
                rng.choice(distinct, size=protected_count, replace=False)
            )
        )
        standardized = _standardize(values[:, protected])
        at_risk = tuple(column for column in eligible if column not in protected)
        for column in at_risk:
            count = min(len(protected), int(rng.integers(1, 4)))
            selected = rng.choice(len(protected), size=count, replace=False)
            weights = rng.normal(size=count)
            score = standardized[:, selected] @ weights
            mask[:, column] = _bernoulli_from_score(rng, score=score, rate=rate)
    elif manifest.family is SCMMissingnessFamily.MNAR:
        standardized = _standardize(values[:, eligible])
        for position, column in enumerate(eligible):
            direction = -1.0 if rng.random() < 0.5 else 1.0
            mask[:, column] = _bernoulli_from_score(
                rng, score=direction * standardized[:, position], rate=rate
            )
    elif manifest.family is SCMMissingnessFamily.BLOCK:
        desired = max(1, round(rate * rows * len(eligible)))
        column_count = min(len(eligible), max(1, math.ceil(desired / rows)))
        selected_columns = rng.choice(eligible, size=column_count, replace=False)
        row_count = min(rows, max(1, math.ceil(desired / column_count)))
        start = int(rng.integers(0, rows - row_count + 1))
        mask[start : start + row_count, selected_columns] = True
    elif manifest.family is SCMMissingnessFamily.CENSORING:
        for column in eligible:
            upper = bool(rng.random() < 0.5)
            threshold = np.quantile(values[:, column], 1.0 - rate if upper else rate)
            mask[:, column] = (
                values[:, column] >= threshold
                if upper
                else values[:, column] <= threshold
            )
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported SCM missingness family: {manifest.family}")

    if not bool(mask[:, eligible].any()):
        mask[int(rng.integers(0, rows)), int(rng.choice(eligible))] = True

    raw_values = values.copy()
    raw_values[mask] = np.nan
    return SCMMissingnessResult(
        raw_values=raw_values,
        missing_mask=mask,
        manifest=manifest,
    )


__all__ = [
    "SCM_MISSINGNESS_COMPONENT_ID",
    "SCM_MISSINGNESS_FAMILIES",
    "SCM_MISSINGNESS_FAMILY_PROBABILITIES",
    "SCM_MISSINGNESS_RATE_RANGE",
    "SCMMissingnessFamily",
    "SCMMissingnessManifest",
    "SCMMissingnessResult",
    "apply_scm_missingness",
    "sample_scm_missingness_manifest",
]
