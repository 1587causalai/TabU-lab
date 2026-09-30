"""Declared, truth-independent feature semantics for tabular episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class FeatureKind(StrEnum):
    """The semantic value family declared before any episode is compiled."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class FeatureRole(StrEnum):
    """Declared semantic use; it is independent of per-episode cell roles."""

    PREDICTOR = "predictor"
    RESPONSE = "response"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One immutable feature schema entry.

    ``domain`` is ordered: it defines categorical codes and ordinal order.  A
    non-numeric feature therefore needs both a declared domain and a stable
    ``codebook_id``.  These are source schema, never values inferred from an
    episode target or its truth sidecar.
    """

    name: str
    kind: FeatureKind | str = FeatureKind.NUMERIC
    domain: tuple[str, ...] = ()
    codebook_id: str | None = None
    role: FeatureRole | str = FeatureRole.PREDICTOR

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise ValueError("FeatureSpec.name cannot be empty")
        try:
            kind = FeatureKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown FeatureKind: {self.kind!r}") from exc
        try:
            role = FeatureRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown FeatureRole: {self.role!r}") from exc
        domain = tuple(self.domain)
        if any(not isinstance(item, str) or not item.strip() for item in domain):
            raise ValueError("FeatureSpec.domain entries must be non-empty strings")
        if len(domain) != len(set(domain)):
            raise ValueError("FeatureSpec.domain entries must be unique")

        codebook_id = self.codebook_id
        if codebook_id is not None:
            if not isinstance(codebook_id, str) or not codebook_id.strip():
                raise ValueError("FeatureSpec.codebook_id must be a non-empty string or None")
            codebook_id = codebook_id.strip()

        if kind is FeatureKind.NUMERIC:
            if domain:
                raise ValueError("numeric FeatureSpec.domain must be empty")
            if codebook_id is not None:
                raise ValueError("numeric FeatureSpec cannot declare a categorical codebook")
        else:
            if not domain:
                raise ValueError("categorical and ordinal FeatureSpec.domain cannot be empty")
            if codebook_id is None:
                raise ValueError(
                    "categorical and ordinal FeatureSpec require a declared codebook_id"
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codebook_id", codebook_id)

    @property
    def semantic_role(self) -> FeatureRole:
        """Explicit alias for consumers that spell out semantic role."""

        return self.role


def normalize_feature_specs(
    *,
    width: int,
    feature_specs: Sequence[FeatureSpec],
    feature_names: Sequence[str],
) -> tuple[FeatureSpec, ...]:
    """Normalize the typed schema while preserving the feature-name alias."""

    if feature_specs:
        specs = tuple(feature_specs)
        if not all(isinstance(spec, FeatureSpec) for spec in specs):
            raise TypeError("feature_specs entries must be FeatureSpec instances")
        if len(specs) != width:
            raise ValueError("feature_specs must match the feature dimension")
        names = tuple(feature_names)
        if names and names != tuple(spec.name for spec in specs):
            raise ValueError("feature_names must exactly match feature_specs names")
    else:
        names = tuple(feature_names) or tuple(f"feature-{index}" for index in range(width))
        specs = tuple(FeatureSpec(name=name) for name in names)

    if len(specs) != width:
        raise ValueError("feature_specs must match the feature dimension")
    names = tuple(spec.name for spec in specs)
    if len(set(names)) != len(names):
        raise ValueError("FeatureSpec names must be unique")
    return specs


__all__ = ["FeatureKind", "FeatureRole", "FeatureSpec"]
