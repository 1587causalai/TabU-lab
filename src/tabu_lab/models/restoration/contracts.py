"""Typed, visible-only model input; clean truth belongs to the scorer alone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


@dataclass(frozen=True)
class ColumnSchema:
    """Declared column type; ordinal order is explicit, not inferred from labels.

    For ordinal columns, ``order`` maps rank position to declared-domain index:
    ``order[0]`` is the index of the lowest category under the declared total
    order. The default ``None`` declares the identity convention — domain index
    i IS the i-th category — which callers must opt into knowingly. Nominal and
    numeric columns have no order.
    """

    key: str
    kind: Literal["numeric", "nominal", "ordinal"]
    domain_size: int | None = None
    order: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("column key must be a nonempty stable string")
        if self.kind not in ("numeric", "nominal", "ordinal"):
            raise ValueError("unknown column kind")
        if self.kind == "numeric":
            if self.domain_size is not None:
                raise ValueError("numeric columns have no discrete domain")
            if self.order is not None:
                raise ValueError("numeric columns have no declared order")
        elif type(self.domain_size) is not int or self.domain_size < 1:
            raise ValueError("discrete columns require a declared positive domain_size")
        elif self.kind == "nominal" and self.order is not None:
            raise ValueError("nominal columns declare no category order")
        if self.kind == "ordinal" and self.order is not None:
            if (
                not isinstance(self.order, tuple)
                or sorted(self.order) != list(range(self.domain_size))
            ):
                raise ValueError("ordinal order must permute the declared domain indices")

    def rank_positions(self) -> tuple[int, ...] | None:
        """Domain index -> 0-based rank position for ordinal columns, else None."""
        if self.kind != "ordinal":
            return None
        order = tuple(range(self.domain_size)) if self.order is None else self.order
        positions = [0] * self.domain_size
        for position, index in enumerate(order):
            positions[index] = position
        return tuple(positions)


def validate_values(schema: ColumnSchema, values: Tensor):
    if values.ndim != 1:
        raise ValueError("column values must be vectors")
    if schema.kind == "numeric":
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError("numeric values must be finite floating values")
    elif values.dtype != torch.long or bool(((values < 0) | (values >= schema.domain_size)).any()):
        raise ValueError("discrete values must be int64 declared-domain indices")


@dataclass(frozen=True)
class RestorationInput:
    """Only visible payload is retained; Query and Null values are zeroed on entry.

    The caller owns schema (including ordinal order), never inferred from truth.
    Boolean masks encode roles, not truth availability or corruption labels.
    """

    schema: tuple[ColumnSchema, ...]
    values: tuple[Tensor, ...]
    visible: Tensor  # [N,M], fixed source eligibility
    query: Tensor  # [N,M], receiver-only; neither mask means semantic Null
    code_seed: int = 0

    def __post_init__(self):
        if not self.schema or len({s.key for s in self.schema}) != len(self.schema):
            raise ValueError("schema needs distinct stable column keys")
        if type(self.code_seed) is not int:
            raise ValueError("code_seed must be an explicit integer")
        if (
            self.visible.ndim != 2
            or self.visible.dtype != torch.bool
            or self.query.dtype != torch.bool
            or self.visible.shape != self.query.shape
            or self.visible.shape[1] != len(self.schema)
            or len(self.values) != len(self.schema)
            or self.visible.device != self.query.device
            or bool((self.visible & self.query).any())
        ):
            raise ValueError("visible/query must be disjoint aligned boolean masks")
        clean = []
        for a, (schema, values) in enumerate(zip(self.schema, self.values, strict=True)):
            if values.shape != (len(self.visible),) or values.device != self.visible.device:
                raise ValueError("column values must align with mask rows and device")
            validate_values(schema, values[self.visible[:, a]])
            clean.append(torch.where(self.visible[:, a], values.detach(), 0).clone())
        object.__setattr__(self, "values", tuple(clean))
        object.__setattr__(self, "visible", self.visible.detach().clone())
        object.__setattr__(self, "query", self.query.detach().clone())


@dataclass(frozen=True)
class RestorationRequest:
    targets: Tensor  # [T,2], rows and columns; may explicitly request Null cells

    def validate(self, inputs: RestorationInput):
        n, m = inputs.visible.shape
        t = self.targets
        if t.ndim != 2 or t.shape[1] != 2 or t.dtype != torch.long:
            raise ValueError("targets must have int64 shape [T,2]")
        if t.device != inputs.visible.device or bool(((t < 0) | (t >= t.new_tensor([n, m]))).any()):
            raise ValueError("target addresses must be in bounds on the input device")
        if len(t.unique(dim=0)) != len(t):
            raise ValueError("duplicate target addresses")


@dataclass(frozen=True)
class TruthSidecar:
    values: tuple[Tensor, ...]  # original observations, NEVER passed to model.forward
    states: Tensor  # -1 absent, 0 retained G, 1 Query Q, 2 Null Z, 3 corrupted B


def make_episode(
    schema: tuple[ColumnSchema, ...],
    values: tuple[Tensor, ...],
    observed: Tensor,
    query: Tensor,
    *,
    code_seed: int,
    null: Tensor | None = None,
    replacements: dict[tuple[int, int], float | int] | None = None,
) -> tuple[RestorationInput, RestorationRequest, TruthSidecar]:
    """Explicit episode construction, not a hidden random-mask retry policy.

    Main restoration requires a nonempty Q. Passing ``null`` or ``replacements``
    explicitly opts into damage restoration. Replacement values are actual input
    supports; original clean values remain exclusively in the returned sidecar.
    """
    damage = null is not None or replacements is not None
    if observed.ndim != 2 or observed.dtype != torch.bool or not bool(observed.any()):
        raise ValueError("observed must be a nonempty boolean table")
    null = torch.zeros_like(observed) if null is None else null
    for mask in (query, null):
        if (
            mask.dtype != torch.bool
            or mask.shape != observed.shape
            or mask.device != observed.device
            or bool((mask & ~observed).any())
        ):
            raise ValueError("damage masks must be subsets of observed")
    if bool((query & null).any()) or (not damage and not bool(query.any())):
        raise ValueError("Query/Null must be disjoint; main restoration needs nonempty Q")
    # Validate clean observed truth before replacing or hiding any of it.
    clean = RestorationInput(schema, values, observed, torch.zeros_like(observed), code_seed)
    visible = observed & ~query & ~null
    payload = [v.clone() for v in clean.values]
    states = torch.full_like(observed, -1, dtype=torch.long)
    states[observed] = 0
    states[query] = 1
    states[null] = 2
    for (r, a), value in (replacements or {}).items():
        if not (0 <= r < observed.shape[0] and 0 <= a < observed.shape[1]):
            raise ValueError("corruption address out of bounds")
        if not bool(visible[r, a]):
            raise ValueError("corruption must be a visible observed address")
        if schema[a].kind != "numeric" and type(value) is not int:
            raise ValueError("discrete replacement must be an integer domain index")
        payload[a][r] = value
        states[r, a] = 3
    inputs = RestorationInput(schema, tuple(payload), visible, query, code_seed)
    request = RestorationRequest(observed.nonzero())
    return inputs, request, TruthSidecar(clean.values, states)
