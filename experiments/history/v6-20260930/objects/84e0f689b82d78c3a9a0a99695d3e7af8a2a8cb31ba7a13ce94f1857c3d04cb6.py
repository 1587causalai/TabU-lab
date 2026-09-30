"""Independent data-origin and forward-computation state channels."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, IntFlag, StrEnum
from typing import TypeVar

import torch


class OriginState(StrEnum):
    """What was present in the source before an episode was constructed."""

    OBSERVED = "observed"
    ARTIFICIAL_MASK = "artificial_mask"
    QUERY = "query"
    NATURAL_MISSING = "natural_missing"
    STRUCTURAL = "structural"
    INTERVENTION = "intervention"


class ForwardRole(IntFlag):
    """Independent, overlap-capable participation bits for one forward pass."""

    NONE = 0
    RECEIVER = 1
    SOURCE = 2
    TARGET = 4


_ORIGIN_ORDER = (
    OriginState.OBSERVED,
    OriginState.ARTIFICIAL_MASK,
    OriginState.QUERY,
    OriginState.NATURAL_MISSING,
    OriginState.STRUCTURAL,
    OriginState.INTERVENTION,
)
_ORIGIN_TO_CODE = {state: code for code, state in enumerate(_ORIGIN_ORDER)}
_E = TypeVar("_E", bound=Enum)


def origin_code(state: OriginState | str) -> int:
    return _ORIGIN_TO_CODE[OriginState(state)]


def forward_role_code(role: ForwardRole | str) -> int:
    if isinstance(role, str):
        bits = ForwardRole.NONE
        for item in role.lower().replace("+", "|").split("|"):
            name = item.strip()
            if not name or name == "none":
                continue
            try:
                bits |= ForwardRole[name.upper()]
            except KeyError as exc:
                raise ValueError(f"unknown ForwardRole bit: {name!r}") from exc
        role = bits
    return int(ForwardRole(role))


def _encode_grid(
    values: torch.Tensor | Sequence[Sequence[_E | str | int]],
    *,
    order: tuple[_E, ...],
    name: str,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 2:
            raise ValueError(f"{name} must be a rank-2 grid")
        if values.dtype is torch.bool or values.is_floating_point() or values.is_complex():
            raise ValueError(f"{name} tensor must contain integer enum codes")
        raw = values.detach().clone()
        if raw.numel() and (int(raw.min()) < 0 or int(raw.max()) >= len(order)):
            raise ValueError(f"{name} contains an unknown enum code")
        encoded = raw.to(dtype=torch.uint8)
    else:
        lookup = {enum_value: code for code, enum_value in enumerate(order)}
        lookup.update({enum_value.value: code for enum_value, code in lookup.items()})
        rows: list[list[int]] = []
        for row in values:
            encoded_row: list[int] = []
            for value in row:
                if isinstance(value, int):
                    encoded_row.append(value)
                else:
                    try:
                        encoded_row.append(lookup[value])
                    except (KeyError, TypeError) as exc:
                        raise ValueError(f"unknown {name} value: {value!r}") from exc
            rows.append(encoded_row)
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError(f"{name} must be a non-empty rectangular grid")
        raw = torch.tensor(rows, dtype=torch.int64)
        if int(raw.min()) < 0 or int(raw.max()) >= len(order):
            raise ValueError(f"{name} contains an unknown enum code")
        encoded = raw.to(dtype=torch.uint8)
    if encoded.numel() == 0:
        raise ValueError(f"{name} cannot be empty")
    if int(encoded.max()) >= len(order):
        raise ValueError(f"{name} contains an unknown enum code")
    return encoded


def encode_origin_states(
    values: torch.Tensor | Sequence[Sequence[OriginState | str | int]],
) -> torch.Tensor:
    return _encode_grid(values, order=_ORIGIN_ORDER, name="origin_states")


def encode_forward_roles(
    values: torch.Tensor | Sequence[Sequence[ForwardRole | str | int]],
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 2:
            raise ValueError("forward_roles must be a rank-2 grid")
        if values.dtype is torch.bool or values.is_floating_point() or values.is_complex():
            raise ValueError("forward_roles tensor must contain integer bitmasks")
        raw = values.detach().clone()
        allowed = int(ForwardRole.RECEIVER | ForwardRole.SOURCE | ForwardRole.TARGET)
        if raw.numel() and (int(raw.min()) < 0 or bool((raw & ~allowed).any())):
            raise ValueError("forward_roles contains unknown role bits")
        encoded = raw.to(dtype=torch.uint8)
    else:
        rows = [[forward_role_code(value) for value in row] for row in values]
        if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("forward_roles must be a non-empty rectangular grid")
        raw = torch.tensor(rows, dtype=torch.int64)
        allowed = int(ForwardRole.RECEIVER | ForwardRole.SOURCE | ForwardRole.TARGET)
        if int(raw.min()) < 0 or bool((raw & ~allowed).any()):
            raise ValueError("forward_roles contains unknown role bits")
        encoded = raw.to(dtype=torch.uint8)
    if encoded.numel() == 0:
        raise ValueError("forward_roles cannot be empty")
    allowed = int(ForwardRole.RECEIVER | ForwardRole.SOURCE | ForwardRole.TARGET)
    if bool((encoded & ~allowed).any()):
        raise ValueError("forward_roles contains unknown role bits")
    return encoded


def _decode_grid(
    values: torch.Tensor,
    *,
    order: tuple[_E, ...],
    name: str,
) -> tuple[tuple[_E, ...], ...]:
    encoded = _encode_grid(values, order=order, name=name)
    return tuple(
        tuple(order[int(code)] for code in row)
        for row in encoded.tolist()
    )


def decode_origin_states(values: torch.Tensor) -> tuple[tuple[OriginState, ...], ...]:
    return _decode_grid(values, order=_ORIGIN_ORDER, name="origin_states")


def decode_forward_roles(values: torch.Tensor) -> tuple[tuple[ForwardRole, ...], ...]:
    encoded = encode_forward_roles(values)
    return tuple(
        tuple(ForwardRole(int(code)) for code in row)
        for row in encoded.tolist()
    )


def origin_mask(values: torch.Tensor, state: OriginState | str) -> torch.Tensor:
    encoded = encode_origin_states(values)
    return encoded == origin_code(state)


def forward_role_mask(values: torch.Tensor, role: ForwardRole | str) -> torch.Tensor:
    encoded = encode_forward_roles(values)
    requested = forward_role_code(role)
    if requested == 0:
        return encoded == 0
    return (encoded & requested) == requested


def origin_value_mask(values: torch.Tensor) -> torch.Tensor:
    """Cells whose source dataset is allowed to carry a factual value."""

    encoded = encode_origin_states(values)
    return origin_mask(encoded, OriginState.OBSERVED) | origin_mask(
        encoded, OriginState.INTERVENTION
    )


__all__ = [
    "ForwardRole",
    "OriginState",
    "decode_forward_roles",
    "decode_origin_states",
    "encode_forward_roles",
    "encode_origin_states",
    "forward_role_code",
    "forward_role_mask",
    "origin_code",
    "origin_mask",
    "origin_value_mask",
]
