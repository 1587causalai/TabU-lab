"""Allow-listed verification check registry.

Suite manifests contain opaque check ids only.  They are resolved here through
explicit registration; arbitrary dotted imports are deliberately unsupported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .contracts import VerificationCheckResult

VerificationCheckFn = Callable[[str, Mapping[str, Any]], VerificationCheckResult]
_CHECKS: dict[str, VerificationCheckFn] = {}


class VerificationRegistryError(ValueError):
    """Raised for an unknown or duplicate verification check id."""


def register_check(check_id: str, fn: VerificationCheckFn, *, replace: bool = False) -> None:
    if not check_id or not check_id.strip():
        raise VerificationRegistryError("check_id cannot be blank")
    if check_id in _CHECKS and not replace:
        raise VerificationRegistryError(f"verification check already registered: {check_id}")
    _CHECKS[check_id] = fn


def get_check(check_id: str) -> VerificationCheckFn:
    try:
        return _CHECKS[check_id]
    except KeyError as exc:
        raise VerificationRegistryError(f"unknown verification check_id: {check_id}") from exc


def list_checks() -> tuple[str, ...]:
    return tuple(sorted(_CHECKS))


def run_check(
    check_id: str, contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    return get_check(check_id)(contract_id, context)


__all__ = [
    "VerificationCheckFn",
    "VerificationRegistryError",
    "get_check",
    "list_checks",
    "register_check",
    "run_check",
]
