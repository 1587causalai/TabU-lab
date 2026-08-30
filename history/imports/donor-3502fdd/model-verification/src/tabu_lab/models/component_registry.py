"""Small explicit component factory used by the TabUBase growth seam."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ComponentRegistry:
    """Typed-by-role registry with duplicate rejection and fail-closed lookup."""

    ROLES = ("tokenizer", "broadcast", "dynamics", "readout", "objective")

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Callable[..., Any]]] = {role: {} for role in self.ROLES}

    def register(self, role: str, name: str, factory: Callable[..., Any], *, replace: bool = False) -> None:
        if role not in self._items:
            raise ValueError(f"unknown component role: {role}")
        if not name.strip() or not callable(factory):
            raise ValueError("component name must be non-empty and factory callable")
        if name in self._items[role] and not replace:
            raise ValueError(f"component already registered: {role}:{name}")
        self._items[role][name] = factory

    def get(self, role: str, name: str) -> Callable[..., Any]:
        if role not in self._items or name not in self._items[role]:
            raise KeyError(f"unknown component: {role}:{name}")
        return self._items[role][name]

    def build(self, role: str, name: str, **kwargs: Any) -> Any:
        return self.get(role, name)(**kwargs)

    def names(self, role: str) -> tuple[str, ...]:
        if role not in self._items:
            raise ValueError(f"unknown component role: {role}")
        return tuple(sorted(self._items[role]))


TABU_CELL_BASE_COMPONENTS = ComponentRegistry()


__all__ = ["ComponentRegistry", "TABU_CELL_BASE_COMPONENTS"]
