"""Explicit builder boundary for the TabUBase model anchor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any, TypeVar

from torch import nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .table_cell import TabUCellBaseModel
from .types import ReferenceConfig

_ModelT = TypeVar("_ModelT", bound=nn.Module)


def _build_float32(
    factory: Callable[..., _ModelT],
    /,
    *args: Any,
    **kwargs: Any,
) -> _ModelT:
    """Construct one executable model under the repository dtype policy."""

    return factory(*args, **kwargs).to(dtype=DEFAULT_FLOAT_DTYPE)


def _config_from_kwargs(kwargs: dict[str, Any]) -> ReferenceConfig:
    explicit = kwargs.pop("config", None)
    if explicit is not None:
        if not isinstance(explicit, ReferenceConfig):
            raise TypeError("config must be a ReferenceConfig")
        return explicit
    names = {field.name for field in fields(ReferenceConfig)}
    values = {name: kwargs.pop(name) for name in tuple(kwargs) if name in names}
    return ReferenceConfig(**values)


def build_tabu_cell_base(**kwargs: Any) -> TabUCellBaseModel:
    """Build ``tabu.cell.base@0.2.0`` with an explicit evidence profile."""

    options = dict(kwargs)
    config = _config_from_kwargs(options)
    numeric_terminal = options.pop("numeric_terminal", "local_linear")
    profile = options.pop("profile", None)
    label_broadcast = options.pop("label_broadcast", None)
    label_broadcast_tau = float(options.pop("label_broadcast_tau", 1.0e-6))
    nominal_tokenizer = options.pop("nominal_tokenizer", "episode_random_sphere")
    nominal_codebook_size = int(options.pop("nominal_codebook_size", 100))
    nominal_codebook_seed = int(options.pop("nominal_codebook_seed", 1729))
    variant_ref = options.pop("variant_ref", None)
    label_columns = options.pop("label_columns", None)
    if label_columns is not None and len(tuple(label_columns)) != 1:
        raise ValueError("tabu.cell.base supervised profile requires exactly one label column")
    if options:
        raise TypeError(f"unknown table-cell base builder options: {sorted(options)}")
    return _build_float32(
        TabUCellBaseModel,
        config,
        numeric_terminal=numeric_terminal,
        profile=profile,
        label_broadcast=label_broadcast,
        label_broadcast_tau=label_broadcast_tau,
        nominal_tokenizer=nominal_tokenizer,
        nominal_codebook_size=nominal_codebook_size,
        nominal_codebook_seed=nominal_codebook_seed,
        variant_ref=variant_ref,
    )


class BuilderRegistry:
    """Small duplicate-rejecting extension seam for model builders."""

    def __init__(self, builders: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._builders: dict[str, Callable[..., Any]] = dict(builders or {})

    def register(
        self,
        model_id: str,
        builder: Callable[..., Any],
        *,
        replace: bool = False,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be blank")
        if not callable(builder):
            raise TypeError("builder must be callable")
        if model_id in self._builders and not replace:
            raise ValueError(f"model builder already registered: {model_id}")
        self._builders[model_id] = builder

    def get(self, model_id: str) -> Callable[..., Any]:
        try:
            return self._builders[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown TabU model id: {model_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def build(self, model_id: str, **kwargs: Any) -> Any:
        return self.get(model_id)(**kwargs)


MODEL_BUILDERS = BuilderRegistry({"tabu.cell.base": build_tabu_cell_base})


def register_model_builder(
    model_id: str,
    builder: Callable[..., Any],
    *,
    replace: bool = False,
) -> None:
    MODEL_BUILDERS.register(model_id, builder, replace=replace)


def build_model(model_id: str, **kwargs: Any) -> Any:
    return MODEL_BUILDERS.build(model_id, **kwargs)


def build_from_spec(spec: Any, **kwargs: Any) -> Any:
    if isinstance(spec, Mapping):
        model_id = spec.get("model_id", spec.get("id"))
    else:
        model_id = getattr(spec, "model_id", getattr(spec, "id", None))
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("spec must expose a non-empty model_id")
    return build_model(model_id, **kwargs)


__all__ = [
    "MODEL_BUILDERS",
    "BuilderRegistry",
    "build_from_spec",
    "build_model",
    "build_tabu_cell_base",
    "register_model_builder",
]
