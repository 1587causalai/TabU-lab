"""Explicit builder boundary for the TabUBase model anchor."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from types import MappingProxyType
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
    if "profile" not in options:
        raise TypeError("tabu.cell.base@0.2.0 requires an explicit profile")
    profile = options.pop("profile")
    label_broadcast = options.pop("label_broadcast", None)
    label_broadcast_tau = float(options.pop("label_broadcast_tau", 1.0e-6))
    component_manifest = options.pop("component_manifest", None)
    component_registry = options.pop("component_registry", None)
    if component_manifest is None:
        if component_registry is not None:
            raise TypeError("component_registry requires an explicit component_manifest")
        numeric_terminal = options.pop("numeric_terminal", "local_linear")
        nominal_tokenizer = options.pop("nominal_tokenizer", "episode_random_sphere")
        nominal_codebook_size = options.pop("nominal_codebook_size", 100)
        nominal_codebook_seed = options.pop("nominal_codebook_seed", 1729)
        if options:
            raise TypeError(f"unknown table-cell base builder options: {sorted(options)}")
        model = _build_float32(
            TabUCellBaseModel,
            config,
            numeric_terminal=numeric_terminal,
            profile=profile,
            label_broadcast=label_broadcast,
            label_broadcast_tau=label_broadcast_tau,
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
        )
    else:
        conflicting = sorted(
            set(options)
            & {
                "numeric_terminal",
                "nominal_tokenizer",
                "nominal_codebook_size",
                "nominal_codebook_seed",
            }
        )
        if conflicting:
            raise TypeError(
                "component_manifest is the only component-selection authority; "
                f"remove {conflicting}"
            )
        if options:
            raise TypeError(f"unknown table-cell base builder options: {sorted(options)}")
        from .component_registry import (
            CANONICAL_COMPONENTS,
            ComponentRegistry,
            ComponentRole,
            TabUBaseComponentManifest,
        )

        if not isinstance(component_manifest, TabUBaseComponentManifest):
            raise TypeError("component_manifest must be a typed TabUBaseComponentManifest")
        registry = CANONICAL_COMPONENTS if component_registry is None else component_registry
        if not isinstance(registry, ComponentRegistry):
            raise TypeError("component_registry must be a ComponentRegistry")
        registry.assert_extends(CANONICAL_COMPONENTS)
        composition = registry.resolve(component_manifest)
        tokenizer = registry.build(
            component_manifest.tokenizer,
            expected_role=ComponentRole.TOKENIZER,
            config=config,
        )
        dynamics = registry.build(
            component_manifest.dynamics,
            expected_role=ComponentRole.DYNAMICS,
            config=config,
        )
        readout = registry.build(
            component_manifest.readout,
            expected_role=ComponentRole.READOUT,
            config=config,
        )
        model = _build_float32(
            TabUCellBaseModel,
            config,
            profile=profile,
            label_broadcast=label_broadcast,
            label_broadcast_tau=label_broadcast_tau,
            _component_tokenizer=tokenizer,
            _component_dynamics=dynamics,
            _component_readout=readout,
            _component_composition=composition,
            _component_registry=registry,
        )
    # The packaged ModelSpec is the semantic authority.  Every public build,
    # including direct ``build_model`` calls, must close the full binding to
    # the concrete runtime composition before it can escape the builder.
    from tabu_lab.registry import get_model_spec

    from .component_contract import resolve_tabu_base_composition

    spec = get_model_spec(model.model_id, model.contract_version)
    resolve_tabu_base_composition(spec, model)
    return model


class BuilderRegistry:
    """Small duplicate-rejecting extension seam for model builders."""

    def __init__(
        self,
        builders: Mapping[str, Callable[..., Any]] | None = None,
        *,
        protected_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._builders: dict[str, Callable[..., Any]] = dict(builders or {})
        self._protected_ids = protected_ids

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
        if model_id in self._protected_ids:
            raise ValueError(f"canonical model builder cannot be replaced: {model_id}")
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


_CANONICAL_MODEL_BUILDERS: Mapping[str, Callable[..., Any]] = MappingProxyType(
    {"tabu.cell.base": build_tabu_cell_base}
)
MODEL_BUILDERS = BuilderRegistry(
    _CANONICAL_MODEL_BUILDERS,
    protected_ids=frozenset(_CANONICAL_MODEL_BUILDERS),
)


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
    """Build only from an exact typed registry contract.

    Accepting a dict or duck-typed object here would let a caller claim one
    version/hash while silently constructing the currently packaged version.
    """

    from tabu_lab.contracts import canonical_hash
    from tabu_lab.registry import ModelSpec, get_model_spec, model_spec_identity_payload

    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a typed ModelSpec")
    registered = get_model_spec(spec.contract_id, spec.contract_version)
    supplied_hash = canonical_hash(model_spec_identity_payload(spec))
    registered_hash = canonical_hash(model_spec_identity_payload(registered))
    if supplied_hash != registered_hash:
        raise ValueError("ModelSpec does not exactly match the registered contract")

    try:
        canonical_builder = _CANONICAL_MODEL_BUILDERS[registered.contract_id]
    except KeyError as exc:
        raise ValueError(f"no immutable canonical builder for {registered.contract_id!r}") from exc
    model = canonical_builder(**kwargs)
    if registered.contract_id == "tabu.cell.base" and not isinstance(model, TabUCellBaseModel):
        raise RuntimeError("canonical builder returned the wrong model type")
    if getattr(model, "contract_version", None) != registered.contract_version:
        raise RuntimeError("builder returned the wrong contract version")
    if getattr(model, "model_spec_hash", None) != registered_hash:
        raise RuntimeError("builder returned the wrong ModelSpec identity")
    return model


__all__ = [
    "MODEL_BUILDERS",
    "BuilderRegistry",
    "build_from_spec",
    "build_model",
    "build_tabu_cell_base",
    "register_model_builder",
]
