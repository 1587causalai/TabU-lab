"""Versioned, source-bound component extension contract for TabUBase."""

from __future__ import annotations

import hashlib
import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType, ModuleType
from typing import Any

from torch import nn

from tabu_lab.contracts import canonical_hash

from .components import CellTokenizer
from .dynamics import CellUnitDynamics
from .readouts import PairUnitReadout
from .types import DynamicsBlockKind, ReferenceConfig

_SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_COMPONENT_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ComponentRole(StrEnum):
    TOKENIZER = "tokenizer"
    DYNAMICS = "dynamics"
    READOUT = "readout"


class ComponentMaturity(StrEnum):
    CANONICAL = "canonical"
    EXPERIMENTAL = "experimental"


_ROLE_INTERFACES = {
    ComponentRole.TOKENIZER: "tabu.cell-tokenizer.v1",
    ComponentRole.DYNAMICS: "tabu.cell-dynamics.v1",
    ComponentRole.READOUT: "tabu.cell-readout.v1",
}
_ROLE_BASE_TYPES: dict[ComponentRole, type[nn.Module]] = {
    ComponentRole.TOKENIZER: CellTokenizer,
    ComponentRole.DYNAMICS: CellUnitDynamics,
    ComponentRole.READOUT: PairUnitReadout,
}


def implementation_source_identity(implementation: Any) -> tuple[str, str]:
    """Return an import identity and direct source digest for a class or factory."""

    if not callable(implementation):
        raise TypeError("component implementation identity requires a callable")
    implementation_ref = f"{implementation.__module__}:{implementation.__qualname__}"
    try:
        source = inspect.getsource(implementation).encode()
    except (OSError, TypeError) as exc:
        raise ValueError("component implementation source must be inspectable") from exc
    return implementation_ref, hashlib.sha256(source).hexdigest()


def factory_dependency_hash(factory: ComponentFactory) -> str:
    """Bind globals and closure values actually referenced by a factory."""

    if not callable(factory):
        raise TypeError("factory dependency identity requires a callable")
    if not inspect.isfunction(factory) or "<locals>" in factory.__qualname__:
        raise TypeError("component factory must be an import-resolvable module-level function")
    closure = inspect.getclosurevars(factory)
    if closure.nonlocals:
        raise ValueError("component factory cannot capture nonlocal closure state")

    def identity(value: Any) -> Any:
        if isinstance(value, ModuleType):
            raise ValueError("component factory cannot depend on a module object")
        if inspect.isfunction(value) or inspect.ismethod(value):
            raise ValueError("component factory cannot call an indirect helper function")
        if isinstance(value, type):
            reference, source_hash = implementation_source_identity(value)
            return {"kind": "type", "ref": reference, "source_sha256": source_hash}
        if callable(value):
            raise ValueError("component factory cannot depend on a callable object")
        try:
            canonical_hash(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("factory dependency is not identity-bindable") from exc
        return {"kind": "value", "value": value}

    payload = {
        "schema": "tabu.component-factory-dependencies.v1",
        "globals": {name: identity(value) for name, value in sorted(closure.globals.items())},
        "nonlocals": {},
        "defaults": identity(factory.__defaults__),
        "keyword_defaults": identity(factory.__kwdefaults__),
        "bytecode_sha256": hashlib.sha256(factory.__code__.co_code).hexdigest(),
    }
    return canonical_hash(payload)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        canonical_hash(value)
        return value
    raise TypeError("component config must contain only canonical JSON-like values")


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("component config must be a mapping")
    payload = _deep_freeze(value)
    canonical_hash(payload)
    return payload


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Semantic and source identity of one registered component implementation."""

    component_id: str
    component_version: str
    role: ComponentRole
    interface_id: str
    implementation_ref: str
    implementation_sha256: str
    factory_ref: str
    factory_sha256: str
    factory_dependency_sha256: str
    maturity: ComponentMaturity
    fixed_config: Mapping[str, Any] = field(default_factory=dict)
    configurable_fields: tuple[str, ...] = ()
    compatible_models: tuple[str, ...] = ("tabu.cell.base@0.2.0",)
    schema_version: str = "tabu.component-spec.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ComponentRole(self.role))
        object.__setattr__(self, "maturity", ComponentMaturity(self.maturity))
        if self.schema_version != "tabu.component-spec.v1":
            raise ValueError("unknown ComponentSpec schema version")
        if not _COMPONENT_ID.fullmatch(self.component_id):
            raise ValueError("component_id must be a namespaced lowercase id")
        if not _SEMVER.fullmatch(self.component_version):
            raise ValueError("component_version must be semantic-versioned")
        if self.interface_id != _ROLE_INTERFACES[self.role]:
            raise ValueError("component interface does not match its role")
        if not self.implementation_ref.strip():
            raise ValueError("implementation_ref cannot be blank")
        if not self.factory_ref.strip():
            raise ValueError("factory_ref cannot be blank")
        for name in (
            "implementation_sha256",
            "factory_sha256",
            "factory_dependency_sha256",
        ):
            if not _SHA256.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.compatible_models or any(not value.strip() for value in self.compatible_models):
            raise ValueError("compatible_models must be non-empty")
        if len(set(self.configurable_fields)) != len(self.configurable_fields):
            raise ValueError("configurable_fields must be unique")
        overlap = set(self.fixed_config).intersection(self.configurable_fields)
        if overlap:
            raise ValueError(f"fixed and configurable component fields overlap: {sorted(overlap)}")
        object.__setattr__(self, "fixed_config", _freeze_mapping(self.fixed_config))
        object.__setattr__(self, "configurable_fields", tuple(self.configurable_fields))
        object.__setattr__(self, "compatible_models", tuple(self.compatible_models))

    @property
    def spec_ref(self) -> str:
        return f"{self.component_id}@{self.component_version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "component_id": self.component_id,
            "component_version": self.component_version,
            "role": self.role.value,
            "interface_id": self.interface_id,
            "implementation_ref": self.implementation_ref,
            "implementation_sha256": self.implementation_sha256,
            "factory_ref": self.factory_ref,
            "factory_sha256": self.factory_sha256,
            "factory_dependency_sha256": self.factory_dependency_sha256,
            "maturity": self.maturity.value,
            "fixed_config": dict(self.fixed_config),
            "configurable_fields": self.configurable_fields,
            "compatible_models": self.compatible_models,
        }

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def resolve_config(self, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = sorted(set(supplied) - set(self.configurable_fields))
        if unknown:
            raise ValueError(f"unknown config for {self.spec_ref}: {unknown}")
        return _freeze_mapping({**dict(self.fixed_config), **dict(supplied)})


@dataclass(frozen=True, slots=True)
class ComponentRef:
    component_id: str
    component_version: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _COMPONENT_ID.fullmatch(self.component_id):
            raise ValueError("component ref id must be a namespaced lowercase id")
        if not _SEMVER.fullmatch(self.component_version):
            raise ValueError("component ref version must be semantic-versioned")
        object.__setattr__(self, "config", _freeze_mapping(self.config))

    @property
    def spec_ref(self) -> str:
        return f"{self.component_id}@{self.component_version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_version": self.component_version,
            "config": dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class TabUBaseComponentManifest:
    tokenizer: ComponentRef
    dynamics: ComponentRef
    readout: ComponentRef
    schema_version: str = "tabu.tabubase-component-manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "tabu.tabubase-component-manifest.v1":
            raise ValueError("unknown TabUBase component manifest version")
        for value in (self.tokenizer, self.dynamics, self.readout):
            if not isinstance(value, ComponentRef):
                raise TypeError("manifest entries must be typed ComponentRef values")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer": self.tokenizer.as_dict(),
            "dynamics": self.dynamics.as_dict(),
            "readout": self.readout.as_dict(),
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class ResolvedComponentComposition:
    manifest: TabUBaseComponentManifest
    tokenizer_spec_hash: str
    dynamics_spec_hash: str
    readout_spec_hash: str
    experimental_axes: tuple[str, ...]
    schema_version: str = "tabu.resolved-component-composition.v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.as_dict(),
            "manifest_hash": self.manifest.manifest_hash,
            "component_spec_hashes": {
                "tokenizer": self.tokenizer_spec_hash,
                "dynamics": self.dynamics_spec_hash,
                "readout": self.readout_spec_hash,
            },
            "experimental_axes": self.experimental_axes,
        }

    @property
    def composition_hash(self) -> str:
        return canonical_hash(self.as_dict())


ComponentFactory = Callable[[ReferenceConfig, Mapping[str, Any]], nn.Module]


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    spec: ComponentSpec
    factory: ComponentFactory
    runtime_type: type[nn.Module]


class ComponentRegistry:
    """Duplicate-rejecting registry with immutable canonical entries."""

    def __init__(self) -> None:
        self._entries: dict[str, _RegistryEntry] = {}
        self._protected: set[str] = set()

    def register(
        self,
        spec: ComponentSpec,
        factory: ComponentFactory,
        runtime_type: type[nn.Module],
    ) -> None:
        if not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be a typed ComponentSpec")
        if spec.maturity is not ComponentMaturity.EXPERIMENTAL:
            raise ValueError("public component registration requires experimental maturity")
        self._register(spec, factory, runtime_type, protected=False)

    def _register_canonical(
        self,
        spec: ComponentSpec,
        factory: ComponentFactory,
        runtime_type: type[nn.Module],
    ) -> None:
        if not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be a typed ComponentSpec")
        if spec.maturity is not ComponentMaturity.CANONICAL:
            raise ValueError("canonical registration requires canonical maturity")
        self._register(spec, factory, runtime_type, protected=True)

    def _register(
        self,
        spec: ComponentSpec,
        factory: ComponentFactory,
        runtime_type: type[nn.Module],
        *,
        protected: bool,
    ) -> None:
        if not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be a typed ComponentSpec")
        if not callable(factory):
            raise TypeError("component factory must be callable")
        if spec.spec_ref in self._entries:
            qualifier = "protected canonical" if spec.spec_ref in self._protected else "registered"
            raise ValueError(f"cannot replace {qualifier} component: {spec.spec_ref}")
        expected_base = _ROLE_BASE_TYPES[spec.role]
        if not isinstance(runtime_type, type) or not issubclass(runtime_type, expected_base):
            raise TypeError(f"{spec.role.value} runtime type violates its interface")
        implementation_ref, implementation_sha256 = implementation_source_identity(runtime_type)
        factory_ref, factory_sha256 = implementation_source_identity(factory)
        dependency_sha256 = factory_dependency_hash(factory)
        if (
            spec.implementation_ref != implementation_ref
            or spec.implementation_sha256 != implementation_sha256
            or spec.factory_ref != factory_ref
            or spec.factory_sha256 != factory_sha256
            or spec.factory_dependency_sha256 != dependency_sha256
        ):
            raise ValueError("ComponentSpec source identity does not match its type and factory")
        self._entries[spec.spec_ref] = _RegistryEntry(spec, factory, runtime_type)
        if protected:
            self._protected.add(spec.spec_ref)

    def fork(self) -> ComponentRegistry:
        registry = ComponentRegistry()
        registry._entries = dict(self._entries)
        registry._protected = set(self._protected)
        return registry

    def assert_extends(self, authority: ComponentRegistry) -> None:
        """Require every authoritative canonical entry byte-for-byte by spec hash."""

        if not isinstance(authority, ComponentRegistry):
            raise TypeError("component authority must be a ComponentRegistry")
        for spec_ref in authority._protected:
            expected = authority._entries[spec_ref]
            actual = self._entries.get(spec_ref)
            if actual is None or actual.spec.spec_hash != expected.spec.spec_hash:
                raise ValueError(
                    f"component registry does not preserve canonical anchor: {spec_ref}"
                )

    def is_authoritative(self, ref: ComponentRef, authority: ComponentRegistry) -> bool:
        """Return whether one selected ref is the exact authoritative canonical spec."""

        expected = authority._entries.get(ref.spec_ref)
        actual = self._entries.get(ref.spec_ref)
        return bool(
            expected is not None
            and ref.spec_ref in authority._protected
            and actual is not None
            and actual.spec.spec_hash == expected.spec.spec_hash
        )

    def validate_runtime(
        self,
        composition: ResolvedComponentComposition,
        modules: Mapping[str, nn.Module],
    ) -> None:
        """Re-resolve specs and bind each actual module to its exact registered type."""

        if not isinstance(composition, ResolvedComponentComposition):
            raise TypeError("runtime composition must be resolved by ComponentRegistry")
        if self.resolve(composition.manifest) != composition:
            raise ValueError("runtime composition no longer matches its registry")
        expected_refs = {
            "tokenizer": composition.manifest.tokenizer.spec_ref,
            "dynamics": composition.manifest.dynamics.spec_ref,
            "readout": composition.manifest.readout.spec_ref,
        }
        if set(modules) != set(expected_refs):
            raise ValueError("runtime component roles do not match the manifest")
        for axis, spec_ref in expected_refs.items():
            entry = self._entries[spec_ref]
            module = modules[axis]
            if type(module) is not entry.runtime_type:
                raise TypeError(f"runtime {axis} type does not match its ComponentSpec")
            implementation_ref, implementation_sha256 = implementation_source_identity(
                type(module)
            )
            if (
                implementation_ref != entry.spec.implementation_ref
                or implementation_sha256 != entry.spec.implementation_sha256
            ):
                raise ValueError(f"runtime {axis} source identity drifted after registration")

    def get(self, ref: ComponentRef, *, expected_role: ComponentRole) -> ComponentSpec:
        if not isinstance(ref, ComponentRef):
            raise TypeError("component selection must be a typed ComponentRef")
        try:
            entry = self._entries[ref.spec_ref]
        except KeyError as exc:
            raise KeyError(f"unknown component: {ref.spec_ref}") from exc
        if entry.spec.role is not ComponentRole(expected_role):
            raise ValueError("component role does not match its manifest position")
        if "tabu.cell.base@0.2.0" not in entry.spec.compatible_models:
            raise ValueError("component is not compatible with tabu.cell.base@0.2.0")
        entry.spec.resolve_config(ref.config)
        return entry.spec

    def build(
        self,
        ref: ComponentRef,
        *,
        expected_role: ComponentRole,
        config: ReferenceConfig,
    ) -> nn.Module:
        spec = self.get(ref, expected_role=expected_role)
        entry = self._entries[ref.spec_ref]
        factory_ref, factory_sha256 = implementation_source_identity(entry.factory)
        if (
            factory_ref != spec.factory_ref
            or factory_sha256 != spec.factory_sha256
            or factory_dependency_hash(entry.factory) != spec.factory_dependency_sha256
        ):
            raise ValueError("component factory identity drifted after registration")
        module = entry.factory(config, spec.resolve_config(ref.config))
        if type(module) is not entry.runtime_type:
            raise TypeError("component factory returned the wrong runtime type")
        return module

    def resolve(self, manifest: TabUBaseComponentManifest) -> ResolvedComponentComposition:
        if not isinstance(manifest, TabUBaseComponentManifest):
            raise TypeError("component_manifest must be a typed TabUBaseComponentManifest")
        specs = {
            "tokenizer": self.get(manifest.tokenizer, expected_role=ComponentRole.TOKENIZER),
            "dynamics": self.get(manifest.dynamics, expected_role=ComponentRole.DYNAMICS),
            "readout": self.get(manifest.readout, expected_role=ComponentRole.READOUT),
        }
        experimental_axes = tuple(
            axis
            for axis in ("tokenizer", "dynamics", "readout")
            if specs[axis].maturity is ComponentMaturity.EXPERIMENTAL
        )
        return ResolvedComponentComposition(
            manifest=manifest,
            tokenizer_spec_hash=specs["tokenizer"].spec_hash,
            dynamics_spec_hash=specs["dynamics"].spec_hash,
            readout_spec_hash=specs["readout"].spec_hash,
            experimental_axes=experimental_axes,
        )

    def refs(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


def _component_spec(
    *,
    component_id: str,
    component_version: str,
    role: ComponentRole,
    runtime_type: type[nn.Module],
    factory: ComponentFactory,
    fixed_config: Mapping[str, Any],
    configurable_fields: tuple[str, ...] = (),
) -> ComponentSpec:
    implementation_ref, implementation_sha256 = implementation_source_identity(runtime_type)
    factory_ref, factory_sha256 = implementation_source_identity(factory)
    factory_dependency_sha256 = factory_dependency_hash(factory)
    return ComponentSpec(
        component_id=component_id,
        component_version=component_version,
        role=role,
        interface_id=_ROLE_INTERFACES[role],
        implementation_ref=implementation_ref,
        implementation_sha256=implementation_sha256,
        factory_ref=factory_ref,
        factory_sha256=factory_sha256,
        factory_dependency_sha256=factory_dependency_sha256,
        maturity=ComponentMaturity.CANONICAL,
        fixed_config=fixed_config,
        configurable_fields=configurable_fields,
    )


def _tokenizer_factory(config: ReferenceConfig, options: Mapping[str, Any]) -> nn.Module:
    return CellTokenizer(config, marker="mask", **dict(options))


def _dynamics_factory(config: ReferenceConfig, options: Mapping[str, Any]) -> nn.Module:
    if options.get("block_kind") != "omab" or config.block_kind is not DynamicsBlockKind.OMAB:
        raise ValueError("canonical dynamics component requires OMAB ReferenceConfig")
    return CellUnitDynamics(config)


def _readout_factory(config: ReferenceConfig, options: Mapping[str, Any]) -> nn.Module:
    return PairUnitReadout(config, numeric_terminal=str(options["numeric_terminal"]))


CANONICAL_COMPONENTS = ComponentRegistry()

_TOKENIZER_V1 = _component_spec(
    component_id="tabu.tokenizer.cell",
    component_version="1.0.0",
    role=ComponentRole.TOKENIZER,
    runtime_type=CellTokenizer,
    factory=_tokenizer_factory,
    fixed_config={"nominal_tokenizer": CellTokenizer.EPISODE_RANDOM_SPHERE_V1},
)
_TOKENIZER_V2 = _component_spec(
    component_id="tabu.tokenizer.cell",
    component_version="2.0.0",
    role=ComponentRole.TOKENIZER,
    runtime_type=CellTokenizer,
    factory=_tokenizer_factory,
    fixed_config={"nominal_tokenizer": CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2},
    configurable_fields=("nominal_codebook_size", "nominal_codebook_seed"),
)
_DYNAMICS_OMAB = _component_spec(
    component_id="tabu.dynamics.cell-unit-three-omab",
    component_version="1.0.0",
    role=ComponentRole.DYNAMICS,
    runtime_type=CellUnitDynamics,
    factory=_dynamics_factory,
    fixed_config={"block_kind": "omab"},
)
_READOUT_LL = _component_spec(
    component_id="tabu.readout.same-column-local-linear",
    component_version="1.0.0",
    role=ComponentRole.READOUT,
    runtime_type=PairUnitReadout,
    factory=_readout_factory,
    fixed_config={"numeric_terminal": "local_linear"},
)
_READOUT_NW = _component_spec(
    component_id="tabu.readout.same-column-nadaraya-watson",
    component_version="1.0.0",
    role=ComponentRole.READOUT,
    runtime_type=PairUnitReadout,
    factory=_readout_factory,
    fixed_config={"numeric_terminal": "nadaraya_watson"},
)

for _spec, _factory, _runtime_type in (
    (_TOKENIZER_V1, _tokenizer_factory, CellTokenizer),
    (_TOKENIZER_V2, _tokenizer_factory, CellTokenizer),
    (_DYNAMICS_OMAB, _dynamics_factory, CellUnitDynamics),
    (_READOUT_LL, _readout_factory, PairUnitReadout),
    (_READOUT_NW, _readout_factory, PairUnitReadout),
):
    CANONICAL_COMPONENTS._register_canonical(_spec, _factory, _runtime_type)


def canonical_tabu_base_manifest(
    *,
    nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
    numeric_terminal: str = "local_linear",
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
) -> TabUBaseComponentManifest:
    if nominal_tokenizer == CellTokenizer.EPISODE_RANDOM_SPHERE_V1:
        tokenizer = ComponentRef("tabu.tokenizer.cell", "1.0.0")
    elif nominal_tokenizer == CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2:
        tokenizer = ComponentRef(
            "tabu.tokenizer.cell",
            "2.0.0",
            config={
                "nominal_codebook_size": nominal_codebook_size,
                "nominal_codebook_seed": nominal_codebook_seed,
            },
        )
    else:
        raise ValueError("unknown canonical tokenizer selection")
    readouts = {
        "local_linear": ComponentRef("tabu.readout.same-column-local-linear", "1.0.0"),
        "nadaraya_watson": ComponentRef(
            "tabu.readout.same-column-nadaraya-watson",
            "1.0.0",
        ),
    }
    try:
        readout = readouts[numeric_terminal]
    except KeyError as exc:
        raise ValueError("unknown canonical readout selection") from exc
    return TabUBaseComponentManifest(
        tokenizer=tokenizer,
        dynamics=ComponentRef("tabu.dynamics.cell-unit-three-omab", "1.0.0"),
        readout=readout,
    )


__all__ = [
    "CANONICAL_COMPONENTS",
    "ComponentMaturity",
    "ComponentRef",
    "ComponentRegistry",
    "ComponentRole",
    "ComponentSpec",
    "ResolvedComponentComposition",
    "TabUBaseComponentManifest",
    "canonical_tabu_base_manifest",
    "factory_dependency_hash",
    "implementation_source_identity",
]
