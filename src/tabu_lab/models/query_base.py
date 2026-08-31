"""Axis-C QueryBase runtime and its typed family-growth seams.

The module deliberately does not inherit the Axis-B ``TabUCellBaseModel``
identity.  It reuses low-level tensor operators through query-specific
adapters and binds a new contract, component authority, source topology, and
checkpoint namespace to the resulting model.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.contracts import canonical_hash
from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE
from tabu_lab.primitives import NumericReadoutOutput

from .component_registry import (
    factory_dependency_hash,
    implementation_source_identity,
)
from .components import CellTokenizer, NumericScaleState
from .dynamics import AugmentedDynamics, CellUnitDynamics, DynamicsPlan
from .readouts import _numeric_terminal
from .reference import DenseReferenceModel, _shape_event
from .table_cell import (
    _apply_cell_null_contract,
    _label_broadcast,
    _reference_config_payload,
)
from .types import ModelVariantRef, ReferenceConfig


class AxisMode(StrEnum):
    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


class RowReadoutMode(StrEnum):
    """Step-4 TabUR readout arms frozen by ``tabu.query.row@0.2.0``."""

    HOMOGENEOUS = "homogeneous"
    ANCHORED = "anchored"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class AxisRoleSpec:
    """One row/column role in the query family generator."""

    axis: Literal["row", "column"]
    mode: AxisMode
    token_bank: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.axis not in {"row", "column"}:
            raise ValueError("axis must be 'row' or 'column'")
        object.__setattr__(self, "mode", AxisMode(self.mode))
        tokens = tuple(self.token_bank)
        if any(not token.strip() for token in tokens):
            raise ValueError("token_bank entries must be non-empty")
        if len(set(tokens)) != len(tokens):
            raise ValueError("token_bank entries must be unique")
        if self.mode is AxisMode.HOMOGENEOUS and tokens:
            raise ValueError("homogeneous axes cannot carry private Unit tokens")
        if self.mode is AxisMode.HETEROGENEOUS and not tokens:
            raise ValueError("heterogeneous axes require an explicit token_bank")
        object.__setattr__(self, "token_bank", tokens)

    def as_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "mode": self.mode.value,
            "token_bank": self.token_bank,
        }


@dataclass(frozen=True, slots=True)
class QueryFamilyPlan:
    """Typed family generator; Base is the H/H specialization."""

    row_axis: AxisRoleSpec = field(
        default_factory=lambda: AxisRoleSpec("row", AxisMode.HOMOGENEOUS)
    )
    column_axis: AxisRoleSpec = field(
        default_factory=lambda: AxisRoleSpec("column", AxisMode.HOMOGENEOUS)
    )
    cell_role: Literal["query"] = "query"
    response_mechanism: str = "shared_W_fallback"

    def __post_init__(self) -> None:
        if self.cell_role != "query":
            raise ValueError("QueryFamilyPlan cell_role is fixed to 'query'")
        if self.row_axis.axis != "row" or self.column_axis.axis != "column":
            raise ValueError("row_axis and column_axis must preserve their axis roles")
        if self.response_mechanism not in {
            "shared_W_fallback",
            "row_unit_projection",
            "row_readout",
        }:
            raise ValueError("unknown QueryFamilyPlan response mechanism")

    @classmethod
    def base(cls) -> QueryFamilyPlan:
        return cls()

    @property
    def geometry(self) -> str:
        row = self.row_axis.mode is AxisMode.HETEROGENEOUS
        column = self.column_axis.mode is AxisMode.HETEROGENEOUS
        if not row and not column:
            return "global_W"
        if row and not column:
            return "row_heterogeneous"
        if not row and column:
            return "column_heterogeneous"
        return "row_column_concat"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tabu.query-family-plan.v1",
            "cell_role": self.cell_role,
            "row_axis": self.row_axis.as_dict(),
            "column_axis": self.column_axis.as_dict(),
            "geometry": self.geometry,
            "response_mechanism": self.response_mechanism,
        }

    @property
    def plan_hash(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class AxisSourcePlan:
    """Source topology separated from model orchestration and hash-bound."""

    column_source_policy: str = "context_visible_only"
    row_source_policy: str = "row_visible_including_query"
    receiver_only_origins: tuple[str, ...] = (
        "query",
        "artificial_mask",
        "null",
        "natural_missing",
    )

    def __post_init__(self) -> None:
        if self.column_source_policy != "context_visible_only":
            raise ValueError("QueryBase column source policy is context_visible_only")
        if self.row_source_policy != "row_visible_including_query":
            raise ValueError("QueryBase row source policy is row_visible_including_query")
        origins = tuple(self.receiver_only_origins)
        if len(set(origins)) != len(origins):
            raise ValueError("receiver_only_origins must be unique")
        object.__setattr__(self, "receiver_only_origins", origins)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tabu.query-axis-source-plan.v1",
            "column_source_policy": self.column_source_policy,
            "row_source_policy": self.row_source_policy,
            "receiver_only_origins": self.receiver_only_origins,
        }

    @property
    def plan_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def resolve(self, inputs: Any, *, supervised: bool) -> tuple[Tensor, Tensor]:
        """Return ``(column_source_mask, row_source_mask)`` for one episode."""

        visible = inputs.visible_mask
        context = visible
        if supervised:
            query_rows = inputs.query_target_mask.any(dim=2, keepdim=True)
            context = visible & ~query_rows
        return context, visible


class TabUQueryProfile(StrEnum):
    COMPLETION_ARTIFICIAL_MASK_V1 = "completion.artificial_mask.v1"
    SUPERVISED_LABEL_BROADCAST_V1 = "supervised.label_broadcast.v1"

    @property
    def uses_label_broadcast(self) -> bool:
        return self is TabUQueryProfile.SUPERVISED_LABEL_BROADCAST_V1


class QueryComponentRole(StrEnum):
    TOKENIZER = "tokenizer"
    AXIS_SOURCE = "axis_source"
    DYNAMICS = "dynamics"
    GEOMETRY = "geometry"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class QueryComponentRef:
    component_id: str
    component_version: str
    role: QueryComponentRole
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id or not self.component_id.strip():
            raise ValueError("component_id cannot be blank")
        if not self.component_version or not self.component_version.strip():
            raise ValueError("component_version cannot be blank")
        object.__setattr__(self, "role", QueryComponentRole(self.role))
        canonical_hash(dict(self.config))
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    @property
    def spec_ref(self) -> str:
        return f"{self.component_id}@{self.component_version}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_version": self.component_version,
            "role": self.role.value,
            "config": dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class QueryComponentSpec:
    component_id: str
    component_version: str
    role: QueryComponentRole
    interface_id: str
    implementation_ref: str
    implementation_sha256: str
    factory_ref: str
    factory_sha256: str
    factory_dependency_sha256: str
    compatible_models: tuple[str, ...] = ("tabu.query.base@0.1.0",)
    fixed_config: Mapping[str, Any] = field(default_factory=dict)
    configurable_fields: tuple[str, ...] = ()
    maturity: Literal["canonical", "experimental"] = "canonical"
    schema_version: str = "tabu.query-component-spec.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", QueryComponentRole(self.role))
        if self.schema_version != "tabu.query-component-spec.v1":
            raise ValueError("unknown QueryComponentSpec schema version")
        if not self.component_id.strip() or not self.component_version.strip():
            raise ValueError("query component identity cannot be blank")
        if not self.implementation_ref.strip() or not self.factory_ref.strip():
            raise ValueError("query component source identity cannot be blank")
        for name in (
            "implementation_sha256",
            "factory_sha256",
            "factory_dependency_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not self.compatible_models:
            raise ValueError("compatible_models cannot be empty")
        overlap = set(self.fixed_config).intersection(self.configurable_fields)
        if overlap:
            raise ValueError(f"fixed/configurable overlap: {sorted(overlap)}")
        canonical_hash(dict(self.fixed_config))
        object.__setattr__(self, "fixed_config", MappingProxyType(dict(self.fixed_config)))
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
            "compatible_models": self.compatible_models,
            "fixed_config": dict(self.fixed_config),
            "configurable_fields": self.configurable_fields,
            "maturity": self.maturity,
        }

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def resolve_config(self, supplied: Mapping[str, Any]) -> Mapping[str, Any]:
        unknown = sorted(set(supplied) - set(self.configurable_fields))
        if unknown:
            raise ValueError(f"unknown config for {self.spec_ref}: {unknown}")
        return {**dict(self.fixed_config), **dict(supplied)}


@dataclass(frozen=True, slots=True)
class QueryComponentManifest:
    tokenizer: QueryComponentRef
    axis_source: QueryComponentRef
    dynamics: QueryComponentRef
    geometry: QueryComponentRef
    terminal: QueryComponentRef
    schema_version: str = "tabu.query-component-manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "tabu.query-component-manifest.v1":
            raise ValueError("unknown QueryComponentManifest schema version")
        expected = {
            "tokenizer": QueryComponentRole.TOKENIZER,
            "axis_source": QueryComponentRole.AXIS_SOURCE,
            "dynamics": QueryComponentRole.DYNAMICS,
            "geometry": QueryComponentRole.GEOMETRY,
            "terminal": QueryComponentRole.TERMINAL,
        }
        for name, role in expected.items():
            value = getattr(self, name)
            if not isinstance(value, QueryComponentRef) or value.role is not role:
                raise TypeError(f"{name} must be a typed {role.value} QueryComponentRef")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tokenizer": self.tokenizer.as_dict(),
            "axis_source": self.axis_source.as_dict(),
            "dynamics": self.dynamics.as_dict(),
            "geometry": self.geometry.as_dict(),
            "terminal": self.terminal.as_dict(),
        }

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class ResolvedQueryComponentComposition:
    manifest: QueryComponentManifest
    component_spec_hashes: Mapping[str, str]
    schema_version: str = "tabu.resolved-query-component-composition.v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest": self.manifest.as_dict(),
            "manifest_hash": self.manifest.manifest_hash,
            "component_spec_hashes": dict(sorted(self.component_spec_hashes.items())),
        }

    @property
    def composition_hash(self) -> str:
        return canonical_hash(self.as_dict())


QueryComponentFactory = Callable[[ReferenceConfig, Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class _QueryRegistryEntry:
    spec: QueryComponentSpec
    factory: QueryComponentFactory
    runtime_type: type[Any]


class QueryComponentRegistry:
    """Protected query component registry independent of Axis-B authority."""

    def __init__(self) -> None:
        self._entries: dict[str, _QueryRegistryEntry] = {}
        self._protected: set[str] = set()

    def _register(
        self,
        spec: QueryComponentSpec,
        factory: QueryComponentFactory,
        runtime_type: type[Any],
        *,
        protected: bool,
    ) -> None:
        if spec.spec_ref in self._entries:
            qualifier = "protected canonical" if spec.spec_ref in self._protected else "registered"
            raise ValueError(f"cannot replace {qualifier} query component: {spec.spec_ref}")
        if not callable(factory) or not isinstance(runtime_type, type):
            raise TypeError("query component factory/runtime_type is invalid")
        expected_base = _QUERY_ROLE_BASE_TYPES[spec.role]
        if not issubclass(runtime_type, expected_base):
            raise TypeError(
                f"{spec.role.value} runtime type must extend the query-neutral interface "
                f"{expected_base.__name__}"
            )
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
            raise ValueError("QueryComponentSpec source identity does not match implementation")
        self._entries[spec.spec_ref] = _QueryRegistryEntry(spec, factory, runtime_type)
        if protected:
            self._protected.add(spec.spec_ref)

    def register(
        self,
        spec: QueryComponentSpec,
        factory: QueryComponentFactory,
        runtime_type: type[Any],
    ) -> None:
        if spec.maturity != "experimental":
            raise ValueError("public query component registration requires experimental maturity")
        self._register(spec, factory, runtime_type, protected=False)

    def fork(self) -> QueryComponentRegistry:
        """Copy the protected authority for an explicitly experimental extension."""

        clone = QueryComponentRegistry()
        clone._entries = dict(self._entries)
        clone._protected = set(self._protected)
        return clone

    def _register_canonical(
        self,
        spec: QueryComponentSpec,
        factory: QueryComponentFactory,
        runtime_type: type[Any],
    ) -> None:
        if spec.maturity != "canonical":
            raise ValueError("canonical query component requires canonical maturity")
        self._register(spec, factory, runtime_type, protected=True)

    def resolve(
        self,
        manifest: QueryComponentManifest,
        *,
        model_ref: str = "tabu.query.base@0.1.0",
    ) -> ResolvedQueryComponentComposition:
        refs = {
            "tokenizer": manifest.tokenizer,
            "axis_source": manifest.axis_source,
            "dynamics": manifest.dynamics,
            "geometry": manifest.geometry,
            "terminal": manifest.terminal,
        }
        hashes: dict[str, str] = {}
        for name, ref in refs.items():
            try:
                entry = self._entries[ref.spec_ref]
            except KeyError as exc:
                raise ValueError(f"unknown query component: {ref.spec_ref}") from exc
            if entry.spec.role is not ref.role:
                raise ValueError(f"query component role mismatch for {ref.spec_ref}")
            if model_ref not in entry.spec.compatible_models:
                raise ValueError(
                    f"query component is not compatible with {model_ref}: "
                    f"{ref.spec_ref}"
                )
            entry.spec.resolve_config(ref.config)
            hashes[name] = entry.spec.spec_hash
        return ResolvedQueryComponentComposition(manifest, hashes)

    def get(self, ref: QueryComponentRef) -> QueryComponentSpec:
        try:
            entry = self._entries[ref.spec_ref]
        except KeyError as exc:
            raise KeyError(f"unknown query component: {ref.spec_ref}") from exc
        if entry.spec.role is not ref.role:
            raise ValueError(f"query component role mismatch for {ref.spec_ref}")
        return entry.spec

    def build(self, ref: QueryComponentRef, *, config: ReferenceConfig) -> Any:
        try:
            entry = self._entries[ref.spec_ref]
        except KeyError as exc:
            raise ValueError(f"unknown query component: {ref.spec_ref}") from exc
        if entry.spec.role is not ref.role:
            raise ValueError(f"query component role mismatch for {ref.spec_ref}")
        resolved = entry.spec.resolve_config(ref.config)
        value = entry.factory(config, resolved)
        if not isinstance(value, entry.runtime_type):
            raise TypeError(f"query component factory returned wrong type for {ref.spec_ref}")
        return value

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


class QueryTokenizerAdapter(nn.Module):
    """Query-specific identity wrapper around the reusable tokenizer operator."""

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
    ) -> None:
        super().__init__()
        self.impl = CellTokenizer(
            config,
            marker="query",
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
        )

    def forward(self, symbols: Any) -> Any:
        return self.impl(symbols)

    @property
    def nominal_tokenizer(self) -> str:
        return self.impl.nominal_tokenizer

    @property
    def nominal_codebook_size(self) -> int:
        return self.impl.nominal_codebook_size

    @property
    def nominal_codebook_seed(self) -> int:
        return self.impl.nominal_codebook_seed

    @property
    def nominal_codebook_hash(self) -> str:
        return self.impl.nominal_codebook_hash


class QueryAxisSourceAdapter:
    def __init__(self, plan: AxisSourcePlan | None = None) -> None:
        self.plan = plan or AxisSourcePlan()

    def resolve(self, inputs: Any, *, supervised: bool) -> tuple[Tensor, Tensor]:
        return self.plan.resolve(inputs, supervised=supervised)


class QueryDynamicsAdapter(nn.Module):
    """Query-family dynamics using the existing O-closed tensor block."""

    plan = DynamicsPlan(
        name="query_axis_three_omab",
        stages=("column_slots", "token_from_slot", "row_axis"),
        carrier="NxM",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.impl = CellUnitDynamics(config)

    def forward(
        self,
        carrier: Tensor,
        *,
        column_source_mask: Tensor,
        row_source_mask: Tensor,
    ) -> Tensor:
        return self.impl(
            carrier,
            column_source_mask=column_source_mask,
            row_source_mask=row_source_mask,
        )


class QueryRowDynamicsAdapter(QueryDynamicsAdapter):
    """Query-row dynamics on the augmented ``N x (M + K)`` carrier."""

    plan = DynamicsPlan(
        name="query_row_augmented_three_omab",
        stages=("column_slots", "token_from_slot", "row_axis"),
        carrier="Nx(M+K)",
    )

    def __init__(self, config: ReferenceConfig) -> None:
        nn.Module.__init__(self)
        self.impl = AugmentedDynamics(config)


class QueryGeometryAdapter(nn.Module):
    """Global $W$ response geometry; $W$ is not a semantic Unit."""

    geometry = "global_W"

    def __init__(self, config: ReferenceConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(
            config.d_model,
            config.matched_slots,
            bias=False,
            dtype=DEFAULT_FLOAT_DTYPE,
        )

    def forward(self, cells: Tensor) -> Tensor:
        return F.linear(
            cells.to(dtype=DEFAULT_FLOAT_DTYPE),
            self.projection.weight.to(dtype=DEFAULT_FLOAT_DTYPE),
        )


class QueryRowGeometryAdapter(QueryGeometryAdapter):
    r"""Symmetric TabUR readout with a global Base anchor.

    The default arm implements

    .. math:: z_{ra}=(W+\gamma\widehat U_r A^\top)c_{ra}.

    ``homogeneous`` is the exact Base readout ``Wc`` and ``free`` retains the
    old ``Uc`` geometry as an explicitly selected ablation.  LayerNorm has no
    affine parameters, so the only anchored-only learned parameters are
    ``A0`` and ``gamma`` in addition to the Base-compatible global ``W``.
    """

    geometry = "row_readout"
    axis_transform_normalization = "exact_spectral_norm_v1"

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        token_count: int = 4,
        row_readout_mode: RowReadoutMode | str = RowReadoutMode.ANCHORED,
        anchored_gamma_initial: float = 1.0e-2,
        axis_transform_normalization: str = "exact_spectral_norm_v1",
    ) -> None:
        if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
            raise ValueError("token_count must be a positive integer")
        if token_count != config.matched_slots:
            raise ValueError(
                "TabUR requires K to equal row_token_count, W rows, and matched_slots"
            )
        mode = RowReadoutMode(row_readout_mode)
        gamma_initial = float(anchored_gamma_initial)
        if not math.isfinite(gamma_initial):
            raise ValueError("anchored_gamma_initial must be finite")
        if mode is not RowReadoutMode.ANCHORED and gamma_initial != 1.0e-2:
            raise ValueError(
                "anchored_gamma_initial is fixed at 0.01 when row_readout_mode "
                "is homogeneous or free"
            )
        if axis_transform_normalization != self.axis_transform_normalization:
            raise ValueError(
                "axis_transform_normalization must be exact_spectral_norm_v1"
            )

        # Construct W in every arm. This preserves Base parameterization and
        # matched-initialization order even when an ablation does not consume W.
        super().__init__(config)
        self.token_count = token_count
        self.row_readout_mode = mode
        self.anchored_gamma_initial = gamma_initial

        identity = torch.eye(config.d_model, dtype=DEFAULT_FLOAT_DTYPE)
        if mode is RowReadoutMode.ANCHORED:
            self.raw_axis_transform = nn.Parameter(identity)
            self.gamma = nn.Parameter(
                torch.tensor(gamma_initial, dtype=DEFAULT_FLOAT_DTYPE)
            )
        else:
            self.register_buffer("raw_axis_transform", identity, persistent=True)
            fixed_gamma = 0.0 if mode is RowReadoutMode.HOMOGENEOUS else 1.0
            self.register_buffer(
                "gamma",
                torch.tensor(fixed_gamma, dtype=DEFAULT_FLOAT_DTYPE),
                persistent=True,
            )
        if mode is RowReadoutMode.FREE:
            self.projection.weight.requires_grad_(False)

    @property
    def beta(self) -> float:
        return 0.0 if self.row_readout_mode is RowReadoutMode.FREE else 1.0

    @staticmethod
    def _exact_spectral_norm(value: Tensor) -> Tensor:
        # PyTorch MPS does not currently implement SVD.  A cross-device copy is
        # autograd-visible, so computing this small dxd norm on CPU preserves
        # the exact contract and propagates its gradient back to the MPS A0.
        if value.device.type == "mps":
            return torch.linalg.matrix_norm(value.to("cpu"), ord=2).to(value.device)
        return torch.linalg.matrix_norm(value, ord=2)

    def effective_axis_transform(self) -> Tensor:
        raw = self.raw_axis_transform.to(dtype=DEFAULT_FLOAT_DTYPE)
        norm = self._exact_spectral_norm(raw)
        detached_norm = norm.detach()
        if not bool(torch.isfinite(detached_norm)) or float(detached_norm) <= 0.0:
            raise ValueError("TabUR axis transform must have finite non-zero spectral norm")
        return raw / norm

    def readout_identity(self) -> dict[str, Any]:
        return {
            "schema_version": "tabu.query-row-readout.v1",
            "mode": self.row_readout_mode.value,
            "beta": self.beta,
            "anchored_gamma_initial": self.anchored_gamma_initial,
            "axis_transform_normalization": self.axis_transform_normalization,
            "row_token_count": self.token_count,
            "global_w_rows": self.projection.out_features,
        }

    def trace_metadata(self) -> dict[str, Any]:
        return {
            **self.readout_identity(),
            "gamma": float(self.gamma.detach()),
            # A successful anchored forward has already validated A0 and
            # divided by its exact norm.  Report the invariant without running
            # a second SVD merely to construct trace metadata.
            "effective_axis_transform_spectral_norm": 1.0,
        }

    def forward(self, carrier: Tensor) -> Tensor:
        if carrier.ndim != 4:
            raise ValueError("row geometry carrier must be [B,N,M+K,D]")
        if carrier.shape[2] <= self.token_count:
            raise ValueError("row geometry carrier is missing ordinary cells")
        if carrier.shape[-1] != self.projection.in_features:
            raise ValueError("row geometry carrier width does not match W")
        cells = carrier[:, :, :-self.token_count, :]
        row_units = carrier[:, :, -self.token_count :, :]

        if self.row_readout_mode is RowReadoutMode.HOMOGENEOUS:
            coordinates = super().forward(cells)
        elif self.row_readout_mode is RowReadoutMode.FREE:
            coordinates = torch.einsum("bnkd,bnmd->bnmk", row_units, cells)
        else:
            normalized_units = F.layer_norm(
                row_units.to(dtype=DEFAULT_FLOAT_DTYPE),
                (row_units.shape[-1],),
            )
            transformed_units = F.linear(
                normalized_units,
                self.effective_axis_transform(),
            )
            anchored = torch.einsum(
                "bnkd,bnmd->bnmk",
                transformed_units,
                cells.to(dtype=DEFAULT_FLOAT_DTYPE),
            )
            coordinates = super().forward(cells) + self.gamma.to(
                dtype=DEFAULT_FLOAT_DTYPE
            ) * anchored

        if coordinates.shape[-1] != self.token_count:
            raise RuntimeError("TabUR coordinate width must equal K")
        return coordinates


class QueryTerminalAdapter(nn.Module):
    def __init__(self, config: ReferenceConfig, *, numeric_terminal: str = "local_linear") -> None:
        super().__init__()
        self.terminal, self.numeric_terminal = _numeric_terminal(
            config,
            numeric_terminal=numeric_terminal,
        )
        self.numeric_terminal_trace = (
            "nw" if self.numeric_terminal == "nadaraya_watson" else "local_linear"
        )

    @property
    def ll_ridge(self) -> float | None:
        value = getattr(self.terminal, "ridge", None)
        return None if value is None else float(value)

    def forward(
        self,
        coordinates: Tensor,
        support_values: Tensor,
        visible_mask: Tensor,
    ) -> NumericReadoutOutput:
        return self.terminal(coordinates, support_values, visible_mask)


_QUERY_ROLE_BASE_TYPES: dict[QueryComponentRole, type[Any]] = {
    QueryComponentRole.TOKENIZER: QueryTokenizerAdapter,
    QueryComponentRole.AXIS_SOURCE: QueryAxisSourceAdapter,
    QueryComponentRole.DYNAMICS: QueryDynamicsAdapter,
    QueryComponentRole.GEOMETRY: QueryGeometryAdapter,
    QueryComponentRole.TERMINAL: QueryTerminalAdapter,
}


def _query_tokenizer_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryTokenizerAdapter:
    return QueryTokenizerAdapter(
        config,
        nominal_tokenizer=str(
            settings.get("nominal_tokenizer", CellTokenizer.EPISODE_RANDOM_SPHERE_V1)
        ),
        nominal_codebook_size=int(settings.get("nominal_codebook_size", 100)),
        nominal_codebook_seed=int(settings.get("nominal_codebook_seed", 1729)),
    )


def _query_axis_source_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryAxisSourceAdapter:
    del config
    return QueryAxisSourceAdapter(AxisSourcePlan(**dict(settings)))


def _query_dynamics_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryDynamicsAdapter:
    del settings
    return QueryDynamicsAdapter(config)


def _query_row_dynamics_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryRowDynamicsAdapter:
    del settings
    return QueryRowDynamicsAdapter(config)


def _query_geometry_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryGeometryAdapter:
    del settings
    return QueryGeometryAdapter(config)


def _query_row_geometry_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryRowGeometryAdapter:
    required = {
        "token_count",
        "row_readout_mode",
        "anchored_gamma_initial",
        "axis_transform_normalization",
    }
    missing = sorted(required - set(settings))
    if missing:
        raise ValueError(f"TabUR row readout manifest is missing required config: {missing}")
    return QueryRowGeometryAdapter(
        config,
        token_count=int(settings["token_count"]),
        row_readout_mode=str(settings["row_readout_mode"]),
        anchored_gamma_initial=float(settings["anchored_gamma_initial"]),
        axis_transform_normalization=str(settings["axis_transform_normalization"]),
    )


def _query_terminal_factory(
    config: ReferenceConfig,
    settings: Mapping[str, Any],
) -> QueryTerminalAdapter:
    return QueryTerminalAdapter(
        config,
        numeric_terminal=str(settings.get("numeric_terminal", "local_linear")),
    )


def _query_component_spec(
    *,
    component_id: str,
    role: QueryComponentRole,
    implementation: type[Any],
    factory: QueryComponentFactory,
    configurable_fields: tuple[str, ...],
    compatible_models: tuple[str, ...] = ("tabu.query.base@0.1.0",),
    component_version: str = "0.1.0",
) -> QueryComponentSpec:
    implementation_ref, implementation_sha256 = implementation_source_identity(implementation)
    factory_ref, factory_sha256 = implementation_source_identity(factory)
    return QueryComponentSpec(
        component_id=component_id,
        component_version=component_version,
        role=role,
        interface_id=f"tabu.query-{role.value}.v1",
        implementation_ref=implementation_ref,
        implementation_sha256=implementation_sha256,
        factory_ref=factory_ref,
        factory_sha256=factory_sha256,
        factory_dependency_sha256=factory_dependency_hash(factory),
        compatible_models=compatible_models,
        configurable_fields=configurable_fields,
    )


def _query_component_specs() -> dict[str, QueryComponentSpec]:
    specs = (
        _query_component_spec(
            component_id="tabu.query.tokenizer",
            role=QueryComponentRole.TOKENIZER,
            implementation=QueryTokenizerAdapter,
            factory=_query_tokenizer_factory,
            configurable_fields=(
                "nominal_tokenizer",
                "nominal_codebook_size",
                "nominal_codebook_seed",
            ),
        ),
        _query_component_spec(
            component_id="tabu.query.axis_source",
            role=QueryComponentRole.AXIS_SOURCE,
            implementation=QueryAxisSourceAdapter,
            factory=_query_axis_source_factory,
            configurable_fields=(
                "column_source_policy",
                "row_source_policy",
                "receiver_only_origins",
            ),
        ),
        _query_component_spec(
            component_id="tabu.query.dynamics",
            role=QueryComponentRole.DYNAMICS,
            implementation=QueryDynamicsAdapter,
            factory=_query_dynamics_factory,
            configurable_fields=(),
        ),
        _query_component_spec(
            component_id="tabu.query.geometry.global_w",
            role=QueryComponentRole.GEOMETRY,
            implementation=QueryGeometryAdapter,
            factory=_query_geometry_factory,
            configurable_fields=(),
        ),
        _query_component_spec(
            component_id="tabu.query.terminal",
            role=QueryComponentRole.TERMINAL,
            implementation=QueryTerminalAdapter,
            factory=_query_terminal_factory,
            configurable_fields=("numeric_terminal",),
        ),
    )
    return {spec.spec_ref: spec for spec in specs}


def _query_row_component_specs() -> dict[str, QueryComponentSpec]:
    """Canonical components for ``tabu.query.row@0.2.0``.

    Tokenization, source policy and terminal semantics are shared with QueryBase,
    while dynamics and geometry receive distinct identities because the carrier
    and readout topology are different.
    """

    shared_models = ("tabu.query.base@0.1.0", "tabu.query.row@0.2.0")
    specs = (
        _query_component_spec(
            component_id="tabu.query.tokenizer",
            role=QueryComponentRole.TOKENIZER,
            implementation=QueryTokenizerAdapter,
            factory=_query_tokenizer_factory,
            configurable_fields=(
                "nominal_tokenizer",
                "nominal_codebook_size",
                "nominal_codebook_seed",
            ),
            compatible_models=shared_models,
        ),
        _query_component_spec(
            component_id="tabu.query.axis_source",
            role=QueryComponentRole.AXIS_SOURCE,
            implementation=QueryAxisSourceAdapter,
            factory=_query_axis_source_factory,
            configurable_fields=(
                "column_source_policy",
                "row_source_policy",
                "receiver_only_origins",
            ),
            compatible_models=shared_models,
        ),
        _query_component_spec(
            component_id="tabu.query.row.dynamics",
            role=QueryComponentRole.DYNAMICS,
            implementation=QueryRowDynamicsAdapter,
            factory=_query_row_dynamics_factory,
            configurable_fields=(),
            compatible_models=("tabu.query.row@0.2.0",),
            component_version="0.2.0",
        ),
        _query_component_spec(
            component_id="tabu.query.geometry.row_readout",
            role=QueryComponentRole.GEOMETRY,
            implementation=QueryRowGeometryAdapter,
            factory=_query_row_geometry_factory,
            configurable_fields=(
                "token_count",
                "row_readout_mode",
                "anchored_gamma_initial",
                "axis_transform_normalization",
            ),
            compatible_models=("tabu.query.row@0.2.0",),
            component_version="0.2.0",
        ),
        _query_component_spec(
            component_id="tabu.query.terminal",
            role=QueryComponentRole.TERMINAL,
            implementation=QueryTerminalAdapter,
            factory=_query_terminal_factory,
            configurable_fields=("numeric_terminal",),
            compatible_models=shared_models,
        ),
    )
    return {spec.spec_ref: spec for spec in specs}


def _ref(spec: QueryComponentSpec, config: Mapping[str, Any]) -> QueryComponentRef:
    spec.resolve_config(config)
    return QueryComponentRef(spec.component_id, spec.component_version, spec.role, dict(config))


def canonical_query_base_manifest(
    *,
    numeric_terminal: str = "local_linear",
    nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
) -> QueryComponentManifest:
    specs = _query_component_specs()
    return QueryComponentManifest(
        tokenizer=_ref(
            specs["tabu.query.tokenizer@0.1.0"],
            {
                "nominal_tokenizer": nominal_tokenizer,
                "nominal_codebook_size": nominal_codebook_size,
                "nominal_codebook_seed": nominal_codebook_seed,
            },
        ),
        axis_source=_ref(specs["tabu.query.axis_source@0.1.0"], {}),
        dynamics=_ref(specs["tabu.query.dynamics@0.1.0"], {}),
        geometry=_ref(specs["tabu.query.geometry.global_w@0.1.0"], {}),
        terminal=_ref(
            specs["tabu.query.terminal@0.1.0"],
            {"numeric_terminal": numeric_terminal},
        ),
    )


def canonical_query_row_manifest(
    *,
    token_count: int = 4,
    row_readout_mode: RowReadoutMode | str = RowReadoutMode.ANCHORED,
    anchored_gamma_initial: float = 1.0e-2,
    numeric_terminal: str = "local_linear",
    nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
    nominal_codebook_size: int = 100,
    nominal_codebook_seed: int = 1729,
) -> QueryComponentManifest:
    """Return the independent TabUR component manifest.

    ``token_count`` is deliberately explicit because it changes the augmented
    carrier, global-W row count, and coordinate width simultaneously.
    """

    if isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0:
        raise ValueError("token_count must be a positive integer")
    mode = RowReadoutMode(row_readout_mode)
    gamma_initial = float(anchored_gamma_initial)
    if not math.isfinite(gamma_initial):
        raise ValueError("anchored_gamma_initial must be finite")
    if mode is not RowReadoutMode.ANCHORED and gamma_initial != 1.0e-2:
        raise ValueError(
            "anchored_gamma_initial is fixed at 0.01 when row_readout_mode "
            "is homogeneous or free"
        )
    specs = _query_row_component_specs()
    return QueryComponentManifest(
        tokenizer=_ref(
            specs["tabu.query.tokenizer@0.1.0"],
            {
                "nominal_tokenizer": nominal_tokenizer,
                "nominal_codebook_size": nominal_codebook_size,
                "nominal_codebook_seed": nominal_codebook_seed,
            },
        ),
        axis_source=_ref(specs["tabu.query.axis_source@0.1.0"], {}),
        dynamics=_ref(specs["tabu.query.row.dynamics@0.2.0"], {}),
        geometry=_ref(
            specs["tabu.query.geometry.row_readout@0.2.0"],
            {
                "token_count": token_count,
                "row_readout_mode": mode.value,
                "anchored_gamma_initial": gamma_initial,
                "axis_transform_normalization": "exact_spectral_norm_v1",
            },
        ),
        terminal=_ref(
            specs["tabu.query.terminal@0.1.0"],
            {"numeric_terminal": numeric_terminal},
        ),
    )


def _build_canonical_query_registry() -> QueryComponentRegistry:
    specs = _query_component_specs()
    factories: dict[str, tuple[QueryComponentFactory, type[Any]]] = {
        "tabu.query.tokenizer@0.1.0": (_query_tokenizer_factory, QueryTokenizerAdapter),
        "tabu.query.axis_source@0.1.0": (_query_axis_source_factory, QueryAxisSourceAdapter),
        "tabu.query.dynamics@0.1.0": (_query_dynamics_factory, QueryDynamicsAdapter),
        "tabu.query.geometry.global_w@0.1.0": (_query_geometry_factory, QueryGeometryAdapter),
        "tabu.query.terminal@0.1.0": (_query_terminal_factory, QueryTerminalAdapter),
    }
    registry = QueryComponentRegistry()
    for spec_ref, spec in specs.items():
        factory, runtime_type = factories[spec_ref]
        registry._register_canonical(spec, factory, runtime_type)
    return registry


def _build_canonical_query_row_registry() -> QueryComponentRegistry:
    specs = _query_row_component_specs()
    factories: dict[str, tuple[QueryComponentFactory, type[Any]]] = {
        "tabu.query.tokenizer@0.1.0": (_query_tokenizer_factory, QueryTokenizerAdapter),
        "tabu.query.axis_source@0.1.0": (_query_axis_source_factory, QueryAxisSourceAdapter),
        "tabu.query.row.dynamics@0.2.0": (_query_row_dynamics_factory, QueryRowDynamicsAdapter),
        "tabu.query.geometry.row_readout@0.2.0": (
            _query_row_geometry_factory,
            QueryRowGeometryAdapter,
        ),
        "tabu.query.terminal@0.1.0": (_query_terminal_factory, QueryTerminalAdapter),
    }
    registry = QueryComponentRegistry()
    for spec_ref, spec in specs.items():
        factory, runtime_type = factories[spec_ref]
        registry._register_canonical(spec, factory, runtime_type)
    return registry


CANONICAL_QUERY_COMPONENTS = _build_canonical_query_registry()
CANONICAL_QUERY_ROW_COMPONENTS = _build_canonical_query_row_registry()


def _query_semantic_hash_payload(
    *,
    config: ReferenceConfig,
    profile: TabUQueryProfile,
    tokenizer_metadata: Mapping[str, Any],
    family_plan: QueryFamilyPlan,
    source_plan: AxisSourcePlan,
    numeric_terminal: str,
    ll_ridge: float | None,
    composition_hash: str,
    response_mechanism: str,
) -> dict[str, Any]:
    return {
        "reference_config": _reference_config_payload(config),
        "profile_id": profile.value,
        "tokenizer": dict(tokenizer_metadata),
        "family_plan": family_plan.as_dict(),
        "source_plan": source_plan.as_dict(),
        "response_mechanism": response_mechanism,
        "numeric_terminal": numeric_terminal,
        "ll_ridge": ll_ridge,
        "component_composition_hash": composition_hash,
    }


class QueryFamilyModelBase(DenseReferenceModel, ABC):
    """Identity-free family kernel; concrete models bind their own contract."""

    @abstractmethod
    def _build_family_plan(self) -> QueryFamilyPlan:
        """Bind a concrete family plan without inheriting model identity."""

        raise NotImplementedError

    def _query_component_manifest_identity(self) -> dict[str, Any]:
        composition = getattr(self, "component_composition", None)
        if composition is None:
            return {}
        return {
            "query_component_manifest_hash": composition.manifest.manifest_hash,
            "query_component_composition_hash": composition.composition_hash,
            "query_component_spec_hashes": dict(composition.component_spec_hashes),
        }


class TabUQueryBaseModel(QueryFamilyModelBase):
    """Axis-C Base: cell=query, both axes homogeneous, global $z=Wc$."""

    model_id = "tabu.query.base"

    def _build_family_plan(self) -> QueryFamilyPlan:
        return QueryFamilyPlan.base()

    @property
    def expected_geometry(self) -> str:
        return "global_W"

    @property
    def model_ref(self) -> str:
        return f"{self.model_id}@{self.contract_version}"

    def _prepare_dynamics(
        self,
        dynamics_input: Tensor,
        resolved: Any,
        context_mask: Tensor,
        row_source_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return dynamics_input, context_mask, row_source_mask

    def _geometry_trace(self) -> tuple[str, ...]:
        return ("global_W",)

    def _geometry_metadata(self) -> dict[str, Any]:
        return {}

    @property
    def unit_semantics(self) -> str:
        return "abstract_axis_roles"

    def _cell_states_for_readout(self, cells: Tensor) -> Tensor:
        return cells

    def _event_source_mask(self, resolved: Any, dynamics_input: Tensor) -> Tensor:
        del dynamics_input
        return resolved.visible_mask

    def _event_null_mask(self, resolved: Any, dynamics_input: Tensor) -> Tensor:
        del dynamics_input
        return resolved.natural_missing_mask

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        profile: TabUQueryProfile | str,
        numeric_terminal: str = "local_linear",
        label_broadcast: bool | None = None,
        label_broadcast_tau: float = 1.0e-6,
        nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
        component_manifest: QueryComponentManifest | None = None,
        component_registry: QueryComponentRegistry | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        super().__init__(config)
        self.profile = TabUQueryProfile(profile)
        expected_broadcast = self.profile.uses_label_broadcast
        if label_broadcast is not None and bool(label_broadcast) != expected_broadcast:
            raise ValueError("label_broadcast is derived from the explicit TabUQueryProfile")
        self.label_broadcast = expected_broadcast
        tau = float(label_broadcast_tau)
        if not math.isfinite(tau) or tau <= 0.0:
            raise ValueError("label_broadcast_tau must be a finite positive float")
        if not self.label_broadcast and tau != 1.0e-6:
            raise ValueError("completion.artificial_mask.v1 requires label_broadcast_tau=1e-6")
        self.label_broadcast_tau = tau
        self.family_plan = self._build_family_plan()
        self.source_plan = AxisSourcePlan()
        manifest = component_manifest or canonical_query_base_manifest(
            numeric_terminal=numeric_terminal,
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
        )
        registry = CANONICAL_QUERY_COMPONENTS if component_registry is None else component_registry
        if not isinstance(manifest, QueryComponentManifest):
            raise TypeError("component_manifest must be QueryComponentManifest")
        if not isinstance(registry, QueryComponentRegistry):
            raise TypeError("component_registry must be QueryComponentRegistry")
        self.component_manifest = manifest
        self.component_registry = registry
        self.component_composition = registry.resolve(manifest, model_ref=self.model_ref)
        self.tokenizer = registry.build(manifest.tokenizer, config=config)
        self.axis_source = registry.build(manifest.axis_source, config=config)
        self.dynamics = registry.build(manifest.dynamics, config=config)
        self.geometry = registry.build(manifest.geometry, config=config)
        self.terminal = registry.build(manifest.terminal, config=config)
        if self.axis_source.plan != self.source_plan:
            raise ValueError("QueryBase canonical source plan must remain fixed")
        if self.geometry.geometry != self.expected_geometry:
            raise ValueError(f"{self.model_ref} requires {self.expected_geometry} geometry")
        self.nominal_tokenizer = self.tokenizer.nominal_tokenizer
        self.nominal_codebook_size = self.tokenizer.nominal_codebook_size
        self.nominal_codebook_seed = self.tokenizer.nominal_codebook_seed
        tokenizer_version = (
            "query-tokenizer.v2"
            if self.nominal_tokenizer == CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2
            else "query-tokenizer.v1"
        )
        self.tokenizer_metadata = {
            "tokenizer_version": tokenizer_version,
            "feature_identity": "forbidden",
            "continuous_tokenizer": "context_only_standardization_then_shared_learnable_fourier",
            "nominal_tokenizer": self.nominal_tokenizer,
            "scale_epsilon": CellTokenizer._scale_epsilon,
        }
        if tokenizer_version == "query-tokenizer.v2":
            self.tokenizer_metadata.update(
                {
                    "nominal_codebook_size": self.nominal_codebook_size,
                    "nominal_codebook_seed": self.nominal_codebook_seed,
                    "nominal_codebook_hash": self.tokenizer.nominal_codebook_hash,
                    "nominal_codebook_scope": "source_codebook_id_and_domain_label",
                }
            )
        semantic_payload = _query_semantic_hash_payload(
            config=config,
            profile=self.profile,
            tokenizer_metadata=self.tokenizer_metadata,
            family_plan=self.family_plan,
            source_plan=self.source_plan,
            numeric_terminal=self.terminal.numeric_terminal,
            ll_ridge=self.terminal.ll_ridge,
            composition_hash=self.component_composition.composition_hash,
            response_mechanism=self.family_plan.response_mechanism,
        )
        semantic_config_hash = hashlib.sha256(
            json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.variant_ref = ModelVariantRef(
            contract_id=self.model_id,
            contract_version=self.contract_version,
            profile_id=self.profile.value,
            model_spec_hash=self.model_spec_hash,
            source_identity="unbound-local-source",
            semantic_config_hash=semantic_config_hash,
        )

    def _validate_profile_input(self, inputs: Any) -> None:
        natural_targets = inputs.natural_missing_mask & inputs.target_mask
        if bool((natural_targets & ~inputs.unsupported_target_mask).any()):
            raise ValueError("natural-missing targets must use the unsupported origin")
        if self.profile is TabUQueryProfile.COMPLETION_ARTIFICIAL_MASK_V1:
            if bool(inputs.query_target_mask.any()):
                raise ValueError("completion.artificial_mask.v1 rejects query target origins")
            response_count = sum(
                getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None))
                == "response"
                for spec in inputs.feature_specs
            )
            if response_count:
                raise ValueError("completion.artificial_mask.v1 requires zero response columns")
            return
        if bool(inputs.artificial_target_mask.any()):
            raise ValueError("supervised.label_broadcast.v1 rejects artificial-mask target origins")
        if not bool(inputs.query_target_mask.any()):
            raise ValueError("supervised.label_broadcast.v1 requires query targets")
        n_features = inputs.values.shape[2]
        response_flags = tuple(
            getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None)) == "response"
            for spec in inputs.feature_specs
        )
        if len(response_flags) != n_features or sum(response_flags) != 1:
            raise ValueError("supervised.label_broadcast.v1 requires exactly one response column")
        response_mask = torch.tensor(response_flags, dtype=torch.bool, device=inputs.values.device)
        if not torch.equal(inputs.query_target_mask.any(dim=(0, 1)), response_mask):
            raise ValueError("query targets must match the single declared response column")

    def _encode_dense_queries(
        self,
        inputs: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any, Tensor, Tensor, NumericScaleState]:
        resolved = self._resolve_inputs(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        self._validate_profile_input(resolved)
        context_mask, row_source_mask = self.axis_source.resolve(
            resolved,
            supervised=self.profile.uses_label_broadcast,
        )
        resolved = replace(
            resolved,
            metadata={
                **dict(resolved.metadata),
                "context_mask": context_mask,
                "profile_id": self.profile.value,
                "cell_role": "query",
                "query_axis_plan": self.family_plan.as_dict(),
                "query_axis_source_plan": self.source_plan.as_dict(),
            },
        )
        symbols = self.symbolizer(resolved)
        tokens = self.tokenizer(symbols)
        if tokens.numeric_scale_state is None:
            raise RuntimeError("TabUQueryBase tokenizer did not expose numeric scale state")
        dynamics_input = _label_broadcast(
            tokens.cells,
            resolved,
            enabled=self.label_broadcast,
            tau=self.label_broadcast_tau,
        )
        dynamics_input, context_mask, row_source_mask = self._prepare_dynamics(
            dynamics_input,
            resolved,
            context_mask,
            row_source_mask,
        )
        cells = self.dynamics(
            dynamics_input,
            column_source_mask=context_mask,
            row_source_mask=row_source_mask,
        )
        return resolved, symbols, dynamics_input, cells, tokens.numeric_scale_state

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        emit_trace = bool(kwargs.get("emit_trace", True))
        resolved, symbols, dynamics_input, cells, numeric_scale_state = self._encode_dense_queries(
            inputs,
            **kwargs,
        )
        coordinates = self.geometry(cells)
        readout = self.terminal(
            coordinates,
            numeric_scale_state.standardized_values,
            resolved.visible_mask,
        )
        readout = _apply_cell_null_contract(
            readout,
            coordinates,
            self._cell_states_for_readout(cells),
            null_mask=resolved.natural_missing_mask,
        )
        numeric_features = torch.tensor(
            tuple(kind == "numeric" for kind in symbols.feature_kinds),
            dtype=torch.bool,
            device=readout.values.device,
        ).view(1, 1, -1)
        numeric_raw_prediction = torch.where(
            readout.support_available & numeric_features & ~resolved.unsupported_target_mask,
            readout.values * numeric_scale_state.scale + numeric_scale_state.mean,
            torch.zeros_like(readout.values),
        )
        events = (
            (
                _shape_event(
                    "symbolizer",
                    symbols.values,
                    input_tensor=resolved.values,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    cell_role="query",
                ),
                _shape_event(
                    "tokenizer",
                    dynamics_input,
                    input_tensor=symbols.values,
                    source_mask=resolved.visible_mask,
                    null_mask=resolved.natural_missing_mask,
                    **self.tokenizer_metadata,
                    label_broadcast=self.label_broadcast,
                    label_broadcast_tau=self.label_broadcast_tau,
                    reference_config=_reference_config_payload(self.config),
                    terminal=self.terminal.numeric_terminal,
                    ll_ridge=self.terminal.ll_ridge,
                    bandwidth=self.config.routing_bandwidth,
                    cell_role="query",
                ),
                _shape_event(
                    "dynamics_plan",
                    cells,
                    input_tensor=dynamics_input,
                    source_mask=self._event_source_mask(resolved, dynamics_input),
                    null_mask=self._event_null_mask(resolved, dynamics_input),
                    operation_trace=self.dynamics.plan.stages,
                    plan=self._dynamics_plan_name(self.dynamics),
                    axis_source_plan=self.source_plan.as_dict(),
                ),
                _shape_event(
                    "readout",
                    readout.values,
                    input_tensor=coordinates,
                    source_mask=resolved.visible_mask,
                    operation_trace=(
                        *self._geometry_trace(),
                        f"numeric_{self.terminal.numeric_terminal_trace}",
                    ),
                    terminal=f"numeric_{self.terminal.numeric_terminal_trace}",
                    geometry=self.geometry.geometry,
                    response_mechanism=self.family_plan.response_mechanism,
                    numeric_prediction_scale="context_standardized",
                    **self._geometry_metadata(),
                ),
                _shape_event(
                    "prediction_boundary",
                    resolved.target_mask,
                    input_tensor=readout.values,
                    source_mask=resolved.visible_mask,
                    operation_trace=("model_forward_complete",),
                    supervision_boundary="sidecar_only",
                    truth_not_available=True,
                    model_forward_complete=True,
                    cell_role="query",
                ),
            )
            if emit_trace
            else ()
        )
        return self._bundle(
            inputs=resolved,
            values=readout.values,
            support_available=readout.support_available,
            coordinates=coordinates,
            routing_weights=readout.routing.weights,
            routing_log_weights=readout.routing.log_weights,
            routing_support_mask=readout.routing.support_mask,
            events=events,
            extra_auxiliaries={
                "numeric_raw_prediction": numeric_raw_prediction,
                "numeric_context_mean": numeric_scale_state.mean,
                "numeric_context_std": numeric_scale_state.std,
                "numeric_context_scale": numeric_scale_state.scale,
                "numeric_context_count": numeric_scale_state.context_count,
            },
            metadata={
                "dynamics_plan": self._dynamics_plan_name(self.dynamics),
                "family_id": "tabu.table_cell_as_query",
                "cell_role": "query",
                "unit_semantics": self.unit_semantics,
                "row_axis_mode": self.family_plan.row_axis.mode.value,
                "column_axis_mode": self.family_plan.column_axis.mode.value,
                "geometry": self.geometry.geometry,
                "response_mechanism": self.family_plan.response_mechanism,
                **self._geometry_metadata(),
                "axis_source_plan": self.source_plan.as_dict(),
                "numeric_terminal": self.terminal.numeric_terminal,
                "numeric_prediction_scale": "context_standardized",
                "profile_id": self.profile.value,
                "contract_version": self.variant_ref.contract_version,
                "variant_ref": self.variant_ref.as_dict(),
                "variant_hash": self.variant_ref.semantic_hash,
                **self._query_component_manifest_identity(),
                **self.tokenizer_metadata,
                "label_broadcast": self.label_broadcast,
                "label_broadcast_tau": self.label_broadcast_tau,
                "reference_config": _reference_config_payload(self.config),
                "terminal": self.terminal.numeric_terminal,
                "ll_ridge": self.terminal.ll_ridge,
                "bandwidth": self.config.routing_bandwidth,
                "query_marker": (
                    "supervised_target_origin"
                    if bool(resolved.query_target_mask.any())
                    else "absent"
                ),
            },
            emit_trace=emit_trace,
        )

    def checkpoint_identity(self) -> dict[str, Any]:
        identity = {
            "model_id": self.model_id,
            "contract_version": self.variant_ref.contract_version,
            "profile_id": self.profile.value,
            "family_plan": self.family_plan.as_dict(),
            "source_plan": self.source_plan.as_dict(),
            "cell_role": "query",
            "unit_semantics": self.unit_semantics,
            "geometry": self.geometry.geometry,
            "response_mechanism": self.family_plan.response_mechanism,
            "tokenizer_version": self.tokenizer_metadata["tokenizer_version"],
            "label_broadcast": self.label_broadcast,
            "label_broadcast_tau": self.label_broadcast_tau,
            "variant_hash": self.variant_ref.semantic_hash,
            "reference_config": _reference_config_payload(self.config),
            "terminal": self.terminal.numeric_terminal,
            "ll_ridge": self.terminal.ll_ridge,
            "bandwidth": self.config.routing_bandwidth,
            "variant_ref": self.variant_ref.as_dict(),
            **self._query_component_manifest_identity(),
        }
        if self.tokenizer_metadata["tokenizer_version"] == "query-tokenizer.v2":
            identity.update(
                {
                    "nominal_tokenizer": self.nominal_tokenizer,
                    "nominal_codebook_size": self.nominal_codebook_size,
                    "nominal_codebook_seed": self.nominal_codebook_seed,
                    "nominal_codebook_hash": self.tokenizer.nominal_codebook_hash,
                }
            )
        return identity

    def validate_checkpoint_identity(self, identity: Mapping[str, Any]) -> None:
        expected = self.checkpoint_identity()
        unexpected = sorted(set(identity) - set(expected))
        if unexpected:
            raise ValueError(f"checkpoint identity has unexpected fields: {unexpected}")
        for key, value in expected.items():
            # Checkpoint identities cross a JSON sidecar boundary.  JSON turns
            # tuple-valued topology fields (for example token banks and source
            # origins) into lists, while both representations have the same
            # contract meaning.  Compare through the repository canonical
            # encoder so serialization does not make a valid checkpoint fail
            # closed; semantic changes still produce a different hash.
            if canonical_hash(identity.get(key)) != canonical_hash(value):
                raise ValueError(f"checkpoint identity mismatch at {key}: expected {value!r}")

    def load_checkpoint_state(
        self,
        state_dict: Mapping[str, Tensor],
        identity: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> Any:
        """Validate semantic identity before allowing tensor loading."""

        self.validate_checkpoint_identity(identity)
        return self.load_state_dict(state_dict, strict=strict)


class TabUQueryRowModel(TabUQueryBaseModel):
    """Axis-C TabUR: heterogeneous row axis with explicit row-unit tokens."""

    model_id = "tabu.query.row"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        profile: TabUQueryProfile | str,
        row_token_count: int = 4,
        row_token_bank: tuple[str, ...] | None = None,
        row_readout_mode: RowReadoutMode | str = RowReadoutMode.ANCHORED,
        anchored_gamma_initial: float = 1.0e-2,
        numeric_terminal: str = "local_linear",
        label_broadcast: bool | None = None,
        label_broadcast_tau: float = 1.0e-6,
        nominal_tokenizer: str = CellTokenizer.EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
        component_manifest: QueryComponentManifest | None = None,
        component_registry: QueryComponentRegistry | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        if (
            isinstance(row_token_count, bool)
            or not isinstance(row_token_count, int)
            or row_token_count <= 0
        ):
            raise ValueError("row_token_count must be a positive integer")
        if row_token_count != config.matched_slots:
            raise ValueError(
                "TabUR requires K to equal row_token_count, W rows, and matched_slots"
            )
        bank = (
            tuple(row_token_bank)
            if row_token_bank is not None
            else tuple(f"row_unit_{index}" for index in range(row_token_count))
        )
        if len(bank) != row_token_count:
            raise ValueError("row_token_bank length must match row_token_count")
        self.row_token_count = row_token_count
        self.row_token_bank = bank
        manifest = component_manifest or canonical_query_row_manifest(
            token_count=row_token_count,
            row_readout_mode=row_readout_mode,
            anchored_gamma_initial=anchored_gamma_initial,
            numeric_terminal=numeric_terminal,
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
        )
        registry = (
            CANONICAL_QUERY_ROW_COMPONENTS
            if component_registry is None
            else component_registry
        )
        super().__init__(
            config,
            profile=profile,
            numeric_terminal=numeric_terminal,
            label_broadcast=label_broadcast,
            label_broadcast_tau=label_broadcast_tau,
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
            component_manifest=manifest,
            component_registry=registry,
        )
        geometry_token_count = getattr(self.geometry, "token_count", None)
        if geometry_token_count != self.row_token_count:
            raise ValueError("row geometry token_count must match row_token_count")
        if not isinstance(self.geometry, QueryRowGeometryAdapter):
            raise TypeError("TabUR geometry must be a QueryRowGeometryAdapter")
        self.row_readout_mode = self.geometry.row_readout_mode
        self.anchored_gamma_initial = self.geometry.anchored_gamma_initial
        self.row_unit_markers = nn.Parameter(
            torch.empty(
                self.row_token_count,
                self.config.d_model,
                dtype=DEFAULT_FLOAT_DTYPE,
            )
        )
        nn.init.normal_(self.row_unit_markers, std=0.02)

    def _build_family_plan(self) -> QueryFamilyPlan:
        return QueryFamilyPlan(
            row_axis=AxisRoleSpec(
                "row",
                AxisMode.HETEROGENEOUS,
                self.row_token_bank,
            ),
            response_mechanism="row_readout",
        )

    @property
    def expected_geometry(self) -> str:
        return "row_readout"

    @property
    def unit_semantics(self) -> str:
        return "abstract_axis_roles_with_row_unit_tokens"

    def _geometry_trace(self) -> tuple[str, ...]:
        return ("row_readout", f"row_readout_{self.row_readout_mode.value}")

    def _geometry_metadata(self) -> dict[str, Any]:
        return {"row_readout": self.geometry.trace_metadata()}

    def _prepare_dynamics(
        self,
        dynamics_input: Tensor,
        resolved: Any,
        context_mask: Tensor,
        row_source_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        del resolved
        batch, n_rows, _, d_model = dynamics_input.shape
        markers = self.row_unit_markers.to(
            device=dynamics_input.device,
            dtype=dynamics_input.dtype,
        ).view(1, 1, self.row_token_count, d_model)
        markers = markers.expand(batch, n_rows, -1, -1)
        false_tokens = torch.zeros(
            batch,
            n_rows,
            self.row_token_count,
            dtype=torch.bool,
            device=dynamics_input.device,
        )
        return (
            torch.cat((dynamics_input, markers), dim=2),
            torch.cat((context_mask, false_tokens), dim=2),
            torch.cat((row_source_mask, false_tokens), dim=2),
        )

    def _cell_states_for_readout(self, cells: Tensor) -> Tensor:
        if cells.shape[2] <= self.row_token_count:
            raise ValueError("row dynamics returned no ordinary cell states")
        return cells[:, :, :-self.row_token_count, :]

    def _event_source_mask(self, resolved: Any, dynamics_input: Tensor) -> Tensor:
        false_tokens = torch.zeros(
            *resolved.visible_mask.shape[:2],
            self.row_token_count,
            dtype=torch.bool,
            device=resolved.visible_mask.device,
        )
        return torch.cat((resolved.visible_mask, false_tokens), dim=2)

    def _event_null_mask(self, resolved: Any, dynamics_input: Tensor) -> Tensor:
        del dynamics_input
        false_tokens = torch.zeros(
            *resolved.natural_missing_mask.shape[:2],
            self.row_token_count,
            dtype=torch.bool,
            device=resolved.natural_missing_mask.device,
        )
        return torch.cat((resolved.natural_missing_mask, false_tokens), dim=2)

    def checkpoint_identity(self) -> dict[str, Any]:
        identity = super().checkpoint_identity()
        identity.update(
            {
                "row_token_count": self.row_token_count,
                "row_token_bank": self.row_token_bank,
                "row_readout": self.geometry.readout_identity(),
            }
        )
        return identity


__all__ = [
    "AxisMode",
    "RowReadoutMode",
    "AxisRoleSpec",
    "AxisSourcePlan",
    "CANONICAL_QUERY_COMPONENTS",
    "QueryAxisSourceAdapter",
    "QueryComponentManifest",
    "QueryComponentRef",
    "QueryComponentRegistry",
    "QueryComponentRole",
    "QueryComponentSpec",
    "QueryDynamicsAdapter",
    "QueryRowDynamicsAdapter",
    "QueryFamilyModelBase",
    "QueryFamilyPlan",
    "QueryGeometryAdapter",
    "QueryRowGeometryAdapter",
    "QueryTerminalAdapter",
    "QueryTokenizerAdapter",
    "ResolvedQueryComponentComposition",
    "TabUQueryBaseModel",
    "TabUQueryRowModel",
    "TabUQueryProfile",
    "canonical_query_base_manifest",
    "canonical_query_row_manifest",
    "CANONICAL_QUERY_ROW_COMPONENTS",
]
