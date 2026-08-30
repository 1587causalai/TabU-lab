"""Public TabUBase model and builder boundary."""

from .builders import (
    MODEL_BUILDERS,
    BuilderRegistry,
    build_from_spec,
    build_model,
    build_tabu_cell_base,
    register_model_builder,
)
from .component_contract import TabUBaseComposition
from .component_registry import (
    CANONICAL_COMPONENTS,
    ComponentMaturity,
    ComponentRef,
    ComponentRegistry,
    ComponentRole,
    ComponentSpec,
    ResolvedComponentComposition,
    TabUBaseComponentManifest,
    canonical_tabu_base_manifest,
    factory_dependency_hash,
    implementation_source_identity,
)
from .table_cell import LabelColumnBroadcast, TabUCellBaseModel
from .types import (
    DenseModelInput,
    DynamicsBlockKind,
    ModelVariantRef,
    ReferenceConfig,
    TabUCellBaseProfile,
)

__all__ = [
    "CANONICAL_COMPONENTS",
    "MODEL_BUILDERS",
    "BuilderRegistry",
    "ComponentMaturity",
    "ComponentRef",
    "ComponentRegistry",
    "ComponentRole",
    "ComponentSpec",
    "DenseModelInput",
    "DynamicsBlockKind",
    "LabelColumnBroadcast",
    "ModelVariantRef",
    "ReferenceConfig",
    "ResolvedComponentComposition",
    "TabUBaseComponentManifest",
    "TabUBaseComposition",
    "TabUCellBaseModel",
    "TabUCellBaseProfile",
    "build_from_spec",
    "build_model",
    "build_tabu_cell_base",
    "canonical_tabu_base_manifest",
    "factory_dependency_hash",
    "implementation_source_identity",
    "register_model_builder",
]
