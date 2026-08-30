"""Public TabUBase model and builder boundary."""

from .builders import (
    MODEL_BUILDERS,
    BuilderRegistry,
    build_from_spec,
    build_model,
    build_tabu_cell_base,
    register_model_builder,
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
    "MODEL_BUILDERS",
    "BuilderRegistry",
    "DenseModelInput",
    "DynamicsBlockKind",
    "LabelColumnBroadcast",
    "ModelVariantRef",
    "ReferenceConfig",
    "TabUCellBaseModel",
    "TabUCellBaseProfile",
    "build_from_spec",
    "build_model",
    "build_tabu_cell_base",
    "register_model_builder",
]
