from __future__ import annotations

from tabu_lab.models import DEFAULT_MODEL_ID, TabUV2CellAsQueryModel, build_model
from tabu_lab.models.types import ReferenceConfig
from tabu_lab.registry import (
    DEFAULT_MODEL_ID as REGISTRY_DEFAULT_MODEL_ID,
)
from tabu_lab.registry import (
    BuildStatus,
)
from tabu_lab.registry import (
    build_model as build_registered_model,
)


def test_model_factory_default_is_tabu_v2() -> None:
    assert DEFAULT_MODEL_ID == "tabu.v2.tabur"
    assert REGISTRY_DEFAULT_MODEL_ID == DEFAULT_MODEL_ID
    model = build_model(
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
            max_features=8,
        )
    )
    assert isinstance(model, TabUV2CellAsQueryModel)
    assert model.model_id == DEFAULT_MODEL_ID


def test_registry_default_is_tabu_v2_and_legacy_is_explicit() -> None:
    result = build_registered_model(
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
            max_features=8,
        )
    )
    assert result.status is BuildStatus.READY
    assert result.contract_id == DEFAULT_MODEL_ID
    assert isinstance(result.model, TabUV2CellAsQueryModel)

    legacy = build_model("tabu.query.row", profile="supervised.label_broadcast.v1")
    assert legacy.model_id == "tabu.query.row"
