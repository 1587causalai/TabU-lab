from __future__ import annotations

import pytest
import torch

from tabu_lab.contracts import FeatureRole, FeatureSpec
from tabu_lab.models import TabUCellBaseProfile, build_model
from tabu_lab.models.types import DenseModelInput


def _completion_input() -> DenseModelInput:
    values = torch.tensor([[[1.0, 0.0, 3.0], [2.0, 4.0, 0.0], [5.0, 6.0, 7.0]]])
    visible = values != 0
    target = ~visible
    return DenseModelInput(values, visible, target, torch.zeros_like(target))


def _supervised_input() -> DenseModelInput:
    values = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 1.0], [5.0, 6.0, 0.0]]])
    visible = torch.tensor([[[True, True, False], [True, True, True], [True, True, False]]])
    query = ~visible
    return DenseModelInput(
        values,
        visible,
        query,
        torch.zeros_like(query),
        artificial_target_mask=torch.zeros_like(query),
        query_target_mask=query,
        feature_specs=(
            FeatureSpec(name="x0"),
            FeatureSpec(name="x1"),
            FeatureSpec(name="y", role=FeatureRole.RESPONSE),
        ),
    )


def test_v020_profiles_are_explicit_and_identity_bound() -> None:
    completion = build_model(
        "tabu.cell.base", profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1
    )
    supervised = build_model(
        "tabu.cell.base", profile=TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
    )
    assert completion.profile is TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1
    assert supervised.profile is TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
    assert completion.variant_ref.contract_version == "0.2.0"
    assert completion.variant_ref.semantic_hash != supervised.variant_ref.semantic_hash
    identity = supervised.checkpoint_identity()
    assert identity["profile_id"] == "supervised.label_broadcast.v1"
    assert identity["tokenizer_version"] == "cell-tokenizer.v1"
    assert identity["label_broadcast"] is True
    assert identity["reference_config"]["block_kind"] == "omab"
    assert identity["terminal"] == supervised.readout.numeric_terminal
    assert identity["bandwidth"] == supervised.config.routing_bandwidth
    with pytest.raises(ValueError, match="derived"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            label_broadcast=True,
        )


def test_source_scoped_codebook_v2_has_distinct_checkpoint_identity() -> None:
    v1 = build_model("tabu.cell.base", profile="supervised.label_broadcast.v1")
    v2 = build_model(
        "tabu.cell.base",
        profile="supervised.label_broadcast.v1",
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        nominal_codebook_seed=1729,
    )
    v1_identity = v1.checkpoint_identity()
    v2_identity = v2.checkpoint_identity()
    assert v1_identity["tokenizer_version"] == "cell-tokenizer.v1"
    assert "nominal_codebook_hash" not in v1_identity
    assert v2_identity["tokenizer_version"] == "cell-tokenizer.v2"
    assert v2_identity["nominal_codebook_size"] == 100
    assert len(v2_identity["nominal_codebook_hash"]) == 64
    assert v1.variant_ref.semantic_hash != v2.variant_ref.semantic_hash
    with pytest.raises(ValueError, match="tokenizer_version"):
        v2.validate_checkpoint_identity(v1_identity)


def test_supervised_profile_has_one_response_and_no_self_support() -> None:
    model = build_model("tabu.cell.base", profile="supervised.label_broadcast.v1")
    prediction = model._forward_dense(_supervised_input())
    assert prediction.metadata["profile_id"] == "supervised.label_broadcast.v1"
    # Query row 0 is never allowed to support its own target.
    assert float(prediction.outputs["support_weights"][0, 0, 2, 2].detach()) == 0.0
    with pytest.raises(ValueError, match="exactly one response"):
        bad = _supervised_input()
        bad = DenseModelInput(
            bad.values,
            bad.visible_mask,
            bad.target_mask,
            bad.natural_missing_mask,
            artificial_target_mask=bad.artificial_target_mask,
            query_target_mask=bad.query_target_mask,
            feature_specs=(
                FeatureSpec(name="x0"),
                FeatureSpec(name="x1", role=FeatureRole.RESPONSE),
                FeatureSpec(name="y", role=FeatureRole.RESPONSE),
            ),
        )
        model._forward_dense(bad)


def test_completion_profile_trace_contains_profile_and_tokenizer_version() -> None:
    model = build_model("tabu.cell.base", profile="completion.artificial_mask.v1")
    prediction = model._forward_dense(_completion_input())
    assert prediction.metadata["profile_id"] == "completion.artificial_mask.v1"
    assert prediction.metadata["variant_ref"]["contract_version"] == "0.2.0"
    tokenizer = next(event for event in prediction.trace.events if event.name == "tokenizer")
    assert tokenizer.metadata["tokenizer_version"] == "cell-tokenizer.v1"
    assert tuple(event.name for event in prediction.trace.events) == (
        "symbolizer",
        "tokenizer",
        "dynamics_plan",
        "readout",
        "prediction_boundary",
    )
