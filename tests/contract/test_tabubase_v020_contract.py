from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tabu_lab.contracts import (
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    TruthSidecar,
    canonical_hash,
)
from tabu_lab.models import TabUCellBaseModel, TabUCellBaseProfile, build_model
from tabu_lab.models.components import CellTokenizer, Symbolizer
from tabu_lab.models.table_cell import _label_broadcast
from tabu_lab.models.types import DenseModelInput, DynamicsBlockKind, ReferenceConfig
from tabu_lab.registry import (
    BuildStatus,
    ModelVersionNotFoundError,
    get_model_spec,
    validate_registry_source_parity,
)
from tabu_lab.registry import build_model as build_contract


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


def _small_config(*, block_kind: DynamicsBlockKind = DynamicsBlockKind.OMAB) -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=8,
        block_kind=block_kind,
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
    nw = build_model(
        "tabu.cell.base",
        profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
        numeric_terminal="nadaraya_watson",
    )
    assert completion.variant_ref.semantic_hash != nw.variant_ref.semantic_hash
    with pytest.raises(ValueError, match="derived"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            label_broadcast=True,
        )
    with pytest.raises(ValueError, match="contract_version"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            variant_ref=replace(completion.variant_ref, contract_version="0.1.0"),
        )
    with pytest.raises(ValueError, match="unexpected fields"):
        completion.validate_checkpoint_identity({**completion.checkpoint_identity(), "extra": True})


def test_registry_build_is_typed_versioned_and_yaml_bound() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    result = build_contract(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )
    assert result.status is BuildStatus.READY
    assert result.executable
    assert isinstance(result.model, TabUCellBaseModel)
    assert result.model.model_spec_hash == canonical_hash(spec.model_dump(mode="json"))
    validate_registry_source_parity()
    with pytest.raises(ModelVersionNotFoundError):
        get_model_spec("tabu.cell.base", "0.1.0")


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
    dynamics = next(event for event in prediction.trace.events if event.name == "dynamics_plan")
    assert dynamics.metadata["shape"] == (1, 3, 3, model.config.d_model)
    assert prediction.metadata["unit"] == "cell"


def test_truth_free_query_marker_drives_label_broadcast_without_label_payload() -> None:
    inputs = _supervised_input()
    model = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    symbols = model.symbolizer(inputs)
    tokens = model.tokenizer(symbols)
    broadcast = _label_broadcast(tokens.cells, inputs, enabled=True)
    assert not torch.equal(broadcast[0, 0, 0], tokens.cells[0, 0, 0])
    assert torch.equal(broadcast[0, 0, 2], tokens.cells[0, 0, 2])
    assert inputs.values[0, 0, 2].item() == 0.0

    prediction = model._forward_dense(inputs)
    assert prediction.metadata["query_marker"] == "unified"
    assert prediction.auxiliaries["query_target_mask"].any()


def test_cell_tokenizer_has_typed_numeric_nominal_and_exact_null_branches() -> None:
    values = torch.tensor(
        [[[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [0.0, 0.0]]]
    )
    visible = torch.tensor(
        [[[True, True], [True, True], [True, True], [False, False]]]
    )
    natural = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=natural,
        natural_missing_mask=natural,
        artificial_target_mask=natural,
        query_target_mask=torch.zeros_like(natural),
        unsupported_target_mask=torch.zeros_like(natural),
        feature_specs=(
            FeatureSpec(name="n", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b", "c"),
                codebook_id="tabubase-tokenizer-contract",
            ),
        ),
        episode_id="tabubase-tokenizer-contract",
    )
    tokenizer = CellTokenizer(_small_config())
    symbols = Symbolizer()(inputs)
    first = tokenizer(symbols).cells
    second = tokenizer(symbols).cells
    assert torch.equal(first, second)
    assert torch.allclose(first[0, :3, 1].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.equal(first[0, 3], torch.zeros_like(first[0, 3]))
    assert not torch.equal(first[0, 0, 1], first[0, 1, 1])


def test_cell_base_keeps_numeric_and_nominal_predictions_typed() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    visible = torch.tensor([[[True, True], [True, False], [True, True]]])
    target = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        feature_specs=(
            FeatureSpec(name="n", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b"),
                codebook_id="tabubase-mixed-contract",
            ),
        ),
        episode_id="tabubase-mixed-contract",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=_small_config(),
    )._forward_dense(inputs)
    assert prediction.entries["numeric"].status.value == "ok"
    assert prediction.entries["categorical"].status.value == "ok"
    assert prediction.outputs["categorical"].shape == (1, 3, 2)
    assert torch.isfinite(prediction.auxiliaries["categorical_log_probabilities"]).all()


@pytest.mark.parametrize("block_kind", [DynamicsBlockKind.OMAB, DynamicsBlockKind.MAB])
def test_natural_missing_is_typed_no_support_for_both_dynamics_blocks(
    block_kind: DynamicsBlockKind,
) -> None:
    values = torch.tensor([[[0.0], [2.0]]])
    visible = torch.tensor([[[False], [True]]])
    target = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.tensor([[[True], [False]]]),
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        episode_id=f"tabubase-null-{block_kind.value}",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=_small_config(block_kind=block_kind),
    )._forward_dense(inputs)
    assert prediction.metadata["status"] == "no_support"
    assert prediction.entries["numeric"].status.value == "no_support"
    assert prediction.auxiliaries["support_available"][0, 0, 0].item() is False


def test_truth_enters_only_after_forward_and_hidden_payload_is_inert() -> None:
    visible = torch.tensor([[True, False], [True, True], [True, True]])
    target = ~visible
    common = {
        "visible_mask": visible,
        "target_mask": target,
        "natural_missing_mask": torch.zeros_like(target),
        "episode_id": "tabubase-truth-boundary",
    }
    clean = SimpleNamespace(
        forward_values=torch.tensor([[1.0, 0.0], [2.0, 4.0], [3.0, 7.0]]),
        **common,
    )
    poisoned = SimpleNamespace(
        forward_values=torch.tensor([[1.0, 999.0], [2.0, 4.0], [3.0, 7.0]]),
        **common,
    )
    clean_dense = DenseModelInput.from_any(clean)
    poisoned_dense = DenseModelInput.from_any(poisoned)
    assert torch.equal(clean_dense.values, poisoned_dense.values)

    torch.manual_seed(5)
    model = build_model("tabu.cell.base", config=_small_config()).eval()
    first = model._forward_dense(clean_dense)
    second = model._forward_dense(poisoned_dense)
    assert torch.equal(first.outputs["numeric"], second.outputs["numeric"])
    assert first.prediction_hash == second.prediction_hash
    boundary = next(event for event in first.trace.events if event.name == "prediction_boundary")
    assert boundary.metadata["truth_not_available"] is True
    assert boundary.metadata["supervision_boundary"] == "sidecar_only"

    def sidecar(value: float) -> TruthSidecar:
        truth_values = torch.zeros(3, 2)
        truth_values[0, 1] = value
        return TruthSidecar(
            episode_id="tabubase-truth-boundary",
            recipe_hash="a" * 64,
            row_ids=("r0", "r1", "r2"),
            feature_names=("x", "y"),
            target_values=truth_values,
            target_mask=target,
        )

    def numeric_loss(truth: TruthSidecar) -> torch.Tensor:
        prediction = first.outputs["numeric"]
        residual = prediction[truth.target_mask] - truth.target_values[truth.target_mask]
        return residual.square().mean()

    assert numeric_loss(sidecar(5.0)).item() != numeric_loss(sidecar(50.0)).item()
    assert first.prediction_hash == second.prediction_hash


def test_fixed_seed_checkpoint_roundtrip_is_deterministic() -> None:
    torch.manual_seed(1729)
    first = build_model("tabu.cell.base", config=_small_config()).eval()
    torch.manual_seed(1729)
    second = build_model("tabu.cell.base", config=_small_config()).eval()
    second.load_state_dict(first.state_dict())

    fixture = _completion_input()
    with torch.no_grad():
        left = first._forward_dense(fixture)
        right = second._forward_dense(fixture)
    assert left.outputs.keys() == right.outputs.keys()
    for name in left.outputs:
        assert torch.equal(left.outputs[name], right.outputs[name]), name
    assert left.prediction_hash == right.prediction_hash
    assert first.checkpoint_identity() == second.checkpoint_identity()

    supervised = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    assert first.variant_ref.semantic_hash != supervised.variant_ref.semantic_hash
