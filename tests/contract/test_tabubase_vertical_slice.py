from __future__ import annotations

import torch

from tabu_lab.contracts import FeatureKind, FeatureSpec, canonical_hash
from tabu_lab.models import build_from_spec
from tabu_lab.models.types import DenseModelInput, DynamicsBlockKind, ReferenceConfig
from tabu_lab.primitives import SameColumnNumericNW
from tabu_lab.registry import get_model_spec, model_spec_identity_payload
from tabu_lab.verification import (
    TabUBaseVerificationStage,
    TabUBaseVerificationStatus,
    inspect_tabu_base_composition,
    verify_tabu_base_component_correctness,
    verify_tabu_base_component_evolvability,
)


def _config(*, block_kind: DynamicsBlockKind = DynamicsBlockKind.OMAB) -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=4,
        block_kind=block_kind,
    )


def _fixture() -> DenseModelInput:
    values = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [3.0, 0.0]]])
    visible = torch.tensor([[[False, True], [True, True], [True, True]]])
    return DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=~visible,
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(
            FeatureSpec(name="numeric"),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("red", "blue"),
                codebook_id="vertical-slice.colors.v1",
            ),
        ),
        episode_id="tabubase-vertical-slice",
    )


def _build(**kwargs):
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    torch.manual_seed(1729)
    return build_from_spec(
        spec,
        config=kwargs.pop("config", _config()),
        profile="completion.artificial_mask.v1",
        **kwargs,
    )


def test_model_spec_builder_and_components_share_one_identity() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    model = _build()
    composition = inspect_tabu_base_composition(model)
    evidence = verify_tabu_base_component_correctness(model)

    assert composition.model_spec_hash == canonical_hash(model_spec_identity_payload(spec))
    assert composition.unit_semantics == "table_cell_as_unit"
    assert composition.tokenizer == "cell-tokenizer.v1"
    assert composition.dynamics == "cell_unit_three_omab"
    assert composition.readout == "same_column.local_linear"
    assert composition.truth_boundary == "loss_sidecar_step_5_only"
    assert evidence.stage is TabUBaseVerificationStage.COMPONENT_CORRECTNESS
    assert evidence.status is TabUBaseVerificationStatus.PASS
    assert evidence.evidence_status == "local_unissued"


def test_one_axis_substitution_produces_bounded_local_evidence() -> None:
    reference = _build()
    candidate = _build(numeric_terminal="nadaraya_watson")
    candidate.load_state_dict(reference.state_dict())
    fixture = _fixture()
    with torch.no_grad():
        reference_prediction = reference._forward_dense(fixture)
        candidate_prediction = candidate._forward_dense(fixture)

    evidence = verify_tabu_base_component_evolvability(
        reference_model=reference,
        candidate_model=candidate,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis="readout",
    )

    assert evidence.stage is TabUBaseVerificationStage.COMPONENT_EVOLVABILITY
    assert evidence.status is TabUBaseVerificationStatus.PASS
    assert evidence.reference_composition_hash != evidence.candidate_composition_hash
    assert evidence.evidence_status == "local_unissued"


def test_two_axis_substitution_stays_a_failed_local_result() -> None:
    reference = _build()
    candidate = _build(
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        numeric_terminal="nadaraya_watson",
    )
    candidate.load_state_dict(reference.state_dict())
    fixture = _fixture()
    with torch.no_grad():
        reference_prediction = reference._forward_dense(fixture)
        candidate_prediction = candidate._forward_dense(fixture)

    evidence = verify_tabu_base_component_evolvability(
        reference_model=reference,
        candidate_model=candidate,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis="readout",
    )

    assert evidence.status is TabUBaseVerificationStatus.FAIL
    assert not next(
        check.passed for check in evidence.checks if check.check_id == "one_declared_axis_changed"
    )


def test_component_binding_rejects_a_terminal_label_class_mismatch() -> None:
    model = _build()
    model.readout.terminal = SameColumnNumericNW(model.config.routing_bandwidth)

    try:
        inspect_tabu_base_composition(model)
    except TypeError as error:
        assert "terminal label" in str(error)
    else:
        raise AssertionError("mismatched terminal class must fail closed")
