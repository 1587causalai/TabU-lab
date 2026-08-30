from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tabu_lab.models import BuilderRegistry, build_model
from tabu_lab.models.types import DenseModelInput, DynamicsBlockKind, ReferenceConfig
from tabu_lab.verification import (
    SubstitutionStatus,
    assess_tabu_base_substitution,
    inspect_tabu_base_composition,
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
    values = torch.tensor([[[0.0, 2.0], [2.0, 4.0], [3.0, 7.0]]])
    visible = torch.tensor([[[False, True], [True, True], [True, True]]])
    target = ~visible
    return DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        episode_id="tabubase-stage2-composability",
    )


def _build(
    *,
    config: ReferenceConfig | None = None,
    nominal_tokenizer: str = "episode_random_sphere",
    numeric_terminal: str = "local_linear",
):
    torch.manual_seed(1729)
    return build_model(
        "tabu.cell.base",
        config=config or _config(),
        profile="completion.artificial_mask.v1",
        nominal_tokenizer=nominal_tokenizer,
        numeric_terminal=numeric_terminal,
    ).eval()


@pytest.mark.parametrize("axis", ["tokenizer", "dynamics", "readout"])
def test_one_component_axis_changes_independently(axis: str) -> None:
    reference = _build()
    if axis == "tokenizer":
        candidate = _build(nominal_tokenizer="source_scoped_frozen_codebook.v2")
    elif axis == "dynamics":
        candidate = _build(config=replace(_config(), block_kind=DynamicsBlockKind.MAB))
    else:
        candidate = _build(numeric_terminal="nadaraya_watson")
    candidate.load_state_dict(reference.state_dict())

    fixture = _fixture()
    with torch.no_grad():
        reference_prediction = reference._forward_dense(fixture)
        candidate_prediction = candidate._forward_dense(fixture)
    assessment = assess_tabu_base_substitution(
        reference_model=reference,
        candidate_model=candidate,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis=axis,
    )

    assert assessment.status is SubstitutionStatus.PASS
    assert assessment.changed_axes == (axis,)
    assert assessment.interface_stable
    assert assessment.variant_identity_changed
    assert len(assessment.assessment_hash) == 64


def test_substitution_gate_rejects_a_two_axis_change() -> None:
    reference = _build()
    candidate = _build(
        config=replace(_config(), block_kind=DynamicsBlockKind.MAB),
        numeric_terminal="nadaraya_watson",
    )
    candidate.load_state_dict(reference.state_dict())
    fixture = _fixture()
    with torch.no_grad():
        reference_prediction = reference._forward_dense(fixture)
        candidate_prediction = candidate._forward_dense(fixture)
    assessment = assess_tabu_base_substitution(
        reference_model=reference,
        candidate_model=candidate,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis="dynamics",
    )

    assert assessment.status is SubstitutionStatus.FAIL
    assert assessment.changed_axes == ("dynamics", "readout")
    assert assessment.interface_stable


def test_model_registry_can_grow_without_replacing_a_protected_anchor() -> None:
    anchor = object()
    extension = object()
    registry = BuilderRegistry(
        {"tabu.cell.base": lambda: anchor},
        protected_ids=frozenset({"tabu.cell.base"}),
    )
    registry.register("research.example", lambda: extension)

    assert registry.build("tabu.cell.base") is anchor
    assert registry.build("research.example") is extension
    assert registry.ids() == ("research.example", "tabu.cell.base")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("research.example", lambda: object())
    with pytest.raises(ValueError, match="cannot be replaced"):
        registry.register("tabu.cell.base", lambda: object(), replace=True)


def test_composition_inspector_rejects_non_tabubase_objects() -> None:
    with pytest.raises(TypeError, match="TabUCellBaseModel"):
        inspect_tabu_base_composition(object())  # type: ignore[arg-type]
