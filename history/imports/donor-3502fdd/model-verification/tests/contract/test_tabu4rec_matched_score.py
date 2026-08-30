from __future__ import annotations

import torch

from tabu_lab.experiments.fixtures import build_f0_fixture
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.models.readouts import MatchedScoreReadout


def test_matched_score_is_literal_diagonal_inner_product() -> None:
    config = ReferenceConfig(d_model=8, n_heads=2, matched_slots=3)
    readout = MatchedScoreReadout(config)
    units = torch.randn(1, 2, 3, 8)
    features = torch.randn(1, 4, 3, 8)

    coordinates, output = readout(
        units,
        features,
        support_values=torch.zeros(1, 2, 4),
        visible_mask=torch.zeros(1, 2, 4, dtype=torch.bool),
    )

    expected_coordinates = torch.einsum("bnkd,bmkd->bnmk", units, features)
    assert torch.allclose(coordinates, expected_coordinates)
    assert torch.allclose(output.values, expected_coordinates.sum(dim=-1))
    assert bool(output.support_available.all())
    assert output.routing.weights.shape[-1] == 0


def test_default_tabu4rec_has_no_empirical_arm_readout() -> None:
    model = build_model("tabu4rec", config=ReferenceConfig())
    fixture = build_f0_fixture("tabu4rec")

    prediction = model(fixture.evidence)

    assert isinstance(model.readout, MatchedScoreReadout)
    assert prediction.metadata["numeric_terminal"] == "parameterized_matching"
    assert prediction.metadata["support"] == "parameterized_matching"
    assert bool(prediction.auxiliaries["support_available"].all())
    assert "rec_user_arm_support_weights" not in prediction.auxiliaries
    assert "rec_item_arm_support_weights" not in prediction.auxiliaries
    matched_event = next(
        event for event in prediction.trace.events if event.name == "recommendation_matched_readout"
    )
    assert matched_event.metadata["operation_trace"] == (
        "matched_special_inner_products",
        "sum_matched_coordinates",
    )
