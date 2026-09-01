from __future__ import annotations

import pytest
import torch

from tabu_lab.contracts import (
    PredictionBundle,
    PredictionEntry,
    PredictionKind,
    PredictionStatus,
    TruthSidecar,
)
from tabu_lab.training import Objective


def _prediction(
    *,
    numeric_prediction: float = 1.5,
    context_mean: float | None = 10.0,
    context_scale: float | None = 2.0,
    value_space: str = "context_standardized",
) -> PredictionBundle:
    target_mask = torch.tensor([[False], [True]])
    values = torch.tensor([[0.0], [numeric_prediction]])
    auxiliaries: dict[str, torch.Tensor] = {
        "target_mask": target_mask,
        "support_available": target_mask,
        "numeric_target_mask": target_mask,
        "numeric_support_available": target_mask,
        "completion_target_mask": torch.zeros_like(target_mask),
        "label_target_mask": target_mask,
    }
    if context_mean is not None:
        auxiliaries["numeric_context_mean"] = torch.full_like(values, context_mean)
    if context_scale is not None:
        auxiliaries["numeric_context_scale"] = torch.full_like(values, context_scale)
    return PredictionBundle(
        episode_id="coordinate-test",
        model_id="tabu.coordinate-test",
        entries={
            "numeric": PredictionEntry(
                kind=PredictionKind.NUMERIC,
                status=PredictionStatus.OK,
                values=values,
                metadata={"value_space": value_space},
            )
        },
        auxiliaries=auxiliaries,
    )


def _truth(raw_target: float) -> TruthSidecar:
    return TruthSidecar(
        episode_id="coordinate-test",
        recipe_hash="a" * 64,
        row_ids=("context", "query"),
        feature_names=("y",),
        target_values=torch.tensor([[0.0], [raw_target]]),
        target_mask=torch.tensor([[False], [True]]),
    )


def test_context_standardized_objective_projects_raw_truth_before_loss() -> None:
    loss = Objective(
        include_categorical=False,
        numeric_target_coordinate="context_standardized",
    )(_prediction(), _truth(14.0))

    # Raw y=14 is standardized by context mean=10 and scale=2 to y~=2.
    assert loss.components["mse"].item() == pytest.approx(0.25)
    assert loss.metadata["numeric_target_coordinate"] == "context_standardized"
    assert loss.total.item() == pytest.approx(0.25)


def test_context_standardized_objective_is_affine_invariant() -> None:
    objective = Objective(
        include_categorical=False,
        numeric_target_coordinate="context_standardized",
    )
    original = objective(_prediction(), _truth(14.0)).total
    affine = objective(
        _prediction(context_mean=35.0, context_scale=6.0),
        _truth(47.0),
    ).total

    torch.testing.assert_close(original, affine)


def test_legacy_raw_objective_semantics_are_unchanged() -> None:
    loss = Objective(include_categorical=False)(_prediction(), _truth(14.0))

    assert loss.components["mse"].item() == pytest.approx((1.5 - 14.0) ** 2)
    assert loss.metadata["numeric_target_coordinate"] == "raw"


@pytest.mark.parametrize(
    ("context_mean", "context_scale", "message"),
    (
        (None, 2.0, "numeric_context_mean"),
        (10.0, None, "numeric_context_scale"),
        (10.0, 0.0, "strictly positive"),
    ),
)
def test_context_standardized_objective_fails_closed_on_invalid_coordinate_state(
    context_mean: float | None,
    context_scale: float | None,
    message: str,
) -> None:
    objective = Objective(
        include_categorical=False,
        numeric_target_coordinate="context_standardized",
    )

    with pytest.raises(ValueError, match=message):
        objective(
            _prediction(context_mean=context_mean, context_scale=context_scale),
            _truth(14.0),
        )


def test_context_standardized_objective_requires_declared_prediction_space() -> None:
    objective = Objective(
        include_categorical=False,
        numeric_target_coordinate="context_standardized",
    )

    with pytest.raises(ValueError, match="declared in context_standardized"):
        objective(_prediction(value_space="raw"), _truth(14.0))
