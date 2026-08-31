from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tabu_lab.contracts import FeatureKind, FeatureRole, FeatureSpec
from tabu_lab.models import SupervisedResponseAdapter
from tabu_lab.models.table_cell import LabelColumnBroadcast


def _evidence() -> SimpleNamespace:
    visible = torch.tensor([[[True, True], [True, False]]])
    query = torch.tensor([[[False, False], [False, True]]])
    return SimpleNamespace(
        feature_specs=(
            FeatureSpec(
                name="x",
                kind=FeatureKind.NUMERIC,
                role=FeatureRole.PREDICTOR,
            ),
            FeatureSpec(
                name="y",
                kind=FeatureKind.NUMERIC,
                role=FeatureRole.RESPONSE,
            ),
        ),
        visible_mask=visible,
        query_target_mask=query,
        target_mask=query,
        natural_missing_mask=torch.zeros_like(query),
    )


def _cells() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[0.25, 0.50], [1.00, 2.00]],
                [[0.75, 1.00], [3.00, 4.00]],
            ]
        ],
        dtype=torch.float32,
    )


def test_zero_gate_exactly_degenerates_to_cell_restoration_core() -> None:
    cells = _cells()
    adapter = SupervisedResponseAdapter(
        residual_gate_initial=0.0,
        trainable_gate=False,
    )
    assert torch.equal(adapter(cells, _evidence()), cells)


def test_unit_gate_matches_the_existing_label_broadcast_route() -> None:
    cells = _cells()
    expected = LabelColumnBroadcast()(cells, _evidence())
    adapter = SupervisedResponseAdapter(
        residual_gate_initial=1.0,
        trainable_gate=False,
    )
    assert torch.allclose(adapter(cells, _evidence()), expected)


def test_gate_interpolates_without_adding_a_second_readout() -> None:
    cells = _cells()
    proposed = LabelColumnBroadcast()(cells, _evidence())
    adapter = SupervisedResponseAdapter(
        residual_gate_initial=0.25,
        trainable_gate=False,
    )
    assert torch.allclose(
        adapter(cells, _evidence()),
        cells + 0.25 * (proposed - cells),
    )


def test_trainable_gate_has_finite_gradient_and_identity() -> None:
    cells = _cells()
    adapter = SupervisedResponseAdapter(residual_gate_initial=1.0e-2)
    loss = adapter(cells, _evidence()).square().mean()
    loss.backward()
    assert adapter.residual_gate.grad is not None
    assert bool(torch.isfinite(adapter.residual_gate.grad))
    identity = adapter.identity_payload()
    assert identity["spec_ref"] == "tabu.query.adapter.supervised_response@0.1.0"
    assert identity["target_origin"] == "query"
    assert identity["truth_visible"] is False
    assert len(identity["composition_hash"]) == 64


def test_adapter_rejects_nonfinite_or_negative_initial_gate() -> None:
    with pytest.raises(ValueError, match="finite"):
        SupervisedResponseAdapter(residual_gate_initial=float("nan"))
    with pytest.raises(ValueError, match="non-negative"):
        SupervisedResponseAdapter(residual_gate_initial=-0.1)
