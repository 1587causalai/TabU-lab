from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureSpec,
    ForwardRole,
    OriginState,
    PredictionStatus,
    TruthSidecar,
)
from tabu_lab.models import TabUV2CellAsQueryModel
from tabu_lab.models.types import DenseModelInput, ReferenceConfig
from tabu_lab.training import MixedObjective


def _config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=3,
        max_features=8,
    )


def _dense_input() -> DenseModelInput:
    values = torch.tensor(
        [[[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [0.0, 1.0]]]
    )
    visible = torch.tensor(
        [[[True, True], [True, True], [True, False], [False, True]]]
    )
    target = ~visible
    return DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        episode_id="tabu-v2-dense",
    )


def _episode() -> EvidenceEpisode:
    values = torch.tensor(
        [[1.0, 0.0], [2.0, 1.0], [3.0, 0.0], [0.0, 1.0]]
    )
    origins = []
    roles = []
    visible = [[True, True], [True, True], [True, False], [False, True]]
    for row in visible:
        origin_row = []
        role_row = []
        for is_visible in row:
            origin_row.append(
                OriginState.OBSERVED if is_visible else OriginState.ARTIFICIAL_MASK
            )
            role_row.append(
                ForwardRole.RECEIVER
                | (ForwardRole.SOURCE if is_visible else ForwardRole.TARGET)
            )
        origins.append(tuple(origin_row))
        roles.append(tuple(role_row))
    return EvidenceEpisode(
        episode_id="tabu-v2-episode",
        dataset_id="tabu-v2-fixture",
        source_partition="validation",
        fit_partition="train",
        row_ids=("r0", "r1", "r2", "r3"),
        feature_names=("numeric", "category"),
        forward_values=values,
        origin_states=tuple(origins),
        forward_roles=tuple(roles),
        feature_specs=(
            FeatureSpec(name="numeric", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("zero", "one", "two"),
                codebook_id="tabu-v2-category",
            ),
        ),
    )


def test_v2_compiles_one_extended_carrier_with_visible_only_sources() -> None:
    model = TabUV2CellAsQueryModel(_config())
    inputs = _dense_input()
    _, tokens, carrier, source_mask = model._compile_extended_carrier(inputs)
    batch, n_rows, n_features = inputs.values.shape
    k = model.k

    assert carrier.shape == (batch, n_rows + k, n_features + k, _config().d_model)
    assert int(source_mask.sum()) == int(inputs.visible_mask.sum())
    assert torch.equal(source_mask[:, :n_rows, :n_features], inputs.visible_mask)
    assert not bool(source_mask[:, n_rows:, :].any())
    assert not bool(source_mask[:, :, n_features:].any())
    assert torch.equal(
        carrier[:, n_rows:, n_features:],
        torch.zeros_like(carrier[:, n_rows:, n_features:]),
    )
    assert torch.equal(
        carrier[:, :n_rows, n_features:],
        model.unit_query.view(1, 1, k, -1).expand(batch, n_rows, -1, -1),
    )
    assert torch.equal(
        carrier[:, n_rows:, :n_features],
        model.feature_query.view(1, k, 1, -1).expand(batch, -1, n_features, -1),
    )
    assert bool(tokens.cells[~inputs.natural_missing_mask].abs().sum() > 0)


def test_v2_dynamics_keep_null_corner_exact_zero_and_emit_one_final_carrier() -> None:
    model = TabUV2CellAsQueryModel(_config())
    inputs = _dense_input()
    _, _, carrier_input, source_mask = model._compile_extended_carrier(inputs)
    final = model.dynamics(carrier_input, source_mask=source_mask)

    n_rows, n_features = inputs.values.shape[1:]
    assert final.shape == carrier_input.shape
    assert torch.equal(
        final[:, n_rows:, n_features:],
        torch.zeros_like(final[:, n_rows:, n_features:]),
    )
    assert torch.isfinite(final).all()

    prediction = model._forward_dense(inputs)
    assert prediction.model_id == "tabu.v2.tabur"
    assert prediction.metadata["carrier_shape"] == tuple(final.shape)
    assert prediction.metadata["dynamics_plan"] == "whole_table_column_then_row_omab"
    assert prediction.trace is not None
    assert [event.name for event in prediction.trace.events] == [
        "symbolizer",
        "tokenizer",
        "extended_carrier",
        "dynamics_plan",
        "response_field",
        "readout",
        "prediction_boundary",
    ]
    assert all(event.metadata["null_norm"] == 0.0 for event in prediction.trace.events)


def test_v2_public_forward_is_typed_and_truth_sidecar_only() -> None:
    torch.manual_seed(19)
    model = TabUV2CellAsQueryModel(_config()).eval()
    episode = _episode()
    prediction = model(episode)

    assert prediction.entries["numeric"].status is PredictionStatus.OK
    assert prediction.entries["categorical"].status is PredictionStatus.OK
    assert prediction.entries["distribution"].status is PredictionStatus.OK
    assert torch.isfinite(prediction.entries["numeric"].values).all()
    assert torch.isfinite(prediction.entries["distribution"].values).all()
    assert prediction.metadata["truth_boundary"] == "objective_sidecar_only"
    assert prediction.auxiliaries["artificial_target_mask"].any()

    truth_a = TruthSidecar(
        episode_id=episode.episode_id,
        recipe_hash="a" * 64,
        row_ids=episode.row_ids,
        feature_names=episode.feature_names,
        target_values=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [4.0, 0.0]]),
        target_mask=torch.tensor(
            [[False, False], [False, False], [False, True], [True, False]]
        ),
    )
    truth_b = TruthSidecar(
        episode_id=episode.episode_id,
        recipe_hash="b" * 64,
        row_ids=episode.row_ids,
        feature_names=episode.feature_names,
        target_values=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [9.0, 0.0]]),
        target_mask=truth_a.target_mask,
    )
    objective = MixedObjective(numeric_target_coordinate="context_standardized")
    first_loss = objective(prediction, truth_a)
    second_loss = objective(prediction, truth_b)
    assert first_loss.total.item() != second_loss.total.item()
    assert prediction.prediction_hash == model(episode).prediction_hash


def test_v2_context_terminal_uses_visible_context_without_truth_sidecar() -> None:
    pytest.importorskip("sklearn")
    model = TabUV2CellAsQueryModel(_config(), context_terminal="linear").eval()
    prediction = model(_episode())

    assert prediction.metadata["context_terminal"] == "linear"
    assert prediction.entries["distribution"].values is not None
    assert torch.isfinite(prediction.entries["distribution"].values).all()


def test_v2_training_objective_has_a_finite_backward_path() -> None:
    torch.manual_seed(23)
    model = TabUV2CellAsQueryModel(_config())
    prediction = model(_episode())
    truth = TruthSidecar(
        episode_id="tabu-v2-episode",
        recipe_hash="c" * 64,
        row_ids=("r0", "r1", "r2", "r3"),
        feature_names=("numeric", "category"),
        target_values=torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 2.0], [4.0, 0.0]]),
        target_mask=torch.tensor(
            [[False, False], [False, False], [False, True], [True, False]]
        ),
    )
    loss = MixedObjective(numeric_target_coordinate="context_standardized")(prediction, truth)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert model.response_base.grad is not None
    assert torch.isfinite(model.response_base.grad).all()
    assert model.tokenizer.query_token.grad is not None
    assert torch.isfinite(model.tokenizer.query_token.grad).all()


def test_v2_trace_identity_distinguishes_response_regimes() -> None:
    baseline = TabUV2CellAsQueryModel(_config())(_episode())
    adjusted = TabUV2CellAsQueryModel(_config(), lambda_F=1.0)(_episode())
    assert baseline.metadata["variant_hash"] != adjusted.metadata["variant_hash"]
    assert baseline.trace.model_hash != adjusted.trace.model_hash


def test_v2_context_override_preserves_other_features_and_domain_padding() -> None:
    pytest.importorskip("sklearn")
    episode = replace(
        _episode(),
        feature_specs=(
            FeatureSpec(
                name="numeric", kind=FeatureKind.CATEGORICAL,
                domain=("zero", "one", "two", "three"), codebook_id="predictor-domain",
            ),
            _episode().feature_specs[1],
        ),
    )
    canonical = TabUV2CellAsQueryModel(_config()).eval()
    adapter = TabUV2CellAsQueryModel(_config(), context_terminal="linear").eval()
    adapter.load_state_dict(canonical.state_dict())
    before = canonical(episode).entries["distribution"].values
    after = adapter(episode).entries["distribution"].values
    assert torch.equal(before[:, 0], after[:, 0])
    assert torch.equal(after[:, 1, 3], torch.zeros_like(after[:, 1, 3]))
    assert torch.allclose(after[2, 1].sum(), torch.tensor(1.0))


def test_v2_linear_context_accepts_single_observed_response_class() -> None:
    pytest.importorskip("sklearn")
    episode = _episode()
    values = episode.forward_values.clone()
    values[:, 1] = 1.0
    values[2, 1] = 0.0  # The held-out response remains physically absent from forward.
    prediction = TabUV2CellAsQueryModel(_config(), context_terminal="linear")(
        replace(episode, forward_values=values)
    )
    assert torch.equal(
        prediction.entries["distribution"].values[2, 1], torch.tensor([0.0, 1.0, 0.0])
    )


def test_v2_empty_same_column_support_is_typed_no_support() -> None:
    model = TabUV2CellAsQueryModel(_config())
    inputs = DenseModelInput(
        values=torch.zeros(1, 1, 1),
        visible_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        target_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        natural_missing_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        episode_id="tabu-v2-empty-support",
    )
    prediction = model._forward_dense(inputs)
    assert prediction.entries["numeric"].status is PredictionStatus.NO_SUPPORT
    assert prediction.entries["numeric"].values is None
    assert not prediction.auxiliaries["support_available"].any()


def test_v2_has_no_physical_row_or_column_parameter_copies() -> None:
    model = TabUV2CellAsQueryModel(_config())
    parameter_shapes = {
        name: tuple(value.shape) for name, value in model.named_parameters()
    }

    assert parameter_shapes["unit_query"] == (model.k, _config().d_model)
    assert parameter_shapes["feature_query"] == (model.k, _config().d_model)
    assert parameter_shapes["response_base"] == (model.k, _config().d_model)
    assert not any(
        token in name
        for name in parameter_shapes
        for token in (
            "row_embedding",
            "column_embedding",
            "row_address",
            "column_address",
        )
    )


def test_v2_row_and_column_permutation_equivariance() -> None:
    torch.manual_seed(31)
    model = TabUV2CellAsQueryModel(_config()).eval()
    inputs = _dense_input()
    row_order = torch.tensor([3, 1, 0, 2])
    column_order = torch.tensor([1, 0])

    def permute(value: torch.Tensor) -> torch.Tensor:
        return value[:, row_order][:, :, column_order]

    permuted = replace(
        inputs,
        values=permute(inputs.values),
        visible_mask=permute(inputs.visible_mask),
        target_mask=permute(inputs.target_mask),
        natural_missing_mask=permute(inputs.natural_missing_mask),
        artificial_target_mask=permute(inputs.artificial_target_mask),
        query_target_mask=permute(inputs.query_target_mask),
        unsupported_target_mask=permute(inputs.unsupported_target_mask),
        episode_id="tabu-v2-permuted",
    )
    original = model._forward_dense(inputs)
    moved = model._forward_dense(permuted)
    inverse_rows = torch.argsort(row_order)
    inverse_columns = torch.argsort(column_order)
    restored_values = moved.entries["numeric"].values[:, inverse_rows][
        :, :, inverse_columns
    ]
    restored_support = moved.auxiliaries["support_available"][:, inverse_rows][
        :, :, inverse_columns
    ]

    assert torch.allclose(
        restored_values,
        original.entries["numeric"].values,
        atol=1.0e-6,
    )
    assert torch.equal(restored_support, original.auxiliaries["support_available"])


def test_v2_response_regime_is_decoupled_after_the_shared_carrier() -> None:
    torch.manual_seed(37)
    shared = TabUV2CellAsQueryModel(_config()).eval()
    feature_conditioned = TabUV2CellAsQueryModel(_config(), lambda_F=0.5).eval()
    feature_conditioned.load_state_dict(shared.state_dict())
    inputs = _dense_input()

    shared_prediction = shared._forward_dense(inputs)
    feature_prediction = feature_conditioned._forward_dense(inputs)
    shared_events = {event.name: event for event in shared_prediction.trace.events}
    feature_events = {event.name: event for event in feature_prediction.trace.events}

    assert (
        shared_events["extended_carrier"].output_hash
        == feature_events["extended_carrier"].output_hash
    )
    assert (
        shared_events["dynamics_plan"].output_hash
        == feature_events["dynamics_plan"].output_hash
    )
    assert (
        shared_events["response_field"].output_hash
        != feature_events["response_field"].output_hash
    )
    assert shared_prediction.metadata["lambda_F"] == 0.0
    assert feature_prediction.metadata["lambda_F"] == 0.5
