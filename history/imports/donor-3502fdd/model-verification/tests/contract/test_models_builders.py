from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    ForwardTrace,
    GraphTopology,
    OriginState,
    PredictionBundle,
    PredictionKind,
    PredictionStatus,
    TruthSidecar,
)
from tabu_lab.models import DesignOpenBuild, ReferenceConfig, TabUFModel, build_model
from tabu_lab.models.dynamics import GraphLocalBlock, PredictorUnitAddressDynamics
from tabu_lab.models.readouts import (
    AxisAddressBootstrap,
    MatchedUFReadout,
    PredictorOnlyLabelReadout,
    PredictorUnitLinkedLabelReadout,
    RowUnitReadout,
)
from tabu_lab.primitives import OMAB, OAttention, SameColumnNumericNW
from tabu_lab.training import Objective

BUILDABLE = (
    "tabu.unit_pair",
    "tabu.unit_row",
    "tabu4graph",
    "tabu4rec",
    "tabuf",
    "tabufl",
    "tabul",
)


def _config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=8,
    )


def test_rec_axis_address_ignores_masked_scalar_payloads() -> None:
    torch.manual_seed(1729)
    bootstrap = AxisAddressBootstrap(8, summary_dim=2, matched_residual_scale=0.1)
    matched = torch.randn(1, 3, 4, 2)
    visible = torch.tensor(
        [
            [
                [True, False, True, False],
                [False, True, True, False],
                [True, True, False, False],
            ]
        ]
    )
    values = torch.randn(1, 3, 4)
    mutated = values.clone()
    mutated[~visible] = 1.0e6

    first = bootstrap(matched, values, visible)
    second = bootstrap(matched, mutated, visible)

    for left, right in zip(first, second, strict=True):
        assert torch.equal(left, right)


def test_predictor_label_address_ignores_response_tokens() -> None:
    torch.manual_seed(1729)
    readout = PredictorOnlyLabelReadout(_config(), n_labels=2)
    typed = torch.randn(1, 4, 6, 8)
    visible_predictors = torch.ones(1, 4, 6, dtype=torch.bool)
    visible_predictors[:, :, 4:] = False
    mutated = typed.clone()
    mutated[:, :, 4:] = 1.0e6

    first = readout.coordinates(
        typed,
        visible_predictors,
        label_columns=(4, 5),
    )
    second = readout.coordinates(
        mutated,
        visible_predictors,
        label_columns=(4, 5),
    )

    assert torch.equal(first, second)


def test_predictor_unit_linked_label_path_excludes_response_tokens() -> None:
    torch.manual_seed(1729)
    config = _config()
    dynamics = PredictorUnitAddressDynamics(config)
    readout = PredictorUnitLinkedLabelReadout(config, n_labels=2)
    typed = torch.randn(1, 4, 6, 8)
    visible_predictors = torch.ones(1, 4, 6, dtype=torch.bool)
    visible_predictors[:, :, 4:] = False
    mutated = typed.clone()
    mutated[:, :, 4:] = 1.0e6

    first_units = dynamics(
        typed,
        visible_predictor_mask=visible_predictors,
    )
    second_units = dynamics(
        mutated,
        visible_predictor_mask=visible_predictors,
    )
    first = readout.coordinates(
        typed,
        visible_predictors,
        first_units,
        label_columns=(4, 5),
    )
    second = readout.coordinates(
        mutated,
        visible_predictors,
        second_units,
        label_columns=(4, 5),
    )

    assert torch.equal(first_units, second_units)
    assert torch.equal(first, second)


def _episode(model_id: str, *, episode_id: str = "episode-model") -> EvidenceEpisode:
    target_origin = (
        OriginState.QUERY if model_id in {"tabul", "tabufl"} else OriginState.ARTIFICIAL_MASK
    )
    values = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    origins = [[OriginState.OBSERVED] * 3 for _ in range(4)]
    roles = [[ForwardRole.RECEIVER | ForwardRole.SOURCE] * 3 for _ in range(4)]
    origins[0][2] = target_origin
    roles[0][2] = ForwardRole.RECEIVER | ForwardRole.TARGET
    feature_specs = (
        FeatureSpec(name="x0"),
        FeatureSpec(name="x1"),
        FeatureSpec(
            name="y",
            role=(
                FeatureRole.RESPONSE
                if model_id in {"tabufl", "tabul", "tabu4rec"}
                else FeatureRole.PREDICTOR
            ),
        ),
    )
    graph_topology = (
        GraphTopology(
            node_ids=("r0", "r1", "r2", "r3"),
            adjacency=torch.tensor(
                [
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, False, True],
                    [False, False, False, False],
                ]
            ),
        )
        if model_id == "tabu4graph"
        else None
    )
    return EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="dense-fixture",
        source_partition="train",
        fit_partition="train",
        row_ids=("r0", "r1", "r2", "r3"),
        feature_names=("x0", "x1", "y"),
        feature_specs=feature_specs,
        graph_topology=graph_topology,
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
    )


def _rec_episode(
    visible_mask: torch.Tensor,
    *,
    target: tuple[int, int] = (0, 0),
    natural_missing_target: bool = False,
    episode_id: str = "rec-response-family",
) -> EvidenceEpisode:
    """Build a Rec matrix whose item columns are one declared RESPONSE family."""

    if visible_mask.ndim != 2 or visible_mask.dtype is not torch.bool:
        raise ValueError("visible_mask must be a rank-2 bool tensor")
    n_rows, n_items = visible_mask.shape
    values = torch.arange(
        1,
        n_rows * n_items + 1,
        dtype=torch.float32,
    ).reshape(n_rows, n_items)
    origins = [[OriginState.NATURAL_MISSING] * n_items for _ in range(n_rows)]
    roles = [[ForwardRole.RECEIVER] * n_items for _ in range(n_rows)]
    for row, item in visible_mask.nonzero(as_tuple=False).tolist():
        origins[row][item] = OriginState.OBSERVED
        roles[row][item] = ForwardRole.RECEIVER | ForwardRole.SOURCE
    target_row, target_item = target
    origins[target_row][target_item] = (
        OriginState.NATURAL_MISSING if natural_missing_target else OriginState.ARTIFICIAL_MASK
    )
    roles[target_row][target_item] = ForwardRole.RECEIVER | ForwardRole.TARGET
    values = values.masked_fill(~visible_mask, 0.0)
    values[target_row, target_item] = 0.0
    feature_names = tuple(f"item-{item}" for item in range(n_items))
    return EvidenceEpisode(
        episode_id=episode_id,
        dataset_id="rec-matrix-fixture",
        source_partition="train",
        fit_partition="train",
        row_ids=tuple(f"user-{row}" for row in range(n_rows)),
        feature_names=feature_names,
        feature_specs=tuple(
            FeatureSpec(name=name, role=FeatureRole.RESPONSE) for name in feature_names
        ),
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
    )


@pytest.mark.parametrize("model_id", BUILDABLE)
def test_seven_buildable_contracts_instantiate_and_forward(model_id: str) -> None:
    kwargs = {"config": _config()}
    if model_id == "tabu4graph":
        kwargs["target_feature"] = 2
    model = build_model(model_id, **kwargs)
    prediction = model(_episode(model_id))

    assert isinstance(prediction, PredictionBundle)
    assert prediction.model_id == model_id
    assert isinstance(prediction.trace, ForwardTrace)
    assert prediction.outputs["numeric"].shape == (4, 3)
    entry = prediction.entries["numeric"]
    assert entry.kind is PredictionKind.NUMERIC
    assert entry.status is PredictionStatus.OK
    assert entry.support_ids.shape == entry.support_weights.shape
    assert prediction.outputs["abstention"].dtype is torch.bool
    expected_contract = "0.2.0" if model_id == "tabu4rec" else "0.1.0"
    assert prediction.contract_version == expected_contract
    assert prediction.metadata["contract_version"] == expected_contract
    assert prediction.metadata["prediction_schema_version"] == "tabu.prediction-bundle.v1"
    assert prediction.metadata["status"] == "ok"
    assert prediction.metadata["categorical"] == "not_declared"
    stages = tuple(event.name for event in prediction.trace.events)
    assert stages[0:2] == ("symbolizer", "tokenizer")
    assert "dynamics_plan" in stages
    assert stages[-2:] == ("readout", "prediction_boundary")
    assert prediction.trace.events[-1].metadata["truth_not_available"]
    assert prediction.trace.events[-1].metadata["model_forward_complete"]
    for event in prediction.trace.events:
        assert event.input_hash is not None and len(event.input_hash) == 64
        assert event.output_hash is not None and len(event.output_hash) == 64
        assert len(event.metadata["source_mask_hash"]) == 64
        assert isinstance(event.metadata["source_count"], int)
        assert isinstance(event.metadata["null_norm"], float)
        assert event.metadata["operation_trace"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this regression")
@pytest.mark.parametrize("model_id", BUILDABLE)
def test_public_forward_materializes_cpu_episode_on_model_device(model_id: str) -> None:
    kwargs = {"config": _config()}
    if model_id == "tabu4graph":
        kwargs["target_feature"] = 2
    model = build_model(model_id, **kwargs).eval().to("cuda")
    episode = _episode(model_id)
    episode_hash = episode.evidence_hash

    prediction = model(episode)

    assert episode.forward_values.device.type == "cpu"
    assert episode.evidence_hash == episode_hash
    for entry in prediction.entries.values():
        for tensor in (entry.values, entry.support_ids, entry.support_weights):
            if tensor is not None:
                assert tensor.device.type == "cuda"
                if tensor.is_floating_point():
                    assert tensor.dtype is torch.float32
    for tensor in prediction.auxiliaries.values():
        assert tensor.device.type == "cuda"
        if tensor.is_floating_point():
            assert tensor.dtype is torch.float32


def test_graph_trace_names_four_operations_and_global_readout() -> None:
    model = build_model("tabu4graph", config=_config(), target_feature=2)
    prediction = model(_episode("tabu4graph"))

    assert prediction.trace is not None
    operations = prediction.trace.metadata["graph_operations"]
    assert operations == (
        "target_feature_broadcast",
        "graph_local_unit_feature_evidence",
        "row_axis_feature_mix",
        "global_feature_prototype_for_readout",
    )
    assert prediction.trace.metadata["readout_path"] == "global_same_column_visible_support"
    assert prediction.trace.metadata["dynamics_plan"] == "graph_four_stage"


def test_graph_local_block_symmetrizes_neighborhood_and_zeros_empty_feature_slots() -> None:
    block = GraphLocalBlock(_config())
    cells = torch.zeros(1, 3, 2, 8)
    unit_tokens = torch.randn(1, 3, 2, 8)
    feature_tokens = torch.randn(1, 2, 2, 8)
    visible = torch.zeros(1, 3, 2, dtype=torch.bool)
    directed_graph = torch.tensor(
        [[False, True, False], [False, False, False], [False, False, False]]
    )

    seen: dict[str, torch.Tensor] = {}

    def capture_pair_mask(module, args, kwargs):
        seen["pair_mask"] = kwargs["pair_mask"].detach().clone()

    handle = block.graph_mab.register_forward_pre_hook(capture_pair_mask, with_kwargs=True)
    try:
        _, _, evolved_features = block(
            cells,
            unit_tokens,
            feature_tokens,
            visible_mask=visible,
            graph=directed_graph,
        )
    finally:
        handle.remove()

    pair_mask = seen["pair_mask"].reshape(1, 2, 3, 3)[0, 0]
    assert pair_mask[0, 1] and pair_mask[1, 0]
    assert pair_mask.diagonal().all()
    assert torch.equal(evolved_features, torch.zeros_like(evolved_features))


def test_matched_coordinates_use_common_float32_path() -> None:
    torch.manual_seed(29)
    units = torch.randn(1, 4, 2, 8)
    features = torch.randn(1, 3, 2, 8)
    raw = (units.float().unsqueeze(2) * features.float().unsqueeze(1)).sum(dim=-1)

    same_column = MatchedUFReadout(_config()).coordinates(units, features)
    bilinear = MatchedUFReadout(_config(), bilinear_support=True).coordinates(units, features)

    assert same_column.dtype is torch.float32
    assert bilinear.dtype is torch.float32
    assert torch.allclose(same_column, raw - raw.mean(dim=1, keepdim=True))
    assert torch.allclose(bilinear, raw - raw.mean(dim=(1, 2), keepdim=True))
    assert torch.allclose(
        same_column[:, 0] - same_column[:, 1],
        raw[:, 0] - raw[:, 1],
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_matched_coordinates_remove_large_common_gauge_before_float32_dot() -> None:
    config = replace(_config(), d_model=64)
    base = 1.0e4
    step = 2.0**-9
    offsets = torch.arange(4, dtype=torch.float32) * step
    units = torch.full((1, 4, 1, 64), base, dtype=torch.float32)
    features = torch.full((1, 4, 1, 64), base, dtype=torch.float32)
    units[0, :, 0, 0] += offsets

    same_column = MatchedUFReadout(config).coordinates(units, features)
    same_float64 = (
        units.double().unsqueeze(2) * features.double().unsqueeze(1)
    ).sum(dim=-1)
    same_expected = same_float64 - same_float64.mean(dim=1, keepdim=True)

    bilinear_features = features.clone()
    bilinear_features[0, :, 0, 1] += offsets
    bilinear = MatchedUFReadout(config, bilinear_support=True).coordinates(
        units,
        bilinear_features,
    )
    bilinear_float64 = (
        units.double().unsqueeze(2) * bilinear_features.double().unsqueeze(1)
    ).sum(dim=-1)
    bilinear_expected = bilinear_float64 - bilinear_float64.mean(
        dim=(1, 2),
        keepdim=True,
    )
    expected_distances = torch.tensor(
        (0.0, 381.4697265625, 1525.87890625, 3433.2275390625),
        dtype=torch.float32,
    )
    same_distances = (same_column[:, 0, 0] - same_column[:, :, 0]).square().sum(dim=-1)
    bilinear_distances = (bilinear[:, 0, 0] - bilinear[:, 0, :]).square().sum(dim=-1)

    assert same_column.dtype is torch.float32
    assert bilinear.dtype is torch.float32
    assert torch.allclose(
        same_column.double(),
        same_expected,
        rtol=1.0e-6,
        atol=1.0e-4,
    )
    assert torch.allclose(
        bilinear.double(),
        bilinear_expected,
        rtol=1.0e-6,
        atol=1.0e-4,
    )
    assert torch.allclose(same_distances[0], expected_distances)
    assert torch.allclose(bilinear_distances[0], expected_distances)


def test_public_model_builders_ignore_process_float64_default() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        models = tuple(build_model(model_id, config=_config()) for model_id in BUILDABLE)
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(previous)

    for model in models:
        floating = tuple(
            value
            for _, value in (*model.named_parameters(), *model.named_buffers())
            if value.is_floating_point()
        )
        assert floating
        assert all(value.dtype is torch.float32 for value in floating)


def test_direct_public_modules_ignore_process_float64_default() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        modules = (
            TabUFModel(_config()),
            OAttention(8, 2),
            OMAB(8, 2, 16),
            SameColumnNumericNW(),
            AxisAddressBootstrap(8),
        )
        assert torch.get_default_dtype() is torch.float64
    finally:
        torch.set_default_dtype(previous)

    for module in modules:
        floating = tuple(
            value
            for _, value in (*module.named_parameters(), *module.named_buffers())
            if value.is_floating_point()
        )
        assert floating
        assert all(value.dtype is torch.float32 for value in floating)


def test_matched_rms_unit_normalizes_tokens_at_router_boundary() -> None:
    torch.manual_seed(31)
    units = torch.randn(1, 4, 2, 8) * 37.0
    features = torch.randn(1, 3, 2, 8) * 0.03
    config = replace(_config(), geometry_normalization="rms_unit")

    normalized_units = torch.nn.functional.normalize(
        units.float(), p=2.0, dim=-1, eps=1.0e-6
    ) * math.sqrt(config.d_model)
    normalized_features = torch.nn.functional.normalize(
        features.float(), p=2.0, dim=-1, eps=1.0e-6
    ) * math.sqrt(config.d_model)
    raw = (normalized_units.unsqueeze(2) * normalized_features.unsqueeze(1)).sum(
        dim=-1
    ) / math.sqrt(config.d_model)

    actual = MatchedUFReadout(config).coordinates(units, features)

    assert torch.allclose(actual, raw - raw.mean(dim=1, keepdim=True))
    assert not torch.equal(
        actual,
        MatchedUFReadout(_config()).coordinates(units, features),
    )


def test_row_unit_rms_normalizes_unit_and_cell_tokens_at_readout() -> None:
    torch.manual_seed(37)
    config = replace(_config(), geometry_normalization="rms_unit")
    units = torch.randn(1, 3, 2, 8) * 19.0
    cells = torch.randn(1, 3, 4, 8) * 0.07
    values = torch.randn(1, 3, 4)
    visible = torch.ones(1, 3, 4, dtype=torch.bool)

    normalized_units = torch.nn.functional.normalize(
        units.float(), p=2.0, dim=-1, eps=1.0e-6
    ) * math.sqrt(config.d_model)
    normalized_cells = torch.nn.functional.normalize(
        cells.float(), p=2.0, dim=-1, eps=1.0e-6
    ) * math.sqrt(config.d_model)
    raw = (normalized_units.unsqueeze(2) * normalized_cells.unsqueeze(3)).sum(dim=-1) / math.sqrt(
        config.d_model
    )

    coordinates, _ = RowUnitReadout(config)(units, cells, values, visible)

    assert torch.allclose(coordinates, raw - raw.mean(dim=1, keepdim=True))


def test_unit_row_specials_are_receiver_only_on_row_axis() -> None:
    model = build_model("tabu.unit_row", config=_config())
    seen: dict[str, torch.Tensor] = {}

    def capture_row_sources(module, args, kwargs):
        seen["source_mask"] = kwargs["source_mask"].detach().clone()

    handle = model.dynamics.blocks[0].row_mab.register_forward_pre_hook(
        capture_row_sources,
        with_kwargs=True,
    )
    try:
        model(_episode("tabu.unit_row"))
    finally:
        handle.remove()

    row_sources = seen["source_mask"]
    assert not row_sources[:, -_config().matched_slots :].any()


def test_unit_pair_row_mates_exclude_receiver_self_edge() -> None:
    model = build_model("tabu.unit_pair", config=_config())
    seen: dict[str, torch.Tensor] = {}

    def capture_row_pair_mask(module, args, kwargs):
        seen["pair_mask"] = kwargs["pair_mask"].detach().clone()

    handle = model.dynamics.blocks[0].row_mab.register_forward_pre_hook(
        capture_row_pair_mask,
        with_kwargs=True,
    )
    try:
        model(_episode("tabu.unit_pair"))
    finally:
        handle.remove()

    pair_mask = seen["pair_mask"]
    assert not pair_mask.diagonal(dim1=-2, dim2=-1).any()
    assert pair_mask.sum(dim=-1).eq(pair_mask.shape[-1] - 1).all()


def test_tabu4do_builder_is_design_open_and_has_no_forward() -> None:
    result = build_model("tabu4do")

    assert isinstance(result, DesignOpenBuild)
    assert result.status == "design_open"
    assert not hasattr(result, "forward")
    assert "identification" in " ".join(result.open_questions)


def test_empty_same_column_support_is_explicit_abstention() -> None:
    episode = EvidenceEpisode(
        episode_id="empty-support",
        dataset_id="dense-fixture",
        source_partition="test",
        fit_partition="train",
        row_ids=("r0",),
        feature_names=("x",),
        forward_values=torch.zeros(1, 1),
        origin_states=((OriginState.ARTIFICIAL_MASK,),),
        forward_roles=((ForwardRole.RECEIVER | ForwardRole.TARGET,),),
    )
    prediction = build_model("tabuf", config=_config())(episode)

    assert prediction.metadata["status"] == "no_support"
    assert prediction.entries["numeric"].status is PredictionStatus.NO_SUPPORT
    assert prediction.entries["numeric"].values is None
    assert prediction.entries["numeric"].support_ids.numel() == 0
    assert prediction.entries["numeric"].support_weights.numel() == 0
    assert prediction.outputs["abstention"].item()
    assert not prediction.outputs["support_available"].item()


def test_public_forward_rejects_raw_tensor_and_dense_helper_is_private() -> None:
    model = build_model("tabuf", config=_config())
    raw = torch.zeros(2, 2)

    with pytest.raises(TypeError, match="EvidenceEpisode only"):
        model(raw)

    dense_prediction = model._forward_dense(
        raw,
        visible_mask=torch.ones_like(raw, dtype=torch.bool),
        target_mask=torch.zeros_like(raw, dtype=torch.bool),
    )
    assert isinstance(dense_prediction, PredictionBundle)


def test_graph_public_topology_changes_state_and_is_row_permutation_equivariant() -> None:
    torch.manual_seed(23)
    model = build_model("tabu4graph", config=_config(), target_feature=2).eval()
    base = _episode("tabu4graph", episode_id="graph-base")
    empty = replace(
        base,
        episode_id="graph-empty",
        graph_topology=GraphTopology(
            node_ids=base.row_ids,
            adjacency=torch.zeros(4, 4, dtype=torch.bool),
        ),
    )
    connected = model(base)
    disconnected = model(empty)
    connected_dynamics = next(
        event for event in connected.trace.events if event.name == "dynamics_plan"
    )
    disconnected_dynamics = next(
        event for event in disconnected.trace.events if event.name == "dynamics_plan"
    )
    assert connected.trace.metadata["raw_topology_hash"] == base.graph_topology.topology_hash
    assert connected_dynamics.output_hash != disconnected_dynamics.output_hash
    assert not torch.equal(connected.outputs["coordinates"], disconnected.outputs["coordinates"])

    permutation = torch.tensor([2, 0, 3, 1])
    permuted_ids = tuple(base.row_ids[index] for index in permutation.tolist())
    permuted = EvidenceEpisode(
        episode_id="graph-permuted",
        dataset_id=base.dataset_id,
        source_partition=base.source_partition,
        fit_partition=base.fit_partition,
        row_ids=permuted_ids,
        feature_names=base.feature_names,
        feature_specs=base.feature_specs,
        graph_topology=base.graph_topology.induced(permuted_ids),
        forward_values=base.forward_values[permutation],
        origin_states=base.origin_states[permutation],
        forward_roles=base.forward_roles[permutation],
    )
    permuted_prediction = model(permuted)
    assert torch.allclose(
        permuted_prediction.outputs["numeric"],
        connected.outputs["numeric"][permutation],
        atol=1.0e-6,
        rtol=1.0e-5,
    )


def test_graph_requires_typed_topology_and_single_tau_target_column() -> None:
    model = build_model("tabu4graph", config=_config(), target_feature=2)
    base = _episode("tabu4graph")
    with pytest.raises(ValueError, match="typed GraphTopology"):
        model(replace(base, graph_topology=None))

    origins = base.origin_states.clone()
    roles = base.forward_roles.clone()
    origins[1, 0] = origins[0, 2]
    roles[1, 0] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    values = base.forward_values.clone()
    values[1, 0] = 0.0
    outside_tau = replace(
        base,
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
    )
    with pytest.raises(ValueError, match="single tau column"):
        model(outside_tau)


def test_rec_uses_all_declared_response_columns_and_equal_active_arm_weights() -> None:
    # Historical dual-arm assertions exercise the explicit appendix plan;
    # the default mainline now uses parameterized matched scoring.
    model = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )
    episode = _rec_episode(
        torch.tensor(
            [
                [False, True, True],
                [True, False, False],
                [True, False, False],
            ]
        )
    )

    prediction = model(episode)

    arm_weights = prediction.outputs["rec_arm_weights"][0, 0]
    assert torch.allclose(arm_weights, arm_weights.new_tensor([0.5, 0.5]))
    assert prediction.outputs["rec_user_arm_support_available"][0, 0]
    assert prediction.outputs["rec_item_arm_support_available"][0, 0]
    user_weights = prediction.outputs["rec_user_arm_support_weights"][0, 0]
    item_weights = prediction.outputs["rec_item_arm_support_weights"][0, 0]
    assert user_weights[0] == 0
    assert bool((user_weights[1:] > 0).all())
    assert item_weights[0] == 0
    assert bool((item_weights[1:] > 0).all())
    assert torch.allclose(user_weights.sum(), user_weights.new_tensor(0.5))
    assert torch.allclose(item_weights.sum(), item_weights.new_tensor(0.5))
    assert prediction.trace.metadata["response_columns"] == (0, 1, 2)
    support_event = next(
        event for event in prediction.trace.events if event.name == "recommendation_support_ledger"
    )
    assert support_event.metadata["operation_trace"] == (
        "same_item_other_users",
        "same_user_other_response_columns",
        "equal_active_arm_mix",
        "single_active_arm_renormalizes_to_one",
    )


def test_rec_categorical_distribution_uses_the_same_two_response_arms() -> None:
    episode = _rec_episode(
        torch.tensor(
            [
                [False, True],
                [True, False],
                [True, False],
            ]
        ),
        episode_id="rec-categorical-response-family",
    )
    values = episode.forward_values.clone()
    values[0, 1] = 1.0
    values[1, 0] = 0.0
    values[2, 0] = 0.0
    categorical = replace(
        episode,
        feature_specs=tuple(
            FeatureSpec(
                name=name,
                kind=FeatureKind.CATEGORICAL,
                domain=("negative", "positive"),
                codebook_id="rec-binary-v1",
                role=FeatureRole.RESPONSE,
            )
            for name in episode.feature_names
        ),
        forward_values=values,
    )

    prediction = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )(categorical)

    assert torch.allclose(
        prediction.outputs["rec_arm_weights"][0, 0],
        prediction.outputs["rec_arm_weights"].new_tensor([0.5, 0.5]),
    )
    assert torch.allclose(
        prediction.entries["distribution"].values[0, 0],
        prediction.entries["distribution"].values.new_tensor([0.5, 0.5]),
    )
    assert torch.allclose(
        prediction.outputs["categorical_log_probabilities"][0, 0].exp(),
        prediction.outputs["categorical_log_probabilities"].new_tensor([0.5, 0.5]),
    )
    assert prediction.outputs["categorical_class_support_available"][0, 0].all()


@pytest.mark.parametrize(
    ("visible_mask", "expected_arm"),
    [
        (
            torch.tensor(
                [
                    [False, False, False],
                    [True, False, False],
                    [True, False, False],
                ]
            ),
            torch.tensor([1.0, 0.0]),
        ),
        (
            torch.tensor(
                [
                    [False, True, True],
                    [False, False, False],
                    [False, False, False],
                ]
            ),
            torch.tensor([0.0, 1.0]),
        ),
    ],
)
def test_rec_renormalizes_one_active_arm_to_one(
    visible_mask: torch.Tensor,
    expected_arm: torch.Tensor,
) -> None:
    prediction = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )(_rec_episode(visible_mask))

    assert prediction.entries["numeric"].status is PredictionStatus.OK
    actual = prediction.outputs["rec_arm_weights"][0, 0]
    assert torch.allclose(actual, expected_arm.to(actual.dtype))


def test_rec_returns_typed_no_support_when_both_arms_are_empty() -> None:
    episode = _rec_episode(
        torch.tensor(
            [
                [False, False, False],
                [False, True, False],
                [False, False, True],
            ]
        )
    )

    prediction = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )(episode)

    assert prediction.metadata["status"] == "no_support"
    assert prediction.entries["numeric"].status is PredictionStatus.NO_SUPPORT
    assert prediction.entries["numeric"].values is None
    assert prediction.entries["numeric"].support_weights.numel() == 0
    assert torch.equal(prediction.outputs["rec_arm_weights"][0, 0], torch.zeros(2))


def test_rec_single_response_is_explicitly_user_arm_only_not_two_arm() -> None:
    model = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )
    valid = _episode("tabu4rec")
    prediction = model(valid)
    assert torch.allclose(
        prediction.outputs["rec_arm_weights"][0, 2],
        prediction.outputs["rec_arm_weights"].new_tensor([1.0, 0.0]),
    )
    assert prediction.trace.metadata["response_columns"] == (2,)
    assert prediction.trace.metadata["support"] == "declared_response_bilinear_arms"

    all_predictors = replace(
        valid,
        feature_specs=tuple(FeatureSpec(name=name) for name in valid.feature_names),
    )
    with pytest.raises(ValueError, match="at least one schema-declared RESPONSE"):
        model(all_predictors)


def test_rec_rejects_natural_missing_as_a_completion_target() -> None:
    episode = _rec_episode(
        torch.tensor(
            [
                [False, True],
                [True, False],
            ]
        ),
        natural_missing_target=True,
    )

    with pytest.raises(ValueError, match="natural-missing cells cannot become targets"):
        build_model("tabu4rec", config=_config())(episode)


def test_tabufl_uses_separate_context_only_F_and_L_support_ledgers() -> None:
    values = torch.tensor(
        [
            [10.0, 20.0, 0.0],
            [0.0, 21.0, 30.0],
            [0.0, 22.0, 31.0],
            [13.0, 23.0, 32.0],
        ]
    )
    origins = [[OriginState.OBSERVED] * 3 for _ in range(4)]
    roles = [[ForwardRole.RECEIVER | ForwardRole.SOURCE] * 3 for _ in range(4)]
    origins[0][2] = OriginState.QUERY
    roles[0][2] = ForwardRole.RECEIVER | ForwardRole.TARGET
    for row in (1, 2):
        origins[row][0] = OriginState.ARTIFICIAL_MASK
        roles[row][0] = ForwardRole.RECEIVER | ForwardRole.TARGET
    episode = EvidenceEpisode(
        episode_id="tabufl-family-ledgers",
        dataset_id="dense-fixture",
        source_partition="train",
        fit_partition="train",
        row_ids=("r0", "r1", "r2", "r3"),
        feature_names=("x0", "x1", "y"),
        feature_specs=(
            FeatureSpec(name="x0"),
            FeatureSpec(name="x1"),
            FeatureSpec(name="y", role=FeatureRole.RESPONSE),
        ),
        forward_values=values,
        origin_states=origins,
        forward_roles=roles,
    )
    prediction = build_model("tabufl", config=_config())(episode)
    completion_weights = prediction.outputs["completion_support_weights"]
    label_weights = prediction.outputs["label_support_weights"]
    assert completion_weights[1, 0, 0] == 0
    assert completion_weights[1, 0, 3] > 0
    assert label_weights[0, 2, 0] == 0
    assert bool((label_weights[0, 2, 1:4] > 0).all())
    stages = tuple(event.name for event in prediction.trace.events)
    assert "completion_support_ledger" in stages
    assert "label_support_ledger" in stages

    truth_values = torch.zeros_like(values)
    truth_values[1, 0] = prediction.outputs["numeric"][1, 0] + 1.0
    truth_values[2, 0] = prediction.outputs["numeric"][2, 0] + 1.0
    truth_values[0, 2] = prediction.outputs["numeric"][0, 2] + 3.0
    truth = TruthSidecar(
        episode_id=episode.episode_id,
        recipe_hash="d" * 64,
        row_ids=episode.row_ids,
        feature_names=episode.feature_names,
        target_values=truth_values.detach(),
        target_mask=episode.target_mask,
    )
    loss = Objective()(prediction, truth)
    assert torch.isclose(loss.components["completion_loss"], torch.tensor(1.0))
    assert torch.isclose(loss.components["label_loss"], torch.tensor(9.0))
    assert torch.isclose(loss.total, torch.tensor(5.0))
    assert loss.metadata["active_families"] == ("F", "L")

    label_completion = replace(
        episode,
        episode_id="tabufl-label-completion",
        origin_states=(
            (OriginState.OBSERVED, OriginState.OBSERVED, OriginState.ARTIFICIAL_MASK),
            *tuple(tuple(OriginState.OBSERVED for _ in range(3)) for _ in range(3)),
        ),
        forward_roles=(
            (
                ForwardRole.RECEIVER | ForwardRole.SOURCE,
                ForwardRole.RECEIVER | ForwardRole.SOURCE,
                ForwardRole.RECEIVER | ForwardRole.TARGET,
            ),
            *tuple(
                tuple(ForwardRole.RECEIVER | ForwardRole.SOURCE for _ in range(3)) for _ in range(3)
            ),
        ),
        forward_values=torch.tensor(
            [
                [10.0, 20.0, 0.0],
                [11.0, 21.0, 30.0],
                [12.0, 22.0, 31.0],
                [13.0, 23.0, 32.0],
            ]
        ),
    )
    with pytest.raises(ValueError, match="reserves label columns"):
        build_model("tabufl", config=_config())(label_completion)
