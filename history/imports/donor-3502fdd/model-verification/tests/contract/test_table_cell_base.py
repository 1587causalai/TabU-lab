from __future__ import annotations

import pytest
import torch

from tabu_lab.contracts import FeatureKind, FeatureSpec
from tabu_lab.models import (
    TabUCellBaseModel,
    TabUCellColumnModel,
    TabUCellRowColumnModel,
    TabUCellRowModel,
    build_model,
)
from tabu_lab.models.components import CellTokenizer, Symbolizer
from tabu_lab.models.readouts import CellMatchingReadout, CellSpecialReadout
from tabu_lab.models.table_cell import _label_broadcast
from tabu_lab.models.types import DenseModelInput, DynamicsBlockKind, ReferenceConfig
from tabu_lab.registry import BuildStatus
from tabu_lab.registry import build_model as build_contract


def _dense_input(*, query: bool = False) -> DenseModelInput:
    values = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [5.0, 6.0]]])
    visible = torch.tensor([[[True, True], [True, False], [True, True]]])
    target = ~visible
    query_mask = torch.zeros_like(target)
    if query:
        query_mask[0, 1, 1] = True
        target[0, 1, 1] = True
    artificial = target & ~query_mask
    return DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        artificial_target_mask=artificial,
        query_target_mask=query_mask,
        unsupported_target_mask=torch.zeros_like(target),
    )


def test_table_cell_contracts_are_distinct_and_expose_the_frozen_experimental_slice() -> None:
    base_result = build_contract("tabu.cell.base")
    assert base_result.status is BuildStatus.READY
    assert base_result.executable
    for contract_id in (
        "tabu.cell.row",
        "tabu.cell.column",
        "tabu.cell.row_column",
    ):
        result = build_contract(contract_id)
        assert result.status is BuildStatus.READY
        assert result.executable
        assert result.model is not None
    rec_result = build_contract("tabu.cell.rec")
    assert rec_result.status is BuildStatus.DESIGN_OPEN
    assert not rec_result.executable
    assert rec_result.model is None


def test_table_cell_base_uses_own_identity_and_artificial_completion_profile() -> None:
    model = build_model("tabu.cell.base")
    assert isinstance(model, TabUCellBaseModel)
    prediction = model._forward_dense(_dense_input())
    assert prediction.model_id == "tabu.cell.base"
    assert prediction.metadata["family_id"] == "tabu.table_cell_as_unit"
    assert prediction.metadata["profile_id"] == "completion.artificial_mask.v1"
    assert prediction.trace is not None
    assert prediction.trace.model_id == "tabu.cell.base"


def test_table_cell_base_accepts_truth_free_query_marker_origins() -> None:
    model = build_model("tabu.cell.base", label_broadcast=True)
    prediction = model._forward_dense(_dense_input(query=True))
    assert prediction.metadata["query_marker"] == "unified"
    assert prediction.metadata["label_broadcast"] is True
    assert prediction.auxiliaries["query_target_mask"].any()
    tokenizer_event = next(event for event in prediction.trace.events if event.name == "tokenizer")
    assert tokenizer_event.metadata["label_broadcast"] is True


def test_label_broadcast_uses_query_marker_without_label_payload() -> None:
    inputs = _dense_input(query=True)
    model = build_model("tabu.cell.base", label_broadcast=True)
    symbols = model.symbolizer(inputs)
    tokens = model.tokenizer(symbols)
    broadcast = _label_broadcast(tokens.cells, inputs, enabled=True)
    # Query row/label is not a source in the ordinary OMAB masks, but its
    # nonzero unified marker can broadcast to the same-row predictor cell.
    assert not torch.equal(broadcast[0, 1, 0], tokens.cells[0, 1, 0])
    assert torch.equal(broadcast[0, 1, 1], tokens.cells[0, 1, 1])
    assert torch.equal(inputs.values[0, 1, 1], torch.tensor(0.0))


def test_label_broadcast_is_in_the_base_dynamics_input() -> None:
    inputs = _dense_input(query=True)
    plain = build_model("tabu.cell.base", label_broadcast=False)
    enabled = build_model("tabu.cell.base", label_broadcast=True)
    enabled.load_state_dict(plain.state_dict())
    first = plain._forward_dense(inputs)
    second = enabled._forward_dense(inputs)
    first_tokenizer = next(event for event in first.trace.events if event.name == "tokenizer")
    second_tokenizer = next(event for event in second.trace.events if event.name == "tokenizer")
    assert first_tokenizer.output_hash != second_tokenizer.output_hash


def test_label_broadcast_tau_is_explicit_and_positive() -> None:
    model = build_model(
        "tabu.cell.base",
        label_broadcast=True,
        label_broadcast_tau=0.25,
    )
    prediction = model._forward_dense(_dense_input(query=True))
    assert prediction.metadata["label_broadcast_tau"] == 0.25
    with pytest.raises(ValueError, match="label_broadcast_tau"):
        build_model("tabu.cell.base", label_broadcast_tau=0.0)


@pytest.mark.parametrize(
    ("model_type", "mode", "expected_shape"),
    [
        (TabUCellRowModel, "row", (1, 3, 2, 2)),
        (TabUCellColumnModel, "column", (1, 3, 2, 2)),
        (TabUCellRowColumnModel, "row_column", (1, 3, 2, 4)),
    ],
)
def test_cell_special_readouts_are_direct_projections(
    model_type: type, mode: str, expected_shape: tuple[int, ...]
) -> None:
    config = ReferenceConfig(d_model=32, n_heads=4, d_ff=64, matched_slots=2)
    readout = CellSpecialReadout(config, mode=mode)
    carrier = torch.zeros(
        1,
        3 + (2 if mode in {"column", "row_column"} else 0),
        2 + (2 if mode in {"row", "row_column"} else 0),
        32,
    )
    carrier[:, :3, :2] = 1.0
    if mode in {"row", "row_column"}:
        carrier[:, :3, 2:4] = 2.0
    if mode in {"column", "row_column"}:
        carrier[:, 3:5, :2] = 3.0
    coordinates = readout.coordinates(carrier, n_rows=3, n_features=2)
    assert coordinates.shape == expected_shape
    first_width = 96.0 if mode == "column" else 64.0
    assert torch.equal(coordinates[..., :2], torch.full(expected_shape, first_width)[..., :2])
    if mode == "row_column":
        assert torch.equal(coordinates[..., 2:], torch.full(expected_shape, 96.0)[..., 2:])


def test_cell_special_carrier_roles_keep_cross_corner_exact_zero() -> None:
    model = TabUCellRowColumnModel()
    cells = torch.ones(1, 2, 3, model.config.d_model)
    visible = torch.ones(1, 2, 3, dtype=torch.bool)
    carrier, column_sources, row_sources, column_receivers, row_receivers = model._initial_carrier(
        cells, visible
    )
    k = model.config.matched_slots
    assert carrier.shape == (1, 2 + k, 3 + k, model.config.d_model)
    assert torch.equal(carrier[:, 2:, 3:], torch.zeros_like(carrier[:, 2:, 3:]))
    assert not bool(column_receivers[:, :2, 3:].any())
    assert not bool(row_receivers[:, 2:, :3].any())
    assert not bool(column_sources[:, 2:, :].any())
    assert not bool(row_sources[:, :, 3:].any())


@pytest.mark.parametrize("profile", ("m", "w", "rc"))
def test_axis_b_rec_profiles_are_explicit_and_truth_free(profile: str) -> None:
    from tabu_lab.models import build_model as build_runtime_model

    model = build_runtime_model(
        "tabu.cell.rec",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
        ),
        profile=profile,
    )
    prediction = model._forward_dense(_dense_input())
    assert prediction.model_id == "tabu.cell.rec"
    assert prediction.metadata["profile_id"] == f"recommendation.{profile}"
    assert prediction.trace.metadata["profile_id"] == f"recommendation.{profile}"


def test_axis_b_rec_profiles_have_distinct_trace_model_identity() -> None:
    config = ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
    )
    traces = {
        profile: build_model("tabu.cell.rec", config=config, profile=profile)
        ._forward_dense(_dense_input())
        .trace
        for profile in ("m", "w", "rc")
    }
    assert all(trace is not None for trace in traces.values())
    model_hashes = {trace.model_hash for trace in traces.values() if trace is not None}
    assert len(model_hashes) == 3
    assert '"profile_id":"recommendation.m"' in traces["m"].metadata["model_variant"]
    assert '"profile_id":"recommendation.w"' in traces["w"].metadata["model_variant"]
    assert '"profile_id":"recommendation.rc"' in traces["rc"].metadata["model_variant"]
    repeated = build_model("tabu.cell.rec", config=config, profile="m")._forward_dense(
        _dense_input()
    )
    assert repeated.trace is not None
    assert repeated.trace.model_hash == traces["m"].model_hash


def test_axis_b_rec_builder_requires_an_explicit_profile() -> None:
    from tabu_lab.models import build_model as build_runtime_model

    with pytest.raises(TypeError, match="requires an explicit profile"):
        build_runtime_model("tabu.cell.rec")


def test_axis_b_rec_matching_readout_pairs_same_special_slot() -> None:
    readout = CellMatchingReadout()
    carrier = torch.zeros(1, 3 + 2, 2 + 2, 4)
    carrier[:, 3:, :2] = 3.0
    carrier[:, :3, 2:] = 5.0
    coordinates = readout.coordinates(carrier, n_rows=3, n_features=2)
    assert coordinates.shape == (1, 3, 2, 2)
    assert torch.equal(coordinates, torch.full_like(coordinates, 60.0))


def test_cell_tokenizer_follows_base_fourier_and_episode_sphere_contract() -> None:
    values = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 2.0, 1.0], [3.0, 3.0, 2.0], [0.0, 0.0, 0.0]]])
    visible = torch.tensor(
        [[[True, True, True], [True, True, True], [True, True, True], [False, False, False]]]
    )
    target = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.tensor(
            [
                [
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                ]
            ]
        ),
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        feature_specs=(
            FeatureSpec(name="n0", kind=FeatureKind.NUMERIC),
            FeatureSpec(name="n1", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b", "c"),
                codebook_id="cell-token-test",
            ),
        ),
        episode_id="cell-token-episode",
    )
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    symbols = Symbolizer()(inputs)
    first = tokenizer(symbols).cells
    second = tokenizer(symbols).cells
    assert torch.equal(first, second)
    assert torch.equal(first[0, :3, 0], first[0, :3, 1])
    assert torch.allclose(first[0, :3, 2].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.equal(first[0, 3], torch.zeros_like(first[0, 3]))
    assert not torch.equal(first[0, 0, 2], first[0, 1, 2])


def test_cell_tokenizer_random_sphere_changes_only_with_episode_identity() -> None:
    values = torch.tensor([[[0.0], [1.0]]])
    visible = torch.ones(1, 2, 1, dtype=torch.bool)
    base = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=torch.zeros_like(visible),
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b"),
                codebook_id="cell-token-test",
            ),
        ),
        episode_id="episode-a",
    )
    other = DenseModelInput(
        values=base.values,
        visible_mask=base.visible_mask,
        target_mask=base.target_mask,
        natural_missing_mask=base.natural_missing_mask,
        feature_specs=base.feature_specs,
        episode_id="episode-b",
    )
    config = ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2)
    tokenizer = CellTokenizer(config)
    first = tokenizer(Symbolizer()(base)).cells
    second = tokenizer(Symbolizer()(other)).cells
    assert not torch.equal(first, second)


def test_cell_tokenizer_nominal_codes_follow_first_appearance_order() -> None:
    visible = torch.ones(1, 3, 1, dtype=torch.bool)
    spec = FeatureSpec(
        name="c",
        kind=FeatureKind.CATEGORICAL,
        domain=("a", "b", "c"),
        codebook_id="cell-token-order",
    )
    common = {
        "visible_mask": visible,
        "target_mask": torch.zeros_like(visible),
        "natural_missing_mask": torch.zeros_like(visible),
        "feature_specs": (spec,),
        "episode_id": "first-appearance-episode",
    }
    first_two = DenseModelInput(values=torch.tensor([[[2.0], [1.0], [2.0]]]), **common)
    second_two = DenseModelInput(values=torch.tensor([[[1.0], [2.0], [1.0]]]), **common)
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    first = tokenizer(Symbolizer()(first_two)).cells
    second = tokenizer(Symbolizer()(second_two)).cells
    # The same Episode sphere is addressed by local rank, so swapping which
    # code appears first swaps the token assignments as well.
    assert torch.equal(first[0, 0], second[0, 0])
    assert torch.equal(first[0, 1], second[0, 1])


def test_cell_tokenizer_does_not_use_static_codebook_identity() -> None:
    values = torch.tensor([[[0.0], [1.0]]])
    visible = torch.ones(1, 2, 1, dtype=torch.bool)
    spec = FeatureSpec(
        name="c",
        kind=FeatureKind.CATEGORICAL,
        domain=("a", "b"),
        codebook_id="codebook-a",
    )
    base = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=torch.zeros_like(visible),
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(spec,),
        episode_id="codebook-independent",
    )
    changed_codebook = DenseModelInput(
        values=base.values,
        visible_mask=base.visible_mask,
        target_mask=base.target_mask,
        natural_missing_mask=base.natural_missing_mask,
        feature_specs=(
            FeatureSpec(
                name=spec.name,
                kind=spec.kind,
                domain=spec.domain,
                codebook_id="codebook-b",
            ),
        ),
        episode_id=base.episode_id,
    )
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    first = tokenizer(Symbolizer()(base)).cells
    second = tokenizer(Symbolizer()(changed_codebook)).cells
    assert torch.equal(first, second)


def test_cell_tokenizer_is_invariant_to_category_relabeling() -> None:
    values = torch.tensor([[[0.0], [1.0], [2.0], [1.0]]])
    relabeled_values = torch.tensor([[[1.0], [0.0], [2.0], [0.0]]])
    visible = torch.ones(1, 4, 1, dtype=torch.bool)
    common = {
        "visible_mask": visible,
        "target_mask": torch.zeros_like(visible),
        "natural_missing_mask": torch.zeros_like(visible),
        "episode_id": "category-relabel",
    }
    first = DenseModelInput(
        values=values,
        feature_specs=(
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("red", "green", "blue"),
                codebook_id="category-relabel-v1",
            ),
        ),
        **common,
    )
    relabeled = DenseModelInput(
        values=relabeled_values,
        feature_specs=(
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("green", "red", "blue"),
                codebook_id="category-relabel-v1",
            ),
        ),
        **common,
    )
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    first_tokens = tokenizer(Symbolizer()(first)).cells
    relabeled_tokens = tokenizer(Symbolizer()(relabeled)).cells
    assert torch.equal(first_tokens, relabeled_tokens)


def test_source_scoped_codebook_v2_is_row_equivariant_and_episode_stable() -> None:
    values = torch.tensor([[[0.0], [1.0], [0.0]]])
    visible = torch.ones_like(values, dtype=torch.bool)
    spec = FeatureSpec(
        name="c",
        kind=FeatureKind.CATEGORICAL,
        domain=("a", "b"),
        codebook_id="stable-source-column-v2",
    )
    common = {
        "visible_mask": visible,
        "target_mask": torch.zeros_like(visible),
        "natural_missing_mask": torch.zeros_like(visible),
        "feature_specs": (spec,),
    }
    base = DenseModelInput(values=values, episode_id="episode-a", **common)
    replay = DenseModelInput(values=values, episode_id="episode-b", **common)
    permutation = torch.tensor([1, 0, 2])
    permuted = DenseModelInput(
        values=values[:, permutation],
        visible_mask=visible[:, permutation],
        target_mask=torch.zeros_like(visible[:, permutation]),
        natural_missing_mask=torch.zeros_like(visible[:, permutation]),
        feature_specs=(spec,),
        episode_id="episode-c",
    )
    tokenizer = CellTokenizer(
        ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2),
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
    )
    base_tokens = tokenizer(Symbolizer()(base)).cells
    replay_tokens = tokenizer(Symbolizer()(replay)).cells
    permuted_tokens = tokenizer(Symbolizer()(permuted)).cells
    assert torch.equal(base_tokens, replay_tokens)
    assert torch.equal(base_tokens[:, permutation], permuted_tokens)


def test_source_scoped_codebook_v2_tracks_domain_labels_and_separates_sources() -> None:
    values = torch.tensor([[[0.0], [1.0], [2.0], [1.0]]])
    relabeled_values = torch.tensor([[[1.0], [0.0], [2.0], [0.0]]])
    visible = torch.ones_like(values, dtype=torch.bool)

    def inputs(
        raw_values: torch.Tensor,
        *,
        domain: tuple[str, ...],
        codebook_id: str,
    ) -> DenseModelInput:
        return DenseModelInput(
            values=raw_values,
            visible_mask=visible,
            target_mask=torch.zeros_like(visible),
            natural_missing_mask=torch.zeros_like(visible),
            feature_specs=(
                FeatureSpec(
                    name="c",
                    kind=FeatureKind.CATEGORICAL,
                    domain=domain,
                    codebook_id=codebook_id,
                ),
            ),
            episode_id="stable-domain",
        )

    tokenizer = CellTokenizer(
        ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2),
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
    )
    base = tokenizer(
        Symbolizer()(inputs(values, domain=("red", "green", "blue"), codebook_id="source-a"))
    ).cells
    reordered = tokenizer(
        Symbolizer()(
            inputs(
                relabeled_values,
                domain=("green", "red", "blue"),
                codebook_id="source-a",
            )
        )
    ).cells
    other_source = tokenizer(
        Symbolizer()(inputs(values, domain=("red", "green", "blue"), codebook_id="source-b"))
    ).cells
    assert torch.equal(base, reordered)
    assert not torch.equal(base, other_source)


def test_source_scoped_codebook_v2_fails_closed_above_100_categories() -> None:
    domain = tuple(f"category-{index}" for index in range(101))
    values = torch.tensor([[[0.0]]])
    visible = torch.ones_like(values, dtype=torch.bool)
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=torch.zeros_like(visible),
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=domain,
                codebook_id="source-over-capacity",
            ),
        ),
    )
    tokenizer = CellTokenizer(
        ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2),
        nominal_tokenizer=CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
    )
    with pytest.raises(ValueError, match="exceeds frozen codebook capacity"):
        tokenizer(Symbolizer()(inputs))


def test_cell_tokenizer_rejects_degenerate_ordinal_domain() -> None:
    inputs = DenseModelInput(
        values=torch.tensor([[[0.0]]]),
        visible_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        target_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        natural_missing_mask=torch.zeros(1, 1, 1, dtype=torch.bool),
        feature_specs=(
            FeatureSpec(
                name="ordered",
                kind=FeatureKind.ORDINAL,
                domain=("only",),
                codebook_id="ordinal-singleton",
            ),
        ),
    )
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    with pytest.raises(ValueError, match="at least two"):
        tokenizer(Symbolizer()(inputs))


def test_cell_base_keeps_numeric_and_nominal_branches_typed() -> None:
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
                codebook_id="cell-mixed-v1",
            ),
        ),
        episode_id="cell-mixed-episode",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=ReferenceConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
        ),
    )._forward_dense(inputs)
    assert prediction.entries["numeric"].status.value == "ok"
    assert prediction.entries["categorical"].status.value == "ok"
    assert prediction.outputs["categorical"].shape == (1, 3, 2)
    assert torch.isfinite(prediction.auxiliaries["categorical_log_probabilities"]).all()


def test_cell_tokenizer_is_equivariant_under_column_permutation() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]])
    visible = torch.ones(1, 3, 2, dtype=torch.bool)
    specs = (
        FeatureSpec(name="n0", kind=FeatureKind.NUMERIC),
        FeatureSpec(
            name="c",
            kind=FeatureKind.CATEGORICAL,
            domain=("a", "b", "c"),
            codebook_id="cell-token-test",
        ),
    )
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=torch.zeros_like(visible),
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=specs,
        episode_id="equivariance-episode",
    )
    permuted = DenseModelInput(
        values=values.flip(-1),
        visible_mask=visible.flip(-1),
        target_mask=torch.zeros_like(visible),
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(specs[1], specs[0]),
        episode_id="equivariance-episode",
    )
    tokenizer = CellTokenizer(ReferenceConfig(d_model=8, n_heads=2, d_ff=16, matched_slots=2))
    first = tokenizer(Symbolizer()(inputs)).cells
    second = tokenizer(Symbolizer()(permuted)).cells.flip(2)
    assert torch.equal(first, second)


def test_cell_base_treats_exact_null_cell_as_typed_no_support() -> None:
    values = torch.tensor([[[0.0], [2.0]]])
    visible = torch.tensor([[[False], [True]]])
    target = ~visible
    natural = torch.tensor([[[True], [False]]])
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=natural,
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        episode_id="null-contract",
    )
    prediction = build_model("tabu.cell.base")._forward_dense(inputs)
    assert prediction.metadata["status"] == "no_support"
    assert prediction.entries["numeric"].status.value == "no_support"
    assert prediction.auxiliaries["support_available"][0, 0, 0].item() is False


def test_cell_base_keeps_natural_null_typed_under_mab_ablation() -> None:
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
        episode_id="null-contract-mab",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=ReferenceConfig(block_kind=DynamicsBlockKind.MAB),
    )._forward_dense(inputs)
    assert prediction.metadata["status"] == "no_support"
    assert prediction.auxiliaries["support_available"][0, 0, 0].item() is False
