from __future__ import annotations

import torch

from tabu_lab.contracts import (
    FeatureKind,
    FeatureRole,
    ForwardRole,
    OriginState,
    forward_role_mask,
    origin_mask,
)
from tabu_lab.experiments import FeasibilityStatus, assess_nw_targets
from tabu_lab.experiments.fixture_registry import (
    build_registered_f0_fixture,
    build_registered_f0_fixture_for_dataset,
)
from tabu_lab.experiments.fixtures import (
    BUILDABLE_CONTRACTS,
    DATA_SEED,
    MODEL_SEEDS,
    SPLIT_SEED,
    InfeasibleReason,
    assert_truth_isolated,
    build_all_f0_fixtures,
    build_f0_feasibility_targets,
    build_f0_fixture,
    build_f0_fixture_for_dataset,
    build_infeasible_f0_fixtures,
)
from tabu_lab.models import ReferenceConfig, build_model


def _reference_config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=16,
    )


def test_f0_fixture_ids_seeds_and_hashes_are_deterministic() -> None:
    first = build_all_f0_fixtures()
    second = build_all_f0_fixtures()

    assert tuple(fixture.contract_id for fixture in first) == BUILDABLE_CONTRACTS
    assert MODEL_SEEDS == (1729, 2718, 31415)
    assert DATA_SEED == 104729
    assert SPLIT_SEED == 130363
    assert tuple(fixture.fixture_hash for fixture in first) == tuple(
        fixture.fixture_hash for fixture in second
    )
    for left, right in zip(first, second, strict=True):
        assert left.dataset.dataset_hash == right.dataset.dataset_hash
        assert left.split_manifest.manifest_hash == right.split_manifest.manifest_hash
        assert left.recipe.recipe_hash == right.recipe.recipe_hash
        assert left.episode_schedule.schedule_hash == right.episode_schedule.schedule_hash
        assert left.episode_schedule.recipe_hashes == (left.recipe.recipe_hash,)
        assert left.episode_schedule.targets_per_episode == int(left.truth.target_mask.sum())
        assert torch.equal(left.evidence.forward_values, right.evidence.forward_values)
        assert torch.equal(left.truth.target_values, right.truth.target_values)


def test_completion_contracts_share_exact_mixed_table_and_target_schedule() -> None:
    fixtures = tuple(
        build_f0_fixture(contract_id)
        for contract_id in ("tabuf", "tabu.unit_row", "tabu.unit_pair")
    )

    assert {fixture.dataset.dataset_hash for fixture in fixtures} == {
        fixtures[0].dataset.dataset_hash
    }
    assert {fixture.recipe.recipe_hash for fixture in fixtures} == {fixtures[0].recipe.recipe_hash}
    assert all(fixture.dataset.shape == (32, 4) for fixture in fixtures)
    assert all(int(fixture.target_family_masks["numeric"].sum()) == 8 for fixture in fixtures)
    assert all(int(fixture.target_family_masks["categorical"].sum()) == 8 for fixture in fixtures)
    numeric_truth = fixtures[0].truth.target_values[fixtures[0].target_family_masks["numeric"]]
    categorical_truth = fixtures[0].truth.target_values[
        fixtures[0].target_family_masks["categorical"]
    ]
    assert torch.unique(numeric_truth).numel() >= 4
    assert torch.unique(categorical_truth).numel() >= 2


def test_completion_v2_is_representation_identifiable_and_not_trivial() -> None:
    fixture = build_f0_fixture("tabuf", fixture_version="v2")
    numeric_mask = fixture.target_family_masks["numeric"]
    categorical_mask = fixture.target_family_masks["categorical"]
    values = fixture.evidence.forward_values
    truth = fixture.truth.target_values
    raw_values = fixture.dataset.values

    assert fixture.dataset.dataset_id == "f0-completion-mixed-identifiable-v2"
    assert fixture.dataset.shape == (32, 4)
    assert int(numeric_mask.sum()) == 8
    assert int(categorical_mask.sum()) == 8
    assert torch.unique(raw_values[numeric_mask]).numel() == 4
    assert torch.unique(truth[categorical_mask]).numel() == 3

    # Every masked numeric value has two visible, same-latent witnesses in its
    # own row.  Categorical truth is determined by that visible latent tier.
    for row, feature in numeric_mask.nonzero(as_tuple=False).tolist():
        witness_columns = tuple(column for column in range(3) if column != feature)
        assert bool(fixture.evidence.source_mask[row, list(witness_columns)].all())
        assert torch.equal(
            raw_values[row, list(witness_columns)],
            raw_values[row, feature].expand(len(witness_columns)),
        )
    for row, _ in categorical_mask.nonzero(as_tuple=False).tolist():
        latent = values[row, 0]
        expected = 0 if latent < -0.5 else 2 if latent > 0.5 else 1
        assert int(truth[row, 3]) == expected

    numeric_truth = truth[numeric_mask]
    assert float((numeric_truth - numeric_truth.mean()).square().mean()) > 0.9
    categorical_truth = truth[categorical_mask].long()
    mode = torch.mode(categorical_truth).values
    assert float((categorical_truth == mode).float().mean()) == 0.5


def test_rec_v2_retains_equal_value_witnesses_in_both_arms() -> None:
    fixture = build_f0_fixture("tabu4rec", fixture_version="v2")
    source = fixture.evidence.source_mask
    values = fixture.evidence.forward_values

    assert fixture.dataset.shape == (16, 12)
    assert int(origin_mask(fixture.dataset.origin_states, OriginState.OBSERVED).sum()) == 134
    assert int(fixture.truth.target_mask.sum()) == 24
    assert torch.unique(fixture.truth.target_values[fixture.truth.target_mask]).numel() == 5
    for user, item in fixture.truth.target_mask.nonzero(as_tuple=False).tolist():
        truth = fixture.truth.target_values[user, item]
        user_arm = values[source[:, item], item]
        item_arm = values[user, source[user]]
        assert bool((user_arm == truth).any())
        assert bool((item_arm == truth).any())

    report = assess_nw_targets(
        build_f0_feasibility_targets(fixture),
        report_id="tabu4rec-F0-v2",
    )
    assert report.ready


def test_tabul_and_tabufl_share_data_but_keep_query_and_completion_ledgers_separate() -> None:
    tabul = build_f0_fixture("tabul")
    tabufl = build_f0_fixture("tabufl")

    assert tabul.dataset.dataset_hash == tabufl.dataset.dataset_hash
    assert tabul.dataset.shape == tabufl.dataset.shape == (64, 6)
    assert tabul.builder_options == tabufl.builder_options == {"label_columns": (4, 5)}
    assert int(tabul.target_family_masks["L"].sum()) == 32
    assert set(tabufl.target_family_masks) == {"F", "L"}
    assert int(tabufl.target_family_masks["F"].sum()) == 16
    assert int(tabufl.target_family_masks["L"].sum()) == 32
    assert torch.equal(tabul.target_family_masks["L"], tabufl.target_family_masks["L"])
    assert int(origin_mask(tabul.evidence.origin_states, OriginState.QUERY).sum()) == 32
    assert int(origin_mask(tabufl.evidence.origin_states, OriginState.ARTIFICIAL_MASK).sum()) == 16
    query_rows = origin_mask(tabul.evidence.origin_states, OriginState.QUERY).any(dim=1)
    assert query_rows.tolist() == [False] * 48 + [True] * 16


def test_supervised_v2_has_predictor_witnesses_and_separate_ledgers() -> None:
    tabul = build_f0_fixture("tabul", fixture_version="v2")
    tabufl = build_f0_fixture("tabufl", fixture_version="v2")

    assert tabul.dataset.dataset_hash == tabufl.dataset.dataset_hash
    assert tabul.dataset.shape == tabufl.dataset.shape == (64, 8)
    assert (
        tabul.builder_options
        == tabufl.builder_options
        == {
            "label_columns": (6, 7),
            "label_address_plan": "predictor_only_per_label_v1",
        }
    )
    assert int(tabul.target_family_masks["L"].sum()) == 32
    assert int(tabufl.target_family_masks["F"].sum()) == 16
    assert int(tabufl.target_family_masks["L"].sum()) == 32
    assert torch.equal(tabul.target_family_masks["L"], tabufl.target_family_masks["L"])
    twin = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    for row, feature in tabufl.target_family_masks["F"].nonzero(as_tuple=False).tolist():
        witness = twin[feature]
        assert bool(tabufl.evidence.source_mask[row, witness])
        assert (
            tabufl.evidence.forward_values[row, witness] == tabufl.truth.target_values[row, feature]
        )


def test_tabufl_v4_balances_identifiable_completion_and_label_ledgers() -> None:
    fixture = build_f0_fixture("tabufl", fixture_version="v4")
    feature_targets = fixture.target_family_masks["F"]
    label_targets = fixture.target_family_masks["L"]

    assert fixture.dataset.shape == (64, 8)
    assert fixture.dataset.dataset_id == ("f0-tabufl-completion-latent-label-composition-v4")
    assert fixture.builder_options == {
        "label_columns": (6, 7),
        "label_address_plan": "predictor_unit_linked_per_label_v2",
    }
    assert int(feature_targets.sum()) == 12
    assert int(label_targets.sum()) == 32
    assert torch.equal(label_targets[48:64, 6:8], torch.ones((16, 2), dtype=torch.bool))

    raw = fixture.dataset.values
    expected_levels = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    for feature in range(3):
        rows = feature_targets[:, feature].nonzero(as_tuple=False).flatten()
        assert rows.tolist() == list(range(feature * 4, feature * 4 + 4))
        assert torch.equal(raw[rows, feature], expected_levels)
        for row in rows.tolist():
            visible_twins = [column for column in range(3) if column != feature]
            assert bool(fixture.evidence.source_mask[row, visible_twins].all())
            assert torch.equal(
                raw[row, visible_twins],
                raw[row, feature].expand(len(visible_twins)),
            )

    # Query rows remain available to their own L lane but never become F
    # completion support.  Feasibility uses the exact model readout ledger.
    completion_targets = tuple(
        target
        for target in build_f0_feasibility_targets(fixture)
        if target.family.value == "completion"
    )
    assert len(completion_targets) == 12
    assert all(
        support_id // fixture.dataset.shape[1] < 48
        for target in completion_targets
        for arm in target.arms
        for support_id in arm.support_ids
    )
    assert (
        build_f0_fixture_for_dataset("tabufl", fixture.dataset.dataset_id).fixture_hash
        == fixture.fixture_hash
    )


def test_tabufl_v5_restores_exact_16f_32l_frozen_contract() -> None:
    fixture = build_registered_f0_fixture("tabufl", fixture_version="v5")
    feature_targets = fixture.target_family_masks["F"]
    label_targets = fixture.target_family_masks["L"]

    assert fixture.dataset.shape == (64, 8)
    assert fixture.dataset.dataset_id == "f0-tabufl-four-latent-label-composition-v5"
    assert int(feature_targets.sum()) == 16
    assert int(label_targets.sum()) == 32
    assert int(fixture.truth.target_mask.sum()) == 48
    assert torch.equal(label_targets[48:64, 6:8], torch.ones((16, 2), dtype=torch.bool))

    raw = fixture.dataset.values
    expected_levels = torch.tensor([-0.75, -0.25, 0.25, 0.75])
    for feature in range(4):
        rows = feature_targets[:, feature].nonzero(as_tuple=False).flatten()
        assert rows.tolist() == list(range(feature * 4, feature * 4 + 4))
        assert torch.equal(raw[rows, feature], expected_levels)
        for row in rows.tolist():
            witnesses = [column for column in range(4) if column != feature]
            assert bool(fixture.evidence.source_mask[row, witnesses].all())
            assert torch.equal(raw[row, witnesses], raw[row, feature].expand(3))

    # Query rows may drive their own label lane, but none can become feature
    # completion support.
    completion_targets = tuple(
        target
        for target in build_f0_feasibility_targets(fixture)
        if target.family.value == "completion"
    )
    assert len(completion_targets) == 16
    assert all(
        support_id // fixture.dataset.shape[1] < 48
        for target in completion_targets
        for arm in target.arms
        for support_id in arm.support_ids
    )
    assert (
        build_registered_f0_fixture_for_dataset(
            "tabufl", fixture.dataset.dataset_id
        ).fixture_hash
        == fixture.fixture_hash
    )


def test_graph_fixture_is_an_8x8_grid_with_nonadjacent_tau_targets() -> None:
    fixture = build_f0_fixture("tabu4graph")
    topology = fixture.dataset.graph_topology

    assert topology is not None
    assert fixture.dataset.shape == (64, 2)
    assert fixture.builder_options == {"target_feature": 1}
    assert torch.equal(topology.adjacency, topology.adjacency.transpose(0, 1))
    assert not bool(topology.adjacency.diagonal().any())
    assert int(topology.adjacency.sum()) == 224
    tau_targets = fixture.target_family_masks["tau"][:, 1]
    assert int(tau_targets.sum()) == 16
    assert not bool(topology.adjacency[tau_targets][:, tau_targets].any()), (
        "each target must retain graph-local source neighbors"
    )
    tau_sources = forward_role_mask(fixture.evidence.forward_roles, ForwardRole.SOURCE)[:, 1]
    assert bool(topology.adjacency[tau_targets][:, tau_sources].any(dim=1).all())


def test_recommendation_fixture_has_exact_density_and_two_active_support_arms() -> None:
    fixture = build_f0_fixture("tabu4rec")
    raw_observed = origin_mask(fixture.dataset.origin_states, OriginState.OBSERVED)
    targets = fixture.target_family_masks["rating"]
    source = forward_role_mask(fixture.evidence.forward_roles, ForwardRole.SOURCE)

    assert fixture.dataset.shape == (16, 12)
    assert int(raw_observed.sum()) == 134
    assert int(targets.sum()) == 24
    assert torch.unique(fixture.truth.target_values[targets]).numel() == 5
    assert all(spec.role is FeatureRole.RESPONSE for spec in fixture.dataset.feature_specs)
    for user, item in torch.nonzero(targets, as_tuple=False).tolist():
        row_support = source[user]
        column_support = source[:, item]
        assert bool(row_support.any())
        assert bool(column_support.any())
        support_values = torch.cat(
            (
                fixture.evidence.forward_values[user, row_support],
                fixture.evidence.forward_values[column_support, item],
            )
        )
        truth = fixture.truth.target_values[user, item]
        assert support_values.min() <= truth <= support_values.max()
        row_values = fixture.evidence.forward_values[user, row_support]
        column_values = fixture.evidence.forward_values[column_support, item]
        dual_lower = 0.5 * (row_values.min() + column_values.min())
        dual_upper = 0.5 * (row_values.max() + column_values.max())
        assert dual_lower <= truth <= dual_upper


def test_all_positive_fixtures_are_truth_isolated_and_terminal_feasible() -> None:
    for fixture in build_all_f0_fixtures():
        assert_truth_isolated(fixture)
        source = forward_role_mask(fixture.evidence.forward_roles, ForwardRole.SOURCE)
        for row, feature in torch.nonzero(fixture.truth.target_mask, as_tuple=False).tolist():
            spec = fixture.dataset.feature_specs[feature]
            if fixture.contract_id == "tabu4rec":
                row_support = fixture.evidence.forward_values[row, source[row]]
                column_support = fixture.evidence.forward_values[source[:, feature], feature]
                support = torch.cat((row_support, column_support))
            else:
                support = fixture.evidence.forward_values[source[:, feature], feature]
            assert support.numel() > 0
            truth = fixture.truth.target_values[row, feature]
            if spec.kind is FeatureKind.CATEGORICAL:
                assert bool((support == truth).any())
            elif fixture.contract_id == "tabu4rec":
                lower = 0.5 * (row_support.min() + column_support.min())
                upper = 0.5 * (row_support.max() + column_support.max())
                assert lower <= truth <= upper
            else:
                assert support.min() <= truth <= support.max()


def test_typed_feasibility_oracle_accepts_every_positive_fixture() -> None:
    for fixture in build_all_f0_fixtures():
        report = assess_nw_targets(
            build_f0_feasibility_targets(fixture),
            report_id=f"{fixture.contract_id.replace('.', '-')}-F0",
        )

        assert report.ready
        assert report.target_count == int(fixture.truth.target_mask.sum())
        assert report.feasible_targets == report.target_count


def test_all_positive_fixtures_forward_with_complete_target_coverage() -> None:
    for fixture in build_all_f0_fixtures():
        model = build_model(
            fixture.contract_id,
            config=_reference_config(),
            **fixture.builder_options,
        )
        prediction = model(fixture.evidence)

        assert prediction.metadata["status"] == "ok"
        assert torch.equal(prediction.outputs["target_mask"], fixture.truth.target_mask)
        assert bool(prediction.outputs["support_available"][fixture.truth.target_mask].all())
        assert not bool(prediction.outputs["abstention"][fixture.truth.target_mask].any())
        if fixture.contract_id == "tabu4rec":
            if prediction.metadata["numeric_terminal"] == "parameterized_matching":
                assert prediction.metadata["support"] == "parameterized_matching"
                assert "rec_arm_weights" not in prediction.outputs
            else:
                targets = fixture.truth.target_mask
                assert bool(prediction.outputs["rec_user_arm_support_available"][targets].all())
                assert bool(prediction.outputs["rec_item_arm_support_available"][targets].all())
                assert torch.allclose(
                    prediction.outputs["rec_arm_weights"][targets],
                    prediction.outputs["rec_arm_weights"].new_full((int(targets.sum()), 2), 0.5),
                )


def test_negative_fixtures_encode_three_distinct_terminal_failures() -> None:
    fixtures = {fixture.reason: fixture for fixture in build_infeasible_f0_fixtures()}
    assert set(fixtures) == set(InfeasibleReason)
    assert len({fixture.fixture_hash for fixture in fixtures.values()}) == 3
    for fixture in fixtures.values():
        assert_truth_isolated(fixture)

    numeric = fixtures[InfeasibleReason.NUMERIC_OUT_OF_HULL]
    numeric_sources = numeric.evidence.forward_values[numeric.evidence.source_mask]
    numeric_truth = numeric.truth.target_values[numeric.truth.target_mask].item()
    assert numeric_truth > numeric_sources.max().item()

    categorical = fixtures[InfeasibleReason.MISSING_CATEGORICAL_CLASS]
    categorical_sources = categorical.evidence.forward_values[categorical.evidence.source_mask]
    categorical_truth = categorical.truth.target_values[categorical.truth.target_mask].item()
    assert not bool((categorical_sources == categorical_truth).any())

    no_support = fixtures[InfeasibleReason.NO_SUPPORT]
    assert not bool(no_support.evidence.source_mask.any())
    prediction = build_model("tabuf", config=_reference_config())(no_support.evidence)
    assert prediction.metadata["status"] == "no_support"
    assert bool(prediction.outputs["abstention"].item())

    assessments = {
        reason: assess_nw_targets(
            build_f0_feasibility_targets(fixture),
            report_id=f"negative-{reason.value}",
        ).targets[0]
        for reason, fixture in fixtures.items()
    }
    assert assessments[InfeasibleReason.NUMERIC_OUT_OF_HULL].status is (
        FeasibilityStatus.TERMINAL_INFEASIBLE
    )
    assert assessments[InfeasibleReason.MISSING_CATEGORICAL_CLASS].status is (
        FeasibilityStatus.TERMINAL_INFEASIBLE
    )
    assert assessments[InfeasibleReason.NO_SUPPORT].status is FeasibilityStatus.NO_SUPPORT
