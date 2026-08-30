from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from tabu_lab.contracts import (
    FeatureKind,
    FeatureRole,
    OriginState,
    TruthSidecar,
    origin_mask,
    origin_value_mask,
)
from tabu_lab.experiments.feasibility import assess_nw_targets
from tabu_lab.experiments.s1_topology import (
    GRAPH_COUNT,
    GRAPH_NODES,
    GRAPH_TARGETS_PER_EPISODE,
    GRAPH_TRAIN_COUNT,
    REC_ITEMS,
    REC_OBSERVED_INTERACTIONS,
    REC_TARGETS_PER_EPISODE,
    REC_TEST_EPISODES,
    REC_TRAIN_EPISODES,
    REC_USERS,
    REC_VALIDATION_EPISODES,
    GraphSyntheticRecipe,
    RecSyntheticRecipe,
    build_s1_graph_corpus,
    build_s1_rec_corpus,
)
from tabu_lab.experiments.splits import GraphSplitManifest, InteractionSplitManifest


@pytest.fixture(scope="module", params=tuple(GraphSyntheticRecipe))
def graph_corpus(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    return build_s1_graph_corpus(request.param)


@pytest.fixture(scope="module", params=tuple(RecSyntheticRecipe))
def rec_corpus(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    return build_s1_rec_corpus(request.param)


def _episodes(corpus: object) -> Iterator[object]:
    for partition in ("train", "validation", "test"):
        yield from corpus.episodes(partition)


def _assert_truth_isolated(episode: object) -> None:
    evidence = episode.evidence
    truth = episode.truth
    evidence_hash = evidence.evidence_hash
    changed_values = truth.target_values.clone()
    changed_values[truth.target_mask] += 17.0
    changed_truth = TruthSidecar(
        episode_id=truth.episode_id,
        recipe_hash=truth.recipe_hash,
        row_ids=truth.row_ids,
        feature_names=truth.feature_names,
        target_values=changed_values,
        target_mask=truth.target_mask,
        metadata=truth.metadata,
    )

    assert changed_truth.truth_hash != truth.truth_hash
    assert evidence.evidence_hash == evidence_hash
    assert not hasattr(evidence, "truth")
    assert not hasattr(evidence, "target_values")
    assert not bool((evidence.source_mask & truth.target_mask).any())
    assert bool((evidence.forward_values[truth.target_mask] == 0).all())


def test_graph_s1_is_typed_6_1_1_sbm_corpus(graph_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = graph_corpus
    assert isinstance(corpus.typed_split, GraphSplitManifest)
    assert corpus.typed_split.scope.value == "graph"
    assert corpus.dataset.shape == (GRAPH_COUNT * GRAPH_NODES, 3)
    assert [
        len(corpus.typed_split.partition(partition).elements)
        for partition in ("train", "validation", "test")
    ] == [GRAPH_TRAIN_COUNT, 1, 1]
    assert [
        len(corpus.episodes(partition))
        for partition in ("train", "validation", "test")
    ] == [GRAPH_TRAIN_COUNT, 1, 1]
    assert corpus.schedule.episode_count == GRAPH_COUNT
    assert corpus.schedule.targets_per_episode == GRAPH_TARGETS_PER_EPISODE
    assert corpus.builder_options["target_feature"] == 2
    assert corpus.builder_options["unit_receiver_plan"] == "same_row_visible_cells"

    topology = corpus.dataset.graph_topology
    assert topology is not None
    assert bool((topology.adjacency.sum(dim=1) > 0).all())
    for left in range(GRAPH_COUNT):
        left_slice = slice(left * GRAPH_NODES, (left + 1) * GRAPH_NODES)
        for right in range(GRAPH_COUNT):
            if left == right:
                continue
            right_slice = slice(right * GRAPH_NODES, (right + 1) * GRAPH_NODES)
            assert not bool(topology.adjacency[left_slice, right_slice].any())


def test_graph_s1_targets_are_diverse_feasible_and_truth_isolated(graph_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = graph_corpus
    target_spec = corpus.dataset.feature_specs[2]
    expected_kind = (
        FeatureKind.CATEGORICAL
        if corpus.builder_options["recipe"] == GraphSyntheticRecipe.COMMUNITY.value
        else FeatureKind.NUMERIC
    )
    assert target_spec.kind is expected_kind
    assert target_spec.role is FeatureRole.RESPONSE
    graph_by_row = dict(corpus.dataset.metadata["graph_ids_by_row"])
    train_graphs = {
        element.graph_id for element in corpus.typed_split.partition("train").elements
    }

    for partition in ("train", "validation", "test"):
        allowed_graphs = {
            element.graph_id for element in corpus.typed_split.partition(partition).elements
        }
        for episode in corpus.episodes(partition):
            target_values = episode.truth.target_values[episode.truth.target_mask]
            assert episode.truth.target_count == GRAPH_TARGETS_PER_EPISODE
            assert len(set(float(value) for value in target_values)) >= 4
            target_rows = torch.nonzero(
                episode.truth.target_mask.any(dim=1), as_tuple=False
            ).flatten()
            assert {
                graph_by_row[episode.evidence.row_ids[int(row)]] for row in target_rows
            } <= allowed_graphs
            source_rows = torch.nonzero(
                episode.evidence.source_mask[:, 2], as_tuple=False
            ).flatten()
            assert {
                graph_by_row[episode.evidence.row_ids[int(row)]] for row in source_rows
            } <= train_graphs
            report = assess_nw_targets(
                episode.feasibility_targets,
                report_id=f"test-{episode.ordinal}-{partition}",
            )
            assert report.ready
            assert all(target.arms[0].active for target in episode.feasibility_targets)
            _assert_truth_isolated(episode)


def test_graph_fit_statistics_exclude_heldout_graphs_and_train_targets(graph_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = graph_corpus
    graph_by_row = dict(corpus.dataset.metadata["graph_ids_by_row"])
    train_graphs = {
        element.graph_id for element in corpus.typed_split.partition("train").elements
    }
    fit_rows = corpus.fit_value_mask.any(dim=1)
    assert {
        graph_by_row[row_id]
        for row_id, included in zip(corpus.dataset.row_ids, fit_rows, strict=True)
        if bool(included)
    } <= train_graphs
    train_targets = torch.zeros(corpus.dataset.shape, dtype=torch.bool)
    for episode in corpus.train_episodes:
        train_targets |= episode.truth.target_mask
    assert not bool((corpus.fit_value_mask & train_targets).any())
    assert (
        corpus.fit_value_mask_hash
        == corpus.numeric_normalizer.statistics.fit_value_mask_hash
    )


def test_rec_s1_is_typed_64_by_48_exact_40_percent_corpus(rec_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = rec_corpus
    assert isinstance(corpus.typed_split, InteractionSplitManifest)
    assert corpus.dataset.shape == (REC_USERS, REC_ITEMS)
    assert int(origin_value_mask(corpus.dataset.origin_states).sum()) == (
        REC_OBSERVED_INTERACTIONS
    )
    assert abs(
        REC_OBSERVED_INTERACTIONS / (REC_USERS * REC_ITEMS) - 0.4
    ) <= 1.0 / (REC_USERS * REC_ITEMS)
    split_sizes = {
        partition: len(corpus.typed_split.partition(partition).interactions)
        for partition in ("train", "validation", "test")
    }
    assert sum(split_sizes.values()) == REC_OBSERVED_INTERACTIONS
    assert [
        len(corpus.episodes(partition))
        for partition in ("train", "validation", "test")
    ] == [REC_TRAIN_EPISODES, REC_VALIDATION_EPISODES, REC_TEST_EPISODES]
    assert corpus.schedule.episode_count == (
        REC_TRAIN_EPISODES + REC_VALIDATION_EPISODES + REC_TEST_EPISODES
    )
    assert corpus.schedule.targets_per_episode == REC_TARGETS_PER_EPISODE
    assert all(spec.role is FeatureRole.RESPONSE for spec in corpus.dataset.feature_specs)
    assert corpus.builder_options["recommendation_address_plan"] == (
        "axis_address_bootstrap_v1"
    )


def test_rec_s1_dual_arms_are_active_feasible_and_train_scoped(rec_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = rec_corpus
    train_interactions = {
        (interaction.user_id, interaction.item_id)
        for interaction in corpus.typed_split.partition("train").interactions
    }
    expected_distinct = (
        2
        if corpus.builder_options["recipe"] == RecSyntheticRecipe.PREFERENCE.value
        else 4
    )
    for episode in _episodes(corpus):
        target_values = episode.truth.target_values[episode.truth.target_mask]
        assert episode.truth.target_count == REC_TARGETS_PER_EPISODE
        assert len(set(float(value) for value in target_values)) >= expected_distinct
        source_coordinates = {
            (episode.evidence.row_ids[row], episode.evidence.feature_names[feature])
            for row, feature in torch.nonzero(
                episode.evidence.source_mask, as_tuple=False
            ).tolist()
        }
        assert source_coordinates <= train_interactions
        report = assess_nw_targets(
            episode.feasibility_targets,
            report_id=f"test-rec-{episode.partition}-{episode.ordinal}",
        )
        assert report.ready
        for target in episode.feasibility_targets:
            assert tuple(arm.arm_id for arm in target.arms) == ("user", "item")
            assert all(arm.active for arm in target.arms)
            target_value = (
                float(target.truth_code)
                if hasattr(target, "truth_code")
                else float(target.truth_value)
            )
            assert all(target_value in arm.support_values for arm in target.arms)
        _assert_truth_isolated(episode)


def test_rec_targets_are_observed_artificial_masks_and_stats_are_train_only(rec_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = rec_corpus
    train_coordinates = {
        (interaction.user_id, interaction.item_id)
        for interaction in corpus.typed_split.partition("train").interactions
    }
    fit_coordinates = {
        (corpus.dataset.row_ids[row], corpus.dataset.feature_names[feature])
        for row, feature in torch.nonzero(corpus.fit_value_mask, as_tuple=False).tolist()
    }
    assert fit_coordinates <= train_coordinates
    train_targets = torch.zeros(corpus.dataset.shape, dtype=torch.bool)
    for episode in corpus.train_episodes:
        train_targets |= episode.truth.target_mask
    assert not bool((corpus.fit_value_mask & train_targets).any())
    for episode in _episodes(corpus):
        assert bool(
            origin_mask(
                episode.evidence.origin_states,
                OriginState.ARTIFICIAL_MASK,
            )[episode.truth.target_mask].all()
        )
        assert not bool(
            origin_mask(
                episode.evidence.origin_states,
                OriginState.NATURAL_MISSING,
            )[episode.truth.target_mask].any()
        )
    if corpus.builder_options["recipe"] == RecSyntheticRecipe.RATING.value:
        assert corpus.numeric_normalizer.shared_numeric_groups == (
            corpus.dataset.feature_names,
        )
    else:
        assert not bool(corpus.fit_value_mask.any())
        assert corpus.numeric_normalizer.shared_numeric_groups == ()


def test_s1_topology_generators_are_content_deterministic() -> None:
    graph = build_s1_graph_corpus(GraphSyntheticRecipe.COMMUNITY)
    rec = build_s1_rec_corpus(RecSyntheticRecipe.RATING)

    assert graph.content_hash == build_s1_graph_corpus(
        GraphSyntheticRecipe.COMMUNITY
    ).content_hash
    assert rec.content_hash == build_s1_rec_corpus(RecSyntheticRecipe.RATING).content_hash
