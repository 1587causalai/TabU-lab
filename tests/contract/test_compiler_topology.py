from __future__ import annotations

import pytest
import torch

from tabu_lab.compiler import (
    TopologyBindingError,
    bind_split_view,
    compile_episode,
    split_dataset,
)
from tabu_lab.contracts import (
    EpisodeRecipe,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    GraphDirection,
    GraphTopology,
    RawDataset,
)

SOURCE = ForwardRole.RECEIVER | ForwardRole.SOURCE
TARGET = ForwardRole.RECEIVER | ForwardRole.TARGET


def _compile_with_dataset_topology(edge_forward: bool):  # type: ignore[no-untyped-def]
    adjacency = torch.zeros(4, 4, dtype=torch.bool)
    adjacency[2, 3] = edge_forward
    adjacency[3, 2] = not edge_forward
    topology = GraphTopology(
        node_ids=("fit-0", "fit-1", "eval-0", "eval-1"),
        adjacency=adjacency,
        direction=GraphDirection.DIRECTED,
    )
    dataset = RawDataset.from_values(
        dataset_id="typed-graph",
        values=torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        row_ids=topology.node_ids,
        feature_specs=(
            FeatureSpec(name="context"),
            FeatureSpec(name="response", role=FeatureRole.RESPONSE),
        ),
        graph_topology=topology,
    )
    manifest = split_dataset(dataset, {"train": (0, 1), "test": (2, 3)})
    fit_view = bind_split_view(dataset, manifest, "train")
    source_view = bind_split_view(dataset, manifest, "test")
    recipe = EpisodeRecipe.create(
        source_view,
        fit_view,
        ((SOURCE, TARGET), (SOURCE, TARGET)),
    )
    return compile_episode(source_view, recipe, fit_view=fit_view)


def test_dataset_topology_and_response_role_roundtrip_without_graph_rewrite() -> None:
    result = _compile_with_dataset_topology(edge_forward=True)
    topology = result.evidence.graph_topology

    assert topology is not None
    assert topology.node_ids == result.evidence.row_ids == ("eval-0", "eval-1")
    assert topology.direction is GraphDirection.DIRECTED
    assert topology.adjacency.tolist() == [[False, True], [False, False]]
    assert not topology.adjacency.diagonal().any()
    assert result.evidence.feature_specs[1].role is FeatureRole.RESPONSE
    assert result.provenance.graph_topology_hash == topology.topology_hash


def test_edge_change_changes_episode_and_evidence_hashes() -> None:
    forward = _compile_with_dataset_topology(edge_forward=True)
    reverse = _compile_with_dataset_topology(edge_forward=False)

    assert forward.evidence.episode_id != reverse.evidence.episode_id
    assert forward.evidence.evidence_hash != reverse.evidence.evidence_hash
    assert forward.provenance.graph_topology_hash != reverse.provenance.graph_topology_hash


def test_explicit_typed_topology_compiles_and_conflicts_fail_closed() -> None:
    dataset = RawDataset.from_values(
        dataset_id="explicit-graph",
        values=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        row_ids=("fit-0", "fit-1", "eval-0", "eval-1"),
    )
    manifest = split_dataset(dataset, {"train": (0, 1), "test": (2, 3)})
    fit_view = bind_split_view(dataset, manifest, "train")
    source_view = bind_split_view(dataset, manifest, "test")
    recipe = EpisodeRecipe.create(source_view, fit_view, ((TARGET,), (SOURCE,)))
    topology = GraphTopology(
        node_ids=source_view.row_ids,
        adjacency=torch.tensor([[False, True], [False, False]]),
    )
    result = compile_episode(
        source_view,
        recipe,
        fit_view=fit_view,
        graph_topology=topology,
    )

    assert result.evidence.graph_topology == topology

    conflicting = GraphTopology(
        node_ids=source_view.row_ids,
        adjacency=topology.adjacency.transpose(0, 1),
    )
    recipe_with_topology = EpisodeRecipe.create(
        source_view,
        fit_view,
        ((TARGET,), (SOURCE,)),
        graph_topology=topology,
    )
    with pytest.raises(TopologyBindingError, match="conflict"):
        compile_episode(
            source_view,
            recipe_with_topology,
            fit_view=fit_view,
            graph_topology=conflicting,
        )


def test_ordinary_table_compiles_without_topology() -> None:
    dataset = RawDataset.from_values(
        dataset_id="ordinary-table",
        values=torch.tensor([[1.0], [2.0]]),
        row_ids=("fit", "eval"),
    )
    manifest = split_dataset(dataset, {"train": (0,), "test": (1,)})
    fit_view = bind_split_view(dataset, manifest, "train")
    source_view = bind_split_view(dataset, manifest, "test")
    recipe = EpisodeRecipe.create(source_view, fit_view, ((TARGET,),))
    result = compile_episode(source_view, recipe, fit_view=fit_view)

    assert result.evidence.graph_topology is None
    assert result.provenance.graph_topology_hash is None
