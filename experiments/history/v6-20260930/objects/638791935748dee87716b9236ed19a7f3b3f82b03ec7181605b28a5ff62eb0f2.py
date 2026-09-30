"""Truth-free graph topology carried alongside tabular episode evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import torch

from .canonical import canonical_hash


class GraphDirection(StrEnum):
    """How an adjacency entry must be interpreted.

    For ``DIRECTED``, ``adjacency[i, j]`` means an edge from ``node_ids[i]``
    to ``node_ids[j]``.  ``UNDIRECTED`` requires a symmetric adjacency.
    """

    DIRECTED = "directed"
    UNDIRECTED = "undirected"


@dataclass(frozen=True, slots=True, eq=False)
class GraphTopology:
    """Raw graph structure with no implicit symmetrization or self loops."""

    node_ids: tuple[str, ...]
    adjacency: torch.Tensor | Sequence[Sequence[bool]]
    direction: GraphDirection | str = GraphDirection.DIRECTED

    def __post_init__(self) -> None:
        node_ids = tuple(self.node_ids)
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise ValueError("GraphTopology.node_ids must be non-empty and unique")
        if any(not isinstance(node_id, str) or not node_id.strip() for node_id in node_ids):
            raise ValueError("GraphTopology.node_ids must be non-empty strings")
        adjacency = torch.as_tensor(self.adjacency).detach().clone().cpu()
        if adjacency.dtype is not torch.bool:
            raise ValueError("GraphTopology.adjacency must have bool dtype")
        expected = (len(node_ids), len(node_ids))
        if adjacency.ndim != 2 or tuple(adjacency.shape) != expected:
            raise ValueError("GraphTopology.adjacency must be square and match node_ids")
        try:
            direction = GraphDirection(self.direction)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown GraphDirection: {self.direction!r}") from exc
        if direction is GraphDirection.UNDIRECTED and not torch.equal(
            adjacency, adjacency.transpose(0, 1)
        ):
            raise ValueError("undirected GraphTopology.adjacency must be symmetric")
        object.__setattr__(self, "node_ids", node_ids)
        object.__setattr__(self, "adjacency", adjacency)
        object.__setattr__(self, "direction", direction)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GraphTopology)
            and self.node_ids == other.node_ids
            and self.direction is other.direction
            and torch.equal(self.adjacency, other.adjacency)
        )

    def __hash__(self) -> int:
        return hash(self.topology_hash)

    @property
    def topology_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.graph-topology.v1",
                "node_ids": self.node_ids,
                "adjacency": self.adjacency,
                "direction": self.direction,
            }
        )

    @property
    def is_directed(self) -> bool:
        return self.direction is GraphDirection.DIRECTED

    def induced(self, node_ids: Sequence[str]) -> GraphTopology:
        """Return the raw induced topology in the requested node order."""

        requested = tuple(node_ids)
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("induced topology node_ids must be non-empty and unique")
        index = {node_id: position for position, node_id in enumerate(self.node_ids)}
        try:
            positions = [index[node_id] for node_id in requested]
        except KeyError as exc:
            raise ValueError(f"unknown topology node id: {exc.args[0]!r}") from exc
        adjacency = self.adjacency[positions][:, positions]
        return GraphTopology(
            node_ids=requested,
            adjacency=adjacency,
            direction=self.direction,
        )


__all__ = ["GraphDirection", "GraphTopology"]
