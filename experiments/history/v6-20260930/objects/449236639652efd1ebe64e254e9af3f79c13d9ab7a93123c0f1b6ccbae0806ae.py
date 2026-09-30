"""Model-independent curriculum selection over existing frozen data artifacts.

This package does not import a model, generate data, or run optimization.
Model-specific episode factories are opt-in consumers of these contracts.
"""

from .catalog import (
    BoundTable,
    Provenance,
    StageSelection,
    bind_legacy_corpus,
    bind_table,
    reference_stage,
    require_disjoint,
    select_stage,
)
from .episodes import EpisodeFactory, EpisodeRequest, EpisodeSeeds

__all__ = [
    "BoundTable", "EpisodeFactory", "EpisodeRequest", "EpisodeSeeds", "Provenance",
    "StageSelection", "bind_legacy_corpus", "bind_table", "reference_stage", "require_disjoint",
    "select_stage",
]
