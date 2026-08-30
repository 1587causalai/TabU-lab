"""Fail-closed split-before-compile episode compiler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tabu_lab.contracts.canonical import canonical_hash, require_sha256
from tabu_lab.contracts.dataset import EpisodeRecipe, RawDataset, SplitView
from tabu_lab.contracts.episode import EvidenceEpisode, TruthSidecar
from tabu_lab.contracts.roles import (
    ForwardRole,
    OriginState,
    forward_role_mask,
    origin_mask,
    origin_value_mask,
)
from tabu_lab.contracts.topology import GraphTopology

if TYPE_CHECKING:
    from .statistics import NumericNormalizer


class CompilationError(ValueError):
    """Base class for L0 episode contract failures."""


class SplitBeforeCompileError(CompilationError):
    """Raised when raw or unbound data reaches the episode compiler."""


class FitPartitionBindingError(CompilationError):
    """Raised when a recipe is not bound to the declared fit partition."""


class TruthIsolationError(CompilationError):
    """Raised when a forward role could expose unavailable or held-out truth."""


class TopologyBindingError(CompilationError):
    """Raised when typed topology sources conflict or do not match episode rows."""


@dataclass(frozen=True, slots=True)
class CompilationProvenance:
    """Host-side content binding that must not be passed to model forward."""

    dataset_hash: str
    split_manifest_hash: str
    source_view_hash: str
    fit_view_hash: str
    recipe_hash: str
    graph_topology_hash: str | None = None
    numeric_normalizer_hash: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_hash",
            "split_manifest_hash",
            "source_view_hash",
            "fit_view_hash",
            "recipe_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field_name=field_name),
            )
        if self.graph_topology_hash is not None:
            object.__setattr__(
                self,
                "graph_topology_hash",
                require_sha256(
                    self.graph_topology_hash,
                    field_name="graph_topology_hash",
                ),
            )
        if self.numeric_normalizer_hash is not None:
            object.__setattr__(
                self,
                "numeric_normalizer_hash",
                require_sha256(
                    self.numeric_normalizer_hash,
                    field_name="numeric_normalizer_hash",
                ),
            )

    @property
    def provenance_hash(self) -> str:
        return canonical_hash(
            {
                "schema": "tabu.compilation-provenance.v2",
                "dataset_hash": self.dataset_hash,
                "split_manifest_hash": self.split_manifest_hash,
                "source_view_hash": self.source_view_hash,
                "fit_view_hash": self.fit_view_hash,
                "recipe_hash": self.recipe_hash,
                "graph_topology_hash": self.graph_topology_hash,
                "numeric_normalizer_hash": self.numeric_normalizer_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Host-side pair; callers must pass only ``evidence`` to model forward."""

    evidence: EvidenceEpisode
    truth: TruthSidecar
    provenance: CompilationProvenance

    def __post_init__(self) -> None:
        if self.evidence.episode_id != self.truth.episode_id:
            raise ValueError("compiled evidence and truth episode ids must match")
        if bool((self.truth.target_mask & ~self.evidence.target_mask).any()):
            raise ValueError("compiled truth target mask must be an evidence target subset")
        if self.truth.recipe_hash != self.provenance.recipe_hash:
            raise ValueError("compiled truth and provenance recipe hashes must match")
        evidence_topology_hash = (
            self.evidence.graph_topology.topology_hash
            if self.evidence.graph_topology is not None
            else None
        )
        if evidence_topology_hash != self.provenance.graph_topology_hash:
            raise ValueError("compiled topology and provenance topology hashes must match")

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.evidence
        yield self.truth


class EpisodeCompiler:
    """Compile a recipe only after source and fit partitions are bound."""

    def compile(
        self,
        source_view: SplitView,
        recipe: EpisodeRecipe,
        *,
        fit_view: SplitView,
        graph_topology: GraphTopology | None = None,
        numeric_normalizer: NumericNormalizer | None = None,
    ) -> CompilationResult:
        if isinstance(source_view, RawDataset) or not isinstance(source_view, SplitView):
            raise SplitBeforeCompileError(
                "compile requires a SplitView; bind a SplitManifest before episode construction"
            )
        if isinstance(fit_view, RawDataset) or not isinstance(fit_view, SplitView):
            raise SplitBeforeCompileError(
                "fit_view must be a SplitView from the manifest's fit partition"
            )
        if not isinstance(recipe, EpisodeRecipe):
            raise TypeError("compile requires an EpisodeRecipe")

        try:
            source_view.assert_bound()
            fit_view.assert_bound()
        except ValueError as exc:
            raise SplitBeforeCompileError(str(exc)) from exc
        if source_view.manifest.manifest_hash != fit_view.manifest.manifest_hash:
            raise FitPartitionBindingError("source and fit views must share one SplitManifest")
        if fit_view.partition != source_view.manifest.fit_partition:
            raise FitPartitionBindingError(
                "fit_view is not the SplitManifest's declared fit partition"
            )
        try:
            recipe.validate_binding(source_view, fit_view)
        except ValueError as exc:
            raise FitPartitionBindingError(str(exc)) from exc

        if graph_topology is not None and not isinstance(graph_topology, GraphTopology):
            raise TypeError("graph_topology must be GraphTopology or None")
        topology_sources = tuple(
            topology
            for topology in (
                source_view.graph_topology,
                recipe.graph_topology,
                graph_topology,
            )
            if topology is not None
        )
        for topology in topology_sources:
            if topology.node_ids != source_view.row_ids:
                raise TopologyBindingError(
                    "graph_topology node_ids must match source SplitView row_ids"
                )
        topology_hashes = {topology.topology_hash for topology in topology_sources}
        if len(topology_hashes) > 1:
            raise TopologyBindingError("RawDataset, recipe, and explicit graph topology conflict")
        resolved_topology = topology_sources[-1] if topology_sources else None

        raw_origins = source_view.origin_states
        origins = recipe.origin_states
        roles = recipe.forward_roles
        receiver = forward_role_mask(roles, ForwardRole.RECEIVER)
        source = forward_role_mask(roles, ForwardRole.SOURCE)
        target = forward_role_mask(roles, ForwardRole.TARGET)
        raw_value_bearing = origin_value_mask(raw_origins)
        episode_value_bearing = origin_value_mask(origins)
        factual_target_origin = origin_mask(
            origins, OriginState.ARTIFICIAL_MASK
        ) | origin_mask(origins, OriginState.QUERY)
        natural_missing_target = origin_mask(origins, OriginState.NATURAL_MISSING)
        target_origin = factual_target_origin | natural_missing_target
        if bool((source & ~episode_value_bearing).any()):
            raise TruthIsolationError(
                "SOURCE role may only select OBSERVED or INTERVENTION cells"
            )
        truth_target = target & factual_target_origin
        if bool((truth_target & ~raw_value_bearing).any()):
            raise TruthIsolationError(
                "ARTIFICIAL_MASK/QUERY target requires factual source truth"
            )
        if bool((target & natural_missing_target & raw_value_bearing).any()):
            raise TruthIsolationError(
                "NATURAL_MISSING target cannot hide factual source truth"
            )
        if bool((target & ~receiver).any()):
            raise TruthIsolationError("TARGET role must also carry RECEIVER")
        if bool((target & source).any()):
            raise TruthIsolationError("TARGET role must never carry SOURCE")
        if bool((target & ~target_origin).any()):
            raise TruthIsolationError(
                "TARGET bits require ARTIFICIAL_MASK, QUERY, or NATURAL_MISSING origin"
            )
        if not torch.equal(origins[~target], raw_origins[~target]):
            raise TruthIsolationError("non-TARGET origin states cannot be rewritten by a recipe")

        normalizer_config_hash: str | None = None
        normalizer_artifact_hash: str | None = None
        if numeric_normalizer is not None:
            from .statistics import NumericNormalizer

            if not isinstance(numeric_normalizer, NumericNormalizer):
                raise TypeError("numeric_normalizer must be NumericNormalizer or None")
            if numeric_normalizer.statistics.fit_view_hash != fit_view.view_hash:
                raise FitPartitionBindingError(
                    "numeric normalizer was not fitted on the bound fit SplitView"
                )
            fit_target_exclusion = (
                truth_target
                if source_view.view_hash == fit_view.view_hash
                else torch.zeros(fit_view.shape, dtype=torch.bool)
            )
            numeric_normalizer.require_fit_value_mask(
                fit_view,
                excluded_mask=fit_target_exclusion,
            )
            values = numeric_normalizer.transform(source_view).to(
                dtype=source_view.values.dtype
            )
            normalizer_config_hash = numeric_normalizer.statistics.config_hash
            normalizer_artifact_hash = numeric_normalizer.artifact_hash
        else:
            values = source_view.values

        forward_values = torch.where(source, values, torch.zeros_like(values))
        target_values = torch.where(truth_target, values, torch.zeros_like(values))
        recipe_hash = recipe.recipe_hash
        # This identity is intentionally independent of all source values and
        # their hashes.  A model cannot branch on a truth-derived identifier.
        truth_free_identity = {
            "dataset_id": source_view.dataset.dataset_id,
            "source_partition": source_view.partition,
            "fit_partition": fit_view.partition,
            "row_ids": source_view.row_ids,
            "feature_specs": source_view.feature_specs,
            "origin_states": origins,
            "forward_roles": roles,
            "graph_topology": resolved_topology,
            "numeric_normalizer_config_hash": normalizer_config_hash,
        }
        episode_id = f"episode-{canonical_hash(truth_free_identity)[:24]}"
        evidence = EvidenceEpisode(
            episode_id=episode_id,
            dataset_id=source_view.dataset.dataset_id,
            source_partition=source_view.partition,
            fit_partition=fit_view.partition,
            row_ids=source_view.row_ids,
            feature_names=source_view.feature_names,
            feature_specs=source_view.feature_specs,
            forward_values=forward_values,
            origin_states=origins,
            forward_roles=roles,
            graph_topology=resolved_topology,
            metadata={
                "numeric_normalized": numeric_normalizer is not None,
                "numeric_normalizer_config_hash": normalizer_config_hash,
            },
        )
        truth = TruthSidecar(
            episode_id=episode_id,
            recipe_hash=recipe_hash,
            row_ids=source_view.row_ids,
            feature_names=source_view.feature_names,
            target_values=target_values,
            target_mask=truth_target,
            metadata={
                "source_partition": source_view.partition,
                "numeric_normalizer_hash": normalizer_artifact_hash,
            },
        )
        provenance = CompilationProvenance(
            dataset_hash=source_view.dataset.dataset_hash,
            split_manifest_hash=source_view.manifest.manifest_hash,
            source_view_hash=source_view.view_hash,
            fit_view_hash=fit_view.view_hash,
            recipe_hash=recipe_hash,
            graph_topology_hash=(
                resolved_topology.topology_hash if resolved_topology is not None else None
            ),
            numeric_normalizer_hash=normalizer_artifact_hash,
        )
        return CompilationResult(evidence=evidence, truth=truth, provenance=provenance)


def compile_episode(
    source_view: SplitView,
    recipe: EpisodeRecipe,
    *,
    fit_view: SplitView,
    graph_topology: GraphTopology | None = None,
    numeric_normalizer: NumericNormalizer | None = None,
) -> CompilationResult:
    return EpisodeCompiler().compile(
        source_view,
        recipe,
        fit_view=fit_view,
        graph_topology=graph_topology,
        numeric_normalizer=numeric_normalizer,
    )


__all__ = [
    "CompilationError",
    "CompilationProvenance",
    "CompilationResult",
    "EpisodeCompiler",
    "FitPartitionBindingError",
    "SplitBeforeCompileError",
    "TopologyBindingError",
    "TruthIsolationError",
    "compile_episode",
]
