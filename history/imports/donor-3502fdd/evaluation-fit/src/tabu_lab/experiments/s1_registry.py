"""Closed registry for the nine frozen S1 synthetic fit experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .corpus import FitEpisodeCorpus
from .s1_table_synthetic import build_s1_table_corpus
from .s1_topology import (
    GraphSyntheticRecipe,
    RecSyntheticRecipe,
    build_s1_graph_corpus,
    build_s1_rec_corpus,
)


class S1GeneratorSource(StrEnum):
    TABLE = "s1_table_synthetic"
    TOPOLOGY = "s1_topology"


class S1Recipe(StrEnum):
    LATENT_MIXED_COMPLETION = "latent_mixed_completion"
    COMPOSITIONAL_XOR_LABEL = "compositional_xor_label"
    JOINT_COMPLETION_COMPOSITIONAL_XOR = "joint_completion_compositional_xor"
    GRAPH_COMMUNITY = GraphSyntheticRecipe.COMMUNITY.value
    GRAPH_DIFFUSION = GraphSyntheticRecipe.DIFFUSION.value
    REC_RATING = RecSyntheticRecipe.RATING.value
    REC_PREFERENCE = RecSyntheticRecipe.PREFERENCE.value
    BASE_COMPLETION = "base_completion"
    BASE_SUPERVISED_REGRESSION = "base_supervised_regression"
    BASE_SUPERVISED_CLASSIFICATION = "base_supervised_classification"


@dataclass(frozen=True, slots=True)
class S1ExperimentRegistration:
    experiment_id: str
    contract_id: str
    recipe: S1Recipe
    generator_source: S1GeneratorSource
    generator_entrypoint: str
    adapter_id: str
    adapter_version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.experiment_id.startswith("S1-"):
            raise ValueError("S1 experiment ids must start with S1-")
        if self.contract_id not in {
            "tabuf",
            "tabu.unit_row",
            "tabu.unit_pair",
            "tabul",
            "tabufl",
            "tabu4graph",
            "tabu4rec",
            "tabu.cell.base",
        }:
            raise ValueError("S1 registration requires a buildable contract")
        if not self.generator_entrypoint.strip() or not self.adapter_id.strip():
            raise ValueError("S1 generator and adapter identifiers cannot be blank")
        if self.generator_source is S1GeneratorSource.TABLE and self.contract_id in {
            "tabu4graph",
            "tabu4rec",
        }:
            raise ValueError("topology contracts must use the topology generator")
        if self.generator_source is S1GeneratorSource.TOPOLOGY and self.contract_id not in {
            "tabu4graph",
            "tabu4rec",
        }:
            raise ValueError("table contracts must use the table generator")

    @property
    def source_uri(self) -> str:
        return (
            f"pkg://tabu_lab.experiments.{self.generator_source.value}#{self.generator_entrypoint}"
        )

    @property
    def source_hash(self) -> str:
        filename = f"{self.generator_source.value}.py"
        return hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()

    def build_corpus(self) -> FitEpisodeCorpus:
        if self.generator_source is S1GeneratorSource.TABLE:
            if self.contract_id == "tabu.cell.base" and self.recipe is S1Recipe.BASE_COMPLETION:
                return build_s1_table_corpus(self.contract_id)
            if self.contract_id == "tabu.cell.base" and self.recipe is S1Recipe.BASE_SUPERVISED_REGRESSION:
                from .s1_table_synthetic import build_s1_base_supervised_corpus
                return build_s1_base_supervised_corpus("regression")
            if self.contract_id == "tabu.cell.base" and self.recipe is S1Recipe.BASE_SUPERVISED_CLASSIFICATION:
                from .s1_table_synthetic import build_s1_base_supervised_corpus
                return build_s1_base_supervised_corpus("classification")
            return build_s1_table_corpus(self.contract_id)
        if self.contract_id == "tabu4graph":
            return build_s1_graph_corpus(self.recipe.value)
        if self.contract_id == "tabu4rec":
            return build_s1_rec_corpus(self.recipe.value)
        raise AssertionError("closed S1 registration union was violated")


S1_EXPERIMENT_REGISTRATIONS = (
    S1ExperimentRegistration(
        experiment_id="S1-001-tabuf-latent-mixed-v1",
        contract_id="tabuf",
        recipe=S1Recipe.LATENT_MIXED_COMPLETION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_completion_corpus",
        adapter_id="tabu-s1-table-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-002-tabu-unit-row-latent-mixed-v1",
        contract_id="tabu.unit_row",
        recipe=S1Recipe.LATENT_MIXED_COMPLETION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_completion_corpus",
        adapter_id="tabu-s1-table-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-003-tabu-unit-pair-latent-mixed-v1",
        contract_id="tabu.unit_pair",
        recipe=S1Recipe.LATENT_MIXED_COMPLETION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_completion_corpus",
        adapter_id="tabu-s1-table-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-004-tabul-compositional-xor-v1",
        contract_id="tabul",
        recipe=S1Recipe.COMPOSITIONAL_XOR_LABEL,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_supervised_corpus",
        adapter_id="tabu-s1-table-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-005-tabufl-joint-compositional-xor-v1",
        contract_id="tabufl",
        recipe=S1Recipe.JOINT_COMPLETION_COMPOSITIONAL_XOR,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_supervised_corpus",
        adapter_id="tabu-s1-table-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-006-tabu4graph-community-v1",
        contract_id="tabu4graph",
        recipe=S1Recipe.GRAPH_COMMUNITY,
        generator_source=S1GeneratorSource.TOPOLOGY,
        generator_entrypoint="build_s1_graph_corpus",
        adapter_id="tabu-s1-topology-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-007-tabu4graph-diffusion-v1",
        contract_id="tabu4graph",
        recipe=S1Recipe.GRAPH_DIFFUSION,
        generator_source=S1GeneratorSource.TOPOLOGY,
        generator_entrypoint="build_s1_graph_corpus",
        adapter_id="tabu-s1-topology-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-008-tabu4rec-rating-v1",
        contract_id="tabu4rec",
        recipe=S1Recipe.REC_RATING,
        generator_source=S1GeneratorSource.TOPOLOGY,
        generator_entrypoint="build_s1_rec_corpus",
        adapter_id="tabu-s1-topology-synthetic",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-010-tabu-cell-base-completion-v1",
        contract_id="tabu.cell.base",
        recipe=S1Recipe.BASE_COMPLETION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_completion_corpus",
        adapter_id="tabu-s1-table-synthetic-base",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-011-tabu-cell-base-supervised-regression-v1",
        contract_id="tabu.cell.base",
        recipe=S1Recipe.BASE_SUPERVISED_REGRESSION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_base_supervised_corpus",
        adapter_id="tabu-s1-table-synthetic-base",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-012-tabu-cell-base-supervised-classification-v1",
        contract_id="tabu.cell.base",
        recipe=S1Recipe.BASE_SUPERVISED_CLASSIFICATION,
        generator_source=S1GeneratorSource.TABLE,
        generator_entrypoint="build_s1_base_supervised_corpus",
        adapter_id="tabu-s1-table-synthetic-base",
    ),
    S1ExperimentRegistration(
        experiment_id="S1-009-tabu4rec-preference-v1",
        contract_id="tabu4rec",
        recipe=S1Recipe.REC_PREFERENCE,
        generator_source=S1GeneratorSource.TOPOLOGY,
        generator_entrypoint="build_s1_rec_corpus",
        adapter_id="tabu-s1-topology-synthetic",
    ),
)

_BY_EXPERIMENT_ID = {
    registration.experiment_id: registration for registration in S1_EXPERIMENT_REGISTRATIONS
}
if len(_BY_EXPERIMENT_ID) != len(S1_EXPERIMENT_REGISTRATIONS):  # pragma: no cover
    raise RuntimeError("S1 experiment ids must be unique")


def list_s1_registrations() -> tuple[S1ExperimentRegistration, ...]:
    return S1_EXPERIMENT_REGISTRATIONS


def get_s1_registration(experiment_id: str) -> S1ExperimentRegistration:
    try:
        return _BY_EXPERIMENT_ID[experiment_id]
    except KeyError as exc:
        raise KeyError(f"unknown S1 experiment id: {experiment_id!r}") from exc


def build_registered_s1_corpus(experiment_id: str) -> FitEpisodeCorpus:
    return get_s1_registration(experiment_id).build_corpus()


__all__ = [
    "S1_EXPERIMENT_REGISTRATIONS",
    "S1ExperimentRegistration",
    "S1GeneratorSource",
    "S1Recipe",
    "build_registered_s1_corpus",
    "get_s1_registration",
    "list_s1_registrations",
]
