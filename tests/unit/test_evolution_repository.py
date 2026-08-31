from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from tabu_lab.evolution import (
    EvolutionManifestError,
    EvolutionRepository,
    check_or_write_lock,
)

ROOT = Path(__file__).resolve().parents[2]


def _copy_evolution_specs(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "specs" / "evolution", repository / "specs" / "evolution")
    return repository


def _rewrite_yaml(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    mutate(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_repository_resolves_machine_independent_snapshot_identities(tmp_path: Path) -> None:
    repository = EvolutionRepository.load(ROOT)
    copied_root = _copy_evolution_specs(tmp_path)
    copied = EvolutionRepository.load(copied_root, verify_sources=False)

    assert len(repository.nodes) == 35
    assert len(repository.edges) == 5
    assert len(repository.programs) == 11
    assert repository.repository_hash == copied.repository_hash
    for program_ref in repository.programs:
        assert repository.resolve(program_ref).snapshot_hash == copied.resolve(
            program_ref
        ).snapshot_hash
    assert check_or_write_lock(ROOT, write=False) == repository.rendered_lock()


def test_manifest_lock_rejects_same_version_semantic_rewrite(tmp_path: Path) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    objective = (
        copied_root
        / "specs"
        / "evolution"
        / "nodes"
        / "objective-supervised-1.0.0.yaml"
    )

    def mutate(payload: dict[str, object]) -> None:
        objectives = payload["objectives"]
        assert isinstance(objectives, list)
        objectives[0]["weight"] = 0.5

    _rewrite_yaml(objective, mutate)
    with pytest.raises(EvolutionManifestError, match="immutable nodes changed"):
        EvolutionRepository.load(copied_root, verify_sources=False)


def test_graph_compiler_rejects_typed_port_mismatch(tmp_path: Path) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    graph = copied_root / "specs/evolution/nodes/graph-query-base-1.0.0.yaml"

    def mutate(payload: dict[str, object]) -> None:
        external_inputs = payload["external_inputs"]
        assert isinstance(external_inputs, list)
        external_inputs[0]["interface_id"] = "tabu.incompatible-evidence@9"

    _rewrite_yaml(graph, mutate)
    with pytest.raises(EvolutionManifestError, match="typed port mismatch"):
        EvolutionRepository.load(
            copied_root,
            verify_sources=False,
            require_lock=False,
            ignore_lock=True,
        )


def test_graph_compiler_rejects_dangling_reference(tmp_path: Path) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    program = copied_root / "specs/evolution/programs/query-base-grow-1.0.0.yaml"

    def mutate(payload: dict[str, object]) -> None:
        payload["world_mixture"] = {
            "node_id": "tabu.mixture.does-not-exist",
            "version": "1.0.0",
        }

    _rewrite_yaml(program, mutate)
    with pytest.raises(EvolutionManifestError, match="dangling evolution node ref"):
        EvolutionRepository.load(
            copied_root,
            verify_sources=False,
            require_lock=False,
            ignore_lock=True,
        )


def test_graph_compiler_rejects_dependency_cycle(tmp_path: Path) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    graph = copied_root / "specs/evolution/nodes/graph-query-base-1.0.0.yaml"

    def mutate(payload: dict[str, object]) -> None:
        components = payload["components"]
        assert isinstance(components, list)
        components[0]["component"] = {
            "node_id": "tabu.graph.query.base",
            "version": "1.0.0",
        }

    _rewrite_yaml(graph, mutate)
    with pytest.raises(EvolutionManifestError, match="dependency cycle"):
        EvolutionRepository.load(
            copied_root,
            verify_sources=False,
            require_lock=False,
            ignore_lock=True,
        )


def test_unverified_compatibility_edge_is_equivalent_to_no_edge(tmp_path: Path) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    edge = (
        copied_root
        / "specs/evolution"
        / "edges"
        / "query-base-identity-warm-start-1.0.0.yaml"
    )

    def mutate(payload: dict[str, object]) -> None:
        payload["verified"] = False

    _rewrite_yaml(edge, mutate)
    with pytest.raises(EvolutionManifestError, match="must remain absent until verified"):
        EvolutionRepository.load(
            copied_root,
            verify_sources=False,
            require_lock=False,
            ignore_lock=True,
        )


def test_manifest_lock_is_canonical_json() -> None:
    repository = EvolutionRepository.load(ROOT)
    payload = json.loads(repository.rendered_lock())

    assert payload["schema_version"] == "tabu.evolution-manifest-lock.v1"
    assert tuple(payload["nodes"]) == tuple(sorted(payload["nodes"]))


def test_description_only_edit_does_not_change_semantic_node_hash() -> None:
    repository = EvolutionRepository.load(ROOT)
    node = repository.node("tabu.query.base@0.1.0")
    edited = node.model_copy(update={"description": "clarified prose only"})

    assert edited.node_hash == node.node_hash


def test_unrelated_program_addition_does_not_change_existing_snapshot(
    tmp_path: Path,
) -> None:
    copied_root = _copy_evolution_specs(tmp_path)
    baseline = EvolutionRepository.load(
        copied_root,
        verify_sources=False,
        require_lock=False,
        ignore_lock=True,
    )
    source = copied_root / "specs/evolution/programs/query-row-grow-1.0.0.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["program_id"] = "tabu.pretraining.unrelated-candidate"
    payload["version"] = "9.0.0-exercise"
    payload["research_question"] = "Does an unrelated candidate perturb existing identity?"
    (source.parent / "unrelated-candidate-9.0.0.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    expanded = EvolutionRepository.load(
        copied_root,
        verify_sources=False,
        require_lock=False,
        ignore_lock=True,
    )

    assert expanded.repository_hash != baseline.repository_hash
    for program_ref in (
        "tabu.pretraining.query-base@1.0.0",
        "tabu.pretraining.query-row@1.0.0",
    ):
        assert expanded.resolve(program_ref).snapshot_hash == baseline.resolve(
            program_ref
        ).snapshot_hash
