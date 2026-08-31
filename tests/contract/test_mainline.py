from pathlib import Path

import yaml

from tabu_lab.evolution import EvolutionRepository

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_mainline_selects_exact_independent_program_snapshots() -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / "MAINLINE.yaml").read_text())
    assert set(payload) == {
        "schema_version",
        "scope",
        "primary_program",
        "sibling_programs",
    }
    assert payload["schema_version"] == "tabu.mainline.v2"
    assert payload["scope"] == "pretraining"

    repository = EvolutionRepository.load(REPOSITORY_ROOT)
    selected = [payload["primary_program"], *payload["sibling_programs"]]
    refs = {
        f"{item['program_id']}@{item['version']}"
        for item in selected
    }
    assert refs == {
        "tabu.pretraining.query-base@1.4.0",
        "tabu.pretraining.query-row@1.4.0",
    }
    resolved = [repository.resolve(ref) for ref in sorted(refs)]
    assert len({snapshot.snapshot_hash for snapshot in resolved}) == 2
    assert len({snapshot.slots["model_contract"].ref for snapshot in resolved}) == 2
    assert all("state_projection" in snapshot.slots for snapshot in resolved)
    assert {
        snapshot.slots["component_graph"].ref for snapshot in resolved
    } == {
        "tabu.graph.query.base@1.1.0",
        "tabu.graph.query.row@1.1.0",
    }
    assert {
        snapshot.slots["world_mixture"].ref for snapshot in resolved
    } == {"tabu.mixture.supervised-v3@1.1.0"}
    assert {
        snapshot.slots["training_recipe"].ref for snapshot in resolved
    } == {"tabu.training.dgx2-grow-continuation@1.0.0"}
    assert {
        snapshot.slots["evaluation_protocol"].ref for snapshot in resolved
    } == {"tabu.eval.transfer-lanes-terminal@1.0.0"}
    assert {
        snapshot.slots["state_projection"].ref for snapshot in resolved
    } == {
        "tabu.projection.query-base-identity@1.1.0",
        "tabu.projection.query-row-identity@1.1.0",
    }
