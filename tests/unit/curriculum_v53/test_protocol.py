"""Protocol boundaries, portable identity and resume-addressable exposure checks."""

import copy
import hashlib
import json
from collections import Counter
from dataclasses import fields

import pytest
import yaml

from tabu_lab.curriculum_v53 import protocol
from tabu_lab.curriculum_v53.protocol import SCHEMA, load_plan, schedule_entry
from tabu_lab.models.restoration_v53 import BackboneConfig, V53Config

_ACTUAL_SOURCE_DIGEST = protocol._source_digest


def _manifest(tmp_path):
    data = {
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i)] for i in range(6)],
        "splits": {"train": [0, 1, 2], "validation": [3], "test": [4, 5]},
    }
    raw = json.dumps(data).encode()
    (tmp_path / "table.json").write_bytes(raw)
    return {
        "schema": SCHEMA, "experiment_id": "bounded-protocol-check",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"backbone": {"layers": 1, "slots": 4}},
        "tables": [{"id": "tiny", "path": "table.json",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "cohort": "old", "kind": "synthetic"}],
        "probes": [{"name": "fit", "cohorts": ["old"], "partition": "train",
                    "purpose": "fit", "recipe": {"kind": "random_cell", "fraction": 0.2},
                    "masks": 2}],
        "stages": [{"name": "fit-old", "question": "Can the model fit this panel?",
                    "max_updates": 12, "max_seconds": 30,
                    "sampling": [{"cohort": "old", "episodes": 1}],
                    "recipe": {"synthetic": {"kind": "random_cell", "fraction": 0.2}},
                    "optimizer": "adamw", "evaluate_every": 4, "checkpoint_every": 4,
                    "probes": ["fit"]}],
    }


def _save(tmp_path, spec, name="plan.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def fixed_source_identity(monkeypatch):
    # Other authors may edit runner/data concurrently. These tests concern the
    # manifest/data identity; source coverage has its own direct test below.
    monkeypatch.setattr(protocol, "_source_digest", lambda: {"files": {}, "sha256": "source"})


def test_defaults_expand_without_execution_and_preserve_probe_boundaries(tmp_path):
    plan = load_plan(_save(tmp_path, _manifest(tmp_path)))
    assert set(plan.spec["model"]) == {field.name for field in fields(V53Config)}
    assert set(plan.spec["model"]["backbone"]) == {field.name for field in fields(BackboneConfig)}
    assert plan.config.backbone.layers == 1
    assert plan.config.backbone.width == 128
    assert plan.optimizer.betas == (0.9, 0.95)
    assert plan.spec["optimizer"]["betas"] == [0.9, 0.95]
    assert plan.spec["stages"][0]["loss"] == {
        "discrete_weight": 1.0, "state_weights": [0.0, 1.0, 0.0, 0.0],
    }
    assert plan.config.codec_version == "unit_gaussian_v2"
    assert plan.spec["stages"][0]["recipe"]["synthetic"]["numeric_query_guard"] == {"kind": "none"}
    assert plan.summary["status"] == "local_unissued"
    assert not plan.summary["execution_started"]
    assert "held-out" in plan.summary["probe_boundary"]
    assert str(tmp_path) not in json.dumps(plan.identity)
    assert str(tmp_path) not in json.dumps(plan.summary)
    json.dumps(plan.spec, allow_nan=False)


def test_source_identity_covers_new_and_reused_implementation_bytes():
    source = _ACTUAL_SOURCE_DIGEST()
    expected = {
        "curriculum_v53/protocol.py", "curriculum_v53/data.py",
        "models/restoration_v53/model.py", "models/restoration_v53/training.py",
        "models/restoration_v53/answers.py", "models/restoration_v53/codec_versions.py",
        "models/restoration/contracts.py", "models/restoration/answers.py",
        "restoration_masking.py", "restoration_optimizers.py", "tar_data.py", "cli.py",
    }
    assert expected <= set(source["files"])
    assert all(not key.startswith("/") for key in source["files"])
    assert len(source["sha256"]) == 64
    assert all(len(value) == 64 for value in source["files"].values())


@pytest.mark.parametrize("mutate,match", [
    (lambda s: s.update(typo=1), "unknown fields"),
    (lambda s: s["model"].update(typo=1), "unknown fields"),
    (lambda s: s["model"]["backbone"].update(typo=1), "unknown fields"),
    (lambda s: s["seeds"].update(model=True), "integer"),
    (lambda s: s["seeds"].pop("evaluation"), "missing fields"),
    (lambda s: s["tables"].append(copy.deepcopy(s["tables"][0])), "duplicate table"),
    (lambda s: s["probes"].append(copy.deepcopy(s["probes"][0])), "duplicate probe"),
    (lambda s: s["stages"].append(copy.deepcopy(s["stages"][0])), "duplicate stage"),
    (lambda s: s["stages"][0]["sampling"][0].update(cohort="missing"), "sampling cohort"),
    (lambda s: s["stages"][0].update(probes=["missing"]), "unknown probe"),
    (lambda s: s["stages"][0].update(recipe={}), "missing fields"),
    (lambda s: s["stages"][0].update(max_seconds=float("nan")), "finite number"),
    (lambda s: s["stages"][0].update(max_updates=False), "integer"),
    (lambda s: s["model"].update(slope_source="feature"), "feature slope"),
    (lambda s: s["tables"][0].update(role="probe"), "probe-only"),
])
def test_strict_manifest_rejects_silent_drift(tmp_path, mutate, match):
    spec = _manifest(tmp_path)
    mutate(spec)
    with pytest.raises(ValueError, match=match):
        load_plan(_save(tmp_path, spec))


@pytest.mark.parametrize("optimizer", [
    {"unknown": 2}, {"betas": [0.9, 1]}, {"betas": [True, 0.9]},
    {"learning_rate": 0}, {"weight_decay": -1}, {"eps": float("inf")},
    {"grad_clip": False}, {"muon_momentum": 1}, {"muon_nesterov": 1},
    {"muon_ns_steps": 0}, {"muon_adjust_lr_fn": "unknown"},
])
def test_optimizer_configuration_is_validated_before_execution(tmp_path, optimizer):
    spec = _manifest(tmp_path)
    spec["optimizer"] = optimizer
    with pytest.raises(ValueError):
        load_plan(_save(tmp_path, spec))


def test_stage_loss_and_optimizer_transition_are_explicit(tmp_path):
    spec = _manifest(tmp_path)
    spec["stages"][0]["loss"] = {"discrete_weight": 16, "state_weights": [0.05, 0.95, 0, 0]}
    second = copy.deepcopy(spec["stages"][0])
    second.update(name="continue", optimizer="muon")
    spec["stages"].append(second)
    plan = load_plan(_save(tmp_path, spec))
    assert plan.spec["stages"][0]["loss"]["discrete_weight"] == 16.0
    assert plan.spec["stages"][1]["optimizer"] == "muon"
    spec["stages"][0]["optimizer"] = "muon"  # A first-stage Muon run is legal.
    load_plan(_save(tmp_path, spec))
    spec["stages"][1]["optimizer"] = "adamw"
    with pytest.raises(ValueError, match="reversal"):
        load_plan(_save(tmp_path, spec))


@pytest.mark.parametrize("loss", [
    {"discrete_weight": 0}, {"state_weights": [0, 0, 0, 0]},
    {"state_weights": [True, 0, 0, 0]}, {"state_weights": [1, -1, 0, 0]},
    {"state_weights": [1, 0, 0, float("nan")]}, {"unknown": 1},
    {"state_weights": [0, 0, 1, 0]}, {"state_weights": [0, 0, 0, 1]},
    {"state_weights": [0.05, 0.95, 1, 0]},
])
def test_stage_loss_rejects_invalid_weights(tmp_path, loss):
    spec = _manifest(tmp_path)
    spec["stages"][0]["loss"] = loss
    with pytest.raises(ValueError):
        load_plan(_save(tmp_path, spec))


@pytest.mark.parametrize("weights", [[1, 0, 0, 0], [0, 1, 0, 0]])
def test_loss_allows_explicit_retained_or_query_only(tmp_path, weights):
    spec = _manifest(tmp_path)
    spec["stages"][0]["loss"] = {"state_weights": weights}
    plan = load_plan(_save(tmp_path, spec))
    assert plan.spec["stages"][0]["loss"]["state_weights"] == weights


def _reserved_probe(partition="validation", purpose="validation"):
    return {"name": "heldout", "cohorts": ["old"], "partition": partition, "purpose": purpose,
            "recipe": {"kind": "supervised_row", "fraction": 0.2}, "masks": 1}


def test_only_validation_can_gate_and_final_test_is_never_a_stage_probe(tmp_path):
    spec = _manifest(tmp_path)
    spec["probes"].append(_reserved_probe())
    spec["stages"][0]["probes"].append("heldout")
    spec["stages"][0]["gate"] = {
        "probe": "heldout", "metric": "query_numeric_mse", "mode": "min", "threshold": 1.0,
    }
    load_plan(_save(tmp_path, spec))
    spec["stages"][0]["gate"]["probe"] = "fit"
    with pytest.raises(ValueError, match="validation probes"):
        load_plan(_save(tmp_path, spec))
    del spec["stages"][0]["gate"]
    spec["probes"][1] = _reserved_probe("test", "transfer")
    with pytest.raises(ValueError, match="retrospective"):
        load_plan(_save(tmp_path, spec))
    spec["probes"][1]["purpose"] = "retrospective"
    load_plan(_save(tmp_path, spec))
    spec["probes"][1]["purpose"] = "final_test"
    with pytest.raises(ValueError, match="independent evaluate"):
        load_plan(_save(tmp_path, spec))
    spec["stages"][0]["probes"] = ["fit"]
    load_plan(_save(tmp_path, spec))  # Registered for later independent evaluation.


@pytest.mark.parametrize("change", [
    {"masks": 2}, {"recipe": {"kind": "random_cell", "fraction": 0.2}},
    {"recipe": {"kind": "supervised_row", "fraction": 0.2,
                "numeric_query_guard": {"kind": "std_iqr_column", "max_std_iqr_ratio": 2}}},
])
def test_reserved_probes_cannot_be_repeated_or_tail_selected(tmp_path, change):
    spec = _manifest(tmp_path)
    probe = _reserved_probe()
    probe.update(change)
    spec["probes"].append(probe)
    with pytest.raises(ValueError):
        load_plan(_save(tmp_path, spec))


def test_reserved_probe_requires_existing_partition_and_full_context(tmp_path):
    spec = _manifest(tmp_path)
    spec["probes"].append(_reserved_probe())
    spec["tables"][0]["window_rows"] = 3
    with pytest.raises(ValueError, match="full context"):
        load_plan(_save(tmp_path, spec))
    del spec["tables"][0]["window_rows"]
    data_path = tmp_path / "table.json"
    data = json.loads(data_path.read_text())
    data["splits"] = {"train": [0, 1, 2, 3], "test": [4, 5]}
    raw = json.dumps(data).encode()
    data_path.write_bytes(raw)
    spec["tables"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="no validation rows"):
        load_plan(_save(tmp_path, spec))


def test_digest_binds_objective_sampling_and_bytes_but_allows_relocation(tmp_path):
    spec = _manifest(tmp_path)
    original = load_plan(_save(tmp_path, spec))
    relocated_dir = tmp_path / "relocated"
    relocated_dir.mkdir()
    (relocated_dir / "copy.json").write_bytes((tmp_path / "table.json").read_bytes())
    moved = copy.deepcopy(spec)
    moved["tables"][0]["path"] = "copy.json"
    relocated = load_plan(_save(relocated_dir, moved))
    assert original.identity == relocated.identity
    changed = copy.deepcopy(spec)
    changed["stages"][0]["loss"] = {"discrete_weight": 2}
    assert load_plan(_save(tmp_path, changed)).identity != original.identity
    changed = copy.deepcopy(spec)
    changed["stages"][0]["sampling"][0]["episodes"] = 2
    assert load_plan(_save(tmp_path, changed)).identity != original.identity
    (tmp_path / "table.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_plan(_save(tmp_path, spec))


def test_schedule_has_fair_replay_and_resume_uses_absolute_address(tmp_path):
    spec = _manifest(tmp_path)
    template = spec["tables"][0]
    spec["tables"] = [dict(template, id=f"old-{i:03d}") for i in range(120)]
    spec["tables"] += [dict(template, id=f"new-{i:03d}", cohort="new") for i in range(12)]
    spec["stages"][0]["sampling"] = [
        {"cohort": "old", "episodes": 30}, {"cohort": "new", "episodes": 12},
    ]
    spec["stages"][0]["max_updates"] = 10**12
    plan = load_plan(_save(tmp_path, spec))
    draws = [schedule_entry(plan, 0, cursor) for cursor in range(4 * 42)]
    old = [(table.name, episode) for table, episode in draws if table.cohort == "old"]
    assert len(old) == 120
    assert len({name for name, _ in old}) == 120
    assert {episode for _, episode in old} == {0}
    new = Counter(table.name for table, _ in draws if table.cohort == "new")
    assert set(new.values()) == {4}
    for cycle in range(4):
        assert Counter(table.cohort for table, _ in draws[cycle * 42:(cycle + 1) * 42]) == {
            "old": 30, "new": 12,
        }
    # Large cursor is direct arithmetic; a linear scan would make this infeasible.
    table, index = schedule_entry(plan, 0, 10**11)
    repeated, repeated_index = schedule_entry(plan, 0, 10**11)
    assert (table.name, index) == (repeated.name, repeated_index)
    assert index > 10**8
    suffix = [schedule_entry(plan, 0, cursor) for cursor in range(75, 85)]
    assert [(t.name, i) for t, i in suffix] == [(t.name, i) for t, i in draws[75:85]]


def test_schedule_never_draws_probe_role_and_addresses_repeated_tables_distinctly(tmp_path):
    spec = _manifest(tmp_path)
    spec["tables"].append(dict(spec["tables"][0], id="probe-only", role="probe"))
    spec["stages"][0]["sampling"][0]["episodes"] = 3
    plan = load_plan(_save(tmp_path, spec))
    draws = [schedule_entry(plan, 0, cursor) for cursor in range(12)]
    assert {table.name for table, _ in draws} == {"tiny"}
    assert sorted(index for _, index in draws) == list(range(12))
    for index, cursor in ((-1, 0), (1, 0), (0, -1), (0, 12)):
        with pytest.raises(ValueError):
            schedule_entry(plan, index, cursor)
