"""CPU-only migration tests; no training, SSH, accelerator or external telemetry."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tabu_lab.curriculum_v53 import artifacts, protocol, runner, loss_replay


spec = importlib.util.spec_from_file_location("migrate_replay", Path(__file__).with_name("migrate_replay_v2.py"))
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def write_payload(path, payload):
    torch.save(payload, path)
    digest = artifacts.sha256(path)
    path.with_suffix(".json").write_text(json.dumps({"sha256": digest}))
    return digest


@pytest.fixture
def prepared(tmp_path):
    data = {"features": [{"kind": "numeric"}, {"kind": "numeric"}],
            "values": [[float(i), float(i * i)] for i in range(8)],
            "splits": {"train": list(range(6)), "validation": [6], "test": [7]}}
    raw = json.dumps(data).encode()
    (tmp_path / "table.json").write_bytes(raw)
    manifest = {
        "schema": protocol.V54_SCHEMA, "experiment_id": "replay-migration-test",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"size": "nano"},
        "tables": [{"id": f"table_{index:03}", "kind": "synthetic", "cohort": "old120",
                    "path": "table.json", "sha256": hashlib.sha256(raw).hexdigest()}
                   for index in range(120)],
        "probes": [{"name": "fit", "cohorts": ["old120"], "partition": "train",
                    "purpose": "fit", "recipe": {"fraction": 0.25}, "masks": 1}],
        "stages": [{"name": "fit", "question": "Does migration preserve state?",
                    "max_updates": 474, "max_seconds": 1200,
                    "sampling": [{"cohort": "old120", "episodes": 120}],
                    "recipe": {"synthetic": {"fraction": 0.25}}, "optimizer": "adamw",
                    "evaluate_every": 120, "checkpoint_every": 120, "probes": ["fit"],
                    "loss_replay": {"kind": migration.POLICY,
                                    "normal_max_updates": 360, "start_normal_cursor": 120,
                                    "start_extra_updates": 30}}],
    }
    manifest_path = tmp_path / "new.json"
    manifest_path.write_text(json.dumps(manifest))
    new = protocol.load_v54_plan(manifest_path)
    old = {"spec": copy.deepcopy(new.spec), "identity": copy.deepcopy(new.identity)}
    old["spec"]["experiment_id"] = "balanced-parent-test"
    old["spec"]["stages"][0]["loss_replay"] = dict(kind=migration.PARENT_POLICY, normal_max_updates=360, start_normal_cursor=0)
    old["spec"]["stages"][0]["max_updates"] = 450
    for key in migration.MUTABLE_SOURCE:
        old["identity"]["source"]["files"][key] = "a" * 64
    old["identity"]["source"]["sha256"] = protocol._digest(old["identity"]["source"]["files"])
    portable = copy.deepcopy(old["spec"])
    for table in portable["tables"]:
        table.pop("path")
    old["identity"]["manifest_sha256"] = protocol._digest(portable)
    old["identity"].pop("sha256")
    old["identity"]["sha256"] = protocol._digest(old["identity"])
    runtime = {"device": "cpu", "dtype": "float32", "test_only": True}
    state = runner._new_state(old["spec"]["stages"])
    state.update(cursor=150, update=150, phase="train", stage_seconds=[8.0],
                 total_seconds=8.5, evaluated_cursor=0,
                 exposure={f"table_{i:03}": {"updates": 1} for i in range(120)})
    old_replay = loss_replay.new_state([f"table_{i:03}" for i in range(120)])
    for i in range(120):
        old_replay = loss_replay.record_normal(old_replay, f"table_{i:03}", float(i))
    for _ in range(30):
        old_replay = loss_replay.record_extra(old_replay)
    state["loss_replay"] = old_replay
    for table, count in old_replay["extra_by_table"].items():
        state["exposure"][table]["updates"] += count
    payload = {
        "schema": "tabu.curriculum.v54.checkpoint.v1", "purpose": "training",
        "identity": old["identity"], "model_config": new.config.as_dict(),
        "runtime": runtime, "model": {"weight": torch.tensor([1., -2.])},
        "optimizer": {"state": {0: {"step": torch.tensor(150.),
                                    "exp_avg": torch.tensor([0.2, 0.3]),
                                    "exp_avg_sq": torch.tensor([0.1, 0.7])}},
                      "param_groups": [{"lr": 1e-4, "params": [0]}]},
        "optimizer_kind": "adamw", "rng": artifacts.rng_state(),
        "state": state, "lineage": [{"mode": "weights_only_initialization"}],
    }
    attempt = tmp_path / "parent"
    attempt.mkdir()
    parent = attempt / "checkpoint-progress.pt"
    digest = write_payload(parent, payload)
    terminal = {"identity": old["identity"], "checkpoint_sha256": digest,
                "update": 150, "durable_update": 150, "outcome": "stopped",
                "total_seconds": 8.8, "stage_seconds": [8.3]}
    (attempt / "terminal.json").write_text(json.dumps(terminal))
    return dict(old=old, new=new, parent=parent, payload=payload,
                runtime=runtime, destination=tmp_path / "migration", terminal=terminal)


def invoke(p):
    return migration.migrate(p["old"], p["new"], p["parent"], p["destination"], p["runtime"])


def test_migration_preserves_entire_parent_state_and_roundtrips(prepared):
    p = prepared
    original = p["parent"].read_bytes()
    final, receipt = invoke(p)
    changed, digest = artifacts.load_checkpoint(final)
    assert receipt["checkpoint_sha256"] == digest == final.stem
    assert receipt["limits"] == dict(normal_max_updates=360, start_normal_cursor=120,
        start_extra_updates=30, remaining_normal_updates=240, additional_replay_updates=84, actual_max_updates=474)
    for key in ("model", "optimizer", "rng", "runtime", "model_config"):
        migration.exact_tree(changed[key], p["payload"][key])
    replay = changed["state"].pop("loss_replay")
    old_state = copy.deepcopy(p["payload"]["state"])
    old_replay = old_state.pop("loss_replay")
    migration.exact_tree(changed["state"], old_state)
    assert replay["normal_cursor"] == 120 and replay["extra_updates"] == 30
    assert replay["base_extra_by_table"] == old_replay["extra_by_table"] == replay["extra_by_table"]
    assert replay["kind"] == migration.POLICY
    assert changed["lineage"][-1]["mode"] == "strategy_change_preserve_optimizer_rng"
    assert p["parent"].read_bytes() == original
    assert json.loads(final.with_suffix(".json").read_text())["sha256"] == digest
    with pytest.raises(FileExistsError):
        invoke(p)


def test_non_boundary_checkpoint_rejected_without_output(prepared):
    p = prepared
    p["payload"]["state"].update(cursor=121, update=121)
    write_payload(p["parent"], p["payload"])
    with pytest.raises(ValueError, match="entire normal-plus-replay cycle"):
        invoke(p)
    assert not p["destination"].exists()


def test_json_lists_and_equivalent_loader_tuples_are_same_contract(prepared):
    p = prepared
    p["old"]["spec"] = json.loads(json.dumps(p["old"]["spec"]))
    weights = p["new"].spec["stages"][0]["loss"]["state_weights"]
    p["new"].spec["stages"][0]["loss"]["state_weights"] = tuple(weights)
    for key, value in p["new"].spec["optimizer"].items():
        if isinstance(value, list):
            p["new"].spec["optimizer"][key] = tuple(value)
    final, receipt = invoke(p)
    assert final.is_file() and receipt["outcome"] == "passed"
    changed, _ = artifacts.load_checkpoint(final)
    migration.exact_tree(changed["optimizer"], p["payload"]["optimizer"])


@pytest.mark.parametrize("drift", ["model", "data", "source", "new_source", "protocol"])
def test_unapproved_drift_rejected(prepared, drift):
    p = prepared
    if drift == "model":
        p["new"].spec["model"]["backbone"]["heads"] = 8
    elif drift == "data":
        p["new"].identity["data_sha256"] = "b" * 64
    elif drift == "source":
        p["new"].identity["source"]["files"]["models/restoration_v54/model.py"] = "b" * 64
    elif drift == "new_source":
        p["new"].identity["source"]["files"]["unexpected.py"] = "b" * 64
    else:
        p["new"].spec["stages"][0]["evaluate_every"] = 240
    with pytest.raises(ValueError, match="unapproved"):
        invoke(p)
    assert not p["destination"].exists()


@pytest.mark.parametrize("field,value", [("max_updates", 450),
    ("normal_max_updates", 480), ("start_normal_cursor", 0), ("start_extra_updates", 0)])
def test_budget_drift_rejected(prepared, field, value):
    p = prepared
    stage = p["new"].spec["stages"][0]
    target = stage if field == "max_updates" else stage["loss_replay"]
    target[field] = value
    with pytest.raises(ValueError, match="budget|cursor"):
        invoke(p)
    assert not p["destination"].exists()


def test_non_durable_parent_rejected(prepared):
    p = prepared
    p["terminal"]["update"] = 121
    (p["parent"].parent / "terminal.json").write_text(json.dumps(p["terminal"]))
    with pytest.raises(ValueError, match="cleanly stopped durable"):
        invoke(p)


def test_resolved_identity_and_runtime_must_match(prepared):
    p = prepared
    p["old"]["identity"] = copy.deepcopy(p["old"]["identity"])
    p["old"]["identity"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="frozen old resolved"):
        invoke(p)
    p["old"]["identity"] = p["payload"]["identity"]
    p["runtime"] = {"device": "mps"}
    with pytest.raises(ValueError, match="runtime drift"):
        invoke(p)


def test_pending_replay_cycle_cannot_be_discarded(prepared):
    p = prepared
    names = list(p["payload"]["state"]["loss_replay"]["extra_by_table"])
    replay = loss_replay.new_state(names)
    for i, table in enumerate(names):
        replay = loss_replay.record_normal(replay, table, float(i))
    for _ in range(29):
        replay = loss_replay.record_extra(replay)
    p["payload"]["state"].update(cursor=149, update=149, loss_replay=replay)
    write_payload(p["parent"], p["payload"])
    with pytest.raises(ValueError, match="entire normal-plus-replay cycle"):
        invoke(p)
    assert not p["destination"].exists()
