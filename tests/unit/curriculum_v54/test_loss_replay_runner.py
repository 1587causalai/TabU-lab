"""Exercise the real runner/checkpoints with a cheap deterministic optimizer step."""
import copy
import hashlib
import json
import random

import pytest
import torch

from tabu_lab.curriculum_v53 import loss_replay, runner
from tabu_lab.curriculum_v53.artifacts import load_checkpoint, restore_rng, save_checkpoint
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v54_plan, schedule_entry
from tabu_lab.restoration_optimizers import adamw


def make_plan(tmp_path, *, start=0, kind=loss_replay.V1, start_extra=0):
    raw = json.dumps({"features": [{"kind": "numeric"}, {"kind": "numeric"}],
                      "values": [[float(i), float(i * i)] for i in range(8)],
                      "splits": {"train": list(range(6)), "validation": [6], "test": [7]}}).encode()
    (tmp_path / "table.json").write_bytes(raw)
    spec = {
        "schema": "tabu.curriculum.v54.v1", "experiment_id": "replay-runner-fixture",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"size": "nano", "backbone": {"layers": 1, "slots": 4, "ff_width": 16},
                  "regression_width": 4, "center_chunk_size": 4},
        "tables": [{"id": f"table-{i:03}", "kind": "synthetic", "cohort": "old120",
                    "path": "table.json", "sha256": hashlib.sha256(raw).hexdigest()}
                   for i in range(120)],
        "probes": [],
        "stages": [{"name": "fit", "question": "Does replay preserve normal scheduling?",
                    "max_updates": (240 + start_extra
                                    + (240 - start) // 120 * loss_replay.extras_per_cycle(kind)),
                    "max_seconds": 120,
                    "sampling": [{"cohort": "old120", "episodes": 120}],
                    "recipe": {"synthetic": {"kind": "supervised_row", "fraction": .25}},
                    "optimizer": "adamw", "evaluate_every": 150, "checkpoint_every": 150,
                    "probes": [], "loss_replay": {"kind": kind,
                                                  "normal_max_updates": 240,
                                                  "start_normal_cursor": start}}],
    }
    if kind == loss_replay.V2:
        spec["stages"][0]["loss_replay"]["start_extra_updates"] = start_extra
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return load_v54_plan(path)


def cheap_step(model, optimizer, plan, table, recipe, index, device, loss_config, *, namespace):
    rank = int(table.name.split("-")[-1])
    optimizer.zero_grad(set_to_none=True)
    parameter = next(model.parameters())
    parameter.grad = torch.full_like(parameter, (rank + 1) * .0001)
    optimizer.step()
    # Checkpoint tests include both RNG streams even though real episode seeds
    # are directly addressable and use a disjoint namespace for extra episodes.
    torch.rand(1)
    random.random()
    return {"table": table.name, "cohort": table.cohort, "table_episode_index": index,
            "loss": float(rank), "seconds": 0., "namespace": namespace,
            "episode": {"query_count": 1, "row_ids": [0], "query_addresses": [[0, 1]]}}


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("stop", [119, 120, 125, 126, 149, 150, 151, 299, 300])
def test_full_and_split_resume_have_same_optimizer_exposure_and_normal_stream(
        tmp_path, monkeypatch, stop):
    monkeypatch.setattr(runner, "train_step", cheap_step)
    plan = make_plan(tmp_path)
    full = runner.run(plan, tmp_path / "full")
    assert full["outcome"] == "completed", full
    part = runner.run(plan, tmp_path / "part", max_updates_this_invocation=stop)
    assert part["outcome"] == "stopped", part
    resumed = runner.run(plan, tmp_path / "resumed",
                         resume=tmp_path / "part/checkpoint-progress.pt")
    assert resumed["outcome"] == "completed", resumed
    a, _ = load_checkpoint(tmp_path / "full/checkpoint-progress.pt")
    b, _ = load_checkpoint(tmp_path / "resumed/checkpoint-progress.pt")
    for key in ("model", "optimizer", "rng"):
        exact(a[key], b[key])
    for key in ("loss_replay", "exposure", "update", "cursor", "phase"):
        exact(a["state"][key], b["state"][key])
    assert resumed["update"] == 300
    assert resumed["normal_update"] == 240
    assert resumed["extra_updates"] == 60
    rows = [json.loads(line) for line in (tmp_path / "full/updates.jsonl").read_text().splitlines()]
    normals = [row for row in rows if row["update_kind"] == "normal"]
    for cursor, row in enumerate(normals):
        table, index = schedule_entry(plan, 0, cursor)
        assert (row["table"], row["table_episode_index"], row["namespace"]) == (
            table.name, index, "fit")
    extras = [row for row in rows if row["update_kind"] != "normal"]
    assert all(row["namespace"] == "fit/loss_replay_v1" for row in extras)
    for i, table in enumerate(sorted(resumed["exposure"])):
        entry = resumed["exposure"][table]
        assert entry["normal_updates"] == 2
        assert entry["extra_updates"] == (4 if i >= 114 else 2 if i >= 96 else 0)


def test_migrated_normal_boundary_and_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "train_step", cheap_step)
    new_plan = make_plan(tmp_path, start=120)
    old_spec = copy.deepcopy(new_plan.spec)
    old_spec["stages"][0].pop("loss_replay")
    old_spec["stages"][0]["max_updates"] = 240
    old_path = tmp_path / "uniform.json"
    old_path.write_text(json.dumps(old_spec))
    old_plan = load_v54_plan(old_path)
    parent = runner.run(old_plan, tmp_path / "uniform", max_updates_this_invocation=120)
    assert parent["outcome"] == "stopped", parent
    payload, _ = load_checkpoint(tmp_path / "uniform/checkpoint-progress.pt")
    model = make_model(new_plan).double()
    model.load_state_dict(payload["model"])
    optimizer = adamw(model, new_plan.optimizer)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    migrated = copy.deepcopy(payload["state"])
    migrated["loss_replay"] = loss_replay.new_state([table.name for table in new_plan.tables], 120)
    checkpoint = tmp_path / "migration/checkpoint-progress.pt"
    checkpoint.parent.mkdir()
    save_checkpoint(checkpoint, plan=new_plan, model=model, optimizer=optimizer,
                    state=migrated, runtime=payload["runtime"],
                    lineage=[{"mode": "test_explicit_strategy_migration"}])
    result = runner.run(new_plan, tmp_path / "replay", resume=checkpoint)
    assert result["outcome"] == "completed", result
    assert (result["update"], result["normal_update"], result["extra_updates"]) == (270, 240, 30)
    rows = [json.loads(line)
            for line in (tmp_path / "replay/updates.jsonl").read_text().splitlines()]
    assert all(row["table_episode_index"] == 1 for row in rows if row["update_kind"] == "normal")
    assert all(row["table_episode_index"] == 0
               for row in rows if row["update_kind"] == "extra_top6")
    for item in result["exposure"].values():
        assert item["normal_updates"] == 2
        assert item["updates"] == item["normal_updates"] + item["extra_updates"]
    finished, _ = load_checkpoint(tmp_path / "replay/checkpoint-progress.pt")
    repeated = runner.run(new_plan, tmp_path / "completed-resume",
                          resume=tmp_path / "replay/checkpoint-progress.pt")
    assert repeated["outcome"] == "completed", repeated
    assert repeated["update"] == 270
    after, _ = load_checkpoint(tmp_path / "completed-resume/checkpoint-progress.pt")
    for key in ("model", "optimizer", "rng"):
        exact(finished[key], after[key])
    assert not (tmp_path / "completed-resume/updates.jsonl").exists()


@pytest.mark.parametrize("change,match", [
    (lambda s: s["stages"][0].update(max_updates=240), "actual optimizer"),
    (lambda s: s["stages"][0]["loss_replay"].update(start_normal_cursor=1), "complete remaining"),
    (lambda s: s["stages"][0]["sampling"][0].update(episodes=121), "exactly 120"),
    (lambda s: s["tables"].pop(), "exactly 120"),
])
def test_invalid_replay_protocol_rejected(tmp_path, change, match):
    plan = make_plan(tmp_path)
    spec = copy.deepcopy(plan.spec)
    change(spec)
    plan.path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_v54_plan(plan.path)


@pytest.mark.parametrize("stop", [119, 120, 125, 126, 137, 138, 161, 162, 163, 323, 324])
def test_v2_real_adamw_split_resume_across_all_pass_boundaries(tmp_path, monkeypatch, stop):
    monkeypatch.setattr(runner, "train_step", cheap_step)
    plan = make_plan(tmp_path, kind=loss_replay.V2)
    assert plan.summary["stages"][0]["actual_cycle_updates"] == 162
    assert runner.run(plan, tmp_path / "full")["outcome"] == "completed"
    part = runner.run(plan, tmp_path / "part", max_updates_this_invocation=stop)
    assert part["outcome"] == "stopped", part
    result = runner.run(plan, tmp_path / "resumed", resume=tmp_path / "part/checkpoint-progress.pt")
    assert result["outcome"] == "completed", result
    a, _ = load_checkpoint(tmp_path / "full/checkpoint-progress.pt")
    b, _ = load_checkpoint(tmp_path / "resumed/checkpoint-progress.pt")
    for key in ("model", "optimizer", "rng"):
        exact(a[key], b[key])
    for key in ("loss_replay", "exposure", "update", "cursor", "phase"):
        exact(a["state"][key], b["state"][key])
    assert (result["update"], result["normal_update"], result["extra_updates"]) == (324, 240, 84)
    rows = [json.loads(x) for x in (tmp_path / "full/updates.jsonl").read_text().splitlines()]
    for cursor, row in enumerate(x for x in rows if x["update_kind"] == "normal"):
        table, index = schedule_entry(plan, 0, cursor)
        assert (row["table"], row["table_episode_index"], row["namespace"]) == (
            table.name, index, "fit")
    extras = [x for x in rows if x["update_kind"] != "normal"]
    assert all(x["namespace"] == "fit/loss_replay_v2" for x in extras)
    for i, table in enumerate(sorted(result["exposure"])):
        entry = result["exposure"][table]
        assert entry["normal_updates"] == 2
        assert entry["extra_updates"] == (
            12 if i >= 118 else 6 if i >= 114 else 2 if i >= 96 else 0)


def test_v2_migrates_old_extra_state_and_validates_completed_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "train_step", cheap_step)
    old = make_plan(tmp_path)
    assert runner.run(old, tmp_path / "v1", max_updates_this_invocation=150)["outcome"] == "stopped"
    parent, _ = load_checkpoint(tmp_path / "v1/checkpoint-progress.pt")
    plan = make_plan(tmp_path, start=120, kind=loss_replay.V2, start_extra=30)
    assert plan.summary["stages"][0]["new_extra_max_updates"] == 42
    model = make_model(plan).double()
    model.load_state_dict(parent["model"])
    optimizer = adamw(model, plan.optimizer)
    optimizer.load_state_dict(parent["optimizer"])
    restore_rng(parent["rng"])
    state = copy.deepcopy(parent["state"])
    base = state["loss_replay"]["extra_by_table"]
    state["loss_replay"] = loss_replay.new_state(
        [t.name for t in plan.tables], 120, kind=loss_replay.V2, base_extra_by_table=base)
    checkpoint = tmp_path / "migration/checkpoint-progress.pt"
    checkpoint.parent.mkdir()
    save_checkpoint(checkpoint, plan=plan, model=model, optimizer=optimizer,
                    state=state, runtime=parent["runtime"], lineage=[{"mode": "test_v1_to_v2"}])
    migrated, _ = load_checkpoint(checkpoint)
    for key in ("model", "optimizer", "rng"):
        exact(parent[key], migrated[key])
    runner._validate_resume(migrated, plan, parent["runtime"])
    wrong_base = copy.deepcopy(migrated)
    # Both count mappings can remain internally valid while drifting from E.
    name = min(base)
    wrong_base["state"]["loss_replay"]["base_extra_by_table"][name] += 1
    wrong_base["state"]["loss_replay"]["extra_by_table"][name] += 1
    wrong_base["state"]["loss_replay"]["extra_updates"] += 1
    with pytest.raises(ValueError, match="baseline"):
        runner._validate_resume(wrong_base, plan, parent["runtime"])
    assert runner.run(plan, tmp_path / "full", resume=checkpoint)["outcome"] == "completed"
    part = runner.run(plan, tmp_path / "part", resume=checkpoint, max_updates_this_invocation=138)
    assert part["outcome"] == "stopped", part
    result = runner.run(plan, tmp_path / "resumed", resume=tmp_path / "part/checkpoint-progress.pt")
    assert result["outcome"] == "completed", result
    assert (result["update"], result["normal_update"], result["extra_updates"]) == (312, 240, 72)
    a, _ = load_checkpoint(tmp_path / "full/checkpoint-progress.pt")
    b, _ = load_checkpoint(tmp_path / "resumed/checkpoint-progress.pt")
    for key in ("model", "optimizer", "rng"):
        exact(a[key], b[key])
    for key in ("loss_replay", "exposure", "update", "cursor", "phase"):
        exact(a["state"][key], b["state"][key])
    rows = [json.loads(x) for x in (tmp_path / "full/updates.jsonl").read_text().splitlines()]
    assert len(rows) == 162 and rows[0]["update"] == 151
    assert all(x["table_episode_index"] == 1 for x in rows[:120])
    used = dict(base)
    for row in rows[120:]:
        assert row["table_episode_index"] == used[row["table"]]
        assert row["namespace"] == "fit/loss_replay_v2"
        used[row["table"]] += 1
    assert b["state"]["loss_replay"]["base_extra_by_table"] == base
    assert b["state"]["loss_replay"]["extra_by_table"] == used
    finished = runner.run(plan, tmp_path / "finished",
                          resume=tmp_path / "resumed/checkpoint-progress.pt")
    assert finished["outcome"] == "completed" and finished["update"] == 312
    assert not (tmp_path / "finished/updates.jsonl").exists()
    c, _ = load_checkpoint(tmp_path / "finished/checkpoint-progress.pt")
    for key in ("model", "optimizer", "rng"):
        exact(b[key], c[key])
    wrong_finished = copy.deepcopy(c)
    wrong_finished["state"]["loss_replay"]["kind"] = loss_replay.V1
    with pytest.raises(ValueError):
        runner._validate_resume(wrong_finished, plan, parent["runtime"])


@pytest.mark.parametrize("change,match", [
    (lambda s: s["stages"][0]["loss_replay"].pop("start_extra_updates"), "explicit start_extra"),
    (lambda s: s["stages"][0]["loss_replay"].update(start_extra_updates=-1), "start_extra"),
    (lambda s: s["stages"][0]["loss_replay"].update(start_extra_updates=True), "start_extra"),
    (lambda s: s["stages"][0]["loss_replay"].update(kind=loss_replay.V1), "v1 does not accept"),
    (lambda s: s["stages"][0].update(max_updates=300), "actual optimizer"),
])
def test_invalid_v2_protocol_rejected(tmp_path, change, match):
    plan = make_plan(tmp_path, kind=loss_replay.V2)
    spec = copy.deepcopy(plan.spec)
    change(spec)
    plan.path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_v54_plan(plan.path)


def test_v2_inherited_extra_requires_explicit_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "train_step", cheap_step)
    plan = make_plan(tmp_path, start=120, kind=loss_replay.V2, start_extra=30)
    result = runner.run(plan, tmp_path / "bad-fresh")
    assert result["outcome"] == "failed"
    assert "explicit migrated checkpoint" in result["error"]
