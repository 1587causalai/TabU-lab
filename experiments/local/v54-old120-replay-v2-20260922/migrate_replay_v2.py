"""Explicit replay-v1 to tiered replay-v2 migration preserving training state.

Only a cleanly stopped complete normal-plus-replay boundary is accepted. The old resolved
artifact is authoritative for the frozen parent source; it is never regenerated
using the new loader. This is a strategy change, not an unchanged-recipe resume.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace


POLICY = "normal120_p99x3_p95x2_p80x1_v2"
PARENT_POLICY = "normal120_top5_top20_v1"
MUTABLE_SOURCE = {"curriculum_v53/protocol.py", "curriculum_v53/runner.py",
                  "curriculum_v53/loss_replay.py"}


def exact_tree(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        if (not isinstance(right, torch.Tensor) or left.dtype != right.dtype
                or left.shape != right.shape or not torch.equal(left.cpu(), right.cpu())):
            raise ValueError("tensor type, dtype, shape or value changed")
        return 1
    if type(left) is not type(right):
        raise ValueError("state type changed")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError("state keys changed")
        return sum(exact_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError("state length changed")
        return sum(exact_tree(a, b) for a, b in zip(left, right, strict=True))
    if left != right:
        raise ValueError("state scalar changed")
    return 0


def validate_change(old_resolved, new, state):
    """Admit exactly the declared strategy/source change and its bounded budget."""
    # A persisted resolved.json uses lists; dataclass normalization may recreate
    # equivalent tuples (for example AdamW betas or loss state_weights). Compare
    # the JSON contract rather than Python container representation.
    old_spec = json.loads(json.dumps(old_resolved["spec"], allow_nan=False))
    new_spec = json.loads(json.dumps(new.spec, allow_nan=False))
    if len(old_spec["stages"]) != 1 or len(new_spec["stages"]) != 1:
        raise ValueError("migration requires exactly one stage")
    old_stage, new_stage = old_spec["stages"][0], new_spec["stages"][0]
    old_policy = old_stage.pop("loss_replay", None)
    if (old_stage.get("gate") or old_stage["optimizer"] != "adamw"
            or not old_policy or old_policy["kind"] != PARENT_POLICY):
        raise ValueError("parent must be ungated replay-v1 AdamW")
    replay = state.get("loss_replay", {})
    normal, extras = replay.get("normal_cursor"), replay.get("extra_updates")
    if (type(normal) is not int or type(extras) is not int or normal < 0
            or extras < 0 or normal % 120 or replay.get("partial")
            or replay.get("queue") or replay.get("queue_cursor")
            or state["update"] != normal + extras or state["cursor"] != state["update"]):
        raise ValueError("parent must finish its entire normal-plus-replay cycle")
    normal_limit = old_policy["normal_max_updates"]
    old_stage.pop("max_updates")
    if normal_limit % 120 or normal >= normal_limit:
        raise ValueError("parent must have complete normal cycles remaining")
    expected_policy = dict(kind=POLICY, normal_max_updates=normal_limit,
                           start_normal_cursor=normal, start_extra_updates=extras)
    if new_stage.pop("loss_replay", None) != expected_policy:
        raise ValueError("loss_replay policy or normal/extra inherited cursor drift")
    actual_limit = new_stage.pop("max_updates")
    additional = (normal_limit - normal) // 120 * 42
    expected_actual = normal_limit + extras + additional
    if actual_limit != expected_actual:
        raise ValueError("actual budget must include normal, inherited extra and future replay")
    for spec in (old_spec, new_spec):
        spec.pop("experiment_id", None)
        spec.pop("description", None)
    if old_spec != new_spec:
        raise ValueError("unapproved protocol/model/optimizer/data-path drift")
    old_identity = old_resolved["identity"]
    for key in ("schema", "data_sha256", "datasets"):
        if old_identity[key] != new.identity[key]:
            raise ValueError(f"unapproved {key} identity drift")
    before = old_identity["source"]["files"]
    after = new.identity["source"]["files"]
    if set(after) != set(before):
        raise ValueError("unapproved added/removed source files")
    drift = [key for key in before if before[key] != after[key] and key not in MUTABLE_SOURCE]
    if drift:
        raise ValueError(f"unapproved source drift: {drift}")
    return {"normal_max_updates": normal_limit, "start_normal_cursor": normal,
            "start_extra_updates": extras,
            "remaining_normal_updates": normal_limit - normal,
            "additional_replay_updates": additional,
            "actual_max_updates": expected_actual}


def migrate(old_resolved, new, parent, destination, runtime):
    import torch
    from tabu_lab.curriculum_v53 import artifacts, loss_replay, runner

    if isinstance(old_resolved, (str, Path)):
        old_resolved = json.loads(Path(old_resolved).read_text())
    parent, destination = Path(parent), Path(destination)
    if parent.name != "checkpoint-progress.pt":
        raise ValueError("parent must be the stopped attempt checkpoint-progress.pt")
    payload, parent_sha = artifacts.load_checkpoint(parent)
    parent_generation = parent.resolve(strict=True)
    if payload["identity"] != old_resolved["identity"]:
        raise ValueError("parent identity does not match frozen old resolved artifact")
    state = payload["state"]
    if state["stage_index"] != 0 or state["phase"] != "train":
        raise ValueError("parent must be in stage 0/train")
    limits = validate_change(old_resolved, new, state)
    old = SimpleNamespace(spec=old_resolved["spec"], identity=old_resolved["identity"],
                          config=new.config, tables=new.tables)
    runner._validate_resume(payload, old, runtime)
    terminal_path = parent.parent / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    if (terminal["identity"] != old.identity
            or terminal["checkpoint_sha256"] != parent_sha
            or terminal["update"] != state["update"]
            or terminal["durable_update"] != state["update"]
            or terminal["outcome"] not in ("interrupted", "stopped")
            or terminal.get("error_type") or terminal.get("error")):
        raise ValueError("parent must be a cleanly stopped durable training checkpoint")
    if max(state["stage_seconds"][0], terminal["stage_seconds"][0]) >= old.spec["stages"][0]["max_seconds"]:
        raise ValueError("parent wall budget already exhausted")
    table_ids = sorted(t.name for t in new.tables if t.role == "train"
                       and t.cohort == new.spec["stages"][0]["sampling"][0]["cohort"])
    if len(table_ids) != 120 or len(set(table_ids)) != 120:
        raise ValueError("replay requires 120 distinct training tables")
    changed = copy.deepcopy(payload)
    changed["identity"] = copy.deepcopy(new.identity)
    changed["state"]["loss_replay"] = loss_replay.new_state(
        table_ids, start_normal_cursor=state["loss_replay"]["normal_cursor"],
        kind=POLICY, base_extra_by_table=state["loss_replay"]["extra_by_table"])
    ancestry = {
        "mode": "strategy_change_preserve_optimizer_rng",
        "claim_boundary": "changed training strategy; not unchanged-recipe strict resume",
        "parent_identity": old.identity, "checkpoint_sha256": parent_sha,
        "parent_update": state["update"], "parent_cursor": state["cursor"],
        "parent_terminal_sha256": artifacts.sha256(terminal_path),
        "parent_terminal_total_seconds": terminal["total_seconds"],
        "parent_terminal_stage_seconds": terminal["stage_seconds"],
        "migration_script_sha256": artifacts.sha256(__file__),
        "policy": new.spec["stages"][0]["loss_replay"], "limits": limits,
    }
    changed["lineage"] = [*payload.get("lineage", []), ancestry]
    runner._validate_resume(changed, new, runtime)
    checks = {key: exact_tree(payload[key], changed[key]) for key in
              ("model", "optimizer", "rng", "runtime", "model_config", "optimizer_kind")}
    state_without_replay = copy.deepcopy(changed["state"])
    del state_without_replay["loss_replay"]
    old_state_without_replay = copy.deepcopy(state)
    del old_state_without_replay["loss_replay"]
    exact_tree(old_state_without_replay, state_without_replay)
    for key in ("normal_cursor", "extra_updates", "extra_by_table", "cycle_index"):
        exact_tree(state["loss_replay"][key], changed["state"]["loss_replay"][key])
    destination.mkdir(parents=True, exist_ok=False)
    generations = destination / "checkpoints"
    generations.mkdir()
    temporary = generations / "migration.tmp"
    with temporary.open("xb") as handle:
        torch.save(changed, handle)
        handle.flush()
        os.fsync(handle.fileno())
    digest = artifacts.sha256(temporary)
    final = generations / f"{digest}.pt"
    temporary.rename(final)
    artifacts.atomic_json(final.with_suffix(".json"), {
        "schema": changed["schema"], "sha256": digest, "update": state["update"],
        "cursor": state["cursor"], "stage_index": 0, "identity": new.identity})
    restored, loaded_sha = artifacts.load_checkpoint(final)
    exact_tree(changed, restored)
    runner._validate_resume(restored, new, runtime)
    if loaded_sha != digest:
        raise ValueError("migration serialization mismatch")
    if parent.resolve(strict=True) != parent_generation or artifacts.sha256(parent_generation) != parent_sha:
        raise ValueError("parent checkpoint changed during migration")
    if artifacts.sha256(terminal_path) != ancestry["parent_terminal_sha256"]:
        raise ValueError("parent terminal changed during migration")
    receipt = {
        "outcome": "passed", "parent_checkpoint": str(parent_generation),
        "checkpoint": str(final.resolve()), "checkpoint_sha256": digest,
        "parent_checkpoint_sha256": parent_sha, "parent_update": state["update"],
        "limits": limits, "preserved_tensor_counts": checks,
        "exact_serialization_roundtrip": True, "parent_state_preserved": True,
        "exposure_and_episode_cursor_preserved": True,
        "old_identity": old.identity["sha256"], "new_identity": new.identity["sha256"],
        "runtime": runtime, "ancestry": ancestry,
    }
    artifacts.atomic_json(destination / "migration.json", receipt)
    return final, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--old-resolved", type=Path, required=True)
    parser.add_argument("--new-manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda:0", "cpu"), required=True)
    args = parser.parse_args()
    from tabu_lab.curriculum_v53 import protocol, runner
    new = protocol.load_v54_plan(args.new_manifest)
    runtime = runner.configure_runtime(args.device)
    _, receipt = migrate(args.old_resolved, new, args.parent, args.destination, runtime)
    print(json.dumps({key: receipt[key] for key in (
        "outcome", "checkpoint", "checkpoint_sha256", "parent_checkpoint_sha256",
        "parent_update", "limits", "preserved_tensor_counts", "new_identity")}, indent=2))


if __name__ == "__main__":
    main()
