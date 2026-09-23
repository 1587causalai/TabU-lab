"""Copy a frozen Nano's two backbone layers four times, without training.

The target sequence is [0, 1, 0, 1, 0, 1, 0, 1], with independent parameters.
Encoder, Unit configuration, codec and readout are unchanged. No output is
zeroed or rescaled; function preservation is neither required nor claimed.
The result is a weights-only initialization with fresh AdamW, RNG and cursors,
consumable by the ordinary V5.4 runner's --initialize-from path.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re


LAYER_SEQUENCE = (0, 1, 0, 1, 0, 1, 0, 1)
NORMAL_MAX_UPDATES = 983_040
ACTUAL_MAX_UPDATES = 1_327_104
VARIANT = "Nano-Dynamics4x-H4-L8-U0"


def _validate_contract(plan, payload):
    from tabu_lab.curriculum_v53 import loss_replay, runner
    from tabu_lab.curriculum_v53.protocol import V54_SCHEMA
    from tabu_lab.models.restoration_v54 import V54Config

    before, after = payload["model_config"], plan.config.as_dict()
    if (payload.get("purpose") != "training"
            or payload["identity"].get("schema") != V54_SCHEMA
            or plan.spec["schema"] != V54_SCHEMA):
        raise ValueError("parent and target must use the V5.4 training contract")
    expected = V54Config(size="nano", center_chunk_size=before["center_chunk_size"]).as_dict()
    if before != expected:
        raise ValueError("parent must be standard two-layer Nano; only center chunk may vary")
    expected = copy.deepcopy(before)
    expected["backbone"]["layers"] = len(LAYER_SEQUENCE)
    if after != expected:
        raise ValueError("target model may change only backbone.layers from 2 to 8")
    if (not runner._model_source(plan.identity)
            or payload["identity"].get("source") != plan.identity["source"]):
        raise ValueError("frozen source identity drift rejected")
    stages = plan.spec["stages"]
    if len(stages) != 1:
        raise ValueError("this conversion requires the one-stage old120 allocation")
    stage = stages[0]
    expected_policy = {
        "kind": loss_replay.V2, "normal_max_updates": NORMAL_MAX_UPDATES,
        "start_normal_cursor": 0, "start_extra_updates": 0,
    }
    if (stage.get("loss_replay") != expected_policy
            or stage["max_updates"] != ACTUAL_MAX_UPDATES
            or stage["optimizer"] != "adamw"
            or len(runner._replay_tables(plan, stage)) != 120):
        raise ValueError("target must start old120 replay v2 at zero with the authorized budget")
    update = payload["state"]["update"]
    if type(update) is not int or update < 0 or not isinstance(payload["state"]["exposure"], dict):
        raise ValueError("invalid parent update/exposure metadata")


def _parent_key(target_key):
    if not target_key.startswith("backbone."):
        return target_key
    match = re.fullmatch(r"backbone\.layers\.(\d+)\.(.+)", target_key)
    if match is None or int(match[1]) >= len(LAYER_SEQUENCE):
        raise ValueError(f"unexpected backbone state key: {target_key}")
    return f"backbone.layers.{LAYER_SEQUENCE[int(match[1])]}.{match[2]}"


def _independent_storage(named_tensors):
    """Reject tied tensors, including distinct Parameter views of one storage."""
    seen = {}
    count = 0
    for name, tensor in named_tensors:
        key = (str(tensor.device), tensor.untyped_storage().data_ptr())
        if tensor.numel() == 0:
            raise ValueError(f"unexpected empty state tensor: {name}")
        if key in seen:
            raise ValueError(f"parameter/storage alias: {seen[key]} and {name}")
        seen[key] = name
        count += 1
    return count


def convert(plan, parent, output_dir, *, device, parent_sha256=None):
    import torch
    from tabu_lab.curriculum_v53 import artifacts, runner
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v54 import V54Config, V54Model
    from tabu_lab.restoration_optimizers import adamw

    if device not in ("cpu", "cuda:0"):
        raise ValueError("use qualified CUDA FP64 or CPU diagnostics; no device fallback")
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("conversion output already exists; refusing overwrite")
    parent = Path(parent).resolve(strict=True)
    payload, parent_sha = artifacts.load_checkpoint(parent)
    if parent_sha256 is not None and parent_sha != parent_sha256:
        raise ValueError("parent differs from explicitly pinned checkpoint SHA256")
    _validate_contract(plan, payload)
    before, after = payload["model_config"], plan.config.as_dict()
    parent_state = payload["model"]

    # Validate every parent tensor against the actual declared Nano, including
    # buffers. Strict target loading alone would not catch omitted parent keys.
    parent_model = V54Model(V54Config.from_dict(before))
    expected = parent_model.state_dict()
    if parent_state.keys() != expected.keys():
        raise ValueError("parent state keys differ from the standard Nano model")
    for name, tensor in parent_state.items():
        other = expected[name]
        if (tensor.shape != other.shape
                or tensor.is_floating_point() != other.is_floating_point()
                or (not tensor.is_floating_point() and tensor.dtype != other.dtype)):
            raise ValueError(f"unexpected parent tensor shape/type: {name}")
    if not torch.equal(parent_state["_codec_signature"], expected["_codec_signature"]):
        raise ValueError("parent codec signature does not match its config")
    del expected, parent_model

    runtime = runner.configure_runtime(device)
    runner._seed_model(plan)
    model = make_model(plan).to(device=device, dtype=execution_dtype(device))
    target_state = model.state_dict()
    mapping = {name: _parent_key(name) for name in target_state}
    if set(mapping.values()) != set(parent_state):
        raise ValueError("conversion must cover every parent state entry exactly by name")
    for name, source in mapping.items():
        if target_state[name].shape != parent_state[source].shape:
            raise ValueError(f"copy shape mismatch: {source} -> {name}")
        # load_state_dict(assign=False) copies into each independently allocated
        # target parameter; never replace modules or Parameters with references.
        target_state[name] = parent_state[source].to(target_state[name])
    model.load_state_dict(target_state, strict=True)
    independent_parameters = _independent_storage(model.named_parameters(remove_duplicate=False))
    converted = model.state_dict()
    for name, source in mapping.items():
        if not torch.equal(converted[name].detach().cpu(), parent_state[source].to(converted[name].dtype)):
            raise ValueError(f"copy changed values: {source} -> {name}")

    optimizer = adamw(model, plan.optimizer)
    if optimizer.state_dict()["state"]:
        raise ValueError("new AdamW unexpectedly has trained moments")
    state = runner._new_state(plan.spec["stages"])
    state["loss_replay"] = runner._initial_replay_state(plan, plan.spec["stages"][0], 0)
    runner._seed_model(plan)
    fresh_rng = artifacts.rng_state()
    changes = {"backbone.layers": {"before": 2, "after": 8}}
    ancestry = {
        "mode": "nano_backbone_four_independent_copies_weights_only",
        "checkpoint_sha256": parent_sha,
        "parent_update": payload["state"]["update"],
        "parent_identity": payload["identity"],
        "parent_model_config": before,
        "parent_runtime": payload["runtime"],
        "config_changes": changes,
        "backbone_layer_sequence": list(LAYER_SEQUENCE),
        "target_to_parent_state_mapping": mapping,
        "independent_parameter_storage_verified": True,
        "non_backbone_values_preserved": True,
        "optimizer_rng_schedule_reset": True,
        "target_variant": VARIANT,
        "output_zeroing": False,
        "weight_rescaling": False,
        "function_preservation_claimed": False,
        "conversion_script_sha256": artifacts.sha256(__file__),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "checkpoint-dynamics4x-init.pt"
    digest = artifacts.save_checkpoint(
        checkpoint, plan=plan, model=model, optimizer=optimizer, state=state,
        runtime=runtime, lineage=[*payload.get("lineage", []), ancestry],
    )
    restored, restored_sha = artifacts.load_checkpoint(checkpoint)
    if (restored_sha != digest or restored["model_config"] != after
            or restored["identity"] != plan.identity or restored["state"] != state
            or restored["optimizer"]["state"] or restored["optimizer_kind"] != "adamw"
            or restored["runtime"] != runtime or restored["lineage"][-1] != ancestry):
        raise ValueError("converted checkpoint metadata/state roundtrip failed")
    if restored["model"].keys() != converted.keys():
        raise ValueError("serialized model state keys changed")
    for name, tensor in converted.items():
        if not torch.equal(restored["model"][name], tensor.detach().cpu()):
            raise ValueError(f"serialization changed tensor: {name}")
    _independent_storage(restored["model"].items())
    if (not torch.equal(restored["rng"]["torch"], fresh_rng["torch"])
            or restored["rng"]["python"] != fresh_rng["python"]
            or len(restored["rng"]["cuda"]) != len(fresh_rng["cuda"])
            or any(not torch.equal(a, b) for a, b in
                   zip(restored["rng"]["cuda"], fresh_rng["cuda"], strict=True))):
        raise ValueError("serialized RNG differs from fresh target-seed RNG")
    if artifacts.sha256(parent) != parent_sha:
        raise ValueError("immutable parent generation changed during conversion")
    receipt = {
        "outcome": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "weights_only_backbone_replication_no_training",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_pointer": str(checkpoint),
        "checkpoint_sha256": digest,
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": parent_sha,
        "parent_update": payload["state"]["update"],
        "parent_exposure": payload["state"]["exposure"],
        "parent_identity": payload["identity"],
        "new_identity": plan.identity,
        "new_model_config": after,
        "target_variant": VARIANT,
        "config_changes": changes,
        "backbone_layer_sequence": list(LAYER_SEQUENCE),
        "target_to_parent_state_mapping": mapping,
        "parent_state_entries": len(parent_state),
        "target_state_entries": len(converted),
        "backbone_state_entries": sum(name.startswith("backbone.") for name in converted),
        "non_backbone_state_entries": sum(not name.startswith("backbone.") for name in converted),
        "parent_parameter_count": sum(t.numel() for k, t in parent_state.items() if k != "_codec_signature"),
        "target_parameter_count": sum(p.numel() for p in model.parameters()),
        "independent_parameter_count": independent_parameters,
        "independent_parameter_storage_verified": True,
        "serialized_state_storage_independent": True,
        "copied_values_exact_after_execution_dtype_conversion": True,
        "serialization_roundtrip_verified": True,
        "parent_checkpoint_unchanged": True,
        "optimizer_fresh": True,
        "fresh_model_seed": plan.spec["seeds"]["model"],
        "fresh_update": 0,
        "fresh_normal_update": 0,
        "fresh_extra_updates": 0,
        "fresh_exposure": {},
        "normal_max_updates": NORMAL_MAX_UPDATES,
        "actual_max_updates": ACTUAL_MAX_UPDATES,
        "runtime": runtime,
        "ancestry": ancestry,
        "claim_boundary": "Only the trained two-layer backbone is copied four times, in order, with independent parameters. No output zeroing, rescaling, function-equivalence, fitting or speed claim. Encoder/codec/Unit/readout contract unchanged; AdamW/RNG/cursors reset.",
    }
    artifacts.atomic_json(output_dir / "conversion.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--parent-sha256", help="optional explicit immutable parent digest")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    from tabu_lab.curriculum_v53.protocol import load_v54_plan
    receipt = convert(load_v54_plan(args.manifest), args.parent, args.output_dir,
                      device=args.device, parent_sha256=args.parent_sha256)
    keys = ("outcome", "checkpoint", "checkpoint_sha256", "parent_checkpoint_sha256",
            "parent_update", "target_variant", "target_state_entries", "target_parameter_count",
            "backbone_layer_sequence", "optimizer_fresh", "normal_max_updates", "actual_max_updates")
    print(json.dumps({key: receipt[key] for key in keys}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
