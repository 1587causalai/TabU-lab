"""Create an audited Small-H4 initialization from frozen V5.4 Nano weights.

This is a weights-only architecture conversion, not training or exact resume.
All Nano state tensors and four attention heads are retained. Newly added
residual outputs start at zero, so the extra axial and Unit layers are expected
to act as identities. A bounded fixed train Query check reports actual output
agreement separately from that construction argument. Small-H4 is an explicit
Small heads=4 variant, not the standard eight-head Small used on dgx2.
The frozen runner can consume the result through ordinary --initialize-from.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def _changes(before, after, prefix=""):
    result = {}
    for key in sorted(before.keys() | after.keys()):
        path = f"{prefix}.{key}" if prefix else key
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            result.update(_changes(left, right, path))
        elif left != right:
            result[path] = {"before": left, "after": right}
    return result


def _verify_fixed_queries(plan, parent_model, model, device, episodes):
    """Compare complete forward paths on bounded, named train-bank episodes."""
    import torch
    from tabu_lab.curriculum_v53.data import build_episode
    from tabu_lab.models.restoration.contracts import RestorationRequest

    if type(episodes) is not int or episodes < 0:
        raise ValueError("verification episodes must be a nonnegative integer")
    probes = [p for p in plan.spec["probes"]
              if p["partition"] == "train" and p["purpose"] == "fit"]
    if not probes:
        raise ValueError("Small-H4 verification requires a fixed training-row fit probe")
    probe = probes[0]
    tables = [t for t in plan.tables if t.cohort in probe["cohorts"] and t.role == "train"]
    # Cover target kinds first, then retain the declared table order.
    selected, kinds = [], set()
    for table in tables:
        kind = table.schema[table.target_column].kind
        if kind not in kinds:
            selected.append(table)
            kinds.add(kind)
    selected.extend(t for t in tables if t.name not in {s.name for s in selected})
    if episodes > len(selected) * probe["masks"]:
        raise ValueError("verification request exceeds this fixed train bank")
    seeds = dict(plan.spec["seeds"])
    seeds["evaluation"] = int.from_bytes(hashlib.sha256(
        f"{seeds['evaluation']}/{probe['name']}".encode()).digest()[:8], "little")
    dtype = next(model.parameters()).dtype
    atol, rtol = (1e-12, 1e-12) if dtype == torch.float64 else (2e-6, 2e-5)
    reports = []
    parent_training, target_training = parent_model.training, model.training
    parent_model.eval()
    model.eval()
    try:
        with torch.no_grad():
            for index in range(episodes):
                table = selected[index % len(selected)]
                mask = index // len(selected)
                inputs, _request, _truth, info = build_episode(
                    table, probe["recipe"], mask, seeds, device, evaluation=True,
                    partition="train", epsilon=plan.config.epsilon,
                    codec_version=plan.config.codec_version,
                )
                request = RestorationRequest(inputs.query.nonzero())
                left, right = parent_model(inputs, request), model(inputs, request)
                tensors = {"carriers": (left.carriers, right.carriers),
                           "units": (left.units, right.units)}
                if len(left.columns) != len(right.columns):
                    raise ValueError("growth changed the requested output columns")
                for a, b in zip(left.columns, right.columns, strict=True):
                    if (a.column != b.column or a.result.status != b.result.status
                            or not torch.equal(a.target_indices, b.target_indices)):
                        raise ValueError("growth changed Query output structure")
                    if a.result.status != "ok":
                        raise ValueError("verification Query has no usable support")
                    tensors[f"query_encoding/{a.column}"] = (a.result.encoding, b.result.encoding)
                    tensors[f"query_decoded/{a.column}"] = (a.decoded, b.decoded)
                comparisons = {}
                for name, (a, b) in tensors.items():
                    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                    same = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
                    comparisons[name] = {
                        "finite": finite, "allclose": same,
                        "exact": torch.equal(a, b),
                        "max_abs_difference": float((a.to(torch.float64) - b.to(torch.float64)).abs().max().cpu())
                        if a.device.type != "mps" else float((a - b).abs().max().cpu()),
                    }
                    if not finite or not same:
                        raise ValueError(f"Small-H4 output agreement failed: {table.name}/{name}")
                reports.append({"table": table.name, "target_kind": table.schema[table.target_column].kind,
                                "mask_index": mask, "query_count": info["query_count"],
                                "comparisons": comparisons})
    finally:
        parent_model.train(parent_training)
        model.train(target_training)
    return {"performed": bool(reports), "passed": bool(reports), "episodes": reports,
            "probe": probe["name"], "partition": "train", "device": device,
            "dtype": str(dtype), "atol": atol, "rtol": rtol,
            "scope": "complete encoder/backbone/Unit/readout forward, all carriers/units and requested Query encodings/decoded predictions, on listed fixed train episodes only"}


def convert(plan, parent, output_dir, *, device, verification_episodes=3):
    import torch
    from tabu_lab.curriculum_v53 import artifacts, runner
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.curriculum_v53.protocol import V54_SCHEMA
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v54 import V54Config, V54Model
    from tabu_lab.restoration_optimizers import adamw

    if device not in ("mps", "cpu"):
        raise ValueError("use the qualified MPS target or CPU diagnostics")
    parent = Path(parent).resolve(strict=True)  # Pin one immutable generation.
    payload, parent_sha = artifacts.load_checkpoint(parent)
    before, after = payload["model_config"], plan.config.as_dict()
    if payload.get("purpose") != "training" or payload["identity"].get("schema") != V54_SCHEMA:
        raise ValueError("parent must be a V5.4 training checkpoint")
    if plan.spec["schema"] != V54_SCHEMA:
        raise ValueError("target must be a V5.4 plan")
    for config, size in ((before, "nano"), (after, "small")):
        fields = {"backbone": {"heads": 4}} if size == "small" else {}
        expected = V54Config(size=size, center_chunk_size=config["center_chunk_size"], **fields).as_dict()
        if config != expected:
            raise ValueError(f"expected standard Nano or Small-H4 config; only center chunk may vary ({size})")
    if before["codec_version"] != after["codec_version"] or before["numeric_scaling"] != after["numeric_scaling"]:
        raise ValueError("codec or numeric scaling drift rejected")
    if (not runner._model_source(plan.identity)
            or payload["identity"].get("source") != plan.identity["source"]):
        raise ValueError("frozen model/execution source drift rejected")

    # Shape/key checks come from actual standard Nano construction, not a loose
    # strict=False load that could silently leave a parent tensor behind.
    parent_model = V54Model(V54Config.from_dict(before))
    parent_expected = parent_model.state_dict()
    parent_state = payload["model"]
    if parent_state.keys() != parent_expected.keys():
        raise ValueError("parent state keys differ from the standard Nano model")
    for name, tensor in parent_state.items():
        if tensor.shape != parent_expected[name].shape:
            raise ValueError(f"unexpected parent tensor shape: {name}")
    if not torch.equal(parent_state["_codec_signature"], parent_expected["_codec_signature"]):
        raise ValueError("parent codec signature buffer does not match its config")
    del parent_expected

    runtime = runner.configure_runtime(device)
    runner._seed_model(plan)
    model = make_model(plan).to(device=device, dtype=execution_dtype(device))
    target_state = model.state_dict()
    copied = []
    for name, tensor in parent_state.items():
        if name not in target_state or tensor.shape != target_state[name].shape:
            raise ValueError(f"unmapped parent tensor rejected: {name}")
        target_state[name] = tensor.to(target_state[name])
        copied.append(name)
    new = sorted(set(target_state) - set(parent_state))
    if any(not name.startswith(("backbone.layers.2.", "unit_blocks.")) for name in new):
        raise ValueError("unexpected newly initialized tensor outside added Small blocks")
    model.load_state_dict(target_state, strict=True)

    zeroed = []
    modules = [(f"backbone.layers.2.{name}", getattr(model.backbone.layers[2], name))
               for name in ("column", "row", "collect")]
    modules.extend((f"unit_blocks.{index}", block)
                   for index, block in enumerate(model.unit_blocks))
    with torch.no_grad():
        for name, block in modules:
            block.out.weight.zero_()
            block.ff[2].weight.zero_()
            zeroed.extend((f"{name}.out.weight", f"{name}.ff.2.weight"))
    if not set(zeroed) <= set(new):
        raise ValueError("zero initialization would overwrite inherited parameters")
    converted = model.state_dict()
    for name, tensor in parent_state.items():
        if not torch.equal(converted[name].detach().cpu(), tensor.to(converted[name].dtype)):
            raise ValueError(f"copied tensor changed: {name}")
    for name in zeroed:
        if torch.count_nonzero(converted[name]).item() != 0:
            raise ValueError(f"residual output was not zeroed: {name}")

    parent_model = parent_model.to(device=device, dtype=execution_dtype(device))
    parent_model.load_state_dict(parent_state, strict=True)
    verification = _verify_fixed_queries(
        plan, parent_model, model, device, verification_episodes,
    )
    del parent_model

    optimizer = adamw(model, plan.optimizer)
    if optimizer.state_dict()["state"]:
        raise ValueError("new optimizer unexpectedly contains trained moments")
    state = runner._new_state(plan.spec["stages"])
    runner._seed_model(plan)  # Fresh run RNG, matching ordinary initialization.
    changes = _changes(before, after)
    ancestry = {
        "mode": "nano_to_small_h4_weights_only_identity_growth",
        "checkpoint_sha256": parent_sha,
        "parent_update": payload["state"]["update"],
        "parent_identity": payload["identity"],
        "parent_model_config": before,
        "config_changes": changes,
        "copied_tensors": sorted(copied),
        "new_tensors": new,
        "zeroed_residual_outputs": sorted(zeroed),
        "optimizer_rng_schedule_reset": True,
        "target_variant": "Small-H4",
        "expected_function_preserving": True,
        "function_preserving_verified": verification["passed"],
        "verification_scope": verification["scope"],
        "verification": verification,
        "construction_reason": "four-head inherited blocks unchanged; all new residual output branches start at zero",
        "conversion_script_sha256": artifacts.sha256(__file__),
    }
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "checkpoint-small-init.pt"
    digest = artifacts.save_checkpoint(
        checkpoint, plan=plan, model=model, optimizer=optimizer, state=state,
        runtime=runtime, lineage=[*payload.get("lineage", []), ancestry],
    )
    restored, restored_sha = artifacts.load_checkpoint(checkpoint)
    if restored_sha != digest or restored["model_config"] != after:
        raise ValueError("converted checkpoint serialization mismatch")
    if restored["optimizer"]["state"] or restored["state"] != state:
        raise ValueError("converted optimizer/scheduler is not fresh")
    for name, tensor in converted.items():
        if not torch.equal(restored["model"][name], tensor.detach().cpu()):
            raise ValueError(f"serialization changed a tensor: {name}")
    if artifacts.sha256(parent) != parent_sha:
        raise ValueError("parent generation changed during conversion")
    receipt = {
        "outcome": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": "weights_only_architecture_conversion_no_training",
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
        "target_variant": "Small-H4",
        "config_changes": changes,
        "copied_tensors": sorted(copied),
        "new_tensors": new,
        "zeroed_residual_outputs": sorted(zeroed),
        "copied_state_entries": len(copied),
        "new_state_entries": len(new),
        "parent_parameter_count": sum(t.numel() for k, t in parent_state.items() if k != "_codec_signature"),
        "target_parameter_count": sum(p.numel() for p in model.parameters()),
        "copied_values_exact_after_execution_dtype_conversion": True,
        "serialization_roundtrip_verified": True,
        "optimizer_fresh": True,
        "fresh_model_seed": plan.spec["seeds"]["model"],
        "fresh_update": 0,
        "fresh_exposure": {},
        "expected_function_preserving": True,
        "function_preserving_verified": verification["passed"],
        "verification": verification,
        "runtime": runtime,
        "ancestry": ancestry,
        "claim_boundary": "Small-H4 explicit heads4 variant, not standard Small; expected identity expansion checked only on listed train Query episodes; optimizer/RNG/schedule reset; no fitting or speed claim",
    }
    artifacts.atomic_json(output_dir / "conversion.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", required=True, choices=("mps", "cpu"))
    parser.add_argument("--verify-fixed-episodes", type=int, default=3,
                        help="bounded end-to-end fixed train Query comparisons (default 3; 0 leaves verification false)")
    args = parser.parse_args()
    from tabu_lab.curriculum_v53.protocol import load_v54_plan
    receipt = convert(load_v54_plan(args.manifest), args.parent, args.output_dir, device=args.device,
                      verification_episodes=args.verify_fixed_episodes)
    keys = ("outcome", "checkpoint", "checkpoint_sha256", "parent_checkpoint_sha256",
            "parent_update", "copied_state_entries", "new_state_entries", "target_parameter_count",
            "config_changes", "optimizer_fresh", "target_variant", "expected_function_preserving",
            "function_preserving_verified")
    print(json.dumps({key: receipt[key] for key in keys}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
