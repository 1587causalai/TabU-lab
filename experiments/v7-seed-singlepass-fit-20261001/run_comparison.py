"""Paired fitting probe for the V7 main default and its donor/four-round control.

All bodyfat rows are used for fitting; there is no held-out-row claim. Fixed
evaluation codebooks measure representation robustness on the SAME Query rows.
Actual update time excludes evaluation and checkpoint I/O. Runs never overwrite
an existing output directory. The frozen source is adjacent to this script.
Both arms inherit each host's validation-selected V6 checkpoint. No V7-trained
checkpoint is eligible. Compatible Axial/Unit weights and seeds are retained;
the 64D value geometry, lift, and typed Query seeds are new in both arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "source/src").is_dir():
    sys.path.insert(0, str(ROOT / "source/src"))

import torch  # noqa: E402

from tabu_lab.models.restoration.backbone import OMAB, AxialBackbone  # noqa: E402
from tabu_lab.models.restoration_v7 import (  # noqa: E402
    V7Config,
    V7Model,
    checkpoint_state,
    evaluate_task,
    load_checkpoint,
    load_typed_table,
    make_optimizer,
    prepare_episode,
    save_checkpoint,
    table_task,
    train_step,
)


def utc():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def r2(prediction, truth):
    prediction = prediction.detach().cpu().double()
    truth = truth.detach().cpu().double()
    total = (truth - truth.mean()).square().sum()
    if float(total) == 0:
        raise ValueError("fixed Query truth has no numeric variation")
    result = float(1 - (prediction - truth).square().sum() / total)
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite R2")
    return result


def evaluate(model, tasks):
    rows = []
    for task in tasks:
        report = evaluate_task(model, task)
        target = report.target
        query = task.inputs.query[:, target].nonzero(as_tuple=True)[0]
        truth = task.truth.values[target][query]
        supports = task.inputs.visible[:, target].nonzero(as_tuple=True)[0]
        mean = task.inputs.values[target][supports].detach().cpu().double().mean()
        rows.append({
            "code_seed": task.inputs.code_seed,
            "donor_seed": task.donor_seed,
            "round_r2": [r2(pred, truth) for pred in report.predictions],
            "r2": r2(report.predictions[-1], truth),
            "code_losses": report.code_losses,
            "donor_r2": r2(report.donor, truth),
            "support_mean_r2": r2(torch.ones(len(query)) * mean, truth),
        })
    scores = sorted(row["r2"] for row in rows)
    return {
        "mean_r2": sum(scores) / len(scores),
        "min_r2": scores[0],
        "median_r2": (scores[(len(scores) - 1) // 2] + scores[len(scores) // 2]) / 2,
        "max_r2": scores[-1],
        "per_codebook": rows,
    }


class InheritedV6Dynamics(torch.nn.Module):
    """Retain the already trained Axial and Unit stack without key remapping."""

    def __init__(self, config, unit_layers):
        super().__init__()
        self.axial = AxialBackbone(config)
        self.unit_blocks = torch.nn.ModuleList(OMAB(config) for _ in range(unit_layers))

    def forward(self, h, visible, query):
        h = self.axial(h, visible, query)
        n, m = visible.shape
        units = h[:n, m]
        for block in self.unit_blocks:
            units = block(units, units, visible.any(-1))
        rows = torch.arange(n, device=h.device)
        return h.index_put((rows, torch.full_like(rows, m)), units)


def model_for(arm, seed, device, dtype, parent):
    if "model_config" not in parent or "config" in parent:
        raise ValueError("only a V6-family checkpoint may initialize this experiment")
    inherited = parent["model_config"]
    if inherited.get("regression_width") is not None:
        raise ValueError("projected V6 regression features need an explicit migration")
    config = V7Config(
        codec="G64",  # Preserve this historical experiment's codec after defaults change.
        backbone=inherited["backbone"],
        query_init="seed" if arm == "seed1" else "donor",
        rounds=1 if arm == "seed1" else 4,
        bandwidth=inherited["bandwidth"], ridge=inherited["ridge"],
        epsilon=inherited["epsilon"],
    )
    torch.manual_seed(seed)
    model = V7Model(config)
    for block in model.rounds:
        block.backbone = InheritedV6Dynamics(config.backbone, inherited["unit_layers"])
    # Load AFTER conversion so a FP64 parent is not rounded through FP32.
    model = model.to(device=device, dtype=dtype)
    weights = parent["model"]
    for value in weights.values():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("parent has nonfinite weights")
    transferred = {}
    for key, value in weights.items():
        if key == "_codec_signature":
            # V6's non-parameter codec identity buffer is not a G64 identity.
            continue
        if key.startswith("backbone."):
            transferred["rounds.0.backbone.axial." + key.removeprefix("backbone.")] = value
        elif key.startswith("unit_blocks."):
            transferred["rounds.0.backbone." + key] = value
        elif key in ("encoder.unit_seed", "encoder.feature_seed"):
            transferred["rounds.0." + key.removeprefix("encoder.")] = value
        elif not key.startswith("encoder."):
            raise ValueError(f"unhandled V6 parameter: {key}")
    expected_loaded = {key for key in model.state_dict()
                       if ".backbone." in key or key.endswith(("unit_seed", "feature_seed"))}
    if set(transferred) != expected_loaded:
        raise ValueError("incomplete V6 Axial/Unit/seed transfer")
    status = model.load_state_dict(transferred, strict=False)
    expected_new = set(model.state_dict()) - expected_loaded
    if set(status.missing_keys) != expected_new or status.unexpected_keys:
        raise ValueError(f"unexpected weight-transfer mismatch: {status}")
    loaded = model.state_dict()
    for key, value in transferred.items():
        if not torch.equal(loaded[key], value.to(loaded[key])):
            raise ValueError(f"inexact parent weight transfer: {key}")
    model.transfer_receipt = {
        "transferred_tensors": len(transferred),
        "transferred_parameters": sum(value.numel() for value in transferred.values()),
        "all_compatible_parent_tensors_verified": True,
        "new_parameters": sorted(expected_new),
        "retained": ["Axial", "Unit stack", "unit/feature seeds"],
        "not_transferred": [key for key in weights if key.startswith("encoder.")
                            and key not in ("encoder.unit_seed", "encoder.feature_seed")],
        "non_parameter_metadata_not_transferred": [key for key in weights
                                                    if key == "_codec_signature"],
        "fresh_optimizer_and_sampler": True,
    }
    return model, make_optimizer(model)


def qualify(task, bank, args, device, dtype):
    out, seed = args.out, args.seed
    model, optimizer = model_for("seed1", seed, device, dtype, args.parent_payload)
    assert model.config.query_init == "seed" and model.config.rounds == 1
    episode = prepare_episode(task.inputs, donor_seed=task.donor_seed, codec=model.config.codec)
    output = model(episode, decode=False)
    assert len(output.states) == 1
    assert output.states[0].device.type == device.type and output.states[0].dtype == dtype
    assert torch.equal(output.initial, model.query_seed_numeric.expand_as(output.initial))
    other = prepare_episode(task.inputs, donor_seed=task.donor_seed + 987,
                            codec=model.config.codec)
    assert torch.equal(output.states[0], model(other, decode=False).states[0])
    records = [train_step(model, optimizer, [task]).loss for _ in range(2)]
    synchronize(device)
    assert model.query_seed_numeric.grad.abs().sum() > 0
    manifest = {"qualification": True, "data_code_seed": task.inputs.code_seed}
    checkpoint = checkpoint_state(model, optimizer, step=2, manifest=manifest)
    save_checkpoint(out / "qualification.pt", checkpoint)
    expected = state_hash(model)
    train_step(model, optimizer, [task])
    load_checkpoint(out / "qualification.pt", model, optimizer, manifest=manifest)
    assert state_hash(model) == expected
    write_json(out / "qualification.json", {
        "status": "passed", "utc": utc(), "device": str(device), "dtype": str(dtype),
        "torch": torch.__version__, "steps": 2, "losses": records,
        "default_single_pass": True, "donor_invariant": True,
        "typed_seed_gradient": True, "checkpoint_restore": True,
        "weight_transfer": model.transfer_receipt,
        "parent_selection": args.parent_evidence,
    })


def run_arm(arm, mode, args, table, rows, query, bank_seeds, bundle, expected_hash):
    out = args.out / f"{mode}-{arm}"
    out.mkdir()
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    model, optimizer = model_for(arm, args.seed, device, dtype, args.parent_payload)
    initial_hash = state_hash(model)
    if expected_hash is not None and initial_hash != expected_hash:
        raise ValueError("paired arms have different initial parameters")

    def task(code, donor):
        return table_task(table, rows, query, code_seed=code, donor_seed=donor,
                          device=device, dtype=dtype)

    fixed = task(args.seed + 1, args.seed + 2)
    bank = [task(code, donor) for code, donor in bank_seeds]
    identity = {
        "experiment": "v7-seed-singlepass-fit-20261001", "arm": arm, "mode": mode,
        "table": table.name, "table_sha256": table.sha256, "row_ids": rows,
        "query_row_ids": query, "config": model.config.as_dict(),
        "initial_parameter_sha256": initial_hash, "seed": args.seed,
        "device": str(device), "dtype": str(dtype), "torch": torch.__version__,
        "actual_update_seconds_budget": args.seconds, "source_manifest": bundle,
        "bank_seeds": bank_seeds, "parameter_parent": str(args.parent),
        "parent_sha256": args.parent_sha, "parent_selection": args.parent_evidence,
        "weight_transfer": model.transfer_receipt,
        "protocol": {
            "claim": "training fit on fixed Query rows; no held-out-row generalization",
            "fixed_mode": "fixed codebook and donor IDs throughout training",
            "resampled_mode": "codebook redrawn per update; donor row IDs stay fixed",
            "paired_prefix": "same row IDs, initial parameters, and sampling prefix per host",
            "evaluation": "fixed codebook plus 16 frozen alternate codebooks on same rows",
            "budget": "synchronized update time; excludes evaluation and checkpoint I/O",
            "source": "Query Cells receiver-only; inherited Unit3 uses visible.any(-1)",
            "architecture": "G64; retain pretrained V6 Axial3/head8 and Unit3 in both arms",
        },
    }
    write_json(out / "identity.json", identity)
    started = utc()
    elapsed, updates, next_eval = 0.0, 0, args.eval_every
    matched_steps = {10, 50, 100, 200, 500, 1000}
    history = []

    def checkpoint_evaluation(final=False):
        result = {
            "utc": utc(), "updates": updates, "train_seconds": elapsed,
            "fixed": evaluate(model, [fixed]),
            "alternate_codebooks": evaluate(model, bank if final else bank[:4]),
        }
        history.append(result)
        write_json(out / "progress.json", result)
        print(json.dumps({"arm": arm, "mode": mode, "updates": updates,
                          "train_seconds": elapsed,
                          "fixed_r2": result["fixed"]["mean_r2"],
                          "alternate_r2": result["alternate_codebooks"]["mean_r2"]}), flush=True)
        return result

    checkpoint_evaluation(final=True)
    try:
        with (out / "updates.jsonl").open("x") as log:
            while elapsed < args.seconds:
                code_seed = random.Random(args.seed + 100000 + updates + 1).getrandbits(31)
                item = fixed if mode == "fixed" else task(code_seed, args.seed + 2)
                synchronize(device)
                begin = time.monotonic()
                record = train_step(model, optimizer, [item])
                synchronize(device)
                elapsed += time.monotonic() - begin
                updates += 1
                log.write(json.dumps({"update": updates, "train_seconds": elapsed,
                                      "loss": record.loss,
                                      "code_seed": item.inputs.code_seed,
                                      "donor_seed": item.donor_seed}) + "\n")
                if elapsed >= next_eval or updates in matched_steps:
                    checkpoint_evaluation()
                    log.flush()
                    while elapsed >= next_eval:
                        next_eval += args.eval_every
        final = checkpoint_evaluation(final=True)
        checkpoint = checkpoint_state(model, optimizer, step=updates, manifest=identity)
        save_checkpoint(out / "final.pt", checkpoint)
        terminal = {"status": "complete", "utc_started": started, "utc_ended": utc(),
                    "updates": updates, "train_seconds": elapsed, "budget_complete": True,
                    "final": final, "history": history, "checkpoint_sha256": sha(out / "final.pt")}
        write_json(out / "terminal.json", terminal)
    except Exception as exc:
        write_json(out / "failure.json", {
            "status": "failed", "utc": utc(), "updates": updates, "train_seconds": elapsed,
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    return initial_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), required=True)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha", required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--eval-every", type=float, default=60)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.eval_every <= 0:
        parser.error("seconds and eval-every must be positive")
    if args.device == "cuda" and args.dtype != "float64":
        parser.error("this CUDA experiment is pinned to FP64")
    if args.device == "mps" and args.dtype != "float32":
        parser.error("this MPS experiment is pinned to FP32")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.5)
    elif args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.5)
    if sha(args.parent) != args.parent_sha:
        raise ValueError("parent checkpoint digest mismatch")
    args.parent_payload = torch.load(args.parent, map_location="cpu", weights_only=True)
    selection = json.loads(args.parent_selection.read_text())
    args.parent_evidence = selection["trials"][selection["selected_trial"]]
    if args.parent_evidence["checkpoint_sha256"] != args.parent_sha:
        raise ValueError("parent is not the previously validation-selected V6 checkpoint")
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / "source-manifest.json"
    bundle = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for rel, expected in bundle.items():
        if sha(ROOT / rel) != expected:
            raise ValueError(f"frozen source mismatch: {rel}")
    table = load_typed_table(args.data, name="bodyfat")
    rows = tuple(range(len(table.values[0])))
    query = tuple(random.Random(args.seed).sample(list(rows), round(len(rows) / 3)))
    if table.sha256 != "f132d5e176d756f24054984ed92a8cb6c48e235164317c2bc05426292b938202":
        raise ValueError("bodyfat data differs from the audited table")
    bank_seeds = [(random.Random(args.seed + 100000 + index).getrandbits(31), args.seed + 2)
                  for index in range(30001, 30017)]
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    fixed = table_task(table, rows, query, code_seed=args.seed + 1, donor_seed=args.seed + 2,
                       device=device, dtype=dtype)
    bank = [table_task(table, rows, query, code_seed=code, donor_seed=donor,
                       device=device, dtype=dtype) for code, donor in bank_seeds]
    qualify(fixed, bank, args, device, dtype)
    if args.qualify_only:
        return
    write_json(args.out / "launch.json", {
        "utc": utc(), "pid": os.getpid(), "device": args.device,
        "dtype": args.dtype, "seconds_per_arm": args.seconds,
        "arms": ["fixed-seed1", "fixed-donor4", "resampled-seed1", "resampled-donor4"],
    })
    initial_hash = None
    for mode in ("fixed", "resampled"):
        for arm in ("seed1", "donor4"):
            initial_hash = run_arm(arm, mode, args, table, rows, query, bank_seeds,
                                   bundle, initial_hash)
    write_json(args.out / "complete.json", {"status": "complete", "utc": utc()})


if __name__ == "__main__":
    main()
