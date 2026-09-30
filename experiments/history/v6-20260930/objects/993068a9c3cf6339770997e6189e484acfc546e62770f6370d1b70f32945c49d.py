"""Bounded ordinal100 CUDA/FP64 smoke using the existing curriculum train_step.

This is a disposable compatibility check.  It initializes the model from the
approved nominal100 endpoint as a weights-only parent, executes one train_step
for each of the 100 ordinal100 tables, writes only a JSON receipt, and never writes a model
checkpoint.  Run only through the already qualified train-20260920 profile.
"""

from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source/src"))

EXPECTED_PARENT_SHA256 = (
    "cb0ed2b5e80612de00a1888dc8a345132f36dc92a3fa097ea3a473065135a776"
)
DEFAULT_PARENT = Path(
    "/home/cms/experiments/tabu-v55-nominal100-retention-dgx2-20260926/"
    "runs/attempt-003-second30min/checkpoint-progress.pt"
)


def _finite_record(record):
    fields = ("loss", "objective_loss", "gradient_norm", "clipped_gradient_norm")
    return all(type(record.get(key)) in (int, float) and math.isfinite(record[key])
               for key in fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "evidence/qualification/ordinal-cuda-smoke.json",
    )
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    from tabu_lab.curriculum_v53.factory import make_model
    from tabu_lab.curriculum_v53.protocol import load_v55_plan
    from tabu_lab.curriculum_v53.runner import configure_runtime, train_step
    from tabu_lab.models.restoration._dtype import execution_dtype
    from tabu_lab.models.restoration_v53 import V53LossConfig
    from tabu_lab.restoration_optimizers import adamw

    plan = load_v55_plan(ROOT / "manifests/candidate.json")
    ordinal = tuple(
        table for table in plan.tables
        if table.cohort == "ordinal100" and table.role == "train"
    )
    if len(ordinal) != 100:
        raise ValueError(f"expected exactly 100 ordinal100 train tables, got {len(ordinal)}")
    if any(not table.name.startswith("ordinal_latent_train_") for table in ordinal):
        raise ValueError("ordinal100 selection contains a non-ordinal table")
    for table in ordinal:
        if table.train_rows != 204:
            raise ValueError(f"{table.name}: expected 204 train rows, got {table.train_rows}")
        target = table.schema[table.target_column]
        if target.kind != "ordinal" or target.domain_size != 8:
            raise ValueError(f"{table.name}: expected an 8-class ordinal target")

    runtime = configure_runtime(args.device)
    if runtime["dtype"] != "float64":
        raise RuntimeError(f"smoke requires FP64, got {runtime['dtype']}")
    if runtime["device"] != "cuda:0":
        raise RuntimeError(f"smoke requires CUDA, got {runtime['device']}")

    payload, checkpoint_sha256 = load_checkpoint(args.checkpoint)
    if checkpoint_sha256 != EXPECTED_PARENT_SHA256:
        raise ValueError(
            "parent checkpoint SHA mismatch: "
            f"{checkpoint_sha256} != {EXPECTED_PARENT_SHA256}"
        )
    if payload.get("model_config") != plan.config.as_dict():
        raise ValueError("weights-only parent model configuration does not match ordinal plan")
    source = payload.get("identity", {}).get("source", {})
    if source.get("sha256") != plan.identity.get("source", {}).get("sha256"):
        raise ValueError("weights-only parent source identity does not match ordinal plan")

    model = make_model(plan).to(device=args.device, dtype=execution_dtype(args.device))
    model.load_state_dict(payload["model"], strict=True)
    optimizer = adamw(model, plan.optimizer)
    stage = plan.spec["stages"][0]
    recipe = stage["recipe"]
    loss_config = V53LossConfig(**stage["loss"])
    receipt = {
        "schema": "tabu.ordinal100.disposable-cuda-smoke.v1",
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "one existing train_step per ordinal100 table; compatibility only; weights discarded",
        "device": args.device,
        "runtime": runtime,
        "parent": {
            "path": str(args.checkpoint),
            "sha256": checkpoint_sha256,
            "expected_sha256": EXPECTED_PARENT_SHA256,
            "checkpoint_update": payload.get("state", {}).get("update"),
            "weights_only": True,
        },
        "plan_identity": plan.identity,
        "table_count_expected": 100,
        "support_rows_expected": 204,
        "query_rows_expected": 51,
        "ordinal_class_count_expected": 8,
        "tables": [],
    }
    started = time.monotonic()
    try:
        for table in ordinal:
            row = train_step(
                model, optimizer, plan, table, recipe[table.kind], 0, args.device,
                loss_config, namespace="ordinal100-cuda-smoke",
                objective=stage["objective"],
            )
            episode = row["episode"]
            record = {
                "table": table.name,
                "cohort": table.cohort,
                "status": "passed",
                "rows": episode["selected_rows"],
                "query": episode["query_count"],
                "loss": row["loss"],
                "objective_loss": row["objective_loss"],
                "gradient_norm": row["gradient_norm"],
                "clipped_gradient_norm": row["clipped_gradient_norm"],
                "seconds": row["seconds"],
            }
            if record["rows"] != 204 or record["query"] != 51:
                raise ValueError(f"{table.name}: support/Query mismatch")
            if not _finite_record(record):
                raise FloatingPointError(f"{table.name}: nonfinite loss or gradient")
            receipt["tables"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        receipt["outcome"] = "passed"
    except Exception as exc:
        receipt.update(
            outcome="failed", error_type=type(exc).__name__, error=str(exc)
        )
    finally:
        # The smoke model and fresh optimizer are intentionally disposable.
        del optimizer, model
        if args.device == "cuda:0":
            import torch
            torch.cuda.empty_cache()

    receipt.update(
        seconds=time.monotonic() - started,
        table_count=len(receipt["tables"]),
        weights_discarded=True,
        checkpoint_written=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return 0 if receipt["outcome"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
