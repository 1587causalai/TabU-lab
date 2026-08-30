#!/usr/bin/env python3
"""Run bounded TabUR scales against matched synthetic MLP/XGBoost ICL baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabu_lab.experiments import run_query_row_classical_icl_benchmark  # noqa: E402


def _ints(value: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--pretrain-rows", type=int, default=64)
    parser.add_argument("--world-schedule", default="64,256,1024")
    parser.add_argument("--step-schedule", default="300,800,3000")
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--eval-worlds", type=int, default=12)
    parser.add_argument("--context-rows", default="8,16,32")
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    worlds = _ints(args.world_schedule, name="world-schedule")
    steps = _ints(args.step_schedule, name="step-schedule")
    context_rows = _ints(args.context_rows, name="context-rows")
    if len(worlds) != len(steps):
        raise SystemExit("--world-schedule and --step-schedule must have equal lengths")
    if args.pretrain_rows < 3 or args.eval_rows < 3 or args.eval_worlds <= 0:
        raise SystemExit("rows must be >= 3 and eval-worlds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, Any]] = []
    for index, (world_count, step_count) in enumerate(zip(worlds, steps, strict=True), start=1):
        stem = f"tabur-classical-icl-scale-{index:02d}-w{world_count}-s{step_count}"
        checkpoint = args.output_dir / f"{stem}.safetensors"
        result = run_query_row_classical_icl_benchmark(
            seed=args.seed,
            pretrain_rows=args.pretrain_rows,
            pretrain_worlds=world_count,
            pretrain_steps=step_count,
            eval_rows=args.eval_rows,
            eval_worlds=args.eval_worlds,
            context_rows=context_rows,
            row_token_count=args.row_token_count,
            learning_rate=args.learning_rate,
            device=args.device,
            checkpoint=checkpoint,
        )
        payload = result.as_dict()
        payload["scale_index"] = index
        attempts.append(payload)
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "scale_index": index,
                    "pretrain_worlds": world_count,
                    "pretrain_steps": step_count,
                    "status": result.status,
                    "threshold_met": result.threshold_met,
                    "tabur_mse": result.tabur_mse,
                    "linear_regression_mse": result.linear_regression_mse,
                    "mlp_mse": result.mlp_mse,
                    "xgboost_mse": result.xgboost_mse,
                    "tabur_vs_mlp_ratio": result.tabur_vs_mlp_ratio,
                    "tabur_vs_xgboost_ratio": result.tabur_vs_xgboost_ratio,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if result.threshold_met:
            break

    threshold_met = any(item["threshold_met"] for item in attempts)
    summary = {
        "status": "threshold_met" if threshold_met else "budget_exhausted",
        "threshold_met": threshold_met,
        "baseline_ids": attempts[0]["baseline_ids"],
        "baseline_config_hash": attempts[0]["baseline_config_hash"],
        "criterion": {
            "aggregate": "tabur_mse <= mlp_mse and tabur_mse <= xgboost_mse",
            "metric": "context_standardized_target_mse",
        },
        "device": attempts[0]["device"],
        "seed": args.seed,
        "pretrain_rows": args.pretrain_rows,
        "eval_rows": args.eval_rows,
        "eval_worlds": args.eval_worlds,
        "context_rows": list(context_rows),
        "attempts": attempts,
        "claim_boundary": (
            "bounded local synthetic frozen-ICL diagnostic; no real-data transfer, "
            "formal receipt, benchmark claim, or accepted capability claim"
        ),
    }
    summary_path = args.output_dir / "tabur-classical-icl-threshold.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "status": summary["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
