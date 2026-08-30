#!/usr/bin/env python3
"""Run the bounded TabUR Stage-6 paired fine-tuning diagnostic."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import run_query_row_finetune_lift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--pretrain-steps", type=int, default=20)
    parser.add_argument("--pretrain-worlds", type=int, default=4)
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--label-budget", type=int, default=64)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_query_row_finetune_lift(
        seed=args.seed,
        updates=args.updates,
        pretrain_steps=args.pretrain_steps,
        pretrain_worlds=args.pretrain_worlds,
        row_token_count=args.row_token_count,
        label_budget=args.label_budget,
        test_limit=args.test_limit,
        device=args.device,
    )
    payload = {"schema_version": "tabu.query-row.finetune-lift-result.v1", **result.as_dict()}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
