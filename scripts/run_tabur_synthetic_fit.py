#!/usr/bin/env python3
"""Run bounded TabUR Stage-3 F0/S1 synthetic fitting gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import (
    run_query_row_fixed_world_fit,
    run_query_row_multi_world_fit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--train-worlds", type=int, default=4)
    parser.add_argument("--validation-worlds", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    payload = {
        "schema_version": "tabu.query-row.synthetic-fit-result.v1",
        "f0_fixed_world": run_query_row_fixed_world_fit(
            seed=args.seed,
            steps=args.steps,
            row_token_count=args.row_token_count,
            device=args.device,
        ).as_dict(),
        "s1_multi_world": run_query_row_multi_world_fit(
            seed=args.seed,
            steps=args.steps,
            train_worlds=args.train_worlds,
            validation_worlds=args.validation_worlds,
            row_token_count=args.row_token_count,
            device=args.device,
        ).as_dict(),
        "evidence_status": "local_unissued",
        "claim_boundary": (
            "bounded TabUR synthetic fitting diagnostics only; not a formal receipt, "
            "benchmark, real-data, frozen-ICL, or fine-tuning claim"
        ),
    }
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
