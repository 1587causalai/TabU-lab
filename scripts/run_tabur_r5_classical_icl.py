#!/usr/bin/env python3
"""Compare frozen R5 TabUR checkpoints with linear, MLP, and XGBoost baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from tabu_lab.experiments import run_query_row_r5_classical_icl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Profile-bound .safetensors checkpoint; repeat for multiple seeds/rungs.",
    )
    parser.add_argument("--panel-root-seed", type=int, default=502729)
    parser.add_argument("--panel-worlds", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for checkpoint in args.checkpoint:
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    result = run_query_row_r5_classical_icl(
        checkpoints=tuple(args.checkpoint),
        panel_root_seed=args.panel_root_seed,
        panel_worlds=args.panel_worlds,
        device=args.device,
    )
    payload = result.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
