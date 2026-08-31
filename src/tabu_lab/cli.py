"""Small installed command boundary for reproducible TabUR experiment plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_preregistration(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"preregistration does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("preregistration root must be a mapping")
    return dict(payload)


def _resolved_config(args: argparse.Namespace, preregistration: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "tabu.tabur.optimize-resolved.v1",
        "preregistration": preregistration,
        "preregistration_path": str(args.preregistration.resolve()),
        "device": str(args.device),
        "output_root": str(args.output_root.resolve()),
        "execute": bool(args.execute),
    }


def _run_optimize(args: argparse.Namespace) -> int:
    preregistration = _load_preregistration(args.preregistration)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved-config.json"
    if resolved_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {resolved_path}")
    resolved = _resolved_config(args, preregistration)
    payload: dict[str, Any] = {"resolved_config": resolved, "run_status": "not_run"}
    if args.execute:
        experiment = preregistration.get("experiment", "query_row_finetune_lift")
        if experiment != "query_row_finetune_lift":
            raise ValueError(f"unsupported TabUR optimize experiment: {experiment!r}")
        from tabu_lab.experiments import run_query_row_finetune_lift

        params = preregistration.get("parameters", {})
        if not isinstance(params, dict):
            raise ValueError("preregistration.parameters must be a mapping")
        result = run_query_row_finetune_lift(device=args.device, **params)
        payload["result"] = result.as_dict()
        payload["run_status"] = result.execution_status
    resolved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabu-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    from tabu_lab.evolution.cli import add_program_commands

    add_program_commands(subparsers)
    tabu = subparsers.add_parser("tabur", help="TabUR experiment commands")
    tabu_subparsers = tabu.add_subparsers(dest="tabur_command", required=True)
    optimize = tabu_subparsers.add_parser(
        "optimize",
        help="resolve and optionally execute a profile-bound TabUR optimization plan",
    )
    optimize.add_argument(
        "--preregistration",
        "--prereg",
        dest="preregistration",
        type=Path,
        required=True,
        help="checked-in YAML/JSON preregistration path",
    )
    optimize.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    optimize.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="new, non-overwriting result directory",
    )
    optimize.add_argument(
        "--execute",
        action="store_true",
        help="execute the declared bounded runner after resolving the plan",
    )
    optimize.set_defaults(handler=_run_optimize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
