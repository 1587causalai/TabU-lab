"""Installed CLI for planning, qualification, execution and frozen evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from .artifacts import atomic_json
from .protocol import SCHEMA, V54_SCHEMA, V55_SCHEMA, load_plan


def handle(args):
    plan = load_plan(args.manifest, expected_schema=args.curriculum_schema)
    if args.curriculum_command == "plan":
        result = {
            "outcome": "planned_not_run",
            "summary": plan.summary,
            "identity": plan.identity,
            "resolved": plan.spec,
        }
        if args.output_root:
            args.output_root.mkdir(parents=True, exist_ok=False)
            atomic_json(args.output_root / "plan.json", result)
    elif args.curriculum_command == "run":
        from .runner import run

        result = run(
            plan,
            args.output_root,
            device=args.device,
            resume=args.resume_checkpoint,
            initialize_from=args.initialize_from,
            max_updates_this_invocation=args.max_updates_this_invocation,
        )
    elif args.curriculum_command == "evaluate":
        from .runner import evaluate_checkpoint

        result = evaluate_checkpoint(
            plan, args.checkpoint, args.output_root, device=args.device, probes=args.probe
        )
    elif args.curriculum_command == "prepare-v54-warm-start":
        from .warm_start import prepare_v54_warm_start

        result = prepare_v54_warm_start(
            plan, args.donor_checkpoint, args.donor_resolved, args.output_root,
            device=args.device,
        )
    else:
        from .preflight import preflight

        result = preflight(plan, args.output_root, device=args.device, max_seconds=args.max_seconds)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result["outcome"] in ("planned_not_run", "completed", "passed", "stopped") else 3


def add_commands(subparsers):
    for version, schema in (("v53", SCHEMA), ("v54", V54_SCHEMA), ("v55", V55_SCHEMA)):
        _add_version(subparsers, version, schema)


def _add_version(subparsers, version, schema):
    parser = subparsers.add_parser(f"curriculum-{version}", help=f"bounded {version} curricula")
    commands = parser.add_subparsers(dest="curriculum_command", required=True)
    names = ("plan", "preflight", "run", "evaluate")
    if version == "v55":
        names += ("prepare-v54-warm-start",)
    for name in names:
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=name != "plan")
        if name != "plan":
            command.add_argument("--device", choices=("cpu", "cuda:0", "mps"), default="cpu")
        if name == "run":
            parent = command.add_mutually_exclusive_group()
            parent.add_argument("--resume-checkpoint", type=Path)
            parent.add_argument("--initialize-from", type=Path)
            command.add_argument("--max-updates-this-invocation", type=int)
        elif name == "evaluate":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument(
                "--probe", action="append", help="named frozen probe; repeat to select several"
            )
        elif name == "preflight":
            command.add_argument("--max-seconds", type=float, default=120.0)
        elif name == "prepare-v54-warm-start":
            command.add_argument("--donor-checkpoint", type=Path, required=True)
            command.add_argument("--donor-resolved", type=Path, required=True)
        command.set_defaults(handler=handle, curriculum_schema=schema)
