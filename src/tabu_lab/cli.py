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


def _run_tar(args: argparse.Namespace) -> int:
    if args.tar_command == "freeze-diverse-corpus":
        from tabu_lab.tar_diverse_corpus import freeze_corpus

        result = freeze_corpus(args)
        print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))
        return 0
    if args.tar_command == "sizes":
        from tabu_lab.tar_sizes import list_sizes

        print(json.dumps(list_sizes(), indent=2))
        return 0
    if args.tar_command == "inspect" and args.size != "standard":
        from tabu_lab.tar_sizes import inspect_size

        print(json.dumps(inspect_size(args.size), indent=2, sort_keys=True))
        return 0
    if args.tar_command == "benchmark":
        from tabu_lab.tar_benchmark import run_benchmark

        result = run_benchmark(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["outcome"] == "benchmark_completed" else 3
    if args.tar_command == "joint-fit":
        from tabu_lab.tar_joint_fit import run_joint_fit

        result = run_joint_fit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["outcome"] not in ("blocked_resources", "failed") else 3
    if args.tar_command == "fit":
        from tabu_lab.tar_fit import run_fit

        result = run_fit(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["outcome"] not in ("blocked_resources", "failed") else 3

    if args.tar_command == "inspect":
        from tabu_lab.models.tar.verification import inspect_model

        result = inspect_model()
    elif args.smoke:
        from tabu_lab.models.tar.verification import verify_model

        result = dict(verify_model(), model_size="cpu-smoke")
    else:
        from tabu_lab.tar_sizes import DEFAULT_VALIDATION_SIZE
        from tabu_lab.tar_validation import verify_size

        result = verify_size("standard" if args.full else args.size or DEFAULT_VALIDATION_SIZE)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _run_restoration_verify(args: argparse.Namespace) -> int:
    from tabu_lab.models.restoration import RestorationConfig
    from tabu_lab.models.restoration.verification import verify_components, verify_model

    if args.restoration_command == "inspect":
        print(json.dumps(
            {"status": "local_unissued", "config": RestorationConfig().as_dict()},
            indent=2, sort_keys=True,
        ))
        return 0
    if args.output is not None and args.output.exists():
        raise FileExistsError("verification output already exists")
    if args.device == "cuda:0":
        from tabu_lab.models.restoration.device_verification import verify_device

        if args.components_only:
            raise ValueError("--components-only requires --device cpu")
        result = verify_device(args.device)
    else:
        result = verify_components() if args.components_only else verify_model()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] == "passed" else 1


def _run_restoration_fit(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_fit import run_fit

    result = run_fit(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] in ("planned", "completed") else 3


def _run_restoration_joint_fit(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_joint_fit import run_joint_fit

    observer = None
    if args.execute:
        from tabu_lab.observers.restoration import create_restoration_observer

        observer = create_restoration_observer()
    try:
        result = run_joint_fit(args, observer=observer)
    finally:
        if observer is not None:
            observer.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] in ("planned", "completed", "segment_completed") else 3


def _run_restoration_curriculum_fit(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_curriculum_fit import run_curriculum_fit

    observer = None
    if args.execute:
        from tabu_lab.observers.restoration import create_restoration_observer

        observer = create_restoration_observer()
    try:
        result = run_curriculum_fit(args, observer=observer)
    finally:
        if observer is not None:
            observer.close()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result["outcome"] in ("planned", "completed") else 3


def _run_restoration_curriculum_preflight(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_curriculum_preflight import run_preflight

    result = run_preflight(args)
    return 0 if result["outcome"] == "passed" else 3


def _run_restoration_benchmark(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_benchmark import run_benchmark

    return run_benchmark(args)


def _run_restoration_joint_preflight(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_joint_preflight import run_preflight

    result = run_preflight(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["outcome"] == "passed" else 3


def _run_restoration_pipeline_benchmark(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_pipeline_benchmark import run_benchmark

    return run_benchmark(args)


def _run_restoration_prepared_benchmark(args: argparse.Namespace) -> int:
    from tabu_lab.restoration_prepared_benchmark import run_benchmark

    return run_benchmark(args)


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
    tar = subparsers.add_parser("tar", help="TabU-TAR local implementation checks")
    tar_sub = tar.add_subparsers(dest="tar_command", required=True)
    inspect = tar_sub.add_parser("inspect", help="inspect allocation-free full model shapes")
    inspect.add_argument(
        "--size", choices=("small", "small-128", "medium", "standard"), default="standard"
    )
    inspect.set_defaults(handler=_run_tar)
    sizes = tar_sub.add_parser("sizes", help="list named TAR model sizes")
    sizes.set_defaults(handler=_run_tar)
    verify = tar_sub.add_parser(
        "verify", help="forward/backward/update/checkpoint correctness probe"
    )
    verify_size = verify.add_mutually_exclusive_group()
    verify_size.add_argument(
        "--size", choices=("small", "small-128", "medium", "standard"),
        help="validation size (default: small)"
    )
    verify_size.add_argument("--full", action="store_true", help="explicit 54M Standard validation")
    verify_size.add_argument("--smoke", action="store_true", help="legacy tiny CPU plumbing check")
    verify.set_defaults(handler=_run_tar)
    bench = tar_sub.add_parser(
        "benchmark", help="matched serial/batched TAR timing and equivalence"
    )
    bench.add_argument("--preregistration", type=Path, required=True)
    bench.add_argument("--reference", type=Path, required=True)
    bench.add_argument("--output-root", type=Path, required=True)
    bench.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    bench.set_defaults(handler=_run_tar)
    fit = tar_sub.add_parser("fit", help="run a preregistered local TAR fitting diagnostic")
    fit.add_argument("--preregistration", type=Path, required=True)
    fit.add_argument("--output-root", type=Path, required=True)
    # Dataset identities are bound by the preregistration and its snapshot
    # digest.  Keep the CLI open to deterministic synthetic recipes and future
    # datasets instead of coupling the runner to the two historical fixtures.
    fit.add_argument("--dataset", required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", default="cuda:0")
    fit.add_argument("--smoke", action="store_true", help="two-step reduced-model plumbing only")
    fit.set_defaults(handler=_run_tar)
    joint = tar_sub.add_parser("joint-fit", help="one-model training-only multi-table fit")
    joint.add_argument("--preregistration", type=Path, required=True)
    joint.add_argument("--output-root", type=Path, required=True)
    joint.add_argument("--device", default="cuda:0")
    joint.add_argument("--smoke", action="store_true")
    joint.add_argument("--resume-checkpoint", type=Path)
    joint.add_argument("--stop-after-round", type=int)
    joint.set_defaults(handler=_run_tar)
    corpus = tar_sub.add_parser("freeze-diverse-corpus", help="freeze 120 diverse typed fit tables")
    corpus.add_argument("--generator-root", type=Path, required=True)
    corpus.add_argument("--output-root", type=Path, required=True)
    corpus.add_argument("--rows", type=int, default=256)
    corpus.add_argument("--seed", type=int, default=20260907)
    corpus.set_defaults(handler=_run_tar)
    restoration = subparsers.add_parser("restoration", help="five-step table-restoration reference")
    restoration_sub = restoration.add_subparsers(dest="restoration_command", required=True)
    restoration_inspect = restoration_sub.add_parser(
        "inspect", help="show the reference configuration"
    )
    restoration_inspect.set_defaults(handler=_run_restoration_verify)
    restoration_verify = restoration_sub.add_parser("verify", help="bounded correctness probes")
    restoration_verify.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    restoration_verify.add_argument("--components-only", action="store_true")
    restoration_verify.add_argument(
        "--output", type=Path, help="new, non-overwriting JSON check file"
    )
    restoration_verify.set_defaults(handler=_run_restoration_verify)
    prepared = restoration_sub.add_parser(
        "prepared-benchmark", help="bounded prepared replay parity and timing"
    )
    prepared.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    prepared.add_argument("--output", type=Path, required=True)
    prepared.set_defaults(handler=_run_restoration_prepared_benchmark)
    restoration_fit = restoration_sub.add_parser(
        "fit", help="plan a bounded numeric fit diagnostic; execution requires --execute"
    )
    restoration_fit.add_argument("--preregistration", type=Path, required=True)
    restoration_fit.add_argument("--dataset", type=Path, required=True)
    restoration_fit.add_argument("--output-root", type=Path, required=True)
    restoration_fit.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    restoration_fit.add_argument("--resume", type=Path, help="checkpoint from a previous attempt")
    restoration_fit.add_argument(
        "--wandb-project",
        default=None,
        help="passive W&B mirror project; requires preregistration telemetry.wandb_mirror: true",
    )
    restoration_fit.add_argument("--wandb-entity", default=None)
    restoration_fit.add_argument("--execute", action="store_true")
    restoration_fit.set_defaults(handler=_run_restoration_fit)
    restoration_joint_fit = restoration_sub.add_parser(
        "joint-fit", help="plan or run bounded mixed-type old120 restoration fitting"
    )
    restoration_joint_fit.add_argument("--preregistration", type=Path, required=True)
    restoration_joint_fit.add_argument("--corpus", type=Path, required=True)
    restoration_joint_fit.add_argument("--output-root", type=Path, required=True)
    restoration_joint_fit.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    restoration_joint_fit.add_argument(
        "--resume-checkpoint", type=Path, help="checkpoint from a previous joint-fit attempt"
    )
    restoration_joint_fit.add_argument(
        "--stop-after-round", type=int, help="optional bounded segment endpoint"
    )
    restoration_joint_fit.add_argument("--execute", action="store_true")
    restoration_joint_fit.set_defaults(handler=_run_restoration_joint_fit)
    restoration_curriculum_fit = restoration_sub.add_parser(
        "curriculum-fit", help="plan or run the three-stage Small-128 restoration curriculum"
    )
    restoration_curriculum_fit.add_argument("--preregistration", type=Path, required=True)
    restoration_curriculum_fit.add_argument("--corpus-root", type=Path, required=True)
    restoration_curriculum_fit.add_argument("--output-root", type=Path, required=True)
    restoration_curriculum_fit.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    restoration_curriculum_fit.add_argument(
        "--resume-checkpoint", type=Path, help="checkpoint from a previous curriculum attempt"
    )
    restoration_curriculum_fit.add_argument("--execute", action="store_true")
    restoration_curriculum_fit.set_defaults(handler=_run_restoration_curriculum_fit)
    restoration_curriculum_preflight = restoration_sub.add_parser(
        "curriculum-preflight", help="qualify largest curriculum episodes and optimizer transition"
    )
    restoration_curriculum_preflight.add_argument("--preregistration", type=Path, required=True)
    restoration_curriculum_preflight.add_argument("--corpus-root", type=Path, required=True)
    restoration_curriculum_preflight.add_argument("--output", type=Path, required=True)
    restoration_curriculum_preflight.add_argument(
        "--device", choices=("cpu", "cuda:0"), default="cuda:0"
    )
    restoration_curriculum_preflight.set_defaults(handler=_run_restoration_curriculum_preflight)
    joint_preflight = restoration_sub.add_parser(
        "joint-fit-preflight", help="one largest-table update with the joint-fit configuration"
    )
    joint_preflight.add_argument("--preregistration", type=Path, required=True)
    joint_preflight.add_argument("--corpus", type=Path, required=True)
    joint_preflight.add_argument("--output", type=Path, required=True)
    joint_preflight.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    joint_preflight.set_defaults(handler=_run_restoration_joint_preflight)
    benchmark = restoration_sub.add_parser(
        "benchmark-vectorization", help="bounded paired serial/batched execution comparison"
    )
    benchmark.add_argument("--preregistration", type=Path, required=True)
    benchmark.add_argument("--dataset", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    benchmark.set_defaults(handler=_run_restoration_benchmark)
    pipeline = restoration_sub.add_parser(
        "benchmark-pipeline", help="frozen full-pipeline parity and paired timing"
    )
    pipeline.add_argument("--preregistration", type=Path, required=True)
    pipeline.add_argument("--dataset", type=Path, required=True)
    pipeline.add_argument("--output", type=Path, required=True)
    pipeline.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    pipeline.set_defaults(handler=_run_restoration_pipeline_benchmark)
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
