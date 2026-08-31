"""Installed ``tabu-lab program`` command boundary."""

from __future__ import annotations

import argparse
from pathlib import Path

from tabu_lab.contracts import canonical_json

from .evaluation import evaluate_program_checkpoint, load_program_evaluation_request
from .impact import diff_snapshots, impact_report
from .models import ProgramLane
from .repository import EvolutionRepository, check_or_write_lock
from .runtime import freeze_program, run_program


def _repository(args: argparse.Namespace) -> EvolutionRepository:
    return EvolutionRepository.load(args.repository)


def _print(value: object) -> None:
    print(canonical_json(value))


def _validate(args: argparse.Namespace) -> int:
    repository = _repository(args)
    resolved = {
        ref: repository.resolve(ref).snapshot_hash for ref in sorted(repository.programs)
    }
    _print(
        {
            "schema_version": "tabu.program-validation.v1",
            "repository_hash": repository.repository_hash,
            "node_count": len(repository.nodes),
            "edge_count": len(repository.edges),
            "programs": resolved,
            "status": "valid",
        }
    )
    return 0


def _resolve(args: argparse.Namespace) -> int:
    repository = _repository(args)
    _print(repository.resolve(args.program).model_dump(mode="python"))
    return 0


def _diff(args: argparse.Namespace) -> int:
    repository = _repository(args)
    source = repository.resolve(args.source_program)
    target = repository.resolve(args.target_program)
    _print(
        {
            "schema_version": "tabu.program-diff.v1",
            "source_snapshot_hash": source.snapshot_hash,
            "target_snapshot_hash": target.snapshot_hash,
            "changes": [
                change.model_dump(mode="python")
                for change in diff_snapshots(source, target)
            ],
        }
    )
    return 0


def _impact(args: argparse.Namespace) -> int:
    repository = _repository(args)
    report = impact_report(
        repository,
        repository.resolve(args.source_program),
        repository.resolve(args.target_program),
    )
    _print(report.model_dump(mode="python"))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    repository = _repository(args)
    destination = args.output
    if destination.exists():
        raise ValueError(f"refusing to overwrite frozen program: {destination}")
    frozen = freeze_program(repository.resolve(args.program))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        canonical_json(frozen.model_dump(mode="python")) + "\n",
        encoding="utf-8",
    )
    _print(
        {
            "schema_version": "tabu.program-freeze-result.v1",
            "freeze_hash": frozen.freeze_hash,
            "output": destination.name,
            "status": "frozen_not_run",
        }
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    repository = _repository(args)
    result = run_program(
        repository,
        lane=ProgramLane(args.lane),
        output_root=args.output_root,
        device=args.device,
        program_ref=args.program,
        frozen_path=args.frozen,
        resume_checkpoint=args.resume_checkpoint,
        warm_start_checkpoint=args.warm_start_checkpoint,
        warm_start_source_program=args.warm_start_source_program,
        max_updates_this_invocation=args.max_updates_this_invocation,
    )
    _print(
        {
            "schema_version": "tabu.program-run-result.v1",
            "receipt": result.receipt.model_dump(mode="python"),
            "checkpoint": result.checkpoint.name,
            "checkpoint_sidecar": result.checkpoint_sidecar.name,
            "receipt_file": result.receipt_path.name,
        }
    )
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    repository = _repository(args)
    request = load_program_evaluation_request(args.request)
    result = evaluate_program_checkpoint(
        repository,
        request=request,
        checkpoint=args.checkpoint,
        training_run_receipt=args.training_run_receipt,
        output=args.output,
        device=args.device,
        evaluation_source_revision=args.evaluation_source_revision,
        evaluation_source_archive_sha256=args.evaluation_source_archive_sha256,
    )
    _print(
        {
            "schema_version": "tabu.program-checkpoint-evaluation-result.v1",
            "request_ref": request.ref,
            "request_hash": request.request_hash,
            "receipt_hash": result.receipt.receipt_hash,
            "output": result.receipt_path.name,
            "mean_loss": result.receipt.metrics.mean_loss,
            "scored_targets": result.receipt.metrics.scored_targets,
            "abstained_targets": result.receipt.metrics.abstained_targets,
            "evidence_status": result.receipt.evidence_status,
        }
    )
    return 0


def _lock(args: argparse.Namespace) -> int:
    check_or_write_lock(args.repository, write=args.write)
    _print(
        {
            "schema_version": "tabu.program-lock-result.v1",
            "status": "written" if args.write else "current",
        }
    )
    return 0


def add_program_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    program = subparsers.add_parser(
        "program",
        help="validate, compare, freeze, and run immutable pretraining programs",
    )
    commands = program.add_subparsers(dest="program_command", required=True)

    def add_repository(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--repository",
            type=Path,
            default=Path.cwd(),
            help="scoped tabu-lab repository root (default: current directory)",
        )

    validate = commands.add_parser("validate", help="validate all manifests and source bindings")
    add_repository(validate)
    validate.set_defaults(handler=_validate)

    resolve = commands.add_parser("resolve", help="resolve one ProgramSnapshot to exact hashes")
    add_repository(resolve)
    resolve.add_argument("--program", required=True, help="program_id@version")
    resolve.set_defaults(handler=_resolve)

    diff = commands.add_parser("diff", help="diff two resolved ProgramSnapshots")
    add_repository(diff)
    diff.add_argument("--from-program", dest="source_program", required=True)
    diff.add_argument("--to-program", dest="target_program", required=True)
    diff.set_defaults(handler=_diff)

    impact = commands.add_parser("impact", help="compute the minimum rerun set")
    add_repository(impact)
    impact.add_argument("--from-program", dest="source_program", required=True)
    impact.add_argument("--to-program", dest="target_program", required=True)
    impact.set_defaults(handler=_impact)

    freeze = commands.add_parser("freeze", help="cross the grow/evidence freeze boundary")
    add_repository(freeze)
    freeze.add_argument("--program", required=True, help="grow program_id@version")
    freeze.add_argument("--output", required=True, type=Path)
    freeze.set_defaults(handler=_freeze)

    run = commands.add_parser("run", help="execute a bounded immutable program")
    add_repository(run)
    run.add_argument("--lane", choices=[lane.value for lane in ProgramLane], required=True)
    selector = run.add_mutually_exclusive_group(required=True)
    selector.add_argument("--program", help="grow program_id@version")
    selector.add_argument("--frozen", type=Path, help="frozen evidence-program JSON")
    restart = run.add_mutually_exclusive_group()
    restart.add_argument("--resume-checkpoint", type=Path)
    restart.add_argument("--warm-start-checkpoint", type=Path)
    run.add_argument(
        "--warm-start-source-program",
        help="required source program identity for a weights-only warm start",
    )
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--device", default="cpu")
    run.add_argument("--max-updates-this-invocation", type=int)
    run.set_defaults(handler=_run)

    evaluate = commands.add_parser(
        "evaluate",
        help="issue an independent local-unissued receipt for one selected checkpoint",
    )
    add_repository(evaluate)
    evaluate.add_argument("--request", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--training-run-receipt", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--evaluation-source-revision", required=True)
    evaluate.add_argument("--evaluation-source-archive-sha256", required=True)
    evaluate.set_defaults(handler=_evaluate)

    lock = commands.add_parser("lock", help="check or append the immutable manifest lock")
    add_repository(lock)
    lock.add_argument("--write", action="store_true")
    lock.set_defaults(handler=_lock)


__all__ = ["add_program_commands"]
