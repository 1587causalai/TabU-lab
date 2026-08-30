"""Command-line access to TabU-lab contracts and fit experiments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from tabu_lab.registry import (
    ModelNotFoundError,
    ModelSpec,
    get_model_spec,
    list_models,
    validate_registry,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _add_json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="emit stable JSON")


def _add_catalog_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--catalog",
        default="catalog.json",
        help="catalog path (default: ./catalog.json)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabu-lab")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="inspect model-factory contracts")
    model_commands = models.add_subparsers(dest="models_command", required=True)

    list_parser = model_commands.add_parser("list", help="list registered model contracts")
    _add_catalog_flag(list_parser)
    _add_json_flag(list_parser)

    show_parser = model_commands.add_parser("show", help="show one model contract")
    show_parser.add_argument("contract_id")
    _add_catalog_flag(show_parser)
    _add_json_flag(show_parser)

    validate_parser = model_commands.add_parser("validate", help="validate one or all contracts")
    validate_parser.add_argument("contract_id", nargs="?")
    validate_parser.add_argument(
        "--skip-upstream",
        action="store_true",
        help="validate manifest structure without checking the local readonly TeX source",
    )
    _add_catalog_flag(validate_parser)
    _add_json_flag(validate_parser)

    render_tex_parser = model_commands.add_parser(
        "render-tex",
        help="render one structured ModelSpec mathematics block as standalone TeX",
    )
    render_tex_parser.add_argument("contract_id")
    render_tex_parser.add_argument(
        "--output",
        required=True,
        help="output .tex path, or '-' to write to stdout",
    )

    experiments = commands.add_parser(
        "experiments", help="validate, run, and verify typed fit experiments"
    )
    experiment_commands = experiments.add_subparsers(dest="experiments_command", required=True)

    experiment_validate = experiment_commands.add_parser(
        "validate", help="validate one fit preregistration"
    )
    experiment_validate.add_argument("preregistration")
    _add_json_flag(experiment_validate)

    experiment_run = experiment_commands.add_parser(
        "run", help="run every frozen seed in one fit preregistration"
    )
    experiment_run.add_argument("preregistration")
    experiment_run.add_argument("--output-root", required=True)
    experiment_run.add_argument(
        "--prepared",
        help="private PreparedEvalDataBundle required for the registered R1 wedge",
    )
    experiment_run.add_argument(
        "--formal",
        action="store_true",
        help=(
            "request formal issuance; fails unless the Git/source identity is clean and retrievable"
        ),
    )
    experiment_run.add_argument(
        "--source-reviewed",
        action="store_true",
        help=(
            "deprecated compatibility annotation; never authorizes formal issuance without "
            "--authorization-catalog"
        ),
    )
    experiment_run.add_argument(
        "--authorization-catalog",
        help=(
            "required for --formal: validated CatalogIndex containing the runnable experiment, "
            "independent approved review, and reviewed source identity"
        ),
    )
    _add_json_flag(experiment_run)

    experiment_verify = experiment_commands.add_parser(
        "verify", help="verify one immutable fit attempt directory"
    )
    experiment_verify.add_argument("run_directory")
    _add_json_flag(experiment_verify)

    experiment_list = experiment_commands.add_parser("list", help="list cataloged experiments")
    _add_catalog_flag(experiment_list)
    _add_json_flag(experiment_list)

    experiment_show = experiment_commands.add_parser("show", help="show one cataloged experiment")
    experiment_show.add_argument("experiment_id")
    _add_catalog_flag(experiment_show)
    _add_json_flag(experiment_show)

    experiment_status = experiment_commands.add_parser(
        "status", help="summarize experiment lifecycle states"
    )
    experiment_status.add_argument("experiment_id", nargs="?")
    _add_catalog_flag(experiment_status)
    _add_json_flag(experiment_status)

    experiment_transfer_manifest = experiment_commands.add_parser(
        "transfer-manifest", help="emit the frozen synthetic-to-real transfer panel"
    )
    _add_json_flag(experiment_transfer_manifest)

    datasets = commands.add_parser("datasets", help="prepare typed datasets")
    dataset_commands = datasets.add_subparsers(dest="datasets_command", required=True)
    dataset_prepare = dataset_commands.add_parser(
        "prepare", help="prepare a typed dataset manifest into a local cache"
    )
    dataset_prepare.add_argument("dataset_manifest")
    dataset_prepare.add_argument("--cache-dir", required=True)
    _add_json_flag(dataset_prepare)

    runs = commands.add_parser("runs", help="inspect and verify cataloged runs")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)
    run_list = run_commands.add_parser("list", help="list cataloged runs")
    _add_catalog_flag(run_list)
    _add_json_flag(run_list)
    run_show = run_commands.add_parser("show", help="show one cataloged run")
    run_show.add_argument("run_id")
    _add_catalog_flag(run_show)
    _add_json_flag(run_show)
    run_verify = run_commands.add_parser(
        "verify", help="verify an attempt directory or cataloged run receipt"
    )
    run_verify.add_argument("run")
    run_verify.add_argument("--receipt-file")
    _add_catalog_flag(run_verify)
    _add_json_flag(run_verify)

    artifacts = commands.add_parser("artifacts", help="inspect and content-verify model artifacts")
    artifact_commands = artifacts.add_subparsers(dest="artifacts_command", required=True)
    artifact_list = artifact_commands.add_parser("list", help="list model artifacts")
    _add_catalog_flag(artifact_list)
    _add_json_flag(artifact_list)
    artifact_show = artifact_commands.add_parser("show", help="show a model artifact")
    artifact_show.add_argument("artifact_id")
    _add_catalog_flag(artifact_show)
    _add_json_flag(artifact_show)
    artifact_verify = artifact_commands.add_parser(
        "verify", help="verify checkpoint bytes against a ModelArtifact"
    )
    artifact_verify.add_argument("artifact_id")
    artifact_verify.add_argument("--file", required=True, dest="artifact_file")
    _add_catalog_flag(artifact_verify)
    _add_json_flag(artifact_verify)

    eval_parser = commands.add_parser("eval", help="validate and run evaluation suites")
    eval_commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_suites = eval_commands.add_parser("suites", help="inspect evaluation suites")
    suite_commands = eval_suites.add_subparsers(dest="suites_command", required=True)
    suite_list = suite_commands.add_parser("list", help="list frozen suites")
    suite_list.add_argument("--directory")
    _add_catalog_flag(suite_list)
    _add_json_flag(suite_list)
    suite_show = suite_commands.add_parser("show", help="show one frozen suite")
    suite_show.add_argument("suite_id")
    suite_show.add_argument("--directory")
    _add_catalog_flag(suite_show)
    _add_json_flag(suite_show)
    suite_validate = suite_commands.add_parser("validate", help="validate one or all frozen suites")
    suite_validate.add_argument("suite_id", nargs="?")
    suite_validate.add_argument("--directory")
    _add_catalog_flag(suite_validate)
    _add_json_flag(suite_validate)

    eval_data = eval_commands.add_parser(
        "data",
        help="prepare, register, and check retained Evaluation v0 data offline",
    )
    eval_data_commands = eval_data.add_subparsers(dest="data_command", required=True)
    eval_data_prepare = eval_data_commands.add_parser(
        "prepare",
        help="materialize a private truth-retaining bundle from exact local bytes",
    )
    eval_data_prepare.add_argument("request")
    eval_data_prepare.add_argument("--source", required=True)
    eval_data_prepare.add_argument("--output", required=True)
    eval_data_prepare.add_argument("--directory")
    _add_json_flag(eval_data_prepare)
    eval_data_register = eval_data_commands.add_parser(
        "register",
        help="register a self-verifying private bundle as a public DatasetSnapshot",
    )
    eval_data_register.add_argument("bundle")
    eval_data_register.add_argument("--output", required=True)
    eval_data_register.add_argument("--directory")
    _add_json_flag(eval_data_register)
    eval_data_check = eval_data_commands.add_parser(
        "check",
        help="read-only verification of a private bundle and optional public snapshot",
    )
    eval_data_check.add_argument("bundle")
    eval_data_check.add_argument("--snapshot")
    eval_data_check.add_argument("--directory")
    _add_json_flag(eval_data_check)

    eval_dry_run = eval_commands.add_parser(
        "dry-run", help="report real-data availability without fabricating results"
    )
    eval_dry_run.add_argument("suite_id")
    eval_dry_run.add_argument(
        "--prepared",
        action="append",
        default=[],
        dest="prepared_paths",
        help=(
            "private PreparedEvalDataBundle (or legacy PreparedScenario); repeat once "
            "per available scenario"
        ),
    )
    eval_dry_run.add_argument("--directory")
    _add_catalog_flag(eval_dry_run)
    _add_json_flag(eval_dry_run)

    eval_run = eval_commands.add_parser(
        "run",
        help=(
            "run a frozen baseline or cataloged checkpoint against a prepared "
            "truth-isolated scenario"
        ),
    )
    eval_run.add_argument("suite_id")
    eval_run.add_argument("--scenario", required=True)
    eval_producer = eval_run.add_mutually_exclusive_group(required=True)
    eval_producer.add_argument("--baseline")
    eval_producer.add_argument("--artifact", dest="artifact_id")
    eval_run.add_argument(
        "--checkpoint-file",
        help="local safetensors bytes bound by --artifact",
    )
    eval_run.add_argument(
        "--device",
        default="cpu",
        help="checkpoint execution device (default: cpu)",
    )
    eval_run.add_argument("--prepared", required=True)
    eval_run.add_argument("--seed", required=True, type=int)
    eval_run.add_argument(
        "--formal",
        action="store_true",
        help=(
            "request a formal evaluation receipt; requires reviewed data and a "
            "replayable evaluator-source authority in --catalog"
        ),
    )
    eval_run.add_argument(
        "--source-authority-experiment",
        help=(
            "cataloged reviewed experiment used only to close evaluator source identity; "
            "it does not authorize the evaluation suite or dataset"
        ),
    )
    eval_run.add_argument(
        "--dataset-snapshot-id",
        help="exact reviewed DatasetSnapshot object used by a formal evaluation",
    )
    eval_run.add_argument("--output")
    eval_run.add_argument("--directory")
    _add_catalog_flag(eval_run)
    _add_json_flag(eval_run)

    eval_compare = eval_commands.add_parser(
        "compare", help="compare complete three-seed results within one frozen suite"
    )
    eval_compare.add_argument("suite_id")
    eval_compare.add_argument("results", nargs="+")
    eval_compare.add_argument(
        "--prepared",
        action="append",
        required=True,
        dest="prepared_paths",
        help=(
            "private PreparedEvalDataBundle (or legacy PreparedScenario); repeat once "
            "per scenario in the results"
        ),
    )
    eval_compare.add_argument("--output")
    eval_compare.add_argument("--directory")
    _add_catalog_flag(eval_compare)
    _add_json_flag(eval_compare)

    mve = commands.add_parser("mve", help="Model Verification & Evaluation evidence lanes")
    mve_commands = mve.add_subparsers(dest="mve_command", required=True)
    mve_suites = mve_commands.add_parser("suites", help="inspect MVE verification suites")
    mve_suite_commands = mve_suites.add_subparsers(dest="mve_suites_command", required=True)
    mve_suite_list = mve_suite_commands.add_parser("list", help="list allow-listed suites")
    _add_json_flag(mve_suite_list)
    mve_suite_show = mve_suite_commands.add_parser("show", help="show one verification suite")
    mve_suite_show.add_argument("suite_id")
    _add_json_flag(mve_suite_show)
    mve_suite_validate = mve_suite_commands.add_parser("validate", help="validate MVE suites")
    mve_suite_validate.add_argument("suite_id", nargs="?")
    _add_json_flag(mve_suite_validate)
    mve_verify = mve_commands.add_parser("verify", help="run one MVE suite against a contract")
    mve_verify.add_argument("--contract", required=True, dest="contract_id")
    mve_verify.add_argument("--suite", required=True, dest="suite_id")
    mve_verify.add_argument("--output", required=True)
    mve_verify.add_argument("--receipt-ref")
    mve_verify.add_argument("--review-ref")
    _add_json_flag(mve_verify)
    mve_r1 = mve_commands.add_parser("r1", help="run the bounded real-data R1 wedge")
    mve_r1_commands = mve_r1.add_subparsers(dest="mve_r1_command", required=True)
    mve_r1_run = mve_r1_commands.add_parser("run", help="score Diabetes R1 baselines")
    mve_r1_run.add_argument("--prepared", required=True, dest="prepared_bundle")
    mve_r1_run.add_argument("--output", required=True)
    mve_r1_run.add_argument("--checkpoint")
    _add_json_flag(mve_r1_run)
    mve_status = mve_commands.add_parser("status", help="project the four-axis MVE matrix")
    mve_status.add_argument("--contract", dest="contract_id")
    mve_status.add_argument("--results-root", default="verification/results")
    _add_json_flag(mve_status)

    lineage = commands.add_parser("lineage", help="inspect typed research lineage")
    lineage_commands = lineage.add_subparsers(dest="lineage_command", required=True)
    lineage_show = lineage_commands.add_parser("show", help="show edges for one object")
    lineage_show.add_argument("object_id")
    _add_catalog_flag(lineage_show)
    _add_json_flag(lineage_show)

    catalog = commands.add_parser("catalog", help="build or check catalog.json")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_build = catalog_commands.add_parser("build", help="build deterministic catalog")
    catalog_build.add_argument("--repository", default=".")
    catalog_build.add_argument("--output", default="catalog.json")
    _add_json_flag(catalog_build)
    catalog_check = catalog_commands.add_parser("check", help="fail on catalog drift")
    catalog_check.add_argument("--repository", default=".")
    catalog_check.add_argument("--catalog", default="catalog.json")
    _add_json_flag(catalog_check)
    return parser


def _catalog_model_specs(catalog_path: str) -> tuple[ModelSpec, ...]:
    from tabu_lab.catalog import CatalogObjectKind

    catalog = _load_catalog_file(catalog_path)
    specs = tuple(
        ModelSpec.model_validate(entry.data)
        for entry in catalog.entries
        if (
            entry.kind is CatalogObjectKind.MODEL_CONTRACT
            # Content-qualified entries are immutable historical identities
            # retained for lineage binding; the model inspection surface lists
            # only the current bare contract aliases.
            and entry.object_id == entry.data.get("contract_id")
        )
    )
    if not specs:
        raise ValueError("catalog contains no model contracts")
    return tuple(sorted(specs, key=lambda spec: spec.contract_id))


def _assert_model_catalog_parity(
    contract_id: str | None,
    *,
    catalog_path: str,
) -> None:
    catalog_specs = {spec.contract_id: spec for spec in _catalog_model_specs(catalog_path)}
    source_specs = {
        spec.contract_id: spec
        for spec in ((get_model_spec(contract_id),) if contract_id is not None else list_models())
    }
    expected_ids = {contract_id} if contract_id is not None else set(catalog_specs)
    if set(source_specs) != expected_ids or not expected_ids.issubset(catalog_specs):
        raise ValueError("model source/catalog object ids differ")
    for model_id in sorted(expected_ids):
        if source_specs[model_id] != catalog_specs[model_id]:
            raise ValueError(f"model source/catalog hash parity failed: {model_id}")


def _models_list(*, catalog_path: str, as_json: bool) -> int:
    specs = _catalog_model_specs(catalog_path)
    if as_json:
        payload = [
            {
                "build_state": spec.maturity.build_state.value,
                "contract_id": spec.contract_id,
                "contract_version": spec.contract_version,
                "display_name": spec.display_name,
                "maturity": spec.maturity.stage.value,
            }
            for spec in specs
        ]
        print(_json(payload))
        return 0

    print("contract_id\tcontract_version\tmaturity\tbuild_state\tdisplay_name")
    for spec in specs:
        print(
            f"{spec.contract_id}\t{spec.contract_version}\t{spec.maturity.stage.value}"
            f"\t{spec.maturity.build_state.value}\t{spec.display_name}"
        )
    return 0


def _models_show(contract_id: str, *, catalog_path: str, as_json: bool) -> int:
    by_id = {spec.contract_id: spec for spec in _catalog_model_specs(catalog_path)}
    try:
        spec = by_id[contract_id]
    except KeyError as exc:
        raise ModelNotFoundError(contract_id, tuple(by_id)) from exc
    payload = spec.model_dump(mode="json")
    if as_json:
        print(_json(payload))
    else:
        print(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).rstrip())
    return 0


def _models_validate(
    contract_id: str | None,
    *,
    as_json: bool,
    catalog_path: str,
    verify_upstream: bool,
) -> int:
    report = validate_registry(contract_id, verify_upstream=verify_upstream)
    if report.ok:
        _assert_model_catalog_parity(contract_id, catalog_path=catalog_path)
    payload = report.model_dump(mode="json") | {"ok": report.ok}
    if as_json:
        print(_json(payload))
    else:
        print(f"{'valid' if report.ok else 'invalid'}: {len(report.checked)} contract(s)")
        for issue in report.issues:
            location = f" [{issue.contract_id}]" if issue.contract_id else ""
            print(f"{issue.severity.value}: {issue.code}{location}: {issue.message}")
    return 0 if report.ok else 1


def _models_render_tex(contract_id: str, *, output: str) -> int:
    from tabu_lab.mathtex import render_model_tex

    tex = render_model_tex(get_model_spec(contract_id))
    if output == "-":
        print(tex, end="")
        return 0
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(tex, encoding="utf-8")
    print(f"wrote: {output_path.resolve()}")
    return 0


def _experiments_validate(path: str, *, as_json: bool) -> int:
    from tabu_lab.experiments.runner import load_fit_experiment

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version") if isinstance(payload, dict) else None
    if schema_version == "tabu.synthetic-prior.v1":
        from tabu_lab.experiments.transfer import SyntheticPriorSpec

        spec = SyntheticPriorSpec.model_validate(payload)
        _emit(
            {
                "prior_id": spec.prior_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.spec_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.pretrain-experiment.v1":
        from tabu_lab.experiments.transfer import PretrainExperimentSpec

        spec = PretrainExperimentSpec.model_validate(payload)
        _emit(
            {
                "experiment_id": spec.experiment_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.content_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.finetune-experiment.v1":
        from tabu_lab.experiments.transfer import FineTuneExperimentSpec

        spec = FineTuneExperimentSpec.model_validate(payload)
        _emit(
            {
                "experiment_id": spec.experiment_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.content_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.transfer-comparison.v1":
        from tabu_lab.experiments.transfer import TransferComparisonSpec

        spec = TransferComparisonSpec.model_validate(payload)
        _emit(
            {
                "comparison_id": spec.comparison_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.content_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.icl-harness.v1":
        from tabu_lab.experiments.transfer import IclHarnessSpec

        spec = IclHarnessSpec.model_validate(payload)
        _emit(
            {
                "harness_id": spec.harness_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.content_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.transfer-panel-manifest.v1":
        from tabu_lab.contracts.canonical import canonical_hash
        from tabu_lab.experiments.transfer import transfer_manifest

        expected = transfer_manifest()
        if canonical_hash(payload) != canonical_hash(expected):
            raise ValueError(
                "transfer panel manifest is stale; regenerate with build_transfer_assets.py"
            )
        _emit(
            {
                "schema_version": schema_version,
                "spec_hash": canonical_hash(payload),
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version == "tabu.transfer-split-manifest.v1":
        from tabu_lab.experiments.transfer import TransferSplitManifest

        spec = TransferSplitManifest.model_validate(payload)
        _emit(
            {
                "task_id": spec.task_id,
                "schema_version": spec.schema_version,
                "spec_hash": spec.content_hash,
                "valid": True,
            },
            as_json=as_json,
        )
        return 0
    if schema_version in {
        "tabu.transfer-base-pretrain.v2",
        "tabu.transfer-base-icl.v2",
        "tabu.transfer-base-finetune.v2",
    }:
        from tabu_lab.contracts.canonical import canonical_hash
        from tabu_lab.experiments.transfer_base import (
            load_finetune_spec,
            load_icl_spec,
            load_pretrain_spec,
        )

        if schema_version == "tabu.transfer-base-pretrain.v2":
            spec = load_pretrain_spec(path)
            identity = spec.reference
        elif schema_version == "tabu.transfer-base-icl.v2":
            spec = load_icl_spec(path)
            identity = spec.reference
        else:
            spec = load_finetune_spec(path)
            identity = spec.reference
        _emit(
            {
                "schema_version": schema_version,
                "contract_id": identity.contract_id,
                "contract_version": identity.contract_version,
                "profile_id": identity.profile_id,
                "spec_hash": canonical_hash(payload),
                "valid": True,
            },
            as_json=as_json,
        )
        return 0

    spec = load_fit_experiment(path)
    payload = {
        "contract_id": spec.contract_id,
        "contract_version": spec.contract_version,
        "experiment_id": spec.experiment_id,
        "model_spec_hash": spec.model_spec_hash,
        "spec_hash": spec.spec_hash,
        "stage": spec.stage.value,
        "valid": True,
    }
    if as_json:
        print(_json(payload))
    else:
        print(
            f"valid: {spec.experiment_id} ({spec.contract_id}@"
            f"{spec.contract_version}, {spec.stage.value})"
        )
        print(f"spec_hash: {spec.spec_hash}")
    return 0


def _experiments_transfer_manifest(*, as_json: bool) -> int:
    from tabu_lab.experiments.transfer import transfer_manifest

    _emit(transfer_manifest(), as_json=as_json)
    return 0


def _experiments_run(
    path: str,
    *,
    output_root: str,
    prepared_bundle: str | None,
    formal: bool,
    source_reviewed: bool,
    authorization_catalog: str | None,
    as_json: bool,
) -> int:
    if source_reviewed and not formal:
        raise ValueError("--source-reviewed is only meaningful with --formal")
    if formal and authorization_catalog is None:
        raise ValueError(
            "--formal requires --authorization-catalog; --source-reviewed cannot self-authorize"
        )
    if not formal and authorization_catalog is not None:
        raise ValueError("--authorization-catalog is only valid with --formal")
    raw_payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    transfer_schema = raw_payload.get("schema_version") if isinstance(raw_payload, dict) else None
    if transfer_schema in {
        "tabu.synthetic-prior.v1",
        "tabu.pretrain-experiment.v1",
        "tabu.finetune-experiment.v1",
        "tabu.transfer-comparison.v1",
        "tabu.transfer-base-pretrain.v2",
        "tabu.transfer-base-icl.v2",
        "tabu.transfer-base-finetune.v2",
    }:
        raise ValueError(
            f"{transfer_schema} is not executable by the fit-first runner; "
            "bind prepared immutable batches, RunIdentity, and receipt writer "
            "through tabu_lab.experiments.transfer before formal execution"
        )
    from tabu_lab.experiments.runner import run_fit_experiment

    command_parts = [
        "tabu-lab",
        "experiments",
        "run",
        str(Path(path)),
        "--output-root",
        str(Path(output_root)),
    ]
    if formal:
        command_parts.append("--formal")
    if source_reviewed:
        command_parts.append("--source-reviewed")
    result = run_fit_experiment(
        path,
        output_root=output_root,
        prepared_bundle=prepared_bundle,
        command=tuple(command_parts),
        formal=formal,
        source_reviewed=source_reviewed,
        authorization_catalog=authorization_catalog,
    )
    if getattr(result, "schema_version", None) == "tabu.r1-run-receipt.v1":
        payload = result.model_dump(mode="json") | {
            "output": str(Path(output_root, result.experiment_id, "r1-receipt.json").resolve())
        }
        _emit(payload, as_json=as_json)
        return 0 if result.outcome == "passed" else 1
    payload = {
        "experiment_id": result.experiment_id,
        "succeeded": result.succeeded,
        "passed": result.passed,
        "seeds": [
            {
                "attempt_directory": str(seed.artifacts.directory.resolve()),
                "model_seed": seed.model_seed,
                "receipt_hash": seed.artifacts.receipt_hash,
                "verdict": seed.verdict,
            }
            for seed in result.seed_results
        ],
        "stage": result.stage.value,
    }
    if as_json:
        print(_json(payload))
    else:
        print(
            f"{result.aggregate.verdict}: "
            f"{result.experiment_id} ({len(result.seed_results)} seed attempts)"
        )
        for seed in result.seed_results:
            print(f"{seed.model_seed}\t{seed.verdict}\t{seed.artifacts.directory.resolve()}")
    return 0 if result.succeeded else 1


def _experiments_verify(path: str, *, as_json: bool) -> int:
    from tabu_lab.evaluation import verify_fit_attempt_artifacts

    receipt = verify_fit_attempt_artifacts(path)
    payload = {
        "attempt_directory": str(Path(path).resolve()),
        "receipt_hash": receipt.receipt_hash,
        "receipt_id": receipt.receipt_id,
        "run_id": receipt.run_id,
        "status": receipt.status.value,
        "valid": True,
    }
    if as_json:
        print(_json(payload))
    else:
        print(f"valid: {receipt.receipt_id}")
        print(f"receipt_hash: {receipt.receipt_hash}")
    return 0


def _datasets_prepare(path: str, *, cache_dir: str, as_json: bool) -> int:
    from tabu_lab.experiments.preparation import load_dataset_spec, prepare_dataset

    spec = load_dataset_spec(path)
    manifest = prepare_dataset(spec, cache_dir=cache_dir)
    payload = {
        "dataset_hash": spec.dataset_hash,
        "dataset_id": spec.dataset_id,
        "manifest": str(manifest.resolve()),
        "prepared": True,
    }
    if as_json:
        print(_json(payload))
    else:
        print(f"prepared: {spec.dataset_id}")
        print(f"manifest: {manifest.resolve()}")
    return 0


def _emit(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(_json(value))
    else:
        print(yaml.safe_dump(value, allow_unicode=True, sort_keys=False).rstrip())


def _load_catalog_file(path: str):
    from tabu_lab.catalog import load_catalog

    return load_catalog(Path(path))


def _entry_summary(entry: Any) -> dict[str, Any]:
    return {
        "kind": entry.kind.value,
        "object_hash": entry.object_hash,
        "object_id": entry.object_id,
        "schema_version": entry.object_schema_version,
        "source_hash": entry.source_hash,
        "source_path": entry.source_path,
        "status": entry.status,
    }


def _catalog_list(kind: str, *, catalog_path: str, as_json: bool) -> int:
    catalog = _load_catalog_file(catalog_path)
    entries = catalog.collections().get(kind, ())
    payload = [_entry_summary(entry) for entry in entries]
    if as_json:
        print(_json(payload))
    else:
        print("object_id\tstatus\tobject_hash\tsource_path")
        for item in payload:
            print(
                f"{item['object_id']}\t{item['status'] or '-'}\t"
                f"{item['object_hash']}\t{item['source_path']}"
            )
    return 0


def _catalog_show(
    object_id: str,
    *,
    expected_kind: str,
    catalog_path: str,
    as_json: bool,
) -> int:
    entry = _load_catalog_file(catalog_path).show(object_id)
    if entry.kind.value != expected_kind:
        raise ValueError(f"catalog object {object_id!r} is {entry.kind.value}, not {expected_kind}")
    _emit(entry.model_dump(mode="json"), as_json=as_json)
    return 0


def _experiments_status(
    experiment_id: str | None,
    *,
    catalog_path: str,
    as_json: bool,
) -> int:
    from collections import Counter

    from tabu_lab.catalog import CatalogObjectKind

    catalog = _load_catalog_file(catalog_path)
    entries = tuple(
        entry for entry in catalog.entries if entry.kind is CatalogObjectKind.EXPERIMENT
    )
    supersedes_edges = tuple(
        edge
        for edge in catalog.lineage
        if edge.relation.value == "supersedes"
        and edge.source.kind is CatalogObjectKind.EXPERIMENT
        and edge.target.kind is CatalogObjectKind.EXPERIMENT
    )
    superseded_by: dict[str, list[str]] = {}
    for edge in supersedes_edges:
        superseded_by.setdefault(edge.target.object_id, []).append(edge.source.object_id)
    if experiment_id is not None:
        entry = catalog.show(experiment_id)
        if entry.kind is not CatalogObjectKind.EXPERIMENT:
            raise ValueError(f"catalog object {experiment_id!r} is not an experiment")
        payload = _entry_summary(entry) | {
            "lineage_role": (
                "superseded_draft"
                if entry.status == "draft" and experiment_id in superseded_by
                else "active_draft"
                if entry.status == "draft"
                else "terminal_or_promoted"
            ),
            "status_history": entry.data.get("status_history", []),
            "superseded_by_experiment_ids": sorted(superseded_by.get(experiment_id, [])),
            "supersedes_experiment_ids": entry.data.get("supersedes_experiment_ids", []),
        }
        _emit(payload, as_json=as_json)
        return 0
    counts = Counter(entry.status or "not-declared" for entry in entries)
    active_drafts = sorted(
        entry.object_id
        for entry in entries
        if (
            entry.status == "draft"
            and entry.object_id not in superseded_by
            and not entry.object_id.startswith("R1-")
        )
    )
    payload = {
        "active_draft_experiment_ids": active_drafts,
        "counts": dict(sorted(counts.items())),
        "experiments": [_entry_summary(entry) for entry in entries],
        "superseded_draft_experiment_ids": sorted(superseded_by),
        "total": len(entries),
    }
    _emit(payload, as_json=as_json)
    return 0


def _runs_verify(
    run: str,
    *,
    receipt_file: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    candidate = Path(run)
    if candidate.is_dir():
        return _experiments_verify(run, as_json=as_json)

    from tabu_lab.catalog import CatalogObjectKind, RunRecord
    from tabu_lab.evidence import read_receipt

    entry = _load_catalog_file(catalog_path).show(run)
    if entry.kind is not CatalogObjectKind.RUN:
        raise ValueError(f"catalog object {run!r} is not a run")
    record = RunRecord.model_validate(entry.data)
    if record.receipt is None:
        raise ValueError("run has no immutable receipt to verify")
    if receipt_file is None:
        raise ValueError("cataloged run verification requires --receipt-file")
    receipt_path = Path(receipt_file)
    receipt = read_receipt(receipt_path)
    if receipt.receipt_hash != record.receipt.sha256:
        raise ValueError("run receipt content hash mismatch")
    if receipt.run_id != record.run_id:
        raise ValueError("run receipt belongs to a different run")
    expected_status = {
        "succeeded": "succeeded",
        "failed": "failed",
        "killed": "cancelled",
    }.get(record.status.value)
    if expected_status is not None and receipt.status.value != expected_status:
        raise ValueError("run receipt status conflicts with the catalog")
    payload = {
        "content_verified": True,
        "receipt_hash": receipt.receipt_hash,
        "run_id": record.run_id,
        "status": record.status.value,
    }
    _emit(payload, as_json=as_json)
    return 0


def _artifacts_verify(
    artifact_id: str,
    *,
    artifact_file: str,
    catalog_path: str,
    as_json: bool,
) -> int:
    from tabu_lab.adapters import verify_cataloged_checkpoint

    catalog = _load_catalog_file(catalog_path)
    verified = verify_cataloged_checkpoint(
        catalog=catalog,
        artifact_id=artifact_id,
        checkpoint_path=Path(artifact_file),
    )
    artifact = catalog.show(artifact_id).data
    payload = {
        "artifact_id": verified.artifact_id,
        "checkpoint_schema_version": verified.checkpoint_schema_version,
        "checkpoint_sha256": verified.checkpoint_sha256,
        "compiler_sha256": verified.compiler_sha256,
        "content_verified": True,
        "model_spec_sha256": verified.model_spec_sha256,
        "producer_run_id": verified.producer_run_id,
        "run_identity_sha256": verified.run_identity_sha256,
        "semantic_config_sha256": verified.semantic_config_sha256,
        "status": artifact["status"],
    }
    _emit(payload, as_json=as_json)
    return 0


def _suite_directory(path: str | None) -> Path | None:
    return None if path is None else Path(path)


def _catalog_eval_suites(catalog_path: str):
    from tabu_lab.catalog import CatalogObjectKind
    from tabu_lab.evaluation.foundry import EvalSuiteSpec

    catalog = _load_catalog_file(catalog_path)
    suites = {
        entry.object_id: EvalSuiteSpec.model_validate(entry.data)
        for entry in catalog.entries
        if entry.kind is CatalogObjectKind.EVAL_SUITE
    }
    if not suites:
        raise ValueError("catalog contains no evaluation suites")
    return suites


def _load_eval_suite_for_cli(
    suite_id: str,
    *,
    catalog_path: str,
    directory: str | None,
):
    from tabu_lab.evaluation.foundry import load_suite

    suites = _catalog_eval_suites(catalog_path)
    try:
        catalog_suite = suites[suite_id]
    except KeyError as exc:
        raise ValueError(f"unknown cataloged evaluation suite: {suite_id}") from exc
    if directory is None:
        return catalog_suite
    source_suite = load_suite(suite_id, directory=_suite_directory(directory))
    if source_suite != catalog_suite:
        raise ValueError(f"evaluation suite source/catalog hash parity failed: {suite_id}")
    return source_suite


def _eval_suites_list(
    *,
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    from tabu_lab.evaluation.foundry import list_suite_ids

    catalog_ids = tuple(sorted(_catalog_eval_suites(catalog_path)))
    if directory is None:
        ids = catalog_ids
    else:
        ids = list_suite_ids(directory=_suite_directory(directory))
        if ids != catalog_ids:
            raise ValueError("evaluation suite source/catalog object ids differ")
        for suite_id in ids:
            _load_eval_suite_for_cli(
                suite_id,
                catalog_path=catalog_path,
                directory=directory,
            )
    if as_json:
        print(_json(list(ids)))
    else:
        for suite_id in ids:
            print(suite_id)
    return 0


def _eval_suite_show(
    suite_id: str,
    *,
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    suite = _load_eval_suite_for_cli(
        suite_id,
        catalog_path=catalog_path,
        directory=directory,
    )
    _emit(suite.model_dump(mode="json"), as_json=as_json)
    return 0


def _eval_suites_validate(
    suite_id: str | None,
    *,
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    from tabu_lab.evaluation.foundry import validate_suite

    ids = (suite_id,) if suite_id is not None else tuple(sorted(_catalog_eval_suites(catalog_path)))
    if not ids:
        raise ValueError("no evaluation suite manifests found")
    reports = tuple(
        validate_suite(
            _load_eval_suite_for_cli(
                item,
                catalog_path=catalog_path,
                directory=directory,
            )
        )
        for item in ids
    )
    payload = [report.model_dump(mode="json") for report in reports]
    _emit(payload[0] if suite_id is not None else payload, as_json=as_json)
    return 0 if all(report.valid for report in reports) else 1


def _eval_dry_run(
    suite_id: str,
    *,
    prepared_paths: Sequence[str],
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    from tabu_lab.evaluation.foundry import dry_run_suite

    suite = _load_eval_suite_for_cli(
        suite_id,
        catalog_path=catalog_path,
        directory=directory,
    )
    prepared_items = tuple(
        _load_prepared_scenario_for_cli(path, suite=suite) for path in prepared_paths
    )
    prepared = {item.scenario_id: item for item in prepared_items}
    if len(prepared) != len(prepared_items):
        raise ValueError("prepared data bundles must have unique scenario ids")
    report = dry_run_suite(suite, prepared=prepared)
    _emit(report.model_dump(mode="json"), as_json=as_json)
    return 0 if report.ready else 1


def _eval_data_prepare(
    request: str,
    *,
    source: str,
    output: str,
    directory: str | None,
    as_json: bool,
) -> int:
    from tabu_lab.adapters.eval_data_workflow import prepare_and_write_eval_data

    bundle, destination = prepare_and_write_eval_data(
        request_path=request,
        source=source,
        destination=output,
        suite_directory=_suite_directory(directory),
    )
    _emit(
        {
            "authority_sha256": bundle.authority_sha256,
            "bundle": str(destination),
            "prepared": True,
            "prepared_sha256": bundle.prepared_sha256,
            "request_sha256": bundle.request_sha256,
            "scenario_id": bundle.request.scenario_id,
            "source_sha256": bundle.source_sha256,
            "suite_id": bundle.request.suite_id,
            "suite_sha256": bundle.suite_sha256,
        },
        as_json=as_json,
    )
    return 0


def _eval_data_register(
    bundle: str,
    *,
    output: str,
    directory: str | None,
    as_json: bool,
) -> int:
    from tabu_lab.adapters.eval_data_workflow import register_prepared_eval_bundle

    verified, snapshot, destination = register_prepared_eval_bundle(
        bundle_path=bundle,
        destination=output,
        suite_directory=_suite_directory(directory),
    )
    assert snapshot.authority_status is not None
    _emit(
        {
            "authority_review_subject_sha256": snapshot.authority_review_subject_sha256,
            "authority_sha256": snapshot.authority_sha256,
            "authority_status": snapshot.authority_status.value,
            "dataset_snapshot_id": snapshot.dataset_snapshot_id,
            "dataset_snapshot_sha256": snapshot.content_hash,
            "manifest": str(destination),
            "prepared_sha256": verified.prepared_sha256,
            "publication_eligible": snapshot.publication_eligible,
            "registered": True,
            "request_sha256": snapshot.request_sha256,
            "review_ids": list(snapshot.review_ids),
            "source_sha256": verified.source_sha256,
        },
        as_json=as_json,
    )
    return 0


def _eval_data_check(
    bundle: str,
    *,
    snapshot: str | None,
    directory: str | None,
    as_json: bool,
) -> int:
    from tabu_lab.adapters.eval_data_workflow import check_prepared_eval_bundle

    report = check_prepared_eval_bundle(
        bundle_path=bundle,
        snapshot_path=snapshot,
        suite_directory=_suite_directory(directory),
    )
    _emit(report.model_dump(mode="json"), as_json=as_json)
    return 0


def _read_mapping(path: str) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    value = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping in {source}")
    return value


def _load_prepared_scenario_for_cli(
    path: str,
    *,
    suite: Any,
    expected_scenario_id: str | None = None,
):
    """Load the output of ``eval data prepare`` without discarding its authority.

    Historical callers may still provide a bare ``PreparedScenario``.  The
    preferred bundle path is fully self-verifying and additionally binds the
    selected suite hash before evaluator-owned truth is exposed to the runner.
    """

    from tabu_lab.evaluation.foundry import PreparedScenario

    raw = _read_mapping(path)
    if raw.get("schema_version") == "tabu.eval-data-prepared-bundle.v1":
        from tabu_lab.adapters.eval_data_workflow import load_prepared_eval_bundle

        bundle = load_prepared_eval_bundle(path)
        if bundle.request.suite_id != suite.suite_id:
            raise ValueError("prepared data bundle belongs to another evaluation suite")
        if bundle.suite_sha256 != suite.suite_hash:
            raise ValueError("prepared data bundle suite hash differs from the frozen suite")
        prepared = bundle.prepared
    else:
        prepared = PreparedScenario.model_validate(raw)
    if expected_scenario_id is not None and prepared.scenario_id != expected_scenario_id:
        raise ValueError("prepared scenario differs from the selected scenario")
    suite_scenarios = {item.scenario_id for item in suite.scenarios}
    if prepared.scenario_id not in suite_scenarios:
        raise ValueError("prepared scenario is absent from the frozen suite")
    return prepared


def _read_catalog_pointer_mapping(
    *,
    catalog_path: str,
    uri: str,
    expected_sha256: str,
) -> dict[str, Any]:
    """Read one local catalog metadata pointer and verify its canonical identity."""

    if "://" in uri or uri.startswith(("/", "\\")):
        raise ValueError(
            "checkpoint evaluation requires locally materialized relative metadata pointers"
        )
    root = Path(catalog_path).resolve().parent
    source = (root / uri).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise ValueError("catalog metadata pointer escapes the catalog root") from exc
    value = _read_mapping(str(source))
    from tabu_lab.contracts import canonical_hash

    if canonical_hash(value) != expected_sha256:
        raise ValueError(f"catalog metadata hash mismatch: {uri}")
    return value


def _write_json_once(path: str, value: Any) -> None:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_json(value) + "\n", encoding="utf-8")


def _formal_eval_authorities(
    *,
    suite: Any,
    prepared_path: str,
    prepared: Any,
    catalog_path: str,
    source_authority_experiment: str,
    dataset_snapshot_id: str,
):
    """Resolve formal evaluator-source and reviewed-data authorities.

    The experiment selected here is only a reviewed carrier for executable
    source identity.  The suite and DatasetSnapshot independently authorize
    protocol and data.
    """

    from tabu_lab.adapters.eval_data_workflow import load_prepared_eval_bundle
    from tabu_lab.adapters.eval_snapshot import dataset_snapshot_from_prepared
    from tabu_lab.catalog import (
        CatalogObjectKind,
        DatasetSnapshotSpec,
        ExperimentRecord,
    )
    from tabu_lab.evidence import SourceIdentity
    from tabu_lab.evidence.formal_authorization import (
        FormalAuthorizationContext,
        verify_formal_authorization,
    )
    from tabu_lab.experiments.runner import source_tree_manifest

    raw = _read_mapping(prepared_path)
    if raw.get("schema_version") != "tabu.eval-data-prepared-bundle.v1":
        raise ValueError(
            "formal evaluation requires a PreparedEvalDataBundle with request/authority lineage"
        )
    bundle = load_prepared_eval_bundle(prepared_path)
    if bundle.prepared != prepared:
        raise ValueError("formal prepared bundle changed after evaluator input validation")
    if bundle.request.suite_id != suite.suite_id or bundle.suite_sha256 != suite.suite_hash:
        raise ValueError("formal prepared bundle differs from the exact evaluation suite")
    prepared_snapshot = dataset_snapshot_from_prepared(
        suite=suite,
        scenario_id=prepared.scenario_id,
        prepared=prepared,
        request_sha256=bundle.request_sha256,
        authority_sha256=bundle.authority_sha256,
    )

    catalog = _load_catalog_file(catalog_path)
    snapshot_entry = catalog.show(dataset_snapshot_id)
    if snapshot_entry.kind is not CatalogObjectKind.DATASET_SNAPSHOT:
        raise ValueError("--dataset-snapshot-id must name a DatasetSnapshot")
    DatasetSnapshotSpec.model_validate(snapshot_entry.data)

    experiment_entry = catalog.show(source_authority_experiment)
    if experiment_entry.kind is not CatalogObjectKind.EXPERIMENT:
        raise ValueError(
            "--source-authority-experiment must name a reviewed source-authority carrier"
        )
    experiment = ExperimentRecord.model_validate(experiment_entry.data)
    if experiment.preregistration is None:
        raise ValueError("evaluator source-authority carrier lacks a preregistration")
    repository = Path(catalog_path).resolve().parent
    preregistration_uri = experiment.preregistration.uri
    _read_catalog_pointer_mapping(
        catalog_path=catalog_path,
        uri=preregistration_uri,
        expected_sha256=experiment.preregistration.sha256,
    )
    preregistration_path = (repository / preregistration_uri).resolve()
    preregistration_text = preregistration_path.read_text(encoding="utf-8")
    context = FormalAuthorizationContext(
        repository=repository,
        catalog=Path(catalog_path).resolve(),
        experiment_id=source_authority_experiment,
    )
    manifest = source_tree_manifest(
        repository,
        preregistration=preregistration_path,
        request_formal=True,
        reviewed=True,
    )
    live_source_identity = SourceIdentity.model_validate(manifest["source_identity"])
    # Preflight before executing adapter code.  Issuance repeats this replay
    # after evaluation, so a source or remote transition during the run fails.
    verify_formal_authorization(
        context,
        preregistration_text=preregistration_text,
        live_source_identity=live_source_identity,
    )
    return (
        catalog,
        context,
        preregistration_text,
        preregistration_path,
        live_source_identity,
        prepared_snapshot,
    )


def _eval_run(
    suite_id: str,
    *,
    scenario_id: str,
    baseline_id: str | None,
    artifact_id: str | None,
    checkpoint_file: str | None,
    device: str,
    prepared_path: str,
    seed: int,
    formal: bool,
    source_authority_experiment: str | None,
    dataset_snapshot_id: str | None,
    output: str | None,
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    from datetime import UTC, datetime

    from tabu_lab.evaluation.foundry import (
        BaselineAdapter,
        EvalProducerBinding,
        bind_evaluation_receipt,
        run_evaluation,
    )
    from tabu_lab.evidence import SourceIdentity

    suite = _load_eval_suite_for_cli(
        suite_id,
        catalog_path=catalog_path,
        directory=directory,
    )
    scenarios = {scenario.scenario_id: scenario for scenario in suite.scenarios}
    scenario = scenarios.get(scenario_id)
    if scenario is None:
        raise ValueError(f"unknown evaluation scenario: {scenario_id}")
    prepared = _load_prepared_scenario_for_cli(
        prepared_path,
        suite=suite,
        expected_scenario_id=scenario_id,
    )
    formal_authorities = None
    if formal:
        if source_authority_experiment is None or dataset_snapshot_id is None:
            raise ValueError(
                "--formal requires --source-authority-experiment and --dataset-snapshot-id"
            )
        formal_authorities = _formal_eval_authorities(
            suite=suite,
            prepared_path=prepared_path,
            prepared=prepared,
            catalog_path=catalog_path,
            source_authority_experiment=source_authority_experiment,
            dataset_snapshot_id=dataset_snapshot_id,
        )
    elif source_authority_experiment is not None or dataset_snapshot_id is not None:
        raise ValueError(
            "source authority and reviewed dataset snapshot flags are only valid with --formal"
        )
    producer = None
    blind_examples = None
    topology_cases = None
    if baseline_id is not None:
        if checkpoint_file is not None:
            raise ValueError("--checkpoint-file is only valid with --artifact")
        baselines = {baseline.baseline_id: baseline for baseline in scenario.baselines}
        baseline = baselines.get(baseline_id)
        if baseline is None:
            raise ValueError(f"baseline {baseline_id!r} is not frozen by {scenario_id}")
        adapter = BaselineAdapter(baseline)
    else:
        if artifact_id is None:  # argparse enforces this; retain a direct-call fail-closed gate.
            raise ValueError("evaluation needs exactly one baseline or model artifact")
        if checkpoint_file is None:
            raise ValueError("--checkpoint-file is required with --artifact")

        from tabu_lab.adapters.cataloged_checkpoint import resolve_model_artifact
        from tabu_lab.adapters.checkpoint_model import cataloged_checkpoint_launch_spec
        from tabu_lab.adapters.real_eval_data import (
            checkpoint_blind_example,
            checkpoint_topology_cases,
        )

        catalog = _load_catalog_file(catalog_path)
        artifact = resolve_model_artifact(catalog, artifact_id)
        model_spec = _read_catalog_pointer_mapping(
            catalog_path=catalog_path,
            uri=artifact.model_spec.uri,
            expected_sha256=artifact.model_spec.sha256,
        )
        semantic_config = _read_catalog_pointer_mapping(
            catalog_path=catalog_path,
            uri=artifact.semantic_config.uri,
            expected_sha256=artifact.semantic_config.sha256,
        )
        compiler_manifest = _read_catalog_pointer_mapping(
            catalog_path=catalog_path,
            uri=artifact.compiler_manifest.uri,
            expected_sha256=artifact.compiler_manifest.sha256,
        )
        adapter = cataloged_checkpoint_launch_spec(
            catalog=catalog,
            artifact_id=artifact_id,
            checkpoint_path=checkpoint_file,
            model_spec=model_spec,
            semantic_config=semantic_config,
            compiler_manifest=compiler_manifest,
            device=device,
        )
        producer = EvalProducerBinding(
            provenance="receipted_run",
            run_id=artifact.producer_run_id,
            receipt_sha256=artifact.producer_receipt.sha256,
            receipt_pointer=artifact.producer_receipt.uri,
            publication_eligible=True,
        )
        blind_examples = tuple(
            checkpoint_blind_example(prepared, example_id=item.example_id) for item in prepared.test
        )
        topology_cases = checkpoint_topology_cases(prepared) if prepared.topology_checks else ()
    started_at = datetime.now(UTC)
    result = run_evaluation(
        suite,
        scenario_id=scenario_id,
        adapter=adapter,
        prepared=prepared,
        seed=seed,
        producer=producer,
        blind_examples=blind_examples,
        topology_cases=topology_cases,
    )
    from tabu_lab.evaluation.fit_artifacts import capture_environment

    environment, _ = capture_environment(device)
    completed_at = datetime.now(UTC)
    if formal_authorities is None:
        result = bind_evaluation_receipt(
            result,
            environment=environment,
            source_identity=SourceIdentity(
                source_kind="local",
                issuance_status="local_unissued",
                reasons=("formal_evaluation_source_identity_not_provided",),
            ),
            started_at=started_at,
            completed_at=completed_at,
        )
    else:
        from tabu_lab.evaluation.formal_receipt import issue_formal_evaluation_receipt
        from tabu_lab.experiments.runner import source_tree_manifest

        (
            authority_catalog,
            source_authorization_context,
            preregistration_text,
            preregistration_path,
            preflight_source_identity,
            prepared_snapshot,
        ) = formal_authorities
        final_manifest = source_tree_manifest(
            source_authorization_context.repository,
            preregistration=preregistration_path,
            request_formal=True,
            reviewed=True,
        )
        final_source_identity = SourceIdentity.model_validate(final_manifest["source_identity"])
        if final_source_identity != preflight_source_identity:
            raise ValueError("evaluator source identity changed during formal evaluation")
        assert dataset_snapshot_id is not None
        result = issue_formal_evaluation_receipt(
            result,
            environment=environment,
            live_source_identity=final_source_identity,
            started_at=started_at,
            completed_at=completed_at,
            source_authorization_context=source_authorization_context,
            preregistration_text=preregistration_text,
            catalog=authority_catalog,
            dataset_snapshot_id=dataset_snapshot_id,
            prepared_snapshot=prepared_snapshot,
        )
    payload = result.model_dump(mode="json")
    if output is not None:
        _write_json_once(output, payload)
        assert result.execution_receipt is not None
        summary = {
            "evaluation_receipt_hash": result.execution_receipt.receipt_hash,
            "evaluation_receipt_id": result.execution_receipt.receipt_id,
            "output": output,
            "result_hash": result.content_hash,
            "result_id": result.result_id,
            "status": result.status.value,
        }
        _emit(summary, as_json=as_json)
    else:
        _emit(payload, as_json=as_json)
    return 0 if result.status.value == "succeeded" else 1


def _eval_compare(
    suite_id: str,
    *,
    result_paths: Sequence[str],
    prepared_paths: Sequence[str],
    output: str | None,
    directory: str | None,
    catalog_path: str,
    as_json: bool,
) -> int:
    from tabu_lab.evaluation.foundry import EvalResult, compare_results

    suite = _load_eval_suite_for_cli(
        suite_id,
        catalog_path=catalog_path,
        directory=directory,
    )
    results = tuple(EvalResult.model_validate(_read_mapping(path)) for path in result_paths)
    prepared_items = tuple(
        _load_prepared_scenario_for_cli(path, suite=suite) for path in prepared_paths
    )
    prepared = {item.scenario_id: item for item in prepared_items}
    if len(prepared) != len(prepared_items):
        raise ValueError("prepared scenario files must have unique scenario ids")
    report = compare_results(suite, results, prepared=prepared)
    payload = report.model_dump(mode="json")
    if output is not None:
        _write_json_once(output, payload)
        _emit(
            {
                "comparison_hash": report.content_hash,
                "comparison_id": report.comparison_id,
                "output": output,
            },
            as_json=as_json,
        )
    else:
        _emit(payload, as_json=as_json)
    return 0


def _lineage_show(object_id: str, *, catalog_path: str, as_json: bool) -> int:
    catalog = _load_catalog_file(catalog_path)
    entry = catalog.show(object_id)
    edges = [
        edge.model_dump(mode="json")
        for edge in catalog.lineage
        if edge.source.object_id == object_id or edge.target.object_id == object_id
    ]
    payload = {"object": _entry_summary(entry), "lineage": edges}
    _emit(payload, as_json=as_json)
    return 0


def _mve_suites_list(*, as_json: bool) -> int:
    from tabu_lab.verification import list_suites

    suites = list_suites()
    payload = [
        {
            "axis": suite.axis.value,
            "checks": [check.check_id for check in suite.checks],
            "suite_hash": suite.suite_hash,
            "suite_id": suite.suite_id,
            "suite_version": suite.suite_version,
            "title": suite.title,
        }
        for suite in suites
    ]
    if as_json:
        print(_json(payload))
    else:
        for item in payload:
            print(f"{item['suite_id']}\t{item['axis']}\t{len(item['checks'])} checks")
    return 0


def _mve_suite_show(suite_id: str, *, as_json: bool) -> int:
    from tabu_lab.verification import list_suites

    try:
        suite = next(item for item in list_suites() if item.suite_id == suite_id)
    except StopIteration as exc:
        raise ValueError(f"unknown MVE suite: {suite_id}") from exc
    _emit(suite.model_dump(mode="json"), as_json=as_json)
    return 0


def _mve_suites_validate(suite_id: str | None, *, as_json: bool) -> int:
    from tabu_lab.verification import list_suites

    suites = list_suites()
    if suite_id is not None:
        suites = tuple(item for item in suites if item.suite_id == suite_id)
        if not suites:
            raise ValueError(f"unknown MVE suite: {suite_id}")
    _emit(
        {
            "suites": [
                {"suite_id": suite.suite_id, "suite_hash": suite.suite_hash, "valid": True}
                for suite in suites
            ],
            "valid": True,
        },
        as_json=as_json,
    )
    return 0


def _mve_verify(
    contract_id: str,
    suite_id: str,
    output: str,
    *,
    receipt_ref: str | None,
    review_ref: str | None,
    as_json: bool,
) -> int:
    from tabu_lab.verification import list_suites, run_suite, write_result

    try:
        suite = next(item for item in list_suites() if item.suite_id == suite_id)
    except StopIteration as exc:
        raise ValueError(f"unknown MVE suite: {suite_id}") from exc
    result = run_suite(
        contract_id,
        suite,
        receipt_ref=receipt_ref,
        review_ref=review_ref,
    )
    path = write_result(result, output)
    _emit(
        result.model_dump(mode="json") | {"output": path.as_posix()},
        as_json=as_json,
    )
    return 0 if result.outcome.value in {"passed", "not_applicable"} else 1


def _mve_status(
    contract_id: str | None,
    *,
    results_root: str,
    as_json: bool,
) -> int:
    from tabu_lab.verification.status import build_status

    report = build_status(contract_id=contract_id, results_root=results_root)
    _emit(report.model_dump(mode="json"), as_json=as_json)
    return 0


def _mve_r1_run(
    prepared_bundle: str,
    output: str,
    checkpoint: str | None,
    *,
    as_json: bool,
) -> int:
    from tabu_lab.contracts import canonical_hash
    from tabu_lab.experiments.r1_runner import run_r1
    from tabu_lab.registry import get_model_spec

    receipt = run_r1(
        prepared_bundle,
        output=output,
        model_spec_hash=canonical_hash(get_model_spec("tabul")),
        checkpoint_ref=checkpoint,
    )
    _emit(
        receipt.model_dump(mode="json") | {"output": str(Path(output).resolve())}, as_json=as_json
    )
    return 0 if receipt.outcome == "passed" else 1


def _catalog_build(*, repository: str, output: str, as_json: bool) -> int:
    from tabu_lab.catalog import build_catalog

    catalog = build_catalog(repository, output_path=output)
    payload = {
        "catalog_hash": catalog.catalog_hash,
        "entries": len(catalog.entries),
        "lineage_edges": len(catalog.lineage),
        "output": output,
        "source_tree_hash": catalog.source_tree_hash,
    }
    _emit(payload, as_json=as_json)
    return 0


def _catalog_check(*, repository: str, catalog_path: str, as_json: bool) -> int:
    from tabu_lab.catalog import check_catalog

    report = check_catalog(repository, catalog_path)
    _emit(report.model_dump(mode="json"), as_json=as_json)
    return 0 if report.ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "models":
            if args.models_command == "list":
                return _models_list(catalog_path=args.catalog, as_json=args.json)
            if args.models_command == "show":
                return _models_show(
                    args.contract_id,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.models_command == "validate":
                return _models_validate(
                    args.contract_id,
                    as_json=args.json,
                    catalog_path=args.catalog,
                    verify_upstream=not args.skip_upstream,
                )
            if args.models_command == "render-tex":
                return _models_render_tex(args.contract_id, output=args.output)
        if args.command == "experiments":
            if args.experiments_command == "list":
                return _catalog_list("experiment", catalog_path=args.catalog, as_json=args.json)
            if args.experiments_command == "show":
                return _catalog_show(
                    args.experiment_id,
                    expected_kind="experiment",
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.experiments_command == "status":
                return _experiments_status(
                    args.experiment_id,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.experiments_command == "validate":
                return _experiments_validate(args.preregistration, as_json=args.json)
            if args.experiments_command == "transfer-manifest":
                return _experiments_transfer_manifest(as_json=args.json)
            if args.experiments_command == "run":
                return _experiments_run(
                    args.preregistration,
                    output_root=args.output_root,
                    prepared_bundle=args.prepared,
                    formal=args.formal,
                    source_reviewed=args.source_reviewed,
                    authorization_catalog=args.authorization_catalog,
                    as_json=args.json,
                )
            if args.experiments_command == "verify":
                return _experiments_verify(args.run_directory, as_json=args.json)
        if args.command == "datasets" and args.datasets_command == "prepare":
            return _datasets_prepare(
                args.dataset_manifest,
                cache_dir=args.cache_dir,
                as_json=args.json,
            )
        if args.command == "runs":
            if args.runs_command == "list":
                return _catalog_list("run", catalog_path=args.catalog, as_json=args.json)
            if args.runs_command == "show":
                return _catalog_show(
                    args.run_id,
                    expected_kind="run",
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.runs_command == "verify":
                return _runs_verify(
                    args.run,
                    receipt_file=args.receipt_file,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
        if args.command == "artifacts":
            if args.artifacts_command == "list":
                return _catalog_list("model_artifact", catalog_path=args.catalog, as_json=args.json)
            if args.artifacts_command == "show":
                return _catalog_show(
                    args.artifact_id,
                    expected_kind="model_artifact",
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.artifacts_command == "verify":
                return _artifacts_verify(
                    args.artifact_id,
                    artifact_file=args.artifact_file,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
        if args.command == "eval":
            if args.eval_command == "suites":
                if args.suites_command == "list":
                    return _eval_suites_list(
                        directory=args.directory,
                        catalog_path=args.catalog,
                        as_json=args.json,
                    )
                if args.suites_command == "show":
                    return _eval_suite_show(
                        args.suite_id,
                        directory=args.directory,
                        catalog_path=args.catalog,
                        as_json=args.json,
                    )
                if args.suites_command == "validate":
                    return _eval_suites_validate(
                        args.suite_id,
                        directory=args.directory,
                        catalog_path=args.catalog,
                        as_json=args.json,
                    )
            if args.eval_command == "data":
                if args.data_command == "prepare":
                    return _eval_data_prepare(
                        args.request,
                        source=args.source,
                        output=args.output,
                        directory=args.directory,
                        as_json=args.json,
                    )
                if args.data_command == "register":
                    return _eval_data_register(
                        args.bundle,
                        output=args.output,
                        directory=args.directory,
                        as_json=args.json,
                    )
                if args.data_command == "check":
                    return _eval_data_check(
                        args.bundle,
                        snapshot=args.snapshot,
                        directory=args.directory,
                        as_json=args.json,
                    )
            if args.eval_command == "dry-run":
                return _eval_dry_run(
                    args.suite_id,
                    prepared_paths=args.prepared_paths,
                    directory=args.directory,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.eval_command == "run":
                return _eval_run(
                    args.suite_id,
                    scenario_id=args.scenario,
                    baseline_id=args.baseline,
                    artifact_id=args.artifact_id,
                    checkpoint_file=args.checkpoint_file,
                    device=args.device,
                    prepared_path=args.prepared,
                    seed=args.seed,
                    formal=args.formal,
                    source_authority_experiment=args.source_authority_experiment,
                    dataset_snapshot_id=args.dataset_snapshot_id,
                    output=args.output,
                    directory=args.directory,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
            if args.eval_command == "compare":
                return _eval_compare(
                    args.suite_id,
                    result_paths=args.results,
                    prepared_paths=args.prepared_paths,
                    output=args.output,
                    directory=args.directory,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
        if args.command == "mve":
            if args.mve_command == "suites":
                if args.mve_suites_command == "list":
                    return _mve_suites_list(as_json=args.json)
                if args.mve_suites_command == "show":
                    return _mve_suite_show(args.suite_id, as_json=args.json)
                if args.mve_suites_command == "validate":
                    return _mve_suites_validate(args.suite_id, as_json=args.json)
            if args.mve_command == "verify":
                return _mve_verify(
                    args.contract_id,
                    args.suite_id,
                    args.output,
                    receipt_ref=args.receipt_ref,
                    review_ref=args.review_ref,
                    as_json=args.json,
                )
            if args.mve_command == "r1" and args.mve_r1_command == "run":
                return _mve_r1_run(
                    args.prepared_bundle,
                    args.output,
                    args.checkpoint,
                    as_json=args.json,
                )
            if args.mve_command == "status":
                return _mve_status(
                    args.contract_id,
                    results_root=args.results_root,
                    as_json=args.json,
                )
        if args.command == "lineage" and args.lineage_command == "show":
            return _lineage_show(args.object_id, catalog_path=args.catalog, as_json=args.json)
        if args.command == "catalog":
            if args.catalog_command == "build":
                return _catalog_build(
                    repository=args.repository,
                    output=args.output,
                    as_json=args.json,
                )
            if args.catalog_command == "check":
                return _catalog_check(
                    repository=args.repository,
                    catalog_path=args.catalog,
                    as_json=args.json,
                )
    except (
        FileExistsError,
        KeyError,
        ModelNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    parser.error("unreachable command")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
