#!/usr/bin/env python3
"""Run a bounded, non-training diagnostic across all six TabUBase links.

This command is intentionally weaker than a formal evaluation.  It proves
that every link has an executable check, validates the closed manifests and
corpus bindings, and records explicit blockers instead of silently skipping a
real-data or transfer stage.  It never downloads data, starts training, or
issues a formal receipt.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from tabu_lab.catalog import DatasetAuthorityStatus, DatasetSnapshotSpec
from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.foundry import load_suite, validate_suite
from tabu_lab.experiments.runner import load_fit_experiment
from tabu_lab.experiments.s1_runner import assess_s1_feasibility, build_s1_corpus
from tabu_lab.experiments.transfer_base import (
    load_finetune_spec,
    load_icl_spec,
    load_pretrain_spec,
)
from tabu_lab.verification import list_suites, run_suite, write_result

ROOT = Path(__file__).resolve().parents[1]
CHAIN_PATH = ROOT / "evaluations" / "chain" / "tabubase-0.2.0-chain.yaml"
PASSPORT_PATH = ROOT / "evaluations" / "data" / "passports" / "tabubase-v1.yaml"
RESULT_ROOT = ROOT / "verification" / "results" / "tabu.cell.base"

F0_PATHS = (
    "experiments/fit-first/F0/F0-023-tabu-cell-base-completion-v1/preregistration.yaml",
    "experiments/fit-first/F0/F0-024-tabu-cell-base-supervised-regression-v1/preregistration.yaml",
    "experiments/fit-first/F0/F0-025-tabu-cell-base-supervised-classification-v1/preregistration.yaml",
)
S1_PATHS = (
    "experiments/fit-first/S1/S1-010-tabu-cell-base-completion-v1/preregistration.yaml",
    "experiments/fit-first/S1/S1-011-tabu-cell-base-supervised-regression-v1/preregistration.yaml",
    "experiments/fit-first/S1/S1-012-tabu-cell-base-supervised-classification-v1/preregistration.yaml",
)


class DiagnosticBlocked(RuntimeError):
    """A diagnostic method is defined but its local prerequisite is absent."""


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _check(check_id: str, fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        evidence = fn()
    except DiagnosticBlocked as exc:
        return {
            "check_id": check_id,
            "status": "blocked",
            "detail": str(exc),
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostic boundary
        return {
            "check_id": check_id,
            "status": "failed",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return {"check_id": check_id, "status": "passed", "evidence": evidence}


def _link(
    *,
    link_id: int,
    name: str,
    mode: str,
    checks: list[dict[str, Any]],
    formal_gate: str,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    failed = [item for item in checks if item["status"] == "failed"]
    soft_blocked = [item for item in checks if item["status"] == "blocked"]
    status = "failed" if failed else ("blocked" if blockers or soft_blocked else "passed")
    return {
        "link_id": link_id,
        "name": name,
        "status": status,
        "evidence_level": "local_unissued",
        "verification_mode": mode,
        "checks": checks,
        "formal_gate": formal_gate,
        "blockers": blockers or [],
        "claim_boundary": "diagnostic only; no formal receipt, score, or model claim",
    }


def _mve_link(
    *,
    link_id: int,
    name: str,
    suite_id: str,
    result_name: str,
) -> dict[str, Any]:
    suite = next(item for item in list_suites() if item.suite_id == suite_id)
    result = run_suite("tabu.cell.base", suite)
    result_path = RESULT_ROOT / result_name
    write_result(result, result_path)
    chain = yaml.safe_load(CHAIN_PATH.read_text(encoding="utf-8"))
    chain_link = next(item for item in chain["links"] if item["link_id"] == link_id)
    checks = [
        _check(
            "suite_outcome",
            lambda: {
                "suite_id": suite_id,
                "outcome": result.outcome.value,
                "result_hash": result.result_hash,
            }
            if result.outcome.value == "passed"
            else (_ for _ in ()).throw(ValueError(result.outcome.value)),
        ),
        _check(
            "chain_hash_binding",
            lambda: {
                "path": _relative(result_path),
                "result_hash": result.result_hash,
                "bound_hash": chain_link["diagnostic_result"]["result_hash"],
            }
            if chain_link["diagnostic_result"]["path"] == _relative(result_path)
            and chain_link["diagnostic_result"]["result_hash"] == result.result_hash
            else (_ for _ in ()).throw(ValueError("chain diagnostic binding is stale")),
        ),
    ]
    return _link(
        link_id=link_id,
        name=name,
        mode="structured_mve_suite",
        checks=checks,
        formal_gate="blocked_until_clean_source_and_independent_review",
    )


def _synthetic_link() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def validate_preregs() -> dict[str, Any]:
        specs: list[dict[str, Any]] = []
        for relative in F0_PATHS + S1_PATHS:
            spec = load_fit_experiment(ROOT / relative)
            if spec.contract_id != "tabu.cell.base" or spec.contract_version != "0.2.0":
                raise ValueError(f"{relative}: wrong Base contract identity")
            specs.append(
                {
                    "path": relative,
                    "experiment_id": spec.experiment_id,
                    "stage": spec.stage.value,
                    "spec_hash": spec.spec_hash,
                    "profile_id": spec.semantic.profile_id,
                }
            )
        return {"count": len(specs), "specs": specs}

    checks.append(_check("f0_s1_preregistrations", validate_preregs))

    def validate_representative_s1_corpus() -> dict[str, Any]:
        # All three S1 preregistrations are checked above.  Construct one
        # representative completion corpus here: constructing all three full
        # truth-opaque corpora in one process is needlessly memory-heavy for a
        # local smoke and does not strengthen the contract check.
        relative = S1_PATHS[0]
        spec = load_fit_experiment(ROOT / relative)
        corpus = build_s1_corpus(spec)
        _, feasibility = assess_s1_feasibility(spec, corpus)
        return {
            "experiment_id": spec.experiment_id,
            "corpus_hash": corpus.corpus_hash,
            "train_episodes": len(corpus.train_episodes),
            "validation_episodes": len(corpus.validation_episodes),
            "test_episodes": len(corpus.test_episodes),
            "feasibility_status": feasibility.status.value,
        }

    checks.append(
        _check(
            "representative_s1_corpus_binding_and_feasibility",
            validate_representative_s1_corpus,
        )
    )
    return _link(
        link_id=3,
        name="synthetic_basic_fit",
        mode="preregistration_and_corpus_compile_smoke",
        checks=checks,
        formal_gate="blocked_until_reviewed_link_1_2_and_training_receipts",
    )


BASE_SNAPSHOT_PATHS = {
    "adult-v2-task-7592-classification-micro-base": (
        "datasets/candidates/openml-adult-v2-task-7592-classification-micro-base.json"
    ),
    "adult-v2-feature-completion-micro-base": (
        "datasets/candidates/openml-adult-v2-feature-completion-micro-base.json"
    ),
    "sklearn-diabetes-regression-micro-base": (
        "datasets/candidates/sklearn-diabetes-regression-micro-base.json"
    ),
    "sklearn-diabetes-feature-completion-micro-base": (
        "datasets/candidates/sklearn-diabetes-feature-completion-micro-base.json"
    ),
}


def _real_data_link() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def validate_suites() -> dict[str, Any]:
        reports = {}
        for suite_id in ("table-completion-micro-v1", "table-supervised-micro-v1"):
            suite = load_suite(suite_id)
            report = validate_suite(suite)
            if not report.valid:
                raise ValueError(f"{suite_id}: {report.issues}")
            reports[suite_id] = {
                "suite_hash": suite.suite_hash,
                "scenario_count": len(suite.scenarios),
            }
        return reports

    checks.append(_check("suite_contracts", validate_suites))

    def validate_projection_parity() -> dict[str, Any]:
        pairs = {}
        for name in ("table-completion-micro-v1.yaml", "table-supervised-micro-v1.yaml"):
            root = ROOT / "evaluations" / "suites" / name
            package = ROOT / "src" / "tabu_lab" / "evaluation" / "foundry" / "suites" / name
            if root.read_bytes() != package.read_bytes():
                raise ValueError(f"suite projection drift: {name}")
            pairs[name] = "byte_identical"
        return pairs

    checks.append(_check("root_package_suite_parity", validate_projection_parity))

    def candidate_snapshot_registration() -> dict[str, Any]:
        """Validate the four v1 candidate snapshots emitted by offline freeze."""

        snapshots: list[dict[str, Any]] = []
        for scenario_id, relative in sorted(BASE_SNAPSHOT_PATHS.items()):
            path = ROOT / relative
            if not path.is_file():
                raise DiagnosticBlocked(f"candidate snapshot is missing: {relative}")
            snapshot = DatasetSnapshotSpec.model_validate_json(path.read_text(encoding="utf-8"))
            if snapshot.evaluation_scenario_id != scenario_id:
                raise ValueError(f"candidate snapshot scenario drift: {relative}")
            if snapshot.authority_status is not DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED:
                raise ValueError(f"candidate snapshot must remain unreviewed: {relative}")
            if snapshot.publication_eligible or snapshot.review_ids:
                raise ValueError(
                    f"candidate snapshot is unexpectedly publication eligible: {relative}"
                )
            snapshots.append(
                {
                    "scenario_id": scenario_id,
                    "snapshot_id": snapshot.dataset_snapshot_id,
                    "snapshot_sha256": snapshot.content_hash,
                    "source_sha256": snapshot.source_sha256,
                    "split_sha256": snapshot.split_manifest_sha256,
                    "truth_sidecar_sha256": snapshot.truth_sidecar_sha256,
                    "authority_review_subject_sha256": snapshot.authority_review_subject_sha256,
                }
            )
        return {"count": len(snapshots), "snapshots": snapshots}

    checks.append(_check("candidate_snapshot_registration", candidate_snapshot_registration))
    passport = yaml.safe_load(PASSPORT_PATH.read_text(encoding="utf-8"))
    datasets = passport.get("datasets", [])
    if not isinstance(datasets, list) or len(datasets) != 2:
        raise ValueError("TabUBase passport must contain Adult and Diabetes")
    candidate_hashes_ready = all(
        dataset.get(key)
        for dataset in datasets
        for key in (
            "raw_snapshot_sha256",
            "row_universe_sha256",
            "split_sha256",
            "truth_sidecar_sha256",
        )
    ) and all(
        len(dataset.get("scenario_bindings", [])) == 2
        and all(
            binding.get("snapshot_sha256")
            and binding.get("split_sha256")
            and binding.get("truth_sidecar_sha256")
            for binding in dataset.get("scenario_bindings", [])
        )
        for dataset in datasets
    )
    reviewed_authority_ready = candidate_hashes_ready and all(
        dataset.get("authority_status") == "reviewed"
        and all(
            binding.get("authority_status") == "reviewed"
            for binding in dataset.get("scenario_bindings", [])
        )
        for dataset in datasets
    )
    checks.append(
        {
            "check_id": "passport_authority_gate",
            "status": "passed",
            "evidence": {
                "passport_status": passport.get("status"),
                "authority_statuses": [
                    dataset.get("authority_status") for dataset in datasets
                ],
                "candidate_hashes_present": candidate_hashes_ready,
                "reviewed_authority_ready": reviewed_authority_ready,
            },
        }
    )
    blockers = (
        []
        if reviewed_authority_ready
        else ["independent Adult/Diabetes dataset authority review and promotion are pending"]
    )
    return _link(
        link_id=4,
        name="real_data_basic_prediction",
        mode="suite_validation_and_authority_gate",
        checks=checks,
        formal_gate="blocked_until_reviewed_dataset_passports",
        blockers=blockers,
    )


def _transfer_link() -> tuple[dict[str, Any], dict[str, Any]]:
    pretrain_path = ROOT / "experiments" / "transfer-base-v1" / "pretrain.yaml"
    icl_path = ROOT / "experiments" / "transfer-base-v1" / "icl-harness.yaml"
    finetune_path = ROOT / "experiments" / "transfer-base-v1" / "finetune-template.yaml"
    pretrain = load_pretrain_spec(pretrain_path)
    icl = load_icl_spec(icl_path)
    finetune = load_finetune_spec(finetune_path)
    raw_icl = yaml.safe_load(icl_path.read_text(encoding="utf-8"))
    checks_5 = [
        _check(
            "pretrain_schema_and_schedule",
            lambda: {
                "worlds": pretrain.worlds,
                "updates": pretrain.updates,
                "pilot_worlds": pretrain.pilot_worlds,
                "pilot_updates": pretrain.pilot_updates,
                "checkpoints": pretrain.checkpoints,
            },
        ),
        _check(
            "icl_schema_and_arms",
            lambda: {
                "heldout_worlds": icl.heldout_worlds,
                "classification_worlds": raw_icl.get("classification_worlds"),
                "regression_worlds": raw_icl.get("regression_worlds"),
                "arms": icl.arms,
                "context_sizes": icl.context_sizes,
            },
        ),
    ]
    link_5 = _link(
        link_id=5,
        name="synthetic_pretraining_icl",
        mode="transfer_schema_and_arm_contract_smoke",
        checks=checks_5,
        formal_gate="blocked_until_link_3_s1_checkpoint",
    )
    checks_6 = [
        _check(
            "finetune_schema_and_budgets",
            lambda: {
                "initialization_arms": finetune.initialization_arms,
                "tasks": [task.task_id for task in finetune.tasks],
                "seeds": finetune.seeds,
                "schedule": {
                    "learning_rates": finetune.learning_rates,
                    "updates": finetune.update_budgets,
                },
            },
        )
    ]
    link_6 = _link(
        link_id=6,
        name="synthetic_pretraining_real_finetune",
        mode="finetune_schema_and_budget_contract_smoke",
        checks=checks_6,
        formal_gate="blocked_until_link_5_selected_checkpoint",
    )
    return link_5, link_6


def build_diagnostic() -> dict[str, Any]:
    links = [
        _mve_link(
            link_id=1,
            name="component_correctness",
            suite_id="component-contract-v0",
            result_name="component-contract-v0.json",
        ),
        _mve_link(
            link_id=2,
            name="component_decoupling_extensibility_growability",
            suite_id="architecture-evolvability-v0",
            result_name="architecture-evolvability-v0.json",
        ),
        _synthetic_link(),
        _real_data_link(),
        *_transfer_link(),
    ]
    payload: dict[str, Any] = {
        "schema_version": "tabu.tabubase-six-link-diagnostic.v1",
        "chain_id": "tabu.cell.base@0.2.0-six-link",
        "model": {
            "contract_id": "tabu.cell.base",
            "contract_version": "0.2.0",
            "profiles": [
                "completion.artificial_mask.v1",
                "supervised.label_broadcast.v1",
            ],
        },
        "verification_scope": (
            "bounded local diagnostic; no training, download, formal receipt, or total score"
        ),
        "links": links,
    }
    payload["diagnostic_hash"] = canonical_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="evaluations/chain/tabubase-0.2.0-diagnostic.json",
        help="repository-relative diagnostic output path",
    )
    args = parser.parse_args()
    output = (
        (ROOT / args.output).resolve()
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    if ROOT.resolve() not in output.parents:
        raise SystemExit("--output must stay inside the repository")
    diagnostic = build_diagnostic()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": _relative(output),
                "diagnostic_hash": diagnostic["diagnostic_hash"],
                "links": [
                    {"link_id": link["link_id"], "status": link["status"]}
                    for link in diagnostic["links"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
