from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import tabu_lab.adapters.cataloged_checkpoint as cataloged_checkpoint_module
import tabu_lab.adapters.checkpoint_model as checkpoint_model_module
import tabu_lab.adapters.real_eval_data as real_eval_data_module
import tabu_lab.cli as cli_module
import tabu_lab.evaluation.foundry as foundry_module
from tabu_lab.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_formal_run_help_and_boolean_self_authorization_gate(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["experiments", "run", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--authorization-catalog" in help_text
    assert "never authorizes" in help_text

    assert (
        main(
            [
                "experiments",
                "run",
                "not-read.yaml",
                "--output-root",
                str(tmp_path / "formal-runs"),
                "--formal",
                "--source-reviewed",
            ]
        )
        == 2
    )
    assert "--formal requires --authorization-catalog" in capsys.readouterr().err


def test_catalog_experiment_and_lineage_commands_share_one_index(capsys) -> None:
    catalog = str(ROOT / "catalog.json")
    command = [
        "catalog",
        "check",
        "--repository",
        str(ROOT),
        "--catalog",
        catalog,
        "--json",
    ]
    assert main(command) == 0
    check = json.loads(capsys.readouterr().out)
    assert check["ok"] is True

    assert main(["experiments", "status", "--catalog", catalog, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert set(status["counts"]) == {"draft"}
    assert status["counts"]["draft"] >= 7
    assert {item["object_id"] for item in status["experiments"]} >= {
        "F0-001-tabuf-v1",
        "F0-007-tabu4rec-v1",
    }
    assert set(status["active_draft_experiment_ids"]) == {
        "F0-008-tabuf-identifiable-v2",
        "F0-009-tabu-unit-row-identifiable-v2",
        "F0-020-tabu-unit-pair-local-linear-contract-v1",
        "F0-011-tabu4graph-row-unit-v2",
        "F0-022-tabu4rec-cell-global-support-v1",
        "F0-015-tabul-unit-linked-address-v3",
        "F0-018-tabufl-balanced-16f-v5",
        "S1-001-tabuf-latent-mixed-v1",
        "S1-002-tabu-unit-row-latent-mixed-v1",
        "S1-003-tabu-unit-pair-latent-mixed-v1",
        "S1-004-tabul-compositional-xor-v1",
        "S1-005-tabufl-joint-compositional-xor-v1",
        "S1-006-tabu4graph-community-v1",
        "S1-007-tabu4graph-diffusion-v1",
        "S1-008-tabu4rec-rating-v1",
        "S1-009-tabu4rec-preference-v1",
        "F0-023-tabu-cell-base-completion-v1",
        "F0-024-tabu-cell-base-supervised-regression-v1",
        "F0-025-tabu-cell-base-supervised-classification-v1",
        "S1-010-tabu-cell-base-completion-v1",
        "S1-011-tabu-cell-base-supervised-regression-v1",
        "S1-012-tabu-cell-base-supervised-classification-v1",
    }
    assert "G000-tabuf-artificial-mask" in status["superseded_draft_experiment_ids"]

    assert main(["lineage", "show", "tabuf", "--catalog", catalog, "--json"]) == 0
    lineage = json.loads(capsys.readouterr().out)
    assert lineage["object"]["kind"] == "model_contract"
    assert any(edge["relation"] == "implements" for edge in lineage["lineage"])


def test_empty_run_and_artifact_collections_are_explicit(capsys) -> None:
    catalog = str(ROOT / "catalog.json")
    assert main(["runs", "list", "--catalog", catalog, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["artifacts", "list", "--catalog", catalog, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_eval_validate_and_dry_run_never_fabricate_real_data(capsys) -> None:
    assert main(["eval", "suites", "validate", "--json"]) == 0
    reports = json.loads(capsys.readouterr().out)
    assert len(reports) == 6
    assert all(report["valid"] for report in reports)

    assert main(["eval", "dry-run", "table-supervised-micro-v0", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ready"] is False
    assert report["would_execute"] is False
    assert all(item["blockers"] for item in report["scenarios"])


def test_eval_suite_inspection_uses_catalog_and_explicit_source_must_match(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_source_read(*args, **kwargs):
        raise AssertionError("default suite inspection must not read packaged manifests")

    monkeypatch.setattr(foundry_module, "list_suite_ids", unexpected_source_read)
    monkeypatch.setattr(foundry_module, "load_suite", unexpected_source_read)
    assert main(["eval", "suites", "list", "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 6
    assert main(["eval", "suites", "show", "table-supervised-micro-v0", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["suite_id"] == "table-supervised-micro-v0"

    monkeypatch.undo()
    source = ROOT / "evaluations" / "suites"
    copied = tmp_path / "suites"
    shutil.copytree(source, copied)
    stale = copied / "table-supervised-micro-v0.yaml"
    stale.write_text(
        stale.read_text().replace(
            "Small real-data supervised classification",
            "Stale real-data supervised classification",
        )
    )
    assert (
        main(
            [
                "eval",
                "suites",
                "validate",
                "table-supervised-micro-v0",
                "--directory",
                str(copied),
                "--json",
            ]
        )
        == 2
    )
    assert "source/catalog hash parity failed" in capsys.readouterr().err


def test_eval_run_wires_cataloged_checkpoint_without_exposing_truth(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.safetensors"
    checkpoint_path.write_bytes(b"resolved by the isolated adapter")
    output_path = tmp_path / "result.json"

    prepared = SimpleNamespace(
        scenario_id="scenario-0",
        test=(SimpleNamespace(example_id="test-0"),),
        topology_checks=(),
    )

    class PreparedSchema:
        @classmethod
        def model_validate(cls, value):
            assert value == {}
            return prepared

    scenario = SimpleNamespace(scenario_id="scenario-0", baselines=())
    suite = SimpleNamespace(scenarios=(scenario,))

    def pointer(uri: str, token: str) -> SimpleNamespace:
        return SimpleNamespace(uri=uri, sha256=token * 64)

    artifact = SimpleNamespace(
        artifact_id="run-0.checkpoint",
        producer_run_id="run-0",
        producer_receipt=pointer("runs/run-0/receipt.json", "d"),
        model_spec=pointer("specs/models/tabuf.yaml", "a"),
        semantic_config=pointer("runs/run-0/resolved-configs/semantic.json", "b"),
        compiler_manifest=pointer("runs/run-0/compiler-manifest.json", "c"),
    )
    catalog = object()
    adapter = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_load_eval_suite_for_cli",
        lambda *args, **kwargs: suite,
    )
    monkeypatch.setattr(cli_module, "_load_catalog_file", lambda path: catalog)
    monkeypatch.setattr(foundry_module, "PreparedScenario", PreparedSchema)
    monkeypatch.setattr(
        cataloged_checkpoint_module,
        "resolve_model_artifact",
        lambda resolved_catalog, artifact_id: artifact,
    )

    def read_pointer(**kwargs):
        return {"uri": kwargs["uri"]}

    monkeypatch.setattr(cli_module, "_read_catalog_pointer_mapping", read_pointer)

    def launch(**kwargs):
        observed["launch"] = kwargs
        return adapter

    monkeypatch.setattr(
        checkpoint_model_module,
        "cataloged_checkpoint_launch_spec",
        launch,
    )
    monkeypatch.setattr(
        real_eval_data_module,
        "checkpoint_blind_example",
        lambda resolved, *, example_id: f"blind:{example_id}",
    )
    monkeypatch.setattr(
        real_eval_data_module,
        "checkpoint_topology_cases",
        lambda resolved: (),
    )

    raw_result = SimpleNamespace(
        content_hash="e" * 64,
        result_id="eval-result-0",
        status=SimpleNamespace(value="succeeded"),
        model_dump=lambda mode: {"result_id": "eval-result-0", "status": "succeeded"},
    )
    execution_receipt = SimpleNamespace(
        receipt_hash="f" * 64,
        receipt_id="evalreceipt-fixture",
    )
    result = SimpleNamespace(
        content_hash="e" * 64,
        result_id="eval-result-0",
        status=SimpleNamespace(value="succeeded"),
        execution_receipt=execution_receipt,
        model_dump=lambda mode: {
            "result_id": "eval-result-0",
            "status": "succeeded",
            "execution_receipt": {"receipt_id": "evalreceipt-fixture"},
        },
    )

    def run_evaluation(*args, **kwargs):
        observed["run"] = kwargs
        return raw_result

    monkeypatch.setattr(foundry_module, "run_evaluation", run_evaluation)

    def bind_evaluation_receipt(value, **kwargs):
        assert value is raw_result
        observed["receipt"] = kwargs
        return result

    monkeypatch.setattr(
        foundry_module,
        "bind_evaluation_receipt",
        bind_evaluation_receipt,
    )

    assert (
        main(
            [
                "eval",
                "run",
                "suite-0",
                "--scenario",
                "scenario-0",
                "--artifact",
                artifact.artifact_id,
                "--checkpoint-file",
                str(checkpoint_path),
                "--prepared",
                str(prepared_path),
                "--seed",
                "17",
                "--output",
                str(output_path),
                "--catalog",
                str(tmp_path / "catalog.json"),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["result_id"] == "eval-result-0"
    assert json.loads(output_path.read_text())["status"] == "succeeded"
    assert observed["receipt"]["source_identity"].issuance_status == "local_unissued"
    launch_args = observed["launch"]
    assert launch_args["catalog"] is catalog
    assert launch_args["artifact_id"] == artifact.artifact_id
    assert launch_args["checkpoint_path"] == str(checkpoint_path)
    assert observed["run"]["adapter"] is adapter
    assert observed["run"]["blind_examples"] == ("blind:test-0",)
    producer = observed["run"]["producer"]
    assert producer.run_id == artifact.producer_run_id
    assert producer.receipt_sha256 == artifact.producer_receipt.sha256
    assert producer.publication_eligible is True
