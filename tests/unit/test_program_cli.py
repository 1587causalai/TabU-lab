from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabu_lab.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_program_validate_and_impact_commands_are_machine_readable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["program", "validate", "--repository", str(ROOT)]) == 0
    validation = json.loads(capsys.readouterr().out)
    assert validation["status"] == "valid"
    assert validation["node_count"] == 44
    assert validation["edge_count"] == 7

    assert main(
        [
            "program",
            "impact",
            "--repository",
            str(ROOT),
            "--from-program",
            "tabu.pretraining.query-base@1.0.0",
            "--to-program",
            "tabu.pretraining.query-base-eval-v2@1.1.0-exercise",
        ]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    actions = {action["object_kind"]: action["disposition"] for action in report["actions"]}
    assert actions["evaluation"] == "rescore"
    assert actions["checkpoint"] == "reuse_exact"


def test_program_freeze_is_non_overwriting(tmp_path: Path) -> None:
    frozen = tmp_path / "base.frozen.json"
    arguments = [
        "program",
        "freeze",
        "--repository",
        str(ROOT),
        "--program",
        "tabu.pretraining.query-base@1.0.0",
        "--output",
        str(frozen),
    ]

    assert main(arguments) == 0
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    assert payload["lane"] == "evidence"
    assert payload["evidence_status"] == "frozen_not_run"
    with pytest.raises(SystemExit):
        main(arguments)


def test_program_help_lists_the_evolution_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["program", "--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    for command in ("validate", "resolve", "diff", "impact", "freeze", "run", "evaluate"):
        assert command in output
