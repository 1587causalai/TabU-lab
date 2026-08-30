from __future__ import annotations

import json
from pathlib import Path

import pytest

from tabu_lab.cli import main


def test_tabur_optimize_resolves_without_overwrite(tmp_path: Path) -> None:
    prereg = tmp_path / "prereg.yaml"
    prereg.write_text(
        "experiment: query_row_finetune_lift\nparameters:\n  dataset_ids: [diabetes]\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    assert main(
        [
            "tabur",
            "optimize",
            "--preregistration",
            str(prereg),
            "--device",
            "cpu",
            "--output-root",
            str(output),
        ]
    ) == 0
    resolved = json.loads((output / "resolved-config.json").read_text(encoding="utf-8"))
    assert resolved["run_status"] == "not_run"
    assert resolved["resolved_config"]["device"] == "cpu"
    with pytest.raises(SystemExit):
        main(
            [
                "tabur",
                "optimize",
                "--preregistration",
                str(prereg),
                "--output-root",
                str(output),
            ]
        )


def test_tabur_optimize_help_is_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["tabur", "optimize", "--help"])
    assert exc.value.code == 0
    assert "--preregistration" in capsys.readouterr().out
