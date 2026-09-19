"""Qualification, interrupted evaluation, and terminal wall-budget boundaries."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabu_lab.curriculum_v53 import preflight, runner
from tabu_lab.curriculum_v53.protocol import load_plan


@pytest.fixture
def manifest(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples" / "curriculum_v53_fixture.py"
    path = runpy.run_path(str(example))["create_fixture"](tmp_path / "data")
    spec = json.loads(path.read_text())
    spec["stages"] = spec["stages"][:1]
    spec["stages"][0].update(max_updates=1, max_seconds=30.0, probes=[])
    path.write_text(json.dumps(spec))
    return path


@pytest.mark.parametrize("native_generation", [False, True])
def test_completed_cursor_cannot_escape_exhausted_final_save(
    manifest,
    tmp_path,
    monkeypatch,
    native_generation,
):
    plan = load_plan(manifest)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now))
    original_save = runner.save_checkpoint

    def save(path, **kwargs):
        digest = original_save(path, **kwargs)
        if Path(path).name == "stage-000-final.pt":
            clock.now += 31.0
        return digest

    monkeypatch.setattr(runner, "save_checkpoint", save)
    first = runner.run(plan, tmp_path / "first")
    assert first["phase"] == "completed"
    assert first["outcome"] == "budget_exhausted"
    checkpoint = tmp_path / "first" / "checkpoint-progress.pt"
    if native_generation:
        checkpoint = checkpoint.resolve()
    second = runner.run(plan, tmp_path / "second", resume=checkpoint)
    assert second["outcome"] == "budget_exhausted"
    assert second["update"] == first["update"] == 1


def test_preflight_never_optimizes_probe_only_table(manifest, tmp_path):
    spec = json.loads(manifest.read_text())
    # A distinct kind in the same cohort would require an unregistered recipe
    # if preflight accidentally treated the probe-only table as training data.
    spec["tables"][2].update(role="probe", cohort="old", kind="real")
    spec["probes"] = []
    manifest.write_text(json.dumps(spec))
    plan = load_plan(manifest)
    result = preflight.preflight(plan, tmp_path / "qualification")
    assert result["outcome"] == "passed", result
    assert {item["table"] for item in result["probes"]} <= {"old_a", "old_b"}


def test_preflight_last_probe_overrun_is_not_passed(manifest, tmp_path, monkeypatch):
    plan = load_plan(manifest)
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(preflight, "time", SimpleNamespace(monotonic=lambda: clock.now))
    original = preflight.train_step

    def step(*args, **kwargs):
        result = original(*args, **kwargs)
        clock.now += 0.5
        return result

    monkeypatch.setattr(preflight, "train_step", step)
    result = preflight.preflight(plan, tmp_path / "qualification", max_seconds=1.0)
    assert result["outcome"] == "failed"
    assert result["error_type"] == "TimeoutError"
    assert result["seconds"] == 1.5


def test_preflight_keyboard_interrupt_has_terminal_receipt(manifest, tmp_path, monkeypatch):
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(preflight, "train_step", interrupt)
    result = preflight.preflight(load_plan(manifest), tmp_path / "qualification")
    assert result["outcome"] == "interrupted"
    assert json.loads((tmp_path / "qualification" / "terminal.json").read_text()) == result


def test_frozen_evaluation_interrupt_has_terminal_receipt(manifest, tmp_path, monkeypatch):
    plan = load_plan(manifest)
    run = runner.run(plan, tmp_path / "train", max_updates_this_invocation=0)
    assert run["outcome"] == "stopped"

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "evaluate_probe", interrupt)
    result = runner.evaluate_checkpoint(
        plan,
        tmp_path / "train" / "checkpoint-progress.pt",
        tmp_path / "evaluation",
        probes=["final_test"],
    )
    assert result["outcome"] == "interrupted"
    assert json.loads((tmp_path / "evaluation" / "terminal.json").read_text()) == result
