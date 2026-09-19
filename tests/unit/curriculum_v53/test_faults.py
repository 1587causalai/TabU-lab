"""Failure injection at checkpoint and stage boundaries, without running fits."""

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tabu_lab.curriculum_v53 import artifacts, runner
from tabu_lab.curriculum_v53.protocol import load_plan
from tabu_lab.models.restoration_v53 import V53Model
from tabu_lab.restoration_optimizers import adamw


@pytest.fixture
def fixture_manifest(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples" / "curriculum_v53_fixture.py"
    create_fixture = runpy.run_path(str(example))["create_fixture"]
    return create_fixture(tmp_path / "fixture")


def _bounded_plan(path):
    spec = json.loads(path.read_text())
    spec["stages"] = spec["stages"][:2]
    for stage in spec["stages"]:
        stage.update(max_updates=1, evaluate_every=1, checkpoint_every=1, max_seconds=1.)
    spec["stages"][0]["probes"] = ["old_fit"]
    spec["stages"][1]["probes"] = ["new_fit"]
    path.write_text(json.dumps(spec))
    return load_plan(path)


@pytest.mark.parametrize("failure_timing", ["before_sidecar", "after_sidecar"])
def test_sidecar_failure_preserves_previous_checkpoint_pointer(
    fixture_manifest, tmp_path, monkeypatch, failure_timing,
):
    plan = load_plan(fixture_manifest)
    model = V53Model(plan.config).double()
    optimizer = adamw(model, plan.optimizer)
    state = runner._new_state(plan.spec["stages"])
    path = tmp_path / "checkpoint-progress.pt"
    options = dict(plan=plan, model=model, optimizer=optimizer, state=state,
                   runtime={"device": "cpu"}, lineage=[])
    original_digest = artifacts.save_checkpoint(path, **options)
    original_generation = path.resolve()
    original_payload, _ = artifacts.load_checkpoint(path)
    with torch.no_grad():
        next(model.parameters()).add_(1.)
    state.update(update=1, cursor=1, phase="train")
    atomic_json = artifacts.atomic_json

    def fail_sidecar(destination, value):
        if Path(destination) == path.with_suffix(".json"):
            if failure_timing == "after_sidecar":
                atomic_json(destination, value)
            raise OSError("injected sidecar write failure")
        atomic_json(destination, value)

    monkeypatch.setattr(artifacts, "atomic_json", fail_sidecar)
    with pytest.raises(OSError, match="injected sidecar"):
        artifacts.save_checkpoint(path, **options)
    assert path.resolve() == original_generation
    payload, digest = artifacts.load_checkpoint(path)
    assert digest == original_digest
    assert payload["state"]["update"] == 0
    for name, value in payload["model"].items():
        assert torch.equal(value, original_payload["model"][name])


def test_final_evaluation_over_budget_cannot_enter_next_stage(
    fixture_manifest, tmp_path, monkeypatch,
):
    plan = _bounded_plan(fixture_manifest)
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: clock.now))
    evaluations, updates = [], []

    def evaluate(_model, _plan, probe, _device, **_kwargs):
        evaluations.append(probe["name"])
        if len(evaluations) == 2:  # The first stage's final bank crosses its deadline.
            clock.now = 2.
        return {"macro": {"query_numeric_mse": 0.}}

    def step(_model, _optimizer, _plan, table, _recipe, index, _device, _loss, **kwargs):
        updates.append(kwargs["namespace"])
        clock.now += 0.1
        return {
            "table": table.name, "cohort": table.cohort, "table_episode_index": index,
            "loss": 0., "seconds": 0.1,
            "episode": {"query_count": 1, "row_ids": [0, 1, 2],
                        "query_addresses": [[0, 0]]},
        }

    monkeypatch.setattr(runner, "evaluate_probe", evaluate)
    monkeypatch.setattr(runner, "train_step", step)
    result = runner.run(plan, tmp_path / "budget-attempt")
    assert result["outcome"] == "budget_exhausted"
    assert result["stage_index"] == 0
    assert result["cursor"] == result["update"] == 1
    assert result["stage_verdicts"] == []
    assert result["stage_seconds"][0] >= plan.spec["stages"][0]["max_seconds"]
    assert result["stage_seconds"][1] == 0
    assert evaluations == ["old_fit", "old_fit"]
    assert updates == ["fit"]


def test_nonfinite_step_does_not_replace_last_valid_checkpoint(
    fixture_manifest, tmp_path, monkeypatch,
):
    plan = _bounded_plan(fixture_manifest)
    # No wall-clock-dependent admission decision is involved in this test.
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 0.))
    monkeypatch.setattr(runner, "evaluate_probe", lambda *_a, **_kw: {"macro": {}})
    before = {}
    output = tmp_path / "nonfinite-attempt"
    checkpoint = output / "checkpoint-progress.pt"

    def fail_step(model, *_args, **_kwargs):
        before["digest"] = artifacts.sha256(checkpoint)
        before["generation"] = checkpoint.resolve()
        before["weights"] = copy.deepcopy(model.state_dict())
        with torch.no_grad():
            next(model.parameters()).fill_(float("inf"))
        raise FloatingPointError("injected nonfinite optimizer result")

    monkeypatch.setattr(runner, "train_step", fail_step)
    result = runner.run(plan, output)
    assert result["outcome"] == "failed"
    assert result["error_type"] == "FloatingPointError"
    assert result["update"] == result["durable_update"] == 0
    assert result["checkpoint_sha256"] == before["digest"]
    assert checkpoint.resolve() == before["generation"]
    payload, digest = artifacts.load_checkpoint(checkpoint)
    assert digest == before["digest"]
    assert artifacts.finite_state(payload["model"])
    for name, tensor in payload["model"].items():
        assert torch.equal(tensor, before["weights"][name])
