"""Pilot execution, dataset boundaries, and deterministic disk continuation."""

import argparse
import dataclasses
import hashlib
import json
import subprocess

import pytest
import torch
import yaml

from tabu_lab import restoration_fit as fit
from tabu_lab.cli import main
from tabu_lab.models.restoration import RestorationModel, TruthSidecar, score_episode
from tabu_lab.models.restoration.end_to_end_checks import small_config


@pytest.fixture(autouse=True)
def bounded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


def commit(prereg):
    """Disposable test repository; never touches the checkout's Git history."""
    subprocess.run(["git", "init", "-q", str(prereg.parent)], check=True)
    subprocess.run(["git", "add", prereg.name], cwd=prereg.parent, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-s",
            "-m",
            "fixture preregistration",
        ],
        cwd=prereg.parent,
        check=True,
    )


@pytest.fixture
def artifacts(tmp_path):
    data = {
        "values": [
            [0.0, 3.0, 0],
            [1.0, 1.0, 1],
            [2.0, 4.0, 0],
            [3.0, 2.0, 1],
            [9000.0, 8000.0, 0],
            [7000.0, 6000.0, 1],
        ],
        "features": [
            {"kind": "numeric", "domain": []},
            {"kind": "numeric", "domain": []},
            {"kind": "nominal", "domain": ["a", "b"]},
        ],
        "target_kind": "nominal",
        "domain": ["a", "b"],
        "splits": {"train": [0, 1, 2, 3], "test": [4, 5]},
    }
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(data))
    spec = {
        "schema": fit.SCHEMA,
        "status": "local_unissued",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "expected_rows": 6,
        "columns": [0, 1],
        "row_count": 4,
        "mask_fraction": 0.25,
        "mask_bank_size": 2,
        "seeds": {"model": 11, "rows": 12, "masks": 13, "codes": 14},
        "model": small_config(kind="direct").as_dict(),
        "optimizer": {
            "kind": "adamw",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "grad_clip": 1.0,
        },
        "max_updates": 3,
        "max_wall_seconds": 120,
        "gradient_accumulation": 1,
        "checkpoint_every": 1,
    }
    prereg = tmp_path / "preregistration.yaml"
    prereg.write_text(yaml.safe_dump(spec))
    return argparse.Namespace(
        preregistration=prereg,
        dataset=dataset,
        output_root=tmp_path / "attempt",
        device="cpu",
        resume=None,
        execute=False,
    )


def alter_spec(args, **changes):
    spec = yaml.safe_load(args.preregistration.read_text())
    spec.update(changes)
    args.preregistration.write_text(yaml.safe_dump(spec))


def execute(args):
    commit(args.preregistration)
    args.execute = True
    return fit.run_fit(args)


def test_plan_is_read_only_and_cli_does_not_allocate_model(artifacts, monkeypatch, capsys):
    def unexpected_model(*args, **kwargs):
        raise AssertionError("plan allocated a model")

    monkeypatch.setattr(fit, "RestorationModel", unexpected_model)
    result = main(
        [
            "restoration",
            "fit",
            "--preregistration",
            str(artifacts.preregistration),
            "--dataset",
            str(artifacts.dataset),
            "--output-root",
            str(artifacts.output_root),
        ]
    )
    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["outcome"] == "planned" and not plan["execution_started"]
    assert not artifacts.output_root.exists()


def test_json_exponents_inside_yaml_filename_remain_numbers(artifacts):
    spec = yaml.safe_load(artifacts.preregistration.read_text())
    artifacts.preregistration.write_text(json.dumps(spec))
    assert "1e-08" in artifacts.preregistration.read_text()
    plan = fit.prepare_plan(artifacts.preregistration, artifacts.dataset)
    assert plan.spec["optimizer"]["eps"] == 1e-8
    assert plan.config.encoder.scale_floor == 1e-6


def test_fixed_masks_train_split_fp64_and_all_observed_targets(artifacts):
    data = json.loads(artifacts.dataset.read_text())
    data["values"][0][0] = 1e10 + 0.25
    artifacts.dataset.write_text(json.dumps(data))
    alter_spec(artifacts, dataset_sha256=fit._hash(artifacts.dataset))
    plan = fit.prepare_plan(artifacts.preregistration, artifacts.dataset)
    same = fit.prepare_plan(artifacts.preregistration, artifacts.dataset)
    assert plan.identity == same.identity
    assert set(plan.identity["selected_train_row_ids"]) == {0, 1, 2, 3}
    assert plan.summary["excluded_columns"] == [2]
    assert plan.values[0].dtype == torch.float64
    assert (plan.values[0] == 1e10 + 0.25).any()
    assert not torch.equal(plan.values[0], plan.values[0].float().double())
    for index in range(2):
        inputs, request, truth = plan.episode(index)
        assert len(request.targets) == 8
        assert (inputs.visible.sum(0) == 3).all()
        assert (inputs.query.sum(0) == 1).all()
        assert torch.equal(inputs.query, same.episode(index)[0].query)
        for col in range(2):
            assert (inputs.values[col][inputs.query[:, col]] == 0).all()
            assert torch.equal(truth.values[col], plan.values[col])


def test_hidden_truth_only_changes_scorer(artifacts):
    plan = fit.prepare_plan(artifacts.preregistration, artifacts.dataset)
    inputs, request, truth = plan.episode(0)
    model = RestorationModel(plan.config).double()
    altered = tuple(value + inputs.query[:, index] * 20 for index, value in enumerate(truth.values))
    first = score_episode(model, inputs, request, truth)
    second = score_episode(model, inputs, request, TruthSidecar(altered, truth.states))
    for left, right in zip(first.output.columns, second.output.columns, strict=True):
        torch.testing.assert_close(left.result.encoding, right.result.encoding, rtol=0, atol=0)
    assert first.loss != second.loss


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"columns": [2]}, "must all be numeric"),
        ({"row_count": 5}, "exceeds training split"),
        ({"mask_fraction": 0.8}, "at least two visible"),
        ({"columns": []}, "explicitly list"),
    ],
)
def test_invalid_predeclared_sampling_fails_without_retry(artifacts, changes, message):
    alter_spec(artifacts, **changes)
    with pytest.raises(ValueError, match=message):
        fit.prepare_plan(artifacts.preregistration, artifacts.dataset)


def test_actual_committed_preregistration_gate_and_failure_receipt(artifacts):
    artifacts.execute = True
    receipt = fit.run_fit(artifacts)
    assert receipt["outcome"] == "failed" and not receipt["execution_started"]
    assert "committed preregistration" in receipt["error"]
    assert (artifacts.output_root / "terminal.json").exists()
    assert not (artifacts.output_root / "checkpoint.pt").exists()
    commit(artifacts.preregistration)
    artifacts.preregistration.write_text(artifacts.preregistration.read_text() + "\n")
    with pytest.raises(ValueError, match="differ from committed"):
        fit._require_committed_preregistration(artifacts.preregistration)


def test_disk_resume_matches_uninterrupted_updates_and_optimizer(artifacts, monkeypatch):
    baseline = execute(artifacts)
    assert baseline["outcome"] == "completed", baseline
    complete = torch.load(artifacts.output_root / "checkpoint.pt", weights_only=True)
    artifacts.output_root = artifacts.output_root.parent / "interrupted"
    actual_step = torch.optim.AdamW.step
    calls = 0

    def interrupt_second(optimizer, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return actual_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", interrupt_second)
    interrupted = fit.run_fit(artifacts)
    assert interrupted["outcome"] == "interrupted" and interrupted["completed_updates"] == 1
    parent = artifacts.output_root / "checkpoint.pt"
    monkeypatch.setattr(torch.optim.AdamW, "step", actual_step)
    artifacts.output_root = artifacts.output_root.parent / "resumed"
    artifacts.resume = parent
    resumed = fit.run_fit(artifacts)
    assert resumed["outcome"] == "completed" and resumed["completed_updates"] == 3, resumed
    continued = torch.load(artifacts.output_root / "checkpoint.pt", weights_only=True)
    for name, value in complete["model"].items():
        torch.testing.assert_close(value, continued["model"][name], rtol=0, atol=0)
    for key, state in complete["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value, continued["optimizer"]["state"][key][name], rtol=0, atol=0
            )
    assert complete["sampler_cursor"] == continued["sampler_cursor"] == 3
    assert set(resumed["final"]["by_state"]) == {"retained", "query"}
    assert resumed["reserved_rows_used"] == 0
    assert resumed["parent_checkpoint_sha256"] == fit._hash(parent)
    with pytest.raises(FileExistsError, match="new attempt"):
        fit.run_fit(artifacts)


def test_resume_rejects_preregistration_and_source_drift(artifacts, monkeypatch):
    assert execute(artifacts)["outcome"] == "completed"
    artifacts.resume = artifacts.output_root / "checkpoint.pt"
    artifacts.output_root = artifacts.output_root.parent / "changed-source"
    original_source = fit.source_identity
    monkeypatch.setattr(fit, "source_identity", lambda: {"sha256": "different"})
    receipt = fit.run_fit(artifacts)
    assert receipt["outcome"] == "failed" and "identity drift" in receipt["error"]
    assert not (artifacts.output_root / "checkpoint.pt").exists()
    monkeypatch.setattr(fit, "source_identity", original_source)
    alter_spec(artifacts, max_updates=4)
    commit(artifacts.preregistration)
    artifacts.output_root = artifacts.output_root.parent / "changed-preregistration"
    receipt = fit.run_fit(artifacts)
    assert receipt["outcome"] == "failed" and "identity drift" in receipt["error"]


def test_nonfinite_loss_kills_before_update_and_keeps_terminal(artifacts, monkeypatch):
    actual_score = fit.score_episode

    def nonfinite(model, *args, **kwargs):
        result = actual_score(model, *args, **kwargs)
        return (
            dataclasses.replace(result, loss=result.loss * float("nan"))
            if model.training
            else result
        )

    monkeypatch.setattr(fit, "score_episode", nonfinite)
    receipt = execute(artifacts)
    assert receipt["outcome"] == "failed" and receipt["completed_updates"] == 0
    assert receipt["error"] == "nonfinite training loss"
    assert json.loads((artifacts.output_root / "terminal.json").read_text())["outcome"] == "failed"


def test_wall_limit_and_nonfinite_gradient_kill(artifacts, monkeypatch):
    actual_score = fit.score_episode

    def bad_gradient(model, *args, **kwargs):
        result = actual_score(model, *args, **kwargs)
        if model.training:
            result.loss.register_hook(lambda gradient: gradient * float("nan"))
        return result

    monkeypatch.setattr(fit, "score_episode", bad_gradient)
    receipt = execute(artifacts)
    assert receipt["outcome"] == "failed" and receipt["completed_updates"] == 0
    assert "nonfinite training gradients" in receipt["error"]
    artifacts.output_root = artifacts.output_root.parent / "wall-limit"

    def hit_limit(deadline):
        raise fit.WallLimit("bounded test wall limit")

    monkeypatch.setattr(fit, "_check_wall", hit_limit)
    receipt = fit.run_fit(artifacts)
    assert receipt["outcome"] == "wall_limit" and receipt["completed_updates"] == 0


def test_sequential_accumulation_commits_episode_cursor(artifacts):
    alter_spec(artifacts, gradient_accumulation=2, max_updates=1)
    result = execute(artifacts)
    assert result["outcome"] == "completed" and result["sampler_cursor"] == 2
    assert result["initial"]["by_state"]["retained"]["count"] == 12
    assert result["initial"]["by_state"]["query"]["count"] == 4


def test_partial_optimizer_update_is_discarded_at_resume_boundary(artifacts, monkeypatch):
    actual_step = torch.optim.AdamW.step

    def partial_step(optimizer, *args, **kwargs):
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].add_(99)
        raise KeyboardInterrupt

    monkeypatch.setattr(torch.optim.AdamW, "step", partial_step)
    result = execute(artifacts)
    assert result["outcome"] == "interrupted" and result["checkpoint_step"] == 0
    saved = torch.load(artifacts.output_root / "checkpoint.pt", weights_only=True)
    torch.manual_seed(11)
    plan = fit.prepare_plan(artifacts.preregistration, artifacts.dataset)
    initial = RestorationModel(plan.config).double()
    for name, value in initial.state_dict().items():
        torch.testing.assert_close(value, saved["model"][name], rtol=0, atol=0)
    monkeypatch.setattr(torch.optim.AdamW, "step", actual_step)


def test_nonfinite_updated_parameters_are_killed_and_not_checkpointed(artifacts, monkeypatch):
    actual_step = torch.optim.AdamW.step

    def poison_step(optimizer, *args, **kwargs):
        actual_step(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(torch.optim.AdamW, "step", poison_step)
    result = execute(artifacts)
    assert result["outcome"] == "failed" and "nonfinite parameters" in result["error"]
    saved = torch.load(artifacts.output_root / "checkpoint.pt", weights_only=True)
    assert saved["step"] == 0 and fit._finite_state(saved["model"])


def test_overflowing_raw_metrics_still_persist_failed_receipt(artifacts, monkeypatch):
    actual_score = fit.score_episode

    def overflow(model, *args, **kwargs):
        score = actual_score(model, *args, **kwargs)
        columns = tuple(
            dataclasses.replace(prediction, decoded=torch.full_like(prediction.decoded, 1e200))
            for prediction in score.output.columns
        )
        return dataclasses.replace(score, output=dataclasses.replace(score.output, columns=columns))

    monkeypatch.setattr(fit, "score_episode", overflow)
    result = execute(artifacts)
    assert result["outcome"] == "failed" and "evaluation error" in result["error"]
    assert json.loads((artifacts.output_root / "terminal.json").read_text())["outcome"] == "failed"


def test_cuda_unavailable_never_falls_back_and_has_receipt(artifacts, monkeypatch):
    artifacts.device = "cuda:0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = execute(artifacts)
    assert result["outcome"] == "failed" and result["error"] == "CUDA unavailable; no CPU fallback"
    assert not result["execution_started"]
