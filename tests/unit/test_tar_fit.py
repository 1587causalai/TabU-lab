from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tabu_lab.tar_data import full_train_test_split, validate_full_dataset
from tabu_lab.tar_fit import run_fit, sha

ROOT = Path(__file__).resolve().parents[2] / "experiments/local/tar-full-data-fit"


def test_frozen_data_and_split_boundaries():
    spec = json.loads((ROOT / "preregistration.yaml").read_text())
    for name, digest in spec["datasets"].items():
        p = ROOT / "data" / f"{name}.json"
        assert sha(p) == digest
        data = json.loads(p.read_text())
        coverage = validate_full_dataset(data, spec["expected_rows"][name])
        assert coverage["unused_rows"] == 0
        assert data["splits"] == full_train_test_split(
            data["values"], seed=spec["split_seed"], stratified=name == "iris"
        )
        expected = (120, 30) if name == "iris" else (353, 89)
        assert (coverage["train_rows"], coverage["test_rows"]) == expected
        if name == "iris":
            assert Counter(data["values"][i][-1] for i in data["splits"]["test"]) == {
                0: 10,
                1: 10,
                2: 10,
            }


def test_busy_gpu_fails_closed_and_leaves_result(tmp_path, monkeypatch):
    from tabu_lab import tar_fit

    monkeypatch.setattr(
        tar_fit, "gpu_preflight", lambda: dict(ready=False, compute_process_count=1)
    )
    args = SimpleNamespace(
        preregistration=ROOT / "preregistration.yaml",
        dataset="diabetes",
        seed=1729,
        device="cuda:0",
        smoke=False,
        output_root=tmp_path / "blocked",
    )
    result = run_fit(args)
    assert result["outcome"] == "blocked_resources"
    assert result["status"] == "local_unissued"
    assert not (args.output_root / "checkpoint").exists()
    frozen = json.loads((args.output_root / "result.json").read_text())
    assert frozen == result
    with pytest.raises(FileExistsError):
        run_fit(args)


def test_preregistered_identity_and_cpu_boundary(tmp_path):
    args = SimpleNamespace(
        preregistration=ROOT / "preregistration.yaml",
        dataset="diabetes",
        seed=123,
        device="cpu",
        smoke=False,
        output_root=tmp_path / "invalid",
    )
    with pytest.raises(ValueError, match="seed"):
        run_fit(args)
    args.seed = 1729
    with pytest.raises(ValueError, match="reserved"):
        run_fit(args)
    assert not args.output_root.exists()


def test_fixed_protocol_requires_historical_source(tmp_path):
    args = SimpleNamespace(
        preregistration=ROOT.parent / "tar-fixed-fit/preregistration.yaml",
        output_root=tmp_path / "obsolete",
    )
    with pytest.raises(ValueError, match="archived"):
        run_fit(args)
    assert not args.output_root.exists()


def test_interrupt_leaves_receipt(tmp_path, monkeypatch):
    from tabu_lab import tar_fit

    def stop():
        raise KeyboardInterrupt("test stop")

    monkeypatch.setattr(tar_fit, "gpu_preflight", stop)
    args = SimpleNamespace(
        preregistration=ROOT / "preregistration.yaml",
        dataset="iris",
        seed=1729,
        device="cuda:0",
        smoke=False,
        output_root=tmp_path / "stop",
    )
    with pytest.raises(KeyboardInterrupt):
        run_fit(args)
    assert json.loads((args.output_root / "result.json").read_text())["outcome"] == "interrupted"
    assert (args.output_root / "checksums.json").exists()


def test_runner_resamples_and_replays_evaluation_with_one_optimizer(tmp_path, monkeypatch):
    from tabu_lab.models.tar import TabUTARModel, TARTrainer

    seen, trainers = [], []
    forward, step = TabUTARModel.forward, TARTrainer.train_step

    def observed_forward(self, ep):
        out = forward(self, ep)
        seen.append(ep.codebook_seed)
        return out

    def observed_step(self, items):
        result = step(self, items)
        trainers.append((id(self), id(self.optimizer), self.step))
        return result

    monkeypatch.setattr(TabUTARModel, "forward", observed_forward)
    monkeypatch.setattr(TARTrainer, "train_step", observed_step)
    args = SimpleNamespace(
        preregistration=ROOT / "preregistration.yaml",
        dataset="iris",
        seed=1729,
        device="cpu",
        smoke=True,
        output_root=tmp_path / "smoke",
    )
    result = run_fit(args)
    assert result["outcome"] == "smoke_completed" and result["updates"] == 2
    assert trainers[0][:2] == trainers[1][:2] and [x[2] for x in trainers] == [1, 2]
    assert seen[:3] == seen[-3:] and len(set(seen)) == 5
    curve = [
        json.loads(line) for line in (args.output_root / "curve.jsonl").read_text().splitlines()
    ]
    first, second = (x["episode"] for x in curve)
    assert first["codebook_seed"] != second["codebook_seed"]
    assert first["codebook_sha256"] != second["codebook_sha256"]
    assert first["context_row_ids"] != second["context_row_ids"]
    held = set(result["splits"]["test"])
    bank = json.loads((args.output_root / "evaluation-episodes.json").read_text())
    assert len(bank["holdout"]) == 1
    test_episode = bank["holdout"][0]
    assert set(test_episode["context_row_ids"]) == set(result["training_pool_row_ids"])
    assert set(test_episode["query_row_ids"]) == held
    assert result["test_context_rows"] == 120 and result["test_query_rows"] == 30
    assert result["evaluation_mode"] == "all_train_context_joint_test"
    assert result["data_coverage"]["unused_rows"] == 0
    assert result["data_coverage"]["training_masked_labels"] == 40
    for x in curve:
        e = x["episode"]
        ids = e["context_row_ids"] + e["query_row_ids"]
        assert held.isdisjoint(ids)
        assert len(ids) == 120 and set(ids) == set(result["training_pool_row_ids"])
    checks = json.loads((args.output_root / "checksums.json").read_text())
    assert all(sha(args.output_root / name) == digest for name, digest in checks.items())


def test_tar_fit_emits_observer_metrics_without_episode_contents(tmp_path, monkeypatch):
    from tabu_lab import observers

    class Capture(observers.NullObserver):
        run_url = "https://wandb.ai/test/tabu-lab/runs/example"

        def __init__(self):
            self.steps, self.summaries, self.closed = [], [], False

        def log_step(self, record):
            self.steps.append(record)

        def log_summary(self, summary):
            self.summaries.append(summary)

        def close(self):
            self.closed = True

    observer = Capture()
    monkeypatch.setattr(observers, "get_observer", lambda **kw: observer)
    args = SimpleNamespace(
        preregistration=ROOT / "preregistration.yaml",
        dataset="iris",
        seed=1729,
        device="cpu",
        smoke=True,
        output_root=tmp_path / "observed",
    )
    result = run_fit(args)
    assert observer.closed and len(observer.steps) == 2
    assert [r["unique_codebooks"] for r in observer.steps] == [1, 2]
    assert all(r["learning_rate"] > 0 for r in observer.steps)
    assert observer.summaries[-1]["final_holdout"] == result["final_holdout"]
    assert all("context_row_ids" not in r and "codebook_seed" not in r for r in observer.steps)
    assert "wandb" not in json.dumps(result)
    assert (tmp_path / "observations/observed.json").exists()


@pytest.mark.parametrize("defect", ["missing", "duplicate", "overlap", "truncated", "out_of_range"])
def test_full_data_coverage_rejects_partial_or_invalid_splits(defect):
    data = deepcopy(json.loads((ROOT / "data/iris.json").read_text()))
    if defect == "missing":
        data["splits"]["train"].pop()
    elif defect == "duplicate":
        data["splits"]["train"].append(data["splits"]["train"][0])
    elif defect == "overlap":
        data["splits"]["test"][0] = data["splits"]["train"][0]
    elif defect == "truncated":
        data["values"].pop()
    else:
        data["splits"]["test"][0] = 150
    with pytest.raises(ValueError):
        validate_full_dataset(data, 150)


def test_batch_deadline_reserves_time_for_final_evaluation(tmp_path, monkeypatch):
    monkeypatch.delenv("TABU_TAR_FIT_DEADLINE_UNIX", raising=False)
    (tmp_path / "deadline.deadline.json").write_text(json.dumps(dict(deadline_unix=1)))
    result = run_fit(
        SimpleNamespace(
            preregistration=ROOT / "preregistration.yaml",
            dataset="iris",
            seed=1729,
            device="cpu",
            smoke=True,
            output_root=tmp_path / "deadline",
        )
    )
    assert result["updates"] == 0 and result["stop_reason"] == "time_budget"
    assert result["final_fit"] == result["initial_fit"]
    assert "final_holdout" in result
    assert (tmp_path / "deadline/checkpoint/manifest.json").exists()


def test_batch_launcher_interrupts_stalled_lane(tmp_path):
    import shutil
    import subprocess
    import sys

    launcher = tmp_path / "run_batch.py"
    shutil.copy2(ROOT / "run_batch.py", launcher)
    spec = json.loads((ROOT / "preregistration.yaml").read_text())
    spec.update(batch_wall_seconds=30, lane_wall_seconds=0.1, finalization_reserve_seconds=0.01)
    (tmp_path / "preregistration.yaml").write_text(json.dumps(spec))
    output = tmp_path / "batch"
    result = subprocess.run(
        [
            sys.executable,
            str(launcher),
            "--output-root",
            str(output),
            "--command",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
        timeout=5,
        check=False,
    )
    state = json.loads((output / "batch-status.json").read_text())
    assert result.returncode == 3 and state["status"] == "blocked"
    assert len(state["lanes"]) == 1 and state["lanes"][0]["timed_out"]
    assert state["elapsed_seconds"] < 5


@pytest.mark.parametrize("alarm", [False, True])
def test_periodic_fit_replay_and_gradient_alarm(tmp_path, alarm):
    import shutil

    prereg = tmp_path / "preregistration.yaml"
    spec = json.loads((ROOT / "preregistration.yaml").read_text())
    spec.update(
        periodic_fit_every=1,
        monitor_terminal_weights=True,
        gradient_alarm_max=0 if alarm else 1e30,
        zero_gradient_patience=3,
    )
    prereg.write_text(json.dumps(spec))
    shutil.copytree(ROOT / "data", tmp_path / "data")
    result = run_fit(
        SimpleNamespace(
            preregistration=prereg,
            dataset="iris",
            seed=1729,
            device="cpu",
            smoke=True,
            output_root=tmp_path / "periodic",
        )
    )
    assert result["updates"] == (1 if alarm else 2)
    assert len(result["periodic_evaluations"]) == result["updates"]
    assert result["periodic_evaluations"][-1]["fit"] == result["final_fit"]
    assert 0 <= result["final_fit"]["support_max_weight_mean"] <= 1
    assert result["final_fit"]["support_entropy_mean"] >= 0
    if alarm:
        assert result["stop_reason"] == "gradient_alarm"
    assert (tmp_path / "periodic/checkpoint/manifest.json").exists()


def test_lr_diagnostic_launcher_runs_four_paired_lanes(tmp_path):
    import shutil
    import subprocess
    import sys

    protocol = ROOT.parent / "tar-lr-diagnostic"
    for name in ("run_batch.py", "lr1e-4.yaml", "lr3e-5.yaml"):
        shutil.copy2(protocol / name, tmp_path / name)
    fake = tmp_path / "fake.py"
    fake.write_text(
        "import sys,json; from pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--output-root')+1]); p.mkdir()\n"
        "assert (p.parent/(p.name+'.deadline.json')).exists()\n"
        "(p/'result.json').write_text(json.dumps(dict(outcome='fit_gate_not_met')))\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(tmp_path / "run_batch.py"),
            "--output-root",
            str(tmp_path / "batch"),
            "--command",
            sys.executable,
            str(fake),
        ],
        check=True,
        timeout=10,
    )
    result = json.loads((tmp_path / "batch/batch-status.json").read_text())
    assert result["status"] == "complete" and len(result["lanes"]) == 4
    assert [(x["dataset"], x["variant"]) for x in result["lanes"]] == [
        (data, rate) for data in ("iris", "diabetes") for rate in ("lr1e-4", "lr3e-5")
    ]


def test_git_source_state_distinguishes_clean_dirty_and_unavailable(tmp_path):
    import subprocess

    from tabu_lab.tar_fit import git_source_state

    assert git_source_state(tmp_path) == {"state": "unavailable", "commit": None}
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert git_source_state(tmp_path) == {"state": "unavailable", "commit": None}
    source = tmp_path / "source.py"
    source.write_text("original")
    subprocess.run(["git", "add", "source.py"], cwd=tmp_path, check=True)
    subprocess.run([
        "git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "fixture"
    ], cwd=tmp_path, check=True)
    clean = git_source_state(tmp_path)
    assert clean["state"] == "clean" and len(clean["commit"]) == 40
    source.write_text("changed")
    assert git_source_state(tmp_path) == dict(clean, state="dirty")
    source.write_text("original")
    (tmp_path / "new.py").write_text("new untracked source")
    assert git_source_state(tmp_path) == dict(clean, state="dirty")
    assert set(clean) == {"state", "commit"}


def test_training_only_covers_every_training_row_without_test_evaluation(tmp_path):
    import shutil

    spec = json.loads((ROOT / 'preregistration.yaml').read_text())
    spec['evaluation_mode'] = 'training_masks_only'
    prep = tmp_path / 'plan'
    prep.mkdir()
    shutil.copytree(ROOT / 'data', prep / 'data')
    prereg = prep / 'preregistration.yaml'
    prereg.write_text(json.dumps(spec))
    out = tmp_path / 'out'
    result = run_fit(SimpleNamespace(preregistration=prereg, dataset='iris', seed=1729,
                                    device='cpu', smoke=True, output_root=out))
    assert result['initial_holdout'] is None and result['final_holdout'] is None
    assert result['test_query_rows'] == 0 and result['evaluation_bank_size']['holdout'] == 0
    bank = json.loads((out / 'evaluation-episodes.json').read_text())
    assert bank['holdout'] == []
    covered = set()
    train = set(result['splits']['train'])
    for ep in bank['fit']:
        c, q = set(ep['context_row_ids']), set(ep['query_row_ids'])
        assert not c & q and c | q == train
        covered.update(q)
    assert covered == train
