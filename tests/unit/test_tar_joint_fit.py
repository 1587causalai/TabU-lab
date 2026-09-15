import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from tabu_lab.models.tar import TARTrainer
from tabu_lab.tar_joint_fit import round_order, run_joint_fit


def test_joint_fit_uses_one_optimizer_and_covers_all_tables(tmp_path, monkeypatch):
    base = Path(__file__).resolve().parents[2] / "experiments/local/tar-unified-joint-fit"
    spec = json.loads((base / "preregistration.yaml").read_text())
    names = ["linear_numeric", "xor_classification"]
    spec["datasets"] = {n: spec["datasets"][n] for n in names}
    (tmp_path / "data").mkdir()
    for n in names:
        shutil.copy2(base / "data" / f"{n}.json", tmp_path / "data" / f"{n}.json")
    plan = tmp_path / "preregistration.yaml"
    plan.write_text(json.dumps(spec))
    created = []
    init = TARTrainer.__init__

    def track(self, *args, **kwargs):
        init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(TARTrainer, "__init__", track)
    out = tmp_path / "out"
    r = run_joint_fit(
        SimpleNamespace(preregistration=plan, device="cpu", smoke=True, output_root=out)
    )
    assert len(created) == 1 and created[0].step == 4
    assert r["outcome"] == "smoke_completed" and r["rounds"] == 2
    assert set(r["final"]["datasets"]) == set(names)
    curves = [json.loads(s) for s in (out / "curve.jsonl").read_text().splitlines()]
    for i in (1, 2):
        rows = [s for s in curves if s["round"] == i]
        assert {s["dataset"] for s in rows} == set(names)
        assert all(s["episode"]["episode_id"] == i - 1 for s in rows)
    assert [s["step"] for s in curves] == [1, 2, 3, 4]
    bank = json.loads((out / "evaluation-episodes.json").read_text())
    for episodes in bank.values():
        train = set(episodes[0]["context_row_ids"]) | set(episodes[0]["query_row_ids"])
        assert {r for e in episodes for r in e["query_row_ids"]} == train
    assert (out / "checkpoint/weights.safetensors").exists()


def test_round_order_reproducible_and_complete():
    names = list("abcdefgh")
    assert round_order(names, 1, 0) == round_order(names, 1, 0)
    assert set(round_order(names, 1, 0)) == set(names)
    assert round_order(names, 1, 0) != round_order(names, 1, 1)


def test_fit_diagnostics_expose_unsupported_and_colliding_query_labels():
    import torch

    from tabu_lab.models.tar import TARFeature
    from tabu_lab.models.tar.episodes import supervised_episode
    from tabu_lab.tar_joint_fit import fit_bank_diagnostics

    # Both queries have identical predictors, but contradictory labels; class 2 is unseen.
    features = (TARFeature(), TARFeature("nominal", ("a", "b", "c"), 1))
    episode, truth = supervised_episode(
        torch.tensor([[1.0, 0.0], [2.0, 1.0]]),
        torch.tensor([[3.0, 1.0], [3.0, 2.0]]),
        features,
        codebook_seed=1,
    )
    table = dict(kind="nominal", features=features, bank=[(episode, truth, {})])
    diagnostics = fit_bank_diagnostics(table, 1e-6)
    assert diagnostics["unsupported_queries"] == 1
    assert diagnostics["query_label_seen_rate"] == 0.5
    assert diagnostics["duplicate_and_support_accuracy_ceiling"] == 0.5
    assert diagnostics["unsupported_nll_floor"] > 6.9


def test_joint_resume_matches_uninterrupted_updates(tmp_path, monkeypatch):
    import pytest

    from tabu_lab import tar_joint_fit
    from tabu_lab.tar_joint_fit import sha

    base = Path(__file__).resolve().parents[2] / "experiments/local/tar-unified-joint-fit"
    spec = json.loads((base / "preregistration.yaml").read_text())
    names = ["linear_numeric", "xor_classification"]
    spec["datasets"] = {n: spec["datasets"][n] for n in names}
    spec["save_periodic_checkpoints"] = True
    (tmp_path / "data").mkdir()
    for n in names:
        shutil.copy2(base / "data" / f"{n}.json", tmp_path / "data" / f"{n}.json")
    plan = tmp_path / "preregistration.yaml"
    plan.write_text(json.dumps(spec))
    args = dict(preregistration=plan, device="cpu", smoke=True)
    full = run_joint_fit(SimpleNamespace(**args, output_root=tmp_path / "full"))
    first = run_joint_fit(
        SimpleNamespace(**args, output_root=tmp_path / "first", stop_after_round=1)
    )
    assert first["rounds"] == 1
    resumed = run_joint_fit(
        SimpleNamespace(
            **args,
            output_root=tmp_path / "resumed",
            resume_checkpoint=tmp_path / "first/checkpoint",
        )
    )
    assert resumed["start_round"] == 1 and resumed["updates"] == 4
    assert resumed["final"] == full["final"]
    assert sha(tmp_path / "full/checkpoint/weights.safetensors") == sha(
        tmp_path / "resumed/checkpoint/weights.safetensors"
    )
    monkeypatch.setattr(
        tar_joint_fit,
        "sha",
        lambda path: "changed-helper" if Path(path).name == "tar_data.py" else sha(path),
    )
    with pytest.raises(ValueError, match="recipe or source mismatch"):
        run_joint_fit(
            SimpleNamespace(
                **args,
                output_root=tmp_path / "rejected",
                resume_checkpoint=tmp_path / "first/checkpoint",
            )
        )
