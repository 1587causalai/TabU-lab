import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tabu_lab.models.restoration import ColumnSchema, RestorationConfig, RestorationModel
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.encoding import EncoderConfig
from tabu_lab.restoration_joint_fit import (
    TablePlan,
    _episode,
    _fixed_bank,
    _metrics,
    _PreparedCache,
    _query_mask,
    prepare_plan,
    run_joint_fit,
)


def _toy_table() -> TablePlan:
    return TablePlan(
        "toy",
        tuple(range(6)),
        (
            torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            torch.tensor([0, 1, 1, 0, 1, 0]),
            torch.tensor([0, 1, 2, 0, 1, 2]),
        ),
        (
            # The names and declared domains mirror the corpus adapter's output.
            ColumnSchema("toy/numeric", "numeric"),
            ColumnSchema("toy/nominal", "nominal", 2),
            ColumnSchema("toy/ordinal", "ordinal", 3),
        ),
        {"numeric": 1, "nominal": 1, "ordinal": 1},
        6,
        2,
    )


def test_query_mask_keeps_every_discrete_class_visible_and_is_reproducible():
    table = _toy_table()
    first, info = _query_mask(table, 2, 17)
    second, same_info = _query_mask(table, 2, 17)
    assert torch.equal(first, second)
    assert info == same_info
    assert first.sum(0).tolist() == [2, 2, 2]
    for column in (1, 2):
        assert set(table.values[column][~first[:, column]].tolist()) == set(
            table.values[column].tolist()
        )


def test_mixed_episode_has_all_observed_targets_and_finite_update():
    table = _toy_table()
    query, _ = _query_mask(table, 2, 19)
    episode = _episode(table, query, 23, "cpu")
    config = RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    )
    from tabu_lab.models.restoration import RestorationModel, score_episode

    model = RestorationModel(config).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    score = score_episode(model, *episode)
    score.loss.backward()
    optimizer.step()
    assert len(score.per_target) == 18
    assert bool(torch.isfinite(score.loss))
    assert score.by_state["query"]["count"] == 6


def test_prepared_cache_replaces_entries_when_mask_identity_changes():
    table = _toy_table()
    q1, _ = _query_mask(table, 2, 29)
    q2, _ = _query_mask(table, 2, 31)
    model = RestorationModel(RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    )).double()
    cache = _PreparedCache(model, capacity=1)
    first = cache.get((table.name, 0), _episode(table, q1, 37, "cpu"))
    assert cache.get((table.name, 0), _episode(table, q1, 37, "cpu")) is first
    second = cache.get((table.name, 1), _episode(table, q2, 41, "cpu"))
    assert second is not first
    assert len(cache.entries) == 1
    assert cache.get((table.name, 0), _episode(table, q1, 37, "cpu")) is not first


def _write_fixture(root: Path, *, table_count=1, max_rounds=2):
    corpus = root / "corpus"
    data_dir = corpus / "data"
    data_dir.mkdir(parents=True)
    data = {
        "schema": "tabu.tar.typed-fit-table.1",
        "dataset": "toy",
        "values": [
            [0.0, 0, 0], [1.0, 1, 1], [2.0, 1, 2], [3.0, 0, 0],
            [4.0, 1, 1], [5.0, 0, 2], [6.0, 1, 0], [7.0, 0, 1],
        ],
        "features": [
            {"kind": "numeric", "domain": []},
            {"kind": "nominal", "domain": ["0", "1"]},
            {"kind": "ordinal", "domain": ["0", "1", "2"]},
        ],
        "target_kind": "ordinal",
        "domain": ["0", "1", "2"],
        "splits": {"train": [0, 1, 2, 3, 4, 5], "test": [6, 7]},
    }
    datasets = {}
    for index in range(table_count):
        name = "toy" if table_count == 1 else f"toy_{index}"
        data["dataset"] = name
        data_path = data_dir / f"{name}.json"
        data_path.write_text(json.dumps(data, separators=(",", ":")))
        datasets[name] = hashlib.sha256(data_path.read_bytes()).hexdigest()
    corpus_spec = {
        "schema": "tabu.tar.joint-training-fit.1",
        "status": "local_unissued",
        "datasets": datasets,
        "expected_rows": {name: 8 for name in datasets},
    }
    corpus_spec_path = corpus / "preregistration.yaml"
    corpus_spec_path.write_text(json.dumps(corpus_spec, sort_keys=True))
    (corpus / "manifest.json").write_text("{}\n")
    config = RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    ).as_dict()
    prereg = {
        "schema": "tabu.restoration.joint-fit.v1",
        "status": "local_unissued",
        "protocol": "old120_all_train_rows_mixed_types_all_observed_targets_v2",
        "corpus_preregistration_sha256": hashlib.sha256(
            corpus_spec_path.read_bytes()
        ).hexdigest(),
        "table_count": table_count,
        "query_count": 2,
        "evaluation_masks": 1,
        "seeds": {"model": 7, "order": 8, "masks": 9, "codes": 10},
        "max_rounds": max_rounds,
        "max_seconds": 120,
        "checkpoint_every_round": 1,
        "model": config,
        "optimizer": {
            "kind": "adamw", "learning_rate": 1e-4, "weight_decay": 0.0,
            "betas": [0.9, 0.95], "eps": 1e-8, "grad_clip": 1.0,
        },
    }
    prereg_path = root / "preregistration.yaml"
    prereg_path.write_text(json.dumps(prereg, sort_keys=True))
    return prereg_path, corpus


def test_joint_fit_plan_is_read_only_and_execution_resumes(tmp_path):
    prereg, corpus = _write_fixture(tmp_path)
    plan = prepare_plan(prereg, corpus)
    assert plan.table_count == 1
    result = run_joint_fit(SimpleNamespace(
        preregistration=prereg, corpus=corpus, output_root=tmp_path / "plan-out",
        device="cpu", execute=False,
    ))
    assert result["outcome"] == "planned"
    assert not (tmp_path / "plan-out").exists()

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "preregistration.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-q", "-s", "-m", "fixture"],
        cwd=tmp_path, check=True,
    )
    args = SimpleNamespace(
        preregistration=prereg, corpus=corpus, output_root=tmp_path / "first",
        device="cpu", execute=True, stop_after_round=1, resume_checkpoint=None,
    )
    first = run_joint_fit(args)
    assert first["outcome"] == "segment_completed"
    assert first["update"] == 1
    assert (args.output_root / "terminal.json").is_file()
    assert (args.output_root / "checkpoint-round-0001.pt").is_file()

    resumed = run_joint_fit(SimpleNamespace(
        preregistration=prereg, corpus=corpus, output_root=tmp_path / "resumed",
        device="cpu", execute=True, stop_after_round=None,
        resume_checkpoint=args.output_root / "checkpoint-round-0001.pt",
    ))
    assert resumed["outcome"] == "completed"
    assert resumed["update"] == 2
    assert (tmp_path / "resumed" / "checkpoint-round-0002.pt").is_file()


def _commit_fixture(root):
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "add", "preregistration.yaml"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-q", "-s", "-m", "fixture"],
        cwd=root, check=True,
    )


def _args(root, prereg, corpus, name, resume=None):
    return SimpleNamespace(preregistration=prereg, corpus=corpus, output_root=root / name,
                           device="cpu", execute=True, stop_after_round=None,
                           resume_checkpoint=resume)


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def test_random_class_representatives_and_singleton_counts():
    table = _toy_table()
    table = replace(
        table,
        values=(table.values[0], torch.tensor([0, 1, 1, 0, 1, 2]), table.values[2]),
        schema=(table.schema[0], ColumnSchema("toy/nominal", "nominal", 3), table.schema[2]),
    )
    masks = []
    for seed in range(30):
        mask, info = _query_mask(table, 2, seed)
        assert not mask[5, 1]  # The singleton must never become a query.
        assert info["unmaskable_discrete_classes"] == 1
        assert info["protected_discrete_cells"] == 6
        masks.append(mask)
    # First occurrence is no longer permanently protected for a repeated class.
    assert any(mask[0, 1] for mask in masks)
    assert any(mask[3, 1] for mask in masks)


def test_prepared_cache_same_key_detects_mask_and_code_changes():
    table = _toy_table()
    config = RestorationConfig(
        encoder=EncoderConfig(),
        backbone=BackboneConfig(kind="inducing", layers=1, heads=4, ff_width=32, slots=2),
    )
    cache = _PreparedCache(RestorationModel(config).double(), capacity=1)
    query, _ = _query_mask(table, 2, 31)
    first = cache.get("same", _episode(table, query, 37, "cpu"))
    assert cache.get("same", _episode(table, query, 37, "cpu")) is first
    second = cache.get("same", _episode(table, query, 41, "cpu"))
    assert second is not first
    other, _ = _query_mask(table, 2, 43)
    assert not torch.equal(query, other)
    assert cache.get("same", _episode(table, other, 41, "cpu")) is not second
    assert len(cache.entries) == 1


def test_vectorized_metrics_match_scalar_mixed_type_oracle(tmp_path):
    from tabu_lab.models.restoration import score_episode
    prereg, corpus = _write_fixture(tmp_path)
    plan = prepare_plan(prereg, corpus)
    model = RestorationModel(plan.config).double()
    measured = _metrics(plan, model, "cpu")
    expected = {state: {"errors": [], "numeric": [], "discrete": []}
                for state in ("retained", "query")}
    with torch.no_grad():
        for (inputs, request, truth), _ in _fixed_bank(plan, plan.tables[0], "cpu"):
            score = score_episode(model, inputs, request, truth)
            for column in score.output.columns:
                for local, position in enumerate(column.target_indices.tolist()):
                    row, column_id = request.targets[position].tolist()
                    state = "query" if truth.states[row, column_id] == 1 else "retained"
                    item = expected[state]
                    item["errors"].append(float(score.per_target[position]))
                    actual = truth.values[column_id][row]
                    if plan.tables[0].schema[column_id].kind == "numeric":
                        item["numeric"].append(float((column.decoded[local] - actual).square()))
                    else:
                        item["discrete"].append(int(column.decoded[local] == actual))
    for state, expected_state in expected.items():
        result = measured["by_state"][state]
        assert result["count"] == len(expected_state["errors"])
        for field, key in (("encoding_mse", "errors"), ("numeric_mse", "numeric"),
                           ("discrete_accuracy", "discrete")):
            expected_mean = sum(expected_state[key]) / len(expected_state[key])
            assert result[field] == pytest.approx(expected_mean)
    assert measured["complete"]


def test_metrics_deadline_reports_partial_table_coverage(tmp_path, monkeypatch):
    import tabu_lab.restoration_joint_fit as runner
    prereg, corpus = _write_fixture(tmp_path, table_count=2)
    plan = prepare_plan(prereg, corpus)
    clock = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def progress(_):
        clock[0] = 2.0

    result = _metrics(plan, RestorationModel(plan.config).double(), "cpu", deadline=1.0,
                      progress=progress)
    assert not result["complete"]
    assert result["completed_episodes"] == 1
    assert result["expected_episodes"] == 2
    assert result["by_table"]["toy_1"]["query"]["count"] == 0
    assert result["by_table"]["toy_1"]["query"]["encoding_mse"] is None


def test_multitable_resume_mid_second_round_matches_uninterrupted(tmp_path):
    prereg, corpus = _write_fixture(tmp_path, table_count=3, max_rounds=3)
    _commit_fixture(tmp_path)
    uninterrupted = run_joint_fit(_args(tmp_path, prereg, corpus, "full"))
    assert uninterrupted["outcome"] == "completed"

    def interrupt(event):
        if event["event"] == "update" and event["update"] == 4:
            raise KeyboardInterrupt("test interruption in second round")

    first = run_joint_fit(_args(tmp_path, prereg, corpus, "partial"), observer=interrupt)
    assert (first["outcome"], first["round"], first["cursor"], first["update"]) == (
        "interrupted", 1, 1, 4,
    )
    events = []
    resumed = run_joint_fit(_args(tmp_path, prereg, corpus, "resumed",
                                 tmp_path / "partial" / "checkpoint.pt"), observer=events.append)
    assert (resumed["outcome"], resumed["round"], resumed["cursor"], resumed["update"]) == (
        "completed", 3, 0, 9,
    )
    assert not any(event.get("stage") == "initial" for event in events)
    full_state = torch.load(tmp_path / "full" / "checkpoint.pt", weights_only=True)
    resumed_state = torch.load(tmp_path / "resumed" / "checkpoint.pt", weights_only=True)
    for key in ("model", "optimizer", "torch_cpu_rng", "evaluation"):
        _assert_tree_equal(full_state[key], resumed_state[key])
    def read_trace(name):
        rows = map(json.loads, (tmp_path / name / "updates.jsonl").read_text().splitlines())
        return [{key: row[key] for key in ("round", "update", "table", "loss", "mask", "code_seed")}
                for row in rows]
    assert read_trace("full") == read_trace("partial") + read_trace("resumed")


def test_resume_after_last_update_finishes_pending_evaluation(tmp_path):
    prereg, corpus = _write_fixture(tmp_path, table_count=2, max_rounds=2)
    spec = json.loads(prereg.read_text())
    spec["evaluate_every_rounds"] = 2
    prereg.write_text(json.dumps(spec, sort_keys=True))
    _commit_fixture(tmp_path)

    def interrupt(event):
        if event["event"] == "phase" and event.get("stage") == "round-0002":
            raise KeyboardInterrupt("test interruption before final evaluation")

    first = run_joint_fit(_args(tmp_path, prereg, corpus, "pending"), observer=interrupt)
    assert (first["outcome"], first["round"], first["cursor"], first["update"]) == (
        "interrupted", 2, 0, 4,
    )
    assert not (tmp_path / "pending" / "round-0001-metrics.json").exists()
    events = []
    result = run_joint_fit(_args(tmp_path, prereg, corpus, "finish-eval",
                                tmp_path / "pending" / "checkpoint.pt"), observer=events.append)
    assert result["outcome"] == "completed"
    assert result["update"] == 4
    assert result["final"]["complete"]
    assert not any(event["event"] == "update" for event in events)


def test_failure_keeps_last_finite_completed_boundary(tmp_path, monkeypatch):
    prereg, corpus = _write_fixture(tmp_path, table_count=2, max_rounds=2)
    _commit_fixture(tmp_path)
    original = torch.optim.AdamW.step
    calls = [0]

    def failing_step(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            with torch.no_grad():
                optimizer.param_groups[0]["params"][0].fill_(float("nan"))
            raise FloatingPointError("simulated failure after partial optimizer mutation")
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", failing_step)
    result = run_joint_fit(_args(tmp_path, prereg, corpus, "failed"))
    assert result["outcome"] == "failed"
    assert (result["round"], result["cursor"], result["update"]) == (0, 1, 1)
    saved = torch.load(tmp_path / "failed" / "checkpoint.pt", weights_only=True)
    assert all(bool(torch.isfinite(value).all()) for value in saved["model"].values())


def test_wall_limit_reserves_final_eval_and_keeps_cumulative_budget(tmp_path, monkeypatch):
    import tabu_lab.restoration_joint_fit as runner
    prereg, corpus = _write_fixture(tmp_path, table_count=2, max_rounds=2)
    _commit_fixture(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock[0])

    def advance(event):
        if event["event"] == "update":
            clock[0] = 109.0

    result = run_joint_fit(_args(tmp_path, prereg, corpus, "limited"), observer=advance)
    assert result["outcome"] == "wall_limit"
    assert result["update"] == 1
    assert result["final"]["complete"]
    assert result["final"]["at_update"] == 1
    assert result["elapsed_seconds"] == 109.0
    resumed = run_joint_fit(_args(tmp_path, prereg, corpus, "limited-resume",
                                 tmp_path / "limited" / "checkpoint.pt"))
    assert resumed["outcome"] == "wall_limit"
    assert resumed["update"] == 1
    assert resumed["elapsed_seconds"] == 109.0
