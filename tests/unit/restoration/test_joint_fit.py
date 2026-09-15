import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

from tabu_lab.models.restoration import ColumnSchema, RestorationConfig
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.encoding import EncoderConfig
from tabu_lab.restoration_joint_fit import (
    TablePlan,
    _episode,
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


def _write_fixture(root: Path):
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
    data_path = data_dir / "toy.json"
    data_path.write_text(json.dumps(data, separators=(",", ":")))
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    corpus_spec = {
        "schema": "tabu.tar.joint-training-fit.1",
        "status": "local_unissued",
        "datasets": {"toy": data_sha},
        "expected_rows": {"toy": 8},
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
        "protocol": "old120_all_train_rows_mixed_types_all_observed_targets_v1",
        "corpus_preregistration_sha256": hashlib.sha256(
            corpus_spec_path.read_bytes()
        ).hexdigest(),
        "table_count": 1,
        "query_count": 2,
        "evaluation_masks": 1,
        "seeds": {"model": 7, "order": 8, "masks": 9, "codes": 10},
        "max_rounds": 2,
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
