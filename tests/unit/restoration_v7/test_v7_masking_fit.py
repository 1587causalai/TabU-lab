"""Configuration, actual mask sampling, data isolation and strict continuation."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest
import torch
import yaml

from tabu_lab.models.restoration_v7 import MaskingSpec, load_typed_table, sample_task
from tabu_lab.models.restoration_v7.fit import run_fit


@pytest.fixture(autouse=True)
def isolated_runtime(request):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    parameters = getattr(getattr(request.node, "callspec", None), "params", {})
    if parameters.get("device") == "mps":
        torch.use_deterministic_algorithms(False)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def fixture(tmp_path, mode="mixed"):
    table_path = tmp_path / "toy.json"
    table_path.write_text(
        json.dumps(
            dict(
                schema="tabu.tar.typed-fit-table.1",
                dataset="toy",
                column_names=["x", "z", "y"],
                features=[{"kind": "numeric"}] * 3,
                values=[[i / 10, (i % 5) / 5, (i / 10) ** 2] for i in range(20)],
                splits=dict(train=list(range(15)), test=list(range(15, 20))),
            )
        )
    )
    cfg = dict(
        schema="tabu.restoration.v7-fit.v1",
        tables=[dict(path="toy.json")],
        steps=4,
        seed=17,
        device="cpu",
        dtype="float64",
        window_rows=10,
        query_rows=3,
        output_dir="run",
        masking=dict(mode=mode, columns=[0, 1, "target"]),
        model=dict(
            backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
            rounds=2,
            unit_layers=1,
            query_init="donor",
            query_source=True,
            coupling_blocks=1,
            coupling_hidden=[16],
            center_chunk_size=4,
        ),
    )
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, load_typed_table(table_path)


def test_masking_counts_endpoints_determinism_and_no_test_exposure(tmp_path):
    _, _, table = fixture(tmp_path)
    spec = MaskingSpec(columns=(0, 1, "target"))
    modes = set()
    for step in range(20):
        task, receipt = sample_task(table, spec, seed=42, step=step, window_rows=10, query_rows=3)
        _, repeated = sample_task(table, spec, seed=42, step=step, window_rows=10, query_rows=3)
        assert receipt == repeated
        modes.add(receipt["mode"])
        assert set(receipt["rows"]) <= set(table.train_rows)
        assert not (set(receipt["rows"]) & set(table.test_rows))
        assert receipt["query_cells"] == int(task.inputs.query.sum())
        assert int(task.inputs.query.any(0).sum()) == (1 if receipt["mode"] == "single" else 3)
    assert modes == {"single", "joint"}
    for p, mode in ((0, "single"), (1, "joint")):
        _, receipt = sample_task(
            table, replace(spec, joint_probability=p), seed=42, step=0, window_rows=10, query_rows=3
        )
        assert receipt["mode"] == mode
    with pytest.raises(ValueError, match="distinct"):
        replace(spec, columns=(2, "target")).eligible(table)
    with pytest.raises(ValueError, match="not enough"):
        replace(spec, columns=("target",)).eligible(table)


@pytest.mark.parametrize("mode", ["single", "joint", "mixed"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_configured_training_resume_and_evaluation(tmp_path, mode, device):
    path, cfg, table = fixture(tmp_path, mode)
    cfg.update(device=device, dtype="float32" if device == "mps" else "float64")
    path.write_text(yaml.safe_dump(cfg))
    plan = run_fit(path)
    assert plan["status"] == "validated-not-run"
    assert not (tmp_path / "run").exists()
    complete = run_fit(path, execute=True)
    evaluation = json.loads((tmp_path / "run/evaluation.json").read_text())
    assert {report["mode"] for report in evaluation} == {"target_only", "joint"}
    for report in evaluation:
        assert set(report["query_rows"]) <= set(table.test_rows)
        assert set(report["support_rows"]) <= set(table.train_rows)
    half = tmp_path / "half.yaml"
    half.write_text(yaml.safe_dump(cfg | dict(steps=2, output_dir="half")))
    prefix = run_fit(half, execute=True)
    resumed = run_fit(
        path, execute=True, resume=prefix["checkpoint"], output_dir=tmp_path / "resumed"
    )
    a = torch.load(complete["checkpoint"], weights_only=True)
    b = torch.load(resumed["checkpoint"], weights_only=True)
    assert a["sampling_totals"] == b["sampling_totals"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    if mode != "mixed":
        assert complete["sampling_totals"]["query_cells"] == (12 if mode == "single" else 36)
    drift = tmp_path / "drift.yaml"
    drift.write_text(
        yaml.safe_dump(cfg | dict(masking=asdict(MaskingSpec(mode="single", columns=(0,)))))
    )
    with pytest.raises(ValueError, match="manifest"):
        run_fit(drift, execute=True, resume=prefix["checkpoint"], output_dir=tmp_path / "bad")


def test_config_rejects_silent_typo_and_mps_fp64(tmp_path):
    path, cfg, _ = fixture(tmp_path)
    path.write_text(yaml.safe_dump(cfg | dict(masking={"mod": "mixed"})))
    with pytest.raises(ValueError, match="unknown masking"):
        run_fit(path)
    path.write_text(yaml.safe_dump(cfg | dict(device="mps")))
    with pytest.raises(ValueError, match="MPS requires"):
        run_fit(path)


def test_weights_only_parent_inherits_shape_and_starts_fresh_optimizer(tmp_path):
    path, cfg, _ = fixture(tmp_path, "single")
    cfg["steps"] = 2
    path.write_text(yaml.safe_dump(cfg))
    parent = run_fit(path, execute=True)
    child = cfg | dict(
        steps=1,
        output_dir="child",
        init_checkpoint=parent["checkpoint"],
        model=dict(query_source=True, rounds=4),
        masking=dict(mode="mixed"),
    )
    path.write_text(yaml.safe_dump(child))
    plan = run_fit(path)
    assert plan["manifest"]["model"]["backbone"]["width"] == 64
    assert plan["manifest"]["model"]["rounds"] == 4
    result = run_fit(path, execute=True)
    state = torch.load(result["checkpoint"], weights_only=True)
    assert state["step"] == 1
    assert all(float(value["step"]) == 1 for value in state["optimizer"]["state"].values())
    assert state["manifest"]["initialization"]["sha256"] == parent["checkpoint_sha256"]
