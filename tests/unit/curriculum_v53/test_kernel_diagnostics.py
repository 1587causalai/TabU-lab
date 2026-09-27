"""The fixed probe measures real Unit kernel weights without changing training."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

from tabu_lab.curriculum_v53.artifacts import rng_state
from tabu_lab.curriculum_v53.evaluation import (
    _kernel_samples,
    _summarize_kernel_samples,
    evaluate_probe,
)
from tabu_lab.curriculum_v53.protocol import load_plan
from tabu_lab.models.restoration_v53 import V53Model


def test_uniform_and_one_hot_weights_have_expected_ess_and_gap():
    uniform = torch.zeros(5, 2, dtype=torch.float64)
    samples = _kernel_samples(
        uniform, torch.tensor([0, 2, 4]), torch.tensor([1, 3]), 1.0,
        chunk_size=2,
    )
    query = _summarize_kernel_samples(samples["query_to_support"])
    centers = _summarize_kernel_samples(samples["ll_centers_to_support"])
    assert query["center_count"] == 2
    assert centers["center_count"] == 5
    for path in (query, centers):
        assert path["ess"]["min"] == pytest.approx(3.0)
        assert path["relative_ess"]["min"] == pytest.approx(1.0)
        assert path["support_count"]["min"] == 3
        assert path["max_weight"]["max"] == pytest.approx(1 / 3)
        assert path["logit_gap"]["max"] == 0

    separated = torch.tensor([[0.0], [10.0], [20.0]], dtype=torch.float64)
    samples = _kernel_samples(
        separated, torch.tensor([0, 2]), torch.tensor([0, 2]), 0.01,
        chunk_size=1,
    )
    query = _summarize_kernel_samples(samples["query_to_support"])
    assert query["ess"]["min"] == pytest.approx(1.0)
    assert query["relative_ess"]["max"] == pytest.approx(0.5)
    assert query["max_weight"]["min"] == pytest.approx(1.0)
    assert query["logit_gap"]["min"] > 1000


def test_chunking_and_single_support_gap_are_unambiguous():
    units = torch.tensor([[0., 1.], [1., 0.], [2., 1.], [3., 2.], [4., 1.]],
                         dtype=torch.float64)
    supports = torch.tensor([0, 2, 4])
    # A repeated Query target remains a repeated kernel exposure.
    query_rows = torch.tensor([1, 3, 3])
    before = rng_state()
    small = _kernel_samples(units, supports, query_rows, 1.3, chunk_size=1)
    large = _kernel_samples(units, supports, query_rows, 1.3, chunk_size=32)
    assert small == large
    assert len(small["query_to_support"]["ess"]) == 3
    assert len(small["ll_centers_to_support"]["ess"]) == len(units)
    assert torch.equal(before["torch"], rng_state()["torch"])
    assert before["python"] == rng_state()["python"]

    one = _kernel_samples(units, torch.tensor([2]), query_rows, 1.3)
    assert _summarize_kernel_samples(one["query_to_support"])["logit_gap"] == {
        "count": 0, "min": None, "p05": None, "median": None,
        "p95": None, "max": None, "mean": None,
    }


def test_fixed_probe_reports_both_paths_by_table_and_column_without_rng_use(tmp_path):
    example = Path(__file__).resolve().parents[3] / "examples" / "curriculum_v53_fixture.py"
    create_fixture = runpy.run_path(str(example))["create_fixture"]
    manifest = create_fixture(tmp_path / "data")
    plan = load_plan(manifest)
    model = V53Model(plan.config).double()
    before = rng_state()
    report = evaluate_probe(model, plan, plan.spec["probes"][0], "cpu")
    after = rng_state()
    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"]
    assert all(torch.equal(a, b) for a, b in zip(before["cuda"], after["cuda"], strict=True))
    assert report["tables"] == 2
    json.dumps(report, allow_nan=False)

    for table in report["by_table"]:
        kernel = table["kernel"]
        assert kernel["mode"] == "gaussian_unit_softmax"
        assert kernel["bandwidth"] == plan.config.bandwidth
        query = kernel["query_to_support"]
        ll = kernel["ll_centers_to_support"]
        assert query["center_count"] > 0
        assert ll["center_count"] >= query["center_count"]
        assert kernel["query_ess_mean"] == query["ess"]["mean"]
        assert kernel["query_ess_min"] == query["ess"]["min"]
        assert kernel["query_max_weight_mean"] == query["max_weight"]["mean"]
        for path in (query, ll):
            assert path["center_count"] == sum(
                column["center_count"] for column in path["by_column"].values()
            )
            for column in path["by_column"].values():
                assert column["ess"]["min"] >= 1 - 1e-10
                assert column["ess"]["max"] <= column["support_count"]["max"] + 1e-10
                assert column["relative_ess"]["max"] <= 1 + 1e-10
                assert 0 < column["max_weight"]["min"] <= 1
