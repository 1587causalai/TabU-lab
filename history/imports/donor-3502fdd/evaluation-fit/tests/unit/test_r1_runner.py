from __future__ import annotations

import json
import runpy
from pathlib import Path

from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments.r1_runner import run_r1
from tabu_lab.registry import get_model_spec


def _prepare_diabetes(tmp_path: Path):
    helpers = runpy.run_path(
        str(Path(__file__).with_name("test_eval_data_workflow.py"))
    )
    return helpers["_prepare_diabetes"](tmp_path)


def test_r1_records_three_seed_baselines_and_truth_free_predictions(tmp_path: Path) -> None:
    _, bundle_path, _ = _prepare_diabetes(tmp_path)
    output = tmp_path / "r1-receipt.json"

    receipt = run_r1(
        bundle_path,
        output=output,
        model_spec_hash=canonical_hash(get_model_spec("tabul")),
    )

    assert receipt.outcome == "failed"
    assert receipt.evidence_level == "local_unissued"
    assert {item.seed for item in receipt.baseline_metrics} == {1729, 2718, 31415}
    assert len(receipt.baseline_metrics) == 6
    assert all(item.predictions for item in receipt.baseline_metrics)
    assert all(
        "target" not in prediction
        for item in receipt.baseline_metrics
        for prediction in item.predictions
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["prepared_bundle_hash"] == receipt.prepared_bundle_hash
