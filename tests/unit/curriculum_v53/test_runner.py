"""Tiny end-to-end campaigns exercise continuation and evidence boundaries."""

import copy
import json
import runpy
from pathlib import Path

import pytest
import torch

from tabu_lab.curriculum_v53.artifacts import load_checkpoint, sha256
from tabu_lab.curriculum_v53.preflight import preflight
from tabu_lab.curriculum_v53.protocol import load_plan
from tabu_lab.curriculum_v53.runner import evaluate_checkpoint, run


def _exact(actual, expected):
    """Compare optimizer trees including every tensor and parameter-group setting."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected, strict=True):
            _exact(value, reference)
    else:
        assert actual == expected


def _write_manifest(path, spec):
    path.write_text(json.dumps(spec), encoding="utf-8")
    return load_plan(path)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("this integration campaign requires native torch.optim.Muon")
    root = tmp_path_factory.mktemp("v53-runner")
    example = Path(__file__).resolve().parents[3] / "examples" / "curriculum_v53_fixture.py"
    create_fixture = runpy.run_path(str(example))["create_fixture"]
    manifest = create_fixture(root / "data")
    spec = json.loads(manifest.read_text())
    for probe in spec["probes"]:
        if probe["partition"] == "train":
            probe["masks"] = 1
    for stage, updates in zip(spec["stages"], (2, 1, 2), strict=True):
        stage.update(max_updates=updates, evaluate_every=updates, checkpoint_every=1,
                     max_seconds=120.)
        stage["probes"] = ["old_fit" if stage["name"] == "fit" else "new_fit"]
    spec["stages"][0]["evaluate_every"] = 1
    spec["stages"][-1]["optimizer"] = "muon"
    plan = _write_manifest(manifest, spec)
    output = root / "uninterrupted"
    result = run(plan, output)
    assert result["outcome"] == "completed", result
    checkpoint = output / "checkpoint-progress.pt"
    payload, digest = load_checkpoint(checkpoint)
    return {"root": root, "manifest": manifest, "spec": spec, "plan": plan,
            "output": output, "result": result, "checkpoint": checkpoint,
            "payload": payload, "digest": digest}


@pytest.mark.parametrize("stop_after,stage_index,cursor,optimizer_kind", [
    (1, 0, 1, "adamw"),  # Pending periodic evaluation.
    (2, 0, 2, "adamw"),  # Pending fit final evaluation and stage transition.
    (3, 1, 1, "adamw"),  # Pending continual final evaluation and Muon initialization.
    (4, 2, 1, "muon"),  # Resume inside the mixed-optimizer stage.
])
def test_split_resume_is_exact_across_adamw_to_muon(
    campaign, stop_after, stage_index, cursor, optimizer_kind,
):
    plan, root = campaign["plan"], campaign["root"]
    partial_root, resumed_root = root / f"partial-{stop_after}", root / f"resumed-{stop_after}"
    partial = run(plan, partial_root, max_updates_this_invocation=stop_after)
    assert partial["outcome"] == "stopped", partial
    assert partial["stage_index"] == stage_index
    assert partial["cursor"] == cursor
    assert partial["update"] == stop_after
    parent = partial_root / "checkpoint-progress.pt"
    parent_payload, parent_digest = load_checkpoint(parent)
    assert parent_payload["optimizer_kind"] == optimizer_kind
    assert parent_payload["state"]["evaluated_cursor"] < cursor
    resumed = run(plan, resumed_root, resume=parent)
    assert resumed["outcome"] == "completed", resumed
    payload, _ = load_checkpoint(resumed_root / "checkpoint-progress.pt")
    _exact(payload["model"], campaign["payload"]["model"])
    _exact(payload["optimizer"], campaign["payload"]["optimizer"])
    _exact(payload["rng"], campaign["payload"]["rng"])
    for key in ("stage_index", "cursor", "update", "phase", "optimizer_kind",
                "evaluated_cursor", "exposure", "stage_verdicts"):
        assert payload["state"][key] == campaign["payload"]["state"][key], key
    assert resumed["update"] == resumed["durable_update"] == 5
    assert resumed["exposure"] == campaign["result"]["exposure"]
    assert resumed["lineage"][-1]["mode"] == "resume"
    assert resumed["lineage"][-1]["checkpoint_sha256"] == parent_digest


def test_frozen_final_test_preserves_checkpoint_and_existing_outputs(campaign):
    before = sha256(campaign["checkpoint"])
    output = campaign["root"] / "frozen-final-test"
    result = evaluate_checkpoint(campaign["plan"], campaign["checkpoint"], output,
                                 probes=["final_test"])
    assert result["outcome"] == "completed", result
    assert set(result["probes"]) == {"final_test"}
    report = result["probes"]["final_test"]
    assert report["partition"] == "test"
    assert report["purpose"] == "final_test"
    assert report["tables"] == 3
    assert all(table["query_rows"] == [10, 11] for table in report["by_table"])
    assert result["checkpoint_update"] == 5
    assert sha256(campaign["checkpoint"]) == before
    assert result["checkpoint_sha256"] == before
    with pytest.raises(FileExistsError):
        run(campaign["plan"], campaign["output"])
    with pytest.raises(FileExistsError):
        evaluate_checkpoint(campaign["plan"], campaign["checkpoint"], output)
    assert sha256(campaign["checkpoint"]) == before


def test_strict_resume_rejects_drift_but_weights_only_records_new_lineage(campaign):
    spec = copy.deepcopy(campaign["spec"])
    spec["experiment_id"] = "new-bounded-recipe"
    spec["seeds"]["model"] += 1
    spec["stages"] = spec["stages"][:1]
    spec["stages"][0]["max_updates"] = 1
    plan = _write_manifest(campaign["manifest"].with_name("new-manifest.json"), spec)
    rejected = run(plan, campaign["root"] / "rejected-drift", resume=campaign["checkpoint"])
    assert rejected["outcome"] == "failed"
    assert rejected["error_type"] == "ValueError"
    assert "identity drift" in rejected["error"]
    assert rejected["checkpoint"] is None
    initialized = run(plan, campaign["root"] / "weights-only",
                      initialize_from=campaign["checkpoint"], max_updates_this_invocation=0)
    assert initialized["outcome"] == "stopped", initialized
    assert initialized["update"] == initialized["cursor"] == initialized["stage_index"] == 0
    assert initialized["phase"] == "entry"
    assert initialized["exposure"] == {}
    payload, _ = load_checkpoint(campaign["root"] / "weights-only" / "checkpoint-progress.pt")
    _exact(payload["model"], campaign["payload"]["model"])
    assert payload["optimizer_kind"] == "adamw"
    assert payload["optimizer"]["state"] == {}
    expected_rng = torch.Generator().manual_seed(plan.spec["seeds"]["model"]).get_state()
    assert torch.equal(payload["rng"]["torch"], expected_rng)
    parent = initialized["lineage"][-1]
    assert parent["mode"] == "weights_only_initialization"
    assert parent["parent_update"] == 5
    assert parent["checkpoint_sha256"] == campaign["digest"]
    assert parent["parent_identity"] == campaign["plan"].identity


def test_cpu_preflight_parity_artifacts_cannot_seed_training(campaign):
    output = campaign["root"] / "qualification"
    qualified = preflight(campaign["plan"], output, device="cpu", max_seconds=120.)
    assert qualified["outcome"] == "passed", qualified
    assert qualified["runtime"]["device"] == "cpu"
    assert len(qualified["probes"]) == 3
    assert all(probe["next_update_exact"] for probe in qualified["probes"])
    assert {probe["optimizer"] for probe in qualified["probes"]} == {"adamw", "muon"}
    checkpoint = output / "probe-000.pt"
    payload, _ = load_checkpoint(checkpoint)
    assert payload["purpose"] == "preflight"
    for option in ("resume", "initialize_from"):
        rejected = run(campaign["plan"], campaign["root"] / f"reject-preflight-{option}",
                       **{option: checkpoint}, max_updates_this_invocation=0)
        assert rejected["outcome"] == "failed", rejected
        assert rejected["error_type"] == "ValueError"
        assert rejected["checkpoint"] is None
    evaluated = evaluate_checkpoint(campaign["plan"], checkpoint,
                                    campaign["root"] / "reject-preflight-evaluation",
                                    probes=["final_test"])
    assert evaluated["outcome"] == "failed"


def test_failed_validation_gate_does_not_enter_next_stage(campaign):
    spec = copy.deepcopy(campaign["spec"])
    spec["experiment_id"] = "forced-negative-validation-gate"
    spec["stages"][0].update(max_updates=1, evaluate_every=1, probes=["validation"])
    spec["stages"][0]["gate"] = {
        "probe": "validation", "metric": "query_numeric_mse", "mode": "min", "threshold": -1.,
    }
    plan = _write_manifest(campaign["manifest"].with_name("gate-manifest.json"), spec)
    output = campaign["root"] / "gate-failed"
    result = run(plan, output)
    assert result["outcome"] == "gate_failed", result
    assert result["stage_index"] == 0
    assert result["update"] == result["cursor"] == 1
    assert result["stage_seconds"][1:] == [0., 0.]
    assert result["stage_verdicts"] == [{"stage": "fit", "verdict": "failed", "update": 1}]
    rows = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["stage"] == "fit"
    assert not (output / "stage-000-final.pt").exists()
