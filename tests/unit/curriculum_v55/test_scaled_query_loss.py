"""The opt-in loss keeps the old square scale observable and replay-stable."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import V54_SCHEMA, V55_SCHEMA, load_v54_plan, load_v55_plan
from tabu_lab.curriculum_v53.runner import _objective_score, train_step
from tabu_lab.models.restoration_v53.training import V53LossConfig, V53Score
from tabu_lab.restoration_optimizers import adamw


def _plan(tmp_path, *, objective=None):
    table = {
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i + 1)] for i in range(12)],
        "splits": {"train": list(range(10)), "validation": [10], "test": [11]},
    }
    data = json.dumps(table).encode()
    (tmp_path / "table.json").write_bytes(data)
    stage = {
        "name": "fit", "question": "Does the scaled objective preserve raw loss?",
        "max_updates": 2, "max_seconds": 120,
        "sampling": [{"cohort": "old", "episodes": 1}],
        "recipe": {"synthetic": {"fraction": 0.25}},
        "optimizer": "adamw", "evaluate_every": 2, "checkpoint_every": 1,
        "probes": ["fit"],
    }
    if objective is not None:
        stage["objective"] = objective
    spec = {
        "schema": V55_SCHEMA, "experiment_id": "scaled-query-loss-test",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"size": "small", "backbone": {"layers": 1, "heads": 4,
                                                "ff_width": 128, "slots": 4}},
        "tables": [{"id": "tiny", "kind": "synthetic", "cohort": "old",
                    "path": "table.json", "sha256": hashlib.sha256(data).hexdigest()}],
        "probes": [{"name": "fit", "cohorts": ["old"], "partition": "train",
                    "purpose": "fit", "recipe": {"fraction": 0.25}, "masks": 1}],
        "stages": [stage],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(spec))
    return path, spec


def test_scaled_query_loss_has_analytic_type_weights_and_no_retained_gradient():
    q = torch.tensor([0., 3., 8., 1., 15.], dtype=torch.float64, requires_grad=True)
    prepared = SimpleNamespace(
        numeric=torch.tensor([True, True, False, False, True]),
        states=torch.tensor([1, 1, 1, 1, 0]),
    )
    config = V53LossConfig(discrete_weight=3.)
    score = V53Score(q[:2].mean() + 3 * q[2:4].mean(), q, None)
    transformed = _objective_score(
        score, prepared, config, {"kind": "query_log1p_scaled", "tau": 2.},
    )
    expected = 2 * (torch.log1p(q[0] / 2) + torch.log1p(q[1] / 2)) / 2
    expected += 3 * 2 * (torch.log1p(q[2] / 2) + torch.log1p(q[3] / 2)) / 2
    torch.testing.assert_close(transformed.loss, expected, rtol=0, atol=0)
    transformed.loss.backward()
    expected_gradient = torch.tensor([0.5, 0.2, 0.3, 1., 0.], dtype=torch.float64)
    torch.testing.assert_close(q.grad, expected_gradient, rtol=1e-15, atol=1e-15)


def test_large_tau_recovers_square_loss_and_local_derivative():
    q = torch.tensor([0., 0.5, 4.], dtype=torch.float64, requires_grad=True)
    prepared = SimpleNamespace(numeric=torch.ones(3, dtype=torch.bool),
                               states=torch.ones(3, dtype=torch.long))
    raw = V53Score(q.mean(), q, None)
    transformed = _objective_score(
        raw, prepared, V53LossConfig(),
        {"kind": "query_log1p_scaled", "tau": 1e8},
    )
    torch.testing.assert_close(transformed.loss, raw.loss, rtol=1e-7, atol=0)
    transformed.loss.backward()
    torch.testing.assert_close(q.grad, torch.full_like(q, 1 / 3), rtol=1e-7, atol=0)


def test_manifest_requires_explicit_positive_scale_and_binds_it_to_identity(tmp_path):
    path, spec = _plan(tmp_path)
    default = load_v55_plan(path)
    assert "objective" not in default.spec["stages"][0]
    spec["stages"][0]["objective"] = {"kind": "query_log1p_scaled", "tau": 2}
    path.write_text(json.dumps(spec))
    scaled = load_v55_plan(path)
    assert scaled.spec["stages"][0]["objective"] == {
        "kind": "query_log1p_scaled", "tau": 2.0,
    }
    assert scaled.summary["stages"][0]["objective"]["tau"] == 2.0
    assert scaled.identity["manifest_sha256"] != default.identity["manifest_sha256"]
    spec["stages"][0]["objective"]["tau"] = 4
    path.write_text(json.dumps(spec))
    assert load_v55_plan(path).identity["manifest_sha256"] != scaled.identity["manifest_sha256"]


@pytest.mark.parametrize("objective,match", [
    ({"kind": "query_log1p_scaled"}, "requires tau"),
    ({"kind": "query_log1p_scaled", "tau": 0}, "finite number"),
    ({"kind": "query_log1p_scaled", "tau": -1}, "finite number"),
    ({"kind": "query_log1p_scaled", "tau": float("inf")}, "finite number"),
    ({"kind": "query_log1p_scaled", "tau": True}, "finite number"),
    ({"kind": "squared", "tau": 1}, "does not accept tau"),
    ({"kind": "unknown"}, "unsupported"),
])
def test_invalid_objective_rejected_before_training(tmp_path, objective, match):
    path, spec = _plan(tmp_path)
    spec["stages"][0]["objective"] = objective
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_v55_plan(path)


def test_scaled_objective_rejects_retained_weight_and_v54_schema(tmp_path):
    path, spec = _plan(tmp_path, objective={"kind": "query_log1p_scaled", "tau": 2})
    spec["stages"][0]["loss"] = {"state_weights": [0.1, 0.9, 0, 0]}
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Query-only"):
        load_v55_plan(path)
    old = copy.deepcopy(spec)
    old["schema"] = V54_SCHEMA
    old["model"]["codec_version"] = "constant_weight_composition_v1"
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match=r"requires a V5\.5 plan"):
        load_v54_plan(path)


def test_one_step_journal_keeps_raw_square_and_logs_optimized_loss(tmp_path):
    path, spec = _plan(tmp_path)
    square = load_v55_plan(path)
    spec["stages"][0]["objective"] = {"kind": "query_log1p_scaled", "tau": 2}
    path.write_text(json.dumps(spec))
    scaled = load_v55_plan(path)
    torch.manual_seed(41)
    original = make_model(square).to(dtype=torch.float64)
    alternate = make_model(scaled).to(dtype=torch.float64)
    alternate.load_state_dict(original.state_dict())
    square_row = train_step(
        original, adamw(original, square.optimizer), square, square.tables[0],
        square.spec["stages"][0]["recipe"]["synthetic"], 0, "cpu",
        V53LossConfig(**square.spec["stages"][0]["loss"]), namespace="fit",
    )
    scaled_row = train_step(
        alternate, adamw(alternate, scaled.optimizer), scaled, scaled.tables[0],
        scaled.spec["stages"][0]["recipe"]["synthetic"], 0, "cpu",
        V53LossConfig(**scaled.spec["stages"][0]["loss"]), namespace="fit",
        objective=scaled.spec["stages"][0]["objective"],
    )
    assert square_row["loss"] == scaled_row["loss"]
    assert square_row["objective_loss"] == square_row["loss"]
    assert 0 <= scaled_row["objective_loss"] < scaled_row["loss"]
    assert scaled_row["objective_kind"] == "query_log1p_scaled"
    assert scaled_row["objective_tau"] == 2.0
