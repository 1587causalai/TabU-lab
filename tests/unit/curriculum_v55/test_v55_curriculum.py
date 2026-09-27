"""V5.5 default, historical candidate, and versioned curriculum boundaries."""

import hashlib
import json
from pathlib import Path

import pytest

from tabu_lab.cli import build_parser
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import V54_SCHEMA, V55_SCHEMA, load_v54_plan, load_v55_plan
from tabu_lab.models.restoration_v55 import V55Model


def _manifest(tmp_path):
    table = {
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i)] for i in range(8)],
        "splits": {"train": list(range(6)), "validation": [6], "test": [7]},
    }
    raw = json.dumps(table).encode()
    (tmp_path / "table.json").write_bytes(raw)
    return {
        "schema": V55_SCHEMA,
        "experiment_id": "v55-curriculum-fixture",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {},
        "tables": [
            {"id": kind, "kind": kind, "cohort": kind, "path": "table.json",
             "sha256": hashlib.sha256(raw).hexdigest()}
            for kind in ("synthetic", "real")
        ],
        "probes": [
            {"name": "fit", "cohorts": ["synthetic", "real"], "partition": "train",
             "purpose": "fit", "recipe": {"fraction": 0.3}, "masks": 1}
        ],
        "stages": [
            {"name": "fit", "question": "Can the V5.5 protocol select its codec?",
             "max_updates": 2, "max_seconds": 120,
             "sampling": [{"cohort": kind, "episodes": 1}
                          for kind in ("synthetic", "real")],
             "recipe": {kind: {"fraction": 0.3} for kind in ("synthetic", "real")},
             "optimizer": "adamw", "evaluate_every": 2, "checkpoint_every": 1,
             "probes": ["fit"]}
        ],
    }


def _save(tmp_path, spec, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps(spec))
    return path


def test_v55_defaults_to_new_composition_and_unit_zero(tmp_path):
    plan = load_v55_plan(_save(tmp_path, _manifest(tmp_path)))

    assert plan.config.codec_version == "constant_weight_composition_v2"
    assert plan.config.unit_layers == 0
    assert plan.summary["schema"] == plan.identity["schema"] == V55_SCHEMA
    assert "models/restoration_v55/model.py" in plan.identity["source"]["files"]
    assert "models/restoration_v53/codec_versions.py" in plan.identity["source"]["files"]
    assert isinstance(make_model(plan), V55Model)
    for recipe in [*plan.spec["stages"][0]["recipe"].values(),
                   plan.spec["probes"][0]["recipe"]]:
        assert recipe["kind"] == "supervised_row"


def test_old_composition_is_explicit_v55_candidate_with_separate_identity(tmp_path):
    default = load_v55_plan(_save(tmp_path, _manifest(tmp_path), "default.json"))
    candidate_spec = _manifest(tmp_path)
    candidate_spec["model"] = {"codec_version": "constant_weight_composition_v1"}
    candidate = load_v55_plan(_save(tmp_path, candidate_spec, "candidate.json"))

    assert candidate.config.codec_version == "constant_weight_composition_v1"
    assert candidate.config.unit_layers == 0
    assert isinstance(make_model(candidate), V55Model)
    assert candidate.identity["sha256"] != default.identity["sha256"]
    assert candidate.identity["source"] == default.identity["source"]


def test_v54_schema_rejects_new_codec_and_versioned_cli_rejects_crossed_manifest(tmp_path):
    spec = _manifest(tmp_path)
    spec["schema"] = V54_SCHEMA
    spec["model"] = {"codec_version": "constant_weight_composition_v2"}
    path = _save(tmp_path, spec, "v54-with-v55-codec.json")
    with pytest.raises(ValueError, match="codec"):
        load_v54_plan(path)

    v55_path = _save(tmp_path, _manifest(tmp_path), "v55.json")
    args = build_parser().parse_args(["curriculum-v54", "plan", "--manifest", str(v55_path)])
    with pytest.raises(ValueError, match=r"manifest\.schema must be tabu\.curriculum\.v54\.v1"):
        args.handler(args)
    args = build_parser().parse_args(["curriculum-v55", "plan", "--manifest", str(path)])
    with pytest.raises(ValueError, match=r"manifest\.schema must be tabu\.curriculum\.v55\.v1"):
        args.handler(args)


def test_v55_cli_plan_and_source_digest_track_actual_model_source(tmp_path, monkeypatch, capsys):
    path = _save(tmp_path, _manifest(tmp_path))
    args = build_parser().parse_args(["curriculum-v55", "plan", "--manifest", str(path)])
    assert args.handler(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "planned_not_run"
    assert result["resolved"]["schema"] == V55_SCHEMA
    before = load_v55_plan(path)
    original = Path.read_bytes

    def changed(self):
        value = original(self)
        if self.name == "model.py" and self.parent.name == "restoration_v55":
            return value + b"\n# simulated source change\n"
        return value

    monkeypatch.setattr(Path, "read_bytes", changed)
    after = load_v55_plan(path)
    assert before.identity["manifest_sha256"] == after.identity["manifest_sha256"]
    assert before.identity["source"] != after.identity["source"]
    assert before.identity["sha256"] != after.identity["sha256"]
