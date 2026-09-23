"""Explicit V5.4 dispatch and unchanged legacy manifest requirements."""

import hashlib
import json
from pathlib import Path

import pytest

from tabu_lab.cli import build_parser
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.protocol import SCHEMA, V54_SCHEMA, load_plan, load_v54_plan


def manifest(tmp_path, *, tiny=False):
    data = {
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i)] for i in range(8)],
        "splits": {"train": list(range(6)), "validation": [6], "test": [7]},
    }
    raw = json.dumps(data).encode()
    (tmp_path / "table.json").write_bytes(raw)
    model = ({"size": "nano", "backbone": {"layers": 1, "slots": 4, "ff_width": 16},
              "regression_width": 4, "center_chunk_size": 4} if tiny else {})
    return {
        "schema": V54_SCHEMA, "experiment_id": "v54-protocol-fixture",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": model,
        "tables": [{"id": kind, "kind": kind, "cohort": kind, "path": "table.json",
                    "sha256": hashlib.sha256(raw).hexdigest()} for kind in ("synthetic", "real")],
        "probes": [{"name": "fit", "cohorts": ["synthetic", "real"], "partition": "train",
                    "purpose": "fit", "recipe": {"fraction": 0.3}, "masks": 1}],
        "stages": [{"name": "fit", "question": "Can the shared protocol train and resume?",
                    "max_updates": 2, "max_seconds": 120,
                    "sampling": [{"cohort": kind, "episodes": 1}
                                 for kind in ("synthetic", "real")],
                    "recipe": {kind: {"fraction": 0.3} for kind in ("synthetic", "real")},
                    "optimizer": "adamw", "evaluate_every": 2, "checkpoint_every": 1,
                    "probes": ["fit"]}],
    }


def save(tmp_path, spec, name="manifest.json"):
    path = tmp_path / name
    path.write_text(json.dumps(spec))
    return path


def test_defaults_are_small_and_supervised_for_every_table_kind(tmp_path):
    plan = load_v54_plan(save(tmp_path, manifest(tmp_path)))
    assert plan.config.size == "small"
    assert plan.config.unit_layers == 3
    assert plan.config.backbone.layers == 3
    assert plan.config.backbone.heads == 8
    assert plan.config.codec_version == "constant_weight_composition_v1"
    for recipe in [*plan.spec["stages"][0]["recipe"].values(), plan.spec["probes"][0]["recipe"]]:
        assert recipe == {"kind": "supervised_row", "fraction": 0.3,
                          "numeric_query_guard": {"kind": "none"}}
    assert plan.summary["schema"] == plan.identity["schema"] == V54_SCHEMA
    assert "affine composition" in plan.summary["codec_boundary"]["ordinal_domain"]
    assert "models/restoration_v54/model.py" in plan.identity["source"]["files"]
    assert "curriculum_v53/factory.py" in plan.identity["source"]["files"]
    assert plan.summary["execution_started"] is False


def test_explicit_random_cell_and_nano_are_resolved_without_relabeling(tmp_path):
    spec = manifest(tmp_path)
    spec["model"] = {"size": "nano", "backbone": {"slots": 8},
                     "codec_version": "unit_gaussian_composition_v1"}
    spec["stages"][0]["recipe"]["synthetic"]["kind"] = "random_cell"
    plan = load_v54_plan(save(tmp_path, spec))
    assert plan.config.size == "nano"
    assert plan.config.unit_layers == 0
    assert plan.config.backbone.layers == 2
    assert plan.config.backbone.heads == 4
    assert plan.config.backbone.slots == 8
    assert plan.spec["stages"][0]["recipe"]["synthetic"]["kind"] == "random_cell"
    assert plan.spec["stages"][0]["recipe"]["real"]["kind"] == "supervised_row"
    resolved = load_v54_plan(save(tmp_path, plan.spec, "resolved.json"))
    assert plan.identity == resolved.identity
    assert plan.config == resolved.config


def test_v53_keeps_required_kind_and_original_model_defaults(tmp_path):
    spec = manifest(tmp_path)
    spec["schema"] = SCHEMA
    with pytest.raises(ValueError, match=r"missing fields.*kind"):
        load_plan(save(tmp_path, spec))
    for recipe in [*spec["stages"][0]["recipe"].values(), spec["probes"][0]["recipe"]]:
        recipe["kind"] = "random_cell"
    plan = load_plan(save(tmp_path, spec))
    assert plan.config.codec_version == "constant_weight_v1"
    assert plan.config.unit_layers == 0
    assert "size" not in plan.spec["model"]
    assert plan.spec["schema"] == SCHEMA
    assert "models/restoration_v54/model.py" not in plan.identity["source"]["files"]


@pytest.mark.parametrize("schema,codec", [
    (SCHEMA, "constant_weight_composition_v1"),
    (V54_SCHEMA, "constant_weight_v1"),
])
def test_schema_cannot_claim_a_codec_from_the_other_generation(tmp_path, schema, codec):
    spec = manifest(tmp_path)
    spec["schema"] = schema
    spec["model"]["codec_version"] = codec
    loader = load_plan if schema == SCHEMA else load_v54_plan
    with pytest.raises(ValueError, match="codec"):
        loader(save(tmp_path, spec))


@pytest.mark.parametrize("change,match", [
    (lambda s: s["stages"][0]["recipe"]["real"].pop("fraction"), "fraction"),
    (lambda s: s["stages"][0].pop("max_updates"), "max_updates"),
    (lambda s: s["stages"][0].pop("max_seconds"), "max_seconds"),
    (lambda s: s["tables"][0].update(target_columns=[0, 1]), "unknown fields.*target_columns"),
    (lambda s: s["tables"][0].update(target_column=[0, 1]), "target_column.*integer"),
])
def test_unspecified_budgets_and_unsupported_multi_target_fail_closed(tmp_path, change, match):
    spec = manifest(tmp_path)
    change(spec)
    with pytest.raises(ValueError, match=match):
        load_v54_plan(save(tmp_path, spec))


@pytest.mark.parametrize("version,schema", [("v53", V54_SCHEMA), ("v54", SCHEMA)])
def test_cli_and_schema_cannot_be_crossed(tmp_path, version, schema):
    spec = manifest(tmp_path)
    spec["schema"] = schema
    path = save(tmp_path, spec)
    args = build_parser().parse_args([f"curriculum-{version}", "plan", "--manifest", str(path)])
    expected = rf"manifest\.schema must be tabu\.curriculum\.{version}\.v1"
    with pytest.raises(ValueError, match=expected):
        args.handler(args)
    loader = load_plan if schema == V54_SCHEMA else load_v54_plan
    with pytest.raises(ValueError, match=r"manifest\.schema"):
        loader(path)


def test_v54_cli_plan_is_explicitly_unstarted(tmp_path, capsys):
    path = save(tmp_path, manifest(tmp_path))
    args = build_parser().parse_args(["curriculum-v54", "plan", "--manifest", str(path)])
    assert args.handler(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["outcome"] == "planned_not_run"
    assert result["resolved"]["schema"] == V54_SCHEMA


def test_composition_ordinal_reserved_rank_does_not_need_nominal_class_support(tmp_path):
    spec = manifest(tmp_path, tiny=True)
    data = json.loads((tmp_path / "table.json").read_text())
    data["features"][1] = {"kind": "ordinal", "domain": ["low", "middle", "high"]}
    for index, row in enumerate(data["values"]):
        row[1] = (0 if index % 2 else 2) if index < 6 else 1
    raw = json.dumps(data).encode()
    (tmp_path / "table.json").write_bytes(raw)
    for entry in spec["tables"]:
        entry["sha256"] = hashlib.sha256(raw).hexdigest()
    plan = load_v54_plan(save(tmp_path, spec))
    inputs, _request, truth, info = build_episode(
        plan.tables[0], plan.spec["probes"][0]["recipe"], 0, plan.spec["seeds"], "cpu",
        evaluation=True, partition="validation", codec_version=plan.config.codec_version,
    )
    assert info["query_addresses"] == [[6, 1]]
    assert info["support_policy"]["protect_ordinal_classes"] is False
    assert not inputs.visible[-1, 1]
    assert truth is not None


def test_source_identity_changes_with_actual_v54_source(tmp_path, monkeypatch):
    path = save(tmp_path, manifest(tmp_path))
    before = load_v54_plan(path)
    original = Path.read_bytes

    def changed(self):
        value = original(self)
        return value + b"\n# simulated source change\n" if (
            self.name == "model.py" and self.parent.name == "restoration_v54"
        ) else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    after = load_v54_plan(path)
    assert before.identity["manifest_sha256"] == after.identity["manifest_sha256"]
    assert before.identity["source"] != after.identity["source"]
    assert before.identity["sha256"] != after.identity["sha256"]


def _exact(actual, expected):
    import torch

    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _exact(actual[key], expected[key])
    elif isinstance(expected, list | tuple):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            _exact(item, reference)
    else:
        assert actual == expected


def test_tiny_training_resume_preflight_and_evaluation_keep_v54_identity(tmp_path):
    from tabu_lab.curriculum_v53.artifacts import load_checkpoint
    from tabu_lab.curriculum_v53.preflight import preflight
    from tabu_lab.curriculum_v53.runner import evaluate_checkpoint, run

    plan = load_v54_plan(save(tmp_path, manifest(tmp_path, tiny=True)))
    full = run(plan, tmp_path / "full")
    assert full["outcome"] == "completed", full
    assert full["schema"] == "tabu.curriculum.v54.terminal.v1"
    assert "# V5.4" in (tmp_path / "full" / "report.md").read_text()
    partial = run(plan, tmp_path / "partial", max_updates_this_invocation=1)
    assert partial["outcome"] == "stopped", partial
    assert partial["update"] == 1
    resumed = run(plan, tmp_path / "resumed", resume=tmp_path / "partial/checkpoint-progress.pt")
    assert resumed["outcome"] == "completed", resumed
    expected, _ = load_checkpoint(tmp_path / "full/checkpoint-progress.pt")
    actual, _ = load_checkpoint(tmp_path / "resumed/checkpoint-progress.pt")
    assert actual["schema"] == "tabu.curriculum.v54.checkpoint.v1"
    assert actual["identity"]["schema"] == V54_SCHEMA
    assert actual["model_config"]["codec_version"] == "constant_weight_composition_v1"
    assert actual["model_config"]["size"] == "nano"
    for key in ("model", "optimizer", "rng"):
        _exact(actual[key], expected[key])
    assert actual["state"]["update"] == 2
    qualified = preflight(plan, tmp_path / "preflight", max_seconds=120)
    assert qualified["outcome"] == "passed", qualified
    assert qualified["schema"] == "tabu.curriculum.v54.preflight.v1"
    assert all(probe["next_update_exact"] for probe in qualified["probes"])
    evaluated = evaluate_checkpoint(plan, tmp_path / "full/checkpoint-progress.pt",
                                    tmp_path / "evaluation")
    assert evaluated["outcome"] == "completed", evaluated
    assert evaluated["schema"] == "tabu.curriculum.v54.evaluation.v1"
    assert evaluated["checkpoint_update"] == 2
    rejected = run(plan, tmp_path / "reject-preflight",
                   resume=tmp_path / "preflight/probe-000.pt")
    assert rejected["outcome"] == "failed"
    assert rejected["checkpoint"] is None
    assert "qualification artifacts" in rejected["error"]


def test_checkpoint_version_cannot_be_relabelled_or_resumed_as_v53(tmp_path):
    import torch

    from tabu_lab.curriculum_v53.artifacts import load_checkpoint, sha256
    from tabu_lab.curriculum_v53.runner import run

    spec = manifest(tmp_path, tiny=True)
    plan = load_v54_plan(save(tmp_path, spec))
    result = run(plan, tmp_path / "v54", max_updates_this_invocation=0)
    assert result["outcome"] == "stopped", result
    checkpoint = tmp_path / "v54/checkpoint-progress.pt"
    payload, _ = load_checkpoint(checkpoint)
    payload["schema"] = "tabu.curriculum.v53.checkpoint.v1"
    forged = tmp_path / "forged.pt"
    torch.save(payload, forged)
    forged.with_suffix(".json").write_text(json.dumps({"sha256": sha256(forged)}))
    with pytest.raises(ValueError, match="schema or identity mismatch"):
        load_checkpoint(forged)
    spec["schema"] = SCHEMA
    spec["model"].pop("size")
    for recipe in [*spec["stages"][0]["recipe"].values(), spec["probes"][0]["recipe"]]:
        recipe["kind"] = "supervised_row"
    legacy = load_plan(save(tmp_path, spec, "legacy.json"))
    for name, options in (("resume", {"resume": checkpoint}),
                          ("initialize", {"initialize_from": checkpoint})):
        rejected = run(legacy, tmp_path / f"reject-{name}", **options)
        assert rejected["outcome"] == "failed"
        assert rejected["checkpoint"] is None
        assert ("identity drift" in rejected["error"]
                or "identical model contract" in rejected["error"])
