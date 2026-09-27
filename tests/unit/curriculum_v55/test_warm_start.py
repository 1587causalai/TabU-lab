"""V5.4 checkpoint conversion is explicit, model-only, and exact."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from tabu_lab.cli import build_parser
from tabu_lab.curriculum_v53.artifacts import (
    WARM_START_SCHEMA,
    load_checkpoint,
    load_warm_start,
    sha256,
)
from tabu_lab.curriculum_v53.protocol import (
    V54_SCHEMA,
    V55_SCHEMA,
    _digest,
    load_v54_plan,
    load_v55_plan,
)
from tabu_lab.curriculum_v53.runner import run
from tabu_lab.curriculum_v53.warm_start import (
    _convert_state,
    _convert_v55_state,
    prepare_v54_warm_start,
    prepare_v55_warm_start,
    validate_v55_warm_start,
    validate_warm_start,
)


def _plans(tmp_path, *, heads=8):
    table = {
        "features": [
            {"kind": "numeric"},
            {"kind": "nominal", "domain": ["a", "b"]},
            {"kind": "ordinal", "domain": ["low", "middle", "high"]},
        ],
        "values": [
            [float(i), i % 2, i % 3]
            for i in range(8)
        ],
        "splits": {"train": list(range(6)), "validation": [6], "test": [7]},
    }
    raw = json.dumps(table).encode()
    (tmp_path / "mixed.json").write_bytes(raw)
    numeric = {
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i)] for i in range(8)],
        "splits": table["splits"],
    }
    numeric_raw = json.dumps(numeric).encode()
    (tmp_path / "numeric.json").write_bytes(numeric_raw)
    base = {
        "schema": V54_SCHEMA, "experiment_id": "v54-donor",
        "seeds": dict(model=1, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"size": "small", "unit_layers": 3,
                  "backbone": {"heads": heads},
                  "codec_version": "constant_weight_composition_v1"},
        "tables": [
            {"id": f"numeric_{index:03d}", "kind": "synthetic", "cohort": "old120",
             "path": "numeric.json", "sha256": hashlib.sha256(numeric_raw).hexdigest()}
            for index in range(60)
        ] + [
            {"id": f"mixed_{index:03d}", "kind": "synthetic", "cohort": "old120",
             "path": "mixed.json", "sha256": hashlib.sha256(raw).hexdigest()}
            for index in range(60)
        ],
        "probes": [{"name": "train_fit", "cohorts": ["old120"],
                    "partition": "train", "purpose": "fit",
                    "recipe": {"fraction": 0.25}, "masks": 1}],
        "stages": [{"name": "old120_small_fit", "question": "Warm start parity?",
                    "max_updates": 1_228_800, "max_seconds": 1200,
                    "sampling": [{"cohort": "old120", "episodes": 120}],
                    "recipe": {"synthetic": {"fraction": 0.25}},
                    "optimizer": "adamw", "evaluate_every": 1,
                    "checkpoint_every": 1, "probes": ["train_fit"],
                    "loss_replay": {"kind": "normal120_top5_top20_v1",
                                    "normal_max_updates": 983_040,
                                    "start_normal_cursor": 0},
                    "loss": {"discrete_weight": 1.0,
                             "state_weights": [0.0, 1.0, 0.0, 0.0]}}],
    }
    old_path = tmp_path / "v54.json"
    old_path.write_text(json.dumps(base))
    old = load_v54_plan(old_path)
    new_spec = copy.deepcopy(base)
    new_spec["schema"] = V55_SCHEMA
    new_spec["experiment_id"] = "v55-target"
    new_spec["model"]["codec_version"] = "constant_weight_composition_v2"
    new_spec["stages"][0]["max_updates"] = 1_327_104
    new_spec["stages"][0]["loss_replay"] = {
        "kind": "normal120_p99x3_p95x2_p80x1_v2", "normal_max_updates": 983_040,
        "start_normal_cursor": 0, "start_extra_updates": 0,
    }
    new_path = tmp_path / "v55.json"
    new_path.write_text(json.dumps(new_spec))
    new = load_v55_plan(new_path)
    return old, new


@pytest.fixture
def donor(tmp_path):
    old, new = _plans(tmp_path)
    origin = tmp_path / "donor-run"
    stopped = run(old, origin, device="cpu", max_updates_this_invocation=0)
    assert stopped["outcome"] == "stopped", stopped
    checkpoint = origin / "checkpoint-progress.pt"
    resolved = origin / "resolved.json"
    return old, new, checkpoint, resolved


def test_prepare_and_initialize_resets_state_and_rejects_resume(donor, tmp_path):
    _old, new, checkpoint, resolved = donor
    prepared = prepare_v54_warm_start(new, checkpoint, resolved, tmp_path / "prepare", device="cpu")
    assert prepared["outcome"] == "passed", prepared
    assert prepared["parent_checkpoint_sha256"] == sha256(checkpoint)
    assert prepared["runtime"]["dtype"] == "float64"
    assert prepared["step0_comparison"]["train_fit"]["table_count"] == 120
    assert prepared["step0_comparison"]["train_fit"]["bank"]["episodes"] == 120
    assert {row["selection"] for row in prepared["clone_one_step"]} == {
        "numeric", "mixed_nominal_ordinal"
    }
    assert all(row["gradient_norm"] >= 0 for row in prepared["clone_one_step"])
    artifact = prepared["artifact"]
    payload, digest = load_warm_start(artifact)
    assert digest == prepared["artifact_sha256"]
    assert payload["model"]["_codec_signature"].tolist() == [6, 1]
    assert len(payload["conversion"]["tensors"]) == 116
    assert payload["conversion"]["trainable_parameters"] == 2_082_688
    assert not {"optimizer", "rng", "state"} & set(payload)
    parent, _ = load_checkpoint(checkpoint)
    for name, value in parent["model"].items():
        if name != "_codec_signature":
            assert torch.equal(payload["model"][name], value)

    started = run(new, tmp_path / "target-run", device="cpu", initialize_from=artifact,
                  max_updates_this_invocation=0)
    assert started["outcome"] == "stopped", started
    assert started["update"] == started["cursor"] == started["stage_index"] == 0
    assert started["exposure"] == {}
    fresh, _ = load_checkpoint(tmp_path / "target-run" / "checkpoint-progress.pt")
    assert fresh["optimizer"]["state"] == {}
    assert fresh["model"]["_codec_signature"].tolist() == [6, 1]
    assert started["lineage"][-1]["initialization_artifact_sha256"] == digest
    rejected = run(new, tmp_path / "reject-resume", device="cpu", resume=artifact,
                   max_updates_this_invocation=0)
    assert rejected["outcome"] == "failed"
    assert rejected["checkpoint"] is None


def test_transfer_casts_every_float_tensor_and_rejects_identity_drift(donor, tmp_path,
                                                                      monkeypatch):
    _, new, checkpoint, resolved = donor
    from tabu_lab.curriculum_v53 import warm_start

    parent, _ = load_checkpoint(checkpoint)
    monkeypatch.setattr(warm_start, "execution_dtype", lambda _device: torch.float32)
    converted, mapping, _ = _convert_state(new, parent, warm_start.V54Config.from_dict(
        parent["model_config"]), "cpu")
    assert len(mapping) == 116
    for name, source in parent["model"].items():
        if name == "_codec_signature":
            continue
        assert converted[name].dtype == torch.float32
        assert torch.equal(converted[name], source.float())

    wrong = copy.deepcopy(new.spec)
    wrong["seeds"]["evaluation"] += 1
    changed_path = tmp_path / "wrong.json"
    changed_path.write_text(json.dumps(wrong))
    changed = load_v55_plan(changed_path)
    failed = prepare_v54_warm_start(changed, checkpoint, resolved, tmp_path / "wrong-prepare")
    assert failed["outcome"] == "failed"
    assert "seeds differ" in failed["error"]
    assert "artifact" not in failed


def test_v54_parent_rejects_inconsistent_source_digest(donor, tmp_path, monkeypatch):
    _, new, checkpoint, resolved = donor
    from tabu_lab.curriculum_v53 import warm_start

    parent, checkpoint_digest = load_checkpoint(checkpoint)
    altered = json.loads(resolved.read_text())
    altered["identity"]["source"]["files"]["cli.py"] = "0" * 64
    altered["identity"]["sha256"] = _digest({
        key: value for key, value in altered["identity"].items() if key != "sha256"
    })
    parent["identity"] = altered["identity"]
    altered_path = tmp_path / "inconsistent-source-resolved.json"
    altered_path.write_text(json.dumps(altered))
    monkeypatch.setattr(warm_start, "load_checkpoint", lambda _path: (
        parent, checkpoint_digest,
    ))

    with pytest.raises(ValueError, match="source digest mismatch"):
        warm_start._validated_parent(new, checkpoint, altered_path)


def test_h4_fp32_parent_reuses_every_tensor_and_rejects_head_drift(tmp_path, monkeypatch):
    old, new = _plans(tmp_path, heads=4)
    origin = tmp_path / "h4-donor-run"
    assert run(old, origin, device="cpu", max_updates_this_invocation=0)["outcome"] == "stopped"
    parent, _ = load_checkpoint(origin / "checkpoint-progress.pt")
    # A content-addressed snapshot with the H4 donor's MPS/FP32 tensor contract.
    # The actual same-device numerical admission is performed on the training host.
    parent["runtime"] = {**parent["runtime"], "device": "mps", "dtype": "float32",
                         "mps_available": True, "mps_cpu_fallback": False}
    parent["model"] = {
        name: value.float() if value.is_floating_point() else value.clone()
        for name, value in parent["model"].items()
    }
    temporary = tmp_path / "h4-parent.tmp"
    torch.save(parent, temporary)
    checkpoint = tmp_path / f"{sha256(temporary)}.pt"
    temporary.rename(checkpoint)

    with monkeypatch.context() as patch:
        from tabu_lab.curriculum_v53 import warm_start
        patch.setattr(warm_start, "execution_dtype", lambda _device: torch.float32)
        converted, mapping, _ = _convert_state(
            new, parent, warm_start.V54Config.from_dict(parent["model_config"]), "cpu")
    assert len(mapping) == 116
    assert sum(item["action"] == "copy_cast" for item in mapping) == 115
    for name, value in parent["model"].items():
        if name != "_codec_signature":
            assert converted[name].dtype == torch.float32
            assert torch.equal(converted[name], value)

    prepared = prepare_v54_warm_start(
        new, checkpoint, origin / "resolved.json", tmp_path / "h4-prepare", device="cpu")
    assert prepared["outcome"] == "passed", prepared
    assert prepared["parent_checkpoint_sha256"] == sha256(checkpoint)
    artifact, _ = load_warm_start(prepared["artifact"])
    assert artifact["parent"]["checkpoint_runtime"]["device"] == "mps"
    assert artifact["target_model_config"]["backbone"]["heads"] == 4
    for name, value in parent["model"].items():
        if name != "_codec_signature":
            assert torch.equal(artifact["model"][name], value.double())

    changed = copy.deepcopy(new.spec)
    changed["model"]["backbone"]["heads"] = 8
    path = tmp_path / "wrong-h4-target.json"
    path.write_text(json.dumps(changed))
    failed = prepare_v54_warm_start(
        load_v55_plan(path), checkpoint, origin / "resolved.json", tmp_path / "wrong-h4-prepare")
    assert failed["outcome"] == "failed"
    assert "permits only the ordinal codec change" in failed["error"]


def test_cli_exposes_dedicated_v55_prepare_only(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "curriculum-v55", "prepare-v54-warm-start", "--manifest", "target.json",
        "--donor-checkpoint", "donor.pt", "--donor-resolved", "resolved.json",
        "--output-root", str(tmp_path / "out"), "--device", "mps",
    ])
    assert args.curriculum_command == "prepare-v54-warm-start"
    assert args.donor_checkpoint.name == "donor.pt"
    with pytest.raises(SystemExit):
        parser.parse_args(["curriculum-v54", "prepare-v54-warm-start"])


@pytest.mark.parametrize("device,recorded_device,dtype", [
    ("cpu", "cpu", torch.float64),
    ("mps", "mps", torch.float32),
    ("mps:0", "mps", torch.float32),
    ("cuda:0", "cuda:0", torch.float64),
    ("cuda:1", "cuda:1", torch.float64),
])
def test_warm_start_validates_full_runtime_device_identity(device, recorded_device, dtype):
    """A CUDA index must survive validation even when no CUDA host is present."""
    class Model:
        def state_dict(self):
            return {"weight": torch.ones(1, dtype=dtype),
                    "_codec_signature": torch.tensor([6, 1])}

        def parameters(self):
            yield SimpleNamespace(device=torch.device(device), dtype=dtype,
                                  numel=lambda: 1)

    model = Model()
    state = model.state_dict()
    plan = SimpleNamespace(
        spec={"schema": V55_SCHEMA}, identity={"sha256": "target"},
        config=SimpleNamespace(as_dict=lambda: {"codec_version": "v2"}),
    )
    payload = {
        "schema": WARM_START_SCHEMA, "purpose": "weights_only_initialization",
        "target_identity": plan.identity, "target_model_config": plan.config.as_dict(),
        "model": state,
        "conversion": {
            "codec_signature": {"source": [4, 1], "target": [6, 1]},
            "tensor_count": 2, "trainable_parameters": 1,
            "tensors": [
                {"source": name, "target": name, "source_shape": list(value.shape),
                 "target_shape": list(value.shape), "target_dtype": str(value.dtype),
                 "action": "replace_codec_signature" if name == "_codec_signature" else "copy_cast"}
                for name, value in state.items()
            ],
        },
        "target_runtime": {"device": recorded_device,
                           "dtype": str(dtype).removeprefix("torch.")},
        "parent": {"identity": {"schema": V54_SCHEMA}, "checkpoint_sha256": "a" * 64},
    }
    validate_warm_start(payload, plan, model)
    wrong = copy.deepcopy(payload)
    wrong["target_runtime"]["device"] = (
        "cuda" if device.startswith("cuda:") else "cuda:0"
    )
    with pytest.raises(ValueError, match="device or dtype"):
        validate_warm_start(wrong, plan, model)
    wrong = copy.deepcopy(payload)
    wrong["target_runtime"]["dtype"] = "float32" if dtype == torch.float64 else "float64"
    with pytest.raises(ValueError, match="device or dtype"):
        validate_warm_start(wrong, plan, model)


def test_v55_warm_start_accepts_mps_zero_alias_but_rejects_other_device():
    class Model:
        def __init__(self, device):
            self.device = device

        def state_dict(self):
            return {"weight": torch.ones(1, dtype=torch.float32),
                    "_codec_signature": torch.tensor([8, 1])}

        def parameters(self):
            yield SimpleNamespace(device=torch.device(self.device), dtype=torch.float32,
                                  numel=lambda: 1)

    parent_identity = {"schema": V55_SCHEMA}
    parent_identity["sha256"] = _digest(parent_identity)
    plan = SimpleNamespace(
        spec={"schema": V55_SCHEMA}, identity={"sha256": "target"},
        config=SimpleNamespace(as_dict=lambda: {
            "codec_version": "constant_weight_composition_v3",
        }, codec_version="constant_weight_composition_v3"),
    )
    state = Model("mps:0").state_dict()
    payload = {
        "schema": WARM_START_SCHEMA, "purpose": "weights_only_initialization",
        "target_identity": plan.identity, "target_model_config": plan.config.as_dict(),
        "target_runtime": {"device": "mps", "dtype": "float32"},
        "parent": {
            "identity": parent_identity,
            "model_config": {"codec_version": "constant_weight_composition_v2"},
            "checkpoint_update": 1, "checkpoint_sha256": "a" * 64,
            "resolved_sha256": "b" * 64,
        },
        "model": state,
        "conversion": {
            "codec_signature": {"source": [6, 1], "target": [8, 1]},
            "tensor_count": 2, "trainable_parameters": 1,
            "tensors": [
                {"source": name, "target": name, "source_shape": list(value.shape),
                 "target_shape": list(value.shape), "target_dtype": str(value.dtype),
                 "action": "replace_codec_signature" if name == "_codec_signature"
                 else "copy_cast"}
                for name, value in state.items()
            ],
        },
    }
    validate_v55_warm_start(payload, plan, Model("mps:0"))
    with pytest.raises(ValueError, match="device or dtype"):
        validate_v55_warm_start(payload, plan, Model("mps:1"))
    wrong = copy.deepcopy(payload)
    wrong["target_runtime"]["device"] = "mps:0"
    with pytest.raises(ValueError, match="device or dtype"):
        validate_v55_warm_start(wrong, plan, Model("mps:0"))


@pytest.mark.parametrize("target_version,signature", [
    ("constant_weight_composition_v2", [6, 1]),
    ("constant_weight_composition_v3", [8, 1]),
])
def test_v55_paired_transfer_reuses_all_weights_and_resets_training_state(
    tmp_path, monkeypatch, target_version, signature,
):
    _, donor_plan = _plans(tmp_path)
    donor_run = tmp_path / "v55-donor-run"
    assert run(donor_plan, donor_run, device="cpu", max_updates_this_invocation=0)[
        "outcome"
    ] == "stopped"
    checkpoint = donor_run / "checkpoint-progress.pt"
    parent, _ = load_checkpoint(checkpoint)
    target_spec = copy.deepcopy(donor_plan.spec)
    target_spec["experiment_id"] = f"v55-paired-{target_version}"
    target_spec["model"]["codec_version"] = target_version
    target_path = tmp_path / f"target-{target_version}.json"
    target_path.write_text(json.dumps(target_spec))
    target_plan = load_v55_plan(target_path)
    from tabu_lab.curriculum_v53 import warm_start

    def paired_comparison(_plan, _donor_config, _source, _target, _device):
        return {"train_fit": {
            "bank": {"sha256": "a" * 64, "episodes": 120},
            "v54": {"macro": {}, "tables": 120},
            "v55": {"macro": {}, "tables": 120},
        }}

    monkeypatch.setattr(warm_start, "_comparison", paired_comparison)
    prepared = prepare_v55_warm_start(
        target_plan, checkpoint, donor_run / "resolved.json",
        tmp_path / f"prepare-{target_version}", device="cpu",
    )
    assert prepared["outcome"] == "passed", prepared
    assert prepared["parent_checkpoint_sha256"] == sha256(checkpoint)
    assert prepared["conversion"]["codec_signature"] == {
        "source": [6, 1], "target": signature,
    }
    artifact, digest = load_warm_start(prepared["artifact"])
    assert digest == prepared["artifact_sha256"]
    assert artifact["model"]["_codec_signature"].tolist() == signature
    assert artifact["parent"]["identity"] == parent["identity"]
    assert artifact["parent"]["checkpoint_update"] == 0
    assert len(artifact["conversion"]["tensors"]) == 116
    assert artifact["conversion"]["trainable_parameters"] == 2_082_688
    assert not {"optimizer", "rng", "state"} & set(artifact)
    for name, value in parent["model"].items():
        if name != "_codec_signature":
            assert torch.equal(artifact["model"][name], value)

    target_run = tmp_path / f"run-{target_version}"
    started = run(target_plan, target_run, initialize_from=prepared["artifact"],
                  device="cpu", max_updates_this_invocation=0)
    assert started["outcome"] == "stopped", started
    assert started["update"] == started["cursor"] == 0
    assert started["exposure"] == {}
    fresh, _ = load_checkpoint(target_run / "checkpoint-progress.pt")
    assert fresh["model"]["_codec_signature"].tolist() == signature
    assert fresh["optimizer"]["state"] == {}
    assert started["lineage"][-1]["initialization_artifact_sha256"] == digest
    refused = run(target_plan, tmp_path / f"resume-{target_version}",
                  resume=prepared["artifact"], device="cpu",
                  max_updates_this_invocation=0)
    assert refused["outcome"] == "failed"
    assert refused["checkpoint"] is None


def test_v55_paired_transfer_rejects_identity_drift_and_casts_every_weight(tmp_path,
                                                                           monkeypatch):
    _, donor_plan = _plans(tmp_path)
    donor_run = tmp_path / "v55-donor-run"
    assert run(donor_plan, donor_run, device="cpu", max_updates_this_invocation=0)[
        "outcome"
    ] == "stopped"
    parent, _ = load_checkpoint(donor_run / "checkpoint-progress.pt")
    target_spec = copy.deepcopy(donor_plan.spec)
    target_spec["experiment_id"] = "v55-paired-v3"
    target_spec["model"]["codec_version"] = "constant_weight_composition_v3"
    path = tmp_path / "target-v3.json"
    path.write_text(json.dumps(target_spec))
    target_plan = load_v55_plan(path)
    from tabu_lab.curriculum_v53 import warm_start

    monkeypatch.setattr(warm_start, "execution_dtype", lambda _device: torch.float32)
    converted, mapping, model = _convert_v55_state(
        target_plan, parent, warm_start.V55Config.from_dict(parent["model_config"]), "cpu",
    )
    assert len(mapping) == 116
    assert converted["_codec_signature"].tolist() == [8, 1]
    for name, value in parent["model"].items():
        if name != "_codec_signature":
            assert converted[name].dtype == torch.float32
            assert torch.equal(converted[name], value.float())
    wrong_shape = copy.deepcopy(parent)
    weight_name = next(name for name, tensor in wrong_shape["model"].items()
                       if name != "_codec_signature" and tensor.numel() > 1)
    wrong_shape["model"][weight_name] = wrong_shape["model"][weight_name].reshape(-1)[:1]
    with pytest.raises(ValueError, match="tensor shape differs"):
        _convert_v55_state(
            target_plan, wrong_shape,
            warm_start.V55Config.from_dict(parent["model_config"]), "cpu",
        )
    bad_spec = copy.deepcopy(target_spec)
    bad_spec["seeds"]["evaluation"] += 1
    bad_path = tmp_path / "wrong-seed.json"
    bad_path.write_text(json.dumps(bad_spec))
    bad_plan = load_v55_plan(bad_path)
    failed = prepare_v55_warm_start(
        bad_plan, donor_run / "checkpoint-progress.pt", donor_run / "resolved.json",
        tmp_path / "wrong-prepare", device="cpu",
    )
    assert failed["outcome"] == "failed"
    assert "seeds differ" in failed["error"]
    wrong_resolved = json.loads((donor_run / "resolved.json").read_text())
    wrong_resolved["identity"]["sha256"] = "0" * 64
    wrong_resolved_path = tmp_path / "wrong-resolved.json"
    wrong_resolved_path.write_text(json.dumps(wrong_resolved))
    failed = prepare_v55_warm_start(
        target_plan, donor_run / "checkpoint-progress.pt", wrong_resolved_path,
        tmp_path / "wrong-resolved-prepare", device="cpu",
    )
    assert failed["outcome"] == "failed"
    assert "resolved identity differs" in failed["error"]
    del model


def test_v55_paired_cli_is_v55_only(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "curriculum-v55", "prepare-v55-warm-start", "--manifest", "target.json",
        "--donor-checkpoint", "donor.pt", "--donor-resolved", "resolved.json",
        "--output-root", str(tmp_path / "out"), "--device", "mps",
    ])
    assert args.curriculum_command == "prepare-v55-warm-start"
    with pytest.raises(SystemExit):
        parser.parse_args(["curriculum-v54", "prepare-v55-warm-start"])
