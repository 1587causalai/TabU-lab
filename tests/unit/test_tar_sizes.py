from __future__ import annotations

import json

import pytest
import torch

from tabu_lab.models.tar import TabUTARModel, TARConfig, TARTrainer, TARTrainingConfig
from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint
from tabu_lab.models.tar.verification import mixed_fixture
from tabu_lab.tar_fit import fit_model_config
from tabu_lab.tar_sizes import config_for_size, inspect_size, list_sizes


@pytest.mark.parametrize(
    "name,count", [("small", 721464), ("medium", 5521008), ("standard", 54071520)]
)
def test_named_size_counts_and_shared_architecture(name, count):
    cfg = config_for_size(name)
    assert inspect_size(name)["parameter_count"] == count
    assert cfg.parameter_count == count
    assert (cfg.heads, cfg.semantic_slots, cfg.inducing_slots) == (8, 32, 128)
    assert cfg.inducing_enabled and cfg.lambda_feature == cfg.lambda_unit == 1
    standard = TARConfig().as_dict()
    changed = {k for k, v in cfg.as_dict().items() if standard[k] != v}
    assert changed <= {"width", "blocks", "ff_width"}
    assert config_for_size("standard") == TARConfig()


@pytest.mark.parametrize("size", ["small", "medium"])
def test_research_sizes_real_forward_backward_update(size):
    torch.set_num_threads(1)
    model = TabUTARModel(config_for_size(size, initialization_seed=1729))
    item = mixed_fixture()
    trainer = TARTrainer(
        model, TARTrainingConfig(effective_episode_batch=1, warmup_steps=0, optimizer_steps=2)
    )
    result = trainer.train_step([item])
    assert result["step"] == 1 and result["gradient_norm"] > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.status == "ok" for p in model(item[0]).predictions)


def test_small_checkpoint_cannot_be_loaded_as_standard(tmp_path):
    cfg = config_for_size("small", initialization_seed=1729)
    model = TabUTARModel(cfg)
    save_checkpoint(model, tmp_path / "small")
    with pytest.raises(ValueError, match="does not match"):
        load_checkpoint(
            tmp_path / "small",
            expected_config=config_for_size("standard", initialization_seed=1729),
        )
    restored = load_checkpoint(tmp_path / "small", expected_config=cfg)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)


def test_size_selection_is_explicit_and_smoke_is_distinct():
    assert fit_model_config({}, 1729)[0] == TARConfig(initialization_seed=1729)
    cfg, name = fit_model_config({"model_size": "small", "expected_parameter_count": 721464}, 1729)
    assert name == "small" and cfg.parameter_count == 721464
    with pytest.raises(ValueError, match="parameter count"):
        fit_model_config({"model_size": "small", "expected_parameter_count": 54071520}, 1729)
    with pytest.raises(ValueError, match="unknown TAR size"):
        config_for_size("typo")
    smoke, label = fit_model_config({"model_size": "small"}, 1729, smoke=True)
    assert label == "cpu-smoke" and smoke != cfg


def test_sizes_cli_and_standard_inspect_remain_explicit(capsys):
    from tabu_lab.cli import main

    assert main(["tar", "sizes"]) == 0
    assert json.loads(capsys.readouterr().out) == list_sizes()
    assert main(["tar", "inspect", "--size", "small"]) == 0
    assert json.loads(capsys.readouterr().out)["parameter_count"] == 721464
    assert main(["tar", "inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["parameter_count"] == 54071520


def test_default_validation_is_small_and_tiny_requires_smoke(capsys):
    from tabu_lab.cli import main
    from tabu_lab.tar_sizes import DEFAULT_VALIDATION_SIZE

    assert DEFAULT_VALIDATION_SIZE == "small"
    assert main(["tar", "verify"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["model_size"] == "small" and result["parameter_count"] == 721464
    assert "strict_checkpoint_roundtrip" in result["checks"]
    assert main(["tar", "verify", "--smoke"]) == 0
    smoke = json.loads(capsys.readouterr().out)
    assert smoke["model_size"] == "cpu-smoke" and smoke["parameter_count"] < 721464
    assert config_for_size() == TARConfig()  # model construction remains Standard


def test_verify_size_routes_are_explicit_and_exclusive(capsys, monkeypatch):
    from tabu_lab import tar_validation
    from tabu_lab.cli import main

    seen = []
    monkeypatch.setattr(
        tar_validation, "verify_size", lambda size: seen.append(size) or dict(model_size=size)
    )
    for argv in (["--size", "medium"], ["--full"], ["--size", "standard"]):
        assert main(["tar", "verify", *argv]) == 0
        capsys.readouterr()
    assert seen == ["medium", "standard", "standard"]
    with pytest.raises(SystemExit):
        main(["tar", "verify", "--size", "small", "--full"])


def test_current_validation_prereg_is_small_and_historical_sizes_stay_explicit():
    from pathlib import Path

    from tabu_lab.tar_data import validate_full_dataset
    from tabu_lab.tar_fit import sha

    root = Path(__file__).resolve().parents[2] / "experiments/local"
    spec = json.loads((root / "tar-small-validation/preregistration.yaml").read_text())
    cfg, label = fit_model_config(spec, 1729)
    assert label == "small" and cfg.parameter_count == spec["expected_parameter_count"] == 721464
    for name, digest in spec["datasets"].items():
        p = root / f"tar-small-validation/data/{name}.json"
        assert sha(p) == digest
        assert (
            validate_full_dataset(json.loads(p.read_text()), spec["expected_rows"][name])[
                "unused_rows"
            ]
            == 0
        )
    old = json.loads((root / "tar-full-data-fit/preregistration.yaml").read_text())
    assert fit_model_config(old, 1729)[0].parameter_count == 54071520


def test_small_128_encoding_fit_configuration():
    legacy, name = fit_model_config({"model_size": "small-128"}, 1729)
    unified, _ = fit_model_config({"model_size": "small-128",
                                  "value_encoding": "unified_constant_weight"}, 1729)
    assert name == "small-128"
    assert legacy.width == unified.width == 128
    assert legacy.blocks == unified.blocks == 3
    assert legacy.ff_width == unified.ff_width == 256
    assert unified.parameter_count - legacy.parameter_count == 8224
    assert unified.value_encoding == "unified_constant_weight"
    smoke, _ = fit_model_config({"model_size": "small-128",
                                "value_encoding": "unified_constant_weight"}, 1729, smoke=True)
    assert smoke.width == 128 and smoke.value_encoding == unified.value_encoding
    with pytest.raises(ValueError, match="width=128"):
        fit_model_config({"model_size": "small", "value_encoding": "constant_weight"}, 1729)
