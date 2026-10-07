"""V6 weight reuse must be exact and paired; failed V7 parents are inadmissible."""

import importlib.util
from pathlib import Path

import pytest
import torch

from tabu_lab.models.restoration_v7 import V7Config, V7Model

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments/v7-seed-singlepass-fit-20261001/run_comparison.py"
)
spec = importlib.util.spec_from_file_location("v7_warm_fit", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def old_parent():
    config = V7Config(
        query_init="donor", rounds=4, codec="C64/8",
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=4),
        coupling_hidden=(16, 16),
    )
    original = V7Model(config)
    original.rounds[0].backbone = probe.InheritedV6Dynamics(config.backbone, 3)
    original = original.double()
    with torch.no_grad():
        for parameter in original.parameters():
            parameter.add_(1e-10)
    weights = {}
    for key, value in original.state_dict().items():
        if key.startswith("rounds.0.backbone.axial."):
            weights["backbone." + key.removeprefix("rounds.0.backbone.axial.")] = value.clone()
        elif key.startswith("rounds.0.backbone.unit_blocks."):
            weights[key.removeprefix("rounds.0.backbone.")] = value.clone()
        elif key in ("rounds.0.unit_seed", "rounds.0.feature_seed"):
            weights["encoder." + key.removeprefix("rounds.0.")] = value.clone()
    weights["encoder.projection.weight"] = torch.eye(128, dtype=torch.float64)
    weights["_codec_signature"] = torch.tensor([6], dtype=torch.int64)
    settings = {"backbone": config.as_dict()["backbone"], "unit_layers": 3,
                "regression_width": None, "ridge": .001, "bandwidth": 1.0, "epsilon": 1e-6}
    return {"model_config": settings, "model": weights}


def test_warm_fit_preserves_all_native_precision_weights_and_pairs_initial_state():
    parent = old_parent()
    new, optimizer = probe.model_for("seed1", 7, torch.device("cpu"), torch.float64, parent)
    old, _ = probe.model_for("donor4", 7, torch.device("cpu"), torch.float64, parent)
    assert new.config.rounds == 1 and new.config.query_init == "seed"
    assert old.config.rounds == 4 and old.config.query_init == "donor"
    assert new.config.codec == "G64"
    assert len(new.rounds[0].backbone.unit_blocks) == 3
    assert not optimizer.state
    assert probe.state_hash(new) == probe.state_hash(old)
    for key, value in parent["model"].items():
        if key in ("encoder.projection.weight", "_codec_signature"):
            continue
        if key.startswith("backbone."):
            dest = "rounds.0.backbone.axial." + key.removeprefix("backbone.")
        elif key.startswith("unit_blocks."):
            dest = "rounds.0.backbone." + key
        else:
            dest = "rounds.0." + key.removeprefix("encoder.")
        assert torch.equal(new.state_dict()[dest], value), key
    assert new.transfer_receipt["transferred_tensors"] == len(parent["model"]) - 2
    assert new.transfer_receipt["not_transferred"] == ["encoder.projection.weight"]
    assert new.transfer_receipt["non_parameter_metadata_not_transferred"] == ["_codec_signature"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "nonfinite"])
def test_warm_fit_rejects_partial_or_invalid_parent_weights(mutation):
    parent = old_parent()
    if mutation == "missing":
        del parent["model"]["encoder.unit_seed"]
    elif mutation == "extra":
        parent["model"]["unexpected.weight"] = torch.ones(3)
    else:
        parent["model"]["encoder.unit_seed"][0] = float("nan")
    with pytest.raises(ValueError):
        probe.model_for("seed1", 7, torch.device("cpu"), torch.float64, parent)


def test_v7_parent_is_rejected_even_if_it_is_loadable():
    parent = {"config": V7Config().as_dict(), "model": V7Model().state_dict()}
    with pytest.raises(ValueError, match="only a V6-family"):
        probe.model_for("seed1", 7, torch.device("cpu"), torch.float64, parent)
