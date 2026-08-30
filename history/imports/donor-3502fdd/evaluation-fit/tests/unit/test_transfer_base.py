from __future__ import annotations

from pathlib import Path

import pytest

from tabu_lab.experiments.transfer_base import (
    BaseFineTuneSpec,
    BaseIclSpec,
    BasePretrainSpec,
    load_finetune_spec,
    load_icl_spec,
    load_pretrain_spec,
)


ROOT = Path(__file__).resolve().parents[2] / "experiments" / "transfer-base-v1"


def test_base_transfer_v2_manifests_bind_profile_and_pt_s0_gate() -> None:
    pretrain = load_pretrain_spec(ROOT / "pretrain.yaml")
    icl = load_icl_spec(ROOT / "icl-harness.yaml")
    finetune = load_finetune_spec(ROOT / "finetune-template.yaml")

    assert isinstance(pretrain, BasePretrainSpec)
    assert (pretrain.pilot_worlds, pretrain.pilot_updates, pretrain.pilot_seeds) == (
        2_048,
        2_000,
        (1729,),
    )
    assert pretrain.predictor_kinds == ("numeric", "ordinal", "nominal")
    assert pretrain.world_split_before_masking is True
    assert pretrain.world_split_before_statistics is True
    assert isinstance(icl, BaseIclSpec)
    assert icl.heldout_worlds == 512
    assert isinstance(finetune, BaseFineTuneSpec)
    assert {task.task_id for task in finetune.tasks} == {
        "adult_classification",
        "diabetes_regression",
    }
    for spec in (pretrain, icl, finetune):
        assert spec.reference.contract_id == "tabu.cell.base"
        assert spec.reference.contract_version == "0.2.0"
        assert spec.reference.profile_id == "supervised.label_broadcast.v1"


def test_base_pretrain_loader_fails_closed_when_pt_s0_gate_drifts(tmp_path: Path) -> None:
    source = (ROOT / "pretrain.yaml").read_text(encoding="utf-8")
    path = tmp_path / "pretrain.yaml"
    path.write_text(source.replace("updates: 2000", "updates: 1999"), encoding="utf-8")
    with pytest.raises(ValueError, match="PT-S0"):
        load_pretrain_spec(path)
