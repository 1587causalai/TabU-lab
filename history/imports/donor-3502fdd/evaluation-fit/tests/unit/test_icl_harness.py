from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tabu_lab.experiments.transfer import IclArm, IclHarnessSpec

YAML_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "transfer-v1" / "icl-harness.yaml"
)


def _payload() -> dict:
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


def test_icl_harness_spec_parses_from_yaml() -> None:
    spec = IclHarnessSpec.model_validate(_payload())
    assert spec.harness_id == "icl-harness-tabul-v1"
    assert set(spec.arms) == set(IclArm)
    assert spec.context_sizes[0] == 0
    assert len(spec.heldout_world_families) >= 1


def test_icl_harness_spec_fails_closed() -> None:
    base = _payload()
    with pytest.raises(ValueError):
        IclHarnessSpec.model_validate({**base, "arms": ["icl_pretrained"]})
    with pytest.raises(ValueError):
        IclHarnessSpec.model_validate({**base, "context_sizes": [8, 4, 2]})
    with pytest.raises(ValueError):
        IclHarnessSpec.model_validate({**base, "pretrain_spec_sha256": "deadbeef"})
