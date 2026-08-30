from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POLICY = runpy.run_path(str(ROOT / "scripts" / "verify_distribution.py"))
_reject_unsafe = POLICY["_reject_unsafe"]
_require = POLICY["_require"]
_wheel_required_members = POLICY["_wheel_required_members"]


def test_wheel_policy_requires_every_runtime_source_module() -> None:
    required = _wheel_required_members()
    representative = {
        "tabu_lab/compiler/episode.py",
        "tabu_lab/contracts/episode.py",
        "tabu_lab/evaluation/evaluator.py",
        "tabu_lab/evidence/receipt_io.py",
        "tabu_lab/models/reference.py",
        "tabu_lab/primitives/oattention.py",
        "tabu_lab/training/trainer.py",
    }
    assert representative <= required
    assert "tabu_lab/specs/models/tabu4rec/0.2.0.yaml" in required
    with pytest.raises(SystemExit, match="missing required public material"):
        _require(required - {"tabu_lab/models/reference.py"}, required, label="mutated wheel")


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "checkpoint.ckpt",
        "weights.pt",
        "weights.pth",
        "weights.bin",
        "estimator.joblib",
        "payload.dill",
        "payload.pickle",
        "payload.pkl",
    ],
)
def test_distribution_policy_rejects_pickle_bearing_artifact_suffixes(
    unsafe_name: str,
) -> None:
    with pytest.raises(SystemExit, match="unsafe serialized artifacts"):
        _reject_unsafe({f"tabu_lab/artifacts/{unsafe_name}"}, label="candidate")
