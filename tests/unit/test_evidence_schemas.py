from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tabu_lab.evidence import Preregistration, generate_public_schema

ROOT = Path(__file__).resolve().parents[2]


def _preregistration_payload() -> dict[str, object]:
    return {
        "schema_version": "tabu-lab.preregistration.v1",
        "experiment_id": "example-contract",
        "status": "proposed",
        "created_at": "2026-08-30",
        "contract": {
            "contract_id": "tabu.compiler",
            "contract_version": "1",
            "maturity_required": "local",
        },
        "hypothesis": "A precommitted gate can be evaluated.",
        "claim_boundary": "A proposal is not evidence.",
        "data": {
            "kind": "synthetic",
            "rows": 16,
            "numeric_features": 2,
            "categorical_features": 0,
            "split_policy": "split_before_compile",
            "fit_partition": "train",
            "target_origin": "artificial_mask",
            "natural_missing_targets": "excluded",
        },
        "baseline": {"name": "constant", "budget": "matched"},
        "model_default": {
            "carrier": "cell",
            "dynamics": "omab",
            "numeric_terminal": "gaussian",
            "categorical_terminal": "categorical",
            "ll_terminal": "disabled",
        },
        "training": {
            "optimizer": "adamw",
            "max_steps": 10,
            "seeds": [1729],
            "device": "cpu",
            "dtype": "float32",
            "wall_clock_budget_minutes": 5,
        },
        "metrics": {"primary": "loss", "diagnostics": ["finite"]},
        "pass_conditions": ["loss_decreases"],
        "kill_conditions": ["non_finite"],
        "exit_conditions": {"pass": "review", "kill": "record_failure"},
        "required_artifacts": ["receipt"],
        "review": {
            "developer_and_reviewer_must_differ": True,
            "gong_approval_required_for_release": True,
        },
    }


def test_preregistration_is_strict_and_content_addressed() -> None:
    payload = _preregistration_payload()
    preregistration = Preregistration.model_validate(payload)

    assert preregistration.status == "proposed"
    assert len(preregistration.content_hash) == 64

    payload["unreviewed_extension"] = True
    with pytest.raises(ValidationError):
        Preregistration.model_validate(payload)


def test_checked_in_public_schemas_match_runtime_generation() -> None:
    for name in ("preregistration", "receipt", "claim", "source-identity"):
        path = ROOT / f"schemas/{name}.schema.json"
        checked_in = json.loads(path.read_text(encoding="utf-8"))

        assert checked_in == generate_public_schema(name)
