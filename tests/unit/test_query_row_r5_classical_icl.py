from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch


def test_r5_classical_runner_reuses_panel_and_keeps_frozen_controls(
    monkeypatch, tmp_path: Path
) -> None:
    module = importlib.import_module("tabu_lab.experiments.query_row_r5_classical_icl")
    checkpoint = tmp_path / "model.safetensors"
    identity = checkpoint.with_suffix(".identity.json")
    checkpoint.write_bytes(b"checkpoint")
    identity.write_text("{}\n", encoding="utf-8")

    calls = {"baseline": 0, "tabur": 0}

    def fake_baseline(episode, *, seed):
        calls["baseline"] += 1
        return {
            "target_count": episode.evidence.forward_values.shape[0] - episode.context_rows,
            "truth": None,
            "response_scale": 1.0,
            "predictions": {},
            "metrics": {
                "linear": {
                    "raw_response_mse": 2.0,
                    "context_standardized_response_mse": 2.0,
                },
                "mlp": {
                    "raw_response_mse": 3.0,
                    "context_standardized_response_mse": 3.0,
                },
                "xgboost": {
                    "raw_response_mse": 4.0,
                    "context_standardized_response_mse": 4.0,
                },
            },
        }

    def fake_checkpoint_model(path, *, device):
        return torch.nn.Identity(), {
            "model_identity": {
                "model_id": "tabu.query.row",
                "contract_version": "0.2.0",
                "profile_id": "supervised.label_broadcast.v1",
                "variant_hash": "b" * 64,
                "variant_ref": {
                    "contract_id": "tabu.query.row",
                    "contract_version": "0.2.0",
                    "profile_id": "supervised.label_broadcast.v1",
                    "model_spec_hash": "a" * 64,
                },
                "row_token_count": 4,
                "reference_config": {"matched_slots": 4},
                "row_readout": {
                    "schema_version": "tabu.query-row-readout.v1",
                    "mode": "anchored",
                    "beta": 1.0,
                    "anchored_gamma_initial": 0.01,
                    "axis_transform_normalization": "exact_spectral_norm_v1",
                    "row_token_count": 4,
                    "global_w_rows": 4,
                },
            },
            "metadata": {
                "rung": "B1",
                "root_seed": 1729,
                "worlds": 2048,
                "updates": 6000,
                "learning_rate": 0.003,
            }
        }

    def fake_tabur(model, episode):
        calls["tabur"] += 1
        return {
            "raw_response_mse": 1.0,
            "context_standardized_response_mse": 1.0,
        }, True

    monkeypatch.setattr(module, "_baseline_world", fake_baseline)
    monkeypatch.setattr(module, "_checkpoint_model", fake_checkpoint_model)
    monkeypatch.setattr(module, "_tabur_world", fake_tabur)
    monkeypatch.setattr(module, "_state_hash", lambda model: "stable")

    result = module.run_query_row_r5_classical_icl(
        checkpoints=(checkpoint,),
        panel_root_seed=502729,
        panel_worlds=3,
        device="cpu",
    )

    assert result.status == "passed"
    assert calls == {"baseline": 3, "tabur": 3}
    checkpoint_result = result.checkpoints[0]
    assert checkpoint_result.status == "passed"
    assert checkpoint_result.parameter_hash_unchanged
    assert checkpoint_result.truth_substitution_prediction_unchanged
    assert checkpoint_result.optimizer_created is False
    assert checkpoint_result.parameter_update_attempted is False
    assert checkpoint_result.contract_version == "0.2.0"
    assert checkpoint_result.model_spec_hash == "a" * 64
    assert checkpoint_result.variant_hash == "b" * 64
    assert checkpoint_result.row_readout_mode == "anchored"
    assert checkpoint_result.row_readout_identity["anchored_gamma_initial"] == 0.01
    assert checkpoint_result.aggregate_metrics["tabur"]["raw_response_mse"] == 1.0
    payload = result.as_dict()
    assert payload["baseline_ids"] == [
        "ordinary_least_squares.context_only.v1",
        "mlp_regressor.context_only.v1",
        "xgboost_regressor.context_only.v1",
    ]
    assert json.loads(json.dumps(payload))["panel_worlds"] == 3
