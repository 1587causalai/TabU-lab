from __future__ import annotations

from pathlib import Path

from tabu_lab.experiments import query_row_r5_pretraining as r5


def test_r5_runner_is_non_overwriting_and_records_frozen_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(r5, "R5_RUNG_SPECS", {"B0": {"worlds": 4, "updates": 2}})
    result = r5.run_query_row_r5_bounded_pretraining(
        output_root=tmp_path / "r5",
        seeds=(1729,),
        rungs=("B0",),
        learning_rates=(1.0e-3,),
        pilot_worlds=4,
        pilot_updates=1,
        validation_worlds=2,
        device="cpu",
    )
    assert result["status"] == "passed"
    assert result["contract_version"] == "0.2.0"
    assert result["row_readout_mode"] == "anchored"
    assert result["row_readout_identity"]["mode"] == "anchored"
    assert len(result["variant_hash"]) == 64
    record = result["records"][0]
    assert record["variant_hash"] == result["variant_hash"]
    controls = record["synthetic_frozen_controls"]
    assert controls["hash_controls"]["pretrained_frozen"]["optimizer_created"] is False
    assert controls["hash_controls"]["same_init_random_frozen"]["parameter_hash_unchanged"]
    assert record["checkpoint"]["checkpoint"].endswith(".safetensors")
