from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file

from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments import (
    load_query_row_pretrain_checkpoint,
    run_query_row_frozen_icl,
    run_query_row_synthetic_pretraining,
)
from tabu_lab.experiments import query_row_pretraining as pretraining_module
from tabu_lab.experiments import query_row_transfer_common as transfer_module
from tabu_lab.experiments.query_row_identity import require_query_row_readout_identity
from tabu_lab.experiments.query_row_pretraining import (
    save_query_row_pretrain_checkpoint,
)
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig


def _config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=4,
        max_features=256,
    )


def test_tabur_synthetic_pretraining_saves_and_loads_profile_bound_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "tabur-completion.safetensors"
    result = run_query_row_synthetic_pretraining(
        profile="completion.artificial_mask.v1",
        rows=8,
        worlds=2,
        steps=8,
        row_token_count=4,
        output=checkpoint,
    )

    assert result.status == "pass"
    assert result.evidence_status == "local_unissued"
    assert result.contract_version == "0.2.0"
    assert result.row_readout_mode == "anchored"
    assert result.row_readout_identity["mode"] == "anchored"
    assert result.row_readout_identity["anchored_gamma_initial"] == pytest.approx(1.0e-2)
    assert result.checkpoint_kind == "weights_only_transfer_snapshot"
    assert result.training_resume_supported is False
    assert len(result.variant_hash) == 64
    assert result.final_loss < result.initial_loss
    assert result.checkpoint == str(checkpoint)
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".identity.json").is_file()
    identity = json.loads(checkpoint.with_suffix(".identity.json").read_text())
    assert identity["schema"] == "tabu.query-row-pretraining-checkpoint.v2"
    assert identity["model_identity"]["row_readout"] == result.row_readout_identity
    assert identity["metadata"]["row_readout_mode"] == "anchored"
    assert identity["metadata"]["row_readout_identity"] == result.row_readout_identity
    assert identity["metadata"]["variant_hash"] == result.variant_hash
    assert identity["metadata"]["checkpoint_kind"] == "weights_only_transfer_snapshot"
    assert identity["metadata"]["training_resume_supported"] is False

    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=4,
    )
    load_query_row_pretrain_checkpoint(model, checkpoint)


def test_tabur_pretraining_checkpoint_rejects_profile_mismatch_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "tabur-completion.safetensors"
    run_query_row_synthetic_pretraining(
        profile="completion.artificial_mask.v1",
        rows=8,
        worlds=2,
        steps=8,
        row_token_count=4,
        output=checkpoint,
    )
    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="supervised.label_broadcast.v1",
        row_token_count=4,
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor loading must not run after identity mismatch")

    monkeypatch.setattr(pretraining_module, "load_file", fail_if_called)
    monkeypatch.setattr(model, "load_state_dict", fail_if_called)
    with pytest.raises(ValueError, match="profile_id"):
        load_query_row_pretrain_checkpoint(model, checkpoint)


def test_tabur_checkpoint_rejects_readout_mode_mismatch_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "tabur-anchored.safetensors"
    run_query_row_synthetic_pretraining(
        rows=8,
        worlds=2,
        steps=2,
        row_token_count=4,
        row_readout_mode="anchored",
        output=checkpoint,
    )
    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=4,
        row_readout_mode="homogeneous",
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor loading must not run after readout mismatch")

    monkeypatch.setattr(pretraining_module, "load_file", fail_if_called)
    monkeypatch.setattr(model, "load_state_dict", fail_if_called)
    with pytest.raises(ValueError, match="row_readout"):
        load_query_row_pretrain_checkpoint(model, checkpoint)


def test_tabur_checkpoint_v1_is_not_migrated_or_defaulted_during_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "tabur-legacy.safetensors"
    run_query_row_synthetic_pretraining(
        rows=8,
        worlds=2,
        steps=2,
        row_token_count=4,
        output=checkpoint,
    )
    identity_path = checkpoint.with_suffix(".identity.json")
    identity = json.loads(identity_path.read_text())
    identity["schema"] = "tabu.query-row-pretraining-checkpoint.v1"
    identity["identity_hash"] = canonical_hash(
        {key: value for key, value in identity.items() if key != "identity_hash"}
    )
    identity_path.write_text(json.dumps(identity))

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("legacy identity must fail before model/tensor construction")

    monkeypatch.setattr(transfer_module, "build_model", fail_if_called)
    monkeypatch.setattr(pretraining_module, "load_file", fail_if_called)
    with pytest.raises(ValueError, match="legacy v1 checkpoints are not migrated"):
        transfer_module.build_model_from_checkpoint(checkpoint, device=torch.device("cpu"))


def test_tabur_checkpoint_embedded_readout_is_required_before_tensor_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "tabur-missing-embedded-readout.safetensors"
    run_query_row_synthetic_pretraining(
        rows=8,
        worlds=2,
        steps=2,
        row_token_count=4,
        output=checkpoint,
    )
    tensors = load_safetensors_file(str(checkpoint), device="cpu")
    identity = json.loads(checkpoint.with_suffix(".identity.json").read_text())
    del identity["model_identity"]["row_readout"]
    identity["identity_hash"] = canonical_hash(
        {key: value for key, value in identity.items() if key != "identity_hash"}
    )
    save_safetensors_file(
        tensors,
        str(checkpoint),
        metadata={"identity": json.dumps(identity, sort_keys=True)},
    )
    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=4,
    )

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("tensor loading must not run without embedded row_readout")

    monkeypatch.setattr(pretraining_module, "load_file", fail_if_called)
    monkeypatch.setattr(model, "load_state_dict", fail_if_called)
    with pytest.raises(ValueError, match=r"model_identity\.row_readout is required"):
        load_query_row_pretrain_checkpoint(model, checkpoint)


def test_tabur_checkpoint_reconstruction_uses_exact_readout_identity(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tabur-custom-gamma.safetensors"
    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=4,
        row_readout_mode="anchored",
        anchored_gamma_initial=0.125,
    )
    save_query_row_pretrain_checkpoint(model, checkpoint, metadata={"purpose": "test"})

    reconstructed, identity = transfer_module.build_model_from_checkpoint(
        checkpoint, device=torch.device("cpu")
    )

    assert reconstructed.row_readout_mode.value == "anchored"
    assert reconstructed.anchored_gamma_initial == pytest.approx(0.125)
    assert identity["metadata"]["row_readout_mode"] == "anchored"
    assert identity["metadata"]["row_readout_identity"][
        "anchored_gamma_initial"
    ] == pytest.approx(0.125)


def test_tabur_nonanchored_identity_rejects_noncanonical_gamma_sentinel() -> None:
    model = build_model(
        "tabu.query.row",
        config=_config(),
        profile="completion.artificial_mask.v1",
        row_token_count=4,
        row_readout_mode="homogeneous",
    )
    identity = model.checkpoint_identity()
    identity["row_readout"] = dict(identity["row_readout"])
    identity["row_readout"]["anchored_gamma_initial"] = 0.125

    with pytest.raises(ValueError, match=r"canonical anchored_gamma_initial=0\.01"):
        require_query_row_readout_identity(identity)


def test_tabur_supervised_pretraining_uses_a_distinct_profile_identity() -> None:
    completion = run_query_row_synthetic_pretraining(
        profile="completion.artificial_mask.v1",
        rows=8,
        worlds=2,
        steps=8,
        row_token_count=4,
    )
    supervised = run_query_row_synthetic_pretraining(
        profile="supervised.label_broadcast.v1",
        rows=8,
        worlds=2,
        steps=8,
        row_token_count=4,
    )

    assert completion.status == supervised.status == "pass"
    assert completion.profile_id != supervised.profile_id
    assert completion.model_spec_hash == supervised.model_spec_hash


def test_tabur_frozen_icl_can_bind_a_completion_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "tabur-completion.safetensors"
    run_query_row_synthetic_pretraining(
        profile="completion.artificial_mask.v1",
        rows=8,
        worlds=2,
        steps=2,
        row_token_count=4,
        output=checkpoint,
    )

    result = run_query_row_frozen_icl(
        seed=1729,
        rows=12,
        context_rows=(4, 8),
        row_token_count=4,
        checkpoint=checkpoint,
    )

    assert result.status == "pass"
    assert result.checkpoint == str(checkpoint)
    assert result.eval_worlds == 1
    assert all(record.parameter_hash_unchanged for record in result.records)
    assert all(not record.optimizer_created for record in result.records)
