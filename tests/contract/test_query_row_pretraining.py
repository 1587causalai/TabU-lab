from __future__ import annotations

from pathlib import Path

import pytest

from tabu_lab.experiments import (
    load_query_row_pretrain_checkpoint,
    run_query_row_synthetic_pretraining,
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
    assert result.final_loss < result.initial_loss
    assert result.checkpoint == str(checkpoint)
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(".identity.json").is_file()

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

    monkeypatch.setattr(model, "load_state_dict", fail_if_called)
    with pytest.raises(ValueError, match="profile_id"):
        load_query_row_pretrain_checkpoint(model, checkpoint)


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

