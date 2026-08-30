from __future__ import annotations

import json

import pytest
import torch

from tabu_lab.contracts import FeatureKind, FeatureRole
from tabu_lab.experiments.tabubase_scale import (
    EXPANDED_SYNTHETIC_GENERATOR_VERSION,
    LONG_CONTEXT_CANDIDATE_ROWS,
    LONG_CONTEXT_ROWS_SCHEDULE,
    QUERY_RESPONSE_TRAINING_FORWARD_MODE,
    PretrainRunConfig,
    SyntheticEpisodePrefetcher,
    _git_commit_or_none,
    _train_one,
    build_synthetic_episode,
    build_tabubase_scale_model,
    load_pretrain_checkpoint,
    pretrain_run_id,
    save_pretrain_checkpoint,
    source_tree_sha256,
)


def test_scale_episode_is_deterministic_mixed_type_and_truth_isolated() -> None:
    episode, truth, metadata = build_synthetic_episode(root_seed=1729, world_index=7)
    replay, replay_truth, replay_metadata = build_synthetic_episode(root_seed=1729, world_index=7)
    assert episode.evidence_hash == replay.evidence_hash
    assert truth.truth_hash == replay_truth.truth_hash
    assert metadata == replay_metadata
    assert episode.feature_specs[-1].role is FeatureRole.RESPONSE
    assert {spec.kind for spec in episode.feature_specs[:-1]} == {
        FeatureKind.NUMERIC,
        FeatureKind.ORDINAL,
        FeatureKind.CATEGORICAL,
    }
    assert torch.equal(episode.forward_values[episode.target_mask], torch.zeros(64))
    assert truth.target_count == 64


def test_scale_episode_supports_explicit_zero_context_icl_condition() -> None:
    episode, truth, metadata = build_synthetic_episode(
        root_seed=1729,
        world_index=7,
        context_rows=0,
        query_rows=16,
    )
    assert metadata["world_index"] == 7
    assert episode.target_mask.shape == (16, 9)
    assert int(episode.target_mask[:, -1].sum()) == 16
    assert truth.target_count == 16


def test_scale_phase_budgets_fail_closed() -> None:
    PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
    ).validate()
    try:
        PretrainRunConfig(
            phase="PT-S1",
            worlds=2_048,
            updates=2_000,
            seed=1729,
            checkpoint_updates=(0, 2_000),
        ).validate()
    except ValueError as exc:
        assert "frozen" in str(exc)
    else:
        raise AssertionError("PT-S1 accepted the pilot budget")


def test_scale_config_accepts_explicit_s2_budget() -> None:
    config = PretrainRunConfig(
        phase="PT-S2",
        worlds=200_000,
        updates=200_000,
        seed=1729,
        checkpoint_updates=(0, 20_000, 50_000, 100_000, 150_000, 200_000),
    )
    assert config.validate() is config


def test_scale_codebook_v2_has_distinct_100_category_run_identity() -> None:
    config = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
    )
    assert config.validate() is config
    assert pretrain_run_id(config).endswith("nominal-codebook-v2-b100-s1729")
    model = build_tabubase_scale_model(
        seed=1729,
        device=torch.device("cpu"),
        nominal_tokenizer=config.nominal_tokenizer,
    )
    identity = model.checkpoint_identity()
    assert identity["tokenizer_version"] == "cell-tokenizer.v2"
    assert identity["nominal_codebook_size"] == 100


def test_scale_variable_k_curriculum_has_distinct_identity() -> None:
    config = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        context_rows_schedule=(2, 4, 8, 16, 32, 64),
    )
    assert config.validate() is config
    assert pretrain_run_id(config).endswith("-icl-kcurriculum-v1")


def test_expanded_generator_has_distinct_identity_and_preserves_legacy_default() -> None:
    legacy = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        context_rows_schedule=(2, 4, 8, 16, 32, 64),
    )
    expanded = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        context_rows_schedule=(2, 4, 8, 16, 32, 64),
        generator_version=EXPANDED_SYNTHETIC_GENERATOR_VERSION,
        validation_worlds=192,
    )
    assert legacy.validate() is legacy
    assert expanded.validate() is expanded
    assert pretrain_run_id(legacy).endswith("-icl-kcurriculum-v1")
    assert pretrain_run_id(expanded).endswith(
        "-icl-kcurriculum-v1-expanded-synthetic-v4-val192-support-k-v1"
    )


def test_expanded_long_context_protocol_has_collision_free_validated_identity() -> None:
    config = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        context_rows_schedule=LONG_CONTEXT_ROWS_SCHEDULE,
        generator_version=EXPANDED_SYNTHETIC_GENERATOR_VERSION,
        validation_worlds=192,
        context_candidate_rows=LONG_CONTEXT_CANDIDATE_ROWS,
        training_forward_mode=QUERY_RESPONSE_TRAINING_FORWARD_MODE,
        query_readout_chunk_rows=64,
    )
    assert config.validate() is config
    assert pretrain_run_id(config).endswith(
        "-expanded-synthetic-v4-val192-support-k-v1-"
        "long-context-v1-k512-query-response-v1"
    )

    mixed_protocol = PretrainRunConfig(
        phase="PT-S0",
        worlds=2_048,
        updates=2_000,
        seed=1729,
        checkpoint_updates=(0, 2_000),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        context_rows_schedule=LONG_CONTEXT_ROWS_SCHEDULE,
        generator_version=EXPANDED_SYNTHETIC_GENERATOR_VERSION,
        validation_worlds=192,
    )
    try:
        mixed_protocol.validate()
    except ValueError as exc:
        assert "protocol" in str(exc) or "candidate bank" in str(exc)
    else:
        raise AssertionError("mixed dense/long-context protocol was accepted")


def test_expanded_generator_episode_has_finite_training_step() -> None:
    from tabu_lab.experiments.tabubase_expanded_synthetic import (
        build_expanded_synthetic_episode,
    )

    episode, truth, metadata = build_expanded_synthetic_episode(
        root_seed=1729,
        world_index=48,
        context_rows=16,
        query_rows=16,
    )
    assert metadata["schema_profile"] == "mixed"
    model = build_tabubase_scale_model(
        seed=1729,
        device=torch.device("cpu"),
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
    loss = _train_one(
        model,
        optimizer,
        episode,
        truth,
        device=torch.device("cpu"),
        gradient_clip_norm=1.0,
    )
    assert torch.isfinite(torch.tensor(loss))


def test_missing_git_binary_does_not_destroy_local_unissued_receipt(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TABU_SOURCE_COMMIT", raising=False)

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("tabu_lab.experiments.tabubase_scale.subprocess.run", missing_git)
    assert _git_commit_or_none() is None


def test_checkpoint_load_requires_matching_embedded_and_sidecar_identity(tmp_path) -> None:
    model = build_tabubase_scale_model(seed=1729, device=torch.device("cpu"))
    identity = {
        "schema_version": "tabu.transfer-base-local-unissued-checkpoint.v1",
        "model_identity": model.checkpoint_identity(),
        "update": 0,
    }
    path = tmp_path / "checkpoint.safetensors"
    save_pretrain_checkpoint(model, path, identity=identity)
    target = build_tabubase_scale_model(seed=2718, device=torch.device("cpu"))
    load_pretrain_checkpoint(target, path)

    sidecar = path.with_suffix(".identity.json")
    drifted = json.loads(sidecar.read_text(encoding="utf-8"))
    drifted["update"] = 1
    sidecar.write_text(json.dumps(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match embedded identity"):
        load_pretrain_checkpoint(target, path)


def test_checkpoint_load_rejects_missing_identity_sidecar(tmp_path) -> None:
    model = build_tabubase_scale_model(seed=1729, device=torch.device("cpu"))
    path = tmp_path / "checkpoint.safetensors"
    save_pretrain_checkpoint(
        model,
        path,
        identity={"model_identity": model.checkpoint_identity()},
    )
    path.with_suffix(".identity.json").unlink()
    with pytest.raises(FileNotFoundError, match="identity sidecar is required"):
        load_pretrain_checkpoint(model, path)


def test_source_tree_identity_binds_runner_config_schema_and_lock(tmp_path) -> None:
    paths = (
        tmp_path / "src/tabu_lab/module.py",
        tmp_path / "specs/models/model.yaml",
        tmp_path / "scripts/run_tabubase_scale_transfer.py",
        tmp_path / "experiments/transfer-base-v2/protocol.yaml",
        tmp_path / "schemas/tabubase-synthetic-world.schema.json",
        tmp_path / "pyproject.toml",
        tmp_path / "uv.lock",
    )
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n", encoding="utf-8")
    expected = source_tree_sha256(tmp_path)
    for path in paths:
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "drift\n", encoding="utf-8")
        assert source_tree_sha256(tmp_path) != expected
        path.write_text(original, encoding="utf-8")


def test_episode_prefetcher_preserves_world_order_and_hashes() -> None:
    with SyntheticEpisodePrefetcher(
        root_seed=1729,
        worlds=20_000,
        first_update=1,
        last_update=3,
        workers=2,
        queue_depth=2,
        context_rows_schedule=(2, 4, 8),
    ) as prefetcher:
        observed = [prefetcher.next() for _ in range(3)]
    expected = [
        build_synthetic_episode(
            root_seed=1729,
            world_index=((update - 1) * 7919 + 1729) % 20_000,
            context_rows=(2, 4, 8)[update - 1],
        )
        for update in range(1, 4)
    ]
    assert [item[0].evidence_hash for item in observed] == [
        item[0].evidence_hash for item in expected
    ]
    assert [item[1].truth_hash for item in observed] == [item[1].truth_hash for item in expected]


def test_training_fast_forward_preserves_predictions_and_omits_trace() -> None:
    episode, _, _ = build_synthetic_episode(root_seed=1729, world_index=7)
    model = build_tabubase_scale_model(seed=1729, device=torch.device("cpu"))
    full = model(episode)
    fast = model._forward_dense(episode.to("cpu"), emit_trace=False)
    assert full.trace is not None
    assert fast.trace is None
    assert full.entries.keys() == fast.entries.keys()
    for name in full.entries:
        full_values = full.entries[name].values
        fast_values = fast.entries[name].values
        if full_values is None or fast_values is None:
            assert full_values is fast_values
        else:
            torch.testing.assert_close(full_values, fast_values, rtol=1e-6, atol=1e-7)
    for name in (
        "target_mask",
        "support_available",
        "numeric_target_mask",
        "categorical_target_mask",
    ):
        torch.testing.assert_close(full.auxiliaries[name], fast.auxiliaries[name])
