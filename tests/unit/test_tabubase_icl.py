from __future__ import annotations

import math

import pytest
import torch

import tabu_lab.experiments.tabubase_icl as icl_module
from tabu_lab.contracts import OriginState, origin_code
from tabu_lab.experiments.tabubase_icl import (
    FROZEN_ARMS,
    K_GRID,
    FrozenIclConfig,
    build_heldout_icl_case,
    paired_aulc,
    paired_world_bootstrap,
    run_frozen_icl,
)


class _TinyFrozenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def checkpoint_identity(self) -> dict[str, object]:
        return {"tokenizer_version": "cell-tokenizer.v2", "nominal_codebook_size": 100}


def test_heldout_icl_world_is_paired_nested_and_truth_free() -> None:
    small = build_heldout_icl_case(
        root_seed=1729,
        world_index=3,
        modality="classification",
        context_size=4,
    )
    large = build_heldout_icl_case(
        root_seed=1729,
        world_index=3,
        modality="classification",
        context_size=8,
    )
    shuffled = build_heldout_icl_case(
        root_seed=1729,
        world_index=3,
        modality="classification",
        context_size=4,
        shuffled_context=True,
    )
    assert small.world_id == large.world_id == shuffled.world_id
    assert torch.equal(small.query_values, large.query_values)
    assert torch.equal(small.query_values, shuffled.query_values)
    assert torch.equal(
        small.episode.forward_values[small.episode.target_mask],
        torch.zeros_like(small.query_values),
    )
    assert small.truth.target_count == 32
    assert small.episode.metadata["generator_family"] == "heteroscedastic_missingness_shift"
    natural = small.episode.origin_states == origin_code(OriginState.NATURAL_MISSING)
    assert bool(natural[:, :-1].any())
    assert not bool(natural[:, -1].any())


def test_zero_context_uses_no_query_statistics_or_visible_labels() -> None:
    case = build_heldout_icl_case(
        root_seed=1729,
        world_index=0,
        modality="regression",
        context_size=0,
        query_rows=7,
    )
    assert case.episode.forward_values.shape == (7, 9)
    assert int(case.episode.source_mask[:, -1].sum()) == 0
    assert case.truth.target_count == 7


def test_train_family_replay_is_distinct_and_has_no_missing_predictors() -> None:
    case = build_heldout_icl_case(
        root_seed=1729,
        world_index=2,
        modality="classification",
        context_size=8,
        world_scope="train_mixture",
    )
    assert case.episode.metadata["world_scope"] == "train_mixture"
    assert case.episode.metadata["generator_family"] == "latent_factor"
    assert not bool((case.episode.origin_states == origin_code(OriginState.NATURAL_MISSING)).any())


def test_aulc_and_world_bootstrap_are_deterministic() -> None:
    values = tuple(float(index) for index in range(len(K_GRID)))
    assert paired_aulc(values) == pytest.approx(3.0)
    first = paired_world_bootstrap((0.2, 0.4, 0.6), replicates=200, seed=1729)
    second = paired_world_bootstrap((0.2, 0.4, 0.6), replicates=200, seed=1729)
    assert first == second
    assert first[0] == pytest.approx(0.4)
    assert all(math.isfinite(value) for value in first)


def test_checkpoint_seed_and_world_seed_are_independent(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"fixture")
    config = FrozenIclConfig(
        checkpoint=checkpoint,
        output_dir=tmp_path / "out",
        seed=2718,
        world_seed=1729,
        heldout_worlds=4,
    )
    assert config.validate() is config
    assert config.seed != config.world_seed


def test_every_synthetic_frozen_arm_has_adjacent_hashes_and_no_optimizer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.safetensors"
    checkpoint.write_bytes(b"fixture")
    evaluated_models: list[torch.nn.Module] = []
    loaded_models: list[torch.nn.Module] = []

    monkeypatch.setattr(
        icl_module,
        "build_tabubase_scale_model",
        lambda **_kwargs: _TinyFrozenModel(),
    )
    monkeypatch.setattr(
        icl_module,
        "load_pretrain_checkpoint",
        lambda model, _path: loaded_models.append(model),
    )

    def fake_evaluate_arm(
        model: torch.nn.Module, *, worlds_per_modality: int, **_kwargs: object
    ) -> dict[str, dict[str, list[float]]]:
        assert torch.is_inference_mode_enabled()
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        evaluated_models.append(model)
        return {
            modality: {str(k): [1.0] * worlds_per_modality for k in K_GRID}
            for modality in ("classification", "regression")
        }

    def reject_optimizer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("frozen ICL must not construct an optimizer")

    monkeypatch.setattr(icl_module, "_evaluate_arm", fake_evaluate_arm)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", reject_optimizer)
    receipt = run_frozen_icl(
        FrozenIclConfig(
            checkpoint=checkpoint,
            output_dir=tmp_path / "out",
            heldout_worlds=2,
            bootstrap_replicates=100,
        ),
        device=torch.device("cpu"),
    )

    assert len({id(model) for model in evaluated_models}) == len(FROZEN_ARMS)
    assert {id(model) for model in loaded_models} == {
        id(evaluated_models[0]),
        id(evaluated_models[2]),
    }
    assert set(receipt["per_arm_parameter_hashes"]) == set(FROZEN_ARMS)
    for arm in FROZEN_ARMS:
        hashes = receipt["per_arm_parameter_hashes"][arm]
        assert hashes["before"] == hashes["after"]
        assert hashes["unchanged"] is True
    assert receipt["all_frozen_arm_parameter_hashes_unchanged"] is True
    assert receipt["parameter_hash_unchanged"] == dict.fromkeys(FROZEN_ARMS, True)
    assert receipt["frozen_arm_optimizer_created"] is False
