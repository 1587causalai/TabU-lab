from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import tabu_lab.experiments.tabubase_real_icl as real_icl_module
from tabu_lab.contracts import OriginState, origin_code
from tabu_lab.experiments.tabubase_icl import FROZEN_ARMS
from tabu_lab.experiments.tabubase_real_benchmark import RealDataset
from tabu_lab.experiments.tabubase_real_icl import (
    FULL_CONTEXT_POLICY,
    LOW_SHOT_CONTEXT_POLICY,
    REAL_FULL_CONTEXT_SCHEMA,
    RealIclConfig,
    build_real_icl_episode,
    prepare_real_icl_split,
    run_real_frozen_icl,
)


class _TinyFrozenModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))


def _fixture(task: str = "classification") -> RealDataset:
    rng = np.random.default_rng(7)
    if task == "classification":
        response = np.repeat(np.arange(3), 60)
    else:
        response = rng.normal(size=180).astype(np.float32)
    return RealDataset(
        dataset_id=f"real-icl-{task}-fixture",
        task=task,  # type: ignore[arg-type]
        features=rng.normal(size=(180, 70)).astype(np.float32),
        response=response,
        source="fixture",
    )


def _small_fixture(task: str) -> RealDataset:
    rng = np.random.default_rng(19)
    response = (
        np.repeat(np.arange(3), 6)
        if task == "classification"
        else np.linspace(-1.0, 1.0, num=18, dtype=np.float32)
    )
    return RealDataset(
        dataset_id=f"small-real-icl-{task}",
        task=task,  # type: ignore[arg-type]
        features=rng.normal(size=(18, 3)).astype(np.float32),
        response=response,
        source="fixture",
    )


def test_real_icl_split_is_nested_class_covered_and_train_only() -> None:
    split = prepare_real_icl_split(_fixture(), split_seed=1729, query_limit=48)
    assert split.features.shape[1] == 63
    assert not set(split.train_indices) & set(split.query_indices)
    assert set(split.response[split.context_order[:3]].tolist()) == {0, 1, 2}
    replay = prepare_real_icl_split(_fixture(), split_seed=1729)
    assert np.array_equal(
        split.context_order[:4],
        replay.context_order[:4],
    )
    assert len(replay.query_indices) == len(_fixture().response) - len(replay.train_indices)


def test_real_icl_episode_hides_query_truth_and_shuffle_only_changes_context_labels() -> None:
    split = prepare_real_icl_split(_fixture(), split_seed=2718, query_limit=32)
    normal, truth = build_real_icl_episode(
        split,
        context_size=8,
        query_indices=split.query_indices[:7],
        shuffled_context=False,
        context_policy=LOW_SHOT_CONTEXT_POLICY,
    )
    shuffled, shuffled_truth = build_real_icl_episode(
        split,
        context_size=8,
        query_indices=split.query_indices[:7],
        shuffled_context=True,
        context_policy=LOW_SHOT_CONTEXT_POLICY,
    )
    assert normal.forward_values.shape[1] == 64
    assert torch.count_nonzero(normal.forward_values[8:, -1]) == 0
    assert torch.all(normal.origin_states[8:, -1] == origin_code(OriginState.QUERY))
    assert torch.equal(truth.target_values, shuffled_truth.target_values)
    assert torch.equal(normal.forward_values[:, :-1], shuffled.forward_values[:, :-1])
    assert not torch.equal(normal.forward_values[:8, -1], shuffled.forward_values[:8, -1])


def test_full_context_episode_exposes_every_train_label_and_hides_every_query_truth() -> None:
    split = prepare_real_icl_split(_fixture(), split_seed=2718, query_limit=None)
    context_size = len(split.train_indices)
    evidence, truth = build_real_icl_episode(
        split,
        context_size=context_size,
        query_indices=split.query_indices,
        shuffled_context=False,
        context_policy=FULL_CONTEXT_POLICY,
    )
    expected_context_ids = tuple(
        f"{split.dataset.dataset_id}-row-{int(index)}" for index in split.context_order
    )
    assert evidence.row_ids[:context_size] == expected_context_ids
    assert evidence.metadata["context_policy"] == FULL_CONTEXT_POLICY
    assert evidence.metadata["context_size"] == len(split.train_indices)
    assert torch.count_nonzero(evidence.forward_values[context_size:, -1]) == 0
    assert torch.all(evidence.origin_states[context_size:, -1] == origin_code(OriginState.QUERY))
    assert torch.equal(
        truth.target_values[context_size:, -1],
        torch.as_tensor(split.response[split.query_indices], dtype=torch.float32),
    )


def test_full_context_config_rejects_query_truncation(tmp_path: Path) -> None:
    config = RealIclConfig(
        checkpoint_root=tmp_path,
        output_path=tmp_path / "result.json",
        context_policy=FULL_CONTEXT_POLICY,
        query_limit=128,
        bootstrap_replicates=100,
    )
    with pytest.raises(ValueError, match="every held-out query row"):
        config.validate()


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_full_context_query_only_readout_matches_dense_reference(task: str) -> None:
    split = prepare_real_icl_split(_small_fixture(task), split_seed=1729, query_limit=None)
    context_size = len(split.train_indices)
    evidence, _ = build_real_icl_episode(
        split,
        context_size=context_size,
        query_indices=split.query_indices,
        shuffled_context=False,
        context_policy=FULL_CONTEXT_POLICY,
    )
    model = real_icl_module.build_tabubase_scale_model(
        seed=1729,
        device=torch.device("cpu"),
    )
    model.requires_grad_(False)
    model.eval()
    with torch.inference_mode():
        dense = model._forward_dense(evidence.to("cpu"), emit_trace=False)
    probabilities, predicted = real_icl_module._forward_full_context_response(
        model,
        evidence,
        context_size=context_size,
        classes=split.classes,
        query_readout_chunk_rows=2,
        device=torch.device("cpu"),
    )
    one_row_probabilities, one_row_predicted = real_icl_module._forward_full_context_response(
        model,
        evidence,
        context_size=context_size,
        classes=split.classes,
        query_readout_chunk_rows=1,
        device=torch.device("cpu"),
    )

    if task == "classification":
        assert probabilities is not None and predicted is None
        dense_values = dense.entries["distribution"].values
        assert dense_values is not None and split.classes is not None
        expected = dense_values[context_size:, -1, : split.classes].detach().cpu().numpy()
        expected = np.clip(expected.astype(np.float64), 1.0e-8, None)
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(probabilities, expected, rtol=1.0e-5, atol=1.0e-7)
        assert one_row_probabilities is not None and one_row_predicted is None
        np.testing.assert_allclose(probabilities, one_row_probabilities, rtol=1.0e-6, atol=1.0e-7)
    else:
        assert probabilities is None and predicted is not None
        dense_values = dense.entries["numeric"].values
        assert dense_values is not None
        expected = dense_values[context_size:, -1].detach().cpu().numpy()
        np.testing.assert_allclose(predicted, expected, rtol=1.0e-5, atol=1.0e-7)
        assert one_row_probabilities is None and one_row_predicted is not None
        np.testing.assert_allclose(predicted, one_row_predicted, rtol=1.0e-6, atol=1.0e-7)


def test_every_real_frozen_arm_has_adjacent_hashes_and_no_optimizer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = RealIclConfig(
        checkpoint_root=tmp_path / "checkpoints",
        output_path=tmp_path / "result.json",
        dataset_ids=("fixture",),
        checkpoint_seeds=(1729,),
        split_seeds=(1729,),
        context_policy=FULL_CONTEXT_POLICY,
        query_limit=None,
        bootstrap_replicates=100,
    )
    checkpoint = real_icl_module._checkpoint_path(config, 1729)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    evaluated_models: list[torch.nn.Module] = []
    loaded_models: list[torch.nn.Module] = []

    monkeypatch.setattr(real_icl_module, "load_real_dataset", lambda _dataset_id: _fixture())
    monkeypatch.setattr(
        real_icl_module,
        "build_tabubase_scale_model",
        lambda **_kwargs: _TinyFrozenModel(),
    )
    monkeypatch.setattr(
        real_icl_module,
        "load_pretrain_checkpoint",
        lambda model, _path: loaded_models.append(model),
    )
    monkeypatch.setattr(real_icl_module, "_source_tree_hash", lambda: "source-tree-fixture")

    def fake_evaluate(
        model: torch.nn.Module, *_args: object, **_kwargs: object
    ) -> dict[str, float]:
        assert torch.is_inference_mode_enabled()
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        evaluated_models.append(model)
        return {"normalized_nll": 1.0, "accuracy": 1.0 / 3.0}

    def reject_optimizer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real frozen ICL must not construct an optimizer")

    monkeypatch.setattr(real_icl_module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", reject_optimizer)
    receipt = run_real_frozen_icl(config, device=torch.device("cpu"))

    distinct_evaluated_models = {id(model) for model in evaluated_models}
    assert len(distinct_evaluated_models) == len(FROZEN_ARMS)
    assert {id(model) for model in loaded_models}.issubset(distinct_evaluated_models)
    assert len({id(model) for model in loaded_models}) == 2
    checkpoint_hashes = receipt["per_arm_parameter_hashes"]["1729"]
    assert set(checkpoint_hashes) == set(FROZEN_ARMS)
    for arm in FROZEN_ARMS:
        hashes = checkpoint_hashes[arm]
        assert hashes["before"] == hashes["after"]
        assert hashes["unchanged"] is True
    assert receipt["all_frozen_arm_parameter_hashes_unchanged"] is True
    assert receipt["all_parameter_hashes_unchanged"] is True
    assert receipt["frozen_arm_optimizer_created"] is False
    assert receipt["schema_version"] == REAL_FULL_CONTEXT_SCHEMA
    assert receipt["context_policy"] == FULL_CONTEXT_POLICY
    assert receipt["context_sizes"] is None
    assert receipt["query_limit"] is None
    assert receipt["query_policy"] == "all_heldout_rows"
    assert receipt["query_chunk_semantics"] == (
        "response_readout_only_after_one_full_transductive_evidence_episode"
    )
    assert len(receipt["records"]) == len(FROZEN_ARMS)
    for row in receipt["records"]:
        assert row["full_context"] is True
        assert row["context_size"] == row["train_rows_total"]


def test_openml_new6_adapter_binds_manifest_and_materialization_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[2]
    panel_path = root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
    config = RealIclConfig(
        checkpoint_root=tmp_path / "checkpoints",
        output_path=tmp_path / "result.json",
        dataset_ids=("banknote_authentication",),
        checkpoint_seeds=(1729,),
        split_seeds=(1729, 2718, 31415),
        context_policy=LOW_SHOT_CONTEXT_POLICY,
        query_limit=256,
        query_chunk_rows=64,
        bootstrap_replicates=100,
        panel_manifest_path=panel_path,
    )
    checkpoint = real_icl_module._checkpoint_path(config, 1729)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    rows = 1_372
    features = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)
    target = np.asarray(["zeta" if index % 2 == 0 else "alpha" for index in range(rows)])
    bunch = SimpleNamespace(
        data=features,
        target=target,
        details={
            "id": "1462",
            "name": "banknote-authentication",
            "version": "1",
            "md5_checksum": "baa2dc5b745775a943ebeb9c276401f8",
            "licence": "Public",
            "status": "active",
            "default_target_attribute": "class",
        },
    )
    fetch_calls: list[dict[str, Any]] = []

    def fake_fetcher(**kwargs: Any) -> SimpleNamespace:
        fetch_calls.append(kwargs)
        return bunch

    def reject_old6_loader(_dataset_id: str) -> RealDataset:
        raise AssertionError("explicit OpenML new6 must not enter the old6 loader")

    monkeypatch.setattr(real_icl_module, "load_real_dataset", reject_old6_loader)
    monkeypatch.setattr(
        real_icl_module,
        "build_tabubase_scale_model",
        lambda **_kwargs: _TinyFrozenModel(),
    )
    monkeypatch.setattr(real_icl_module, "load_pretrain_checkpoint", lambda *_args: None)
    monkeypatch.setattr(real_icl_module, "_source_tree_hash", lambda: "source-tree-fixture")

    def fake_evaluate(
        model: torch.nn.Module, *_args: object, **_kwargs: object
    ) -> dict[str, float]:
        assert torch.is_inference_mode_enabled()
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        return {"normalized_nll": 1.0, "accuracy": 0.5}

    def reject_optimizer(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("OpenML frozen ICL must not construct an optimizer")

    monkeypatch.setattr(real_icl_module, "_evaluate", fake_evaluate)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", reject_optimizer)
    receipt = run_real_frozen_icl(
        config,
        device=torch.device("cpu"),
        openml_fetcher=fake_fetcher,
        openml_sklearn_version="9.9-test",
    )

    assert fetch_calls == [
        {
            "data_id": 1462,
            "target_column": "default-target",
            "as_frame": False,
            "parser": "liac-arff",
            "cache": True,
        }
    ]
    panel = receipt["panel_manifest"]
    assert panel["file_sha256"] == hashlib.sha256(panel_path.read_bytes()).hexdigest()
    assert panel["evaluated_dataset_ids"] == ["banknote_authentication"]
    assert len(panel["materialization_manifest_sha256"]) == 64
    provenance = receipt["dataset_provenance"]["banknote_authentication"]
    source = provenance["source_manifest"]
    assert source["source"]["data_id"] == 1462
    assert source["source"]["resolved_target_column"] == "class"
    assert source["fetch"] == {
        "api": "sklearn.datasets.fetch_openml",
        "fetcher_mode": "injected_callable",
        "scikit_learn_version": "9.9-test",
        "as_frame": False,
        "parser": "liac-arff",
        "cache": True,
        "data_home": None,
    }
    assert source["materialized"]["label_mapping"] == {"alpha": 0, "zeta": 1}
    assert len(source["materialized"]["array_sha256"]) == 64
    assert receipt["all_frozen_arm_parameter_hashes_unchanged"] is True
    assert receipt["frozen_arm_optimizer_created"] is False


def test_openml_new6_dataset_requires_explicit_panel_manifest(tmp_path: Path) -> None:
    config = RealIclConfig(
        checkpoint_root=tmp_path,
        output_path=tmp_path / "result.json",
        dataset_ids=("banknote_authentication",),
        checkpoint_seeds=(1729,),
        split_seeds=(1729,),
        bootstrap_replicates=100,
    )
    with pytest.raises(ValueError, match="explicit panel manifest"):
        config.validate()


@pytest.mark.parametrize(
    ("split_seeds", "query_limit", "query_chunk_rows", "message"),
    [
        ((1729,), 256, 64, "three split seeds"),
        ((1729, 2718, 31415), 128, 64, "query_limit=256"),
        ((1729, 2718, 31415), 256, 32, "query_limit=256"),
    ],
)
def test_openml_new6_runtime_cannot_drift_from_preregistered_panel(
    tmp_path: Path,
    split_seeds: tuple[int, ...],
    query_limit: int,
    query_chunk_rows: int,
    message: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = RealIclConfig(
        checkpoint_root=tmp_path,
        output_path=tmp_path / "result.json",
        dataset_ids=("banknote_authentication",),
        checkpoint_seeds=(1729,),
        split_seeds=split_seeds,
        context_policy=LOW_SHOT_CONTEXT_POLICY,
        query_limit=query_limit,
        query_chunk_rows=query_chunk_rows,
        bootstrap_replicates=100,
        panel_manifest_path=(
            root / "experiments/transfer-base-v2/real-frozen-icl-openml-new6.yaml"
        ),
    )
    with pytest.raises(ValueError, match=message):
        config.validate()
