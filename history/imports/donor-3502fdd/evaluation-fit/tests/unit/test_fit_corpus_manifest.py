from __future__ import annotations

import json
from copy import deepcopy

import pytest

from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.fit_artifacts import _validate_compiler_manifest
from tabu_lab.experiments import (
    AugmentedReadoutGeometry,
    BaselineRole,
    BaselineSpec,
    DatasetAdapterSpec,
    DatasetOrigin,
    FitDatasetSpec,
    FitDevice,
    FitExecutionConfig,
    FitExperimentSpec,
    FitPassGate,
    FitSeedConfig,
    FitStage,
    FitTrainingConfig,
    ModelSemanticConfig,
    RedistributionPolicy,
    ReferenceBackendConfig,
    build_corpus_compiler_binding_manifest,
    corpus_compiler_episode_recipe_hashes,
    validate_corpus_compiler_binding_manifest,
)
from tabu_lab.experiments.s1_table_synthetic import build_s1_completion_corpus
from tabu_lab.registry import get_model_spec


def _s1_spec():  # type: ignore[no-untyped-def]
    corpus = build_s1_completion_corpus("tabuf")
    model_spec = get_model_spec("tabuf")
    source_preimage = {
        "schema": "tabu.s1-generator-source.v1",
        "module": "tabu_lab.experiments.s1_table_synthetic",
        "builder": "build_s1_completion_corpus",
    }
    spec = FitExperimentSpec(
        experiment_id="S1-001-tabuf-v1",
        stage=FitStage.S1,
        contract_id="tabuf",
        contract_version=model_spec.contract_version,
        model_spec=model_spec,
        model_spec_hash=canonical_hash(model_spec),
        dataset=FitDatasetSpec(
            dataset_id=corpus.dataset.dataset_id,
            origin=DatasetOrigin.GENERATED,
            source_uri="generator://tabu-lab/s1-table-completion-v1",
            source_sha256=canonical_hash(source_preimage),
            dataset_hash=corpus.dataset.dataset_hash,
            license_id="Apache-2.0",
            redistribution=RedistributionPolicy.ALLOWED,
            adapter=DatasetAdapterSpec(
                adapter_id="tabu-s1-table-synthetic",
                adapter_version="1.0.0",
            ),
        ),
        split=corpus.typed_split,
        episode_schedule=corpus.schedule,
        semantic=ModelSemanticConfig(
            reference=ReferenceBackendConfig(),
            augmented_readout_geometry=AugmentedReadoutGeometry.MATCHED_UF,
        ),
        training=FitTrainingConfig(
            learning_rate=1.0e-3,
            weight_decay=0.0,
            gradient_clip_norm=1.0,
            max_updates=3_000,
            wall_clock_budget_minutes=30,
            exact_resume=True,
        ),
        execution=FitExecutionConfig(
            device=FitDevice.CPU,
            deterministic_algorithms=True,
        ),
        seeds=FitSeedConfig(
            model_seeds=(1729, 2718, 31415),
            data_seed=104729,
            split_seed=130363,
            episode_order_seed=130363,
        ),
        target_families=corpus.schedule.target_families,
        baselines=(
            BaselineSpec(
                baseline_id="exact-support-mean-mode",
                role=BaselineRole.TRIVIAL,
            ),
        ),
        pass_gate=FitPassGate(
            stage=FitStage.S1,
            max_loss_ratio=0.10,
            max_trivial_baseline_ratio=0.50,
            max_numeric_nrmse=0.05,
            min_categorical_accuracy=0.98,
            max_categorical_nll=0.10,
        ),
    )
    return corpus, spec


def _validate(manifest, corpus) -> None:  # type: ignore[no-untyped-def]
    validate_corpus_compiler_binding_manifest(
        manifest,
        expected_hash=canonical_hash(manifest),
        contract_id="tabuf",
        dataset_hash=corpus.dataset.dataset_hash,
        typed_split_hash=corpus.typed_split.content_hash,
        typed_split_kind=corpus.typed_split.kind.value,
        fit_partition=corpus.typed_split.fit_partition,
        episode_schedule=corpus.schedule,
        expected_corpus_hash=corpus.corpus_hash,
    )


def test_corpus_compiler_manifest_is_canonical_complete_and_truth_opaque() -> None:
    corpus, _ = _s1_spec()
    first = build_corpus_compiler_binding_manifest(corpus, contract_id="tabuf")
    replay = build_corpus_compiler_binding_manifest(corpus, contract_id="tabuf")

    assert first == replay
    assert canonical_hash(first) == canonical_hash(replay)
    assert first["corpus_hash"] == corpus.corpus_hash
    assert [episode["partition"] for episode in first["episodes"]] == [
        *("train" for _ in corpus.train_episodes),
        *("validation" for _ in corpus.validation_episodes),
        *("test" for _ in corpus.test_episodes),
    ]
    assert [episode["ordinal"] for episode in first["episodes"]] == [
        *(episode.ordinal for episode in corpus.train_episodes),
        *(episode.ordinal for episode in corpus.validation_episodes),
        *(episode.ordinal for episode in corpus.test_episodes),
    ]
    serialized = json.dumps(first, sort_keys=True)
    assert "target_values" not in serialized
    assert "truth_value" not in serialized
    assert "support_values" not in serialized
    assert all("sidecar_hash" in episode for episode in first["episodes"])
    assert corpus_compiler_episode_recipe_hashes(first) == tuple(
        episode.recipe_hash
        for partition in ("train", "validation", "test")
        for episode in corpus.episodes(partition)  # type: ignore[arg-type]
    )
    _validate(first, corpus)


def test_corpus_compiler_manifest_mapping_order_does_not_change_identity() -> None:
    corpus, _ = _s1_spec()
    manifest = build_corpus_compiler_binding_manifest(corpus, contract_id="tabuf")
    reordered = dict(reversed(tuple(manifest.items())))
    reordered["carrier_view_hashes"] = dict(
        reversed(tuple(manifest["carrier_view_hashes"].items()))
    )

    assert canonical_hash(reordered) == canonical_hash(manifest)
    _validate(reordered, corpus)


def _tamper_typed_split(manifest):  # type: ignore[no-untyped-def]
    manifest["typed_split_hash"] = "f" * 64


def _tamper_schedule(manifest):  # type: ignore[no-untyped-def]
    manifest["schedule"]["order_seed"] += 1
    manifest["schedule_hash"] = canonical_hash(manifest["schedule"])
    manifest["schedule_realization"]["schedule_hash"] = manifest["schedule_hash"]
    manifest["schedule_realization_hash"] = canonical_hash(
        manifest["schedule_realization"]
    )


def _tamper_episode_order(manifest):  # type: ignore[no-untyped-def]
    manifest["episodes"][0], manifest["episodes"][1] = (
        manifest["episodes"][1],
        manifest["episodes"][0],
    )


def _tamper_provenance(manifest):  # type: ignore[no-untyped-def]
    episode = manifest["episodes"][0]
    episode["compilation_provenance"]["source_view_hash"] = "e" * 64
    episode["compilation_provenance_hash"] = canonical_hash(
        {
            "schema": "tabu.compilation-provenance.v2",
            **episode["compilation_provenance"],
        }
    )


def _tamper_evidence(manifest):  # type: ignore[no-untyped-def]
    manifest["episodes"][0]["evidence_hash"] = "d" * 64


def _tamper_source_ledger(manifest):  # type: ignore[no-untyped-def]
    manifest["episodes"][0]["source_ledger_hash"] = "c" * 64


def _tamper_normalizer(manifest):  # type: ignore[no-untyped-def]
    manifest["numeric_normalizer"]["artifact_hash"] = "b" * 64


def _tamper_corpus_hash(manifest):  # type: ignore[no-untyped-def]
    manifest["corpus_hash"] = "a" * 64


@pytest.mark.parametrize(
    "tamper",
    (
        _tamper_typed_split,
        _tamper_schedule,
        _tamper_episode_order,
        _tamper_provenance,
        _tamper_evidence,
        _tamper_source_ledger,
        _tamper_normalizer,
        _tamper_corpus_hash,
    ),
)
def test_corpus_compiler_manifest_tampering_fails_after_hash_remint(tamper) -> None:  # type: ignore[no-untyped-def]
    corpus, _ = _s1_spec()
    manifest = deepcopy(
        build_corpus_compiler_binding_manifest(corpus, contract_id="tabuf")
    )
    tamper(manifest)

    with pytest.raises(ValueError):
        _validate(manifest, corpus)


def test_fit_artifact_compiler_dispatch_accepts_s1_corpus_schema() -> None:
    corpus, spec = _s1_spec()
    manifest = build_corpus_compiler_binding_manifest(corpus, contract_id="tabuf")

    _validate_compiler_manifest(
        manifest,
        spec=spec,
        expected_hash=canonical_hash(manifest),
    )


def test_fit_artifact_compiler_dispatch_rejects_single_episode_schema_for_s1() -> None:
    _, spec = _s1_spec()

    with pytest.raises(ValueError, match="multi-episode corpus compiler schema"):
        _validate_compiler_manifest(
            {"schema": "tabu.fit-compiler-binding.v1"},
            spec=spec,
            expected_hash="0" * 64,
        )
