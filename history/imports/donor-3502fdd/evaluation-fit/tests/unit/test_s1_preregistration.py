from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from tabu_lab.contracts import canonical_hash
from tabu_lab.experiments import FitDevice, FitEvidenceMode, FitStage
from tabu_lab.experiments.s1_preregistration import (
    build_all_s1_preregistrations,
    build_s1_preregistration,
    s1_dataset_manifest,
    validate_s1_binding,
)
from tabu_lab.experiments.s1_registry import (
    S1GeneratorSource,
    S1Recipe,
    get_s1_registration,
    list_s1_registrations,
)

ROOT = Path(__file__).resolve().parents[2]

EXPECTED_EXPERIMENT_IDS = (
    "S1-001-tabuf-latent-mixed-v1",
    "S1-002-tabu-unit-row-latent-mixed-v1",
    "S1-003-tabu-unit-pair-latent-mixed-v1",
    "S1-004-tabul-compositional-xor-v1",
    "S1-005-tabufl-joint-compositional-xor-v1",
    "S1-006-tabu4graph-community-v1",
    "S1-007-tabu4graph-diffusion-v1",
    "S1-008-tabu4rec-rating-v1",
    "S1-010-tabu-cell-base-completion-v1",
    "S1-011-tabu-cell-base-supervised-regression-v1",
    "S1-012-tabu-cell-base-supervised-classification-v1",
    "S1-009-tabu4rec-preference-v1",
)


@pytest.fixture(scope="module")
def registrations():  # type: ignore[no-untyped-def]
    return list_s1_registrations()


@pytest.fixture(scope="module")
def specs():  # type: ignore[no-untyped-def]
    return build_all_s1_preregistrations()


def test_s1_registry_is_closed_over_twelve_experiments_and_eight_models(registrations) -> None:  # type: ignore[no-untyped-def]
    assert tuple(item.experiment_id for item in registrations) == EXPECTED_EXPERIMENT_IDS
    assert Counter(item.contract_id for item in registrations) == {
        "tabuf": 1,
        "tabu.unit_row": 1,
        "tabu.unit_pair": 1,
        "tabul": 1,
        "tabufl": 1,
        "tabu4graph": 2,
        "tabu4rec": 2,
        "tabu.cell.base": 3,
    }
    assert {item.recipe for item in registrations if item.contract_id == "tabu4graph"} == {
        S1Recipe.GRAPH_COMMUNITY,
        S1Recipe.GRAPH_DIFFUSION,
    }
    assert {item.recipe for item in registrations if item.contract_id == "tabu4rec"} == {
        S1Recipe.REC_RATING,
        S1Recipe.REC_PREFERENCE,
    }


def test_generator_uri_and_hash_bind_the_exact_registered_source(registrations) -> None:  # type: ignore[no-untyped-def]
    for registration in registrations:
        source = (
            ROOT
            / "src"
            / "tabu_lab"
            / "experiments"
            / (f"{registration.generator_source.value}.py")
        )
        assert registration.source_uri == (
            f"pkg://tabu_lab.experiments.{registration.generator_source.value}"
            f"#{registration.generator_entrypoint}"
        )
        assert registration.source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
        assert registration.adapter_version == "1.0.0"
        if registration.contract_id in {"tabu4graph", "tabu4rec"}:
            assert registration.generator_source is S1GeneratorSource.TOPOLOGY
        else:
            assert registration.generator_source is S1GeneratorSource.TABLE


def test_all_s1_specs_are_canonical_cuda_three_seed_gate_specs(specs, registrations) -> None:  # type: ignore[no-untyped-def]
    by_id = {item.experiment_id: item for item in registrations}
    assert tuple(spec.experiment_id for spec in specs) == EXPECTED_EXPERIMENT_IDS
    for spec in specs:
        registration = by_id[spec.experiment_id]
        assert spec.stage is FitStage.S1
        assert spec.contract_id == registration.contract_id
        assert spec.execution.device is FitDevice.CUDA
        assert spec.execution.device_index == 0
        assert spec.execution.deterministic_algorithms is True
        assert spec.execution.evidence_mode is FitEvidenceMode.GATE
        assert spec.training.learning_rate == pytest.approx(1.0e-3)
        assert spec.training.max_updates == 3000
        assert spec.training.wall_clock_budget_minutes == 15
        assert spec.training.exact_resume is True
        assert spec.seeds.model_seeds == (1729, 2718, 31415)
        assert spec.seeds.data_seed == 104729
        assert spec.seeds.split_seed == 130363
        assert spec.seeds.episode_order_seed == 130363
        assert spec.pass_gate.max_loss_ratio == pytest.approx(0.10)
        assert spec.pass_gate.max_trivial_baseline_ratio == pytest.approx(0.50)
        assert spec.pass_gate.max_numeric_nrmse == pytest.approx(0.05)
        assert spec.pass_gate.min_categorical_accuracy == pytest.approx(0.98)
        assert spec.pass_gate.max_categorical_nll == pytest.approx(0.10)
        assert spec.dataset.source_uri == registration.source_uri
        assert spec.dataset.source_sha256 == registration.source_hash
        assert spec.model_spec_hash == canonical_hash(spec.model_spec)
        assert spec.dataset.dataset_id == spec.split.dataset_id
        assert spec.dataset.dataset_hash == spec.split.dataset_hash
        assert spec.target_families == spec.episode_schedule.target_families
        assert spec.episode_schedule.sampler_seed == spec.seeds.data_seed
        assert spec.episode_schedule.order_seed == spec.seeds.episode_order_seed


def test_s1_semantics_preserve_the_contract_repaired_f0_architectures(specs) -> None:  # type: ignore[no-untyped-def]
    by_contract: dict[str, list[object]] = {}
    for spec in specs:
        by_contract.setdefault(spec.contract_id, []).append(spec)

    unit_row = by_contract["tabu.unit_row"][0]
    assert unit_row.semantic.reference.geometry_normalization == "rms_unit"
    assert unit_row.semantic.reference.routing_bandwidth == pytest.approx(2.5)

    for contract_id in ("tabul", "tabufl"):
        supervised = by_contract[contract_id][0]
        assert supervised.semantic.label_columns == (6, 7)
        assert supervised.semantic.label_address_plan == ("predictor_unit_linked_per_label_v2")

    for graph in by_contract["tabu4graph"]:
        assert graph.semantic.target_feature == 2
        assert graph.semantic.graph_unit_receiver_plan == "same_row_visible_cells"

    rec_families = {spec.semantic.response_family for spec in by_contract["tabu4rec"]}
    assert rec_families == {"numeric_rating", "categorical_preference"}
    for rec in by_contract["tabu4rec"]:
        assert rec.semantic.recommendation_address_plan == "axis_address_bootstrap_v1"
        assert rec.semantic.rec_axis_summary_dim == 2
        assert rec.semantic.rec_matched_residual_scale == pytest.approx(0.1)


def test_s1_binding_validator_rejects_semantic_or_generator_drift(specs) -> None:  # type: ignore[no-untyped-def]
    spec = specs[0]
    validate_s1_binding(spec)

    drifted_semantic = spec.semantic.model_copy(
        update={"augmented_readout_geometry": "matched_ufc"}
    )
    drifted = spec.model_copy(update={"semantic": drifted_semantic})
    with pytest.raises(ValueError, match="registered generator, corpus, or config"):
        validate_s1_binding(drifted)


def test_s1_dataset_manifest_exports_the_full_corpus_binding() -> None:
    registration = get_s1_registration(EXPECTED_EXPERIMENT_IDS[0])
    corpus = registration.build_corpus()
    manifest = s1_dataset_manifest(registration.experiment_id)

    assert manifest["dataset"].dataset_hash == corpus.dataset.dataset_hash
    assert manifest["typed_split_hash"] == corpus.typed_split.content_hash
    assert manifest["schedule_hash"] == corpus.schedule.content_hash
    assert manifest["schedule_realization_hash"] == (corpus.schedule_realization.content_hash)
    assert manifest["fit_value_mask_hash"] == corpus.fit_value_mask_hash
    assert manifest["source_ledger_hashes"] == dict(corpus.source_ledger_hashes)
    assert len(manifest["source_ledger_hashes"]) == corpus.schedule.episode_count
    assert manifest["corpus_hash"] == corpus.corpus_hash


def test_registry_lookup_and_builder_fail_closed() -> None:
    registration = get_s1_registration(EXPECTED_EXPERIMENT_IDS[0])
    assert registration.contract_id == "tabuf"
    assert build_s1_preregistration(registration.experiment_id).experiment_id == (
        registration.experiment_id
    )
    with pytest.raises(KeyError, match="unknown S1 experiment"):
        get_s1_registration("S1-999-unknown-v1")
    with pytest.raises(KeyError, match="unknown S1 experiment"):
        build_s1_preregistration("S1-999-unknown-v1")
