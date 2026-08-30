from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tabu_lab.contracts import (
    FeatureKind,
    GraphTopology,
    OriginState,
    PredictionBundle,
    PredictionEntry,
    PredictionKind,
    PredictionStatus,
    TruthSidecar,
    canonical_hash,
    origin_code,
)
from tabu_lab.evaluation import verify_fit_attempt_artifacts
from tabu_lab.evidence import RunBundle
from tabu_lab.experiments.contracts import (
    FitDevice,
    FitExperimentSpec,
    FitMetricKind,
    ReferenceBackendConfig,
)
from tabu_lab.experiments.corpus import EpisodeScheduleRealization, FitEpisodeCorpus
from tabu_lab.experiments.fixtures import build_f0_fixture
from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.experiments.runner import (
    FitExperimentError,
    _build_model,
    _device,
    _identity,
    _trainer,
    _training_and_execution_configs,
    source_tree_manifest,
    trivial_baseline,
)
from tabu_lab.experiments.s1_preregistration import build_s1_preregistration
from tabu_lab.experiments.s1_registry import build_registered_s1_corpus
from tabu_lab.experiments.s1_runner import (
    _aggregate_family_metrics,
    _graph_mechanism_gradient_probe,
    _run_s1_seed,
    _validate_rec_dual_arm_trace,
    assess_s1_feasibility,
    s1_compiler_binding_manifest,
)


def _typed_prediction(
    episode_id: str,
    *,
    numeric: torch.Tensor,
    categorical: torch.Tensor,
    completion_mask: torch.Tensor | None = None,
    label_mask: torch.Tensor | None = None,
) -> PredictionBundle:
    target = torch.ones(2, 2, dtype=torch.bool)
    numeric_mask = torch.tensor([[True, False], [True, False]])
    categorical_mask = ~numeric_mask
    completion = target if completion_mask is None else completion_mask
    label = torch.zeros_like(target) if label_mask is None else label_mask
    return PredictionBundle(
        episode_id=episode_id,
        model_id="tabuf",
        contract_version="0.1.0",
        entries={
            "numeric": PredictionEntry(
                kind=PredictionKind.NUMERIC,
                status=PredictionStatus.OK,
                values=numeric,
            ),
            "distribution": PredictionEntry(
                kind=PredictionKind.DISTRIBUTION,
                status=PredictionStatus.OK,
                values=categorical,
            ),
        },
        auxiliaries={
            "target_mask": target,
            "support_available": target,
            "numeric_target_mask": numeric_mask,
            "categorical_target_mask": categorical_mask,
            "completion_target_mask": completion,
            "label_target_mask": label,
        },
    )


def _metric_episode(index: int, numeric_truth: tuple[float, float]) -> SimpleNamespace:
    values = torch.tensor(
        [[numeric_truth[0], 0.0], [numeric_truth[1], 1.0]],
        dtype=torch.float32,
    )
    truth = TruthSidecar(
        episode_id=f"metric-{index}",
        recipe_hash=f"{index + 1:064x}",
        row_ids=(f"r{index}-0", f"r{index}-1"),
        feature_names=("numeric", "categorical"),
        target_values=values,
        target_mask=torch.ones_like(values, dtype=torch.bool),
    )
    numeric_mask = torch.tensor([[True, False], [True, False]])
    categorical_mask = ~numeric_mask
    origins = torch.full(
        values.shape,
        origin_code(OriginState.ARTIFICIAL_MASK),
        dtype=torch.uint8,
    )
    evidence = SimpleNamespace(
        feature_specs=(
            SimpleNamespace(kind=FeatureKind.NUMERIC),
            SimpleNamespace(kind=FeatureKind.CATEGORICAL),
        ),
        origin_states=origins,
    )
    return SimpleNamespace(
        truth=truth,
        evidence=evidence,
        target_family_masks={"numeric": numeric_mask, "categorical": categorical_mask},
    )


def test_s1_family_metrics_use_global_truth_scale_and_target_level_categorical() -> None:
    episodes = (_metric_episode(0, (0.0, 2.0)), _metric_episode(1, (100.0, 102.0)))
    initial = tuple(
        _typed_prediction(
            episode.truth.episode_id,
            numeric=episode.truth.target_values + torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            categorical=torch.tensor(
                [
                    [[0.5, 0.5], [0.6, 0.4]],
                    [[0.5, 0.5], [0.4, 0.6]],
                ]
            ),
        )
        for episode in episodes
    )
    final_probabilities = (
        torch.tensor(
            [
                [[0.5, 0.5], [0.9, 0.1]],
                [[0.5, 0.5], [0.2, 0.8]],
            ]
        ),
        torch.tensor(
            [
                [[0.5, 0.5], [0.7, 0.3]],
                [[0.5, 0.5], [0.4, 0.6]],
            ]
        ),
    )
    final = tuple(
        _typed_prediction(
            episode.truth.episode_id,
            numeric=episode.truth.target_values + torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            categorical=probabilities,
        )
        for episode, probabilities in zip(episodes, final_probabilities, strict=True)
    )
    metrics = _aggregate_family_metrics(
        initial=initial,
        final=final,
        episodes=episodes,  # type: ignore[arg-type]
        baseline={"families": {"completion": {"numeric_mse": 5.0, "categorical_nll": 1.0}}},
    )
    numeric = next(entry for entry in metrics if entry.kind is FitMetricKind.NUMERIC)
    categorical = next(entry for entry in metrics if entry.kind is FitMetricKind.CATEGORICAL)

    all_numeric_truth = torch.tensor([0.0, 2.0, 100.0, 102.0])
    assert numeric.mse == pytest.approx(1.0)
    assert numeric.nrmse == pytest.approx(
        1.0 / float(all_numeric_truth.std(unbiased=False)), rel=1.0e-6
    )
    expected_nll = float(-torch.log(torch.tensor((0.9, 0.8, 0.7, 0.6))).mean())
    assert categorical.accuracy == 1.0
    assert categorical.nll == pytest.approx(expected_nll, rel=1.0e-6)


def test_s1_family_metrics_fail_closed_on_tabufl_f_l_lane_swap() -> None:
    values = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    completion = torch.tensor([[True, True], [False, False]])
    label = ~completion
    origins = torch.where(
        completion,
        torch.full_like(completion, origin_code(OriginState.ARTIFICIAL_MASK), dtype=torch.uint8),
        torch.full_like(label, origin_code(OriginState.QUERY), dtype=torch.uint8),
    )
    episode = SimpleNamespace(
        truth=TruthSidecar(
            episode_id="tabufl-ledger",
            recipe_hash="a" * 64,
            row_ids=("context", "query"),
            feature_names=("numeric", "categorical"),
            target_values=values,
            target_mask=torch.ones_like(values, dtype=torch.bool),
        ),
        evidence=SimpleNamespace(
            feature_specs=(
                SimpleNamespace(kind=FeatureKind.NUMERIC),
                SimpleNamespace(kind=FeatureKind.CATEGORICAL),
            ),
            origin_states=origins,
        ),
        target_family_masks={"F": completion, "L": label},
    )
    probabilities = torch.tensor(
        [
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.1, 0.9], [0.1, 0.9]],
        ],
        dtype=torch.float32,
    )
    correct = _typed_prediction(
        episode.truth.episode_id,
        numeric=values,
        categorical=probabilities,
        completion_mask=completion,
        label_mask=label,
    )
    tampered = replace(
        correct,
        auxiliaries={
            **dict(correct.auxiliaries),
            "completion_target_mask": torch.ones_like(completion),
            "label_target_mask": torch.zeros_like(label),
        },
    )

    with pytest.raises(FitExperimentError, match="host-side ledger"):
        _aggregate_family_metrics(
            initial=(correct,),
            final=(tampered,),
            episodes=(episode,),  # type: ignore[arg-type]
            baseline={"families": {"completion": {}, "label": {}}},
        )


def test_s1_graph_probe_separates_local_and_global_readout_paths() -> None:
    spec = build_f0_preregistration(
        "tabu4graph",
        device=FitDevice.CPU,
        fixture_version="v2",
        reference=ReferenceBackendConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
        ),
    )
    fixture = build_f0_fixture("tabu4graph", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    sources, active, gradients, scored = _graph_mechanism_gradient_probe(
        model,
        fixture,  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    assert scored == fixture.truth.target_count
    assert set(sources) == {"graph_local_path", "graph_global_readout"}
    assert all(count > 0 for count in sources.values())
    assert active == {
        "graph_local_path": scored,
        "graph_global_readout": scored,
    }
    assert all(norm > 0.0 for norm in gradients.values())


def test_s1_graph_active_counts_follow_each_target_local_source_path() -> None:
    spec = build_f0_preregistration(
        "tabu4graph",
        device=FitDevice.CPU,
        fixture_version="v2",
        reference=ReferenceBackendConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
        ),
    )
    fixture = build_f0_fixture("tabu4graph", fixture_version="v2")
    evidence = fixture.evidence
    assert evidence.graph_topology is not None
    isolated = replace(
        evidence,
        graph_topology=GraphTopology(
            node_ids=evidence.row_ids,
            adjacency=torch.zeros(
                len(evidence.row_ids), len(evidence.row_ids), dtype=torch.bool
            ),
            direction=evidence.graph_topology.direction,
        ),
    )
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))

    sources, active, _, scored = _graph_mechanism_gradient_probe(
        model,
        SimpleNamespace(evidence=isolated, truth=fixture.truth),  # type: ignore[arg-type]
        device=torch.device("cpu"),
    )

    assert scored == fixture.truth.target_count
    assert sources["graph_local_path"] == 0
    assert active["graph_local_path"] == 0
    assert sources["graph_global_readout"] > 0
    assert active["graph_global_readout"] == scored


def test_s1_rec_trace_requires_declared_dual_arm_operations() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
        reference=ReferenceBackendConfig(
            d_model=8,
            n_heads=2,
            d_ff=16,
            n_blocks=1,
            inducing_slots=2,
            matched_slots=2,
        ),
    )
    fixture = build_f0_fixture("tabu4rec", fixture_version="v2")
    model = _build_model(spec, seed=1729, device=torch.device("cpu"))
    prediction = model(fixture.evidence)
    _validate_rec_dual_arm_trace(prediction)

    assert prediction.trace is not None
    events = list(prediction.trace.events)
    index = next(
        index
        for index, event in enumerate(events)
        if event.name == "recommendation_support_ledger"
    )
    ledger = events[index]
    events[index] = replace(
        ledger,
        metadata={
            **dict(ledger.metadata),
            "operation_trace": tuple(
                operation
                for operation in ledger.metadata["operation_trace"]
                if operation != "equal_active_arm_mix"
            ),
        },
    )
    tampered = replace(prediction, trace=replace(prediction.trace, events=tuple(events)))
    with pytest.raises(FitExperimentError, match="equal_active_arm_mix"):
        _validate_rec_dual_arm_trace(tampered)


def _reduced_s1_case() -> tuple[FitExperimentSpec, FitEpisodeCorpus]:
    experiment_id = "S1-001-tabuf-latent-mixed-v1"
    base = build_s1_preregistration(experiment_id)
    full = build_registered_s1_corpus(experiment_id)
    schedule = full.schedule.model_copy(update={"episode_count": 4})
    train = full.train_episodes[:2]
    validation = full.validation_episodes[:1]
    test = full.test_episodes[:1]
    realization = EpisodeScheduleRealization.create(
        schedule,
        typed_split_hash=full.typed_split.content_hash,
        fit_value_mask_hash=full.fit_value_mask_hash,
        train_recipe_hashes=tuple(episode.recipe_hash for episode in train),
        validation_recipe_hashes=tuple(episode.recipe_hash for episode in validation),
        test_recipe_hashes=tuple(episode.recipe_hash for episode in test),
    )
    corpus = FitEpisodeCorpus(
        dataset=full.dataset,
        typed_split=full.typed_split,
        carrier_manifest=full.carrier_manifest,
        carrier_views=full.carrier_views,
        fit_value_mask=full.fit_value_mask,
        numeric_normalizer=full.numeric_normalizer,
        train_episodes=train,
        validation_episodes=validation,
        test_episodes=test,
        schedule=schedule,
        schedule_realization=realization,
        builder_options=full.builder_options,
    )
    payload = base.model_dump(mode="python")
    payload["episode_schedule"] = schedule.model_dump(mode="python")
    payload["execution"].update(device="cpu", device_index=None)
    payload["semantic"]["reference"].update(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
    )
    return FitExperimentSpec.model_validate(payload), corpus


def _execute_reduced_s1_seed(
    tmp_path: Path,
    *,
    max_updates_override: int,
    monotonic_clock: Callable[[], float] | None = None,
) -> SimpleNamespace:
    spec, corpus = _reduced_s1_case()
    targets, feasibility = assess_s1_feasibility(spec, corpus)
    baseline = trivial_baseline(targets)
    compiler_manifest = s1_compiler_binding_manifest(spec, corpus)
    device = _device(spec)
    code_manifest = source_tree_manifest(".")
    code_hash = canonical_hash(code_manifest)
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        training, execution, environment, environment_payload = _training_and_execution_configs(
            spec, device=device
        )
        identity = _identity(
            spec,
            corpus,  # type: ignore[arg-type]
            seed=1729,
            code_hash=code_hash,
            compiler_hash=canonical_hash(compiler_manifest),
            training=training,
            execution=execution,
        )
        result = _run_s1_seed(
            spec=spec,
            corpus=corpus,
            identity=identity,
            seed=1729,
            device=device,
            output_root=tmp_path,
            preregistration_text=yaml.safe_dump(spec.model_dump(mode="json"), sort_keys=True),
            code_manifest=code_manifest,
            code_hash=code_hash,
            compiler_manifest=compiler_manifest,
            feasibility=feasibility,
            baseline=baseline,
            training=training,
            execution=execution,
            environment=environment,
            environment_payload=environment_payload,
            command=("pytest", "reduced-s1-one-seed"),
            formal_authorization=None,
            max_updates_override=max_updates_override,
            monotonic_clock=monotonic_clock,
        )
    finally:
        torch.use_deterministic_algorithms(previous)
    return SimpleNamespace(
        result=result,
        spec=spec,
        corpus=corpus,
        identity=identity,
        training=training,
        execution=execution,
        device=device,
    )


def test_s1_reduced_one_seed_writes_recipe_bound_failed_gate_bundle(
    tmp_path: Path,
) -> None:
    case = _execute_reduced_s1_seed(tmp_path, max_updates_override=1)
    result = case.result
    corpus = case.corpus

    assert result.verdict == "failed"
    assert result.failure_phase is None
    assert result.fit_evaluation is not None
    assert result.fit_evaluation.checkpoint_reloaded
    assert result.artifacts.checkpoint is not None
    verified = verify_fit_attempt_artifacts(result.artifacts.directory)
    assert verified.receipt_hash == result.artifacts.receipt_hash
    run_bundle = RunBundle.model_validate_json(
        result.artifacts.run_bundle.read_text(encoding="utf-8")
    )
    expected_recipes = tuple(
        episode.recipe_hash
        for episode in (
            *corpus.train_episodes,
            *corpus.validation_episodes,
            *corpus.test_episodes,
        )
    )
    assert run_bundle.episode_recipe_hashes == expected_recipes
    assert run_bundle.metadata["issuance_status"] == "local_unissued"
    assert run_bundle.metadata["stop_reason"] == "max_updates"
    assert run_bundle.metadata["wall_time_seconds"] >= 0.0
    assert run_bundle.metadata["checkpoint_continuation_verified"] is True
    rows = tuple(
        json.loads(line)
        for line in (result.artifacts.directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    summary = next(row for row in rows if row["record_type"] == "summary")
    assert summary["executed_steps"] == 1
    assert summary["stop_reason"] == "max_updates"
    assert summary["wall_time_seconds"] >= 0.0
    assert len(summary["checkpoint_train_prediction_hashes"]) == len(corpus.train_episodes)
    assert summary["next_episode_recipe_hash"] == corpus.episode_at_update(1).recipe_hash
    assert summary["checkpoint_continuation_verified"] is True
    assert summary["checkpoint_continuation_step"] == 2
    assert len(summary["checkpoint_continuation_state_hash"]) == 64
    assert len(summary["checkpoint_continuation_prediction_hashes"]) == len(
        corpus.train_episodes
    )

    restored_model = _build_model(case.spec, seed=1729, device=case.device)
    restored_trainer = _trainer(
        restored_model,
        case.spec,
        case.identity,
        case.training,
        case.execution,
    )
    restored_trainer.load_checkpoint(result.artifacts.checkpoint)
    assert restored_trainer.step == 1


def test_s1_wall_clock_budget_records_typed_stop_reason(tmp_path: Path) -> None:
    ticks = iter((0.0, 901.0))
    case = _execute_reduced_s1_seed(
        tmp_path,
        max_updates_override=2,
        monotonic_clock=ticks.__next__,
    )
    result = case.result
    assert result.failure_phase is None
    verified = verify_fit_attempt_artifacts(result.artifacts.directory)
    assert verified.receipt_hash == result.artifacts.receipt_hash
    run_bundle = RunBundle.model_validate_json(
        result.artifacts.run_bundle.read_text(encoding="utf-8")
    )
    rows = tuple(
        json.loads(line)
        for line in (result.artifacts.directory / "metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    summary = next(row for row in rows if row["record_type"] == "summary")
    assert summary["executed_steps"] == 1
    assert summary["stop_reason"] == "wall_clock_budget"
    assert summary["wall_time_seconds"] == 901.0
    assert run_bundle.metadata["stop_reason"] == "wall_clock_budget"
    assert run_bundle.metadata["wall_time_seconds"] == 901.0
