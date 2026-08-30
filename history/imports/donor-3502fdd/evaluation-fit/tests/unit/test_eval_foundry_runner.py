from __future__ import annotations

import inspect
import math
import time as wall_time
from collections.abc import Sequence

import pytest

from tabu_lab.evaluation.foundry import (
    AdapterKind,
    AdapterLaunchSpec,
    AdapterSpec,
    AdapterStateError,
    BaselineAdapter,
    BlindExample,
    DatasetSnapshotBinding,
    DatasetUnavailableError,
    EvalProducerBinding,
    EvalResult,
    EvalSuiteSpec,
    EvaluationStatus,
    FailureCategory,
    PreparationContract,
    PreparedExample,
    PreparedScenario,
    RawPrediction,
    SourceMaterial,
    TargetKind,
    TopologyCheckCase,
    compare_results,
    dry_run_suite,
    load_suite,
    rescore_result_samples,
    run_evaluation,
    score_predictions,
    verify_result_against_prepared,
)
from tabu_lab.evaluation.foundry.contracts import deterministic_result_id


def _producer(run_id: str = "eval-run-spy") -> EvalProducerBinding:
    return EvalProducerBinding(
        provenance="receipted_run",
        run_id=run_id,
        receipt_sha256="d" * 64,
        receipt_pointer=f"runs/{run_id}/receipt.json",
        publication_eligible=True,
    )


def _regression_data(*, dataset_id: str = "sklearn-diabetes") -> PreparedScenario:
    scenario = _micro_suite().scenarios[1]
    train = (
        PreparedExample(
            example_id="train-0",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 0.0},
            target=0.0,
        ),
        PreparedExample(
            example_id="train-1",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 1.0},
            target=2.0,
        ),
        PreparedExample(
            example_id="train-2",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 2.0},
            target=4.0,
        ),
    )
    validation = (
        PreparedExample(
            example_id="validation-0",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 3.0},
            target=6.0,
        ),
    )
    test = (
        PreparedExample(
            example_id="test-0",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 4.0},
            target=8.0,
        ),
        PreparedExample(
            example_id="test-1",
            target_kind=TargetKind.NUMERIC,
            target_family="diabetes-target",
            features={"x": 5.0},
            target=10.0,
        ),
    )
    preparation = PreparationContract(
        preprocessing={
            "implementation_sha256": "1" * 64,
            "fitted_state_sha256": "2" * 64,
            "fit_partition": "train",
        },
        selection=scenario.selection.model_dump(mode="python"),
        mask={"kind": "none"},
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id=dataset_id,
        content=f"{dataset_id}:frozen-micro-source-v1".encode(),
        media_type="text/plain",
    )
    binding = DatasetSnapshotBinding(
        dataset_id=dataset_id,
        source_sha256=source_material.raw_sha256,
        split_sha256=PreparedScenario.split_sha256_for(
            train=train,
            validation=validation,
            test=test,
        ),
        recipe_sha256=PreparedScenario.recipe_sha256_for(preparation=preparation),
        truth_sidecar_sha256=PreparedScenario.truth_sidecar_sha256_for(test=test),
        partition_counts={"train": 3, "validation": 1, "test": 2},
    )
    return PreparedScenario(
        scenario_id="sklearn-diabetes-regression-micro",
        binding=binding,
        source_material=source_material,
        preparation=preparation,
        train=train,
        validation=validation,
        test=test,
    )


def _micro_suite() -> EvalSuiteSpec:
    payload = load_suite("table-supervised-micro-v0").model_dump(mode="python")
    payload["scenarios"][1]["selection"]["partition_limits"] = {
        "train": 3,
        "validation": 1,
        "test": 2,
    }
    return EvalSuiteSpec.model_validate(payload)


def _reidentified_result(
    result: EvalResult,
    *,
    suite: EvalSuiteSpec,
    binding: DatasetSnapshotBinding,
) -> EvalResult:
    scenario = next(item for item in suite.scenarios if item.scenario_id == result.scenario_id)
    return result.model_copy(
        update={
            "result_id": deterministic_result_id(
                suite=suite,
                scenario=scenario,
                adapter=result.adapter,
                binding=binding,
                producer=result.producer,
                seed=result.seed,
                status=result.status,
                raw_predictions=result.raw_predictions,
                topology_checks=result.topology_checks,
                per_example=result.per_example,
                metrics=result.metrics,
                counts=result.counts,
                failure_counts=result.failure_counts,
                coverage=result.coverage,
                failure=result.failure,
                claim_boundary=result.claim_boundary,
            )
        }
    )


def _model_adapter_spec(
    *,
    adapter_id: str = "spy-model",
    fit_iterations: int = 0,
    deterministic: bool = True,
    contract_id: str = "tabul",
    artifact_id: str = "artifact-spy",
) -> AdapterSpec:
    return AdapterSpec(
        adapter_id=adapter_id,
        adapter_version="1.0.0",
        kind=AdapterKind.MODEL,
        fit_iterations=fit_iterations,
        device_class="single_device",
        deterministic=deterministic,
        contract_id=contract_id,
        artifact_id=artifact_id,
    )


def _adapter_launch(
    adapter_type: type[object],
    *,
    declared_spec: AdapterSpec,
    kwargs: dict[str, object] | None = None,
) -> AdapterLaunchSpec:
    return AdapterLaunchSpec(
        module=adapter_type.__module__,
        qualname=adapter_type.__qualname__,
        kwargs=kwargs or {},
        declared_spec=declared_spec,
    )


class SpyAdapter:
    def __init__(
        self,
        *,
        adapter_id: str = "spy-model",
        value: float = 9.0,
        fit_iterations: int = 0,
        deterministic: bool = True,
    ) -> None:
        self._spec = _model_adapter_spec(
            adapter_id=adapter_id,
            fit_iterations=fit_iterations,
            deterministic=deterministic,
        )
        self.value = value
        self.saw_truth = False

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del scenario, fit_examples, seed
        self.saw_truth = any(hasattr(item, "target") for item in examples)
        return tuple(
            RawPrediction(example_id=item.example_id, value=self.value) for item in examples
        )


def _spy_launch(
    *,
    adapter_id: str = "spy-model",
    value: float = 9.0,
    fit_iterations: int = 0,
    deterministic: bool = True,
) -> AdapterLaunchSpec:
    return _adapter_launch(
        SpyAdapter,
        declared_spec=_model_adapter_spec(
            adapter_id=adapter_id,
            fit_iterations=fit_iterations,
            deterministic=deterministic,
        ),
        kwargs={
            "adapter_id": adapter_id,
            "value": value,
            "fit_iterations": fit_iterations,
            "deterministic": deterministic,
        },
    )


def test_run_strips_test_truth_and_retains_rescorable_outputs() -> None:
    suite = _micro_suite()
    prepared = _regression_data()

    result = run_evaluation(
        suite,
        scenario_id=prepared.scenario_id,
        adapter=_spy_launch(),
        prepared=prepared,
        seed=1729,
        producer=_producer(),
    )

    assert result.status is EvaluationStatus.SUCCEEDED
    assert result.coverage == 1.0
    assert len(result.raw_predictions) == 2
    assert {item.target_family for item in result.per_example} == {"diabetes-target"}
    rescored = rescore_result_samples(
        scenario=suite.scenarios[1],
        fit_examples=prepared.train,
        truth=prepared.test,
        predictions=result.raw_predictions,
    )
    assert rescored.metrics == result.metrics
    assert rescored.per_example == result.per_example


def test_run_accepts_projected_blind_inputs_with_exact_truth_sidecar_parity() -> None:
    suite = _micro_suite()
    prepared = _regression_data()
    projected = tuple(
        BlindExample(
            example_id=item.example_id,
            target_kind=item.target_kind,
            target_family=item.target_family,
            features={"projection": {"schema_version": "test.explicit-projection.v1"}},
            context={"selector": item.example_id},
        )
        for item in reversed(prepared.test)
    )

    result = run_evaluation(
        suite,
        scenario_id=prepared.scenario_id,
        adapter=_spy_launch(),
        prepared=prepared,
        seed=1729,
        producer=_producer("eval-run-projected"),
        blind_examples=projected,
    )

    assert result.status is EvaluationStatus.SUCCEEDED
    assert [item.example_id for item in result.raw_predictions] == [
        item.example_id for item in prepared.test
    ]
    with pytest.raises(ValueError, match="exact prepared test ids"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(),
            prepared=prepared,
            seed=1729,
            producer=_producer("eval-run-projected-missing"),
            blind_examples=projected[:-1],
        )
    wrong_family = projected[0].model_copy(update={"target_family": "wrong-family"})
    with pytest.raises(ValueError, match="target kind and family"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(),
            prepared=prepared,
            seed=1729,
            producer=_producer("eval-run-projected-family"),
            blind_examples=(wrong_family, projected[1]),
        )


def test_dry_run_and_run_refuse_missing_or_wrong_real_snapshot() -> None:
    canonical_suite = load_suite("table-supervised-micro-v0")
    report = dry_run_suite(canonical_suite)

    assert not report.ready
    adult = next(item for item in report.scenarios if item.scenario_id.startswith("adult"))
    assert adult.network_required
    assert "explicit_fetch" in adult.blockers[0]

    suite = _micro_suite()
    prepared = _regression_data(dataset_id="wrong-dataset")
    with pytest.raises(DatasetUnavailableError, match="prepared_dataset_id_mismatch"):
        run_evaluation(
            suite,
            scenario_id="sklearn-diabetes-regression-micro",
            adapter=_spy_launch(),
            prepared=prepared,
            seed=1729,
            producer=_producer(),
        )


def test_deterministic_baselines_match_closed_form_mean_and_ridge() -> None:
    suite = _micro_suite()
    scenario = suite.scenarios[1]
    prepared = _regression_data()
    baselines = {item.baseline_id: item for item in scenario.baselines}

    mean_result = run_evaluation(
        suite,
        scenario_id=scenario.scenario_id,
        adapter=BaselineAdapter(baselines["train-mean"]),
        prepared=prepared,
        seed=1729,
    )
    ridge_result = run_evaluation(
        suite,
        scenario_id=scenario.scenario_id,
        adapter=BaselineAdapter(baselines["standardized-ridge"]),
        prepared=prepared,
        seed=1729,
    )

    assert [item.value for item in mean_result.raw_predictions] == [2.0, 2.0]
    assert ridge_result.metrics["mae"] < mean_result.metrics["mae"]
    assert (
        ridge_result.content_hash
        == run_evaluation(
            suite,
            scenario_id=scenario.scenario_id,
            adapter=BaselineAdapter(baselines["standardized-ridge"]),
            prepared=prepared,
            seed=1729,
        ).content_hash
    )


class HalfAbstainingAdapter(SpyAdapter):
    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del scenario, fit_examples, seed
        return (
            RawPrediction(example_id=examples[0].example_id, value=8.0),
            RawPrediction(
                example_id=examples[1].example_id,
                abstained=True,
                failure_category=FailureCategory.MODEL,
                failure_code="unsupported_example",
            ),
        )


def test_coverage_and_failure_aggregation_count_explicit_abstention() -> None:
    suite = _micro_suite()
    result = run_evaluation(
        suite,
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_adapter_launch(
            HalfAbstainingAdapter,
            declared_spec=_model_adapter_spec(),
        ),
        prepared=_regression_data(),
        seed=1729,
        producer=_producer(),
    )

    assert result.status is EvaluationStatus.SUCCEEDED
    assert result.coverage == 0.5
    assert result.metrics["coverage"] == 0.5
    assert result.counts == {
        "abstained": 1,
        "categorical_targets": 0,
        "numeric_targets": 2,
        "scored": 1,
        "targets": 2,
    }
    assert result.failure_counts == {FailureCategory.MODEL: 1}


def test_compare_reports_per_scenario_mean_std_without_composite_rank() -> None:
    suite = _micro_suite()
    prepared = _regression_data()
    results = []
    for seed in suite.budget.model_seeds:
        results.append(
            run_evaluation(
                suite,
                scenario_id=prepared.scenario_id,
                adapter=_spy_launch(adapter_id="model-a", value=9.0),
                prepared=prepared,
                seed=seed,
                producer=_producer(f"run-model-a-{seed}"),
            )
        )
        results.append(
            run_evaluation(
                suite,
                scenario_id=prepared.scenario_id,
                adapter=_spy_launch(adapter_id="model-b", value=8.0),
                prepared=prepared,
                seed=seed,
                producer=_producer(f"run-model-b-{seed}"),
            )
        )

    prepared_by_scenario = {prepared.scenario_id: prepared}
    report = compare_results(suite, results, prepared=prepared_by_scenario)

    assert report.composite_score is False
    assert report.overall_rank is None
    assert report.aggregates
    assert {item.scenario_id for item in report.aggregates} == {prepared.scenario_id}
    assert all(item.seeds == (1729, 2718, 31415) for item in report.aggregates)
    assert all(item.std == 0.0 for item in report.aggregates)

    with pytest.raises(ValueError, match="at least two adapters"):
        compare_results(suite, results[::2], prepared=prepared_by_scenario)

    with pytest.raises(ValueError, match="canonical prepared scenarios"):
        compare_results(suite, results)

    mixed_data = []
    for item in results:
        if item.adapter.adapter_id != "model-b":
            mixed_data.append(item)
            continue
        changed = item.model_copy(update={"split_sha256": "d" * 64})
        mixed_data.append(
            _reidentified_result(
                changed,
                suite=suite,
                binding=prepared.binding.model_copy(update={"split_sha256": "d" * 64}),
            )
        )
    with pytest.raises(ValueError, match="canonical prepared snapshot"):
        compare_results(suite, mixed_data, prepared=prepared_by_scenario)

    washed_scale_results: list[EvalResult] = []
    for item in results:
        if item.adapter.adapter_id != "model-b":
            washed_scale_results.append(item)
            continue
        washed_scores = tuple(
            score.model_copy(
                update={
                    "normalization_scale": 1.0e9,
                    "metrics": {
                        **score.metrics,
                        "normalized_squared_error": score.metrics["squared_error"] / 1.0e18,
                    },
                }
            )
            for score in item.per_example
        )
        washed_metrics = dict(item.metrics)
        washed_metrics["regression_nrmse"] = (
            sum(score.metrics["normalized_squared_error"] for score in washed_scores)
            / len(washed_scores)
        ) ** 0.5
        washed = item.model_copy(
            update={"per_example": washed_scores, "metrics": washed_metrics}
        )
        washed = _reidentified_result(
            washed,
            suite=suite,
            binding=prepared.binding,
        )
        washed_scale_results.append(
            EvalResult.model_validate(washed.model_dump(mode="python"))
        )
    with pytest.raises(ValueError, match="canonical prepared truth"):
        compare_results(
            suite,
            washed_scale_results,
            prepared=prepared_by_scenario,
        )


def test_model_results_require_explicit_receipted_producer_binding() -> None:
    suite = _micro_suite()
    prepared = _regression_data()

    with pytest.raises(ValueError, match="producer run receipt"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(),
            prepared=prepared,
            seed=1729,
        )

    producer = _producer("eval-run-bound")
    result = run_evaluation(
        suite,
        scenario_id=prepared.scenario_id,
        adapter=_spy_launch(),
        prepared=prepared,
        seed=1729,
        producer=producer,
    )
    assert result.producer == producer
    assert result.producer.receipt_sha256 == "d" * 64
    assert result.producer.publication_eligible is True

    scenario = suite.scenarios[1]
    baseline = BaselineAdapter(scenario.baselines[0])
    baseline_result = run_evaluation(
        suite,
        scenario_id=prepared.scenario_id,
        adapter=baseline,
        prepared=prepared,
        seed=1729,
    )
    assert baseline_result.producer.provenance.value == "unissued_baseline"
    assert baseline_result.producer.publication_eligible is False


def test_prepared_payload_hashes_bind_examples_and_all_preparation_contracts() -> None:
    prepared = _regression_data()
    changed_train = (
        prepared.train[0].model_copy(update={"target": 99.0}),
        *prepared.train[1:],
    )
    with pytest.raises(ValueError, match="split_sha256"):
        PreparedScenario(
            scenario_id=prepared.scenario_id,
            binding=prepared.binding,
            source_material=prepared.source_material,
            preparation=prepared.preparation,
            train=changed_train,
            validation=prepared.validation,
            test=prepared.test,
        )

    changed_preparation = prepared.preparation.model_copy(
        update={"mask": {"kind": "artificial", "fraction": 0.15}}
    )
    with pytest.raises(ValueError, match="recipe_sha256"):
        PreparedScenario(
            scenario_id=prepared.scenario_id,
            binding=prepared.binding,
            source_material=prepared.source_material,
            preparation=changed_preparation,
            train=prepared.train,
            validation=prepared.validation,
            test=prepared.test,
        )

    forged_source_binding = prepared.binding.model_copy(update={"source_sha256": "f" * 64})
    with pytest.raises(ValueError, match="retained source bytes"):
        PreparedScenario(
            scenario_id=prepared.scenario_id,
            binding=forged_source_binding,
            source_material=prepared.source_material,
            preparation=prepared.preparation,
            train=prepared.train,
            validation=prepared.validation,
            test=prepared.test,
        )

    forged_truth_binding = prepared.binding.model_copy(
        update={"truth_sidecar_sha256": "e" * 64}
    )
    with pytest.raises(ValueError, match="truth_sidecar_sha256"):
        PreparedScenario(
            scenario_id=prepared.scenario_id,
            binding=forged_truth_binding,
            source_material=prepared.source_material,
            preparation=prepared.preparation,
            train=prepared.train,
            validation=prepared.validation,
            test=prepared.test,
        )

    bypassed = prepared.model_copy(update={"train": changed_train})
    with pytest.raises(DatasetUnavailableError, match="payload_hash_or_contract_drift"):
        run_evaluation(
            _micro_suite(),
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(),
            prepared=bypassed,
            seed=1729,
            producer=_producer(),
        )

    wrong_selection = dict(prepared.preparation.selection)
    wrong_selection["method"] = "all"
    wrong_selection_preparation = PreparationContract(
        preprocessing=prepared.preparation.preprocessing,
        selection=wrong_selection,
        mask=prepared.preparation.mask,
    )
    wrong_selection_binding = prepared.binding.model_copy(
        update={
            "recipe_sha256": PreparedScenario.recipe_sha256_for(
                preparation=wrong_selection_preparation
            )
        }
    )
    wrong_selection_scenario = PreparedScenario(
        scenario_id=prepared.scenario_id,
        binding=wrong_selection_binding,
        source_material=prepared.source_material,
        preparation=wrong_selection_preparation,
        train=prepared.train,
        validation=prepared.validation,
        test=prepared.test,
    )
    with pytest.raises(DatasetUnavailableError, match="selection_contract"):
        run_evaluation(
            _micro_suite(),
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(),
            prepared=wrong_selection_scenario,
            seed=1729,
            producer=_producer(),
        )


@pytest.mark.parametrize(
    "field,payload",
    [
        ("features", {"target": 1.0}),
        ("context", {"nested": {"groundTruth": "secret"}}),
        ("context", {"label": "secret"}),
    ],
)
def test_adapter_visible_payload_rejects_nested_truth_keys(
    field: str, payload: dict[str, object]
) -> None:
    kwargs = {
        "example_id": "leak",
        "target_kind": TargetKind.NUMERIC,
        "target_family": "family",
        "features": {},
        "context": {},
        "target": 1.0,
        field: payload,
    }
    with pytest.raises(ValueError, match="target/truth key"):
        PreparedExample(**kwargs)

    kwargs.pop("target")
    with pytest.raises(ValueError, match="target/truth key"):
        BlindExample(**kwargs)


def test_run_rejects_declared_budget_or_determinism_violation() -> None:
    suite = _micro_suite()
    prepared = _regression_data()
    with pytest.raises(ValueError, match="fit iterations exceed"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(fit_iterations=suite.budget.max_fit_iterations + 1),
            prepared=prepared,
            seed=1729,
            producer=_producer(),
        )
    with pytest.raises(ValueError, match="deterministic execution"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_spy_launch(deterministic=False),
            prepared=prepared,
            seed=1729,
            producer=_producer(),
        )


class FlakyAdapter(SpyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del scenario, fit_examples, seed
        self.calls += 1
        return tuple(
            RawPrediction(example_id=item.example_id, value=float(self.calls)) for item in examples
        )


class SlowAdapter(SpyAdapter):
    def __init__(self, *, delay_seconds: float) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds

    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        wall_time.sleep(self.delay_seconds)
        return super().predict(
            scenario=scenario,
            fit_examples=fit_examples,
            examples=examples,
            seed=seed,
        )


class SlowSpecAdapter(SpyAdapter):
    def __init__(self, *, delay_seconds: float) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds

    @property
    def spec(self) -> AdapterSpec:
        wall_time.sleep(self.delay_seconds)
        return self._spec


def _caller_frames_contain_prepared_truth() -> bool:
    frame = inspect.currentframe()
    while frame is not None:
        if any(isinstance(value, PreparedScenario) for value in frame.f_locals.values()):
            return True
        frame = frame.f_back
    return False


class FrameInspectingAdapter(SpyAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.spec_saw_prepared_truth = False
        self.pickle_saw_prepared_truth = False

    @property
    def spec(self) -> AdapterSpec:
        self.spec_saw_prepared_truth = _caller_frames_contain_prepared_truth()
        return self._spec

    def __reduce_ex__(self, protocol: int) -> object:
        self.pickle_saw_prepared_truth = _caller_frames_contain_prepared_truth()
        return super().__reduce_ex__(protocol)

    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del scenario, fit_examples, seed
        saw_prepared_truth = (
            self.spec_saw_prepared_truth or _caller_frames_contain_prepared_truth()
        )
        return tuple(
            RawPrediction(
                example_id=item.example_id,
                value=8.0 if saw_prepared_truth else 0.0,
                diagnostics={"saw_prepared_truth": saw_prepared_truth},
            )
            for item in examples
        )


class DescriptorInspectingAdapter:
    def __init__(self) -> None:
        self.descriptor_accessed = False
        self.descriptor_saw_prepared_truth = False

    @property
    def __dict__(self) -> dict[str, object]:
        self.descriptor_accessed = True
        self.descriptor_saw_prepared_truth = _caller_frames_contain_prepared_truth()
        return {}


def test_evaluator_detects_nondeterministic_adapter_replay() -> None:
    result = run_evaluation(
        _micro_suite(),
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_adapter_launch(
            FlakyAdapter,
            declared_spec=_model_adapter_spec(),
        ),
        prepared=_regression_data(),
        seed=1729,
        producer=_producer(),
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.failure is not None
    assert result.failure.code == "nondeterministic_replay"


def test_evaluator_enforces_cumulative_hard_adapter_deadline() -> None:
    payload = _micro_suite().model_dump(mode="python")
    payload["budget"]["max_adapter_seconds"] = 1
    suite = EvalSuiteSpec.model_validate(payload)
    result = run_evaluation(
        suite,
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_adapter_launch(
            SlowAdapter,
            declared_spec=_model_adapter_spec(),
            kwargs={"delay_seconds": 0.6},
        ),
        prepared=_regression_data(),
        seed=1729,
        producer=_producer(),
    )
    assert result.status is EvaluationStatus.FAILED
    assert result.failure is not None
    assert result.failure.category is FailureCategory.BUDGET
    assert result.failure.code == "adapter_time_budget_exceeded"

    spec_result = run_evaluation(
        suite,
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_adapter_launch(
            SlowSpecAdapter,
            declared_spec=_model_adapter_spec(),
            kwargs={"delay_seconds": 1.5},
        ),
        prepared=_regression_data(),
        seed=1729,
        producer=_producer("eval-run-slow-spec"),
    )
    assert spec_result.status is EvaluationStatus.FAILED
    assert spec_result.failure is not None
    assert spec_result.failure.category is FailureCategory.BUDGET
    assert spec_result.failure.code == "adapter_time_budget_exceeded"


def test_adapter_process_cannot_inspect_parent_frame_for_test_truth() -> None:
    result = run_evaluation(
        _micro_suite(),
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_adapter_launch(
            FrameInspectingAdapter,
            declared_spec=_model_adapter_spec(),
        ),
        prepared=_regression_data(),
        seed=1729,
        producer=_producer("eval-run-frame-isolation"),
    )
    assert result.status is EvaluationStatus.SUCCEEDED
    assert all(
        prediction.diagnostics == {"saw_prepared_truth": False}
        for prediction in result.raw_predictions
    )
    assert result.metrics["mae"] > 0.0

    in_process = DescriptorInspectingAdapter()
    with pytest.raises(AdapterStateError, match="plain AdapterLaunchSpec"):
        run_evaluation(
            _micro_suite(),
            scenario_id="sklearn-diabetes-regression-micro",
            adapter=in_process,  # type: ignore[arg-type]
            prepared=_regression_data(),
            seed=1729,
            producer=_producer("eval-run-reject-in-process"),
        )
    assert in_process.descriptor_accessed is False
    assert in_process.descriptor_saw_prepared_truth is False


def test_eval_result_rejects_forged_identity_counts_and_metrics() -> None:
    suite = _micro_suite()
    prepared = _regression_data()
    result = run_evaluation(
        suite,
        scenario_id="sklearn-diabetes-regression-micro",
        adapter=_spy_launch(),
        prepared=prepared,
        seed=1729,
        producer=_producer("eval-run-result-integrity"),
    )
    assert verify_result_against_prepared(
        suite,
        result=result,
        prepared=prepared,
    ) == result
    payload = result.model_dump(mode="python")

    forged_id = {**payload, "result_id": "eval-forged"}
    with pytest.raises(ValueError, match="result_id"):
        EvalResult.model_validate(forged_id)

    changed_metrics = result.model_copy(
        update={"metrics": {**result.metrics, "mae": 0.0}}
    )
    forged_metrics = _reidentified_result(
        changed_metrics,
        suite=suite,
        binding=prepared.binding,
    ).model_dump(mode="python")
    with pytest.raises(ValueError, match="not derivable"):
        EvalResult.model_validate(forged_metrics)

    changed_counts = result.model_copy(
        update={"counts": {**result.counts, "scored": 0}}
    )
    forged_counts = _reidentified_result(
        changed_counts,
        suite=suite,
        binding=prepared.binding,
    ).model_dump(mode="python")
    with pytest.raises(ValueError, match="counts"):
        EvalResult.model_validate(forged_counts)

    changed_raw = result.model_copy(
        update={
            "raw_predictions": (
                result.raw_predictions[0].model_copy(update={"value": 0.0}),
                *result.raw_predictions[1:],
            )
        }
    )
    forged_raw = _reidentified_result(
        changed_raw,
        suite=suite,
        binding=prepared.binding,
    ).model_dump(mode="python")
    with pytest.raises(ValueError, match="bind its retained raw prediction"):
        EvalResult.model_validate(forged_raw)

    zero_scores = tuple(
        score.model_copy(
            update={
                "metrics": {
                    **score.metrics,
                    "absolute_error": 0.0,
                    "normalized_squared_error": 0.0,
                    "squared_error": 0.0,
                }
            }
        )
        for score in result.per_example
    )
    coherent_forge = result.model_copy(
        update={
            "per_example": zero_scores,
            "metrics": {
                **result.metrics,
                "mae": 0.0,
                "r2": 1.0,
                "regression_nrmse": 0.0,
            },
        }
    )
    coherent_forge = _reidentified_result(
        coherent_forge,
        suite=suite,
        binding=prepared.binding,
    )
    with pytest.raises(ValueError, match="does not rescore"):
        EvalResult.model_validate(coherent_forge.model_dump(mode="python"))

    changed_truth_scores = tuple(
        score.model_copy(
            update={
                "truth": float(prediction.value),
                "metrics": {
                    "absolute_error": 0.0,
                    "normalized_squared_error": 0.0,
                    "squared_error": 0.0,
                    "truth_centered_squared": 0.0,
                },
            }
        )
        for score, prediction in zip(
            result.per_example,
            result.raw_predictions,
            strict=True,
        )
    )
    changed_truth_examples = tuple(
        PreparedExample(
            example_id=score.example_id,
            target_kind=score.target_kind,
            target_family=score.target_family,
            target=score.truth,
        )
        for score in changed_truth_scores
    )
    changed_truth_hash = PreparedScenario.truth_sidecar_sha256_for(
        test=changed_truth_examples
    )
    changed_truth_binding = prepared.binding.model_copy(
        update={"truth_sidecar_sha256": changed_truth_hash}
    )
    changed_truth_result = result.model_copy(
        update={
            "truth_sidecar_sha256": changed_truth_hash,
            "per_example": changed_truth_scores,
            "metrics": {
                metric_id: (0.0 if metric_id in {"mae", "regression_nrmse"} else value)
                for metric_id, value in result.metrics.items()
                if metric_id != "r2"
            },
        }
    )
    changed_truth_result = _reidentified_result(
        changed_truth_result,
        suite=suite,
        binding=changed_truth_binding,
    )
    loaded_forge = EvalResult.model_validate(changed_truth_result.model_dump(mode="python"))
    with pytest.raises(ValueError, match="canonical prepared snapshot"):
        verify_result_against_prepared(
            suite,
            result=loaded_forge,
            prepared=prepared,
        )


def test_numeric_nrmse_uses_train_only_scale_per_target_family() -> None:
    scenario = load_suite("table-completion-micro-v0").scenarios[1]
    fit = (
        PreparedExample(
            example_id="fit-small-0",
            target_kind="numeric",
            target_family="small",
            target=0.0,
        ),
        PreparedExample(
            example_id="fit-small-1",
            target_kind="numeric",
            target_family="small",
            target=2.0,
        ),
        PreparedExample(
            example_id="fit-large-0",
            target_kind="numeric",
            target_family="large",
            target=0.0,
        ),
        PreparedExample(
            example_id="fit-large-1",
            target_kind="numeric",
            target_family="large",
            target=200.0,
        ),
    )
    truth = (
        PreparedExample(
            example_id="test-small",
            target_kind="numeric",
            target_family="small",
            target=1.0,
        ),
        PreparedExample(
            example_id="test-large",
            target_kind="numeric",
            target_family="large",
            target=100.0,
        ),
    )
    predictions = (
        RawPrediction(example_id="test-small", value=2.0),
        RawPrediction(example_id="test-large", value=200.0),
    )
    scored = score_predictions(
        scenario=scenario,
        fit_examples=fit,
        truth=truth,
        predictions=predictions,
    )
    assert scored.metrics["numeric_nrmse"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="no train-only normalization scale"):
        score_predictions(
            scenario=scenario,
            fit_examples=fit[:2],
            truth=truth,
            predictions=predictions,
        )


def test_categorical_nll_requires_complete_train_supported_distribution() -> None:
    scenario = load_suite("table-supervised-micro-v0").scenarios[0]
    fit = (
        PreparedExample(
            example_id="fit-negative",
            target_kind="categorical",
            target_family="income",
            target="no",
        ),
        PreparedExample(
            example_id="fit-positive",
            target_kind="categorical",
            target_family="income",
            target="yes",
        ),
    )
    truth = (
        PreparedExample(
            example_id="test-income",
            target_kind="categorical",
            target_family="income",
            target="yes",
        ),
    )
    for probabilities in (
        {"no": 0.4, "yes": 0.4},
        {"no": -0.1, "yes": 1.1},
        {"no": float("nan"), "yes": float("nan")},
    ):
        with pytest.raises(ValueError, match="probabilities"):
            RawPrediction(
                example_id="test-income",
                value="yes",
                probabilities=probabilities,
            )
    mutated = RawPrediction(
        example_id="test-income",
        value="yes",
        probabilities={"no": 0.25, "yes": 0.75},
    )
    assert mutated.probabilities is not None
    mutated.probabilities["yes"] = float("nan")
    with pytest.raises(ValueError, match="probabilities"):
        score_predictions(
            scenario=scenario,
            fit_examples=fit,
            truth=truth,
            predictions=(mutated,),
        )
    with pytest.raises(ValueError, match="complete probability distribution"):
        score_predictions(
            scenario=scenario,
            fit_examples=fit,
            truth=truth,
            predictions=(RawPrediction(example_id="test-income", value="yes"),),
        )
    with pytest.raises(ValueError, match="exactly match"):
        score_predictions(
            scenario=scenario,
            fit_examples=fit,
            truth=truth,
            predictions=(
                RawPrediction(
                    example_id="test-income",
                    value="yes",
                    probabilities={"yes": 1.0},
                ),
            ),
        )
    scored = score_predictions(
        scenario=scenario,
        fit_examples=fit,
        truth=truth,
        predictions=(
            RawPrediction(
                example_id="test-income",
                value="yes",
                probabilities={"no": 0.25, "yes": 0.75},
            ),
        ),
    )
    assert scored.metrics["classification_nll"] == pytest.approx(-math.log(0.75))


def test_scoring_can_use_explicit_full_train_fitted_state() -> None:
    scenario = load_suite("table-supervised-micro-v0").scenarios[0]
    fit = (
        PreparedExample(
            example_id="fit-negative",
            target_kind="categorical",
            target_family="income",
            target="no",
        ),
    )
    truth = (
        PreparedExample(
            example_id="test-income",
            target_kind="categorical",
            target_family="income",
            target="yes",
        ),
    )
    prediction = RawPrediction(
        example_id="test-income",
        value="yes",
        probabilities={"no": 0.25, "yes": 0.75},
    )
    with pytest.raises(ValueError, match="outside train-only label support"):
        score_predictions(
            scenario=scenario,
            fit_examples=fit,
            truth=truth,
            predictions=(prediction,),
        )
    scored = score_predictions(
        scenario=scenario,
        fit_examples=fit,
        truth=truth,
        predictions=(prediction,),
        train_only_fitted_state={
            "families": {"income": {"kind": "categorical", "domain": ["no", "yes"]}}
        },
    )
    assert scored.metrics["classification_nll"] == pytest.approx(-math.log(0.75))


def test_full_train_domain_reaches_smoothed_mode_and_knn_categorical_fallback() -> None:
    scenario = load_suite("table-completion-micro-v1").scenarios[0]
    fit = (
        PreparedExample(
            example_id="fit-country-us",
            target_kind="categorical",
            target_family="country",
            target="US",
        ),
        PreparedExample(
            example_id="fit-age",
            target_kind="numeric",
            target_family="age",
            features={"x": 0.0},
            target=20.0,
        ),
    )
    examples = (
        BlindExample(
            example_id="test-country",
            target_kind="categorical",
            target_family="country",
        ),
    )
    fitted_state = {
        "families": {
            "country": {
                "kind": "categorical",
                "domain": ["CA", "US"],
            },
            "age": {"kind": "numeric", "mean": 20.0, "scale": 1.0},
        }
    }
    for baseline_id in ("train-mean-mode", "numeric-knn-imputation"):
        frozen = next(
            item for item in scenario.baselines if item.baseline_id == baseline_id
        )
        prediction = BaselineAdapter(
            frozen,
            train_only_fitted_state=fitted_state,
        ).predict(
            scenario=scenario,
            fit_examples=fit,
            examples=examples,
            seed=1729,
        )[0]
        assert prediction.value == "US"
        assert prediction.probabilities == {"CA": 1 / 3, "US": 2 / 3}


class TopologyAdapter:
    def __init__(self) -> None:
        self._spec = AdapterSpec(
            adapter_id="topology-model",
            adapter_version="1.0.0",
            kind="model",
            fit_iterations=0,
            device_class="single_device",
            deterministic=True,
            contract_id="tabu4graph",
            artifact_id="artifact-topology",
        )

    @property
    def spec(self) -> AdapterSpec:
        return self._spec

    def predict(
        self,
        *,
        scenario: object,
        fit_examples: Sequence[PreparedExample],
        examples: Sequence[BlindExample],
        seed: int,
    ) -> Sequence[RawPrediction]:
        del scenario, fit_examples, seed
        outputs = []
        for item in examples:
            changed = item.features.get("signal") == 1
            outputs.append(
                RawPrediction(
                    example_id=item.example_id,
                    value="B" if changed else "A",
                    probabilities={"A": 0.25 if changed else 0.75, "B": 0.75 if changed else 0.25},
                    diagnostics={
                        "topology_perturbation_pass": True,
                        "locality_contract_pass": True,
                    },
                )
            )
        return tuple(outputs)


def _topology_launch() -> AdapterLaunchSpec:
    return _adapter_launch(
        TopologyAdapter,
        declared_spec=_model_adapter_spec(
            adapter_id="topology-model",
            contract_id="tabu4graph",
            artifact_id="artifact-topology",
        ),
    )


def _graph_data(*, include_pairs: bool) -> PreparedScenario:
    scenario = _graph_suite().scenarios[0]
    train = (
        PreparedExample(
            example_id="graph-train-a",
            target_kind="categorical",
            target_family="club",
            target="A",
        ),
        PreparedExample(
            example_id="graph-train-b",
            target_kind="categorical",
            target_family="club",
            target="B",
        ),
    )
    validation = (
        PreparedExample(
            example_id="graph-validation",
            target_kind="categorical",
            target_family="club",
            target="A",
        ),
    )
    test = (
        PreparedExample(
            example_id="graph-test",
            target_kind="categorical",
            target_family="club",
            features={"signal": 0},
            context={"neighbor_labels": ["A", "B"]},
            target="A",
        ),
    )
    topology_checks = (
        (
            TopologyCheckCase(
                check_id="topology_perturbation_pass",
                base_example_id="graph-test",
                perturbed_example=BlindExample(
                    example_id="graph-perturbed-topology",
                    target_kind="categorical",
                    target_family="club",
                    features={"signal": 1},
                    context={"neighbor_labels": ["B"]},
                ),
                expected_relation="different",
            ),
            TopologyCheckCase(
                check_id="locality_contract_pass",
                base_example_id="graph-test",
                perturbed_example=BlindExample(
                    example_id="graph-perturbed-locality",
                    target_kind="categorical",
                    target_family="club",
                    features={"signal": 0, "unrelated_component": 1},
                    context={"neighbor_labels": ["A", "B"]},
                ),
                expected_relation="equal",
            ),
        )
        if include_pairs
        else ()
    )
    preparation = PreparationContract(
        preprocessing={
            "implementation_sha256": "1" * 64,
            "fitted_state_sha256": "2" * 64,
            "fit_partition": "train",
        },
        selection=scenario.selection.model_dump(mode="python"),
        mask={"kind": "none"},
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id="zachary-karate-club",
        content=b"zachary-karate-club:frozen-topology-v1",
        media_type="text/plain",
    )
    return PreparedScenario(
        scenario_id="zachary-karate-club-label-completion",
        binding=DatasetSnapshotBinding(
            dataset_id="zachary-karate-club",
            source_sha256=source_material.raw_sha256,
            split_sha256=PreparedScenario.split_sha256_for(
                train=train, validation=validation, test=test
            ),
            recipe_sha256=PreparedScenario.recipe_sha256_for(
                preparation=preparation,
                topology_checks=topology_checks,
            ),
            truth_sidecar_sha256=PreparedScenario.truth_sidecar_sha256_for(test=test),
            partition_counts={"train": 2, "validation": 1, "test": 1},
        ),
        source_material=source_material,
        preparation=preparation,
        train=train,
        validation=validation,
        test=test,
        topology_checks=topology_checks,
    )


def _graph_suite() -> EvalSuiteSpec:
    payload = load_suite("graph-completion-micro-v0").model_dump(mode="python")
    payload["scenarios"][0]["selection"]["partition_limits"] = {
        "train": 2,
        "validation": 1,
        "test": 1,
    }
    return EvalSuiteSpec.model_validate(payload)


def test_topology_checks_use_evaluator_owned_pairs_and_fail_closed_without_them() -> None:
    suite = _graph_suite()
    with pytest.raises(DatasetUnavailableError, match="topology_pairs"):
        run_evaluation(
            suite,
            scenario_id="zachary-karate-club-label-completion",
            adapter=_topology_launch(),
            prepared=_graph_data(include_pairs=False),
            seed=1729,
            producer=_producer("eval-run-graph-missing"),
        )

    prepared = _graph_data(include_pairs=True)
    result = run_evaluation(
        suite,
        scenario_id="zachary-karate-club-label-completion",
        adapter=_topology_launch(),
        prepared=prepared,
        seed=1729,
        producer=_producer("eval-run-graph"),
    )
    assert result.status is EvaluationStatus.SUCCEEDED
    assert result.metrics["topology_perturbation_pass"] == 1.0
    assert result.metrics["locality_contract_pass"] == 1.0
    assert len(result.topology_checks) == 2

    forged_check = result.topology_checks[0].model_copy(
        update={"passed": not result.topology_checks[0].passed}
    )
    forged = result.model_copy(
        update={"topology_checks": (forged_check, *result.topology_checks[1:])}
    )
    forged = _reidentified_result(forged, suite=suite, binding=prepared.binding)
    with pytest.raises(ValueError, match="topology pass flag"):
        EvalResult.model_validate(forged.model_dump(mode="python"))


def test_projected_graph_inputs_require_atomic_exact_topology_parity() -> None:
    suite = _graph_suite()
    prepared = _graph_data(include_pairs=True)
    projected_test = tuple(
        BlindExample(
            example_id=item.example_id,
            target_kind=item.target_kind,
            target_family=item.target_family,
            features={**item.features, "projection_schema": "test.graph-projection.v1"},
            context=item.context,
        )
        for item in prepared.test
    )
    projected_cases = tuple(
        case.model_copy(
            update={
                "perturbed_example": case.perturbed_example.model_copy(
                    update={
                        "features": {
                            **case.perturbed_example.features,
                            "projection_schema": "test.graph-projection.v1",
                        }
                    }
                )
            }
        )
        for case in prepared.topology_checks
    )

    result = run_evaluation(
        suite,
        scenario_id=prepared.scenario_id,
        adapter=_topology_launch(),
        prepared=prepared,
        seed=1729,
        producer=_producer("eval-run-graph-projected"),
        blind_examples=projected_test,
        topology_cases=projected_cases,
    )

    assert result.status is EvaluationStatus.SUCCEEDED
    assert [item.perturbed_example_sha256 for item in result.topology_checks] == [
        item.perturbed_example.content_hash for item in projected_cases
    ]
    with pytest.raises(ValueError, match="blind examples and topology cases together"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_topology_launch(),
            prepared=prepared,
            seed=1729,
            producer=_producer("eval-run-graph-half-projected"),
            blind_examples=projected_test,
        )
    wrong_family = projected_cases[0].model_copy(
        update={
            "perturbed_example": projected_cases[0].perturbed_example.model_copy(
                update={"target_family": "wrong-family"}
            )
        }
    )
    with pytest.raises(ValueError, match="target kind and family"):
        run_evaluation(
            suite,
            scenario_id=prepared.scenario_id,
            adapter=_topology_launch(),
            prepared=prepared,
            seed=1729,
            producer=_producer("eval-run-graph-wrong-family"),
            blind_examples=projected_test,
            topology_cases=(wrong_family, projected_cases[1]),
        )


def test_neighbor_mode_retains_complete_train_label_support_for_nll() -> None:
    suite = _graph_suite()
    scenario = suite.scenarios[0]
    baseline = next(item for item in scenario.baselines if item.baseline_id == "graph-one-hop-mode")
    result = run_evaluation(
        suite,
        scenario_id=scenario.scenario_id,
        adapter=BaselineAdapter(baseline),
        prepared=_graph_data(include_pairs=True),
        seed=1729,
    )
    assert result.status is EvaluationStatus.SUCCEEDED
    assert result.metrics["categorical_nll"] > 0.0
    assert result.raw_predictions[0].value == "A"
    assert result.raw_predictions[0].probabilities == {"A": 0.5, "B": 0.5}
    for prediction in (
        *result.raw_predictions,
        *(item.perturbed_prediction for item in result.topology_checks),
    ):
        assert prediction.probabilities is not None
        assert set(prediction.probabilities) == {"A", "B"}
        assert all(value > 0.0 for value in prediction.probabilities.values())
