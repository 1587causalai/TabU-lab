"""Validate, dry-run, execute, and compare frozen evaluation suites."""

from __future__ import annotations

import importlib
import math
import multiprocessing as mp
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from multiprocessing.connection import Connection
from typing import Any

from tabu_lab.contracts import canonical_hash

from .adapters import BaselineAdapter
from .contracts import (
    AdapterKind,
    AdapterLaunchSpec,
    AdapterSpec,
    BaselineSpec,
    BlindExample,
    ComparisonReport,
    DatasetSnapshotBinding,
    DryRunReport,
    EvalProducerBinding,
    EvalResult,
    EvalSuiteSpec,
    EvaluationFailure,
    EvaluationStatus,
    FailureCategory,
    PreparedExample,
    PreparedScenario,
    ProducerProvenance,
    RawPrediction,
    ScenarioAvailability,
    ScenarioSpec,
    SuiteValidationReport,
    TopologyCheckCase,
    TopologyCheckResult,
    TopologyRelation,
    comparison_publication_eligible,
    derive_comparison_summary,
    deterministic_result_id,
)
from .scoring import score_predictions


class DatasetUnavailableError(RuntimeError):
    """Raised when a real, hash-bound prepared snapshot is not available."""


class AdapterIsolationError(RuntimeError):
    """Raised when an adapter cannot complete inside the isolated worker."""


class AdapterStateError(ValueError):
    """Raised when an adapter cannot be reconstructed from evaluator-safe state."""


def _plain_json_value(value: Any) -> Any:
    value_type = type(value)
    if value is None or value_type in {bool, int, float, str}:
        return value
    if value_type is list:
        return [_plain_json_value(item) for item in value]
    if value_type is tuple:
        return [_plain_json_value(item) for item in value]
    if value_type is dict:
        if any(type(key) is not str for key in value):
            raise AdapterStateError("adapter launch mappings require plain string keys")
        return {key: _plain_json_value(item) for key, item in dict.items(value)}
    value_type_name = type.__getattribute__(value_type, "__qualname__")
    raise AdapterStateError(
        f"adapter launch manifests allow only inert JSON values; found {value_type_name}"
    )


def _validated_declared_spec(spec: object) -> AdapterSpec:
    if type(spec) is not AdapterSpec:
        raise AdapterStateError("adapter launch declared_spec must be the exact contract type")
    state = object.__getattribute__(spec, "__dict__")
    if type(state) is not dict:
        raise AdapterStateError("adapter launch declared_spec state is malformed")
    payload: dict[str, Any] = {}
    for field_name in AdapterSpec.model_fields:
        if field_name not in state:
            raise AdapterStateError("adapter launch declared_spec is incomplete")
        value = dict.__getitem__(state, field_name)
        payload[field_name] = (
            value.value if type(value) is AdapterKind else _plain_json_value(value)
        )
    return AdapterSpec.model_validate(payload)


def _validated_plain_launch(launch: AdapterLaunchSpec) -> AdapterLaunchSpec:
    state = object.__getattribute__(launch, "__dict__")
    if type(state) is not dict:
        raise AdapterStateError("adapter launch manifest state is malformed")
    required = {"schema_version", "module", "qualname", "kwargs", "declared_spec"}
    if set(state) != required:
        raise AdapterStateError("adapter launch manifest fields are incomplete")
    return AdapterLaunchSpec(
        schema_version=_plain_json_value(dict.__getitem__(state, "schema_version")),
        module=_plain_json_value(dict.__getitem__(state, "module")),
        qualname=_plain_json_value(dict.__getitem__(state, "qualname")),
        kwargs=_plain_json_value(dict.__getitem__(state, "kwargs")),
        declared_spec=_validated_declared_spec(dict.__getitem__(state, "declared_spec")),
    )


def _baseline_launch_spec(
    adapter: BaselineAdapter,
    *,
    train_only_fitted_state: Mapping[str, object] | None = None,
) -> AdapterLaunchSpec:
    baseline = object.__getattribute__(adapter, "baseline")
    declared_spec = object.__getattribute__(adapter, "_spec")
    if type(baseline) is not BaselineSpec or type(declared_spec) is not AdapterSpec:
        raise AdapterStateError("evaluator-owned baseline adapter state was mutated")
    baseline_state = object.__getattribute__(baseline, "__dict__")
    if type(baseline_state) is not dict:
        raise AdapterStateError("evaluator-owned baseline contract state was mutated")
    baseline_payload = {
        field_name: _plain_json_value(dict.__getitem__(baseline_state, field_name))
        for field_name in BaselineSpec.model_fields
    }
    kwargs: dict[str, Any] = {"baseline": baseline_payload}
    if train_only_fitted_state is not None:
        kwargs["train_only_fitted_state"] = _plain_json_value(
            dict(train_only_fitted_state)
        )
    return AdapterLaunchSpec(
        module="tabu_lab.evaluation.foundry.adapters",
        qualname="BaselineAdapter",
        kwargs=kwargs,
        declared_spec=_validated_declared_spec(declared_spec),
    )


def _coerce_adapter_launch(
    adapter: object,
    *,
    train_only_fitted_state: Mapping[str, object] | None = None,
) -> AdapterLaunchSpec:
    """Accept only inert launch data or the exact evaluator-owned baseline class."""

    if type(adapter) is AdapterLaunchSpec:
        return _validated_plain_launch(adapter)
    if type(adapter) is BaselineAdapter:
        return _baseline_launch_spec(
            adapter,
            train_only_fitted_state=train_only_fitted_state,
        )
    raise AdapterStateError(
        "formal model evaluation requires a plain AdapterLaunchSpec; "
        "untrusted in-process adapter objects are forbidden"
    )


def _adapter_from_launch_manifest(envelope: Any) -> tuple[Any, AdapterSpec]:
    launch = AdapterLaunchSpec.model_validate(envelope)
    resolved: Any = importlib.import_module(launch.module)
    for component in launch.qualname.split("."):
        namespace = vars(resolved)
        if component not in namespace:
            raise AdapterStateError("isolated adapter class identity is not importable")
        resolved = namespace[component]
    if not isinstance(resolved, type):
        raise AdapterStateError("isolated adapter identity does not resolve to a class")
    instance = resolved(**dict(launch.kwargs))
    raw_spec = instance.spec
    actual_spec = AdapterSpec.model_validate(
        raw_spec.model_dump(mode="python") if isinstance(raw_spec, AdapterSpec) else raw_spec
    )
    if actual_spec != launch.declared_spec:
        raise AdapterStateError("isolated adapter spec differs from its launch manifest")
    return instance, actual_spec


def _isolated_adapter_worker(
    sender: Connection,
    adapter_envelope: dict[str, Any],
    scenario: ScenarioSpec,
    fit_examples: tuple[PreparedExample, ...],
    batches: tuple[tuple[BlindExample, ...], ...],
    seed: int,
) -> None:
    """Execute all adapter calls without any parent frame or test truth object."""

    try:
        adapter, _ = _adapter_from_launch_manifest(adapter_envelope)
        outputs: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
        for examples in batches:
            predictions = tuple(
                adapter.predict(
                    scenario=scenario,
                    fit_examples=fit_examples,
                    examples=examples,
                    seed=seed,
                )
            )
            replay = tuple(
                adapter.predict(
                    scenario=scenario,
                    fit_examples=fit_examples,
                    examples=examples,
                    seed=seed,
                )
            )
            outputs.append((predictions, replay))
        sender.send(("ok", tuple(outputs)))
    except BaseException as error:
        with suppress(BaseException):
            sender.send(("error", type(error).__name__))
    finally:
        sender.close()


def _isolated_spec_worker(sender: Connection, adapter_envelope: dict[str, Any]) -> None:
    """Resolve adapter identity without exposing any evaluation caller frame."""

    try:
        _, adapter_spec = _adapter_from_launch_manifest(adapter_envelope)
        sender.send(("ok", adapter_spec))
    except BaseException as error:
        with suppress(BaseException):
            sender.send(("error", type(error).__name__))
    finally:
        sender.close()


def _terminate_process(process: mp.Process) -> None:
    if process.pid is None:
        return
    if not process.is_alive():
        process.join(timeout=0.1)
        return
    process.terminate()
    process.join(timeout=0.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=0.5)


def _execute_adapter_isolated(
    *,
    adapter_envelope: dict[str, Any],
    scenario: ScenarioSpec,
    fit_examples: tuple[PreparedExample, ...],
    batches: tuple[tuple[BlindExample, ...], ...],
    seed: int,
    deadline_seconds: float,
) -> tuple[tuple[tuple[RawPrediction, ...], tuple[RawPrediction, ...]], ...]:
    """Run every adapter batch and replay under one cumulative hard deadline."""

    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_adapter_worker,
        args=(sender, adapter_envelope, scenario, fit_examples, batches, seed),
        daemon=True,
    )
    started = time.monotonic()
    try:
        process.start()
        sender.close()
        elapsed = time.monotonic() - started
        remaining = max(0.0, deadline_seconds - elapsed)
        if not receiver.poll(remaining):
            raise TimeoutError("adapter exceeded cumulative isolated deadline")
        try:
            status, payload = receiver.recv()
        except EOFError as error:
            raise AdapterIsolationError("isolated adapter exited without a result") from error
        if time.monotonic() - started > deadline_seconds:
            raise TimeoutError("adapter exceeded cumulative isolated deadline")
        if status != "ok":
            raise AdapterIsolationError(f"isolated adapter raised {payload}")
        validated: list[tuple[tuple[RawPrediction, ...], tuple[RawPrediction, ...]]] = []
        for predictions, replay in payload:
            validated.append(
                (
                    tuple(
                        RawPrediction.model_validate(
                            item.model_dump(mode="python")
                            if isinstance(item, RawPrediction)
                            else item
                        )
                        for item in predictions
                    ),
                    tuple(
                        RawPrediction.model_validate(
                            item.model_dump(mode="python")
                            if isinstance(item, RawPrediction)
                            else item
                        )
                        for item in replay
                    ),
                )
            )
        if len(validated) != len(batches):
            raise AdapterIsolationError("isolated adapter returned the wrong batch count")
        return tuple(validated)
    finally:
        receiver.close()
        sender.close()
        _terminate_process(process)


def _read_adapter_spec_isolated(
    *, adapter_envelope: dict[str, Any], deadline_seconds: float
) -> AdapterSpec:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_spec_worker,
        args=(sender, adapter_envelope),
        daemon=True,
    )
    started = time.monotonic()
    try:
        process.start()
        sender.close()
        remaining = max(0.0, deadline_seconds - (time.monotonic() - started))
        if not receiver.poll(remaining):
            raise TimeoutError("adapter spec exceeded cumulative isolated deadline")
        try:
            status, payload = receiver.recv()
        except EOFError as error:
            raise AdapterIsolationError("isolated adapter spec exited without a result") from error
        if time.monotonic() - started > deadline_seconds:
            raise TimeoutError("adapter spec exceeded cumulative isolated deadline")
        if status != "ok":
            raise AdapterIsolationError(f"isolated adapter spec raised {payload}")
        return AdapterSpec.model_validate(
            payload.model_dump(mode="python") if isinstance(payload, AdapterSpec) else payload
        )
    finally:
        receiver.close()
        sender.close()
        _terminate_process(process)


_ALLOWED_METRICS = {
    "classification_nll",
    "regression_nrmse",
    "numeric_nrmse",
    "numeric_mae",
    "categorical_nll",
    "categorical_accuracy",
    "accuracy",
    "auroc",
    "mae",
    "r2",
    "rmse",
    "coverage",
    "abstention",
    "topology_perturbation_pass",
    "locality_contract_pass",
}

_TOPOLOGY_CHECK_RELATIONS = {
    "topology_perturbation_pass": TopologyRelation.DIFFERENT,
    "locality_contract_pass": TopologyRelation.EQUAL,
}


def validate_suite(suite: EvalSuiteSpec) -> SuiteValidationReport:
    issues: list[str] = []
    for scenario in suite.scenarios:
        unknown_metrics = sorted(
            metric.metric_id
            for metric in scenario.metrics
            if metric.metric_id not in _ALLOWED_METRICS
        )
        if unknown_metrics:
            issues.append(f"{scenario.scenario_id}: unknown metrics {unknown_metrics}")
        if "tabu4do" in scenario.applicable_contracts:
            issues.append(f"{scenario.scenario_id}: tabu4do remains design_open")
        if scenario.task.value.startswith("supervised_") and scenario.mask is not None:
            issues.append(f"{scenario.scenario_id}: supervised truth cannot be artificial masking")
        if scenario.task.value.endswith("completion") and "coverage" not in {
            metric.metric_id for metric in scenario.metrics
        }:
            issues.append(f"{scenario.scenario_id}: completion metrics must include coverage")
        if scenario.task.value == "table_completion" and (
            scenario.mask is None or scenario.mask.fraction != 0.15
        ):
            issues.append(f"{scenario.scenario_id}: table completion mask must be exactly 15%")
        if scenario.task.value == "recsys_completion" and (
            scenario.selection.users != 64 or scenario.selection.items != 128
        ):
            issues.append(f"{scenario.scenario_id}: recsys v0 must freeze 64 users x 128 items")
        target_kinds = set(scenario.target_kinds)
        for metric in scenario.metrics:
            if metric.target_kind is not None and metric.target_kind not in target_kinds:
                issues.append(
                    f"{scenario.scenario_id}: metric {metric.metric_id} has absent target kind"
                )
            if "nrmse" in metric.metric_id and not metric.train_only_normalization:
                issues.append(
                    f"{scenario.scenario_id}: {metric.metric_id} must use train-only normalization"
                )
    return SuiteValidationReport(
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        valid=not issues,
        issues=tuple(issues),
    )


def _scenario(suite: EvalSuiteSpec, scenario_id: str) -> ScenarioSpec:
    matches = [item for item in suite.scenarios if item.scenario_id == scenario_id]
    if not matches:
        raise KeyError(f"unknown evaluation scenario: {scenario_id}")
    return matches[0]


def _prepared_blockers(
    scenario: ScenarioSpec, prepared: PreparedScenario | None
) -> tuple[str, ...]:
    if prepared is None:
        detail = "prepared_snapshot_missing"
        if scenario.dataset.network_required:
            detail += ":external_source_requires_explicit_fetch_and_sha256_verification"
        return (detail,)
    blockers: list[str] = []
    try:
        PreparedScenario.model_validate(prepared.model_dump(mode="python"))
    except ValueError:
        blockers.append("prepared_payload_hash_or_contract_drift")
    if prepared.scenario_id != scenario.scenario_id:
        blockers.append("prepared_scenario_id_mismatch")
    if prepared.binding.dataset_id != scenario.dataset.dataset_id:
        blockers.append("prepared_dataset_id_mismatch")
    if scenario.selection.partition_limits and scenario.task.value == "table_completion":
        execution = prepared.preparation.preprocessing.get("execution")
        selected_row_counts = (
            execution.get("selected_row_counts") if isinstance(execution, Mapping) else None
        )
        masked_target_counts = (
            execution.get("masked_target_counts") if isinstance(execution, Mapping) else None
        )
        if selected_row_counts != scenario.selection.partition_limits:
            blockers.append(
                "prepared_selected_row_counts_do_not_match_frozen_micro_split"
            )
        if masked_target_counts != prepared.binding.partition_counts:
            blockers.append(
                "prepared_masked_target_counts_do_not_match_snapshot_binding"
            )
    elif scenario.selection.partition_limits:
        expected_counts = scenario.selection.partition_limits
        if prepared.binding.partition_counts != expected_counts:
            blockers.append("prepared_partition_counts_do_not_match_frozen_micro_split")
    if prepared.preparation.selection != scenario.selection.model_dump(mode="python"):
        blockers.append("prepared_selection_contract_does_not_match_suite")
    expected_mask = (
        scenario.mask.model_dump(mode="python") if scenario.mask is not None else {"kind": "none"}
    )
    if prepared.preparation.mask != expected_mask:
        blockers.append("prepared_mask_contract_does_not_match_suite")
    if (
        prepared.preparation.preprocessing.get("fit_partition")
        != scenario.preprocessing_fit_partition
    ):
        blockers.append("prepared_preprocessing_partition_does_not_match_suite")
    kinds = {item.target_kind for item in (*prepared.train, *prepared.test)}
    if not kinds.issubset(set(scenario.target_kinds)):
        blockers.append("prepared_target_kind_outside_scenario_contract")
    if scenario.mask is None and prepared.binding.mask_applied_after_split is not True:
        blockers.append("invalid_split_boundary")
    declared_topology_checks = set(scenario.topology_contract_checks)
    prepared_topology_checks = {item.check_id for item in prepared.topology_checks}
    if declared_topology_checks != prepared_topology_checks:
        blockers.append("prepared_topology_pairs_do_not_match_frozen_contract")
    if any(
        item.expected_relation is not _TOPOLOGY_CHECK_RELATIONS.get(item.check_id)
        for item in prepared.topology_checks
    ):
        blockers.append("prepared_topology_relation_does_not_match_evaluator_protocol")
    return tuple(blockers)


def dry_run_suite(
    suite: EvalSuiteSpec,
    *,
    prepared: Mapping[str, PreparedScenario] | None = None,
) -> DryRunReport:
    validation = validate_suite(suite)
    prepared = prepared or {}
    availability: list[ScenarioAvailability] = []
    for scenario in suite.scenarios:
        blockers = list(_prepared_blockers(scenario, prepared.get(scenario.scenario_id)))
        blockers.extend(validation.issues)
        availability.append(
            ScenarioAvailability(
                scenario_id=scenario.scenario_id,
                ready=not blockers,
                network_required=scenario.dataset.network_required,
                blockers=tuple(blockers),
            )
        )
    return DryRunReport(
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        ready=all(item.ready for item in availability),
        scenarios=tuple(availability),
    )


def _validate_run_inputs(
    *,
    suite: EvalSuiteSpec,
    scenario: ScenarioSpec,
    prepared: PreparedScenario,
    adapter_spec: AdapterSpec,
    producer: EvalProducerBinding,
    seed: int,
) -> DatasetSnapshotBinding:
    validation = validate_suite(suite)
    if not validation.valid:
        raise ValueError(f"invalid evaluation suite: {validation.issues}")
    blockers = _prepared_blockers(scenario, prepared)
    if blockers:
        raise DatasetUnavailableError(
            "prepared dataset is unavailable or incompatible: " + ", ".join(blockers)
        )
    if seed not in suite.budget.model_seeds:
        raise ValueError("evaluation seed is outside the frozen suite budget")
    if adapter_spec.kind is AdapterKind.MODEL:
        if adapter_spec.contract_id not in scenario.applicable_contracts:
            raise ValueError("model adapter contract is not applicable to this scenario")
        if (
            scenario.applicable_profiles
            and adapter_spec.profile_id not in scenario.applicable_profiles
        ):
            raise ValueError("model adapter profile is not applicable to this scenario")
        if producer.provenance is not ProducerProvenance.RECEIPTED_RUN:
            raise ValueError("model evaluation requires a receipted producer run")
    else:
        baseline_by_id = {item.baseline_id: item for item in scenario.baselines}
        baseline = baseline_by_id.get(adapter_spec.adapter_id)
        if baseline is None or baseline.family != adapter_spec.baseline_family:
            raise ValueError("baseline adapter is not frozen by this scenario")
        if producer.provenance is ProducerProvenance.UNISSUED_BASELINE and (
            producer.publication_eligible
        ):
            raise ValueError("unissued baseline evaluation cannot be public evidence")
    if adapter_spec.fit_iterations > suite.budget.max_fit_iterations:
        raise ValueError("adapter fit iterations exceed the frozen evaluation budget")
    if adapter_spec.device_class != suite.budget.device_class:
        raise ValueError("adapter device class differs from the frozen evaluation budget")
    if suite.budget.deterministic and not adapter_spec.deterministic:
        raise ValueError("adapter refuses the frozen deterministic execution contract")
    return prepared.binding


def _failed_result(
    *,
    suite: EvalSuiteSpec,
    scenario: ScenarioSpec,
    adapter_spec: AdapterSpec,
    producer: EvalProducerBinding,
    binding: DatasetSnapshotBinding,
    seed: int,
    category: FailureCategory,
    code: str,
    public_detail: str,
) -> EvalResult:
    counts = {"targets": 0, "scored": 0, "abstained": 0}
    failure_counts = {category: 1}
    failure = EvaluationFailure(
        category=category,
        code=code,
        public_detail=public_detail,
    )
    return EvalResult(
        result_id=deterministic_result_id(
            suite=suite,
            scenario=scenario,
            adapter=adapter_spec,
            binding=binding,
            producer=producer,
            seed=seed,
            status=EvaluationStatus.FAILED,
            raw_predictions=(),
            topology_checks=(),
            per_example=(),
            metrics={},
            counts=counts,
            failure_counts=failure_counts,
            coverage=0.0,
            failure=failure,
            claim_boundary=suite.claim_boundary,
        ),
        status=EvaluationStatus.FAILED,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        adapter=adapter_spec,
        producer=producer,
        seed=seed,
        source_sha256=binding.source_sha256,
        split_sha256=binding.split_sha256,
        recipe_sha256=binding.recipe_sha256,
        budget_hash=suite.budget.content_hash,
        truth_sidecar_sha256=binding.truth_sidecar_sha256,
        counts=counts,
        failure_counts=failure_counts,
        coverage=0.0,
        failure=failure,
        claim_boundary=suite.claim_boundary,
    )


def _unissued_baseline_producer() -> EvalProducerBinding:
    return EvalProducerBinding(
        provenance=ProducerProvenance.UNISSUED_BASELINE,
        publication_eligible=False,
    )


def _prediction_semantics_hash(prediction: object) -> str:
    from .contracts import RawPrediction

    parsed = RawPrediction.model_validate(prediction)
    if parsed.abstained:
        raise ValueError("topology contract checks cannot be verified from abstentions")
    return canonical_hash(
        {
            "schema": "tabu.eval-prediction-semantics.v1",
            "value": parsed.value,
            "probabilities": parsed.probabilities,
        }
    )


def _blind_test_examples(
    prepared: PreparedScenario,
    supplied: Sequence[BlindExample] | None,
) -> tuple[BlindExample, ...]:
    """Return evaluator-visible test inputs in canonical truth-sidecar order."""

    if supplied is None:
        return tuple(
            BlindExample(
                example_id=item.example_id,
                target_kind=item.target_kind,
                target_family=item.target_family,
                features=item.features,
                context=item.context,
            )
            for item in prepared.test
        )
    parsed = tuple(
        BlindExample.model_validate(
            item.model_dump(mode="python") if isinstance(item, BlindExample) else item
        )
        for item in supplied
    )
    by_id = {item.example_id: item for item in parsed}
    if len(by_id) != len(parsed):
        raise ValueError("supplied blind example ids must be unique")
    expected_ids = {item.example_id for item in prepared.test}
    if set(by_id) != expected_ids:
        raise ValueError("supplied blind examples must cover exact prepared test ids")
    retained: list[BlindExample] = []
    for truth in prepared.test:
        blind = by_id[truth.example_id]
        if (
            blind.target_kind is not truth.target_kind
            or blind.target_family != truth.target_family
        ):
            raise ValueError(
                "supplied blind examples must retain prepared target kind and family"
            )
        retained.append(blind)
    return tuple(retained)


def _blind_topology_cases(
    prepared: PreparedScenario,
    supplied: Sequence[TopologyCheckCase] | None,
) -> tuple[TopologyCheckCase, ...]:
    """Validate projected topology inputs against evaluator-owned pair identities."""

    if supplied is None:
        return prepared.topology_checks
    parsed = tuple(
        TopologyCheckCase.model_validate(
            item.model_dump(mode="python") if isinstance(item, TopologyCheckCase) else item
        )
        for item in supplied
    )
    by_check_id = {item.check_id: item for item in parsed}
    if len(by_check_id) != len(parsed):
        raise ValueError("supplied topology check ids must be unique")
    canonical_by_check_id = {item.check_id: item for item in prepared.topology_checks}
    if set(by_check_id) != set(canonical_by_check_id):
        raise ValueError("supplied topology cases must cover exact prepared check ids")
    test_by_id = {item.example_id: item for item in prepared.test}
    retained: list[TopologyCheckCase] = []
    for canonical in prepared.topology_checks:
        projected = by_check_id[canonical.check_id]
        if (
            projected.base_example_id != canonical.base_example_id
            or projected.perturbed_example.example_id
            != canonical.perturbed_example.example_id
            or projected.expected_relation is not canonical.expected_relation
        ):
            raise ValueError(
                "supplied topology cases must retain prepared pair identity and relation"
            )
        base = test_by_id[canonical.base_example_id]
        if (
            projected.perturbed_example.target_kind is not base.target_kind
            or projected.perturbed_example.target_family != base.target_family
        ):
            raise ValueError(
                "supplied topology cases must retain prepared target kind and family"
            )
        retained.append(projected)
    return tuple(retained)


def run_evaluation(
    suite: EvalSuiteSpec,
    *,
    scenario_id: str,
    adapter: AdapterLaunchSpec | BaselineAdapter,
    prepared: PreparedScenario,
    seed: int,
    producer: EvalProducerBinding | None = None,
    blind_examples: Sequence[BlindExample] | None = None,
    topology_cases: Sequence[TopologyCheckCase] | None = None,
) -> EvalResult:
    scenario = _scenario(suite, scenario_id)
    execution_started = time.monotonic()
    fitted_state = prepared.preparation.preprocessing.get("fitted_state")
    if not isinstance(fitted_state, Mapping):
        fitted_state = None
    launch = _coerce_adapter_launch(
        adapter,
        train_only_fitted_state=fitted_state,
    )
    adapter_spec = launch.declared_spec
    if producer is None:
        if adapter_spec.kind is not AdapterKind.BASELINE:
            raise ValueError("model evaluations must explicitly bind a producer run receipt")
        producer = _unissued_baseline_producer()
    binding = _validate_run_inputs(
        suite=suite,
        scenario=scenario,
        prepared=prepared,
        adapter_spec=adapter_spec,
        producer=producer,
        seed=seed,
    )
    if (blind_examples is not None or topology_cases is not None) and (
        adapter_spec.kind is not AdapterKind.MODEL
    ):
        raise ValueError("explicit projected inputs are reserved for model adapters")
    if prepared.topology_checks and ((blind_examples is None) != (topology_cases is None)):
        raise ValueError(
            "projected graph evaluation requires blind examples and topology cases together"
        )
    examples = _blind_test_examples(prepared, blind_examples)
    resolved_topology_cases = _blind_topology_cases(prepared, topology_cases)
    adapter_envelope = AdapterLaunchSpec.model_dump(launch, mode="json")
    spec_deadline_seconds = suite.budget.max_adapter_seconds - (
        time.monotonic() - execution_started
    )
    if spec_deadline_seconds <= 0.0:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.BUDGET,
            code="adapter_time_budget_exceeded",
            public_detail="adapter launch exceeded the frozen wall-clock budget",
        )
    try:
        actual_spec = _read_adapter_spec_isolated(
            adapter_envelope=adapter_envelope,
            deadline_seconds=spec_deadline_seconds,
        )
    except TimeoutError:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.BUDGET,
            code="adapter_time_budget_exceeded",
            public_detail="adapter initialization exceeded the frozen wall-clock budget",
        )
    except Exception as error:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.MODEL,
            code="adapter_initialization_failed",
            public_detail=(
                f"adapter initialization raised {type(error).__name__}; detail omitted"
            ),
        )
    if actual_spec != adapter_spec:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.ARTIFACT,
            code="adapter_spec_mismatch",
            public_detail="isolated adapter identity differs from its declared launch manifest",
        )
    perturbed_examples = tuple(case.perturbed_example for case in resolved_topology_cases)
    batches = (examples,) + ((perturbed_examples,) if perturbed_examples else ())
    remaining_seconds = suite.budget.max_adapter_seconds - (time.monotonic() - execution_started)
    if remaining_seconds <= 0.0:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.BUDGET,
            code="adapter_time_budget_exceeded",
            public_detail="adapter exceeded the frozen wall-clock budget",
        )
    try:
        isolated_outputs = _execute_adapter_isolated(
            adapter_envelope=adapter_envelope,
            scenario=scenario,
            fit_examples=prepared.train,
            batches=batches,
            seed=seed,
            deadline_seconds=remaining_seconds,
        )
    except TimeoutError:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.BUDGET,
            code="adapter_time_budget_exceeded",
            public_detail="adapter exceeded the frozen wall-clock budget",
        )
    except Exception as error:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.MODEL,
            code="adapter_exception",
            public_detail=f"adapter raised {type(error).__name__}; private exception text omitted",
        )
    predictions, replay = isolated_outputs[0]
    if canonical_hash(predictions) != canonical_hash(replay):
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.MODEL,
            code="nondeterministic_replay",
            public_detail="adapter outputs changed during evaluator-owned deterministic replay",
        )

    topology_results: tuple[TopologyCheckResult, ...] = ()
    if resolved_topology_cases:
        perturbed_predictions, perturbed_replay = isolated_outputs[1]
        if canonical_hash(perturbed_predictions) != canonical_hash(perturbed_replay):
            return _failed_result(
                suite=suite,
                scenario=scenario,
                adapter_spec=adapter_spec,
                producer=producer,
                binding=binding,
                seed=seed,
                category=FailureCategory.MODEL,
                code="nondeterministic_topology_replay",
                public_detail="paired topology outputs changed during deterministic replay",
            )
        original_by_id = {item.example_id: item for item in predictions}
        perturbed_by_id = {item.example_id: item for item in perturbed_predictions}
        try:
            if len(perturbed_by_id) != len(perturbed_predictions):
                raise ValueError("paired topology prediction ids must be unique")
            expected_ids = {
                case.perturbed_example.example_id for case in resolved_topology_cases
            }
            if set(perturbed_by_id) != expected_ids:
                raise ValueError("paired topology predictions must cover exact prepared ids")
            retained: list[TopologyCheckResult] = []
            for case in resolved_topology_cases:
                base_prediction = original_by_id[case.base_example_id]
                perturbed_prediction = perturbed_by_id[case.perturbed_example.example_id]
                equal = _prediction_semantics_hash(base_prediction) == _prediction_semantics_hash(
                    perturbed_prediction
                )
                passed = equal if case.expected_relation is TopologyRelation.EQUAL else not equal
                retained.append(
                    TopologyCheckResult(
                        check_id=case.check_id,
                        base_example_id=case.base_example_id,
                        perturbed_example_sha256=case.perturbed_example.content_hash,
                        expected_relation=case.expected_relation,
                        base_prediction=base_prediction,
                        perturbed_prediction=perturbed_prediction,
                        passed=passed,
                    )
                )
            topology_results = tuple(retained)
        except Exception as error:
            return _failed_result(
                suite=suite,
                scenario=scenario,
                adapter_spec=adapter_spec,
                producer=producer,
                binding=binding,
                seed=seed,
                category=FailureCategory.EVALUATOR,
                code="topology_contract_unverifiable",
                public_detail=(
                    f"evaluator rejected paired topology outputs as {type(error).__name__}; "
                    "detail omitted"
                ),
            )
    if "prepared_payload_hash_or_contract_drift" in _prepared_blockers(scenario, prepared):
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.EVALUATOR,
            code="prepared_snapshot_mutated",
            public_detail="adapter execution changed the hash-bound prepared snapshot",
        )
    try:
        scored = score_predictions(
            scenario=scenario,
            fit_examples=prepared.train,
            truth=prepared.test,
            predictions=predictions,
            train_only_fitted_state=fitted_state,
        )
        metrics = dict(scored.metrics)
        for check_id in scenario.topology_contract_checks:
            outcomes = [item.passed for item in topology_results if item.check_id == check_id]
            if not outcomes:
                raise ValueError(f"topology contract has no evaluator-owned pairs: {check_id}")
            metrics[check_id] = sum(outcomes) / len(outcomes)
    except Exception as error:
        return _failed_result(
            suite=suite,
            scenario=scenario,
            adapter_spec=adapter_spec,
            producer=producer,
            binding=binding,
            seed=seed,
            category=FailureCategory.EVALUATOR,
            code="scoring_contract_violation",
            public_detail=f"scoring rejected outputs as {type(error).__name__}; detail omitted",
        )

    return EvalResult(
        result_id=deterministic_result_id(
            suite=suite,
            scenario=scenario,
            adapter=adapter_spec,
            binding=binding,
            producer=producer,
            seed=seed,
            status=EvaluationStatus.SUCCEEDED,
            raw_predictions=predictions,
            topology_checks=topology_results,
            per_example=scored.per_example,
            metrics=metrics,
            counts=scored.counts,
            failure_counts=scored.failure_counts,
            coverage=scored.coverage,
            failure=None,
            claim_boundary=suite.claim_boundary,
        ),
        status=EvaluationStatus.SUCCEEDED,
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        adapter=adapter_spec,
        producer=producer,
        seed=seed,
        source_sha256=binding.source_sha256,
        split_sha256=binding.split_sha256,
        recipe_sha256=binding.recipe_sha256,
        budget_hash=suite.budget.content_hash,
        truth_sidecar_sha256=binding.truth_sidecar_sha256,
        raw_predictions=predictions,
        topology_checks=topology_results,
        per_example=scored.per_example,
        metrics=metrics,
        counts=scored.counts,
        failure_counts=scored.failure_counts,
        coverage=scored.coverage,
        claim_boundary=suite.claim_boundary,
    )


def verify_result_against_prepared(
    suite: EvalSuiteSpec,
    *,
    result: EvalResult,
    prepared: PreparedScenario,
) -> EvalResult:
    """Authenticate a loaded result by rescoring it against the canonical truth sidecar."""

    result = EvalResult.model_validate(result.model_dump(mode="python"))
    scenario = _scenario(suite, result.scenario_id)
    binding = _validate_run_inputs(
        suite=suite,
        scenario=scenario,
        prepared=prepared,
        adapter_spec=result.adapter,
        producer=result.producer,
        seed=result.seed,
    )
    expected_identity = (
        binding.source_sha256,
        binding.split_sha256,
        binding.recipe_sha256,
        binding.truth_sidecar_sha256,
    )
    actual_identity = (
        result.source_sha256,
        result.split_sha256,
        result.recipe_sha256,
        result.truth_sidecar_sha256,
    )
    if actual_identity != expected_identity:
        raise ValueError("evaluation result does not bind the canonical prepared snapshot")
    if result.status is EvaluationStatus.FAILED:
        return result

    fitted_state = prepared.preparation.preprocessing.get("fitted_state")
    if not isinstance(fitted_state, Mapping):
        fitted_state = None
    rescored = score_predictions(
        scenario=scenario,
        fit_examples=prepared.train,
        truth=prepared.test,
        predictions=result.raw_predictions,
        train_only_fitted_state=fitted_state,
    )
    expected_metrics = dict(rescored.metrics)
    for check_id in scenario.topology_contract_checks:
        outcomes = [item.passed for item in result.topology_checks if item.check_id == check_id]
        if not outcomes:
            raise ValueError("canonical topology result is missing evaluator-owned paired outputs")
        expected_metrics[check_id] = sum(outcomes) / len(outcomes)
    expected_topology = {
        (case.check_id, case.base_example_id, case.perturbed_example.content_hash)
        for case in prepared.topology_checks
    }
    actual_topology = {
        (item.check_id, item.base_example_id, item.perturbed_example_sha256)
        for item in result.topology_checks
    }
    if actual_topology != expected_topology:
        raise ValueError("evaluation topology outputs do not bind canonical prepared pairs")
    if (
        result.per_example != rescored.per_example
        or result.metrics != expected_metrics
        or result.counts != rescored.counts
        or result.failure_counts != rescored.failure_counts
        or not math.isclose(
            result.coverage,
            rescored.coverage,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("evaluation result does not rescore against canonical prepared truth")
    return result


def compare_results(
    suite: EvalSuiteSpec,
    results: Sequence[EvalResult],
    *,
    prepared: Mapping[str, PreparedScenario] | None = None,
) -> ComparisonReport:
    if not results:
        raise ValueError("comparison requires evaluation results")
    if prepared is None:
        raise ValueError("comparison requires canonical prepared scenarios for rescoring")
    results = tuple(
        EvalResult.model_validate(result.model_dump(mode="python")) for result in results
    )
    verified_results: list[EvalResult] = []
    for result in results:
        canonical_prepared = prepared.get(result.scenario_id)
        if canonical_prepared is None:
            raise ValueError(
                f"comparison lacks canonical prepared scenario: {result.scenario_id}"
            )
        verified_results.append(
            verify_result_against_prepared(
                suite,
                result=result,
                prepared=canonical_prepared,
            )
        )
    results = tuple(verified_results)
    aggregates, failure_counts = derive_comparison_summary(suite, results)

    result_hashes = tuple(sorted(result.content_hash for result in results))
    digest = canonical_hash(
        {
            "schema": "tabu.eval-comparison-identity.v2",
            "suite_hash": suite.suite_hash,
            "result_hashes": result_hashes,
        }
    )
    return ComparisonReport(
        comparison_id=f"compare-{digest[:24]}",
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        variable_under_test=suite.variable_under_test,
        result_hashes=result_hashes,
        aggregates=aggregates,
        failure_counts=failure_counts,
        publication_eligible=comparison_publication_eligible(suite, results),
        claim_boundary=suite.claim_boundary,
    )
