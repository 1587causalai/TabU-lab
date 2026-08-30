"""Versioned contracts for reproducible, task-scoped TabU evaluations.

The foundry deliberately does not fetch datasets.  A formal evaluation must be
bound to a prepared snapshot with content, split, and recipe hashes before a
model adapter can run.  Test truth is stored here only so the runner can score
predictions after the adapter has returned.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import canonical_hash, require_sha256
from tabu_lab.evidence import EnvironmentDisclosure, SourceIdentity
from tabu_lab.evidence.public_safety import (
    contains_absolute_local_path,
    contains_private_identity_or_secret,
    is_sensitive_public_key,
)
from tabu_lab.evidence.schemas import EvidenceSchema

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Scalar = str | int | float


class TaskKind(StrEnum):
    SUPERVISED_CLASSIFICATION = "supervised_classification"
    SUPERVISED_REGRESSION = "supervised_regression"
    TABLE_COMPLETION = "table_completion"
    GRAPH_COMPLETION = "graph_completion"
    RECSYS_COMPLETION = "recsys_completion"


class TargetKind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class MetricDirection(StrEnum):
    LOWER = "lower"
    HIGHER = "higher"


class AdapterKind(StrEnum):
    MODEL = "model"
    BASELINE = "baseline"


class ProducerProvenance(StrEnum):
    RECEIPTED_RUN = "receipted_run"
    UNISSUED_BASELINE = "unissued_baseline"


class TopologyRelation(StrEnum):
    EQUAL = "equal"
    DIFFERENT = "different"


class EvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailureCategory(StrEnum):
    MODEL = "model"
    DATA = "data"
    EVALUATOR = "evaluator"
    ARTIFACT = "artifact"
    INFRASTRUCTURE = "infrastructure"
    BUDGET = "budget"
    KILL_CONDITION = "kill-condition"


class DatasetRequirement(EvidenceSchema):
    schema_version: Literal["tabu.eval-dataset-requirement.v1"] = "tabu.eval-dataset-requirement.v1"
    dataset_id: Identifier
    source_uri: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    task_id: str | None = None
    license_id: str = Field(min_length=1)
    redistribution: Literal["allowed", "metadata_only", "forbidden"]
    network_required: bool
    snapshot_hash_policy: Literal["require_sha256_at_run"] = "require_sha256_at_run"
    required_partitions: tuple[Literal["train", "validation", "test"], ...] = (
        "train",
        "validation",
        "test",
    )

    @field_validator("required_partitions")
    @classmethod
    def _partitions_are_complete_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != ("train", "validation", "test"):
            raise ValueError("evaluation datasets must freeze train, validation, test partitions")
        return values


class SelectionSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-selection.v1"] = "tabu.eval-selection.v1"
    method: Literal["sha256_rank", "support_desc_stable_id", "all"]
    fit_partition: Literal["train"] = "train"
    stable_id_tiebreak: Literal[True] = True
    partition_limits: dict[str, int] = Field(default_factory=dict)
    users: int | None = Field(default=None, gt=0)
    items: int | None = Field(default=None, gt=0)

    @field_validator("partition_limits")
    @classmethod
    def _valid_partition_limits(cls, values: dict[str, int]) -> dict[str, int]:
        allowed = {"train", "validation", "test"}
        if set(values) - allowed:
            raise ValueError("partition_limits contains an unknown partition")
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("partition limits must be positive integers")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _recsys_dimensions_are_paired(self) -> SelectionSpec:
        if (self.users is None) != (self.items is None):
            raise ValueError("user and item limits must be declared together")
        if self.method == "support_desc_stable_id" and self.users is None:
            raise ValueError("support-based interaction selection requires users and items")
        if self.method != "support_desc_stable_id" and self.users is not None:
            raise ValueError("user/item limits are only valid for support-based selection")
        return self


class MaskSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-mask.v1"] = "tabu.eval-mask.v1"
    fraction: float = Field(gt=0.0, lt=1.0)
    applied_after_split: Literal[True] = True
    independent_seed_role: Literal["mask_seed"] = "mask_seed"
    statistics_partition: Literal["train"] = "train"
    truth_sidecar_required: Literal[True] = True
    natural_missing_excluded: Literal[True] = True


class MetricSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-metric.v1"] = "tabu.eval-metric.v1"
    metric_id: Identifier
    direction: MetricDirection
    primary: bool = False
    target_kind: TargetKind | None = None
    train_only_normalization: bool = False
    sample_rescorable: Literal[True] = True


class BaselineSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-baseline.v1"] = "tabu.eval-baseline.v1"
    baseline_id: Identifier
    family: Literal[
        "majority",
        "mean",
        "mean_mode",
        "ohe_logistic",
        "standardized_ridge",
        "numeric_knn",
        "global_mode",
        "neighbor_mode",
        "global_mean",
        "user_item_bias",
    ]
    hyperparameters: dict[str, JsonValue] = Field(default_factory=dict)
    fit_partition: Literal["train"] = "train"


class ScenarioSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-scenario.v1"] = "tabu.eval-scenario.v1"
    scenario_id: Identifier
    task: TaskKind
    dataset: DatasetRequirement
    selection: SelectionSpec
    target_kinds: tuple[TargetKind, ...] = Field(min_length=1)
    applicable_contracts: tuple[Identifier, ...] = Field(min_length=1)
    applicable_profiles: tuple[Identifier, ...] = ()
    response_columns: int | None = Field(default=None, ge=1)
    baselines: tuple[BaselineSpec, ...] = Field(min_length=1)
    metrics: tuple[MetricSpec, ...] = Field(min_length=1)
    mask: MaskSpec | None = None
    truth_partition: Literal["test"] = "test"
    checkpoint_selection_partition: Literal["validation"] = "validation"
    preprocessing_fit_partition: Literal["train"] = "train"
    statistics_fit_scope: Literal["selected_train", "full_train_partition"] = (
        "selected_train"
    )
    topology_contract_checks: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("target_kinds", "applicable_contracts")
    @classmethod
    def _unique_sequence(cls, values: tuple[object, ...]) -> tuple[object, ...]:
        if len(values) != len(set(values)):
            raise ValueError("scenario declarations must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _scenario_is_closed(self) -> ScenarioSpec:
        baseline_ids = [item.baseline_id for item in self.baselines]
        metric_ids = [item.metric_id for item in self.metrics]
        if len(baseline_ids) != len(set(baseline_ids)):
            raise ValueError("baseline ids must be unique within a scenario")
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("metric ids must be unique within a scenario")
        if not any(metric.primary for metric in self.metrics):
            raise ValueError("every scenario needs at least one primary metric")
        completion_tasks = {
            TaskKind.TABLE_COMPLETION,
            TaskKind.GRAPH_COMPLETION,
            TaskKind.RECSYS_COMPLETION,
        }
        if self.task is TaskKind.TABLE_COMPLETION and self.mask is None:
            raise ValueError("table completion requires a post-split mask contract")
        if self.task not in completion_tasks and self.mask is not None:
            raise ValueError("mask contracts are only valid for completion scenarios")
        if self.task is TaskKind.GRAPH_COMPLETION and not self.topology_contract_checks:
            raise ValueError("graph completion requires topology contract checks")
        if self.task is not TaskKind.GRAPH_COMPLETION and self.topology_contract_checks:
            raise ValueError("topology checks are only valid for graph completion")
        return self


class EvaluationBudget(EvidenceSchema):
    schema_version: Literal["tabu.eval-budget.v1"] = "tabu.eval-budget.v1"
    model_seeds: tuple[int, ...] = (1729, 2718, 31415)
    max_fit_iterations: int = Field(default=400, gt=0)
    max_adapter_seconds: int = Field(default=600, gt=0)
    device_class: Literal["single_device"] = "single_device"
    deterministic: Literal[True] = True

    @field_validator("model_seeds")
    @classmethod
    def _three_fixed_seeds(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != 3 or len(set(values)) != 3:
            raise ValueError("evaluation v0 requires exactly three unique fixed seeds")
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError("evaluation seeds must be non-negative integers")
        return values


class EvalSuiteSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-suite.v1"] = "tabu.eval-suite.v1"
    suite_id: Identifier
    suite_version: Literal["0.1.0", "0.2.0"] = "0.1.0"
    status: Literal["frozen_v0", "preregistered"] = "frozen_v0"
    description: str = Field(min_length=1)
    variable_under_test: Literal["model_contract_or_artifact"] = "model_contract_or_artifact"
    frozen_variables: tuple[str, ...] = Field(min_length=1)
    scenarios: tuple[ScenarioSpec, ...] = Field(min_length=1)
    budget: EvaluationBudget = Field(default_factory=EvaluationBudget)
    allowed_pilot: Literal["one_train_validation_unissued"] = "one_train_validation_unissued"
    test_sweeps_allowed: Literal[False] = False
    retain_raw_predictions: Literal[True] = True
    retain_per_example_scores: Literal[True] = True
    report_per_seed: Literal[True] = True
    report_mean_std: Literal[True] = True
    composite_score: Literal[False] = False
    claim_boundary: str = Field(min_length=1)

    @field_validator("frozen_variables")
    @classmethod
    def _unique_frozen_variables(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(not value.strip() for value in values):
            raise ValueError("frozen_variables must be unique non-empty names")
        return values

    @model_validator(mode="after")
    def _suite_is_closed(self) -> EvalSuiteSpec:
        scenario_ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario ids must be unique within a suite")
        if "data_snapshot" not in self.frozen_variables:
            raise ValueError("data_snapshot must be frozen for controlled comparison")
        if "evaluation_budget" not in self.frozen_variables:
            raise ValueError("evaluation_budget must be frozen for controlled comparison")
        return self

    @property
    def suite_hash(self) -> str:
        return self.content_hash


class DatasetSnapshotBinding(EvidenceSchema):
    schema_version: Literal["tabu.eval-dataset-binding.v2"] = "tabu.eval-dataset-binding.v2"
    dataset_id: Identifier
    source_sha256: Sha256
    split_sha256: Sha256
    recipe_sha256: Sha256
    truth_sidecar_sha256: Sha256 | None = None
    partition_counts: dict[str, int]
    preprocessing_fit_partition: Literal["train"] = "train"
    normalizer_fit_partition: Literal["train"] = "train"
    codebook_fit_partition: Literal["train"] = "train"
    selection_fit_partition: Literal["train"] = "train"
    mask_applied_after_split: Literal[True] = True
    test_truth_isolated: Literal[True] = True
    adapter_receives_test_truth: Literal[False] = False

    @field_validator("source_sha256", "split_sha256", "recipe_sha256")
    @classmethod
    def _hashes_are_valid(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("truth_sidecar_sha256")
    @classmethod
    def _truth_hash_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="truth_sidecar_sha256")

    @field_validator("partition_counts")
    @classmethod
    def _partition_counts_are_complete(cls, values: dict[str, int]) -> dict[str, int]:
        if set(values) != {"train", "validation", "test"}:
            raise ValueError("snapshot binding needs exact train/validation/test counts")
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("snapshot partition counts must be positive integers")
        return {key: values[key] for key in ("train", "validation", "test")}


class SourceMaterial(EvidenceSchema):
    """Retained source bytes whose raw SHA-256 anchors the prepared snapshot."""

    schema_version: Literal["tabu.eval-source-material.v1"] = "tabu.eval-source-material.v1"
    dataset_id: Identifier
    media_type: str = Field(min_length=1)
    content_base64: str = Field(min_length=1)

    @field_validator("content_base64")
    @classmethod
    def _content_is_canonical_base64(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError("source material must be canonical base64") from error
        if not decoded:
            raise ValueError("source material bytes cannot be empty")
        canonical = base64.b64encode(decoded).decode("ascii")
        if canonical != value:
            raise ValueError("source material must use canonical base64 encoding")
        return value

    @classmethod
    def from_bytes(
        cls,
        *,
        dataset_id: str,
        content: bytes,
        media_type: str = "application/octet-stream",
    ) -> SourceMaterial:
        return cls(
            dataset_id=dataset_id,
            media_type=media_type,
            content_base64=base64.b64encode(content).decode("ascii"),
        )

    @property
    def content_bytes(self) -> bytes:
        return base64.b64decode(self.content_base64, validate=True)

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.content_bytes).hexdigest()


def _finite_scalar(value: Scalar) -> Scalar:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("evaluation scalar values must be finite")
    return value


def _truth_key_path(value: JsonValue, *, path: str) -> str | None:
    """Return the first adapter-visible key path that can carry hidden truth."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            tokens = [token.lower() for token in re.split(r"[^A-Za-z0-9]+", snake) if token]
            normalized_key = "_".join(tokens)
            leaks_truth = any(token.rstrip("s") in {"target", "truth"} for token in tokens) or (
                any(token.rstrip("s") == "label" for token in tokens)
                and normalized_key != "neighbor_labels"
            )
            if leaks_truth:
                return f"{path}.{key}"
            nested = _truth_key_path(child, path=f"{path}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for index, child in enumerate(value):
            nested = _truth_key_path(child, path=f"{path}[{index}]")
            if nested is not None:
                return nested
    return None


def _truth_free_mapping(values: dict[str, JsonValue], *, field_name: str) -> dict[str, JsonValue]:
    leaked = _truth_key_path(values, path=field_name)
    if leaked is not None:
        raise ValueError(
            f"adapter-visible payload contains a target/truth key (including label): {leaked}"
        )
    return dict(sorted(values.items()))


class PreparedExample(EvidenceSchema):
    schema_version: Literal["tabu.eval-prepared-example.v2"] = "tabu.eval-prepared-example.v2"
    example_id: Identifier
    target_kind: TargetKind
    target_family: Identifier
    features: dict[str, JsonValue] = Field(default_factory=dict)
    target: Scalar
    context: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def _target_is_finite(cls, value: Scalar) -> Scalar:
        return _finite_scalar(value)

    @field_validator("features", "context")
    @classmethod
    def _adapter_payload_is_truth_free(
        cls, values: dict[str, JsonValue], info: object
    ) -> dict[str, JsonValue]:
        return _truth_free_mapping(
            values,
            field_name=getattr(info, "field_name", "adapter_payload"),
        )


class BlindExample(EvidenceSchema):
    """Truth-free example passed to model and baseline adapters."""

    schema_version: Literal["tabu.eval-blind-example.v2"] = "tabu.eval-blind-example.v2"
    example_id: Identifier
    target_kind: TargetKind
    target_family: Identifier
    features: dict[str, JsonValue] = Field(default_factory=dict)
    context: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("features", "context")
    @classmethod
    def _adapter_payload_is_truth_free(
        cls, values: dict[str, JsonValue], info: object
    ) -> dict[str, JsonValue]:
        return _truth_free_mapping(
            values,
            field_name=getattr(info, "field_name", "adapter_payload"),
        )


class PreparationContract(EvidenceSchema):
    """Hashable declarations for every train-only preparation boundary."""

    schema_version: Literal["tabu.eval-preparation-contract.v1"] = (
        "tabu.eval-preparation-contract.v1"
    )
    preprocessing: dict[str, JsonValue] = Field(min_length=1)
    selection: dict[str, JsonValue] = Field(min_length=1)
    mask: dict[str, JsonValue] = Field(min_length=1)

    @field_validator("preprocessing", "selection", "mask")
    @classmethod
    def _contract_maps_are_canonical(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _preprocessing_has_train_only_fitted_identity(self) -> PreparationContract:
        required = {"fit_partition", "implementation_sha256", "fitted_state_sha256"}
        if not required.issubset(self.preprocessing):
            raise ValueError(
                "preprocessing contract needs train fit partition, implementation hash, "
                "and fitted-state hash"
            )
        if self.preprocessing["fit_partition"] != "train":
            raise ValueError("preprocessing contract must be fitted on train")
        for field_name in ("implementation_sha256", "fitted_state_sha256"):
            value = self.preprocessing[field_name]
            if not isinstance(value, str):
                raise ValueError(f"preprocessing {field_name} must be a SHA-256 string")
            require_sha256(value, field_name=field_name)
        return self


class TopologyCheckCase(EvidenceSchema):
    """Evaluator-owned paired input used to verify a topology relation."""

    schema_version: Literal["tabu.eval-topology-check-case.v1"] = "tabu.eval-topology-check-case.v1"
    check_id: Identifier
    base_example_id: Identifier
    perturbed_example: BlindExample
    expected_relation: TopologyRelation


class PreparedScenario(EvidenceSchema):
    schema_version: Literal["tabu.eval-prepared-scenario.v2"] = "tabu.eval-prepared-scenario.v2"
    scenario_id: Identifier
    binding: DatasetSnapshotBinding
    source_material: SourceMaterial
    preparation: PreparationContract
    train: tuple[PreparedExample, ...] = Field(min_length=1)
    validation: tuple[PreparedExample, ...] = Field(min_length=1)
    test: tuple[PreparedExample, ...] = Field(min_length=1)
    topology_checks: tuple[TopologyCheckCase, ...] = ()

    @staticmethod
    def split_sha256_for(
        *,
        train: Sequence[PreparedExample],
        validation: Sequence[PreparedExample],
        test: Sequence[PreparedExample],
    ) -> str:
        return canonical_hash(
            {
                "schema": "tabu.eval-prepared-split.v2",
                "partitions": {
                    "train": tuple(train),
                    "validation": tuple(validation),
                    "test": tuple(test),
                },
            }
        )

    @staticmethod
    def truth_sidecar_sha256_for(*, test: Sequence[PreparedExample]) -> str:
        return canonical_hash(
            {
                "schema": "tabu.eval-truth-sidecar.v1",
                "targets": tuple(
                    {
                        "example_id": item.example_id,
                        "target_kind": item.target_kind,
                        "target_family": item.target_family,
                        "truth": (
                            float(item.target)
                            if item.target_kind is TargetKind.NUMERIC
                            else str(item.target)
                        ),
                    }
                    for item in test
                ),
            }
        )

    @staticmethod
    def recipe_sha256_for(
        *,
        preparation: PreparationContract,
        topology_checks: Sequence[TopologyCheckCase] = (),
    ) -> str:
        return canonical_hash(
            {
                "schema": "tabu.eval-preparation-recipe.v2",
                "preparation": preparation,
                "topology_checks": tuple(topology_checks),
            }
        )

    @model_validator(mode="after")
    def _partitions_match_binding(self) -> PreparedScenario:
        if self.source_material.dataset_id != self.binding.dataset_id:
            raise ValueError("source material dataset id does not match snapshot binding")
        if self.source_material.raw_sha256 != self.binding.source_sha256:
            raise ValueError("source_sha256 does not bind the retained source bytes")
        expected_truth_sidecar = self.truth_sidecar_sha256_for(test=self.test)
        if self.binding.truth_sidecar_sha256 != expected_truth_sidecar:
            raise ValueError("truth_sidecar_sha256 does not bind the held-out test targets")
        partitions = {"train": self.train, "validation": self.validation, "test": self.test}
        all_ids: list[str] = []
        for name, examples in partitions.items():
            expected = self.binding.partition_counts[name]
            if len(examples) != expected:
                raise ValueError(f"{name} examples do not match snapshot partition count")
            all_ids.extend(example.example_id for example in examples)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("prepared example ids must be unique across all partitions")
        expected_split = self.split_sha256_for(
            train=self.train,
            validation=self.validation,
            test=self.test,
        )
        if self.binding.split_sha256 != expected_split:
            raise ValueError("split_sha256 does not bind the actual prepared partitions")
        expected_recipe = self.recipe_sha256_for(
            preparation=self.preparation,
            topology_checks=self.topology_checks,
        )
        if self.binding.recipe_sha256 != expected_recipe:
            raise ValueError(
                "recipe_sha256 does not bind preprocessing, selection, mask, and topology cases"
            )
        test_by_id = {example.example_id: example for example in self.test}
        topology_keys: list[tuple[str, str, str]] = []
        perturbed_ids: list[str] = []
        for case in self.topology_checks:
            base = test_by_id.get(case.base_example_id)
            if base is None:
                raise ValueError("topology check base_example_id must name a test example")
            if (
                case.perturbed_example.target_kind is not base.target_kind
                or case.perturbed_example.target_family != base.target_family
            ):
                raise ValueError("topology paired examples must retain target kind and family")
            topology_keys.append(
                (case.check_id, case.base_example_id, case.perturbed_example.example_id)
            )
            perturbed_ids.append(case.perturbed_example.example_id)
        if len(topology_keys) != len(set(topology_keys)):
            raise ValueError("topology check cases must be unique")
        if len(perturbed_ids) != len(set(perturbed_ids)) or set(perturbed_ids) & set(all_ids):
            raise ValueError("topology perturbed example ids must be unique and partition-external")
        return self


class AdapterSpec(EvidenceSchema):
    schema_version: Literal["tabu.eval-adapter.v2"] = "tabu.eval-adapter.v2"
    adapter_id: Identifier
    adapter_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
    kind: AdapterKind
    fit_iterations: int = Field(ge=0)
    device_class: Literal["single_device"]
    deterministic: bool
    contract_id: Identifier | None = None
    artifact_id: Identifier | None = None
    baseline_family: str | None = None
    profile_id: str | None = None

    @model_validator(mode="after")
    def _adapter_identity_is_consistent(self) -> AdapterSpec:
        if self.kind is AdapterKind.MODEL:
            if self.contract_id is None:
                raise ValueError("model adapters must declare contract_id")
            if self.baseline_family is not None:
                raise ValueError("model adapters cannot declare baseline_family")
        else:
            if self.baseline_family is None:
                raise ValueError("baseline adapters must declare baseline_family")
            if self.contract_id is not None or self.artifact_id is not None:
                raise ValueError("baseline adapters cannot claim a model contract or artifact")
        return self


class AdapterLaunchSpec(EvidenceSchema):
    """Plain-data factory manifest evaluated only inside the isolated process."""

    schema_version: Literal["tabu.eval-adapter-launch.v1"] = "tabu.eval-adapter-launch.v1"
    module: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    qualname: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    declared_spec: AdapterSpec

    @field_validator("qualname")
    @classmethod
    def _qualname_is_importable(cls, value: str) -> str:
        if "<locals>" in value.split("."):
            raise ValueError("adapter launch class must have a module-level identity")
        return value

    @field_validator("kwargs")
    @classmethod
    def _kwargs_are_deterministic(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return dict(sorted(values.items()))


class RawPrediction(EvidenceSchema):
    schema_version: Literal["tabu.eval-raw-prediction.v1"] = "tabu.eval-raw-prediction.v1"
    example_id: Identifier
    value: Scalar | None = None
    probabilities: dict[str, float] | None = None
    abstained: bool = False
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    failure_category: FailureCategory | None = None
    failure_code: str | None = None

    @field_validator("value")
    @classmethod
    def _prediction_is_finite(cls, value: Scalar | None) -> Scalar | None:
        return None if value is None else _finite_scalar(value)

    @field_validator("probabilities")
    @classmethod
    def _probabilities_form_distribution(
        cls, values: dict[str, float] | None
    ) -> dict[str, float] | None:
        if values is None:
            return None
        if not values or any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("probabilities must be a non-empty finite non-negative map")
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
            raise ValueError("probabilities must sum to one")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _abstention_is_explicit(self) -> RawPrediction:
        if self.abstained:
            if self.value is not None or self.probabilities is not None:
                raise ValueError("abstentions cannot contain a value or probability distribution")
            if self.failure_category is None or not self.failure_code:
                raise ValueError("abstentions need a failure category and stable failure code")
        elif self.value is None and self.probabilities is None:
            raise ValueError("non-abstaining predictions need a value or probabilities")
        return self


class PerExampleScore(EvidenceSchema):
    schema_version: Literal["tabu.eval-example-score.v2"] = "tabu.eval-example-score.v2"
    example_id: Identifier
    prediction_sha256: Sha256
    target_kind: TargetKind
    target_family: Identifier
    truth: Scalar
    normalization_scale: float | None = None
    categorical_support: tuple[str, ...] = ()
    scored: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    failure_category: FailureCategory | None = None

    @field_validator("prediction_sha256")
    @classmethod
    def _prediction_hash_is_valid(cls, value: str) -> str:
        return require_sha256(value, field_name="prediction_sha256")

    @field_validator("truth")
    @classmethod
    def _truth_is_finite(cls, value: Scalar) -> Scalar:
        return _finite_scalar(value)

    @field_validator("normalization_scale")
    @classmethod
    def _scale_is_positive(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("numeric normalization scale must be positive and finite")
        return value

    @field_validator("categorical_support")
    @classmethod
    def _support_is_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if values != tuple(sorted(set(values))) or any(not value for value in values):
            raise ValueError("categorical support must be sorted, unique, and non-empty when used")
        return values

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("per-example metrics must be finite")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _score_payload_is_consistent(self) -> PerExampleScore:
        if self.target_kind is TargetKind.NUMERIC:
            if isinstance(self.truth, str) or self.normalization_scale is None:
                raise ValueError("numeric scores need numeric truth and a train-only scale")
            if self.categorical_support:
                raise ValueError("numeric scores cannot carry categorical label support")
        else:
            if self.normalization_scale is not None or not self.categorical_support:
                raise ValueError("categorical scores need train-only support and no numeric scale")
            if str(self.truth) not in self.categorical_support:
                raise ValueError("categorical truth must lie in train-only label support")
        if self.scored:
            if not self.metrics or self.failure_category is not None:
                raise ValueError("scored examples need metrics and no failure category")
        elif self.metrics or self.failure_category is None:
            raise ValueError("unscored examples need an empty metric map and failure category")
        nonnegative = {
            "absolute_error",
            "squared_error",
            "normalized_squared_error",
            "negative_log_likelihood",
            "truth_centered_squared",
        }
        if any(self.metrics.get(name, 0.0) < 0.0 for name in nonnegative):
            raise ValueError("loss and squared metric contributions must be non-negative")
        if "auc_label" in self.metrics and self.metrics["auc_label"] not in {0.0, 1.0}:
            raise ValueError("auc_label contribution must be binary")
        if "auc_score" in self.metrics and not 0.0 <= self.metrics["auc_score"] <= 1.0:
            raise ValueError("auc_score contribution must be a probability")
        if "correct" in self.metrics and self.metrics["correct"] not in {0.0, 1.0}:
            raise ValueError("categorical correctness contribution must be binary")
        return self


class EvaluationFailure(EvidenceSchema):
    schema_version: Literal["tabu.eval-failure.v1"] = "tabu.eval-failure.v1"
    category: FailureCategory
    code: Identifier
    public_detail: str = Field(min_length=1, max_length=240)


class EvalProducerBinding(EvidenceSchema):
    """Lineage of the subject being evaluated, never the evaluation execution.

    A model points back to the immutable training receipt that produced its
    checkpoint.  That producer receipt may be ``local_unissued``: it is enough
    to make an exploratory result reproducible, but it does not make the result
    public evidence.  An evaluator-owned baseline has no training run.  Both
    still require a separate :class:`EvalExecutionReceipt` for every execution.
    """

    schema_version: Literal["tabu.eval-producer-binding.v1"] = "tabu.eval-producer-binding.v1"
    provenance: ProducerProvenance
    run_id: Identifier | None = None
    receipt_sha256: Sha256 | None = None
    receipt_pointer: str | None = Field(default=None, min_length=1)
    publication_eligible: bool

    @field_validator("receipt_sha256")
    @classmethod
    def _receipt_hash_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="receipt_sha256")

    @field_validator("receipt_pointer")
    @classmethod
    def _receipt_pointer_is_public_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace("\\", "/")
        if (
            normalized.startswith(("/", "file:"))
            or "/Users/" in normalized
            or ".." in normalized.split("/")
        ):
            raise ValueError("receipt pointer must not disclose a private absolute path")
        return value

    @model_validator(mode="after")
    def _provenance_is_complete(self) -> EvalProducerBinding:
        fields = (self.run_id, self.receipt_sha256, self.receipt_pointer)
        if self.provenance is ProducerProvenance.RECEIPTED_RUN:
            if any(value is None for value in fields):
                raise ValueError("receipted evaluation producers need run id, hash, and pointer")
        else:
            if any(value is not None for value in fields):
                raise ValueError("unissued baselines cannot carry partial run receipt identity")
            if self.publication_eligible:
                raise ValueError("unissued baseline results are explicitly non-public")
        return self


class EvaluatorSourceAuthorization(EvidenceSchema):
    """Path-free replay handle for the evaluator implementation source.

    This object deliberately does *not* authorize an evaluation protocol or a
    dataset.  Those authorities remain the exact ``EvalSuiteSpec`` hash and a
    separately reviewed ``DatasetSnapshotSpec``.  The fields below mirror the
    verified Git authorization summary without importing the catalog-backed
    verifier into this low-level schema module (which would create an import
    cycle).
    """

    schema_version: Literal["tabu.eval-source-authorization.v1"] = (
        "tabu.eval-source-authorization.v1"
    )
    purpose: Literal["evaluator_source"] = "evaluator_source"
    authorization_schema_version: Literal["tabu.formal-run-authorization.v3"] = (
        "tabu.formal-run-authorization.v3"
    )
    canonical_commit: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    catalog_hash: Sha256
    catalog_source_tree_hash: Sha256
    experiment_id: Identifier
    experiment_status: str = Field(min_length=1)
    preregistration_sha256: Sha256
    source_identity_sha256: Sha256
    review_ids: tuple[Identifier, ...] = Field(min_length=1)
    review_report_sha256s: tuple[Sha256, ...] = Field(min_length=1)
    gong_approval_sha256s: tuple[Sha256, ...] = Field(min_length=1)

    @field_validator(
        "catalog_hash",
        "catalog_source_tree_hash",
        "preregistration_sha256",
        "source_identity_sha256",
    )
    @classmethod
    def _authorization_hashes_are_valid(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("review_ids", "review_report_sha256s", "gong_approval_sha256s")
    @classmethod
    def _authorization_evidence_is_unique(
        cls, values: tuple[str, ...], info: object
    ) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError(
                f"{getattr(info, 'field_name', 'authorization evidence')} must be unique"
            )
        return values

    @property
    def formal_authorization_payload(self) -> dict[str, object]:
        """Return the exact payload accepted by ``FormalAuthorizationSummary``."""

        payload = self.model_dump(
            mode="python",
            exclude={"schema_version", "purpose", "authorization_schema_version"},
        )
        return {
            "schema_version": self.authorization_schema_version,
            **payload,
        }


def _assert_public_eval_receipt_payload(value: object) -> None:
    """Reject host-private paths and secret-shaped fields in public receipts."""

    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, child in item.items():
                if is_sensitive_public_key(key):
                    raise ValueError(
                        f"evaluation receipt cannot expose sensitive field {key!r}"
                    )
                stack.append(child)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str):
            if contains_absolute_local_path(item):
                raise ValueError("evaluation receipt cannot expose an absolute local path")
            if contains_private_identity_or_secret(item):
                raise ValueError(
                    "evaluation receipt cannot expose a private identity or secret"
                )


def _eval_run_id_preimage(
    *,
    suite_id: str,
    suite_hash: str,
    scenario_id: str,
    adapter_sha256: str,
    producer_sha256: str,
    seed: int,
    source_sha256: str,
    split_sha256: str,
    recipe_sha256: str,
    budget_hash: str,
    truth_sidecar_sha256: str | None,
    baseline_spec_sha256: str | None,
    dataset_snapshot_id: str | None,
    dataset_snapshot_sha256: str | None,
    dataset_request_sha256: str | None,
    dataset_authority_sha256: str | None,
    environment: EnvironmentDisclosure,
    source_identity: SourceIdentity,
    started_at: datetime,
) -> dict[str, object]:
    return {
        "schema": "tabu.eval-run-identity.v1",
        "suite_id": suite_id,
        "suite_hash": suite_hash,
        "scenario_id": scenario_id,
        "adapter_sha256": adapter_sha256,
        "producer_sha256": producer_sha256,
        "seed": seed,
        "source_sha256": source_sha256,
        "split_sha256": split_sha256,
        "recipe_sha256": recipe_sha256,
        "budget_hash": budget_hash,
        "truth_sidecar_sha256": truth_sidecar_sha256,
        "baseline_spec_sha256": baseline_spec_sha256,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_snapshot_sha256": dataset_snapshot_sha256,
        "dataset_request_sha256": dataset_request_sha256,
        "dataset_authority_sha256": dataset_authority_sha256,
        "environment": environment,
        "source_identity": source_identity,
        "started_at": started_at,
    }


def derive_eval_run_id(**values: object) -> str:
    """Derive one evaluation-attempt identity from its frozen execution inputs."""

    return f"evalrun-{canonical_hash(_eval_run_id_preimage(**values))}"


class EvalExecutionReceipt(EvidenceSchema):
    """Independent immutable receipt for one evaluation execution.

    ``result_content_sha256`` binds the complete canonical ``EvalResult``
    payload *before* this receipt is attached.  Excluding the receipt from that
    one hash is intentional and avoids a circular self-hash; every substantive
    result field remains covered.
    """

    schema_version: Literal["tabu.eval-execution-receipt.v2"] = (
        "tabu.eval-execution-receipt.v2"
    )
    receipt_id: Identifier
    eval_run_id: str = Field(pattern=r"^evalrun-[0-9a-f]{64}$")
    issuance_status: Literal["formal", "local_unissued"]
    status: EvaluationStatus
    suite_id: Identifier
    suite_version: str
    suite_hash: Sha256
    scenario_id: Identifier
    adapter_sha256: Sha256
    producer_sha256: Sha256
    seed: int = Field(ge=0)
    source_sha256: Sha256
    split_sha256: Sha256
    recipe_sha256: Sha256
    budget_hash: Sha256
    truth_sidecar_sha256: Sha256 | None = None
    baseline_spec_sha256: Sha256 | None = None
    dataset_snapshot_id: Identifier | None = None
    dataset_snapshot_sha256: Sha256 | None = None
    dataset_request_sha256: Sha256 | None = None
    dataset_authority_sha256: Sha256 | None = None
    dataset_authority_status: Literal["reviewed"] | None = None
    result_id: Identifier
    result_content_sha256: Sha256
    environment: EnvironmentDisclosure
    source_identity: SourceIdentity
    source_authorization: EvaluatorSourceAuthorization | None = None
    started_at: datetime
    completed_at: datetime
    failure_category: FailureCategory | None = None

    @field_validator(
        "suite_hash",
        "adapter_sha256",
        "producer_sha256",
        "source_sha256",
        "split_sha256",
        "recipe_sha256",
        "budget_hash",
        "result_content_sha256",
    )
    @classmethod
    def _receipt_hashes_are_valid(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("truth_sidecar_sha256")
    @classmethod
    def _receipt_truth_hash_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="truth_sidecar_sha256")

    @field_validator(
        "baseline_spec_sha256",
        "dataset_snapshot_sha256",
        "dataset_request_sha256",
        "dataset_authority_sha256",
    )
    @classmethod
    def _optional_receipt_hashes_are_valid(
        cls, value: str | None, info: object
    ) -> str | None:
        return (
            None
            if value is None
            else require_sha256(value, field_name=getattr(info, "field_name", "sha256"))
        )

    @model_validator(mode="after")
    def _receipt_is_closed_and_public_safe(self) -> EvalExecutionReceipt:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("evaluation receipt timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("evaluation receipt completion cannot precede its start")
        if self.issuance_status != self.source_identity.issuance_status:
            raise ValueError("evaluation receipt issuance differs from its source identity")
        if self.issuance_status == "formal":
            if self.source_authorization is None:
                raise ValueError(
                    "formal evaluation receipt requires replayable evaluator-source "
                    "authorization"
                )
            if canonical_hash(self.source_identity) != (
                self.source_authorization.source_identity_sha256
            ):
                raise ValueError(
                    "formal evaluator source differs from its authorization summary"
                )
            dataset_authority = (
                self.dataset_snapshot_id,
                self.dataset_snapshot_sha256,
                self.dataset_request_sha256,
                self.dataset_authority_sha256,
                self.dataset_authority_status,
            )
            if any(value is None for value in dataset_authority):
                raise ValueError(
                    "formal evaluation receipts require one exact reviewed dataset snapshot"
                )
        elif self.source_authorization is not None:
            raise ValueError(
                "local_unissued evaluation receipts cannot carry source authorization"
            )
        if self.status is EvaluationStatus.FAILED:
            if self.failure_category is None:
                raise ValueError("failed evaluation receipts need a failure category")
        elif self.failure_category is not None:
            raise ValueError("successful evaluation receipts cannot claim a failure category")
        expected_run_id = derive_eval_run_id(
            suite_id=self.suite_id,
            suite_hash=self.suite_hash,
            scenario_id=self.scenario_id,
            adapter_sha256=self.adapter_sha256,
            producer_sha256=self.producer_sha256,
            seed=self.seed,
            source_sha256=self.source_sha256,
            split_sha256=self.split_sha256,
            recipe_sha256=self.recipe_sha256,
            budget_hash=self.budget_hash,
            truth_sidecar_sha256=self.truth_sidecar_sha256,
            baseline_spec_sha256=self.baseline_spec_sha256,
            dataset_snapshot_id=self.dataset_snapshot_id,
            dataset_snapshot_sha256=self.dataset_snapshot_sha256,
            dataset_request_sha256=self.dataset_request_sha256,
            dataset_authority_sha256=self.dataset_authority_sha256,
            environment=self.environment,
            source_identity=self.source_identity,
            started_at=self.started_at,
        )
        if self.eval_run_id != expected_run_id:
            raise ValueError("eval_run_id does not bind the frozen evaluation execution")
        expected_receipt_id = (
            "evalreceipt-"
            + canonical_hash(
                self.model_dump(mode="python", exclude={"receipt_id"})
            )[:24]
        )
        if self.receipt_id != expected_receipt_id:
            raise ValueError("receipt_id does not bind the immutable evaluation receipt")
        _assert_public_eval_receipt_payload(self.model_dump(mode="python"))
        return self

    @property
    def publication_eligible(self) -> bool:
        return self.issuance_status == "formal"

    @property
    def receipt_hash(self) -> str:
        return self.content_hash


class TopologyCheckResult(EvidenceSchema):
    """Retained paired outputs scored by the evaluator, never model diagnostics."""

    schema_version: Literal["tabu.eval-topology-check-result.v1"] = (
        "tabu.eval-topology-check-result.v1"
    )
    check_id: Identifier
    base_example_id: Identifier
    perturbed_example_sha256: Sha256
    expected_relation: TopologyRelation
    base_prediction: RawPrediction
    perturbed_prediction: RawPrediction
    passed: bool

    @field_validator("perturbed_example_sha256")
    @classmethod
    def _perturbed_hash_is_valid(cls, value: str) -> str:
        return require_sha256(value, field_name="perturbed_example_sha256")


def _binary_auc_from_contributions(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=True) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _derived_metrics_from_scores(
    scores: Sequence[PerExampleScore],
    topology_checks: Sequence[TopologyCheckResult],
) -> dict[str, float]:
    targets = len(scores)
    scored = [item for item in scores if item.scored]
    derived: dict[str, float] = {
        "coverage": len(scored) / targets,
        "abstention": 1.0 - len(scored) / targets,
    }
    numeric = [item for item in scored if item.target_kind is TargetKind.NUMERIC]
    categorical = [item for item in scored if item.target_kind is TargetKind.CATEGORICAL]
    if numeric:
        absolute = [item.metrics["absolute_error"] for item in numeric]
        squared = [item.metrics["squared_error"] for item in numeric]
        normalized = [item.metrics["normalized_squared_error"] for item in numeric]
        derived["mae"] = sum(absolute) / len(absolute)
        derived["numeric_mae"] = derived["mae"]
        derived["rmse"] = (sum(squared) / len(squared)) ** 0.5
        derived["regression_nrmse"] = (sum(normalized) / len(normalized)) ** 0.5
        derived["numeric_nrmse"] = derived["regression_nrmse"]
        centered = [item.metrics.get("truth_centered_squared") for item in numeric]
        if all(value is not None for value in centered):
            denominator = sum(float(value) for value in centered if value is not None)
            if denominator:
                derived["r2"] = 1.0 - sum(squared) / denominator
    if categorical:
        correct = [item.metrics["correct"] for item in categorical]
        derived["accuracy"] = sum(correct) / len(correct)
        derived["categorical_accuracy"] = derived["accuracy"]
        nll = [item.metrics.get("negative_log_likelihood") for item in categorical]
        if all(value is not None for value in nll):
            mean_nll = sum(float(value) for value in nll if value is not None) / len(nll)
            derived["classification_nll"] = mean_nll
            derived["categorical_nll"] = mean_nll
        auc_labels = [item.metrics.get("auc_label") for item in categorical]
        auc_scores = [item.metrics.get("auc_score") for item in categorical]
        if all(value is not None for value in (*auc_labels, *auc_scores)):
            auc = _binary_auc_from_contributions(
                [int(value) for value in auc_labels if value is not None],
                [float(value) for value in auc_scores if value is not None],
            )
            if auc is not None:
                derived["auroc"] = auc
    by_check: dict[str, list[bool]] = {}
    for check in topology_checks:
        by_check.setdefault(check.check_id, []).append(check.passed)
    for check_id, outcomes in by_check.items():
        derived[check_id] = sum(outcomes) / len(outcomes)
    return derived


def _truth_sidecar_sha256_from_scores(scores: Sequence[PerExampleScore]) -> str:
    return canonical_hash(
        {
            "schema": "tabu.eval-truth-sidecar.v1",
            "targets": tuple(
                {
                    "example_id": item.example_id,
                    "target_kind": item.target_kind,
                    "target_family": item.target_family,
                    "truth": item.truth,
                }
                for item in scores
            ),
        }
    )


def _validate_scores_against_raw(
    scores: Sequence[PerExampleScore],
    predictions: Mapping[str, RawPrediction],
) -> None:
    numeric_truth = [
        float(item.truth)
        for item in scores
        if item.scored and item.target_kind is TargetKind.NUMERIC
    ]
    numeric_mean = sum(numeric_truth) / len(numeric_truth) if numeric_truth else 0.0
    for score in scores:
        prediction = predictions[score.example_id]
        if not score.scored:
            continue
        expected: dict[str, float]
        if score.target_kind is TargetKind.NUMERIC:
            if (
                prediction.value is None
                or isinstance(prediction.value, str)
                or prediction.probabilities is not None
                or score.normalization_scale is None
            ):
                raise ValueError("numeric retained prediction cannot be rescored")
            actual = float(score.truth)
            error = float(prediction.value) - actual
            expected = {
                "absolute_error": abs(error),
                "normalized_squared_error": (error / score.normalization_scale) ** 2,
                "squared_error": error**2,
                "truth_centered_squared": (actual - numeric_mean) ** 2,
            }
        else:
            support = score.categorical_support
            probabilities = prediction.probabilities
            if probabilities is not None and set(probabilities) != set(support):
                raise ValueError("categorical retained probability support changed")
            if prediction.value is None:
                if probabilities is None:
                    raise ValueError("categorical retained prediction cannot be rescored")
                predicted_label = min(
                    probabilities,
                    key=lambda label: (-probabilities[label], label),
                )
            else:
                predicted_label = str(prediction.value)
            if predicted_label not in support:
                raise ValueError("categorical retained prediction lies outside train support")
            actual_label = str(score.truth)
            expected = {"correct": float(predicted_label == actual_label)}
            if probabilities is not None:
                probability = probabilities[actual_label]
                if probability <= 0.0:
                    raise ValueError("categorical retained truth probability must be positive")
                distribution_label = min(
                    probabilities,
                    key=lambda label: (-probabilities[label], label),
                )
                if prediction.value is not None and predicted_label != distribution_label:
                    raise ValueError(
                        "categorical value disagrees with retained probability distribution"
                    )
                expected["negative_log_likelihood"] = -math.log(probability)
                if len(support) == 2:
                    positive = support[-1]
                    expected["auc_label"] = float(actual_label == positive)
                    expected["auc_score"] = probabilities[positive]
        if set(score.metrics) != set(expected) or any(
            not math.isclose(
                score.metrics[name],
                value,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            for name, value in expected.items()
        ):
            raise ValueError(
                "per-example score does not rescore from retained truth and prediction"
            )


def _result_id_from_components(
    *,
    suite_id: str,
    suite_version: str,
    suite_hash: str,
    scenario_id: str,
    task: TaskKind,
    adapter: AdapterSpec,
    producer: EvalProducerBinding,
    seed: int,
    source_sha256: str,
    split_sha256: str,
    recipe_sha256: str,
    budget_hash: str,
    truth_sidecar_sha256: str | None = None,
    status: EvaluationStatus,
    raw_predictions: Sequence[RawPrediction],
    topology_checks: Sequence[TopologyCheckResult],
    per_example: Sequence[PerExampleScore],
    metrics: Mapping[str, float],
    counts: Mapping[str, int],
    failure_counts: Mapping[FailureCategory, int],
    coverage: float,
    failure: EvaluationFailure | None,
    claim_boundary: str,
) -> str:
    digest = canonical_hash(
        {
            "schema": "tabu.eval-result-identity.v5",
            "suite_id": suite_id,
            "suite_version": suite_version,
            "suite_hash": suite_hash,
            "scenario_id": scenario_id,
            "task": task,
            "adapter": adapter,
            "producer": producer,
            "seed": seed,
            "source_sha256": source_sha256,
            "split_sha256": split_sha256,
            "recipe_sha256": recipe_sha256,
            "budget_hash": budget_hash,
            "truth_sidecar_sha256": truth_sidecar_sha256,
            "status": status,
            "raw_predictions": tuple(raw_predictions),
            "topology_checks": tuple(topology_checks),
            "per_example": tuple(per_example),
            "metrics": dict(metrics),
            "counts": dict(counts),
            "failure_counts": dict(failure_counts),
            "coverage": coverage,
            "failure": failure,
            "claim_boundary": claim_boundary,
        }
    )
    return f"eval-{digest[:24]}"


class EvalResult(EvidenceSchema):
    schema_version: Literal["tabu.eval-result.v2"] = "tabu.eval-result.v2"
    result_id: Identifier
    status: EvaluationStatus
    suite_id: Identifier
    suite_version: str
    suite_hash: Sha256
    scenario_id: Identifier
    task: TaskKind
    adapter: AdapterSpec
    producer: EvalProducerBinding
    seed: int = Field(ge=0)
    source_sha256: Sha256
    split_sha256: Sha256
    recipe_sha256: Sha256
    budget_hash: Sha256
    truth_sidecar_sha256: Sha256 | None = None
    raw_predictions: tuple[RawPrediction, ...] = ()
    topology_checks: tuple[TopologyCheckResult, ...] = ()
    per_example: tuple[PerExampleScore, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    failure_counts: dict[FailureCategory, int] = Field(default_factory=dict)
    coverage: float = Field(ge=0.0, le=1.0)
    failure: EvaluationFailure | None = None
    claim_boundary: str = Field(min_length=1)
    execution_receipt: EvalExecutionReceipt | None = None

    @property
    def result_content_hash(self) -> str:
        """Hash every substantive result field before receipt attachment."""

        return canonical_hash(
            self.model_dump(mode="python", exclude={"execution_receipt"})
        )

    @field_validator("suite_hash", "source_sha256", "split_sha256", "recipe_sha256", "budget_hash")
    @classmethod
    def _result_hashes_are_valid(cls, value: str, info: object) -> str:
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("truth_sidecar_sha256")
    @classmethod
    def _result_truth_hash_is_valid(cls, value: str | None) -> str | None:
        return None if value is None else require_sha256(value, field_name="truth_sidecar_sha256")

    @field_validator("metrics")
    @classmethod
    def _aggregate_metrics_are_finite(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("aggregate evaluation metrics must be finite")
        return dict(sorted(values.items()))

    @field_validator("counts")
    @classmethod
    def _counts_are_nonnegative(cls, values: dict[str, int]) -> dict[str, int]:
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("evaluation counts must be non-negative integers")
        return dict(sorted(values.items()))

    @model_validator(mode="after")
    def _status_matches_payload(self) -> EvalResult:
        if (
            self.adapter.kind is AdapterKind.MODEL
            and self.producer.provenance is not ProducerProvenance.RECEIPTED_RUN
        ):
            raise ValueError("model evaluation results require a receipted producer run")
        if (
            self.producer.provenance is ProducerProvenance.UNISSUED_BASELINE
            and self.adapter.kind is not AdapterKind.BASELINE
        ):
            raise ValueError("only baseline results may be explicitly unissued")
        expected_result_id = _result_id_from_components(
            suite_id=self.suite_id,
            suite_version=self.suite_version,
            suite_hash=self.suite_hash,
            scenario_id=self.scenario_id,
            task=self.task,
            adapter=self.adapter,
            producer=self.producer,
            seed=self.seed,
            source_sha256=self.source_sha256,
            split_sha256=self.split_sha256,
            recipe_sha256=self.recipe_sha256,
            budget_hash=self.budget_hash,
            truth_sidecar_sha256=self.truth_sidecar_sha256,
            status=self.status,
            raw_predictions=self.raw_predictions,
            topology_checks=self.topology_checks,
            per_example=self.per_example,
            metrics=self.metrics,
            counts=self.counts,
            failure_counts=self.failure_counts,
            coverage=self.coverage,
            failure=self.failure,
            claim_boundary=self.claim_boundary,
        )
        if self.result_id != expected_result_id:
            raise ValueError("result_id does not match the immutable evaluation identity")
        if self.status is EvaluationStatus.SUCCEEDED:
            if self.failure is not None:
                raise ValueError("successful evaluation cannot contain a terminal failure")
            if not self.raw_predictions or not self.per_example or not self.metrics:
                raise ValueError("successful evaluation needs predictions, scores, and metrics")
            predictions = {item.example_id: item for item in self.raw_predictions}
            scores = {item.example_id: item for item in self.per_example}
            if len(predictions) != len(self.raw_predictions):
                raise ValueError("successful evaluation raw prediction ids must be unique")
            if len(scores) != len(self.per_example):
                raise ValueError("successful evaluation score ids must be unique")
            if set(predictions) != set(scores):
                raise ValueError("predictions and per-example scores must cover identical ids")
            if self.truth_sidecar_sha256 is None or self.truth_sidecar_sha256 != (
                _truth_sidecar_sha256_from_scores(self.per_example)
            ):
                raise ValueError("result truth sidecar does not match retained held-out targets")
            for example_id, prediction in predictions.items():
                score = scores[example_id]
                if score.prediction_sha256 != prediction.content_hash:
                    raise ValueError("per-example score does not bind its retained raw prediction")
                if prediction.abstained == score.scored:
                    raise ValueError("prediction abstention and score status disagree")
                if prediction.abstained and (
                    prediction.failure_category is not score.failure_category
                ):
                    raise ValueError("prediction and score failure categories disagree")
            _validate_scores_against_raw(self.per_example, predictions)
            scored = [item for item in self.per_example if item.scored]
            expected_counts = {
                "targets": len(self.per_example),
                "scored": len(scored),
                "abstained": len(self.per_example) - len(scored),
                "numeric_targets": sum(
                    item.target_kind is TargetKind.NUMERIC for item in self.per_example
                ),
                "categorical_targets": sum(
                    item.target_kind is TargetKind.CATEGORICAL for item in self.per_example
                ),
            }
            if self.counts != expected_counts:
                raise ValueError("evaluation counts do not match per-example scores")
            expected_coverage = len(scored) / len(self.per_example)
            if not math.isclose(
                self.coverage,
                expected_coverage,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("evaluation coverage does not match per-example scores")
            expected_failures = Counter(
                item.failure_category for item in self.per_example if not item.scored
            )
            if self.failure_counts != dict(expected_failures):
                raise ValueError("evaluation failure counts do not match abstained examples")
            raw_by_id = {item.example_id: item for item in self.raw_predictions}
            topology_keys: list[tuple[str, str, str]] = []
            for check in self.topology_checks:
                base = raw_by_id.get(check.base_example_id)
                if base is None or base != check.base_prediction:
                    raise ValueError("topology check base prediction is not retained raw output")
                if base.abstained or check.perturbed_prediction.abstained:
                    raise ValueError("topology checks cannot be reconstructed from abstentions")
                base_semantics = canonical_hash(
                    {
                        "schema": "tabu.eval-prediction-semantics.v1",
                        "value": base.value,
                        "probabilities": base.probabilities,
                    }
                )
                perturbed_semantics = canonical_hash(
                    {
                        "schema": "tabu.eval-prediction-semantics.v1",
                        "value": check.perturbed_prediction.value,
                        "probabilities": check.perturbed_prediction.probabilities,
                    }
                )
                equal = base_semantics == perturbed_semantics
                expected_pass = (
                    equal
                    if check.expected_relation is TopologyRelation.EQUAL
                    else not equal
                )
                if check.passed != expected_pass:
                    raise ValueError(
                        "topology pass flag does not match retained evaluator-owned outputs"
                    )
                topology_keys.append(
                    (
                        check.check_id,
                        check.base_example_id,
                        check.perturbed_example_sha256,
                    )
                )
            if len(topology_keys) != len(set(topology_keys)):
                raise ValueError("topology check results must be unique")
            derived_metrics = _derived_metrics_from_scores(self.per_example, self.topology_checks)
            for metric_id, value in self.metrics.items():
                expected = derived_metrics.get(metric_id)
                if expected is None or not math.isclose(
                    value,
                    expected,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise ValueError(
                        f"evaluation metric is not derivable from retained scores: {metric_id}"
                    )
        else:
            if self.failure is None:
                raise ValueError("failed evaluation needs a classified terminal failure")
            if self.raw_predictions or self.topology_checks or self.per_example or self.metrics:
                raise ValueError("failed evaluation cannot masquerade as scored evidence")
            if self.coverage != 0.0 or self.counts != {
                "targets": 0,
                "scored": 0,
                "abstained": 0,
            }:
                raise ValueError("failed evaluation must retain zero scored counts")
            if self.failure_counts != {self.failure.category: 1}:
                raise ValueError("failed evaluation failure counts must match terminal failure")
        if self.execution_receipt is not None:
            receipt = self.execution_receipt
            expected_producer_sha256 = self.producer.content_hash
            if self.adapter.kind is AdapterKind.BASELINE:
                if receipt.issuance_status == "formal" and (
                    receipt.baseline_spec_sha256 is None
                ):
                    raise ValueError(
                        "formal baseline receipts must bind the frozen baseline spec"
                    )
                if receipt.baseline_spec_sha256 is not None:
                    expected_producer_sha256 = receipt.baseline_spec_sha256
            elif receipt.baseline_spec_sha256 is not None:
                raise ValueError("model evaluation receipts cannot bind a baseline spec")
            expected = (
                receipt.status is self.status
                and receipt.suite_id == self.suite_id
                and receipt.suite_version == self.suite_version
                and receipt.suite_hash == self.suite_hash
                and receipt.scenario_id == self.scenario_id
                and receipt.adapter_sha256 == self.adapter.content_hash
                and receipt.producer_sha256 == expected_producer_sha256
                and receipt.seed == self.seed
                and receipt.source_sha256 == self.source_sha256
                and receipt.split_sha256 == self.split_sha256
                and receipt.recipe_sha256 == self.recipe_sha256
                and receipt.budget_hash == self.budget_hash
                and receipt.truth_sidecar_sha256 == self.truth_sidecar_sha256
                and receipt.result_id == self.result_id
                and receipt.result_content_sha256 == self.result_content_hash
                and receipt.failure_category
                == (self.failure.category if self.failure is not None else None)
            )
            if not expected:
                raise ValueError("evaluation execution receipt does not bind this exact result")
        return self


def bind_evaluation_receipt(
    result: EvalResult,
    *,
    environment: EnvironmentDisclosure,
    source_identity: SourceIdentity,
    started_at: datetime,
    completed_at: datetime | None = None,
) -> EvalResult:
    """Attach a local, explicitly unissued execution receipt to a result.

    Formal issuance is intentionally unavailable through this low-level helper;
    callers must use the Git-backed issuer that verifies evaluator source and
    reviewed dataset authority.
    """

    if source_identity.issuance_status == "formal":
        raise ValueError(
            "self-declared formal evaluation source is forbidden; use the Git-backed "
            "formal evaluation receipt issuer"
        )
    return _bind_evaluation_receipt(
        result,
        environment=environment,
        source_identity=source_identity,
        started_at=started_at,
        completed_at=completed_at,
    )


def _bind_evaluation_receipt(
    result: EvalResult,
    *,
    environment: EnvironmentDisclosure,
    source_identity: SourceIdentity,
    started_at: datetime,
    completed_at: datetime | None = None,
    source_authorization: EvaluatorSourceAuthorization | None = None,
    producer_sha256: str | None = None,
    baseline_spec_sha256: str | None = None,
    dataset_snapshot_id: str | None = None,
    dataset_snapshot_sha256: str | None = None,
    dataset_request_sha256: str | None = None,
    dataset_authority_sha256: str | None = None,
    dataset_authority_status: Literal["reviewed"] | None = None,
) -> EvalResult:
    """Construct one receipt after the caller has closed its authority chain."""

    result = EvalResult.model_validate(result.model_dump(mode="python"))
    if result.execution_receipt is not None:
        raise ValueError("evaluation result already has an execution receipt")
    completed_at = completed_at or datetime.now(UTC)
    resolved_producer_sha256 = producer_sha256 or result.producer.content_hash
    receipt_values = {
        "schema_version": "tabu.eval-execution-receipt.v2",
        "eval_run_id": derive_eval_run_id(
            suite_id=result.suite_id,
            suite_hash=result.suite_hash,
            scenario_id=result.scenario_id,
            adapter_sha256=result.adapter.content_hash,
            producer_sha256=resolved_producer_sha256,
            seed=result.seed,
            source_sha256=result.source_sha256,
            split_sha256=result.split_sha256,
            recipe_sha256=result.recipe_sha256,
            budget_hash=result.budget_hash,
            truth_sidecar_sha256=result.truth_sidecar_sha256,
            baseline_spec_sha256=baseline_spec_sha256,
            dataset_snapshot_id=dataset_snapshot_id,
            dataset_snapshot_sha256=dataset_snapshot_sha256,
            dataset_request_sha256=dataset_request_sha256,
            dataset_authority_sha256=dataset_authority_sha256,
            environment=environment,
            source_identity=source_identity,
            started_at=started_at,
        ),
        "issuance_status": source_identity.issuance_status,
        "status": result.status,
        "suite_id": result.suite_id,
        "suite_version": result.suite_version,
        "suite_hash": result.suite_hash,
        "scenario_id": result.scenario_id,
        "adapter_sha256": result.adapter.content_hash,
        "producer_sha256": resolved_producer_sha256,
        "seed": result.seed,
        "source_sha256": result.source_sha256,
        "split_sha256": result.split_sha256,
        "recipe_sha256": result.recipe_sha256,
        "budget_hash": result.budget_hash,
        "truth_sidecar_sha256": result.truth_sidecar_sha256,
        "baseline_spec_sha256": baseline_spec_sha256,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_snapshot_sha256": dataset_snapshot_sha256,
        "dataset_request_sha256": dataset_request_sha256,
        "dataset_authority_sha256": dataset_authority_sha256,
        "dataset_authority_status": dataset_authority_status,
        "result_id": result.result_id,
        "result_content_sha256": result.result_content_hash,
        "environment": environment,
        "source_identity": source_identity,
        "source_authorization": source_authorization,
        "started_at": started_at,
        "completed_at": completed_at,
        "failure_category": result.failure.category if result.failure is not None else None,
    }
    receipt_id = "evalreceipt-" + canonical_hash(receipt_values)[:24]
    receipt = EvalExecutionReceipt(receipt_id=receipt_id, **receipt_values)
    payload = result.model_dump(mode="python")
    payload["execution_receipt"] = receipt
    return EvalResult.model_validate(payload)


class MetricAggregate(EvidenceSchema):
    schema_version: Literal["tabu.eval-metric-aggregate.v1"] = "tabu.eval-metric-aggregate.v1"
    scenario_id: Identifier
    task: TaskKind
    adapter_id: Identifier
    metric_id: Identifier
    seeds: tuple[int, ...]
    values: tuple[float, ...]
    mean: float
    std: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _aggregate_is_self_consistent(self) -> MetricAggregate:
        if len(self.seeds) != len(self.values) or not self.values:
            raise ValueError("metric aggregate needs one value per seed")
        if any(not math.isfinite(value) for value in (*self.values, self.mean, self.std)):
            raise ValueError("metric aggregate values must be finite")
        expected_mean = sum(self.values) / len(self.values)
        expected_std = (
            sum((value - expected_mean) ** 2 for value in self.values) / len(self.values)
        ) ** 0.5
        if not math.isclose(self.mean, expected_mean, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("metric aggregate mean does not match seed values")
        if not math.isclose(self.std, expected_std, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("metric aggregate std does not match seed values")
        return self


class ComparisonReport(EvidenceSchema):
    schema_version: Literal["tabu.eval-comparison.v2"] = "tabu.eval-comparison.v2"
    comparison_id: Identifier
    suite_id: Identifier
    suite_version: str
    suite_hash: Sha256
    variable_under_test: Literal["model_contract_or_artifact"]
    result_hashes: tuple[Sha256, ...]
    aggregates: tuple[MetricAggregate, ...]
    failure_counts: dict[FailureCategory, int] = Field(default_factory=dict)
    composite_score: Literal[False] = False
    overall_rank: Literal[None] = None
    publication_eligible: bool
    claim_boundary: str = Field(min_length=1)

    @field_validator("suite_hash")
    @classmethod
    def _suite_hash_is_valid(cls, value: str) -> str:
        return require_sha256(value, field_name="suite_hash")

    @field_validator("result_hashes")
    @classmethod
    def _result_hashes_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(require_sha256(value, field_name="result_hash") for value in values)
        if len(normalized) != len(set(normalized)):
            raise ValueError("comparison result hashes must be unique")
        return normalized

    @model_validator(mode="after")
    def _never_aggregates_across_tasks(self) -> ComparisonReport:
        keys = [(item.scenario_id, item.adapter_id, item.metric_id) for item in self.aggregates]
        if len(keys) != len(set(keys)):
            raise ValueError("comparison aggregates must be scenario-scoped and unique")
        return self


def comparison_publication_eligible(
    suite: EvalSuiteSpec,
    results: Sequence[EvalResult],
) -> bool:
    """Return whether one controlled comparison has a shared formal authority.

    Individual formal receipts are necessary but not sufficient for a public
    comparison.  The compared executions must also share the exact evaluator
    source identity and its reviewed authorization, while each scenario must
    use one reviewed dataset authority.  Model/baseline adapter and producer
    identity remain the declared variable under test; seeds are the suite's
    frozen repetitions.

    Local or mixed formal/local inputs remain valid for an explicitly unissued
    comparison, so this predicate returns ``False`` instead of rejecting them.
    Structural controlled-comparison errors are rejected separately by
    :func:`derive_comparison_summary`.
    """

    results = tuple(results)
    receipts = tuple(result.execution_receipt for result in results)
    if not results or any(
        receipt is None or not receipt.publication_eligible for receipt in receipts
    ):
        return False

    formal_receipts = tuple(receipt for receipt in receipts if receipt is not None)
    if any(
        receipt.suite_id != suite.suite_id
        or receipt.suite_version != suite.suite_version
        or receipt.suite_hash != suite.suite_hash
        or receipt.budget_hash != suite.budget.content_hash
        for receipt in formal_receipts
    ):
        return False

    source_identities = {
        canonical_hash(receipt.source_identity) for receipt in formal_receipts
    }
    source_authorizations = {
        canonical_hash(receipt.source_authorization)
        for receipt in formal_receipts
        if receipt.source_authorization is not None
    }
    if len(source_identities) != 1 or len(source_authorizations) != 1:
        return False

    for scenario_id in {result.scenario_id for result in results}:
        dataset_authorities = {
            (
                receipt.dataset_snapshot_id,
                receipt.dataset_snapshot_sha256,
                receipt.dataset_request_sha256,
                receipt.dataset_authority_sha256,
                receipt.dataset_authority_status,
            )
            for result, receipt in zip(results, formal_receipts, strict=True)
            if result.scenario_id == scenario_id
        }
        if len(dataset_authorities) != 1:
            return False

    return (
        suite.variable_under_test == "model_contract_or_artifact"
        and len({result.adapter.adapter_id for result in results}) >= 2
    )


def derive_comparison_summary(
    suite: EvalSuiteSpec,
    results: Sequence[EvalResult],
) -> tuple[tuple[MetricAggregate, ...], dict[FailureCategory, int]]:
    """Recompute the only aggregate summary authorized by exact result objects.

    Both the evaluator and the public catalog call this function.  A comparison
    manifest therefore cannot publish hand-entered numbers that merely cite
    otherwise valid ``result_hashes``.
    """

    results = tuple(results)
    if not results:
        raise ValueError("comparison requires evaluation results")
    if any(
        result.suite_id != suite.suite_id
        or result.suite_version != suite.suite_version
        or result.suite_hash != suite.suite_hash
        or result.budget_hash != suite.budget.content_hash
        for result in results
    ):
        raise ValueError("comparison requires one frozen suite and evaluation budget")
    if len({result.adapter.adapter_id for result in results}) < 2:
        raise ValueError("comparison requires at least two adapters")

    keys = [(item.scenario_id, item.adapter.adapter_id, item.seed) for item in results]
    if len(keys) != len(set(keys)):
        raise ValueError("comparison contains duplicate scenario/adapter/seed results")
    scenarios = {scenario.scenario_id: scenario for scenario in suite.scenarios}
    by_group: defaultdict[tuple[str, str], list[EvalResult]] = defaultdict(list)
    for result in results:
        scenario = scenarios.get(result.scenario_id)
        if scenario is None:
            raise ValueError("comparison result scenario is absent from its suite")
        if result.task is not scenario.task:
            raise ValueError("result task does not match its scenario")
        primary_metrics = {metric.metric_id for metric in scenario.metrics if metric.primary}
        if result.status is EvaluationStatus.SUCCEEDED and not primary_metrics.issubset(
            result.metrics
        ):
            raise ValueError("successful comparison input is missing a frozen primary metric")
        by_group[(result.scenario_id, result.adapter.adapter_id)].append(result)

    expected_seeds = tuple(sorted(suite.budget.model_seeds))
    for (scenario_id, _adapter_id), items in by_group.items():
        seeds = tuple(sorted(item.seed for item in items))
        if seeds != expected_seeds:
            raise ValueError(
                f"{scenario_id} comparison groups need all three frozen seeds; got {seeds}"
            )
        identities = {
            (
                item.source_sha256,
                item.split_sha256,
                item.recipe_sha256,
                item.truth_sidecar_sha256,
            )
            for item in items
        }
        if len(identities) != 1:
            raise ValueError("comparison group mixes data snapshots or split recipes")
    for scenario_id in sorted({item.scenario_id for item in results}):
        identities = {
            (
                item.source_sha256,
                item.split_sha256,
                item.recipe_sha256,
                item.truth_sidecar_sha256,
            )
            for item in results
            if item.scenario_id == scenario_id
        }
        if len(identities) != 1:
            raise ValueError("controlled comparison requires one data identity per scenario")

    aggregates: list[MetricAggregate] = []
    failure_counts: Counter[FailureCategory] = Counter()
    for (scenario_id, adapter_id), items in sorted(by_group.items()):
        for item in items:
            failure_counts.update(item.failure_counts)
        if any(item.status is EvaluationStatus.FAILED for item in items):
            continue
        scenario = scenarios[scenario_id]
        metric_ids = set.intersection(*(set(item.metrics) for item in items))
        ordered_items = sorted(items, key=lambda item: item.seed)
        for metric_id in sorted(metric_ids):
            values = tuple(item.metrics[metric_id] for item in ordered_items)
            mean = sum(values) / len(values)
            std = (
                0.0
                if len(set(values)) == 1
                else (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
            )
            aggregates.append(
                MetricAggregate(
                    scenario_id=scenario_id,
                    task=scenario.task,
                    adapter_id=adapter_id,
                    metric_id=metric_id,
                    seeds=tuple(item.seed for item in ordered_items),
                    values=values,
                    mean=mean,
                    std=std,
                )
            )
    return (
        tuple(aggregates),
        dict(sorted(failure_counts.items(), key=lambda item: item[0].value)),
    )


class ScenarioAvailability(EvidenceSchema):
    schema_version: Literal["tabu.eval-scenario-availability.v1"] = (
        "tabu.eval-scenario-availability.v1"
    )
    scenario_id: Identifier
    ready: bool
    network_required: bool
    blockers: tuple[str, ...] = ()


class DryRunReport(EvidenceSchema):
    schema_version: Literal["tabu.eval-dry-run.v1"] = "tabu.eval-dry-run.v1"
    suite_id: Identifier
    suite_hash: Sha256
    ready: bool
    scenarios: tuple[ScenarioAvailability, ...]
    would_execute: Literal[False] = False


class SuiteValidationReport(EvidenceSchema):
    schema_version: Literal["tabu.eval-suite-validation.v1"] = "tabu.eval-suite-validation.v1"
    suite_id: Identifier
    suite_hash: Sha256
    valid: bool
    issues: tuple[str, ...] = ()


def deterministic_result_id(
    *,
    suite: EvalSuiteSpec,
    scenario: ScenarioSpec,
    adapter: AdapterSpec,
    binding: DatasetSnapshotBinding,
    producer: EvalProducerBinding,
    seed: int,
    status: EvaluationStatus,
    raw_predictions: Sequence[RawPrediction],
    topology_checks: Sequence[TopologyCheckResult],
    per_example: Sequence[PerExampleScore],
    metrics: Mapping[str, float],
    counts: Mapping[str, int],
    failure_counts: Mapping[FailureCategory, int],
    coverage: float,
    failure: EvaluationFailure | None,
    claim_boundary: str,
) -> str:
    return _result_id_from_components(
        suite_id=suite.suite_id,
        suite_version=suite.suite_version,
        suite_hash=suite.suite_hash,
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        adapter=adapter,
        producer=producer,
        seed=seed,
        source_sha256=binding.source_sha256,
        split_sha256=binding.split_sha256,
        recipe_sha256=binding.recipe_sha256,
        budget_hash=suite.budget.content_hash,
        truth_sidecar_sha256=binding.truth_sidecar_sha256,
        status=status,
        raw_predictions=raw_predictions,
        topology_checks=topology_checks,
        per_example=per_example,
        metrics=metrics,
        counts=counts,
        failure_counts=failure_counts,
        coverage=coverage,
        failure=failure,
        claim_boundary=claim_boundary,
    )
