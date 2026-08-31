"""Versioned frozen full-context OpenML evaluation for query-family programs.

The historical TabUR full-context panel is intentionally frozen to the 0.1
contract.  This module defines a separate Program-level request that can bind
independent TabUBase and TabUR snapshots without making either checkpoint a
dependency of the other.  Dataset receipts are emitted one at a time so a
failed or memory-bound dataset cannot invalidate completed siblings.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import platform
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import TruthSidecar, canonical_hash, canonical_json, require_sha256
from tabu_lab.experiments.query_row_openml_full_context import (
    _forward_query_response_only,
    _tabur_full_metrics,
)
from tabu_lab.experiments.query_row_transfer_common import aligned_probabilities
from tabu_lab.experiments.tabubase_openml_new6 import (
    OPENML_NEW6_BY_ID,
    OPENML_NEW6_SPECS,
    fetch_openml_new6_dataset,
)
from tabu_lab.experiments.tabubase_real_benchmark import _source_tree_hash
from tabu_lab.experiments.tabubase_real_icl import (
    FULL_CONTEXT_POLICY,
    PreparedRealIclSplit,
    build_real_icl_episode,
    prepare_real_icl_split,
    real_icl_split_manifest,
)
from tabu_lab.experiments.tabubase_real_metrics import (
    classification_metrics,
    regression_metrics,
)

from .checkpoint import (
    file_sha256,
    program_sidecar_path,
    read_checkpoint_model_state,
    read_program_checkpoint,
)
from .evaluation import ProgramCheckpointEvaluationReceipt
from .models import ComponentGraphNode, EvidenceStatus, ProgramArtifact, StrictModel
from .repository import EvolutionRepository
from .runtime import _build_runtime_model

_ID = re.compile(r"^[a-z][a-z0-9_.-]*$")
_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_DATASET_IDS = tuple(spec.dataset_id for spec in OPENML_NEW6_SPECS)
_TASK_PRIMARY = {"classification": "normalized_nll", "regression": "scaled_rmse"}


class OpenMLSourceContract(StrictModel):
    provider: Literal["OpenML"] = "OpenML"
    api: Literal["sklearn.datasets.fetch_openml"] = "sklearn.datasets.fetch_openml"
    identity_key: Literal["data_id"] = "data_id"
    target_column: Literal["default-target"] = "default-target"
    as_frame: Literal[False] = False
    parser: Literal["liac-arff"] = "liac-arff"
    cache: Literal[True] = True


class OpenMLDatasetIdentity(StrictModel):
    dataset_id: str
    openml_name: str
    data_id: int = Field(gt=0)
    version: int = Field(gt=0)
    upstream_md5: str
    license: str
    task: Literal["classification", "regression"]
    rows: int = Field(gt=0)
    predictors: int = Field(gt=0)
    classes: int | None = Field(default=None, ge=2)
    missing_values: Literal[0] = 0

    @model_validator(mode="after")
    def _matches_pinned_source(self) -> OpenMLDatasetIdentity:
        try:
            spec = OPENML_NEW6_BY_ID[self.dataset_id]
        except KeyError as exc:
            raise ValueError(f"dataset is not in OpenML new6: {self.dataset_id}") from exc
        expected = {
            "openml_name": spec.openml_name,
            "data_id": spec.data_id,
            "version": spec.version,
            "upstream_md5": spec.upstream_md5,
            "license": spec.license,
            "task": spec.task,
            "rows": spec.rows,
            "predictors": spec.predictors,
            "classes": spec.classes,
        }
        observed = {key: getattr(self, key) for key in expected}
        if observed != expected:
            raise ValueError(
                f"OpenML new6 identity drift for {self.dataset_id}: "
                f"expected {expected!r}, observed {observed!r}"
            )
        return self


class OpenMLSplitProtocol(StrictModel):
    split_seeds: tuple[int, ...]
    train_fraction: Literal[0.7] = 0.7
    context_policy: Literal["full_train"] = "full_train"
    context_rows: Literal["all_train_partition_rows"] = "all_train_partition_rows"
    query_policy: Literal["all_heldout_rows"] = "all_heldout_rows"
    query_limit: None = None
    split_before_feature_selection: Literal[True] = True
    max_predictors: Literal[63] = 63
    feature_selection: Literal["train_only_variance_if_needed"] = (
        "train_only_variance_if_needed"
    )
    target_truth_visibility: Literal["scorer_only_sidecar"] = "scorer_only_sidecar"

    @field_validator("split_seeds")
    @classmethod
    def _three_frozen_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (1729, 2718, 31415):
            raise ValueError("OpenML new6 full-context panel requires seeds 1729, 2718, 31415")
        return value


class OpenMLDataPanel(StrictModel):
    schema_version: Literal["tabu.openml-data-panel.v1"] = "tabu.openml-data-panel.v1"
    panel_id: Literal["tabu.openml.numeric-nomissing-new6"] = (
        "tabu.openml.numeric-nomissing-new6"
    )
    version: Literal["1.0.0"] = "1.0.0"
    status: Literal["active_for_local_unissued_evaluation"]
    description: str
    source_contract: OpenMLSourceContract
    datasets: tuple[OpenMLDatasetIdentity, ...]
    split_protocol: OpenMLSplitProtocol

    @model_validator(mode="after")
    def _complete_panel(self) -> OpenMLDataPanel:
        ids = tuple(item.dataset_id for item in self.datasets)
        if ids != _DATASET_IDS:
            raise ValueError(
                f"OpenML data panel must contain the ordered new6 set {_DATASET_IDS!r}"
            )
        return self

    @property
    def ref(self) -> str:
        return f"{self.panel_id}@{self.version}"

    @property
    def panel_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class DataPanelBinding(StrictModel):
    ref: Literal["tabu.openml.numeric-nomissing-new6@1.0.0"]
    path: str
    canonical_payload_sha256: str

    @field_validator("path")
    @classmethod
    def _repository_relative_path(cls, value: str) -> str:
        normalized = value.removeprefix("./")
        if not normalized or Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise ValueError("data panel path must be repository-relative")
        return normalized

    @field_validator("canonical_payload_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="data panel payload hash")


class FullContextSplitBinding(StrictModel):
    split_seeds: tuple[int, ...]
    train_fraction: Literal[0.7] = 0.7
    context_policy: Literal["full_train"] = "full_train"
    query_policy: Literal["all_heldout_rows"] = "all_heldout_rows"
    query_chunk_rows: Literal[64] = 64

    @field_validator("split_seeds")
    @classmethod
    def _frozen_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != (1729, 2718, 31415):
            raise ValueError("full-context request requires the frozen three split seeds")
        return value


class LinearClassificationSpec(StrictModel):
    estimator: Literal["sklearn.linear_model.LogisticRegression"]
    C: Literal[1.0] = 1.0
    max_iter: Literal[500] = 500
    solver: Literal["lbfgs"] = "lbfgs"
    preprocessing: Literal["none"] = "none"
    random_state: Literal["split_seed"] = "split_seed"


class LinearRegressionSpec(StrictModel):
    estimator: Literal["sklearn.linear_model.Ridge"]
    alpha: Literal[0.0001] = 0.0001
    preprocessing: Literal["none"] = "none"


class LinearBaselineSpec(StrictModel):
    classification: LinearClassificationSpec
    regression: LinearRegressionSpec


class CapabilityRule(StrictModel):
    lower_is_better: Literal[True] = True
    classification_primary_metric: Literal["normalized_nll"] = "normalized_nll"
    regression_primary_metric: Literal["scaled_rmse"] = "scaled_rmse"
    panel_success: str


class FrozenProgramArm(StrictModel):
    arm_id: Literal["tabu_base", "tabu_row"]
    program_ref: str
    model_contract_ref: str
    component_graph_ref: str
    snapshot_hash: str
    run_identity_hash: str
    training_run_receipt_hash: str
    source_training_commit: str
    checkpoint_name: str
    checkpoint_step: int = Field(gt=0)
    checkpoint_sha256: str
    checkpoint_sidecar_sha256: str
    checkpoint_metadata_hash: str
    selection_evaluation_request_hash: str
    selection_evaluation_receipt_hash: str

    @field_validator(
        "snapshot_hash",
        "run_identity_hash",
        "training_run_receipt_hash",
        "checkpoint_sha256",
        "checkpoint_sidecar_sha256",
        "checkpoint_metadata_hash",
        "selection_evaluation_request_hash",
        "selection_evaluation_receipt_hash",
    )
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @field_validator("source_training_commit")
    @classmethod
    def _valid_revision(cls, value: str) -> str:
        if _GIT_REVISION.fullmatch(value) is None:
            raise ValueError("source_training_commit must be a full Git revision")
        return value

    @model_validator(mode="after")
    def _arm_family_is_explicit(self) -> FrozenProgramArm:
        expected = {
            "tabu_base": (
                "tabu.pretraining.query-base@1.3.0",
                "tabu.query.base@0.1.0",
                "tabu.graph.query.base@1.1.0",
            ),
            "tabu_row": (
                "tabu.pretraining.query-row@1.3.0",
                "tabu.query.row@0.2.0",
                "tabu.graph.query.row@1.1.0",
            ),
        }[self.arm_id]
        observed = (self.program_ref, self.model_contract_ref, self.component_graph_ref)
        if observed != expected:
            raise ValueError(f"{self.arm_id} family refs drifted: {observed!r} != {expected!r}")
        if Path(self.checkpoint_name).name != self.checkpoint_name:
            raise ValueError("checkpoint_name must be a basename")
        return self


class ProgramOpenMLFullContextRequest(StrictModel):
    schema_version: Literal["tabu.program-openml-full-context-request.v1"] = (
        "tabu.program-openml-full-context-request.v1"
    )
    request_id: str
    version: str
    evidence_status: Literal[EvidenceStatus.LOCAL_UNISSUED] = EvidenceStatus.LOCAL_UNISSUED
    claim_boundary: str
    source_repository_hash: str
    data_panel: DataPanelBinding
    split_protocol: FullContextSplitBinding
    linear_baseline: LinearBaselineSpec
    capability_rule: CapabilityRule
    arms: tuple[FrozenProgramArm, ...]

    @field_validator("request_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if _ID.fullmatch(value) is None:
            raise ValueError("request_id must be a namespaced lowercase identifier")
        return value

    @field_validator("version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        if _VERSION.fullmatch(value) is None:
            raise ValueError("request version must be semantic-versioned")
        return value

    @field_validator("source_repository_hash")
    @classmethod
    def _valid_repository_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source repository hash")

    @model_validator(mode="after")
    def _complete_request(self) -> ProgramOpenMLFullContextRequest:
        if not self.claim_boundary.strip() or not self.capability_rule.panel_success.strip():
            raise ValueError("claim boundary and capability rule cannot be blank")
        if tuple(arm.arm_id for arm in self.arms) != ("tabu_base", "tabu_row"):
            raise ValueError("request must bind independent tabu_base then tabu_row arms")
        return self

    @property
    def ref(self) -> str:
        return f"{self.request_id}@{self.version}"

    @property
    def request_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="python"))


class BaselineSplitResult(StrictModel):
    split_seed: int
    metrics: dict[str, float]
    fit: dict[str, JsonValue]


class ArmSplitResult(StrictModel):
    split_seed: int
    metrics: dict[str, float]
    prediction_sha256: str
    truth_sidecar_sha256: str
    truth_substitution_checked: bool
    substituted_truth_sidecar_sha256: str | None = None
    truth_substitution_prediction_sha256: str | None = None
    truth_substitution_prediction_unchanged: bool | None = None

    @field_validator(
        "prediction_sha256",
        "truth_sidecar_sha256",
        "substituted_truth_sidecar_sha256",
        "truth_substitution_prediction_sha256",
    )
    @classmethod
    def _valid_optional_hash(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _truth_control_is_complete(self) -> ArmSplitResult:
        controls = (
            self.substituted_truth_sidecar_sha256,
            self.truth_substitution_prediction_sha256,
            self.truth_substitution_prediction_unchanged,
        )
        if self.truth_substitution_checked:
            if any(item is None for item in controls):
                raise ValueError("truth-substitution control is incomplete")
            if self.truth_substitution_prediction_unchanged is not True:
                raise ValueError("TruthSidecar substitution changed model prediction")
            if self.prediction_sha256 != self.truth_substitution_prediction_sha256:
                raise ValueError("truth-substitution prediction hash differs")
        elif any(item is not None for item in controls):
            raise ValueError("unchecked truth-substitution control must be empty")
        return self


class ArmDatasetResult(StrictModel):
    arm_id: Literal["tabu_base", "tabu_row"]
    program_ref: str
    snapshot_hash: str
    run_identity_hash: str
    checkpoint: ProgramArtifact
    checkpoint_sidecar: ProgramArtifact
    checkpoint_metadata_hash: str
    selection_evaluation_receipt_hash: str
    optimizer_created: Literal[False] = False
    trainable_parameter_count: Literal[0] = 0
    model_state_hash_before: str
    model_state_hash_after: str
    splits: tuple[ArmSplitResult, ...]

    @field_validator(
        "snapshot_hash",
        "run_identity_hash",
        "checkpoint_metadata_hash",
        "selection_evaluation_receipt_hash",
        "model_state_hash_before",
        "model_state_hash_after",
    )
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _frozen_and_covered(self) -> ArmDatasetResult:
        if self.model_state_hash_before != self.model_state_hash_after:
            raise ValueError("frozen OpenML evaluation mutated model state")
        seeds = tuple(item.split_seed for item in self.splits)
        if seeds != (1729, 2718, 31415):
            raise ValueError("arm result does not cover the frozen split seeds")
        if sum(item.truth_substitution_checked for item in self.splits) != 1:
            raise ValueError("each arm requires exactly one TruthSidecar substitution control")
        return self


class DatasetExecution(StrictModel):
    evaluation_source_revision: str
    evaluation_source_archive_sha256: str
    source_tree_sha256: str
    evaluation_tool: ProgramArtifact
    environment: dict[str, JsonValue]

    @field_validator("evaluation_source_revision")
    @classmethod
    def _valid_revision(cls, value: str) -> str:
        if _GIT_REVISION.fullmatch(value) is None:
            raise ValueError("evaluation source revision must be a full Git revision")
        return value

    @field_validator("evaluation_source_archive_sha256", "source_tree_sha256")
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))


class ProgramOpenMLDatasetReceipt(StrictModel):
    schema_version: Literal["tabu.program-openml-full-context-dataset-receipt.v1"] = (
        "tabu.program-openml-full-context-dataset-receipt.v1"
    )
    evidence_status: Literal[EvidenceStatus.LOCAL_UNISSUED] = EvidenceStatus.LOCAL_UNISSUED
    claim_boundary: str
    request: ProgramOpenMLFullContextRequest
    request_hash: str
    source_repository_hash: str
    data_panel: OpenMLDataPanel
    data_panel_hash: str
    data_panel_artifact: ProgramArtifact
    dataset_id: str
    task: Literal["classification", "regression"]
    primary_metric: Literal["normalized_nll", "scaled_rmse"]
    source_manifest: dict[str, JsonValue]
    source_manifest_sha256: str
    split_manifests: tuple[dict[str, JsonValue], ...]
    linear_baseline: tuple[BaselineSplitResult, ...]
    arms: tuple[ArmDatasetResult, ...]
    execution: DatasetExecution
    receipt_hash: str

    @field_validator(
        "request_hash",
        "source_repository_hash",
        "data_panel_hash",
        "source_manifest_sha256",
        "receipt_hash",
    )
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _receipt_is_consistent(self) -> ProgramOpenMLDatasetReceipt:
        if self.request_hash != self.request.request_hash:
            raise ValueError("dataset receipt request hash differs")
        if self.claim_boundary != self.request.claim_boundary:
            raise ValueError("dataset receipt claim boundary differs")
        if self.source_repository_hash != self.request.source_repository_hash:
            raise ValueError("dataset receipt repository hash differs")
        if self.data_panel_hash != self.data_panel.panel_hash:
            raise ValueError("dataset receipt data panel hash differs")
        if self.request.data_panel.canonical_payload_sha256 != self.data_panel_hash:
            raise ValueError("request and receipt data panel hashes differ")
        panel_by_id = {item.dataset_id: item for item in self.data_panel.datasets}
        if self.dataset_id not in panel_by_id:
            raise ValueError("dataset receipt is outside the frozen data panel")
        if panel_by_id[self.dataset_id].task != self.task:
            raise ValueError("dataset receipt task differs from frozen data panel")
        if self.primary_metric != _TASK_PRIMARY[self.task]:
            raise ValueError("dataset primary metric does not match task")
        if canonical_hash(self.source_manifest) != self.source_manifest_sha256:
            raise ValueError("source manifest hash does not match source manifest")
        seeds = self.request.split_protocol.split_seeds
        if tuple(item["split_seed"] for item in self.split_manifests) != seeds:
            raise ValueError("split manifests do not cover the frozen split seeds")
        if tuple(item.split_seed for item in self.linear_baseline) != seeds:
            raise ValueError("Linear baseline does not cover the frozen split seeds")
        if tuple(item.arm_id for item in self.arms) != tuple(
            item.arm_id for item in self.request.arms
        ):
            raise ValueError("dataset receipt arm set differs from request")
        for frozen, result in zip(self.request.arms, self.arms, strict=True):
            observed = (
                result.program_ref,
                result.snapshot_hash,
                result.run_identity_hash,
                result.checkpoint.name,
                result.checkpoint.sha256,
                result.checkpoint_sidecar.sha256,
                result.checkpoint_metadata_hash,
                result.selection_evaluation_receipt_hash,
            )
            expected = (
                frozen.program_ref,
                frozen.snapshot_hash,
                frozen.run_identity_hash,
                frozen.checkpoint_name,
                frozen.checkpoint_sha256,
                frozen.checkpoint_sidecar_sha256,
                frozen.checkpoint_metadata_hash,
                frozen.selection_evaluation_receipt_hash,
            )
            if observed != expected:
                raise ValueError(f"dataset receipt identity differs for {frozen.arm_id}")
        payload = self.model_dump(mode="python", exclude={"receipt_hash"})
        if canonical_hash(payload) != self.receipt_hash:
            raise ValueError("receipt_hash does not match OpenML dataset receipt")
        return self


class ProgramOpenMLPanelReceipt(StrictModel):
    schema_version: Literal["tabu.program-openml-full-context-panel-receipt.v1"] = (
        "tabu.program-openml-full-context-panel-receipt.v1"
    )
    evidence_status: Literal[EvidenceStatus.LOCAL_UNISSUED] = EvidenceStatus.LOCAL_UNISSUED
    claim_boundary: str
    request: ProgramOpenMLFullContextRequest
    request_hash: str
    data_panel_hash: str
    dataset_receipt_hashes: dict[str, str]
    datasets: dict[str, JsonValue]
    task_macros: dict[str, JsonValue]
    arm_panel_success: dict[str, bool]
    receipt_hash: str

    @field_validator("request_hash", "data_panel_hash", "receipt_hash")
    @classmethod
    def _valid_hash(cls, value: str, info: Any) -> str:
        return require_sha256(value, field_name=str(info.field_name))

    @field_validator("dataset_receipt_hashes")
    @classmethod
    def _valid_receipt_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            key: require_sha256(item, field_name=f"dataset receipt {key}")
            for key, item in value.items()
        }

    @model_validator(mode="after")
    def _panel_is_consistent(self) -> ProgramOpenMLPanelReceipt:
        if self.request_hash != self.request.request_hash:
            raise ValueError("panel receipt request hash differs")
        if self.claim_boundary != self.request.claim_boundary:
            raise ValueError("panel receipt claim boundary differs")
        if tuple(self.dataset_receipt_hashes) != _DATASET_IDS:
            raise ValueError("panel receipt must bind all six ordered dataset receipts")
        if set(self.arm_panel_success) != {arm.arm_id for arm in self.request.arms}:
            raise ValueError("panel success map differs from request arms")
        payload = self.model_dump(mode="python", exclude={"receipt_hash"})
        if canonical_hash(payload) != self.receipt_hash:
            raise ValueError("receipt_hash does not match OpenML panel receipt")
        return self


@dataclass(frozen=True, slots=True)
class LoadedDataPanel:
    panel: OpenMLDataPanel
    artifact: ProgramArtifact


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a mapping")
    return payload


def load_openml_data_panel(path: str | Path) -> LoadedDataPanel:
    source = Path(path).resolve()
    payload = _load_yaml_mapping(source, label="OpenML data panel")
    panel = OpenMLDataPanel.model_validate(payload)
    return LoadedDataPanel(
        panel=panel,
        artifact=ProgramArtifact(
            name=source.name,
            sha256=file_sha256(source),
            size_bytes=source.stat().st_size,
        ),
    )


def load_program_openml_full_context_request(
    path: str | Path,
) -> ProgramOpenMLFullContextRequest:
    return ProgramOpenMLFullContextRequest.model_validate(
        _load_yaml_mapping(Path(path).resolve(), label="Program OpenML full-context request")
    )


def load_program_openml_dataset_receipt(path: str | Path) -> ProgramOpenMLDatasetReceipt:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid OpenML dataset receipt: {exc}") from exc
    return ProgramOpenMLDatasetReceipt.model_validate(payload)


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(canonical_json(tuple(value.shape)).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _prediction_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(canonical_json(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_receipt(destination: Path, receipt: StrictModel) -> None:
    if destination.exists():
        raise ValueError(f"refusing to overwrite OpenML receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(receipt.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _fit_linear_baseline(
    split: PreparedRealIclSplit,
    *,
    seed: int,
    spec: LinearBaselineSpec,
) -> BaselineSplitResult:
    try:
        import sklearn
        from sklearn.linear_model import LogisticRegression, Ridge
    except ImportError as exc:  # pragma: no cover - optional evaluation dependency
        raise RuntimeError("Linear OpenML evaluation requires scikit-learn") from exc

    context = split.context_order
    x_train = np.asarray(split.features[context], dtype=np.float32)
    x_query = np.asarray(split.features[split.query_indices], dtype=np.float32)
    y_train = np.asarray(split.response[context])
    truth = np.asarray(split.response[split.query_indices])
    fit: dict[str, JsonValue] = {
        "fit_rows": len(context),
        "query_rows": len(split.query_indices),
        "preprocessing": "none",
        "scikit_learn_version": sklearn.__version__,
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if split.dataset.task == "classification":
            assert split.classes is not None
            config = spec.classification
            model = LogisticRegression(
                C=config.C,
                max_iter=config.max_iter,
                solver=config.solver,
                random_state=seed,
            )
            model.fit(x_train, y_train)
            probabilities = aligned_probabilities(model, x_query, classes=split.classes)
            metrics = classification_metrics(truth, probabilities, classes=split.classes)
            fit["estimator"] = config.estimator
            fit["estimator_config"] = {
                "C": config.C,
                "max_iter": config.max_iter,
                "solver": config.solver,
                "random_state": seed,
            }
            fit["n_iter"] = [int(item) for item in np.asarray(model.n_iter_).tolist()]
        else:
            config = spec.regression
            model = Ridge(alpha=config.alpha)
            model.fit(x_train, y_train)
            predicted = np.asarray(model.predict(x_query), dtype=np.float64)
            metrics = regression_metrics(truth, predicted, target_scale=split.target_scale)
            fit["estimator"] = config.estimator
            fit["estimator_config"] = {"alpha": config.alpha}
        fit["warnings"] = [str(item.message) for item in caught]
    return BaselineSplitResult(split_seed=seed, metrics=metrics, fit=fit)


def _substituted_truth(truth: TruthSidecar) -> TruthSidecar:
    values = truth.target_values.clone()
    values[truth.target_mask] = values[truth.target_mask] + 17.0
    return TruthSidecar(
        episode_id=truth.episode_id,
        recipe_hash=canonical_hash(
            {"source_truth_hash": truth.truth_hash, "control": "add-17-to-targets"}
        ),
        row_ids=truth.row_ids,
        feature_names=truth.feature_names,
        target_values=values,
        target_mask=truth.target_mask,
        metadata={"control": "truth-substitution-forward-invariance"},
    )


def _repeat_prediction(
    model: torch.nn.Module,
    evidence: Any,
    split: PreparedRealIclSplit,
    *,
    context_rows: int,
    query_chunk_rows: int,
    device: torch.device,
) -> torch.Tensor:
    with torch.inference_mode():
        probabilities, predicted = _forward_query_response_only(
            model,
            evidence,
            context_rows=context_rows,
            classes=split.classes,
            query_chunk_rows=query_chunk_rows,
            device=device,
        )
    value = probabilities if probabilities is not None else predicted
    assert value is not None
    if value.ndim >= 2 and value.shape[0] == 1:
        value = value[0]
    return torch.from_numpy(value)


def _load_selection_receipt(path: Path) -> ProgramCheckpointEvaluationReceipt:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint selection evaluation receipt: {exc}") from exc
    return ProgramCheckpointEvaluationReceipt.model_validate(payload)


def _validated_frozen_model(
    repository: EvolutionRepository,
    arm: FrozenProgramArm,
    *,
    checkpoint: Path,
    selection_receipt_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, ProgramArtifact, ProgramArtifact]:
    resolved = repository.resolve(arm.program_ref)
    if resolved.snapshot_hash != arm.snapshot_hash:
        raise ValueError(f"{arm.arm_id} resolved snapshot differs from request")
    if resolved.slots["model_contract"].ref != arm.model_contract_ref:
        raise ValueError(f"{arm.arm_id} model contract differs from request")
    if resolved.slots["component_graph"].ref != arm.component_graph_ref:
        raise ValueError(f"{arm.arm_id} component graph differs from request")
    graph = repository.node(resolved.slots["component_graph"].ref)
    if not isinstance(graph, ComponentGraphNode):
        raise TypeError(f"{arm.arm_id} component graph is invalid")

    checkpoint = checkpoint.resolve()
    if checkpoint.name != arm.checkpoint_name:
        raise ValueError(f"{arm.arm_id} checkpoint basename differs from request")
    sidecar = program_sidecar_path(checkpoint)
    checkpoint_hash = file_sha256(checkpoint)
    sidecar_hash = file_sha256(sidecar)
    if checkpoint_hash != arm.checkpoint_sha256:
        raise ValueError(f"{arm.arm_id} checkpoint hash differs from request")
    if sidecar_hash != arm.checkpoint_sidecar_sha256:
        raise ValueError(f"{arm.arm_id} checkpoint sidecar hash differs from request")
    metadata = read_program_checkpoint(checkpoint)
    if metadata.metadata_hash != arm.checkpoint_metadata_hash:
        raise ValueError(f"{arm.arm_id} checkpoint metadata hash differs from request")
    if metadata.resolved_snapshot.snapshot_hash != arm.snapshot_hash:
        raise ValueError(f"{arm.arm_id} checkpoint snapshot differs from request")
    if metadata.run_identity_hash != arm.run_identity_hash:
        raise ValueError(f"{arm.arm_id} checkpoint run identity differs from request")
    if metadata.update_cursor != arm.checkpoint_step:
        raise ValueError(f"{arm.arm_id} checkpoint step differs from request")

    selection = _load_selection_receipt(selection_receipt_path.resolve())
    if selection.receipt_hash != arm.selection_evaluation_receipt_hash:
        raise ValueError(f"{arm.arm_id} selection evaluation receipt hash differs")
    if selection.request_hash != arm.selection_evaluation_request_hash:
        raise ValueError(f"{arm.arm_id} selection evaluation request hash differs")
    if selection.snapshot_hash != arm.snapshot_hash:
        raise ValueError(f"{arm.arm_id} selection snapshot differs")
    if selection.run_identity_hash != arm.run_identity_hash:
        raise ValueError(f"{arm.arm_id} selection run identity differs")
    if selection.request.training_run_receipt_hash != arm.training_run_receipt_hash:
        raise ValueError(f"{arm.arm_id} training receipt lineage differs")
    if selection.request.source_training_commit != arm.source_training_commit:
        raise ValueError(f"{arm.arm_id} training source revision differs")
    if selection.checkpoint.sha256 != checkpoint_hash:
        raise ValueError(f"{arm.arm_id} selection checkpoint differs")
    if selection.checkpoint_sidecar.sha256 != sidecar_hash:
        raise ValueError(f"{arm.arm_id} selection checkpoint sidecar differs")
    if selection.checkpoint_metadata_hash != metadata.metadata_hash:
        raise ValueError(f"{arm.arm_id} selection checkpoint metadata differs")

    torch.manual_seed(1729)
    model = _build_runtime_model(graph, str(device))
    model.load_state_dict(read_checkpoint_model_state(checkpoint), strict=True)
    model.eval()
    model.requires_grad_(False)
    checkpoint_artifact = ProgramArtifact(
        name=checkpoint.name,
        sha256=checkpoint_hash,
        size_bytes=checkpoint.stat().st_size,
    )
    sidecar_artifact = ProgramArtifact(
        name=sidecar.name,
        sha256=sidecar_hash,
        size_bytes=sidecar.stat().st_size,
    )
    return model, checkpoint_artifact, sidecar_artifact


def _environment(device: torch.device) -> dict[str, JsonValue]:
    cuda_name: str | None = None
    if device.type == "cuda":
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "architecture": platform.machine().lower(),
        "cuda_device_name": cuda_name,
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "device": str(device),
        "numpy_version": str(np.__version__),
        "operating_system": platform.system().lower(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
    }


def run_program_openml_full_context_dataset(
    repository: EvolutionRepository,
    *,
    request: ProgramOpenMLFullContextRequest,
    dataset_id: str,
    checkpoints: dict[str, Path],
    selection_receipts: dict[str, Path],
    openml_data_home: Path,
    device: str,
    evaluation_source_revision: str,
    evaluation_source_archive_sha256: str,
    output: Path,
) -> ProgramOpenMLDatasetReceipt:
    """Evaluate both frozen siblings plus one matched Linear baseline on one dataset."""

    if output.exists():
        raise ValueError(f"refusing to overwrite OpenML receipt: {output}")
    if repository.repository_hash != request.source_repository_hash:
        raise ValueError("evaluation repository hash differs from frozen request")
    if _GIT_REVISION.fullmatch(evaluation_source_revision) is None:
        raise ValueError("evaluation_source_revision must be a full Git revision")
    require_sha256(
        evaluation_source_archive_sha256,
        field_name="evaluation_source_archive_sha256",
    )
    expected_arms = {arm.arm_id for arm in request.arms}
    if set(checkpoints) != expected_arms or set(selection_receipts) != expected_arms:
        raise ValueError("checkpoint and selection receipt maps must cover both request arms")
    if dataset_id not in _DATASET_IDS:
        raise ValueError(f"dataset is not in frozen OpenML new6 panel: {dataset_id}")

    panel_path = repository.root / request.data_panel.path
    loaded_panel = load_openml_data_panel(panel_path)
    if loaded_panel.panel.ref != request.data_panel.ref:
        raise ValueError("data panel ref differs from frozen request")
    if loaded_panel.panel.panel_hash != request.data_panel.canonical_payload_sha256:
        raise ValueError("data panel payload hash differs from frozen request")
    fetched = fetch_openml_new6_dataset(
        dataset_id,
        cache=True,
        data_home=openml_data_home,
    )
    splits = tuple(
        prepare_real_icl_split(fetched.dataset, split_seed=seed)
        for seed in request.split_protocol.split_seeds
    )
    split_manifests = tuple(real_icl_split_manifest(split) for split in splits)
    baselines = tuple(
        _fit_linear_baseline(split, seed=split.split_seed, spec=request.linear_baseline)
        for split in splits
    )

    target_device = torch.device(device)
    arm_results: list[ArmDatasetResult] = []
    for arm in request.arms:
        model, checkpoint_artifact, sidecar_artifact = _validated_frozen_model(
            repository,
            arm,
            checkpoint=checkpoints[arm.arm_id],
            selection_receipt_path=selection_receipts[arm.arm_id],
            device=target_device,
        )
        state_before = _model_state_hash(model)
        split_results: list[ArmSplitResult] = []
        for index, split in enumerate(splits):
            evidence, truth = build_real_icl_episode(
                split,
                context_size=len(split.train_indices),
                query_indices=split.query_indices,
                shuffled_context=False,
                context_policy=FULL_CONTEXT_POLICY,
            )
            metrics, prediction = _tabur_full_metrics(
                model,
                evidence,
                split,
                context_rows=len(split.train_indices),
                device=target_device,
            )
            prediction_sha = _prediction_hash(prediction)
            if index == 0:
                substituted = _substituted_truth(truth)
                repeated = _repeat_prediction(
                    model,
                    evidence,
                    split,
                    context_rows=len(split.train_indices),
                    query_chunk_rows=request.split_protocol.query_chunk_rows,
                    device=target_device,
                )
                repeated_sha = _prediction_hash(repeated)
                split_result = ArmSplitResult(
                    split_seed=split.split_seed,
                    metrics=metrics,
                    prediction_sha256=prediction_sha,
                    truth_sidecar_sha256=truth.truth_hash,
                    truth_substitution_checked=True,
                    substituted_truth_sidecar_sha256=substituted.truth_hash,
                    truth_substitution_prediction_sha256=repeated_sha,
                    truth_substitution_prediction_unchanged=(prediction_sha == repeated_sha),
                )
                del substituted, repeated
            else:
                split_result = ArmSplitResult(
                    split_seed=split.split_seed,
                    metrics=metrics,
                    prediction_sha256=prediction_sha,
                    truth_sidecar_sha256=truth.truth_hash,
                    truth_substitution_checked=False,
                )
            split_results.append(split_result)
            del evidence, truth, prediction
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        state_after = _model_state_hash(model)
        arm_results.append(
            ArmDatasetResult(
                arm_id=arm.arm_id,
                program_ref=arm.program_ref,
                snapshot_hash=arm.snapshot_hash,
                run_identity_hash=arm.run_identity_hash,
                checkpoint=checkpoint_artifact,
                checkpoint_sidecar=sidecar_artifact,
                checkpoint_metadata_hash=arm.checkpoint_metadata_hash,
                selection_evaluation_receipt_hash=arm.selection_evaluation_receipt_hash,
                optimizer_created=False,
                trainable_parameter_count=sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                model_state_hash_before=state_before,
                model_state_hash_after=state_after,
                splits=tuple(split_results),
            )
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tool_path = Path(__file__).resolve()
    execution = DatasetExecution(
        evaluation_source_revision=evaluation_source_revision,
        evaluation_source_archive_sha256=evaluation_source_archive_sha256,
        source_tree_sha256=_source_tree_hash(),
        evaluation_tool=ProgramArtifact(
            name="src/tabu_lab/evolution/openml_full_context.py",
            sha256=file_sha256(tool_path),
            size_bytes=tool_path.stat().st_size,
        ),
        environment=_environment(target_device),
    )
    payload: dict[str, Any] = {
        "schema_version": "tabu.program-openml-full-context-dataset-receipt.v1",
        "evidence_status": EvidenceStatus.LOCAL_UNISSUED,
        "claim_boundary": request.claim_boundary,
        "request": request,
        "request_hash": request.request_hash,
        "source_repository_hash": repository.repository_hash,
        "data_panel": loaded_panel.panel,
        "data_panel_hash": loaded_panel.panel.panel_hash,
        "data_panel_artifact": loaded_panel.artifact,
        "dataset_id": dataset_id,
        "task": fetched.spec.task,
        "primary_metric": _TASK_PRIMARY[fetched.spec.task],
        "source_manifest": fetched.source_manifest,
        "source_manifest_sha256": fetched.source_manifest_sha256,
        "split_manifests": split_manifests,
        "linear_baseline": baselines,
        "arms": tuple(arm_results),
        "execution": execution,
    }
    receipt = ProgramOpenMLDatasetReceipt(**payload, receipt_hash=canonical_hash(payload))
    _write_receipt(output, receipt)
    return receipt


def _mean_primary(values: tuple[dict[str, float], ...], primary: str) -> float:
    selected = tuple(float(item[primary]) for item in values)
    if not selected or not all(math.isfinite(item) for item in selected):
        raise ValueError(f"primary metric {primary} is absent or non-finite")
    return float(np.mean(selected))


def aggregate_program_openml_full_context(
    *,
    request: ProgramOpenMLFullContextRequest,
    receipt_paths: tuple[Path, ...],
    output: Path,
) -> ProgramOpenMLPanelReceipt:
    """Bind six independently completed dataset receipts into one comparison panel."""

    if output.exists():
        raise ValueError(f"refusing to overwrite OpenML receipt: {output}")
    receipts = tuple(load_program_openml_dataset_receipt(path) for path in receipt_paths)
    by_id = {receipt.dataset_id: receipt for receipt in receipts}
    if len(by_id) != len(receipts) or tuple(by_id) != _DATASET_IDS:
        raise ValueError(f"aggregate requires ordered unique receipts for {_DATASET_IDS!r}")
    for receipt in receipts:
        if receipt.request_hash != request.request_hash:
            raise ValueError(f"dataset {receipt.dataset_id} uses a different request")
    source_identities = {
        (
            receipt.execution.evaluation_source_revision,
            receipt.execution.evaluation_source_archive_sha256,
            receipt.execution.source_tree_sha256,
            receipt.data_panel_hash,
        )
        for receipt in receipts
    }
    if len(source_identities) != 1:
        raise ValueError("dataset receipts do not share one source and data identity")

    dataset_summaries: dict[str, JsonValue] = {}
    for dataset_id, receipt in by_id.items():
        primary = receipt.primary_metric
        linear_mean = _mean_primary(
            tuple(item.metrics for item in receipt.linear_baseline), primary
        )
        arms: dict[str, JsonValue] = {}
        for arm in receipt.arms:
            model_mean = _mean_primary(tuple(item.metrics for item in arm.splits), primary)
            arms[arm.arm_id] = {
                "dataset_mean_primary": model_mean,
                "delta_vs_linear": model_mean - linear_mean,
                "beats_linear": model_mean < linear_mean,
            }
        dataset_summaries[dataset_id] = {
            "task": receipt.task,
            "primary_metric": primary,
            "linear_dataset_mean_primary": linear_mean,
            "arms": arms,
        }

    task_macros: dict[str, JsonValue] = {}
    for task, primary in _TASK_PRIMARY.items():
        selected = [
            dataset_summaries[dataset_id]
            for dataset_id in _DATASET_IDS
            if dataset_summaries[dataset_id]["task"] == task
        ]
        linear_macro = float(
            np.mean([float(item["linear_dataset_mean_primary"]) for item in selected])
        )
        arms: dict[str, JsonValue] = {}
        for arm in request.arms:
            model_macro = float(
                np.mean(
                    [
                        float(item["arms"][arm.arm_id]["dataset_mean_primary"])
                        for item in selected
                    ]
                )
            )
            arms[arm.arm_id] = {
                "dataset_macro_mean_primary": model_macro,
                "delta_vs_linear": model_macro - linear_macro,
                "beats_linear": model_macro < linear_macro,
                "dataset_wins": sum(
                    bool(item["arms"][arm.arm_id]["beats_linear"]) for item in selected
                ),
                "dataset_count": len(selected),
            }
        task_macros[task] = {
            "primary_metric": primary,
            "dataset_count": len(selected),
            "linear_dataset_macro_mean_primary": linear_macro,
            "arms": arms,
        }
    panel_success = {
        arm.arm_id: all(
            bool(task_macros[task]["arms"][arm.arm_id]["beats_linear"])
            for task in _TASK_PRIMARY
        )
        for arm in request.arms
    }
    payload: dict[str, Any] = {
        "schema_version": "tabu.program-openml-full-context-panel-receipt.v1",
        "evidence_status": EvidenceStatus.LOCAL_UNISSUED,
        "claim_boundary": request.claim_boundary,
        "request": request,
        "request_hash": request.request_hash,
        "data_panel_hash": receipts[0].data_panel_hash,
        "dataset_receipt_hashes": {
            dataset_id: by_id[dataset_id].receipt_hash for dataset_id in _DATASET_IDS
        },
        "datasets": dataset_summaries,
        "task_macros": task_macros,
        "arm_panel_success": panel_success,
    }
    panel = ProgramOpenMLPanelReceipt(**payload, receipt_hash=canonical_hash(payload))
    _write_receipt(output, panel)
    return panel


__all__ = [
    "OpenMLDataPanel",
    "ProgramOpenMLDatasetReceipt",
    "ProgramOpenMLFullContextRequest",
    "ProgramOpenMLPanelReceipt",
    "aggregate_program_openml_full_context",
    "load_openml_data_panel",
    "load_program_openml_dataset_receipt",
    "load_program_openml_full_context_request",
    "run_program_openml_full_context_dataset",
]
