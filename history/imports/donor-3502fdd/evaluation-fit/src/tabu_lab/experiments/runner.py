"""Executable fit-first runner for preregistered dense-reference experiments."""

from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import yaml

from tabu_lab.contracts import (
    FeatureKind,
    FeatureRole,
    OriginState,
    PredictionBundle,
    TruthSidecar,
    canonical_hash,
    canonical_json,
    origin_mask,
)
from tabu_lab.evaluation import (
    Evaluator,
    FitAttemptArtifacts,
    assert_public_artifact_tree_safe,
    capture_environment,
    verify_fit_attempt_artifacts,
    write_fit_attempt_artifacts,
)
from tabu_lab.evaluation.fit_artifacts import assert_public_payload_safe
from tabu_lab.evidence import (
    ArtifactRef,
    EnvironmentDisclosure,
    Receipt,
    ReceiptStatus,
    RunBundle,
    RunIdentity,
    read_receipt,
    write_receipt,
)
from tabu_lab.evidence.formal_authorization import (
    FormalAuthorizationContext,
    FormalAuthorizationError,
    FormalAuthorizationSummary,
    VerifiedFormalAuthorization,
    verify_formal_authorization,
)
from tabu_lab.evidence.source_identity import (
    SourceIdentity,
    distribution_source_identity,
    git_source_identity,
)
from tabu_lab.models import ModelVariantRef, ReferenceConfig, build_model
from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE, DEFAULT_FLOAT_DTYPE_NAME
from tabu_lab.observers import get_observer
from tabu_lab.registry import get_model_spec
from tabu_lab.training import Objective, Trainer

from .contracts import (
    FitEvaluationBundle,
    FitEvidenceMode,
    FitExperimentSpec,
    FitFamilyMetrics,
    FitMetricKind,
    FitStage,
    FitTargetFamily,
    derive_attempt_id,
)
from .feasibility import (
    CategoricalNWTarget,
    FeasibilityReportStatus,
    FitFeasibilityReport,
    NumericNWTarget,
    NWFeasibilityTarget,
    NWSupportArm,
    assess_nw_targets,
)
from .fixture_registry import (
    build_registered_f0_fixture_for_dataset,
    f0_generator_source_hash,
    f0_generator_source_uri,
    fixture_version_from_adapter,
)
from .fixtures import F0Fixture
from .splits import GraphSplitManifest, InteractionSplitManifest, RowSplitManifest

if TYPE_CHECKING:
    from .r1_runner import R1RunReceipt


class FitExperimentError(RuntimeError):
    """A preregistration or execution boundary was violated."""


@dataclass(frozen=True, slots=True)
class SeedRunResult:
    model_seed: int
    verdict: str
    fit_evaluation: FitEvaluationBundle | None
    artifacts: FitAttemptArtifacts
    failure_phase: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ExperimentAggregateArtifacts:
    directory: Path
    summary: Path
    checksums: Path
    aggregate_hash: str
    verdict: str


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    experiment_id: str
    stage: FitStage
    seed_results: tuple[SeedRunResult, ...]
    aggregate: ExperimentAggregateArtifacts

    @property
    def passed(self) -> bool:
        return self.aggregate.verdict == "pass"

    @property
    def succeeded(self) -> bool:
        return self.aggregate.verdict in {"pass", "diagnostic_pass"}


def _repository_source_root(
    repository: str | os.PathLike[str] | None,
) -> Path | None:
    if repository is not None:
        return Path(repository).resolve()
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "src" / "tabu_lab").is_dir():
        return candidate
    return None


def _assert_formal_output_root_safe(
    output_root: str | os.PathLike[str],
    *,
    repository: str | os.PathLike[str] | None,
) -> None:
    """Require in-repository formal staging paths to be ignored by Git."""

    source_root = _repository_source_root(repository)
    if source_root is None:
        return
    try:
        result = subprocess.run(
            ("git", "-C", str(source_root), "rev-parse", "--show-toplevel"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise FitExperimentError(
            "cannot verify the formal output_root Git-ignore boundary"
        ) from exc
    if result.returncode != 0:
        return
    git_root = Path(result.stdout.strip()).resolve()
    output = Path(output_root).resolve()
    try:
        relative = output.relative_to(git_root)
    except ValueError:
        return
    ignored = subprocess.run(
        (
            "git",
            "-C",
            str(git_root),
            "check-ignore",
            "--quiet",
            "--",
            relative.as_posix(),
        ),
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise FitExperimentError(
            "formal output_root inside the Git repository must be Git ignored; "
            "use .local-runs/formal-staging"
        )


def _resolve_formal_authorization(
    authorization_catalog: str | os.PathLike[str],
    *,
    spec: FitExperimentSpec,
    preregistration_path: Path,
    preregistration_text: str,
    repository: str | os.PathLike[str] | None,
) -> tuple[FormalAuthorizationContext, VerifiedFormalAuthorization]:
    """Resolve a grant only by replaying one clean canonical repository."""

    # ``repository`` is the live executable-source scope.  The authorization
    # checkout may intentionally be a later reviewed commit (or a separate
    # clean checkout) so that it can contain the immutable catalog and review
    # objects without creating a Git self-reference with SourceIdentity.
    del preregistration_path, repository
    catalog_path = Path(authorization_catalog).resolve()
    context = FormalAuthorizationContext(
        repository=catalog_path.parent,
        catalog=catalog_path,
        experiment_id=spec.experiment_id,
    )
    try:
        verified = verify_formal_authorization(
            context,
            preregistration_text=preregistration_text,
        )
    except (FormalAuthorizationError, TypeError, ValueError) as exc:
        raise FitExperimentError("formal authorization replay failed") from exc
    return context, verified


def _authorization_safe_command(
    command: Sequence[str],
    authorization: FormalAuthorizationSummary | None,
    *,
    output_root: str | os.PathLike[str] | None = None,
    preregistration_path: Path | None = None,
) -> tuple[str, ...]:
    """Replace formal host paths with public, content-addressed command values."""

    if authorization is None:
        return tuple(command)
    sanitized: list[str] = []
    private_preregistration_tokens: set[str] = set()
    if preregistration_path is not None and preregistration_path.is_absolute():
        private_preregistration_tokens = {
            str(preregistration_path),
            str(preregistration_path.resolve()),
        }
    index = 0
    while index < len(command):
        token = command[index]
        if token == "--authorization-catalog":
            index += 2
            continue
        if token.startswith("--authorization-catalog="):
            index += 1
            continue
        if output_root is not None and token == "--output-root":
            sanitized.extend(("--output-root", "formal-staging://output"))
            index += 2
            continue
        if output_root is not None and token.startswith("--output-root="):
            sanitized.append("--output-root=formal-staging://output")
            index += 1
            continue
        if token in private_preregistration_tokens:
            sanitized.append(f"preregistration://sha256/{authorization.preregistration_sha256}")
            index += 1
            continue
        sanitized.append(token)
        index += 1
    sanitized.extend(("--authorization-catalog", f"sha256:{authorization.catalog_hash}"))
    return tuple(sanitized)


def load_fit_experiment(path: str | os.PathLike[str]) -> FitExperimentSpec:
    """Load one strict preregistration and cross-check the live registry."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FitExperimentError(f"cannot load fit experiment: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FitExperimentError("fit experiment YAML must contain one mapping")
    try:
        spec = FitExperimentSpec.model_validate(payload)
    except ValueError as exc:
        raise FitExperimentError(f"invalid fit experiment: {exc}") from exc
    live = get_model_spec(spec.contract_id)
    if live.contract_version != spec.contract_version:
        raise FitExperimentError("preregistered contract_version does not match the live registry")
    if canonical_hash(live) != spec.model_spec_hash:
        raise FitExperimentError("preregistered ModelSpec hash does not match the registry")
    if spec.stage is FitStage.F0:
        validate_f0_binding(spec)
    elif spec.stage is FitStage.S1:
        from .s1_preregistration import validate_s1_binding

        try:
            validate_s1_binding(spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise FitExperimentError(f"invalid registered S1 binding: {exc}") from exc
    elif spec.stage is FitStage.R1:
        from .r1_runner import validate_r1_binding

        try:
            validate_r1_binding(spec)
        except (KeyError, TypeError, ValueError) as exc:
            raise FitExperimentError(f"invalid registered R1 binding: {exc}") from exc
    return spec


def source_tree_manifest(
    repository: str | os.PathLike[str] | None = None,
    *,
    preregistration: str | os.PathLike[str] | None = None,
    request_formal: bool = False,
    reviewed: bool = False,
    source_identity: SourceIdentity | None = None,
    distribution_artifact: bytes | str | os.PathLike[str] | None = None,
    distribution_lock: bytes | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Capture source files and independently resolve the issuance boundary.

    A supplied ``source_identity`` is an expected public binding, never authority.
    Repository identities are reconstructed from the live Git state. Installed-package
    identities are reconstructed from bytes read from ``distribution_artifact`` and
    ``distribution_lock``; their local paths are never retained.
    """

    if repository is not None:
        root = Path(repository).resolve()
        mode = "repository"
    else:
        candidate = Path(__file__).resolve().parents[3]
        if (candidate / "src" / "tabu_lab").is_dir():
            root = candidate
            mode = "repository"
        else:
            root = Path(__file__).resolve().parents[1]
            mode = "installed_package"
    candidates: list[Path] = []
    if mode == "repository":
        for relative in ("src/tabu_lab", "specs/models", "schemas"):
            directory = root / relative
            if directory.is_dir():
                candidates.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix not in {".pyc", ".pyo"}
                )
        candidates.extend(
            path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file()
        )
    else:
        candidates.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    if not candidates:
        raise FitExperimentError(f"no executable source files found under {root}")
    files = tuple(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
        for path in sorted(set(candidates))
    )
    preimage = {
        "schema_version": "tabu.source-tree-preimage.v1",
        "mode": mode,
        "root_label": "repository" if mode == "repository" else "tabu_lab_package",
        "files": files,
    }
    tree_hash = canonical_hash(preimage)
    lock_path = root / "uv.lock" if mode == "repository" else None
    lock_hash = (
        hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock_path is not None and lock_path.is_file()
        else None
    )
    if mode == "repository":
        resolved_identity = git_source_identity(
            root,
            preregistration=preregistration,
            source_tree_hash=tree_hash,
            lock_hash=lock_hash,
            request_formal=request_formal,
            reviewed=reviewed,
            source_files=files,
        )
        if (
            source_identity is not None
            and source_identity.issuance_status == "formal"
            and resolved_identity.issuance_status == "formal"
            and source_identity != resolved_identity
        ):
            resolved_identity = _local_source_identity(
                resolved_identity,
                "provided_source_identity_mismatch",
            )
    else:
        if (
            source_identity is None
            or source_identity.source_kind != "distribution"
            or source_identity.distribution_uri is None
            or source_identity.distribution_sha256 is None
            or source_identity.lock_hash is None
        ):
            resolved_identity = SourceIdentity(
                source_kind="local",
                issuance_status="local_unissued",
                source_tree_hash=tree_hash,
                reasons=("installed_distribution_identity_not_provided",),
            )
        else:
            resolved_identity = distribution_source_identity(
                uri=source_identity.distribution_uri,
                sha256=source_identity.distribution_sha256,
                lock_hash=source_identity.lock_hash,
                reviewed=reviewed,
                retrieved_distribution=distribution_artifact,
                retrieved_lock=distribution_lock,
                source_tree_hash=tree_hash,
                live_source_root=root,
            )
            if not request_formal and resolved_identity.issuance_status == "formal":
                resolved_identity = _local_source_identity(
                    resolved_identity,
                    "formal_issuance_not_requested",
                )
    return {
        "schema_version": "tabu.source-tree.v3",
        "mode": preimage["mode"],
        "root_label": preimage["root_label"],
        "files": preimage["files"],
        "source_identity": resolved_identity.model_dump(mode="json"),
    }


def _local_source_identity(identity: SourceIdentity, reason: str) -> SourceIdentity:
    """Downgrade a resolved identity while preserving only public bindings."""

    payload = identity.model_dump(mode="python")
    payload.update(
        issuance_status="local_unissued",
        reviewed=False,
        reasons=tuple(dict.fromkeys((*identity.reasons, reason))),
    )
    return SourceIdentity.model_validate(payload)


def source_tree_hash(repository: str | os.PathLike[str] | None = None) -> str:
    """Hash the executable source preimage independent of Git SHA format."""

    return canonical_hash(source_tree_manifest(repository))


class _NonfiniteFitError(FitExperimentError):
    """A frozen nonfinite kill condition was reached."""


def _write_canonical_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_artifact_ref(root: Path, path: Path) -> ArtifactRef:
    relative = path.relative_to(root).as_posix()
    return ArtifactRef(
        artifact_id=relative.replace("/", "--"),
        kind=("runtime_failure" if path.name == "failure.json" else "fit_failure_artifact"),
        uri=relative,
        sha256=_file_sha256(path),
        size_bytes=path.stat().st_size,
        media_type=(
            "application/json"
            if path.suffix == ".json"
            else "application/yaml"
            if path.suffix in {".yaml", ".yml"}
            else "text/markdown"
            if path.suffix == ".md"
            else "application/octet-stream"
        ),
    )


def _publish_directory_create_once(staging: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an earlier attempt."""

    lock = destination.parent / f".{destination.name}.publish.lock"
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise FileExistsError(
                f"another publisher owns the immutable path: {destination}"
            ) from exc
        if os.path.lexists(destination):
            raise FileExistsError(f"immutable output already exists: {destination}")
        os.rename(staging, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                lock.unlink()


def _exception_boundary(
    phase: str,
    error: Exception,
    *,
    formal: bool = False,
) -> dict[str, Any]:
    text = str(error).strip() or type(error).__name__
    is_oom = isinstance(error, (MemoryError, torch.OutOfMemoryError)) or (
        "out of memory" in text.lower()
    )
    return {
        "schema": "tabu.fit-runtime-failure.v1",
        "phase": phase,
        "code": (
            "nonfinite"
            if isinstance(error, _NonfiniteFitError)
            else "out_of_memory"
            if is_oom
            else "execution_error"
        ),
        "exception_type": type(error).__name__,
        "message": (
            "formal runtime failure details withheld; use typed phase and code" if formal else text
        ),
        "retryable": bool(is_oom),
    }


def _typed_fit_members(
    spec: FitExperimentSpec,
) -> tuple[str | tuple[str, str | None], ...]:
    """Return the ordered fit-partition members named by a typed split."""

    partition = spec.split.partition(spec.split.fit_partition)
    if isinstance(spec.split, RowSplitManifest):
        return tuple(partition.row_ids)
    if isinstance(spec.split, GraphSplitManifest):
        return tuple((item.graph_id, item.node_id) for item in partition.elements)
    if isinstance(spec.split, InteractionSplitManifest):
        return tuple((item.user_id, item.item_id) for item in partition.interactions)
    raise FitExperimentError("unsupported typed split manifest")  # pragma: no cover


def _expected_typed_fit_members(
    spec: FitExperimentSpec,
    fixture: F0Fixture,
) -> tuple[str | tuple[str, str | None], ...]:
    if isinstance(spec.split, RowSplitManifest):
        return tuple(fixture.fit_view.row_ids)
    if isinstance(spec.split, GraphSplitManifest):
        return tuple((fixture.dataset.dataset_id, row_id) for row_id in fixture.fit_view.row_ids)
    if isinstance(spec.split, InteractionSplitManifest):
        observed = origin_mask(fixture.dataset.origin_states, OriginState.OBSERVED)
        return tuple(
            (
                fixture.dataset.row_ids[row],
                fixture.dataset.feature_specs[feature].name,
            )
            for row, feature in observed.nonzero(as_tuple=False).tolist()
        )
    raise FitExperimentError("unsupported typed split manifest")  # pragma: no cover


def _require_float32_tensor(name: str, value: torch.Tensor) -> None:
    if value.dtype is not DEFAULT_FLOAT_DTYPE:
        raise FitExperimentError(f"{name} must be canonical float32, observed {value.dtype}")


def validate_f0_binding(
    spec: FitExperimentSpec,
    fixture: F0Fixture | None = None,
) -> F0Fixture:
    """Fail closed unless a typed F0 preregistration names the executed fixture."""

    if spec.stage is not FitStage.F0:
        raise FitExperimentError("F0 binding validation requires an F0 experiment")
    if fixture is None:
        try:
            bound = build_registered_f0_fixture_for_dataset(
                spec.contract_id,
                spec.dataset.dataset_id,
            )
        except KeyError as exc:
            raise FitExperimentError(str(exc)) from exc
    else:
        bound = fixture
    if bound.contract_id != spec.contract_id:
        raise FitExperimentError("F0 fixture contract_id does not match preregistration")
    try:
        fixture_version = fixture_version_from_adapter(spec.dataset.adapter.adapter_version)
    except ValueError as exc:
        raise FitExperimentError(str(exc)) from exc
    if spec.dataset.source_uri != f0_generator_source_uri(fixture_version=fixture_version):
        raise FitExperimentError("F0 dataset source URI is not the canonical generator")
    if spec.dataset.source_sha256 != f0_generator_source_hash(fixture_version=fixture_version):
        raise FitExperimentError("F0 generator source hash does not match this build")
    if spec.dataset.dataset_id != bound.dataset.dataset_id:
        raise FitExperimentError("preregistered dataset_id does not match the F0 fixture")
    if spec.dataset.dataset_hash != bound.dataset.dataset_hash:
        raise FitExperimentError("preregistered dataset hash does not match the F0 fixture")
    if spec.episode_schedule.recipe_hashes != (bound.recipe.recipe_hash,):
        raise FitExperimentError("preregistered recipe hash does not match compiled F0 truth")
    if spec.episode_schedule.content_hash != bound.episode_schedule.content_hash:
        raise FitExperimentError("preregistered episode schedule does not match the F0 fixture")
    if spec.episode_schedule.targets_per_episode != int(bound.truth.target_mask.sum()):
        raise FitExperimentError("preregistered target count does not match compiled F0 truth")
    if set(spec.target_families) != {family for family in bound.episode_schedule.target_families}:
        raise FitExperimentError("preregistered target families do not match F0 fixture")

    if len(spec.split.partitions) != 1:
        raise FitExperimentError("F0 typed split must contain exactly its fit partition")
    if spec.split.partitions[0].name != spec.split.fit_partition:
        raise FitExperimentError("F0 typed split partition must be the fit partition")
    if _typed_fit_members(spec) != _expected_typed_fit_members(spec, bound):
        kind = spec.split.kind.value
        raise FitExperimentError(
            f"typed {kind} fit partition does not bind ordered compiler members"
        )

    _require_float32_tensor("F0 evidence", bound.evidence.forward_values)
    _require_float32_tensor("F0 truth sidecar", bound.truth.target_values)
    provenance = bound.compilation.provenance
    if provenance.dataset_hash != spec.dataset.dataset_hash:
        raise FitExperimentError("compiler provenance does not bind the dataset hash")
    if provenance.split_manifest_hash != bound.split_manifest.manifest_hash:
        raise FitExperimentError("compiler provenance does not bind its SplitManifest")
    if provenance.source_view_hash != bound.source_view.view_hash:
        raise FitExperimentError("compiler provenance does not bind the source SplitView")
    if provenance.fit_view_hash != bound.fit_view.view_hash:
        raise FitExperimentError("compiler provenance does not bind the fit SplitView")
    if provenance.recipe_hash != bound.recipe.recipe_hash:
        raise FitExperimentError("compiler provenance does not bind the recipe hash")
    normalizer = bound.numeric_normalizer
    statistics = normalizer.statistics
    if statistics.fit_view_hash != bound.fit_view.view_hash:
        raise FitExperimentError("numeric normalizer does not bind the fit SplitView")
    expected_normalizer_config_hash = canonical_hash(
        {
            "kind": "numeric_normalizer",
            "epsilon": float(normalizer.epsilon),
            "shared_numeric_groups": normalizer.shared_numeric_groups,
        }
    )
    if statistics.config_hash != expected_normalizer_config_hash:
        raise FitExperimentError("numeric normalizer config hash does not match its preimage")
    if provenance.numeric_normalizer_hash != normalizer.artifact_hash:
        raise FitExperimentError("compiler provenance does not bind the numeric normalizer")
    expected_topology_hash = (
        None
        if bound.evidence.graph_topology is None
        else bound.evidence.graph_topology.topology_hash
    )
    if provenance.graph_topology_hash != expected_topology_hash:
        raise FitExperimentError("compiler provenance does not bind graph topology")
    if bound.evidence.source_partition != bound.source_view.partition:
        raise FitExperimentError("compiled evidence does not bind the source partition")
    if bound.evidence.fit_partition != bound.fit_view.partition:
        raise FitExperimentError("compiled evidence does not bind the fit partition")
    if bound.truth.recipe_hash != bound.recipe.recipe_hash:
        raise FitExperimentError("compiled truth does not bind the recipe hash")
    return bound


def _numeric_normalizer_manifest(fixture: F0Fixture) -> dict[str, Any]:
    normalizer = fixture.numeric_normalizer
    statistics = normalizer.statistics
    statistics_preimage = {
        "schema": "tabu.fitted-statistics.v2",
        "fit_view_hash": statistics.fit_view_hash,
        "split_definition_hash": statistics.split_definition_hash,
        "config_hash": statistics.config_hash,
        "fit_value_mask_hash": statistics.fit_value_mask_hash,
        "feature_names": statistics.feature_names,
        "feature_kinds": statistics.feature_kinds,
        "counts": statistics.counts,
        "means": statistics.means,
        "scales": statistics.scales,
    }
    artifact_hash = canonical_hash(statistics_preimage)
    if artifact_hash != statistics.artifact_hash:
        raise FitExperimentError("numeric normalizer artifact hash does not match statistics")
    return {
        "schema": "tabu.numeric-normalizer-binding.v1",
        **{name: value for name, value in statistics_preimage.items() if name != "schema"},
        "epsilon": float(normalizer.epsilon),
        "shared_numeric_groups": normalizer.shared_numeric_groups,
        "artifact_hash": artifact_hash,
    }


def compiler_binding_manifest(
    spec: FitExperimentSpec,
    fixture: F0Fixture,
) -> dict[str, Any]:
    """Bind typed split semantics to the row-carrier compiler provenance."""

    validate_f0_binding(spec, fixture)
    provenance = fixture.compilation.provenance
    provenance_manifest = {
        "dataset_hash": provenance.dataset_hash,
        "split_manifest_hash": provenance.split_manifest_hash,
        "source_view_hash": provenance.source_view_hash,
        "fit_view_hash": provenance.fit_view_hash,
        "recipe_hash": provenance.recipe_hash,
        "graph_topology_hash": provenance.graph_topology_hash,
        "numeric_normalizer_hash": provenance.numeric_normalizer_hash,
    }
    provenance_hash = canonical_hash(
        {"schema": "tabu.compilation-provenance.v2", **provenance_manifest}
    )
    if provenance_hash != provenance.provenance_hash:
        raise FitExperimentError("compiler provenance hash does not match its preimage")
    return {
        "schema": "tabu.fit-compiler-binding.v1",
        "typed_split_hash": spec.split.content_hash,
        "typed_split_kind": spec.split.kind.value,
        "fit_partition": spec.split.fit_partition,
        "compiler_provenance": provenance_manifest,
        "compiler_provenance_hash": provenance_hash,
        "numeric_normalizer": _numeric_normalizer_manifest(fixture),
        "projection": (
            "observed_interactions_to_full_matrix_row_carrier"
            if isinstance(spec.split, InteractionSplitManifest)
            else "nodes_to_graph_row_carrier"
            if isinstance(spec.split, GraphSplitManifest)
            else "rows_to_tabular_row_carrier"
        ),
    }


def _device(spec: FitExperimentSpec) -> torch.device:
    if spec.execution.device.value == "cuda":
        if not torch.cuda.is_available():
            raise FitExperimentError("CUDA execution was preregistered but CUDA is unavailable")
        assert spec.execution.device_index is not None
        return torch.device("cuda", spec.execution.device_index)
    if spec.execution.device.value == "mps":
        if not torch.backends.mps.is_available():
            raise FitExperimentError("MPS execution was preregistered but MPS is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def _reference_config(spec: FitExperimentSpec) -> ReferenceConfig:
    values = spec.semantic.reference.model_dump(mode="python")
    values.pop("backend")
    values["block_kind"] = spec.semantic.dynamics.block_kind
    return ReferenceConfig(**values)


def _tabubase_identity_metadata(
    spec: FitExperimentSpec, *, model: Any | None = None
) -> dict[str, Any]:
    """Return the explicit Base identity envelope copied into every receipt."""

    if spec.contract_id != "tabu.cell.base":
        return {}
    config = _reference_config(spec)
    payload = {
        key: getattr(value, "value", value)
        for key, value in config.__dict__.items()
    }
    variant_ref = getattr(model, "variant_ref", None)
    if variant_ref is None:
        variant_ref = ModelVariantRef(
            contract_id=spec.contract_id,
            contract_version=spec.contract_version,
            profile_id=spec.semantic.profile_id or "",
            model_spec_hash=spec.model_spec_hash,
            source_identity=spec.dataset.source_sha256,
            semantic_config_hash=spec.semantic.content_hash,
        )
    return {
        "profile_id": spec.semantic.profile_id,
        "tokenizer_version": "cell-tokenizer.v1",
        "label_broadcast": spec.semantic.profile_id == "supervised.label_broadcast.v1",
        "label_broadcast_tau": 1.0e-6,
        "reference_config": payload,
        "terminal": spec.semantic.numeric_terminal.value,
        "ll_ridge": getattr(getattr(model, "readout", None), "ll_ridge", None),
        "bandwidth": config.routing_bandwidth,
        "variant_ref": (
            variant_ref.as_dict() if hasattr(variant_ref, "as_dict") else variant_ref
        ),
        "variant_hash": (
            variant_ref.semantic_hash if hasattr(variant_ref, "semantic_hash") else None
        ),
    }


def _seed_everything(seed: int, *, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def _build_model(spec: FitExperimentSpec, *, seed: int, device: torch.device):
    if spec.execution.dtype != DEFAULT_FLOAT_DTYPE_NAME:  # defensive: schema freezes it
        raise FitExperimentError("fit-first execution supports only float32")
    _seed_everything(seed, device=device)
    kwargs: dict[str, Any] = {"config": _reference_config(spec)}
    # Numeric kernel choice is an independent semantic axis shared by every
    # executable readout.  The default remains NW for legacy contracts.
    kwargs["numeric_terminal"] = spec.semantic.numeric_terminal.value
    if spec.contract_id in {"tabuf", "tabul", "tabufl", "tabu4rec"}:
        geometry = spec.semantic.augmented_readout_geometry
        if geometry is None:  # defensive: FitExperimentSpec closes this field
            raise FitExperimentError(
                f"{spec.contract_id} requires an explicit augmented readout geometry"
            )
        kwargs["readout_geometry"] = geometry.value
    if spec.contract_id in {"tabul", "tabufl"}:
        kwargs["label_columns"] = spec.semantic.label_columns
        label_plan = spec.semantic.label_address_plan
        if label_plan is None:  # defensive: FitExperimentSpec closes this field
            raise FitExperimentError(f"{spec.contract_id} requires a label address plan")
        kwargs["label_address_plan"] = label_plan.value
    if spec.contract_id == "tabu.cell.base":
        profile = spec.semantic.profile_id
        if profile is None:
            raise FitExperimentError("tabu.cell.base requires an explicit profile_id")
        kwargs["profile"] = profile
        if spec.semantic.label_columns:
            kwargs["label_columns"] = spec.semantic.label_columns
    if spec.contract_id == "tabu4graph":
        kwargs["target_feature"] = spec.semantic.target_feature
        plan = spec.semantic.graph_unit_receiver_plan
        if plan is None:  # defensive: FitExperimentSpec closes this field
            raise FitExperimentError("tabu4graph requires a graph Unit receiver plan")
        kwargs["unit_receiver_plan"] = plan.value
    if spec.contract_id == "tabu4rec":
        rec_plan = spec.semantic.recommendation_address_plan
        if rec_plan is None:  # defensive: FitExperimentSpec closes this field
            raise FitExperimentError("tabu4rec requires a recommendation address plan")
        kwargs["recommendation_address_plan"] = rec_plan.value
        if spec.semantic.rec_axis_summary_dim is not None:
            kwargs["rec_axis_summary_dim"] = spec.semantic.rec_axis_summary_dim
        if spec.semantic.rec_matched_residual_scale is not None:
            kwargs["rec_matched_residual_scale"] = spec.semantic.rec_matched_residual_scale
    model = build_model(spec.contract_id, **kwargs)
    if spec.contract_id == "tabu.cell.base":
        model.variant_ref = ModelVariantRef(
            contract_id=spec.contract_id,
            contract_version=spec.contract_version,
            profile_id=spec.semantic.profile_id or "",
            model_spec_hash=spec.model_spec_hash,
            source_identity=spec.dataset.source_sha256,
            semantic_config_hash=spec.semantic.content_hash,
        )
    model.semantic_config_hash = spec.semantic.content_hash
    model = model.to(device=device, dtype=DEFAULT_FLOAT_DTYPE)
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if value.is_floating_point() and value.dtype is not DEFAULT_FLOAT_DTYPE:
            raise FitExperimentError(
                f"model tensor {name!r} did not resolve to float32: {value.dtype}"
            )
    return model


def _response_columns(fixture: F0Fixture) -> tuple[int, ...]:
    return tuple(
        index
        for index, feature in enumerate(fixture.evidence.feature_specs)
        if feature.role is FeatureRole.RESPONSE
    )


def fixture_nw_targets(
    fixture: F0Fixture,
    *,
    max_categorical_nll: float,
) -> tuple[NWFeasibilityTarget, ...]:
    """Bind one compiled F0 fixture to the exact frozen readout support geometry."""

    evidence = fixture.evidence
    truth = fixture.truth
    source = evidence.source_mask.clone()
    query_rows = origin_mask(evidence.origin_states, OriginState.QUERY).any(dim=1)
    if fixture.contract_id in {"tabul", "tabufl"}:
        source[query_rows] = False
    response_columns = _response_columns(fixture)
    _, n_features = evidence.forward_values.shape
    targets: list[NWFeasibilityTarget] = []
    for row, feature in truth.target_mask.nonzero(as_tuple=False).tolist():
        target_id = f"r{row}:c{feature}"
        family = (
            FitTargetFamily.LABEL
            if bool(origin_mask(evidence.origin_states, OriginState.QUERY)[row, feature])
            else FitTargetFamily.COMPLETION
        )
        arms: tuple[NWSupportArm, ...]
        if fixture.contract_id == "tabu4rec":
            user_rows = source[:, feature].clone()
            user_rows[row] = False
            user_ids = tuple(
                source_row * n_features + feature
                for source_row in user_rows.nonzero(as_tuple=False).flatten().tolist()
            )
            item_columns = source[row].clone()
            allowed = torch.zeros(n_features, dtype=torch.bool)
            allowed[list(response_columns)] = True
            item_columns &= allowed
            item_columns[feature] = False
            item_ids = tuple(
                row * n_features + source_feature
                for source_feature in item_columns.nonzero(as_tuple=False).flatten().tolist()
            )
            arms = (
                NWSupportArm(
                    arm_id="user",
                    support_ids=user_ids,
                    support_values=tuple(
                        float(evidence.forward_values[index // n_features, feature])
                        for index in user_ids
                    ),
                    arm_weight=1.0,
                ),
                NWSupportArm(
                    arm_id="item",
                    support_ids=item_ids,
                    support_values=tuple(
                        float(evidence.forward_values[row, index % n_features])
                        for index in item_ids
                    ),
                    arm_weight=1.0,
                ),
            )
        else:
            support_rows = source[:, feature].clone()
            support_rows[row] = False
            support_ids = tuple(
                source_row * n_features + feature
                for source_row in support_rows.nonzero(as_tuple=False).flatten().tolist()
            )
            arms = (
                NWSupportArm(
                    arm_id="same_column",
                    support_ids=support_ids,
                    support_values=tuple(
                        float(evidence.forward_values[index // n_features, feature])
                        for index in support_ids
                    ),
                ),
            )
        feature_spec = evidence.feature_specs[feature]
        truth_value = float(truth.target_values[row, feature])
        if feature_spec.kind is FeatureKind.NUMERIC:
            targets.append(
                NumericNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_value=truth_value,
                    arms=arms,
                )
            )
        else:
            targets.append(
                CategoricalNWTarget(
                    target_id=target_id,
                    family=family,
                    truth_code=round(truth_value),
                    arms=arms,
                    max_nll=max_categorical_nll,
                )
            )
    return tuple(targets)


def assess_fixture_feasibility(
    fixture: F0Fixture,
    spec: FitExperimentSpec,
) -> tuple[tuple[NWFeasibilityTarget, ...], FitFeasibilityReport]:
    max_nll = spec.pass_gate.max_categorical_nll
    targets = fixture_nw_targets(
        fixture,
        max_categorical_nll=float("inf") if max_nll is None else max_nll,
    )
    return targets, assess_nw_targets(
        targets,
        report_id=f"{spec.experiment_id}-{fixture.fixture_id}",
    )


def _normalized_arm_weights(target: NWFeasibilityTarget) -> dict[str, float]:
    active = tuple(arm for arm in target.arms if arm.active)
    total = sum(arm.arm_weight for arm in active)
    return {arm.arm_id: arm.arm_weight / total for arm in active} if total else {}


def trivial_baseline(targets: Sequence[NWFeasibilityTarget]) -> dict[str, Any]:
    """Mean/mode baseline computed from exactly the same declared supports."""

    numeric_errors: dict[str, list[float]] = defaultdict(list)
    categorical_nll: dict[str, list[float]] = defaultdict(list)
    categorical_correct: dict[str, list[float]] = defaultdict(list)
    for target in targets:
        weights = _normalized_arm_weights(target)
        key = target.family.value
        if isinstance(target, NumericNWTarget):
            prediction = sum(
                weights[arm.arm_id] * (sum(arm.support_values) / len(arm.support_values))
                for arm in target.arms
                if arm.active
            )
            numeric_errors[key].append((prediction - target.truth_value) ** 2)
            continue
        probabilities: dict[int, float] = defaultdict(float)
        for arm in target.arms:
            if not arm.active:
                continue
            mass = weights[arm.arm_id] / len(arm.support_values)
            for value in arm.support_values:
                probabilities[int(value)] += mass
        target_probability = probabilities.get(target.truth_code, 0.0)
        categorical_nll[key].append(-math.log(max(target_probability, 1.0e-8)))
        predicted = max(sorted(probabilities), key=probabilities.get)
        categorical_correct[key].append(float(predicted == target.truth_code))
    families: dict[str, Any] = {}
    active_family_losses: list[float] = []
    for family in sorted(set(numeric_errors) | set(categorical_nll)):
        typed: list[float] = []
        payload: dict[str, Any] = {}
        if numeric_errors[family]:
            mse = sum(numeric_errors[family]) / len(numeric_errors[family])
            payload["numeric_mse"] = mse
            typed.append(mse)
        if categorical_nll[family]:
            nll = sum(categorical_nll[family]) / len(categorical_nll[family])
            payload["categorical_nll"] = nll
            payload["categorical_accuracy"] = sum(categorical_correct[family]) / len(
                categorical_correct[family]
            )
            typed.append(nll)
        payload["loss"] = sum(typed) / len(typed)
        families[family] = payload
        active_family_losses.append(payload["loss"])
    return {
        "schema": "tabu.fit-trivial-baseline.v1",
        "baseline_id": "exact_support_mean_mode",
        "families": families,
        "loss": sum(active_family_losses) / len(active_family_losses),
    }


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.named_parameters()}


def _parameter_delta_norm(before: Mapping[str, torch.Tensor], model: torch.nn.Module) -> float:
    squared = 0.0
    for name, parameter in model.named_parameters():
        difference = parameter.detach().cpu().float() - before[name].float()
        squared += float(difference.square().sum().item())
    return squared**0.5


def _forward_in_eval(
    model: torch.nn.Module,
    evidence: Any,
    *,
    device: torch.device,
) -> PredictionBundle:
    """Run deterministic metric/checkpoint inference without changing train state."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            prediction = model(evidence.to(device))
    finally:
        model.train(was_training)
    if not isinstance(prediction, PredictionBundle):
        raise TypeError("fit model forward must return PredictionBundle")
    return prediction


def _mechanism_gradient_probe(
    model: torch.nn.Module,
    evidence: Any,
    truth: TruthSidecar,
    *,
    contract_id: str,
    device: torch.device,
) -> tuple[dict[str, int], dict[str, int], dict[str, float], int]:
    """Measure declared sub-path connectivity without exposing truth to the model."""

    if contract_id != "tabu4rec":
        return {}, {}, {}, 0
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        device_evidence = evidence.to(device)
        prediction = model(device_evidence)
        # The current TabU4Rec mainline is a parameterized matched score.  It
        # intentionally has no empirical user/item support arms; those
        # auxiliaries belong to the historical axis-address appendix only.
        if (
            contract_id == "tabu4rec"
            and prediction.trace is not None
            and prediction.trace.metadata.get("recommendation_address_plan") == "matched_uf"
            and prediction.trace.metadata.get("numeric_terminal") == "parameterized_matching"
        ):
            target_mask = truth.target_mask.to(device=device, dtype=torch.bool)
            numeric_targets = prediction.auxiliaries["numeric_target_mask"].to(torch.bool)
            categorical_targets = prediction.auxiliaries["categorical_target_mask"].to(torch.bool)
            scored = target_mask & (numeric_targets | categorical_targets)
            if not bool(scored.any()):
                raise FitExperimentError("Rec matched-score probe requires typed targets")
            if bool((target_mask & ~scored).any()):
                raise FitExperimentError("Rec matched-score probe cannot type every truth target")
            return {}, {}, {}, int(scored.sum().item())
        target_mask = truth.target_mask.to(device=device, dtype=torch.bool)
        numeric_targets = prediction.auxiliaries["numeric_target_mask"].to(torch.bool)
        categorical_targets = prediction.auxiliaries["categorical_target_mask"].to(torch.bool)
        if (
            numeric_targets.shape != target_mask.shape
            or categorical_targets.shape != target_mask.shape
        ):
            raise FitExperimentError("Rec arm gradient probe has invalid typed target masks")
        if bool((numeric_targets & categorical_targets).any()):
            raise FitExperimentError("Rec arm gradient probe target families must be disjoint")
        scored = target_mask & (numeric_targets | categorical_targets)
        if not bool(scored.any()):
            raise FitExperimentError("Rec arm gradient probe requires typed targets")
        if bool((target_mask & ~scored).any()):
            raise FitExperimentError("Rec arm gradient probe cannot type every truth target")
        numeric_scored = scored & numeric_targets
        categorical_scored = scored & categorical_targets
        scored_target_count = int(scored.sum().item())
        truth_values = truth.target_values.to(device=device)
        parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        source_counts: dict[str, int] = {}
        active_target_counts: dict[str, int] = {}
        gradient_norms: dict[str, float] = {}
        arms = (
            (
                "rec_user_arm",
                "rec_user_arm_values",
                "rec_user_arm_support_weights",
                "user",
            ),
            (
                "rec_item_arm",
                "rec_item_arm_values",
                "rec_item_arm_support_weights",
                "item",
            ),
        )
        for index, (name, values_key, support_key, axis) in enumerate(arms):
            values = prediction.auxiliaries.get(values_key)
            support_weights = prediction.auxiliaries.get(support_key)
            if values is None or support_weights is None:
                raise FitExperimentError(f"Rec mechanism probe is missing {name} auxiliaries")
            if values.shape != target_mask.shape or support_weights.shape[:-1] != target_mask.shape:
                raise FitExperimentError(f"Rec mechanism probe has invalid {name} shapes")
            if not bool(torch.isfinite(support_weights).all()) or bool((support_weights < 0).any()):
                raise FitExperimentError(
                    f"Rec mechanism probe requires finite nonnegative {name} weights"
                )
            active_targets = support_weights.sum(dim=-1) > 0
            active_target_counts[name] = int((active_targets & scored).sum().item())
            source_counts[name] = int(
                ((support_weights.detach() > 0) & scored.unsqueeze(-1)).sum().item()
            )
            active_family_losses: list[torch.Tensor] = []
            numeric_active = numeric_scored & active_targets
            if bool(numeric_active.any()):
                error = (values - truth_values.to(values.dtype)).square()
                numeric_loss = torch.where(
                    numeric_active,
                    error,
                    torch.zeros_like(error),
                ).sum()
                numeric_loss = numeric_loss / numeric_active.sum().to(numeric_loss.dtype).clamp_min(
                    1
                )
                active_family_losses.append(numeric_loss)

            categorical_active = categorical_scored & active_targets
            if bool(categorical_active.any()):
                n_rows, n_features = target_mask.shape
                support_values = (
                    device_evidence.values.transpose(0, 1)
                    .unsqueeze(0)
                    .expand(n_rows, n_features, n_rows)
                    if axis == "user"
                    else device_evidence.values.unsqueeze(1).expand(n_rows, n_features, n_features)
                )
                if support_values.shape != support_weights.shape:
                    raise FitExperimentError(f"Rec mechanism probe has invalid {name} support axis")
                rounded_support = support_values.round()
                visible_categorical_support = (support_weights > 0) & categorical_active.unsqueeze(
                    -1
                )
                if bool(
                    (
                        visible_categorical_support
                        & ~torch.isclose(support_values, rounded_support)
                    ).any()
                ):
                    raise FitExperimentError(
                        f"Rec categorical {name} support values must be integer codes"
                    )
                rounded_truth = truth_values.round()
                if bool((categorical_active & ~torch.isclose(truth_values, rounded_truth)).any()):
                    raise FitExperimentError("Rec categorical truth values must be integer codes")
                target_codes = rounded_truth.to(torch.int64)
                support_codes = rounded_support.to(torch.int64)
                matching_class = support_codes == target_codes.unsqueeze(-1)
                target_class_mass = (
                    support_weights * matching_class.to(support_weights.dtype)
                ).sum(dim=-1)
                arm_mass = support_weights.sum(dim=-1)
                target_probability = target_class_mass / arm_mass.clamp_min(
                    torch.finfo(arm_mass.dtype).tiny
                )
                nll = -target_probability.clamp_min(1.0e-8).log()
                categorical_loss = torch.where(
                    categorical_active,
                    nll,
                    torch.zeros_like(nll),
                ).sum()
                categorical_loss = categorical_loss / categorical_active.sum().to(
                    categorical_loss.dtype
                ).clamp_min(1)
                active_family_losses.append(categorical_loss)

            arm_loss = (
                sum(active_family_losses) / len(active_family_losses)
                if active_family_losses
                else support_weights.sum() * 0.0
            )
            gradients = torch.autograd.grad(
                arm_loss,
                parameters,
                retain_graph=index + 1 < len(arms),
                allow_unused=True,
            )
            squared = sum(
                float(gradient.detach().float().square().sum().cpu())
                for gradient in gradients
                if gradient is not None
            )
            gradient_norms[name] = squared**0.5
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
    return source_counts, active_target_counts, gradient_norms, scored_target_count


def _selected_categorical_nll(
    prediction: PredictionBundle,
    probabilities: torch.Tensor,
    scored_mask: torch.Tensor,
    target_codes: torch.Tensor,
) -> torch.Tensor:
    """Use the training log-domain terminal when it is available."""

    log_probabilities = prediction.auxiliaries.get("categorical_log_probabilities")
    class_support = prediction.auxiliaries.get("categorical_class_support_available")
    if log_probabilities is None:
        selected = probabilities[scored_mask].gather(-1, target_codes.unsqueeze(-1)).squeeze(-1)
        return -selected.clamp_min(1.0e-8).log()
    selected_log = log_probabilities[scored_mask].gather(-1, target_codes.unsqueeze(-1)).squeeze(-1)
    if class_support is None:
        return (-selected_log).clamp_min(0.0)
    selected_class_support = (
        class_support[scored_mask].gather(-1, target_codes.unsqueeze(-1)).squeeze(-1)
    )
    epsilon_log = selected_log.new_full(selected_log.shape, 1.0e-8).log()
    return (-torch.where(selected_class_support, selected_log, epsilon_log)).clamp_min(0.0)


def _typed_family_metrics(
    *,
    initial: PredictionBundle,
    final: PredictionBundle,
    truth: TruthSidecar,
    baseline: Mapping[str, Any],
) -> tuple[FitFamilyMetrics, ...]:
    device = final.auxiliaries["target_mask"].device
    truth_values = truth.target_values.to(device)
    truth_targets = truth.target_mask.to(device)
    numeric = final.auxiliaries["numeric_target_mask"].to(torch.bool)
    categorical = final.auxiliaries["categorical_target_mask"].to(torch.bool)
    support = final.auxiliaries["support_available"].to(torch.bool)
    completion = final.auxiliaries["completion_target_mask"].to(torch.bool)
    label = final.auxiliaries["label_target_mask"].to(torch.bool)
    family_masks = (
        (FitTargetFamily.COMPLETION, completion),
        (FitTargetFamily.LABEL, label),
    )
    entries: list[FitFamilyMetrics] = []
    for family, family_mask in family_masks:
        numeric_targets = truth_targets & family_mask & numeric
        if bool(numeric_targets.any()):
            scored = numeric_targets & support
            initial_values = initial.entries["numeric"].values
            final_values = final.entries["numeric"].values
            assert initial_values is not None and final_values is not None
            if bool(scored.any()):
                initial_error = initial_values[scored] - truth_values[scored]
                final_error = final_values[scored] - truth_values[scored]
                mse = float(final_error.float().square().mean().item())
                scale = float(truth_values[scored].float().std(unbiased=False).item())
                nrmse = mse**0.5 / max(scale, 1.0e-8)
                initial_loss = float(initial_error.float().square().mean().item())
            else:
                mse = nrmse = initial_loss = 0.0
            entries.append(
                FitFamilyMetrics(
                    family=family,
                    kind=FitMetricKind.NUMERIC,
                    targets=int(numeric_targets.sum().item()),
                    scored_targets=int(scored.sum().item()),
                    initial_loss=initial_loss,
                    final_loss=mse,
                    trivial_baseline_loss=baseline["families"]
                    .get(family.value, {})
                    .get("numeric_mse"),
                    mse=mse,
                    nrmse=nrmse,
                )
            )
        categorical_targets = truth_targets & family_mask & categorical
        if bool(categorical_targets.any()):
            scored = categorical_targets & support
            initial_probabilities = initial.entries["distribution"].values
            final_probabilities = final.entries["distribution"].values
            assert initial_probabilities is not None and final_probabilities is not None
            if bool(scored.any()):
                codes = truth_values[scored].round().long()
                initial_nll = float(
                    _selected_categorical_nll(
                        initial,
                        initial_probabilities,
                        scored,
                        codes,
                    )
                    .mean()
                    .item()
                )
                final_nll = float(
                    _selected_categorical_nll(
                        final,
                        final_probabilities,
                        scored,
                        codes,
                    )
                    .mean()
                    .item()
                )
                accuracy = float(
                    (final_probabilities[scored].argmax(-1) == codes).float().mean().item()
                )
            else:
                initial_nll = final_nll = accuracy = 0.0
            entries.append(
                FitFamilyMetrics(
                    family=family,
                    kind=FitMetricKind.CATEGORICAL,
                    targets=int(categorical_targets.sum().item()),
                    scored_targets=int(scored.sum().item()),
                    initial_loss=initial_nll,
                    final_loss=final_nll,
                    trivial_baseline_loss=baseline["families"]
                    .get(family.value, {})
                    .get("categorical_nll"),
                    accuracy=accuracy,
                    nll=final_nll,
                )
            )
    return tuple(entries)


def _gate_reasons(
    spec: FitExperimentSpec,
    evaluation: FitEvaluationBundle,
    *,
    initial_objective: float,
    final_objective: float,
) -> tuple[str, ...]:
    gate = spec.pass_gate
    reasons: list[str] = []
    count_validation = evaluation.count_validation
    if not count_validation.ready:
        reasons.extend(count_validation.reasons)
    if evaluation.nonfinite_seen:
        reasons.append("nonfinite_seen")
    gradient_deadline = spec.kill_conditions.require_nonzero_gradient_by_step
    if evaluation.gradient_nonzero_by_step is None:
        reasons.append("no_nonzero_gradient_by_required_step")
    elif evaluation.gradient_nonzero_by_step > gradient_deadline:
        reasons.append("nonzero_gradient_after_required_step")
    required_gradient_groups = {"dynamics", "readout"}
    # The current TabU4Rec mainline readout is the literal matched
    # User-special/Item-special inner product.  It has no trainable readout
    # parameters by design; gradients must instead reach the carrier and
    # dynamics that form the two special-token axes.
    if (
        spec.contract_id == "tabu4rec"
        and spec.semantic.recommendation_address_plan is not None
        and spec.semantic.recommendation_address_plan.value == "matched_uf"
    ):
        required_gradient_groups.discard("readout")
        required_gradient_groups.add("carrier")
    if spec.contract_id == "tabu.unit_row":
        required_gradient_groups.add("carrier")
    elif spec.contract_id == "tabu.unit_pair":
        required_gradient_groups.add("tokenizer")
    for group in sorted(required_gradient_groups):
        first_nonzero = evaluation.gradient_group_nonzero_by_step.get(group)
        if first_nonzero is None:
            reasons.append(f"no_{group}_gradient_by_required_step")
        elif first_nonzero > gradient_deadline:
            reasons.append(f"{group}_gradient_after_required_step")
    if evaluation.parameter_delta_norm <= 0.0:
        reasons.append("zero_parameter_delta")
    if (
        spec.contract_id == "tabu4rec"
        and spec.semantic.recommendation_address_plan.value != "matched_uf"
    ):
        if evaluation.mechanism_scored_target_count != evaluation.scored_targets:
            reasons.append("rec_mechanism_scored_target_count_mismatch")
        for mechanism in ("rec_user_arm", "rec_item_arm"):
            if evaluation.mechanism_source_counts.get(mechanism, 0) <= 0:
                reasons.append(f"{mechanism}_has_no_target_support")
            if (
                evaluation.mechanism_active_target_counts.get(mechanism, 0)
                != evaluation.scored_targets
            ):
                reasons.append(f"{mechanism}_not_active_for_every_scored_target")
            if evaluation.mechanism_gradient_norms.get(mechanism, 0.0) <= 0.0:
                reasons.append(f"{mechanism}_has_no_target_gradient")
    if not evaluation.checkpoint_reloaded:
        reasons.append("checkpoint_reload_failed")
    if final_objective / max(initial_objective, 1.0e-12) > gate.max_loss_ratio:
        reasons.append("aggregate_loss_ratio")
    for family in evaluation.families:
        ratio = family.final_loss / max(family.initial_loss, 1.0e-12)
        if ratio > gate.max_loss_ratio:
            reasons.append(f"{family.family.value}_{family.kind.value}_loss_ratio")
        if family.kind is FitMetricKind.NUMERIC:
            if gate.max_numeric_mse is not None and (
                family.mse is None or family.mse > gate.max_numeric_mse
            ):
                reasons.append(f"{family.family.value}_numeric_mse")
            if gate.max_numeric_nrmse is not None and (
                family.nrmse is None or family.nrmse > gate.max_numeric_nrmse
            ):
                reasons.append(f"{family.family.value}_numeric_nrmse")
        else:
            if gate.min_categorical_accuracy is not None and (
                family.accuracy is None or family.accuracy < gate.min_categorical_accuracy
            ):
                reasons.append(f"{family.family.value}_categorical_accuracy")
            if gate.max_categorical_nll is not None and (
                family.nll is None or family.nll > gate.max_categorical_nll
            ):
                reasons.append(f"{family.family.value}_categorical_nll")
        if gate.max_trivial_baseline_ratio is not None:
            if family.trivial_baseline_loss is None:
                reasons.append(f"{family.family.value}_{family.kind.value}_baseline_missing")
            elif family.final_loss / max(family.trivial_baseline_loss, 1.0e-12) > (
                gate.max_trivial_baseline_ratio
            ):
                reasons.append(f"{family.family.value}_{family.kind.value}_baseline_ratio")
    return tuple(dict.fromkeys(reasons))


def _seed_verdict(
    evaluation: FitEvaluationBundle,
    reasons: Sequence[str],
    *,
    diagnostic: bool = False,
) -> str:
    if not evaluation.count_validation.ready:
        return "invalid"
    if reasons:
        return "failed"
    return "diagnostic_pass" if diagnostic else "pass"


def _training_and_execution_configs(
    spec: FitExperimentSpec, *, device: torch.device
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    EnvironmentDisclosure,
    dict[str, Any],
]:
    training = spec.training.model_dump(mode="json")
    disclosure, environment = capture_environment(str(device))
    execution = {
        **spec.execution.model_dump(mode="json"),
        "resolved_device": str(device),
        "environment_hash": canonical_hash(environment),
        "host_class": environment["host_class"],
        "torch_version": environment["torch_version"],
        "python_version": environment["python_version"],
    }
    return training, execution, disclosure, environment


def _identity(
    spec: FitExperimentSpec,
    fixture: F0Fixture,
    *,
    seed: int,
    code_hash: str,
    compiler_hash: str,
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> RunIdentity:
    seeds = {
        "episode": spec.seeds.data_seed,
        "model_init": seed,
        "numpy": seed,
        "python": seed,
        "sampler": spec.seeds.episode_order_seed,
        "torch_cpu": seed,
    }
    if spec.execution.device.value == "cuda":
        seeds["torch_cuda"] = seed
    elif spec.execution.device.value == "mps":
        seeds["torch_mps"] = seed
    return RunIdentity.create(
        spec_hash=spec.spec_hash,
        code_hash=code_hash,
        data_hash=fixture.dataset.dataset_hash,
        split_hash=spec.split.content_hash,
        compiler_hash=compiler_hash,
        semantic_config_hash=spec.semantic.content_hash,
        execution_config_hash=canonical_hash(execution),
        training_config_hash=canonical_hash(training),
        seeds=seeds,
    )


def _runtime_failure_artifacts(
    *,
    destination: Path,
    attempt_id: str,
    attempt_nonce: str,
    identity: RunIdentity,
    spec: FitExperimentSpec,
    fixture: F0Fixture,
    episode_recipe_hashes: Sequence[str] | None = None,
    preregistration_text: str,
    code_manifest: Mapping[str, Any],
    compiler_manifest: Mapping[str, Any],
    feasibility: FitFeasibilityReport,
    baseline: Mapping[str, Any],
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
    environment: EnvironmentDisclosure,
    environment_payload: Mapping[str, Any],
    command: Sequence[str],
    formal_authorization: FormalAuthorizationContext | None,
    phase: str,
    error: Exception,
    started_at: datetime,
) -> FitAttemptArtifacts:
    """Publish a typed, self-verifying receipt when formal artifacts cannot exist.

    The formal fit writer correctly requires real predictions and evaluations,
    which do not exist after build/train/evaluation crashes.  This narrower
    bundle uses the public RunBundle and Receipt schemas without fabricating
    those artifacts.  It is deliberately not accepted by the formal fit
    artifact verifier.
    """

    try:
        source_identity = SourceIdentity.model_validate(code_manifest["source_identity"])
    except (KeyError, ValueError) as exc:
        raise FitExperimentError("runtime failure source identity is invalid") from exc
    source_identity_hash = canonical_hash(source_identity)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable fit failure attempt already exists: {destination}")
    is_formal = source_identity.issuance_status == "formal"
    if is_formal:
        if formal_authorization is None:
            raise FitExperimentError(
                "formal runtime failure issuance requires replayable authorization"
            )
        try:
            verified_authorization = verify_formal_authorization(
                formal_authorization,
                preregistration_text=preregistration_text,
                live_source_identity=source_identity,
            )
        except (FormalAuthorizationError, TypeError, ValueError) as exc:
            raise FitExperimentError("formal runtime failure authorization replay failed") from exc
        authorization_summary = verified_authorization.summary.model_dump(mode="json")
    else:
        if formal_authorization is not None:
            raise FitExperimentError(
                "local runtime failure evidence cannot use formal authorization"
            )
        authorization_summary = None
    boundary = _exception_boundary(phase, error, formal=is_formal)
    if is_formal:
        assert_public_payload_safe(
            {
                "boundary": boundary,
                "command": tuple(command),
                "code_manifest": code_manifest,
                "environment": environment_payload,
            },
            location="formal runtime failure evidence",
        )
    completed_at = datetime.now(UTC)
    error_text = f"{phase}: {boundary['exception_type']}: {boundary['message']}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        preregistration = staging / "preregistration.yaml"
        preregistration.write_text(preregistration_text, encoding="utf-8")
        if not preregistration_text.endswith("\n"):
            with preregistration.open("a", encoding="utf-8") as target:
                target.write("\n")
        resolved = staging / "resolved-configs"
        resolved.mkdir()
        for name, payload in sorted(
            {
                "code": code_manifest,
                "experiment": spec,
                "semantic": spec.semantic,
                "training": training,
                "execution": execution,
                "seeds": identity.seeds,
            }.items()
        ):
            _write_canonical_json(resolved / f"{name}.json", payload)
        _write_canonical_json(
            staging / "dataset-manifest.json",
            {
                "schema": "tabu.fit-dataset-manifest.v1",
                "dataset": spec.dataset,
                "dataset_id": fixture.dataset.dataset_id,
                "dataset_hash": fixture.dataset.dataset_hash,
                "feature_specs": fixture.dataset.feature_specs,
                "row_ids": fixture.dataset.row_ids,
                "metadata": fixture.dataset.metadata,
            },
        )
        _write_canonical_json(staging / "split-manifest.json", spec.split)
        _write_canonical_json(staging / "compiler-manifest.json", compiler_manifest)
        _write_canonical_json(staging / "feasibility.json", feasibility)
        _write_canonical_json(staging / "baselines.json", baseline)
        _write_canonical_json(staging / "environment.json", environment_payload)
        _write_canonical_json(
            staging / "failure.json",
            {
                **boundary,
                "attempt_id": attempt_id,
                "attempt_nonce": attempt_nonce,
                "run_id": identity.run_id,
                "experiment_id": spec.experiment_id,
                "contract_id": spec.contract_id,
                "stage": spec.stage,
                "model_seed": identity.seeds["model_init"],
                "started_at": started_at,
                "completed_at": completed_at,
                "claim_boundary": "runtime_failure_only_no_fit_result_no_accepted_claim",
            },
        )
        (staging / "verdict.md").write_text(
            "\n".join(
                (
                    "# Fit verdict: failed",
                    "",
                    f"- experiment: `{spec.experiment_id}`",
                    f"- contract: `{spec.contract_id}`",
                    f"- stage: `{spec.stage.value}`",
                    f"- model seed: `{identity.seeds['model_init']}`",
                    f"- failure phase: `{phase}`",
                    f"- failure code: `{boundary['code']}`",
                    "- boundary: no fit result; this receipt records only the runtime failure",
                    "",
                )
            ),
            encoding="utf-8",
        )

        payload_paths = tuple(
            path
            for path in sorted(
                staging.rglob("*"), key=lambda item: item.relative_to(staging).as_posix()
            )
            if path.is_file()
        )
        artifact_refs = tuple(_failure_artifact_ref(staging, path) for path in payload_paths)
        run_metadata: dict[str, Any] = {
            "attempt_id": attempt_id,
            "attempt_nonce": attempt_nonce,
            "experiment_id": spec.experiment_id,
            "contract_version": spec.contract_version,
            "block_kind": spec.semantic.dynamics.block_kind.value,
            "numeric_terminal": spec.semantic.numeric_terminal.value,
            "variant_role": (
                "canonical"
                if spec.semantic.dynamics.block_kind.value == "omab"
                else "non_o_ablation"
            ),
            "checkpoint_license_id": "Apache-2.0",
            "stage": spec.stage.value,
            "model_seed": identity.seeds["model_init"],
            "code_hash": identity.code_hash,
            "failure_phase": phase,
            "failure_code": boundary["code"],
            "verdict": "failed",
            "issuance_status": source_identity.issuance_status,
            "source_identity_hash": source_identity_hash,
            "claim_boundary": "runtime_failure_only_no_fit_result_no_accepted_claim",
        }
        run_metadata.update(_tabubase_identity_metadata(spec))
        if authorization_summary is not None:
            run_metadata["formal_authorization"] = authorization_summary
        recipe_hashes = (
            (fixture.recipe.recipe_hash,)
            if episode_recipe_hashes is None
            else tuple(episode_recipe_hashes)
        )
        run_bundle = RunBundle(
            identity=identity,
            model_id=spec.contract_id,
            dataset_id=fixture.dataset.dataset_id,
            fit_partition=spec.split.fit_partition,
            environment=environment,
            episode_recipe_hashes=recipe_hashes,
            artifacts=artifact_refs,
            metadata=run_metadata,
        )
        _write_canonical_json(staging / "run_bundle.json", run_bundle)
        _write_canonical_json(
            staging / "run_manifest.json",
            {
                "schema": "tabu.fit-runtime-failure-manifest.v1",
                "attempt_id": attempt_id,
                "run_id": identity.run_id,
                "run_identity": identity,
                "run_bundle_hash": run_bundle.run_bundle_hash,
                "status": ReceiptStatus.FAILED,
                "failure_phase": phase,
                "failure_code": boundary["code"],
                "issuance_status": source_identity.issuance_status,
                "source_identity_hash": source_identity_hash,
                "claim_boundary": "runtime_failure_only_no_fit_result_no_accepted_claim",
            },
        )
        receipt = Receipt.from_run_bundle(
            run_bundle,
            receipt_id=f"receipt-{attempt_id}",
            status=ReceiptStatus.FAILED,
            created_at=started_at,
            completed_at=completed_at,
            command=tuple(command),
            artifacts=artifact_refs,
            error=error_text,
            metadata={
                "attempt_id": attempt_id,
                "verdict": "failed",
                "failure_phase": phase,
                "failure_code": boundary["code"],
                "bundle_schema": "tabu.fit-runtime-failure.v1",
                "issuance_status": source_identity.issuance_status,
                "source_identity_hash": source_identity_hash,
                "block_kind": spec.semantic.dynamics.block_kind.value,
                "numeric_terminal": spec.semantic.numeric_terminal.value,
                "variant_role": (
                    "canonical"
                    if spec.semantic.dynamics.block_kind.value == "omab"
                    else "non_o_ablation"
                ),
                **_tabubase_identity_metadata(spec),
                **(
                    {"formal_authorization": authorization_summary}
                    if authorization_summary is not None
                    else {}
                ),
            },
        )
        write_receipt(staging / "receipt.json", receipt)
        if source_identity.issuance_status == "formal":
            assert_public_artifact_tree_safe(
                staging,
                location="formal runtime failure evidence",
            )
        checksum_paths = tuple(
            path
            for path in sorted(
                staging.rglob("*"), key=lambda item: item.relative_to(staging).as_posix()
            )
            if path.is_file() and path.name != "artifacts.sha256"
        )
        (staging / "artifacts.sha256").write_text(
            "".join(
                f"{_file_sha256(path)}  {path.relative_to(staging).as_posix()}\n"
                for path in checksum_paths
            ),
            encoding="utf-8",
        )
        verified = verify_fit_attempt_artifacts(
            staging,
            formal_authorization=formal_authorization,
        )
        if verified.receipt_hash != receipt.receipt_hash:
            raise FitExperimentError("runtime failure receipt read-back hash mismatch")
        _publish_directory_create_once(staging, destination)
        return FitAttemptArtifacts(
            directory=destination,
            receipt=destination / "receipt.json",
            checksums=destination / "artifacts.sha256",
            run_bundle=destination / "run_bundle.json",
            checkpoint=None,
            receipt_hash=receipt.receipt_hash,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _aggregate_verdict(results: Sequence[SeedRunResult]) -> str:
    if len(results) != 3:
        raise FitExperimentError("fit-first aggregate requires exactly three seed attempts")
    diagnostic = all(
        result.verdict in {"diagnostic_pass", "failed", "invalid"} for result in results
    ) and any(result.verdict == "diagnostic_pass" for result in results)
    passing_verdict = "diagnostic_pass" if diagnostic else "pass"
    passed = sum(result.verdict == passing_verdict for result in results)
    if passed == len(results):
        return passing_verdict
    if passed == len(results) - 1:
        return "diagnostic_unstable" if diagnostic else "unstable"
    return "failed"


def _write_experiment_aggregate(
    *,
    output_root: Path,
    spec: FitExperimentSpec,
    results: Sequence[SeedRunResult],
) -> ExperimentAggregateArtifacts:
    """Bind all frozen seed receipts into one immutable aggregate summary."""

    if tuple(result.model_seed for result in results) != spec.seeds.model_seeds:
        raise FitExperimentError("aggregate requires all frozen model seeds in order")
    attempts: list[dict[str, Any]] = []
    for result in results:
        receipt = read_receipt(result.artifacts.receipt)
        try:
            run_bundle = RunBundle.model_validate_json(
                result.artifacts.run_bundle.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise FitExperimentError("aggregate attempt has an invalid RunBundle") from error
        run_id = result.artifacts.directory.parent.name
        attempt_id = result.artifacts.directory.name
        if receipt.receipt_hash != result.artifacts.receipt_hash:
            raise FitExperimentError("aggregate attempt receipt hash mismatch")
        if (
            run_bundle.run_bundle_hash != receipt.run_bundle_hash
            or run_bundle.run_id != receipt.run_id
        ):
            raise FitExperimentError("aggregate RunBundle is not bound to its receipt")
        if receipt.run_id != run_id or receipt.metadata.get("attempt_id") != attempt_id:
            raise FitExperimentError("aggregate attempt path is not bound to its receipt")
        declared_verdict = run_bundle.metadata.get(
            "attempt_verdict", run_bundle.metadata.get("verdict")
        )
        if declared_verdict != result.verdict:
            raise FitExperimentError("aggregate verdict conflicts with RunBundle metadata")
        expected_status = (
            ReceiptStatus.SUCCEEDED
            if result.verdict in {"pass", "diagnostic_pass"}
            else ReceiptStatus.FAILED
        )
        if receipt.status is not expected_status:
            raise FitExperimentError("aggregate verdict conflicts with receipt status")
        attempts.append(
            {
                "model_seed": result.model_seed,
                "run_id": run_id,
                "attempt_id": attempt_id,
                "receipt_id": receipt.receipt_id,
                "receipt_hash": receipt.receipt_hash,
                "receipt_status": receipt.status,
                "run_bundle_hash": receipt.run_bundle_hash,
                "verdict": result.verdict,
                "failure_phase": result.failure_phase,
            }
        )
    verdict = _aggregate_verdict(results)
    aggregate = {
        "schema": "tabu.fit-seed-aggregate.v1",
        "experiment_id": spec.experiment_id,
        "contract_id": spec.contract_id,
        "contract_version": spec.contract_version,
        "stage": spec.stage,
        "spec_hash": spec.spec_hash,
        "expected_model_seeds": spec.seeds.model_seeds,
        "seed_attempts": tuple(attempts),
        "passed_seed_count": sum(
            result.verdict in {"pass", "diagnostic_pass"} for result in results
        ),
        "verdict": verdict,
        "gate_passed": verdict == "pass",
        "fit_succeeded": verdict in {"pass", "diagnostic_pass"},
        "claim_boundary": (
            "nondeterministic_diagnostic_only_no_gate_no_accepted_claim"
            if spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
            else "three_seed_fit_aggregate_only_no_accepted_claim"
        ),
    }
    aggregate_hash = canonical_hash(aggregate)
    envelope = {
        "schema": "tabu.fit-seed-aggregate-envelope.v1",
        "aggregate_hash": aggregate_hash,
        "aggregate": aggregate,
    }
    destination = output_root / "aggregates" / spec.experiment_id / f"aggregate-{aggregate_hash}"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"immutable aggregate already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        summary = staging / "aggregate.json"
        _write_canonical_json(summary, envelope)
        checksums = staging / "artifacts.sha256"
        checksums.write_text(
            f"{_file_sha256(summary)}  aggregate.json\n",
            encoding="utf-8",
        )
        _publish_directory_create_once(staging, destination)
        return ExperimentAggregateArtifacts(
            directory=destination,
            summary=destination / "aggregate.json",
            checksums=destination / "artifacts.sha256",
            aggregate_hash=aggregate_hash,
            verdict=verdict,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _trainer(
    model: torch.nn.Module,
    spec: FitExperimentSpec,
    identity: RunIdentity,
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> Trainer:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.training.learning_rate,
        weight_decay=spec.training.weight_decay,
    )
    generators = {
        "episode": torch.Generator(device="cpu").manual_seed(identity.seeds["episode"]),
        "sampler": torch.Generator(device="cpu").manual_seed(identity.seeds["sampler"]),
    }
    return Trainer(
        model,
        objective=Objective(),
        optimizer=optimizer,
        max_gradient_norm=spec.training.gradient_clip_norm,
        run_identity=identity,
        training_config=training,
        execution_config=execution,
        named_generators=generators,
    )


def _checkpoint_roundtrip(
    trainer: Trainer,
    spec: FitExperimentSpec,
    identity: RunIdentity,
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
    evidence: Any,
    expected: PredictionBundle,
    *,
    seed: int,
    device: torch.device,
) -> bool:
    with tempfile.TemporaryDirectory(prefix="tabu-fit-checkpoint-") as directory:
        path = Path(directory) / "checkpoint.safetensors"
        trainer.save_checkpoint(path)
        restored_model = _build_model(spec, seed=seed, device=device)
        restored = _trainer(
            restored_model,
            spec,
            identity,
            training,
            execution,
        )
        restored.load_checkpoint(path)
        actual = _forward_in_eval(restored_model, evidence, device=device)
        return actual.prediction_hash == expected.prediction_hash


def _run_f0_seed(
    *,
    spec: FitExperimentSpec,
    fixture: F0Fixture,
    identity: RunIdentity,
    seed: int,
    device: torch.device,
    output_root: Path,
    preregistration_text: str,
    code_manifest: Mapping[str, Any],
    code_hash: str,
    compiler_manifest: Mapping[str, Any],
    feasibility: FitFeasibilityReport,
    baseline: Mapping[str, Any],
    training: Mapping[str, Any],
    execution: Mapping[str, Any],
    environment: EnvironmentDisclosure,
    environment_payload: Mapping[str, Any],
    command: Sequence[str],
    formal_authorization: FormalAuthorizationContext | None,
) -> SeedRunResult:
    try:
        source_identity = SourceIdentity.model_validate(code_manifest["source_identity"])
    except (KeyError, ValueError) as exc:
        raise FitExperimentError("fit source identity is invalid") from exc
    source_identity_hash = canonical_hash(source_identity)
    started_at = datetime.now(UTC)
    attempt_nonce = f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}-{uuid.uuid4().hex}"
    attempt_id = derive_attempt_id(
        run_id=identity.run_id,
        attempt_nonce=attempt_nonce,
    )
    attempt_directory = output_root / identity.run_id / attempt_id
    observer = get_observer(
        run_id=identity.run_id,
        attempt_id=attempt_id,
        experiment_id=spec.experiment_id,
        contract_id=spec.contract_id,
        seed=seed,
        stage=spec.stage.value,
        environment_payload=environment_payload,
    )
    phase = "build"
    try:
        model = _build_model(spec, seed=seed, device=device)
        trainer = _trainer(model, spec, identity, training, execution)
        evidence = fixture.evidence
        truth = fixture.truth
        initial_parameters = _parameter_snapshot(model)

        phase = "initial_evaluation"
        initial_prediction = _forward_in_eval(model, evidence, device=device)
        initial_loss = Objective()(initial_prediction, truth.to(device))
        initial_objective = float(initial_loss.total.detach().cpu())
        if not math.isfinite(initial_objective):
            raise _NonfiniteFitError("initial objective is NaN or Inf")
        phase = "mechanism_probe"
        (
            mechanism_source_counts,
            mechanism_active_target_counts,
            mechanism_gradient_norms,
            mechanism_scored_target_count,
        ) = _mechanism_gradient_probe(
            model,
            evidence,
            truth,
            contract_id=spec.contract_id,
            device=device,
        )
        gradient_nonzero_by_step: int | None = None
        gradient_group_nonzero_by_step: dict[str, int] = {}
        gradient_group_max_norms: dict[str, float] = {}
        history: list[dict[str, Any]] = [
            {
                "record_type": "step",
                "step": 0,
                "loss": initial_objective,
                "gradient_norm": None,
                "gradient_norms": {},
                "mechanism_source_counts": mechanism_source_counts,
                "mechanism_active_target_counts": mechanism_active_target_counts,
                "mechanism_scored_target_count": mechanism_scored_target_count,
                "mechanism_gradient_norms": mechanism_gradient_norms,
                "elapsed_seconds": 0.0,
            }
        ]
        observer.log_step(history[-1])

        phase = "train"
        started = time.monotonic()
        for _ in range(spec.training.max_updates):
            step = trainer.train_step(evidence, truth)
            elapsed = time.monotonic() - started
            loss_value = float(step.loss.total.detach().cpu())
            if not math.isfinite(loss_value) or not math.isfinite(step.gradient_norm):
                raise _NonfiniteFitError(f"nonfinite training state observed at step {step.step}")
            if step.gradient_norm > 0.0 and gradient_nonzero_by_step is None:
                gradient_nonzero_by_step = step.step
            for group, norm in step.gradient_norms.items():
                gradient_group_max_norms[group] = max(
                    gradient_group_max_norms.get(group, 0.0),
                    float(norm),
                )
                if norm > 0.0 and group not in gradient_group_nonzero_by_step:
                    gradient_group_nonzero_by_step[group] = step.step
            if step.step == 1 or step.step % 10 == 0:
                history.append(
                    {
                        "record_type": "step",
                        "step": step.step,
                        "loss": loss_value,
                        "gradient_norm": step.gradient_norm,
                        "gradient_norms": dict(step.gradient_norms),
                        "elapsed_seconds": elapsed,
                    }
                )
                observer.log_step(history[-1])
            if elapsed >= spec.training.wall_clock_budget_minutes * 60:
                break

        phase = "final_evaluation"
        final_prediction = _forward_in_eval(model, evidence, device=device)
        final_loss = Objective()(final_prediction, truth.to(device))
        final_objective = float(final_loss.total.detach().cpu())
        if not math.isfinite(final_objective):
            raise _NonfiniteFitError("final objective is NaN or Inf")
        evaluation = Evaluator().evaluate(
            (final_prediction,),
            (truth,),
            evaluation_id=f"{spec.experiment_id}-{seed}",
        )
        parameter_delta = _parameter_delta_norm(initial_parameters, model)

        phase = "checkpoint"
        checkpoint_reloaded = _checkpoint_roundtrip(
            trainer,
            spec,
            identity,
            training,
            execution,
            evidence,
            final_prediction,
            seed=seed,
            device=device,
        )

        phase = "evaluation"
        families = _typed_family_metrics(
            initial=initial_prediction,
            final=final_prediction,
            truth=truth,
            baseline=baseline,
        )
        fit_evaluation = FitEvaluationBundle(
            evaluation_id=f"{spec.experiment_id}-{seed}-fit",
            experiment_id=spec.experiment_id,
            stage=spec.stage,
            model_seed=seed,
            targets=evaluation.counts["targets"],
            scored_targets=evaluation.counts["scored_targets"],
            coverage=float(evaluation.metrics["coverage"] or 0.0),
            families=families,
            gradient_nonzero_by_step=gradient_nonzero_by_step,
            gradient_group_nonzero_by_step=gradient_group_nonzero_by_step,
            gradient_group_max_norms=gradient_group_max_norms,
            mechanism_source_counts=mechanism_source_counts,
            mechanism_active_target_counts=mechanism_active_target_counts,
            mechanism_scored_target_count=mechanism_scored_target_count,
            mechanism_gradient_norms=mechanism_gradient_norms,
            parameter_delta_norm=parameter_delta,
            nonfinite_seen=False,
            checkpoint_reloaded=checkpoint_reloaded,
        )
        reasons = _gate_reasons(
            spec,
            fit_evaluation,
            initial_objective=initial_objective,
            final_objective=final_objective,
        )
        diagnostic = spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC
        verdict = _seed_verdict(
            fit_evaluation,
            reasons,
            diagnostic=diagnostic,
        )
        status = (
            ReceiptStatus.SUCCEEDED
            if verdict in {"pass", "diagnostic_pass"}
            else ReceiptStatus.FAILED
        )
        verdict_text = "\n".join(
            (
                f"# Fit verdict: {verdict}",
                "",
                f"- experiment: `{spec.experiment_id}`",
                f"- contract: `{spec.contract_id}`",
                f"- stage: `{spec.stage.value}`",
                f"- model seed: `{seed}`",
                f"- reasons: `{', '.join(reasons) if reasons else 'none'}`",
                "- boundary: support-realizable fixed-episode fit only; no generalization claim",
            )
        )
        summary = {
            "record_type": "summary",
            "fit_evaluation": fit_evaluation,
            "initial_objective": initial_objective,
            "final_objective": final_objective,
            "loss_ratio": final_objective / max(initial_objective, 1.0e-12),
            "steps": trainer.step,
            "verdict": verdict,
            "reasons": reasons,
        }
        observer.log_summary(summary)

        phase = "artifact"
        fit_metadata: dict[str, Any] = {
            "attempt_nonce": attempt_nonce,
            "attempt_verdict": verdict,
            "experiment_id": spec.experiment_id,
            "contract_version": spec.contract_version,
            "block_kind": spec.semantic.dynamics.block_kind.value,
            "numeric_terminal": spec.semantic.numeric_terminal.value,
            "variant_role": (
                "canonical"
                if spec.semantic.dynamics.block_kind.value == "omab"
                else "non_o_ablation"
            ),
            "checkpoint_license_id": "Apache-2.0",
            "stage": spec.stage.value,
            "model_seed": seed,
            "code_hash": code_hash,
            "issuance_status": source_identity.issuance_status,
            "source_identity_hash": source_identity_hash,
            "evidence_mode": spec.execution.evidence_mode,
        }
        fit_metadata.update(_tabubase_identity_metadata(spec, model=model))
        artifacts = write_fit_attempt_artifacts(
            attempt_directory,
            attempt_id=attempt_id,
            run_identity=identity,
            model_id=spec.contract_id,
            dataset_id=fixture.dataset.dataset_id,
            fit_partition=spec.split.fit_partition,
            preregistration_text=preregistration_text,
            resolved_configs={
                "code": code_manifest,
                "experiment": spec,
                "semantic": spec.semantic,
                "training": training,
                "execution": execution,
                "seeds": identity.seeds,
            },
            dataset_manifest={
                "schema": "tabu.fit-dataset-manifest.v1",
                "dataset": spec.dataset,
                "dataset_id": fixture.dataset.dataset_id,
                "dataset_hash": fixture.dataset.dataset_hash,
                "feature_specs": fixture.dataset.feature_specs,
                "row_ids": fixture.dataset.row_ids,
                "metadata": fixture.dataset.metadata,
            },
            split_manifest=spec.split,
            compiler_manifest=compiler_manifest,
            feasibility=feasibility,
            metrics={"summary": summary, "history": tuple(history)},
            evaluation=evaluation,
            predictions=(final_prediction,),
            baselines=baseline,
            verdict=verdict_text,
            status=status,
            error=(None if verdict in {"pass", "diagnostic_pass"} else "; ".join(reasons)),
            command=tuple(command),
            checkpoint_writer=lambda path, bound=trainer: bound.save_checkpoint(path),
            metadata=fit_metadata,
            formal_authorization=formal_authorization,
        )
        observer.close()
        return SeedRunResult(
            model_seed=seed,
            verdict=verdict,
            fit_evaluation=fit_evaluation,
            artifacts=artifacts,
            error=(None if verdict in {"pass", "diagnostic_pass"} else "; ".join(reasons)),
        )
    except Exception as error:
        boundary = _exception_boundary(
            phase,
            error,
            formal=source_identity.issuance_status == "formal",
        )
        artifacts = _runtime_failure_artifacts(
            destination=attempt_directory,
            attempt_id=attempt_id,
            attempt_nonce=attempt_nonce,
            identity=identity,
            spec=spec,
            fixture=fixture,
            preregistration_text=preregistration_text,
            code_manifest=code_manifest,
            compiler_manifest=compiler_manifest,
            feasibility=feasibility,
            baseline=baseline,
            training=training,
            execution=execution,
            environment=environment,
            environment_payload=environment_payload,
            command=command,
            formal_authorization=formal_authorization,
            phase=phase,
            error=error,
            started_at=started_at,
        )
        if boundary["code"] == "out_of_memory" and device.type == "cuda":
            torch.cuda.empty_cache()
        observer.close()
        return SeedRunResult(
            model_seed=seed,
            verdict="failed",
            fit_evaluation=None,
            artifacts=artifacts,
            failure_phase=phase,
            error=f"{boundary['exception_type']}: {boundary['message']}",
        )


def run_fit_experiment(
    preregistration: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str],
    prepared_bundle: str | os.PathLike[str] | None = None,
    repository: str | os.PathLike[str] | None = None,
    command: Sequence[str] = (),
    formal: bool = False,
    source_reviewed: bool = False,
    authorization_catalog: str | os.PathLike[str] | None = None,
    source_identity: SourceIdentity | None = None,
    distribution_artifact: bytes | str | os.PathLike[str] | None = None,
    distribution_lock: bytes | str | os.PathLike[str] | None = None,
) -> ExperimentRunResult | R1RunReceipt:
    """Run every preregistered seed with an explicit formal/local evidence boundary."""

    source_path = Path(preregistration)
    preregistration_text = source_path.read_text(encoding="utf-8")
    spec = load_fit_experiment(source_path)
    if spec.stage is FitStage.S1:
        # Keep F0 stable while the multi-episode execution contract lives in a
        # separate module.  The lazy import avoids a module cycle because the
        # S1 implementation reuses the shared runner primitives above.
        from .s1_runner import run_s1_experiment

        return run_s1_experiment(
            source_path,
            output_root=output_root,
            repository=repository,
            command=command,
            formal=formal,
            source_reviewed=source_reviewed,
            authorization_catalog=authorization_catalog,
            source_identity=source_identity,
            distribution_artifact=distribution_artifact,
            distribution_lock=distribution_lock,
        )
    if spec.stage is FitStage.R1:
        if formal or source_reviewed or authorization_catalog is not None:
            raise FitExperimentError(
                "R1 v0 is a local_unissued wedge; formal issuance is deferred until "
                "a reviewed model artifact and evaluator receipt exist"
            )
        if prepared_bundle is None:
            raise FitExperimentError(
                "R1 execution requires --prepared with a private PreparedEvalDataBundle"
            )
        from .r1_runner import run_r1

        receipt_path = Path(output_root) / spec.experiment_id / "r1-receipt.json"
        return run_r1(
            prepared_bundle,
            output=receipt_path,
            model_spec_hash=spec.model_spec_hash,
            contract_version=spec.contract_version,
        )
    if source_reviewed and not formal:
        raise FitExperimentError("source_reviewed is only meaningful for a formal request")
    if formal and authorization_catalog is None:
        raise FitExperimentError(
            "formal runs require authorization_catalog; source_reviewed cannot self-authorize"
        )
    if not formal and authorization_catalog is not None:
        raise FitExperimentError("authorization_catalog is only valid for a formal request")
    authorization_context: FormalAuthorizationContext | None = None
    verified_authorization: VerifiedFormalAuthorization | None = None
    if formal:
        assert authorization_catalog is not None
        _assert_formal_output_root_safe(output_root, repository=repository)
        authorization_context, verified_authorization = _resolve_formal_authorization(
            authorization_catalog,
            spec=spec,
            preregistration_path=source_path,
            preregistration_text=preregistration_text,
            repository=repository,
        )
    if formal and (spec.execution.evidence_mode is FitEvidenceMode.DIAGNOSTIC_NONDETERMINISTIC):
        raise FitExperimentError(
            "nondeterministic diagnostic execution cannot issue formal or Gate 1 evidence"
        )
    if spec.stage is not FitStage.F0:
        raise FitExperimentError(
            "S1/R1 execution is gated on the complete seven-model F0 checkpoint review"
        )
    fixture = validate_f0_binding(spec)
    compiler_manifest = compiler_binding_manifest(spec, fixture)
    targets, feasibility = assess_fixture_feasibility(fixture, spec)
    if feasibility.status is not FeasibilityReportStatus.READY:
        raise FitExperimentError("positive F0 fixture is not terminal-feasible")
    baseline = trivial_baseline(targets)
    device = _device(spec)
    code_manifest = source_tree_manifest(
        repository,
        preregistration=source_path,
        request_formal=formal,
        reviewed=authorization_context is not None,
        source_identity=source_identity,
        distribution_artifact=distribution_artifact,
        distribution_lock=distribution_lock,
    )
    resolved_source_identity = SourceIdentity.model_validate(code_manifest["source_identity"])
    if formal and resolved_source_identity.issuance_status != "formal":
        reasons = ", ".join(resolved_source_identity.reasons)
        raise FitExperimentError(
            f"formal receipt refused by SourceIdentity: {reasons or 'source is not formal'}"
        )
    if authorization_context is not None:
        assert verified_authorization is not None
        try:
            verified_authorization = verify_formal_authorization(
                authorization_context,
                preregistration_text=preregistration_text,
                live_source_identity=resolved_source_identity,
                expected_summary=verified_authorization.summary,
            )
        except (FormalAuthorizationError, TypeError, ValueError) as exc:
            raise FitExperimentError(
                "live formal SourceIdentity does not match canonical authorization"
            ) from exc
    recorded_command = _authorization_safe_command(
        command,
        (None if verified_authorization is None else verified_authorization.summary),
        output_root=output_root if formal else None,
        preregistration_path=source_path if formal else None,
    )
    if formal:
        assert_public_payload_safe(
            {"command": recorded_command},
            location="formal recorded command",
        )
    code_hash = canonical_hash(code_manifest)
    compiler_hash = canonical_hash(compiler_manifest)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(spec.execution.deterministic_algorithms)
    results: list[SeedRunResult] = []
    try:
        try:
            training, execution, environment, environment_payload = _training_and_execution_configs(
                spec, device=device
            )
            identities = tuple(
                _identity(
                    spec,
                    fixture,
                    seed=seed,
                    code_hash=code_hash,
                    compiler_hash=compiler_hash,
                    training=training,
                    execution=execution,
                )
                for seed in spec.seeds.model_seeds
            )
        except Exception as error:
            raise FitExperimentError(
                "fit preflight failed before RunIdentity formation; the current public "
                "Receipt schema cannot safely represent an identity-free attempt"
            ) from error
        for seed, identity in zip(spec.seeds.model_seeds, identities, strict=True):
            results.append(
                _run_f0_seed(
                    spec=spec,
                    fixture=fixture,
                    identity=identity,
                    seed=seed,
                    device=device,
                    output_root=Path(output_root),
                    preregistration_text=preregistration_text,
                    code_manifest=code_manifest,
                    code_hash=code_hash,
                    compiler_manifest=compiler_manifest,
                    feasibility=feasibility,
                    baseline=baseline,
                    training=training,
                    execution=execution,
                    environment=environment,
                    environment_payload=environment_payload,
                    command=recorded_command,
                    formal_authorization=authorization_context,
                )
            )
        aggregate = _write_experiment_aggregate(
            output_root=Path(output_root),
            spec=spec,
            results=results,
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
    return ExperimentRunResult(
        experiment_id=spec.experiment_id,
        stage=spec.stage,
        seed_results=tuple(results),
        aggregate=aggregate,
    )


__all__ = [
    "ExperimentAggregateArtifacts",
    "ExperimentRunResult",
    "FitExperimentError",
    "SeedRunResult",
    "assess_fixture_feasibility",
    "compiler_binding_manifest",
    "fixture_nw_targets",
    "load_fit_experiment",
    "run_fit_experiment",
    "source_tree_hash",
    "source_tree_manifest",
    "trivial_baseline",
    "validate_f0_binding",
]
