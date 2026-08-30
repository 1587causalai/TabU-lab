"""Model contract registry for the public TabU-lab package.

The registry deliberately loads only small YAML manifests.  Importing this module must
not import torch or instantiate a model: validation and documentation tooling should be
usable in a minimal environment.  Runtime construction is delegated lazily to
``tabu_lab.models.build_model``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from importlib import import_module, resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from tabu_lab.mathspec import Mathematics


class BuildStatus(StrEnum):
    """Typed runtime outcome of a registry build request."""

    READY = "ready"
    DESIGN_OPEN = "design_open"
    BUILDER_UNAVAILABLE = "builder_unavailable"
    BUILD_ERROR = "build_error"


class ContractBuildState(StrEnum):
    """Build state asserted by a source contract, independent of installed code."""

    BUILDABLE_CONTRACT = "buildable_contract"
    DESIGN_OPEN = "design_open"


class MaturityStage(StrEnum):
    """Monotone public maturity vocabulary; transitions require explicit evidence."""

    DESIGN_OPEN = "design_open"
    SPECIFIED = "specified"
    EXPERIMENTAL = "experimental"
    CONTRACT_TESTED = "contract-tested"
    GATE0_SANITY = "gate0_sanity"
    GATE1_REPRODUCIBLE = "gate1_reproducible"
    SUPPORTED = "supported"
    EVIDENCE_BACKED = "evidence-backed"


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class UpstreamSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    readonly: Literal[True]
    observed_at: str = Field(min_length=1)
    license_boundary: str = Field(min_length=1)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        normalized = value.lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("sha256 must be exactly 64 lowercase hexadecimal characters")
        return normalized


class OpenItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    question: str = Field(min_length=1)
    blocking_build: bool = False


class Alternative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    description: str = Field(min_length=1)
    status: Literal["first_alternative", "deferred", "rejected"]


class Gate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    condition: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class Maturity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_contract: Literal["factory_source"]
    stage: MaturityStage
    contract: MaturityStage
    implementation: MaturityStage
    evidence: MaturityStage
    build_state: ContractBuildState

    @model_validator(mode="after")
    def design_open_is_explicit(self) -> Maturity:
        if self.build_state is ContractBuildState.DESIGN_OPEN:
            if self.stage is not MaturityStage.DESIGN_OPEN:
                raise ValueError("design_open build_state requires design_open maturity stage")
        elif self.stage is MaturityStage.DESIGN_OPEN:
            raise ValueError("a buildable contract cannot have design_open maturity stage")
        return self


class MaturityEvidence(BaseModel):
    """Immutable gate references required before public maturity promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate1_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    independent_review_report_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gong_approval_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    accepted_claim_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )

    @field_validator(
        "gate1_receipt_hash",
        "independent_review_report_hash",
        "gong_approval_hash",
        "accepted_claim_hash",
    )
    @classmethod
    def normalized_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("maturity evidence hashes must be lowercase SHA-256")
        return normalized


class ModelSpec(BaseModel):
    """Machine-readable boundary for one model-factory contract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "https://research.wehub.us/schemas/model-spec.schema.json",
        },
    )

    schema_version: Literal["1.0.0"]
    contract_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    contract_version: str = Field(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
    display_name: str = Field(min_length=1)
    upstream: UpstreamSource
    carrier: dict[str, Any] = Field(min_length=1)
    dynamics: dict[str, Any] = Field(min_length=1)
    readout: dict[str, Any] = Field(min_length=1)
    loss: dict[str, Any] = Field(default_factory=dict)
    interfaces: dict[str, Any] = Field(default_factory=dict)
    capabilities: tuple[str, ...] = Field(min_length=1)
    maturity: Maturity
    maturity_evidence: MaturityEvidence | None = None
    mathematics: Mathematics | None = None
    experimental_defaults: dict[str, Any] = Field(min_length=1)
    known_open: tuple[OpenItem, ...]
    alternatives: tuple[Alternative, ...]
    kill: tuple[Gate, ...] = Field(min_length=1)
    exit: tuple[Gate, ...] = Field(min_length=1)

    @field_validator("capabilities")
    @classmethod
    def unique_capabilities(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("capabilities must be unique")
        return values

    @model_validator(mode="after")
    def public_maturity_is_evidence_gated(self) -> ModelSpec:
        dimensions = (
            self.maturity.stage,
            self.maturity.contract,
            self.maturity.implementation,
            self.maturity.evidence,
        )
        gated = {MaturityStage.SUPPORTED, MaturityStage.EVIDENCE_BACKED}
        if (
            any(dimension in gated for dimension in dimensions)
            and self.maturity_evidence is None
        ):
            raise ValueError(
                "supported/evidence-backed maturity requires an immutable Gate 1 "
                "receipt, independent review report, and gong approval"
            )
        if (
            any(dimension is MaturityStage.EVIDENCE_BACKED for dimension in dimensions)
            and (
                self.maturity_evidence is None
                or self.maturity_evidence.accepted_claim_hash is None
            )
        ):
            raise ValueError("evidence-backed maturity requires an accepted claim hash")
        return self

    @property
    def model_id(self) -> str:
        """Compatibility alias for model builders."""

        return self.contract_id

    @property
    def id(self) -> str:
        """Compatibility alias for model builders."""

        return self.contract_id


class ValidationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: IssueSeverity
    code: str
    message: str
    contract_id: str | None = None
    path: str | None = None


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checked: tuple[str, ...]
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(issue.severity is IssueSeverity.ERROR for issue in self.issues)


@dataclass(frozen=True, slots=True)
class BuildResult:
    """A build request result; ``model`` is populated only for ``READY``."""

    status: BuildStatus
    contract_id: str
    spec: ModelSpec
    model: Any | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {BuildStatus.READY, BuildStatus.DESIGN_OPEN}

    @property
    def executable(self) -> bool:
        """Whether callers may pass the build result into training/inference.

        ``ok`` remains a compatibility predicate for registry inspection: a
        typed ``DESIGN_OPEN`` response is an expected, non-error outcome.  It
        must not, however, be treated as an executable model by generic
        runners.
        """

        return self.status is BuildStatus.READY and self.model is not None


class RegistryError(RuntimeError):
    """Base registry failure."""


class ModelNotFoundError(RegistryError, KeyError):
    def __init__(self, contract_id: str, available: Sequence[str]) -> None:
        self.contract_id = contract_id
        self.available = tuple(available)
        super().__init__(
            f"unknown model contract {contract_id!r}; available: {', '.join(self.available)}"
        )


class ModelVersionNotFoundError(RegistryError, KeyError):
    """An exact contract version is not present in the immutable registry history."""

    def __init__(
        self,
        contract_id: str,
        contract_version: str,
        available: Sequence[str],
    ) -> None:
        self.contract_id = contract_id
        self.contract_version = contract_version
        self.available = tuple(available)
        rendered = ", ".join(self.available) or "none"
        super().__init__(
            f"unknown version {contract_version!r} for model contract {contract_id!r}; "
            f"available versions: {rendered}"
        )


class RegistryValidationError(RegistryError):
    """A packaged manifest is not a valid ModelSpec."""


def _model_resource_dir() -> Any:
    return resources.files("tabu_lab.specs.models")


@dataclass(frozen=True, slots=True)
class _RegistrySnapshot:
    current: tuple[ModelSpec, ...]
    versions: tuple[ModelSpec, ...]


def _load_manifest(resource: Any, *, relative_path: str) -> tuple[ModelSpec, bytes]:
    try:
        payload = resource.read_bytes()
        raw = yaml.safe_load(payload.decode("utf-8"))
        return ModelSpec.model_validate(raw), payload
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValidationError) as exc:
        message = f"invalid packaged manifest {relative_path}: {exc}"
        raise RegistryValidationError(message) from exc


@lru_cache(maxsize=1)
def _load_registry() -> _RegistrySnapshot:
    current: list[ModelSpec] = []
    current_payloads: dict[tuple[str, str], bytes] = {}
    history: dict[tuple[str, str], ModelSpec] = {}
    history_payloads: dict[tuple[str, str], bytes] = {}
    root = _model_resource_dir()

    for resource in sorted(root.iterdir(), key=lambda item: item.name):
        if resource.is_file() and resource.name.endswith((".yaml", ".yml")):
            spec, payload = _load_manifest(resource, relative_path=resource.name)
            expected_id = resource.name.rsplit(".", maxsplit=1)[0]
            if spec.contract_id != expected_id:
                raise RegistryValidationError(
                    f"current manifest filename {resource.name!r} does not match "
                    f"contract_id {spec.contract_id!r}"
                )
            current.append(spec)
            current_payloads[(spec.contract_id, spec.contract_version)] = payload
            continue

        if not resource.is_dir() or resource.name.startswith((".", "__")):
            continue
        for version_resource in sorted(resource.iterdir(), key=lambda item: item.name):
            if not version_resource.is_file() or not version_resource.name.endswith(
                (".yaml", ".yml")
            ):
                continue
            relative_path = f"{resource.name}/{version_resource.name}"
            spec, payload = _load_manifest(version_resource, relative_path=relative_path)
            filename_version = version_resource.name.rsplit(".", maxsplit=1)[0]
            if spec.contract_id != resource.name:
                raise RegistryValidationError(
                    f"history directory {resource.name!r} does not match "
                    f"contract_id {spec.contract_id!r} in {relative_path}"
                )
            if spec.contract_version != filename_version:
                raise RegistryValidationError(
                    f"history filename version {filename_version!r} does not match "
                    f"contract_version {spec.contract_version!r} in {relative_path}"
                )
            key = (spec.contract_id, spec.contract_version)
            if key in history:
                raise RegistryValidationError(
                    f"duplicate historical model contract version: {spec.contract_id}@"
                    f"{spec.contract_version}"
                )
            history[key] = spec
            history_payloads[key] = payload

    if not current:
        raise RegistryValidationError("no packaged model manifests found")

    ids = [spec.contract_id for spec in current]
    duplicates = sorted({contract_id for contract_id in ids if ids.count(contract_id) > 1})
    if duplicates:
        raise RegistryValidationError(f"duplicate contract ids: {', '.join(duplicates)}")

    for key, payload in current_payloads.items():
        if key in history_payloads and history_payloads[key] != payload:
            contract_id, contract_version = key
            raise RegistryValidationError(
                "current alias and immutable history differ for "
                f"{contract_id}@{contract_version}"
            )

    versions = dict(history)
    for spec in current:
        versions.setdefault((spec.contract_id, spec.contract_version), spec)
    return _RegistrySnapshot(
        current=tuple(sorted(current, key=lambda spec: spec.contract_id)),
        versions=tuple(
            sorted(versions.values(), key=lambda spec: (spec.contract_id, spec.contract_version))
        ),
    )


def clear_registry_cache() -> None:
    """Clear the resource cache, primarily for tests and editable development."""

    _load_registry.cache_clear()


def list_models() -> tuple[ModelSpec, ...]:
    """Return all model contracts in stable contract-id order."""

    return _load_registry().current


def list_model_versions(contract_id: str) -> tuple[ModelSpec, ...]:
    """Return every archived/current version for one contract in stable version order."""

    current_ids = tuple(spec.contract_id for spec in list_models())
    if contract_id not in current_ids:
        raise ModelNotFoundError(contract_id, current_ids)
    return tuple(
        spec for spec in _load_registry().versions if spec.contract_id == contract_id
    )


def get_model_spec(contract_id: str, contract_version: str | None = None) -> ModelSpec:
    """Return the current alias or an exact immutable model contract version."""

    by_id = {spec.contract_id: spec for spec in list_models()}
    try:
        current = by_id[contract_id]
    except KeyError as exc:
        raise ModelNotFoundError(contract_id, tuple(by_id)) from exc
    if contract_version is None:
        return current

    versions = {spec.contract_version: spec for spec in list_model_versions(contract_id)}
    try:
        return versions[contract_version]
    except KeyError as exc:
        raise ModelVersionNotFoundError(
            contract_id,
            contract_version,
            tuple(versions),
        ) from exc


def validate_registry_source_parity(
    *,
    public_dir: Path | None = None,
    packaged_dir: Path | None = None,
) -> None:
    """Fail unless public and packaged ModelSpec trees have identical paths and bytes."""

    repository_root = Path(__file__).resolve().parents[2]
    public = public_dir or repository_root / "specs" / "models"
    packaged = packaged_dir or Path(__file__).resolve().parent / "specs" / "models"

    def manifests(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in {".yaml", ".yml"}
        }

    public_manifests = manifests(public)
    packaged_manifests = manifests(packaged)
    if public_manifests.keys() != packaged_manifests.keys():
        missing_public = sorted(packaged_manifests.keys() - public_manifests.keys())
        missing_packaged = sorted(public_manifests.keys() - packaged_manifests.keys())
        raise RegistryValidationError(
            "public/package ModelSpec path mismatch; "
            f"missing public={missing_public}, missing packaged={missing_packaged}"
        )
    mismatches = sorted(
        path
        for path, payload in public_manifests.items()
        if packaged_manifests[path] != payload
    )
    if mismatches:
        raise RegistryValidationError(
            f"public/package ModelSpec byte mismatch: {', '.join(mismatches)}"
        )


def _source_path(spec: ModelSpec, source_root: Path | None) -> Path:
    root = source_root if source_root is not None else Path(__file__).resolve().parents[2]
    return (root / spec.upstream.path).resolve()


def validate_model_spec(
    spec: ModelSpec | Mapping[str, Any],
    *,
    verify_upstream: bool = True,
    source_root: Path | None = None,
) -> ValidationReport:
    """Validate one manifest and, when locally available, its readonly source hash.

    A public wheel need not contain the private owner workspace.  A missing upstream
    source is therefore a warning, while a present source with the wrong hash is an
    error.  ``source_root`` is the directory against which ``upstream.path`` resolves.
    """

    try:
        validated = spec if isinstance(spec, ModelSpec) else ModelSpec.model_validate(spec)
    except ValidationError as exc:
        issue = ValidationIssue(
            severity=IssueSeverity.ERROR,
            code="invalid_model_spec",
            message=str(exc),
        )
        return ValidationReport(checked=(), issues=(issue,))

    issues: list[ValidationIssue] = []
    if validated.contract_id == "tabu4do":
        if validated.maturity.build_state is not ContractBuildState.DESIGN_OPEN:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="tabu4do_must_be_design_open",
                    message="TabU4Do cannot be marked buildable until its realization is frozen",
                    contract_id=validated.contract_id,
                )
            )
        if not any(item.blocking_build for item in validated.known_open):
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    code="tabu4do_missing_blocking_open",
                    message="TabU4Do must retain at least one build-blocking realization question",
                    contract_id=validated.contract_id,
                )
            )

    if verify_upstream:
        source = _source_path(validated, source_root)
        if not source.is_file():
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    code="upstream_source_unavailable",
                    message="readonly upstream source is not present in this checkout",
                    contract_id=validated.contract_id,
                    path=str(source),
                )
            )
        else:
            digest = sha256(source.read_bytes()).hexdigest()
            if digest != validated.upstream.sha256:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        code="upstream_hash_mismatch",
                        message=(
                            f"expected {validated.upstream.sha256}, observed {digest}; "
                            "refresh the manifest only after contract review"
                        ),
                        contract_id=validated.contract_id,
                        path=str(source),
                    )
                )

    return ValidationReport(checked=(validated.contract_id,), issues=tuple(issues))


def validate_registry(
    contract_id: str | None = None,
    *,
    verify_upstream: bool = True,
    source_root: Path | None = None,
) -> ValidationReport:
    """Validate one or all packaged contracts without importing model code."""

    specs = (get_model_spec(contract_id),) if contract_id is not None else list_models()
    issues: list[ValidationIssue] = []
    for spec in specs:
        report = validate_model_spec(
            spec,
            verify_upstream=verify_upstream,
            source_root=source_root,
        )
        issues.extend(report.issues)
    return ValidationReport(
        checked=tuple(spec.contract_id for spec in specs),
        issues=tuple(issues),
    )


def build_model(contract_id: str, **kwargs: Any) -> BuildResult:
    """Build a model lazily, preserving a typed boundary for unavailable designs."""

    spec = get_model_spec(contract_id)
    if spec.maturity.build_state is ContractBuildState.DESIGN_OPEN:
        blockers = "; ".join(item.question for item in spec.known_open if item.blocking_build)
        return BuildResult(
            status=BuildStatus.DESIGN_OPEN,
            contract_id=contract_id,
            spec=spec,
            detail=blockers or "the source contract is explicitly design-open",
        )

    try:
        models = import_module("tabu_lab.models")
        builder = models.build_model
    except (ModuleNotFoundError, AttributeError) as exc:
        return BuildResult(
            status=BuildStatus.BUILDER_UNAVAILABLE,
            contract_id=contract_id,
            spec=spec,
            detail=f"tabu_lab.models.build_model is unavailable: {exc}",
        )

    try:
        model = builder(contract_id, **kwargs)
    except Exception as exc:
        return BuildResult(
            status=BuildStatus.BUILD_ERROR,
            contract_id=contract_id,
            spec=spec,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return BuildResult(status=BuildStatus.READY, contract_id=contract_id, spec=spec, model=model)


def instantiate_model(contract_id: str, **kwargs: Any) -> BuildResult:
    """Explicit alias for callers that use instantiate terminology."""

    return build_model(contract_id, **kwargs)


# Small compatibility aliases keep the public boundary unsurprising without duplicating logic.
list_model_specs = list_models
show_model = get_model_spec
validate_models = validate_registry
build = build_model
instantiate = instantiate_model


__all__ = [
    "Alternative",
    "BuildResult",
    "BuildStatus",
    "ContractBuildState",
    "Gate",
    "IssueSeverity",
    "Maturity",
    "MaturityEvidence",
    "MaturityStage",
    "ModelNotFoundError",
    "ModelSpec",
    "ModelVersionNotFoundError",
    "OpenItem",
    "RegistryError",
    "RegistryValidationError",
    "UpstreamSource",
    "ValidationIssue",
    "ValidationReport",
    "build",
    "build_model",
    "clear_registry_cache",
    "get_model_spec",
    "instantiate",
    "instantiate_model",
    "list_model_specs",
    "list_model_versions",
    "list_models",
    "show_model",
    "validate_model_spec",
    "validate_models",
    "validate_registry",
    "validate_registry_source_parity",
]
