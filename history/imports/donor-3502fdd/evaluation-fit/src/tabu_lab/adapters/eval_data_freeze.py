"""Deterministic, offline candidate freezes for Evaluation v0 source data.

This module closes the gap between upstream retained bytes and
``EvalDataPreparationRequest``.  It does not download anything and it cannot
create a reviewed dataset authority.  Every emitted freeze is permanently
marked ``self_consistent_unreviewed`` and ``publication_eligible = false``.

The exporter is intentionally narrow.  It accepts only the source bytes pinned
below, emits a canonical retained representation, constructs exhaustive split
or validation-carve authorities, and validates each request through the live
real-data materializer before writing anything.  Adult additionally requires
an explicit OpenML fold, row-id semantics confirmation, and retained license
evidence; those decisions cannot be inferred from the ARFF bytes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import canonical_hash, require_sha256, to_canonical_data
from tabu_lab.evaluation.foundry import TaskKind, load_suite
from tabu_lab.evidence.schemas import EvidenceSchema

from .eval_data_workflow import EvalDataPreparationRequest
from .real_eval_data import (
    ColumnAuthority,
    CompletionMaskAuthority,
    DelimitedTableAuthority,
    GraphPerturbationAuthority,
    KarateAuthority,
    MovieLensAuthority,
    SplitAuthority,
    materialize_karate,
    materialize_movielens,
    materialize_table_completion,
    materialize_table_supervised,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]

FREEZE_SCHEMA = "tabu.eval-data-authority-freeze.v1"
SPLIT_RANK_SCHEMA = "tabu.eval-authority-split-rank.v1"
MOVIELENS_VALIDATION_SCHEMA = "tabu.eval-movielens-validation-rank.v1"
ADULT_ROWID_SEMANTICS = "openml-task-rowid-zero-based-arff-data-order-v1"
_MAX_INPUT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class _PinnedSource:
    retained_name: str
    sha256: str
    size_bytes: int
    media_type: str


_PINS: dict[str, _PinnedSource] = {
    "adult_data": _PinnedSource(
        retained_name="openml-adult-v2-1590.arff",
        sha256="77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0",
        size_bytes=5_962_724,
        media_type="text/x-arff",
    ),
    "adult_splits": _PinnedSource(
        retained_name="openml-task-7592-splits.arff",
        sha256="dac4caf27b44e897f40a4c63c205f4748729db59b6575d4fdfc07d8fa9ebd437",
        size_bytes=7_654_912,
        media_type="text/x-arff",
    ),
    "diabetes_data": _PinnedSource(
        retained_name="diabetes_data_raw.csv.gz",
        sha256="7fc0ded571454b1982210d3bb43f0aca44eae01a0b8654a3b24022bdb6b38009",
        size_bytes=7_073,
        media_type="application/gzip",
    ),
    "diabetes_target": _PinnedSource(
        retained_name="diabetes_target.csv.gz",
        sha256="8e53f65eb811df43c206f3534bb3af0e5fed213bc37ed6ba36310157d6023803",
        size_bytes=1_050,
        media_type="application/gzip",
    ),
    "movielens": _PinnedSource(
        retained_name="ml-100k.zip",
        sha256="50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229",
        size_bytes=4_924_029,
        media_type="application/zip",
    ),
    "networkx_social": _PinnedSource(
        retained_name="networkx-3.6.1/networkx/generators/social.py",
        sha256="6ebcc049ca40c5619113113d27b16e889dbd419c419bc6d7a518027342281cfd",
        size_bytes=23_416,
        media_type="text/x-python",
    ),
}

_EXPORTER_SHA256 = canonical_hash(
    {
        "schema_version": FREEZE_SCHEMA,
        "source_pins": {
            key: {
                "retained_name": value.retained_name,
                "sha256": value.sha256,
                "size_bytes": value.size_bytes,
                "media_type": value.media_type,
            }
            for key, value in sorted(_PINS.items())
        },
        "table_serialization": "utf8-csv-rfc4180-lf-v1",
        "split_rank": SPLIT_RANK_SCHEMA,
        "movielens_validation_rank": MOVIELENS_VALIDATION_SCHEMA,
        "karate_features": "unweighted-degree-v1",
        "adult_rowid_semantics": ADULT_ROWID_SEMANTICS,
    }
)


class EvalDataFreezeError(ValueError):
    """Retained bytes or choices cannot produce an unreviewed candidate freeze."""


class FreezeSourceRef(EvidenceSchema):
    """Path-free identity of one source consumed by the exporter."""

    schema_version: Literal["tabu.eval-data-freeze-source.v1"] = (
        "tabu.eval-data-freeze-source.v1"
    )
    role: Identifier
    retained_name: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0, le=_MAX_INPUT_BYTES)
    media_type: str = Field(min_length=1)
    provenance_status: Literal["retained_candidate_unreviewed"] = (
        "retained_candidate_unreviewed"
    )

    @field_validator("sha256")
    @classmethod
    def _valid_sha256(cls, value: str) -> str:
        return require_sha256(value, field_name="sha256")

    @field_validator("retained_name")
    @classmethod
    def _path_free_name(cls, value: str) -> str:
        if (
            value.startswith(("/", "\\"))
            or ".." in PurePosixPath(value).parts
            or "\\" in value
            or ":" in value
        ):
            raise ValueError("freeze source retained_name must be a public relative label")
        return value


class FreezeOutputRef(EvidenceSchema):
    """Content identity of one create-once file in the private freeze bundle."""

    schema_version: Literal["tabu.eval-data-freeze-output.v1"] = (
        "tabu.eval-data-freeze-output.v1"
    )
    role: Identifier
    relative_path: str = Field(min_length=1)
    sha256: Sha256
    size_bytes: int = Field(gt=0, le=_MAX_INPUT_BYTES)
    media_type: str = Field(min_length=1)
    content_sha256: Sha256 | None = None

    @field_validator("sha256", "content_sha256")
    @classmethod
    def _valid_hash(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        return require_sha256(value, field_name=getattr(info, "field_name", "sha256"))

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or value.endswith("/")
            or "\\" in value
            or ":" in value
        ):
            raise ValueError("freeze output must be a safe relative file path")
        return path.as_posix()


class EvalDataAuthorityFreezeManifest(EvidenceSchema):
    """Review subject for one offline, explicitly unreviewed data freeze."""

    schema_version: Literal["tabu.eval-data-authority-freeze.v1"] = FREEZE_SCHEMA
    freeze_id: str = Field(pattern=r"^eval-data-freeze-[0-9a-f]{64}$")
    dataset_id: Identifier
    source_version: str = Field(min_length=1)
    authority_status: Literal["self_consistent_unreviewed"] = (
        "self_consistent_unreviewed"
    )
    publication_eligible: Literal[False] = False
    review_ids: tuple[str, ...] = ()
    network_access: Literal[False] = False
    exporter_sha256: Sha256 = _EXPORTER_SHA256
    source_inputs: tuple[FreezeSourceRef, ...] = Field(min_length=1)
    decisions: dict[str, JsonValue]
    outputs: tuple[FreezeOutputRef, ...] = Field(min_length=2)
    blockers: tuple[str, ...] = ()

    @field_validator("exporter_sha256")
    @classmethod
    def _valid_exporter_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="exporter_sha256")

    @model_validator(mode="after")
    def _candidate_is_closed(self) -> EvalDataAuthorityFreezeManifest:
        if self.review_ids:
            raise ValueError("candidate freeze cannot carry review ids")
        if self.blockers:
            raise ValueError("completed candidate freeze cannot carry unresolved blockers")
        paths = [item.relative_path for item in self.outputs]
        roles = [item.role for item in self.outputs]
        source_roles = [item.role for item in self.source_inputs]
        if len(paths) != len(set(paths)) or len(roles) != len(set(roles)):
            raise ValueError("freeze outputs need unique paths and roles")
        if len(source_roles) != len(set(source_roles)):
            raise ValueError("freeze source inputs need unique roles")
        expected = _derive_freeze_id(
            dataset_id=self.dataset_id,
            source_version=self.source_version,
            source_inputs=self.source_inputs,
            decisions=self.decisions,
            outputs=self.outputs,
        )
        if self.freeze_id != expected:
            raise ValueError("freeze_id does not bind sources, decisions, and outputs")
        return self


@dataclass(frozen=True)
class EvalDataFreezeBundle:
    """In-memory files plus their self-verifying unreviewed manifest."""

    manifest: EvalDataAuthorityFreezeManifest
    files: Mapping[str, bytes]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            to_canonical_data(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _source_ref(*, role: str, pin: _PinnedSource) -> FreezeSourceRef:
    return FreezeSourceRef(
        role=role,
        retained_name=pin.retained_name,
        sha256=pin.sha256,
        size_bytes=pin.size_bytes,
        media_type=pin.media_type,
    )


def _unpinned_source_ref(
    *, role: str, retained_name: str, content: bytes, media_type: str
) -> FreezeSourceRef:
    return FreezeSourceRef(
        role=role,
        retained_name=retained_name,
        sha256=_sha256(content),
        size_bytes=len(content),
        media_type=media_type,
    )


def _derive_freeze_id(
    *,
    dataset_id: str,
    source_version: str,
    source_inputs: Sequence[FreezeSourceRef],
    decisions: Mapping[str, JsonValue],
    outputs: Sequence[FreezeOutputRef],
) -> str:
    digest = canonical_hash(
        {
            "schema_version": FREEZE_SCHEMA,
            "authority_status": "self_consistent_unreviewed",
            "publication_eligible": False,
            "network_access": False,
            "exporter_sha256": _EXPORTER_SHA256,
            "dataset_id": dataset_id,
            "source_version": source_version,
            "source_inputs": tuple(source_inputs),
            "decisions": dict(decisions),
            "outputs": tuple(outputs),
        }
    )
    return f"eval-data-freeze-{digest}"


def _read_local_file(path: str | os.PathLike[str], *, role: str) -> bytes:
    raw = os.fspath(path)
    if "://" in raw:
        raise EvalDataFreezeError(f"{role} must be retained local bytes; URLs are forbidden")
    source = Path(raw)
    try:
        metadata_value = source.lstat()
    except OSError as error:
        raise EvalDataFreezeError(f"{role} must name a regular local file") from error
    if not stat.S_ISREG(metadata_value.st_mode):
        raise EvalDataFreezeError(f"{role} must name a non-symlink regular local file")
    if metadata_value.st_size > _MAX_INPUT_BYTES:
        raise EvalDataFreezeError(f"{role} exceeds the offline size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise EvalDataFreezeError(f"{role} must name a non-symlink regular local file") from error
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_INPUT_BYTES:
            raise EvalDataFreezeError(f"{role} changed during safe open")
        content = stream.read(_MAX_INPUT_BYTES + 1)
    if not content or len(content) > _MAX_INPUT_BYTES:
        raise EvalDataFreezeError(f"{role} is empty or exceeds the offline size limit")
    return content


def _read_pinned(path: str | os.PathLike[str], *, role: str, pin: _PinnedSource) -> bytes:
    content = _read_local_file(path, role=role)
    if len(content) != pin.size_bytes or _sha256(content) != pin.sha256:
        raise EvalDataFreezeError(
            f"{role} differs from the frozen {pin.retained_name} size/SHA-256 pin"
        )
    return content


def _ranked_ids(
    row_ids: Iterable[str], *, dataset_id: str, seed: int, role: str
) -> tuple[str, ...]:
    if type(seed) is not int or seed < 0:
        raise EvalDataFreezeError("split seed must be a non-negative integer")
    values = tuple(row_ids)
    if not values or len(values) != len(set(values)):
        raise EvalDataFreezeError("split candidates must be non-empty unique row ids")
    return tuple(
        sorted(
            values,
            key=lambda row_id: (
                canonical_hash(
                    {
                        "schema_version": SPLIT_RANK_SCHEMA,
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "role": role,
                        "row_id": row_id,
                    }
                ),
                row_id,
            ),
        )
    )


def _split_exact_counts(
    *,
    row_ids: Sequence[str],
    dataset_id: str,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
) -> dict[str, tuple[str, ...]]:
    if min(train_count, validation_count, test_count) <= 0:
        raise EvalDataFreezeError("all exhaustive split counts must be positive")
    if train_count + validation_count + test_count != len(row_ids):
        raise EvalDataFreezeError("exhaustive split counts differ from retained row count")
    ranked = _ranked_ids(row_ids, dataset_id=dataset_id, seed=seed, role="partition")
    train_end = train_count
    validation_end = train_end + validation_count
    return {
        "train": tuple(sorted(ranked[:train_end], key=int)),
        "validation": tuple(sorted(ranked[train_end:validation_end], key=int)),
        "test": tuple(sorted(ranked[validation_end:], key=int)),
    }


def _csv_bytes(header: Sequence[str], rows: Iterable[Sequence[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _request_file(
    *, relative_path: str, role: str, request: EvalDataPreparationRequest
) -> tuple[FreezeOutputRef, bytes]:
    content = _canonical_json_bytes(request)
    return (
        FreezeOutputRef(
            role=role,
            relative_path=relative_path,
            sha256=_sha256(content),
            size_bytes=len(content),
            media_type="application/json",
            content_sha256=request.content_hash,
        ),
        content,
    )


def _retained_file(
    *, relative_path: str, role: str, content: bytes, media_type: str
) -> FreezeOutputRef:
    return FreezeOutputRef(
        role=role,
        relative_path=relative_path,
        sha256=_sha256(content),
        size_bytes=len(content),
        media_type=media_type,
    )


def _assemble_bundle(
    *,
    dataset_id: str,
    source_version: str,
    source_inputs: Sequence[FreezeSourceRef],
    decisions: Mapping[str, JsonValue],
    files: Mapping[str, tuple[FreezeOutputRef, bytes]],
) -> EvalDataFreezeBundle:
    outputs = tuple(files[path][0] for path in sorted(files))
    manifest = EvalDataAuthorityFreezeManifest(
        freeze_id=_derive_freeze_id(
            dataset_id=dataset_id,
            source_version=source_version,
            source_inputs=source_inputs,
            decisions=decisions,
            outputs=outputs,
        ),
        dataset_id=dataset_id,
        source_version=source_version,
        source_inputs=tuple(source_inputs),
        decisions=dict(decisions),
        outputs=outputs,
    )
    return EvalDataFreezeBundle(
        manifest=manifest,
        files={path: value[1] for path, value in sorted(files.items())},
    )


def _table_requests(
    *,
    dataset_id: str,
    source_content: bytes,
    partitions: Mapping[str, Sequence[str]],
    authority_id: str,
    feature_columns: Sequence[ColumnAuthority],
    response_column: ColumnAuthority,
    supervised_source_version: str,
    completion_source_version: str,
    supervised_suite_id: str,
    supervised_scenario_id: str,
    completion_suite_id: str,
    completion_scenario_id: str,
    mask_seed: int,
) -> tuple[EvalDataPreparationRequest, EvalDataPreparationRequest]:
    source_sha256 = _sha256(source_content)
    header = (
        "row_id",
        *(item.source_name for item in feature_columns),
        response_column.source_name,
    )

    def authority(source_version: str, suffix: str) -> DelimitedTableAuthority:
        return DelimitedTableAuthority(
            delimiter=",",
            field_whitespace="preserve",
            header=header,
            row_id_column="row_id",
            feature_columns=tuple(feature_columns),
            response_column=response_column,
            split=SplitAuthority(
                authority_id=f"{authority_id}-{suffix}",
                dataset_id=dataset_id,
                source_version=source_version,
                source_sha256=source_sha256,
                stable_id_kind="decimal_integer",
                partitions={key: tuple(value) for key, value in partitions.items()},
            ),
        )

    supervised_authority = authority(supervised_source_version, "supervised")
    completion_authority = authority(completion_source_version, "completion")
    supervised = EvalDataPreparationRequest(
        suite_id=supervised_suite_id,
        scenario_id=supervised_scenario_id,
        source_sha256=source_sha256,
        source_size_bytes=len(source_content),
        source_media_type="text/csv",
        authority=supervised_authority,
    )
    completion = EvalDataPreparationRequest(
        suite_id=completion_suite_id,
        scenario_id=completion_scenario_id,
        source_sha256=source_sha256,
        source_size_bytes=len(source_content),
        source_media_type="text/csv",
        authority=completion_authority,
        completion_mask_authority=CompletionMaskAuthority(mask_seed=mask_seed),
    )
    _validate_request_materializes(supervised, source_content)
    _validate_request_materializes(completion, source_content)
    return supervised, completion


def _validate_request_materializes(
    request: EvalDataPreparationRequest, source_content: bytes
) -> None:
    suite = load_suite(request.suite_id)
    scenarios = [item for item in suite.scenarios if item.scenario_id == request.scenario_id]
    if len(scenarios) != 1:
        raise EvalDataFreezeError("candidate request is absent from the live frozen suite")
    scenario = scenarios[0]
    authority = request.authority
    if isinstance(authority, DelimitedTableAuthority):
        if scenario.task is TaskKind.TABLE_COMPLETION:
            if request.completion_mask_authority is None:
                raise EvalDataFreezeError("completion request lacks its mask authority")
            materialize_table_completion(
                scenario=scenario,
                source=source_content,
                authority=authority,
                mask_authority=request.completion_mask_authority,
            )
        else:
            materialize_table_supervised(
                scenario=scenario,
                source=source_content,
                authority=authority,
            )
    elif isinstance(authority, KarateAuthority):
        materialize_karate(scenario=scenario, source=source_content, authority=authority)
    elif isinstance(authority, MovieLensAuthority):
        materialize_movielens(scenario=scenario, source=source_content, authority=authority)
    else:  # pragma: no cover - discriminated request schema excludes this branch
        raise EvalDataFreezeError("candidate request has an unsupported authority")


def build_diabetes_freeze(
    *,
    data_source: str | os.PathLike[str],
    target_source: str | os.PathLike[str],
    split_seed: int,
    mask_seed: int,
    suite_version: Literal["v0", "v1"] = "v0",
) -> EvalDataFreezeBundle:
    """Export pinned Diabetes arrays into one canonical CSV candidate freeze.

    The default retains the historical v0 suite identity.  Base 0.2.0 callers
    must opt into v1 explicitly so old candidate snapshots are not rewritten.
    """

    data_pin = _PINS["diabetes_data"]
    target_pin = _PINS["diabetes_target"]
    data_bytes = _read_pinned(data_source, role="diabetes data", pin=data_pin)
    target_bytes = _read_pinned(target_source, role="diabetes target", pin=target_pin)
    try:
        data_lines = gzip.decompress(data_bytes).decode("ascii", errors="strict").splitlines()
        target_lines = gzip.decompress(target_bytes).decode("ascii", errors="strict").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise EvalDataFreezeError(
            "pinned Diabetes gzip inputs are not strict ASCII tables"
        ) from error
    if len(data_lines) != 442 or len(target_lines) != 442:
        raise EvalDataFreezeError("Diabetes retained arrays must contain exactly 442 rows")
    rows: list[tuple[str, ...]] = []
    for row_id, (data_line, target_line) in enumerate(zip(data_lines, target_lines, strict=True)):
        features = tuple(data_line.split())
        if len(features) != 10:
            raise EvalDataFreezeError("Diabetes data row must contain exactly ten features")
        values = (*features, target_line.strip())
        try:
            parsed = tuple(float(value) for value in values)
        except ValueError as error:
            raise EvalDataFreezeError(
                "Diabetes retained arrays contain non-numeric data"
            ) from error
        if not all(math.isfinite(value) for value in parsed):
            raise EvalDataFreezeError("Diabetes retained arrays contain non-finite data")
        rows.append((str(row_id), *values))
    feature_names = tuple(f"x{index}" for index in range(10))
    retained = _csv_bytes(("row_id", *feature_names, "outcome"), rows)
    partitions = _split_exact_counts(
        row_ids=tuple(str(value) for value in range(442)),
        dataset_id="sklearn-diabetes",
        seed=split_seed,
        train_count=256,
        validation_count=64,
        test_count=122,
    )
    supervised, completion = _table_requests(
        dataset_id="sklearn-diabetes",
        source_content=retained,
        partitions=partitions,
        authority_id=f"sklearn-diabetes-1.9.0-candidate-seed-{split_seed}",
        feature_columns=tuple(
            ColumnAuthority(source_name=name, family_id=name, kind="numeric")
            for name in feature_names
        ),
        response_column=ColumnAuthority(
            source_name="outcome", family_id="outcome", kind="numeric"
        ),
        supervised_source_version="scikit-learn-1.x-bundled-snapshot",
        completion_source_version="scikit-learn-1.x-bundled-feature-table",
        supervised_suite_id=(
            "table-supervised-micro-v1"
            if suite_version == "v1"
            else "table-supervised-micro-v0"
        ),
        supervised_scenario_id=(
            "sklearn-diabetes-regression-micro-base"
            if suite_version == "v1"
            else "sklearn-diabetes-regression-micro"
        ),
        completion_suite_id=(
            "table-completion-micro-v1"
            if suite_version == "v1"
            else "table-completion-micro-v0"
        ),
        completion_scenario_id=(
            "sklearn-diabetes-feature-completion-micro-base"
            if suite_version == "v1"
            else "sklearn-diabetes-feature-completion-micro"
        ),
        mask_seed=mask_seed,
    )
    files: dict[str, tuple[FreezeOutputRef, bytes]] = {}
    retained_path = "retained/sklearn-diabetes-1.9.0-raw.csv"
    files[retained_path] = (
        _retained_file(
            relative_path=retained_path,
            role="canonical_retained_source",
            content=retained,
            media_type="text/csv",
        ),
        retained,
    )
    for path, role, request in (
        (
            "requests/diabetes-supervised.request.json",
            "supervised_request",
            supervised,
        ),
        (
            "requests/diabetes-completion.request.json",
            "completion_request",
            completion,
        ),
    ):
        reference, content = _request_file(relative_path=path, role=role, request=request)
        files[path] = (reference, content)
    return _assemble_bundle(
        dataset_id="sklearn-diabetes",
        source_version="scikit-learn-1.9.0-bundled-raw-files",
        source_inputs=(
            _source_ref(role="raw_feature_array", pin=data_pin),
            _source_ref(role="target_array", pin=target_pin),
        ),
        decisions={
            "representation": "raw_unscaled_10_feature_csv_with_zero_based_row_id_v1",
            "source_package_exact_version": "scikit-learn-1.9.0",
            "suite_compatibility_source_versions": [
                "scikit-learn-1.x-bundled-snapshot",
                "scikit-learn-1.x-bundled-feature-table",
            ],
            "version_binding_boundary": (
                "outer_freeze_pins_exact_1.9.0_inputs_requests_retain_frozen_suite_ids"
            ),
            "split_recipe": SPLIT_RANK_SCHEMA,
            "split_seed": split_seed,
            "partition_counts": {"train": 256, "validation": 64, "test": 122},
            "completion_mask_seed": mask_seed,
        },
        files=files,
    )


_ARFF_ATTRIBUTE = re.compile(r"^@attribute\s+([^\s]+)\s+(.+)$", re.IGNORECASE)


def _parse_arff(
    content: bytes, *, role: str
) -> tuple[tuple[tuple[str, str], ...], list[list[str]]]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise EvalDataFreezeError(f"{role} ARFF must be strict UTF-8") from error
    attributes: list[tuple[str, str]] = []
    data: list[list[str]] = []
    in_data = False
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            if line.lower() == "@data":
                in_data = True
                continue
            match = _ARFF_ATTRIBUTE.match(line)
            if match is not None:
                attributes.append((match.group(1), match.group(2).strip()))
                continue
            if line.lower().startswith("@relation"):
                continue
            raise EvalDataFreezeError(f"{role} ARFF has unsupported header line {line_number}")
        try:
            row = next(csv.reader([raw_line], skipinitialspace=True, strict=True))
        except csv.Error as error:
            raise EvalDataFreezeError(f"{role} ARFF has malformed row {line_number}") from error
        values = [value.strip() for value in row]
        if len(values) != len(attributes):
            raise EvalDataFreezeError(f"{role} ARFF row width differs from its attributes")
        data.append(values)
    if not in_data or not attributes or not data:
        raise EvalDataFreezeError(f"{role} ARFF lacks attributes or data")
    return tuple(attributes), data


_ADULT_COLUMNS: tuple[tuple[str, Literal["numeric", "categorical"]], ...] = (
    ("age", "numeric"),
    ("workclass", "categorical"),
    ("fnlwgt", "numeric"),
    ("education", "categorical"),
    ("education-num", "numeric"),
    ("marital-status", "categorical"),
    ("occupation", "categorical"),
    ("relationship", "categorical"),
    ("race", "categorical"),
    ("sex", "categorical"),
    ("capital-gain", "numeric"),
    ("capital-loss", "numeric"),
    ("hours-per-week", "numeric"),
    ("native-country", "categorical"),
    ("class", "categorical"),
)


def _adult_rows(content: bytes) -> tuple[bytes, tuple[str, ...]]:
    attributes, raw_rows = _parse_arff(content, role="Adult data")
    if tuple(name for name, _ in attributes) != tuple(name for name, _ in _ADULT_COLUMNS):
        raise EvalDataFreezeError("Adult ARFF attributes differ from the v2 retained contract")
    for (name, arff_kind), (_, expected_kind) in zip(attributes, _ADULT_COLUMNS, strict=True):
        observed_kind = "numeric" if arff_kind.lower() == "numeric" else "categorical"
        if observed_kind != expected_kind:
            raise EvalDataFreezeError(f"Adult ARFF attribute kind drifted for {name!r}")
    if len(raw_rows) != 48_842:
        raise EvalDataFreezeError("Adult v2 retained ARFF must contain exactly 48,842 rows")
    output_rows = [(str(index), *row) for index, row in enumerate(raw_rows)]
    retained = _csv_bytes(("row_id", *(name for name, _ in _ADULT_COLUMNS)), output_rows)
    return retained, tuple(str(value) for value in range(len(raw_rows)))


def _adult_task_fold(
    content: bytes, *, fold: int, row_ids: Sequence[str], validation_seed: int
) -> dict[str, tuple[str, ...]]:
    if type(fold) is not int or fold < 0 or fold > 9:
        raise EvalDataFreezeError("Adult OpenML task fold must be an explicit integer in 0..9")
    attributes, rows = _parse_arff(content, role="OpenML task 7592 split")
    if tuple(name.lower() for name, _ in attributes) != ("type", "rowid", "repeat", "fold"):
        raise EvalDataFreezeError("OpenML task 7592 split ARFF schema drifted")
    by_fold: dict[int, dict[str, set[str]]] = {
        value: {"TRAIN": set(), "TEST": set()} for value in range(10)
    }
    for row in rows:
        split_type, row_id, repeat_text, fold_text = row
        if split_type not in {"TRAIN", "TEST"}:
            raise EvalDataFreezeError("OpenML task split type must be TRAIN or TEST")
        if not row_id.isdigit() or not repeat_text.isdigit() or not fold_text.isdigit():
            raise EvalDataFreezeError("OpenML task split ids must be non-negative integers")
        repeat = int(repeat_text)
        fold_value = int(fold_text)
        canonical_row_id = str(int(row_id))
        if repeat != 0 or fold_value not in by_fold or canonical_row_id != row_id:
            raise EvalDataFreezeError("OpenML task split repeat/fold/rowid semantics drifted")
        target = by_fold[fold_value][split_type]
        if canonical_row_id in target:
            raise EvalDataFreezeError("OpenML task split contains a duplicate assignment")
        target.add(canonical_row_id)
    expected_ids = set(row_ids)
    for fold_value, assignment in by_fold.items():
        train = assignment["TRAIN"]
        test = assignment["TEST"]
        if train & test or train | test != expected_ids:
            raise EvalDataFreezeError(
                f"OpenML task fold {fold_value} is not an exhaustive disjoint assignment"
            )
    official_train = by_fold[fold]["TRAIN"]
    official_test = by_fold[fold]["TEST"]
    validation_rank = _ranked_ids(
        official_train,
        dataset_id="openml-adult-v2-task-7592",
        seed=validation_seed,
        role=f"fold-{fold}-train-side-validation",
    )
    validation_count = len(official_test)
    validation = set(validation_rank[:validation_count])
    train = official_train - validation
    return {
        "train": tuple(sorted(train, key=int)),
        "validation": tuple(sorted(validation, key=int)),
        "test": tuple(sorted(official_test, key=int)),
    }


def build_adult_freeze(
    *,
    data_source: str | os.PathLike[str],
    task_split_source: str | os.PathLike[str],
    license_evidence: str | os.PathLike[str] | None,
    fold: int | None,
    rowid_semantics: str | None,
    validation_seed: int,
    mask_seed: int,
    suite_version: Literal["v0", "v1"] = "v0",
) -> EvalDataFreezeBundle:
    """Build an Adult candidate after all non-byte OpenML choices are explicit.

    The default retains the historical v0 suite identity.  Base 0.2.0 callers
    must opt into v1 explicitly so old candidate snapshots are not rewritten.
    """

    missing: list[str] = []
    if fold is None:
        missing.append("explicit OpenML task fold 0..9")
    if rowid_semantics != ADULT_ROWID_SEMANTICS:
        missing.append(f"row-id semantics confirmation {ADULT_ROWID_SEMANTICS!r}")
    if license_evidence is None:
        missing.append("retained OpenML/Adult license evidence")
    if missing:
        raise EvalDataFreezeError(
            "Adult authority freeze is blocked; missing " + "; ".join(missing)
        )
    assert fold is not None and license_evidence is not None
    data_pin = _PINS["adult_data"]
    split_pin = _PINS["adult_splits"]
    data_bytes = _read_pinned(data_source, role="Adult data", pin=data_pin)
    split_bytes = _read_pinned(
        task_split_source, role="OpenML task 7592 split", pin=split_pin
    )
    license_bytes = _read_local_file(license_evidence, role="Adult license evidence")
    retained, row_ids = _adult_rows(data_bytes)
    partitions = _adult_task_fold(
        split_bytes,
        fold=fold,
        row_ids=row_ids,
        validation_seed=validation_seed,
    )
    features = tuple(
        ColumnAuthority(
            source_name=name,
            family_id=name,
            kind=kind,
            missing_tokens=("?",),
        )
        for name, kind in _ADULT_COLUMNS[:-1]
    )
    response = ColumnAuthority(
        source_name="class",
        family_id="class",
        kind="categorical",
    )
    supervised, completion = _table_requests(
        dataset_id="openml-adult-v2-task-7592",
        source_content=retained,
        partitions=partitions,
        authority_id=(
            f"openml-adult-v2-task-7592-fold-{fold}-candidate-seed-{validation_seed}"
        ),
        feature_columns=features,
        response_column=response,
        supervised_source_version="OpenML-task-7592",
        completion_source_version="OpenML-task-7592-feature-table",
        supervised_suite_id=(
            "table-supervised-micro-v1"
            if suite_version == "v1"
            else "table-supervised-micro-v0"
        ),
        supervised_scenario_id=(
            "adult-v2-task-7592-classification-micro-base"
            if suite_version == "v1"
            else "adult-v2-task-7592-classification-micro"
        ),
        completion_suite_id=(
            "table-completion-micro-v1"
            if suite_version == "v1"
            else "table-completion-micro-v0"
        ),
        completion_scenario_id=(
            "adult-v2-feature-completion-micro-base"
            if suite_version == "v1"
            else "adult-v2-feature-completion-micro"
        ),
        mask_seed=mask_seed,
    )
    files: dict[str, tuple[FreezeOutputRef, bytes]] = {}
    retained_path = "retained/openml-adult-v2-task-7592.csv"
    files[retained_path] = (
        _retained_file(
            relative_path=retained_path,
            role="canonical_retained_source",
            content=retained,
            media_type="text/csv",
        ),
        retained,
    )
    for path, role, request in (
        ("requests/adult-supervised.request.json", "supervised_request", supervised),
        ("requests/adult-completion.request.json", "completion_request", completion),
    ):
        reference, content = _request_file(relative_path=path, role=role, request=request)
        files[path] = (reference, content)
    license_output_path = "evidence/adult-license-evidence.bin"
    files[license_output_path] = (
        _retained_file(
            relative_path=license_output_path,
            role="license_evidence_candidate",
            content=license_bytes,
            media_type="application/octet-stream",
        ),
        license_bytes,
    )
    license_name = Path(os.fspath(license_evidence)).name
    return _assemble_bundle(
        dataset_id="openml-adult-v2-task-7592",
        source_version="OpenML-task-7592-fold-candidate-v1",
        source_inputs=(
            _source_ref(role="adult_v2_data_arff", pin=data_pin),
            _source_ref(role="task_7592_split_arff", pin=split_pin),
            _unpinned_source_ref(
                role="license_evidence_candidate",
                retained_name=license_name,
                content=license_bytes,
                media_type="application/octet-stream",
            ),
        ),
        decisions={
            "openml_task_fold": fold,
            "rowid_semantics": ADULT_ROWID_SEMANTICS,
            "validation_recipe": "hash-rank-official-train-match-test-count-v1",
            "validation_seed": validation_seed,
            "partition_counts": {key: len(value) for key, value in partitions.items()},
            "completion_mask_seed": mask_seed,
            "license_claim_status": "candidate_evidence_bound_unreviewed",
            "license_evidence_output": license_output_path,
        },
        files=files,
    )


def _networkx_source() -> tuple[bytes, object]:
    try:
        import networkx as nx
        import networkx.generators.social as social
    except ImportError as error:  # pragma: no cover - torch currently supplies networkx
        raise EvalDataFreezeError(
            "Karate freeze requires the lock-resolved networkx package"
        ) from error
    version = metadata.version("networkx")
    if version != "3.6.1":
        raise EvalDataFreezeError("Karate freeze requires exact networkx 3.6.1")
    source_path = Path(social.__file__ or "")
    source_bytes = _read_pinned(
        source_path,
        role="networkx social generator source",
        pin=_PINS["networkx_social"],
    )
    return source_bytes, nx.karate_club_graph()


def build_karate_freeze(*, split_seed: int) -> EvalDataFreezeBundle:
    """Export NetworkX 3.6.1 Karate topology with an explicit degree feature."""

    _, graph = _networkx_source()
    node_ids = tuple(str(value) for value in sorted(graph.nodes))
    if node_ids != tuple(str(value) for value in range(34)) or graph.number_of_edges() != 78:
        raise EvalDataFreezeError("NetworkX Karate topology differs from the frozen 34/78 contract")
    edges = tuple(
        sorted(tuple(sorted((str(left), str(right)))) for left, right in graph.edges)
    )
    payload = {
        "schema_version": "tabu.karate-retained-json.v1",
        "dataset_id": "zachary-karate-club",
        "source_version": "networkx-3.x-frozen-topology-contract",
        "nodes": [
            {
                "node_id": node_id,
                "club": graph.nodes[int(node_id)]["club"],
                "features": {"degree": graph.degree[int(node_id)]},
            }
            for node_id in node_ids
        ],
        "edges": [list(edge) for edge in edges],
    }
    retained = _canonical_json_bytes(payload)
    partitions = _split_exact_counts(
        row_ids=node_ids,
        dataset_id="zachary-karate-club",
        seed=split_seed,
        train_count=20,
        validation_count=7,
        test_count=7,
    )
    train_ids = tuple(partitions["train"])
    test_ids = tuple(partitions["test"])
    base_node_id = min(test_ids, key=int)
    topology_other = min(train_ids, key=int)
    locality_candidates = [item for item in node_ids if item != base_node_id]
    locality_edge = (locality_candidates[0], locality_candidates[1])
    topology_sha256 = canonical_hash(
        {
            "schema": "tabu.eval-karate-topology.v1",
            "node_ids": sorted(node_ids),
            "edges": edges,
        }
    )
    authority = KarateAuthority(
        split=SplitAuthority(
            authority_id=f"zachary-karate-networkx-3.6.1-candidate-seed-{split_seed}",
            dataset_id="zachary-karate-club",
            source_version="networkx-3.x-frozen-topology-contract",
            source_sha256=_sha256(retained),
            stable_id_kind="decimal_integer",
            partitions=partitions,
        ),
        feature_columns=(
            ColumnAuthority(source_name="degree", family_id="degree", kind="numeric"),
        ),
        club_domain=("Mr. Hi", "Officer"),
        topology_sha256=topology_sha256,
        perturbations=GraphPerturbationAuthority(
            base_node_id=base_node_id,
            topology_toggle_edge=(base_node_id, topology_other),
            locality_toggle_edge=locality_edge,
        ),
    )
    request = EvalDataPreparationRequest(
        suite_id="graph-completion-micro-v0",
        scenario_id="zachary-karate-club-label-completion",
        source_sha256=_sha256(retained),
        source_size_bytes=len(retained),
        source_media_type="application/json",
        authority=authority,
    )
    _validate_request_materializes(request, retained)
    retained_path = "retained/zachary-karate-networkx-3.6.1.json"
    request_path = "requests/karate.request.json"
    request_ref, request_bytes = _request_file(
        relative_path=request_path,
        role="graph_completion_request",
        request=request,
    )
    return _assemble_bundle(
        dataset_id="zachary-karate-club",
        source_version="networkx-3.6.1-karate-degree-candidate-v1",
        source_inputs=(
            _source_ref(role="networkx_social_generator", pin=_PINS["networkx_social"]),
        ),
        decisions={
            "node_feature_contract": "unweighted_degree_v1",
            "club_label_source": "networkx_node_attribute_club",
            "split_recipe": SPLIT_RANK_SCHEMA,
            "split_seed": split_seed,
            "partition_counts": {"train": 20, "validation": 7, "test": 7},
            "topology_perturbation": "smallest_test_node_to_smallest_train_node_toggle_v1",
            "locality_perturbation": "smallest_two_nonbase_nodes_toggle_v1",
        },
        files={
            retained_path: (
                _retained_file(
                    relative_path=retained_path,
                    role="canonical_retained_source",
                    content=retained,
                    media_type="application/json",
                ),
                retained,
            ),
            request_path: (request_ref, request_bytes),
        },
    )


def _movielens_base_interactions(content: bytes, *, member: str) -> tuple[str, ...]:
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            infos = [item for item in archive.infolist() if item.filename == member]
            if len(infos) != 1 or infos[0].flag_bits & 0x1:
                raise EvalDataFreezeError("MovieLens archive lacks one unencrypted u1.base")
            raw = archive.read(infos[0]).decode("ascii", errors="strict")
    except (zipfile.BadZipFile, UnicodeDecodeError) as error:
        raise EvalDataFreezeError("MovieLens source is not the expected ASCII ZIP") from error
    interaction_ids: list[str] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 4 or any(not item.isdigit() for item in fields):
            raise EvalDataFreezeError(f"MovieLens u1.base row {line_number} is malformed")
        user_id, item_id, rating, _ = fields
        if int(user_id) <= 0 or int(item_id) <= 0 or not 1 <= int(rating) <= 5:
            raise EvalDataFreezeError("MovieLens u1.base value lies outside the official domain")
        interaction_ids.append(f"{int(user_id)}:{int(item_id)}")
    if len(interaction_ids) != 80_000 or len(interaction_ids) != len(set(interaction_ids)):
        raise EvalDataFreezeError("MovieLens u1.base must contain 80,000 unique interactions")
    return tuple(interaction_ids)


def build_movielens_freeze(
    *,
    zip_source: str | os.PathLike[str],
    validation_seed: int,
    validation_count: int,
) -> EvalDataFreezeBundle:
    """Bind the official ZIP and deterministically carve validation from u1.base."""

    pin = _PINS["movielens"]
    retained = _read_pinned(zip_source, role="MovieLens-100K ZIP", pin=pin)
    interaction_ids = _movielens_base_interactions(retained, member="ml-100k/u1.base")
    if type(validation_count) is not int or not 1 <= validation_count < len(interaction_ids):
        raise EvalDataFreezeError("MovieLens validation count must lie in 1..79,999")
    ranked = tuple(
        sorted(
            interaction_ids,
            key=lambda interaction_id: (
                canonical_hash(
                    {
                        "schema_version": MOVIELENS_VALIDATION_SCHEMA,
                        "dataset_id": "movielens-100k",
                        "seed": validation_seed,
                        "interaction_id": interaction_id,
                    }
                ),
                tuple(int(value) for value in interaction_id.split(":")),
            ),
        )
    )
    validation_ids = tuple(
        sorted(
            ranked[:validation_count],
            key=lambda value: tuple(int(item) for item in value.split(":")),
        )
    )
    authority = MovieLensAuthority(
        source_sha256=pin.sha256,
        base_member="ml-100k/u1.base",
        test_member="ml-100k/u1.test",
        validation_interaction_ids=validation_ids,
    )
    request = EvalDataPreparationRequest(
        suite_id="recsys-completion-micro-v0",
        scenario_id="movielens-100k-interaction-completion",
        source_sha256=pin.sha256,
        source_size_bytes=len(retained),
        source_media_type="application/zip",
        authority=authority,
    )
    _validate_request_materializes(request, retained)
    retained_path = "retained/ml-100k.zip"
    request_path = "requests/movielens.request.json"
    request_ref, request_bytes = _request_file(
        relative_path=request_path,
        role="recsys_completion_request",
        request=request,
    )
    return _assemble_bundle(
        dataset_id="movielens-100k",
        source_version="MovieLens-100K-u1-candidate-v1",
        source_inputs=(_source_ref(role="official_zip", pin=pin),),
        decisions={
            "base_member": "ml-100k/u1.base",
            "test_member": "ml-100k/u1.test",
            "validation_recipe": MOVIELENS_VALIDATION_SCHEMA,
            "validation_seed": validation_seed,
            "validation_count": validation_count,
            "selection_after_validation": "independent_train_support_desc_stable_id_v1",
        },
        files={
            retained_path: (
                _retained_file(
                    relative_path=retained_path,
                    role="canonical_retained_source",
                    content=retained,
                    media_type="application/zip",
                ),
                retained,
            ),
            request_path: (request_ref, request_bytes),
        },
    )


def verify_freeze_bundle(bundle: EvalDataFreezeBundle) -> EvalDataAuthorityFreezeManifest:
    """Recompute every in-memory output reference and manifest invariant."""

    manifest = EvalDataAuthorityFreezeManifest.model_validate(
        bundle.manifest.model_dump(mode="python")
    )
    expected = {item.relative_path: item for item in manifest.outputs}
    if set(bundle.files) != set(expected):
        raise EvalDataFreezeError("freeze file set differs from its manifest outputs")
    for path, content in bundle.files.items():
        reference = expected[path]
        if _sha256(content) != reference.sha256 or len(content) != reference.size_bytes:
            raise EvalDataFreezeError(f"freeze output bytes drifted: {path}")
        if reference.content_sha256 is not None:
            try:
                payload = json.loads(content)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvalDataFreezeError(
                    f"request output is not canonical JSON: {path}"
                ) from error
            request = EvalDataPreparationRequest.model_validate(payload)
            if request.content_hash != reference.content_sha256:
                raise EvalDataFreezeError(f"request content hash drifted: {path}")
    return manifest


def _git_worktree_root(path: Path) -> Path | None:
    candidate = path.absolute()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.exists() and not candidate.is_dir():
        candidate = candidate.parent
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(candidate), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def _require_no_existing_symlink_component(path: Path) -> None:
    """Reject an output root whose existing lexical path traverses a symlink.

    The freeze contains retained labels and truth.  Resolving a caller path
    before the Git-ignore test would let an unignored link borrow the policy of
    its target; checking only the final component would still let an ancestor
    redirect later ``mkdir``/``open`` calls.  Inspect every existing component
    with ``lstat`` and repeat the check after directory creation.
    """

    lexical = Path(os.path.abspath(os.fspath(path)))
    anchor = Path(lexical.anchor)
    current = anchor
    for component in lexical.parts[1:]:
        current /= component
        try:
            metadata_value = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise EvalDataFreezeError(
                "cannot verify freeze output ancestor symlink safety"
            ) from error
        if stat.S_ISLNK(metadata_value.st_mode):
            raise EvalDataFreezeError(
                "freeze output root must not traverse an existing symlink component"
            )


def _require_private_freeze_root(path: Path) -> None:
    lexical = Path(os.path.abspath(os.fspath(path)))
    _require_no_existing_symlink_component(lexical)
    root = _git_worktree_root(lexical)
    if root is None:
        return
    try:
        relative = lexical.relative_to(root)
    except ValueError:
        return
    result = subprocess.run(
        ["git", "-C", os.fspath(root), "check-ignore", "--quiet", "--", relative.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvalDataFreezeError(
            "freeze bundle contains retained truth and must be Git-ignored inside a worktree"
        )


def write_freeze_bundle(
    bundle: EvalDataFreezeBundle, output_root: str | os.PathLike[str]
) -> Path:
    """Write one complete private bundle create-once; identical retries are idempotent."""

    manifest = verify_freeze_bundle(bundle)
    root = Path(output_root)
    _require_private_freeze_root(root)
    manifest_path = "freeze-manifest.json"
    all_files = {**bundle.files, manifest_path: _canonical_json_bytes(manifest)}
    targets = {relative: root / PurePosixPath(relative) for relative in all_files}
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise EvalDataFreezeError("freeze output root must be a non-symlink directory")
    for relative, target in targets.items():
        if target.exists() and (
            not target.is_file()
            or target.is_symlink()
            or target.read_bytes() != all_files[relative]
        ):
            raise FileExistsError(f"freeze output already differs: {target}")
    root.mkdir(parents=True, exist_ok=True)
    _require_no_existing_symlink_component(root)
    created: list[Path] = []
    try:
        for relative in sorted(all_files):
            target = targets[relative]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            descriptor = os.open(target, flags, 0o600)
            created.append(target)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(all_files[relative])
                stream.flush()
                os.fsync(stream.fileno())
    except Exception:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
    return root / manifest_path


def load_freeze_bundle(output_root: str | os.PathLike[str]) -> EvalDataFreezeBundle:
    """Load and byte-verify an existing private candidate freeze."""

    root = Path(output_root)
    _require_private_freeze_root(root)
    manifest_path = root / "freeze-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise EvalDataFreezeError("freeze manifest must be one regular local file")
    try:
        manifest = EvalDataAuthorityFreezeManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise EvalDataFreezeError("freeze manifest is invalid") from error
    files: dict[str, bytes] = {}
    for output in manifest.outputs:
        source = root / output.relative_path
        if not source.is_file() or source.is_symlink():
            raise EvalDataFreezeError(f"freeze output is missing or unsafe: {output.relative_path}")
        files[output.relative_path] = source.read_bytes()
    bundle = EvalDataFreezeBundle(manifest=manifest, files=files)
    verify_freeze_bundle(bundle)
    return bundle


def _summary(bundle: EvalDataFreezeBundle, manifest_path: Path) -> dict[str, object]:
    return {
        "authority_status": bundle.manifest.authority_status,
        "dataset_id": bundle.manifest.dataset_id,
        "freeze_id": bundle.manifest.freeze_id,
        "manifest": os.fspath(manifest_path),
        "network_access": bundle.manifest.network_access,
        "outputs": [item.model_dump(mode="json") for item in bundle.manifest.outputs],
        "publication_eligible": bundle.manifest.publication_eligible,
        "review_ids": list(bundle.manifest.review_ids),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freeze_eval_data_authority.py",
        description="Build offline self-consistent-unreviewed Evaluation v0 authority requests",
    )
    commands = parser.add_subparsers(dest="dataset", required=True)

    check = commands.add_parser("check")
    check.add_argument("--output-root", required=True)

    diabetes = commands.add_parser("diabetes")
    diabetes.add_argument("--data", required=True)
    diabetes.add_argument("--target", required=True)
    diabetes.add_argument("--split-seed", required=True, type=int)
    diabetes.add_argument("--mask-seed", required=True, type=int)
    diabetes.add_argument("--output-root", required=True)
    diabetes.add_argument("--suite-version", choices=("v0", "v1"), default="v0")

    adult = commands.add_parser("adult")
    adult.add_argument("--data-arff", required=True)
    adult.add_argument("--task-splits-arff", required=True)
    adult.add_argument("--license-evidence")
    adult.add_argument("--fold", type=int)
    adult.add_argument("--rowid-semantics")
    adult.add_argument("--validation-seed", required=True, type=int)
    adult.add_argument("--mask-seed", required=True, type=int)
    adult.add_argument("--output-root", required=True)
    adult.add_argument("--suite-version", choices=("v0", "v1"), default="v0")

    karate = commands.add_parser("karate")
    karate.add_argument("--split-seed", required=True, type=int)
    karate.add_argument("--output-root", required=True)

    movielens = commands.add_parser("movielens")
    movielens.add_argument("--zip", required=True, dest="zip_source")
    movielens.add_argument("--validation-seed", required=True, type=int)
    movielens.add_argument("--validation-count", required=True, type=int)
    movielens.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry point; failures write no authority bundle."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.dataset == "check":
            bundle = load_freeze_bundle(args.output_root)
            manifest_path = Path(args.output_root) / "freeze-manifest.json"
        elif args.dataset == "diabetes":
            bundle = build_diabetes_freeze(
                data_source=args.data,
                target_source=args.target,
                split_seed=args.split_seed,
                mask_seed=args.mask_seed,
                suite_version=args.suite_version,
            )
        elif args.dataset == "adult":
            bundle = build_adult_freeze(
                data_source=args.data_arff,
                task_split_source=args.task_splits_arff,
                license_evidence=args.license_evidence,
                fold=args.fold,
                rowid_semantics=args.rowid_semantics,
                validation_seed=args.validation_seed,
                mask_seed=args.mask_seed,
                suite_version=args.suite_version,
            )
        elif args.dataset == "karate":
            bundle = build_karate_freeze(split_seed=args.split_seed)
        else:
            bundle = build_movielens_freeze(
                zip_source=args.zip_source,
                validation_seed=args.validation_seed,
                validation_count=args.validation_count,
            )
        if args.dataset != "check":
            manifest_path = write_freeze_bundle(bundle, args.output_root)
    except (EvalDataFreezeError, FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(bundle, manifest_path), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "ADULT_ROWID_SEMANTICS",
    "EvalDataAuthorityFreezeManifest",
    "EvalDataFreezeBundle",
    "EvalDataFreezeError",
    "FreezeOutputRef",
    "FreezeSourceRef",
    "build_adult_freeze",
    "build_diabetes_freeze",
    "build_karate_freeze",
    "build_movielens_freeze",
    "load_freeze_bundle",
    "main",
    "verify_freeze_bundle",
    "write_freeze_bundle",
]
