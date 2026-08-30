"""Offline, fail-closed materializers for the frozen table evaluation suites.

The suite YAML files intentionally name datasets and high-level selection rules,
but they do not authorize a downloader, a raw serialization, or a validation
carve.  This module therefore accepts retained bytes (or an explicit local
``Path``) plus a versioned, hash-bound authority manifest.  It never performs
network I/O and never guesses a split or source format.

Every materializer performs these operations in order:

1. bind the exact retained bytes;
2. validate an exhaustive split authority;
3. select rows/interactions inside each already-created partition;
4. fit statistics and categorical codebooks on selected train data only;
5. apply any artificial mask independently inside each partition; and
6. construct a :class:`PreparedScenario` whose test truth remains solely in
   the evaluator-owned sidecar fields.

``checkpoint_blind_example`` is the adapter projection boundary.  It constructs
the explicit, versioned ``EvidenceEpisode`` payload required by
``CatalogedCheckpointModelAdapter`` while physically zeroing the selected
held-out cell.  Only the returned ``BlindExample`` is model-facing.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import Field, JsonValue, field_validator, model_validator

from tabu_lab.contracts import (
    EvidenceEpisode,
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    ForwardRole,
    GraphTopology,
    OriginState,
    canonical_hash,
    require_sha256,
)
from tabu_lab.evaluation.foundry import (
    BlindExample,
    DatasetSnapshotBinding,
    PreparationContract,
    PreparedExample,
    PreparedScenario,
    ScenarioSpec,
    SourceMaterial,
    TargetKind,
    TaskKind,
    TopologyCheckCase,
)
from tabu_lab.evidence.schemas import EvidenceSchema

from .checkpoint_model import (
    EPISODE_PAYLOAD_KEY,
    EPISODE_PAYLOAD_SCHEMA,
    READOUT_SELECTOR_KEY,
    READOUT_SELECTOR_SCHEMA,
)

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PartitionName = Literal["train", "validation", "test"]
RawSource = bytes | bytearray | memoryview | Path

MATERIALIZER_SCHEMA = "tabu.real-eval-materializer.v1"
CHECKPOINT_PROJECTION_SCHEMA = "tabu.real-eval-checkpoint-projection.v1"
MASK_EXECUTION_SCHEMA = "tabu.eval-cell-target-enumeration.v2"
_IMPLEMENTATION_SHA256 = canonical_hash(
    {
        "schema": MATERIALIZER_SCHEMA,
        "split_before_preprocess": True,
        "selection": "sha256-rank-v1",
        "completion_mask": "sha256-row-cover-train-support-floor-v2",
        "target_enumeration": "all-masked-cells-v1",
        "movie_lens_selection": "independent-train-support-desc-stable-id-v1",
        "checkpoint_projection": CHECKPOINT_PROJECTION_SCHEMA,
    }
)
_PARTITIONS: tuple[PartitionName, ...] = ("train", "validation", "test")
_MAX_SOURCE_BYTES = 128 * 1024 * 1024

_SUPPORTED_SCENARIOS: dict[str, tuple[str, TaskKind]] = {
    "adult-v2-task-7592-classification-micro": (
        "openml-adult-v2-task-7592",
        TaskKind.SUPERVISED_CLASSIFICATION,
    ),
    "adult-v2-task-7592-classification-micro-base": (
        "openml-adult-v2-task-7592",
        TaskKind.SUPERVISED_CLASSIFICATION,
    ),
    "sklearn-diabetes-regression-micro": (
        "sklearn-diabetes",
        TaskKind.SUPERVISED_REGRESSION,
    ),
    "sklearn-diabetes-regression-micro-base": (
        "sklearn-diabetes",
        TaskKind.SUPERVISED_REGRESSION,
    ),
    "adult-v2-feature-completion-micro": (
        "openml-adult-v2-task-7592",
        TaskKind.TABLE_COMPLETION,
    ),
    "adult-v2-feature-completion-micro-base": (
        "openml-adult-v2-task-7592",
        TaskKind.TABLE_COMPLETION,
    ),
    "sklearn-diabetes-feature-completion-micro": (
        "sklearn-diabetes",
        TaskKind.TABLE_COMPLETION,
    ),
    "sklearn-diabetes-feature-completion-micro-base": (
        "sklearn-diabetes",
        TaskKind.TABLE_COMPLETION,
    ),
    "zachary-karate-club-label-completion": (
        "zachary-karate-club",
        TaskKind.GRAPH_COMPLETION,
    ),
    "movielens-100k-interaction-completion": (
        "movielens-100k",
        TaskKind.RECSYS_COMPLETION,
    ),
}


class RealEvalDataError(ValueError):
    """Retained data cannot satisfy the frozen evaluation contract."""


class SplitAuthority(EvidenceSchema):
    """Exhaustive caller-supplied partition authority bound to retained bytes."""

    schema_version: Literal["tabu.eval-split-authority.v1"] = (
        "tabu.eval-split-authority.v1"
    )
    authority_id: Identifier
    dataset_id: Identifier
    source_version: str = Field(min_length=1)
    source_sha256: Sha256
    validation_origin: Literal["train_side"] = "train_side"
    complete_row_assignment: Literal[True] = True
    stable_id_kind: Literal["utf8", "decimal_integer"]
    partitions: dict[str, tuple[str, ...]]

    @field_validator("source_sha256")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source_sha256")

    @field_validator("partitions")
    @classmethod
    def _complete_disjoint_partitions(
        cls, values: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        if set(values) != set(_PARTITIONS):
            raise ValueError("split authority needs exact train/validation/test partitions")
        normalized: dict[str, tuple[str, ...]] = {}
        all_ids: list[str] = []
        for partition in _PARTITIONS:
            ids = tuple(values[partition])
            if not ids or any(not isinstance(row_id, str) or not row_id for row_id in ids):
                raise ValueError("split partition ids must be non-empty strings")
            if len(ids) != len(set(ids)):
                raise ValueError(f"split partition {partition} contains duplicate ids")
            normalized[partition] = ids
            all_ids.extend(ids)
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("split partitions must be disjoint")
        return normalized


class ColumnAuthority(EvidenceSchema):
    """Raw column name plus its stable evaluation family identity."""

    schema_version: Literal["tabu.eval-column-authority.v1"] = (
        "tabu.eval-column-authority.v1"
    )
    source_name: str = Field(min_length=1)
    family_id: Identifier
    kind: TargetKind
    missing_tokens: tuple[str, ...] = ()

    @field_validator("source_name")
    @classmethod
    def _source_name_is_exact(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("column source_name must be a non-empty exact header token")
        return value

    @field_validator("missing_tokens")
    @classmethod
    def _missing_tokens_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("column missing tokens must be unique")
        return values


class DelimitedTableAuthority(EvidenceSchema):
    """Exact UTF-8 delimited serialization and split declaration."""

    schema_version: Literal["tabu.eval-delimited-table-authority.v1"] = (
        "tabu.eval-delimited-table-authority.v1"
    )
    format: Literal["utf8-delimited-header-v1"] = "utf8-delimited-header-v1"
    delimiter: Literal[",", "\t"]
    quotechar: Literal['"'] = '"'
    field_whitespace: Literal["preserve", "strip"]
    header: tuple[str, ...] = Field(min_length=3)
    row_id_column: str = Field(min_length=1)
    feature_columns: tuple[ColumnAuthority, ...] = Field(min_length=1)
    response_column: ColumnAuthority
    unknown_category_policy: Literal["reject"] = "reject"
    split: SplitAuthority

    @model_validator(mode="after")
    def _table_schema_is_closed(self) -> DelimitedTableAuthority:
        if len(self.header) != len(set(self.header)):
            raise ValueError("delimited source header must be unique")
        feature_names = [item.source_name for item in self.feature_columns]
        family_ids = [item.family_id for item in self.feature_columns]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("feature source columns must be unique")
        if len(family_ids) != len(set(family_ids)):
            raise ValueError("feature family ids must be unique")
        consumed = {self.row_id_column, self.response_column.source_name, *feature_names}
        if len(consumed) != 2 + len(feature_names):
            raise ValueError("row id, response, and feature columns must be distinct")
        if set(self.header) != consumed:
            raise ValueError("header must contain exactly row id, response, and feature columns")
        if self.response_column.family_id in set(family_ids):
            raise ValueError("response family must differ from feature families")
        return self


class CompletionMaskAuthority(EvidenceSchema):
    """Exact per-seed all-cell target projection for the suite's 15% mask."""

    schema_version: Literal["tabu.eval-completion-mask-authority.v2"] = (
        "tabu.eval-completion-mask-authority.v2"
    )
    mask_seed: int = Field(ge=0)
    mask_algorithm: Literal["sha256-row-cover-train-support-floor-v2"] = (
        "sha256-row-cover-train-support-floor-v2"
    )
    target_enumeration: Literal["all-masked-cells-v1"] = (
        "all-masked-cells-v1"
    )


class GraphPerturbationAuthority(EvidenceSchema):
    """Two explicit evaluator-owned edge toggles for graph contract checks."""

    schema_version: Literal["tabu.eval-graph-perturbation-authority.v1"] = (
        "tabu.eval-graph-perturbation-authority.v1"
    )
    base_node_id: str = Field(min_length=1)
    topology_toggle_edge: tuple[str, str]
    locality_toggle_edge: tuple[str, str]

    @field_validator("topology_toggle_edge", "locality_toggle_edge")
    @classmethod
    def _edge_is_valid(cls, value: tuple[str, str]) -> tuple[str, str]:
        if len(value) != 2 or any(not item for item in value) or value[0] == value[1]:
            raise ValueError("graph perturbation edges need two distinct node ids")
        return value


class KarateAuthority(EvidenceSchema):
    """Strict retained JSON snapshot authority for Zachary Karate Club."""

    schema_version: Literal["tabu.eval-karate-authority.v1"] = (
        "tabu.eval-karate-authority.v1"
    )
    format: Literal["tabu.karate-retained-json.v1"] = "tabu.karate-retained-json.v1"
    split: SplitAuthority
    feature_columns: tuple[ColumnAuthority, ...] = Field(min_length=1)
    club_family_id: Identifier = "club"
    club_domain: tuple[Literal["Mr. Hi", "Officer"], Literal["Mr. Hi", "Officer"]]
    topology_sha256: Sha256
    expected_node_count: Literal[34] = 34
    expected_edge_count: Literal[78] = 78
    perturbations: GraphPerturbationAuthority

    @field_validator("topology_sha256")
    @classmethod
    def _valid_topology_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="topology_sha256")

    @model_validator(mode="after")
    def _karate_schema_is_closed(self) -> KarateAuthority:
        if set(self.club_domain) != {"Mr. Hi", "Officer"}:
            raise ValueError("karate club domain must be exactly Mr. Hi and Officer")
        names = [item.source_name for item in self.feature_columns]
        families = [item.family_id for item in self.feature_columns]
        if len(names) != len(set(names)) or len(families) != len(set(families)):
            raise ValueError("karate feature columns and families must be unique")
        if self.club_family_id in set(families):
            raise ValueError("karate club family must differ from feature families")
        return self


class MovieLensAuthority(EvidenceSchema):
    """Exact official-archive split and train-side subset authority."""

    schema_version: Literal["tabu.eval-movielens-authority.v1"] = (
        "tabu.eval-movielens-authority.v1"
    )
    format: Literal["movielens-100k-official-zip-v1"] = (
        "movielens-100k-official-zip-v1"
    )
    dataset_id: Literal["movielens-100k"] = "movielens-100k"
    source_version: Literal["MovieLens-100K-official"] = "MovieLens-100K-official"
    source_sha256: Sha256
    base_member: str = Field(min_length=1)
    test_member: str = Field(min_length=1)
    validation_interaction_ids: tuple[str, ...] = Field(min_length=1)
    validation_origin: Literal["train_side"] = "train_side"
    stable_id_kind: Literal["decimal_integer"] = "decimal_integer"
    selection_algorithm: Literal["independent-train-support-desc-stable-id-v1"] = (
        "independent-train-support-desc-stable-id-v1"
    )
    expected_interactions: Literal[100000] = 100000
    expected_users: Literal[943] = 943
    expected_items: Literal[1682] = 1682

    @field_validator("source_sha256")
    @classmethod
    def _valid_source_hash(cls, value: str) -> str:
        return require_sha256(value, field_name="source_sha256")

    @field_validator("base_member", "test_member")
    @classmethod
    def _safe_archive_member(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or value.endswith("/"):
            raise ValueError("MovieLens member must be a safe relative file path")
        return value

    @field_validator("validation_interaction_ids")
    @classmethod
    def _validation_ids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("MovieLens validation interaction ids must be unique")
        for value in values:
            pieces = value.split(":")
            if len(pieces) != 2 or any(not piece.isdigit() or int(piece) <= 0 for piece in pieces):
                raise ValueError("MovieLens interaction ids must be positive user:item ids")
        return values

    @model_validator(mode="after")
    def _members_differ(self) -> MovieLensAuthority:
        if self.base_member == self.test_member:
            raise ValueError("MovieLens base and test members must differ")
        return self


def _read_source(source: RawSource) -> bytes:
    if isinstance(source, bytes):
        content = source
    elif isinstance(source, bytearray | memoryview):
        content = bytes(source)
    elif isinstance(source, Path):
        if not source.is_file():
            raise RealEvalDataError("retained source path must name a regular file")
        if source.stat().st_size > _MAX_SOURCE_BYTES:
            raise RealEvalDataError("retained source exceeds the offline materializer size limit")
        content = source.read_bytes()
    else:
        raise TypeError("source must be retained bytes or an explicit pathlib.Path")
    if not content:
        raise RealEvalDataError("retained source bytes cannot be empty")
    if len(content) > _MAX_SOURCE_BYTES:
        raise RealEvalDataError("retained source exceeds the offline materializer size limit")
    return content


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_scenario(scenario: ScenarioSpec) -> None:
    expected = _SUPPORTED_SCENARIOS.get(scenario.scenario_id)
    if expected is None:
        raise RealEvalDataError("real-data materializer only supports the frozen table suites")
    dataset_id, task = expected
    if scenario.dataset.dataset_id != dataset_id or scenario.task is not task:
        raise RealEvalDataError("scenario identity differs from the frozen dataset/task")
    if scenario.preprocessing_fit_partition != "train":
        raise RealEvalDataError("real-data preprocessing must be train-only")
    if scenario.dataset.required_partitions != _PARTITIONS:
        raise RealEvalDataError("scenario must require exact train/validation/test partitions")


def _bind_split(
    *,
    scenario: ScenarioSpec,
    split: SplitAuthority,
    source_sha256: str,
    row_ids: set[str],
) -> None:
    if split.dataset_id != scenario.dataset.dataset_id:
        raise RealEvalDataError("split authority dataset differs from the scenario")
    if split.source_version != scenario.dataset.source_version:
        raise RealEvalDataError("split authority source version differs from the scenario")
    if split.source_sha256 != source_sha256:
        raise RealEvalDataError("split authority does not bind the retained source bytes")
    assigned = {row_id for ids in split.partitions.values() for row_id in ids}
    if assigned != row_ids:
        missing = len(row_ids - assigned)
        unknown = len(assigned - row_ids)
        raise RealEvalDataError(
            "split authority must exhaustively cover parsed rows "
            f"(missing={missing}, unknown={unknown})"
        )


def _stable_id_key(value: str, kind: Literal["utf8", "decimal_integer"]) -> tuple[object, ...]:
    if kind == "utf8":
        return (value.encode("utf-8"),)
    if not value.isdigit() or int(value) < 0 or str(int(value)) != value:
        raise RealEvalDataError("decimal stable ids must use canonical non-negative digits")
    return (int(value), value)


def _selection_rank(*, dataset_id: str, partition: str, row_id: str) -> tuple[str, str]:
    digest = canonical_hash(
        {
            "schema": "tabu.eval-sha256-row-rank.v1",
            "dataset_id": dataset_id,
            "partition": partition,
            "row_id": row_id,
        }
    )
    return digest, row_id


def _select_partition_rows(
    *, scenario: ScenarioSpec, rows: Mapping[str, object], split: SplitAuthority
) -> dict[str, tuple[str, ...]]:
    if scenario.selection.method != "sha256_rank":
        raise RealEvalDataError("tabular materializer requires sha256_rank selection")
    if set(scenario.selection.partition_limits) != set(_PARTITIONS):
        raise RealEvalDataError("tabular scenario lacks exact partition limits")
    selected: dict[str, tuple[str, ...]] = {}
    for partition in _PARTITIONS:
        candidates = split.partitions[partition]
        limit = scenario.selection.partition_limits[partition]
        if len(candidates) < limit:
            raise RealEvalDataError(f"{partition} split is smaller than its frozen limit")
        ranked = sorted(
            candidates,
            key=lambda row_id: _selection_rank(
                dataset_id=scenario.dataset.dataset_id,
                partition=partition,
                row_id=row_id,
            ),
        )
        chosen = tuple(ranked[:limit])
        if any(row_id not in rows for row_id in chosen):
            raise RealEvalDataError("selected row is absent from parsed source")
        selected[partition] = chosen
    return selected


def _normalize_field(value: str, policy: Literal["preserve", "strip"]) -> str:
    return value if policy == "preserve" else value.strip()


def _parse_value(
    raw: object,
    *,
    column: ColumnAuthority,
    whitespace: Literal["preserve", "strip"] = "preserve",
) -> str | float | None:
    if not isinstance(raw, str):
        if raw is None:
            raw_value = ""
        elif isinstance(raw, int | float) and not isinstance(raw, bool):
            raw_value = str(raw)
        else:
            raise RealEvalDataError(f"column {column.source_name!r} has a non-scalar value")
    else:
        raw_value = raw
    value = _normalize_field(raw_value, whitespace)
    missing = {
        _normalize_field(token, whitespace)
        for token in column.missing_tokens
    }
    if value in missing:
        return None
    if column.kind is TargetKind.CATEGORICAL:
        if not value:
            raise RealEvalDataError(
                f"empty categorical value in {column.source_name!r} needs an explicit missing token"
            )
        return value
    try:
        numeric = float(value)
    except ValueError as error:
        raise RealEvalDataError(
            f"column {column.source_name!r} is not finite numeric data"
        ) from error
    if not math.isfinite(numeric):
        raise RealEvalDataError(f"column {column.source_name!r} is not finite numeric data")
    return numeric


class _TableRow:
    __slots__ = ("features", "response", "row_id")

    def __init__(
        self,
        *,
        row_id: str,
        features: Mapping[str, str | float | None],
        response: str | float,
    ) -> None:
        self.row_id = row_id
        self.features = dict(features)
        self.response = response


def _parse_delimited_table(
    content: bytes, authority: DelimitedTableAuthority
) -> dict[str, _TableRow]:
    try:
        text = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RealEvalDataError("delimited retained source must be strict UTF-8") from error
    if "\x00" in text:
        raise RealEvalDataError("delimited retained source cannot contain NUL bytes")
    stream = io.StringIO(text, newline="")
    try:
        reader = csv.reader(
            stream,
            delimiter=authority.delimiter,
            quotechar=authority.quotechar,
            strict=True,
        )
        raw_header = next(reader)
    except (StopIteration, csv.Error) as error:
        raise RealEvalDataError("delimited retained source has no valid header") from error
    header = tuple(
        _normalize_field(value, authority.field_whitespace) for value in raw_header
    )
    if header != authority.header:
        raise RealEvalDataError("delimited source header differs from its authority manifest")
    index = {name: position for position, name in enumerate(header)}
    rows: dict[str, _TableRow] = {}
    try:
        for line_number, fields in enumerate(reader, start=2):
            if not fields or (len(fields) == 1 and not fields[0]):
                raise RealEvalDataError(f"blank record at source line {line_number}")
            if len(fields) != len(header):
                raise RealEvalDataError(f"field count differs from header at line {line_number}")
            normalized = [
                _normalize_field(value, authority.field_whitespace) for value in fields
            ]
            row_id = normalized[index[authority.row_id_column]]
            if not row_id:
                raise RealEvalDataError(f"empty row id at source line {line_number}")
            if row_id in rows:
                raise RealEvalDataError(f"duplicate row id {row_id!r}")
            feature_values = {
                column.family_id: _parse_value(
                    normalized[index[column.source_name]],
                    column=column,
                    whitespace="preserve",
                )
                for column in authority.feature_columns
            }
            response = _parse_value(
                normalized[index[authority.response_column.source_name]],
                column=authority.response_column,
                whitespace="preserve",
            )
            if response is None:
                raise RealEvalDataError("supervised response cannot be naturally missing")
            rows[row_id] = _TableRow(
                row_id=row_id,
                features=feature_values,
                response=response,
            )
    except csv.Error as error:
        raise RealEvalDataError("malformed delimited retained source") from error
    if not rows:
        raise RealEvalDataError("delimited retained source has no data rows")
    return rows


def _mean_scale(values: Sequence[float], *, family_id: str) -> tuple[float, float]:
    if not values:
        raise RealEvalDataError(f"train-only numeric family {family_id!r} has no observed values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0.0:
        raise RealEvalDataError(
            f"train-only numeric family {family_id!r} needs positive finite variation"
        )
    return mean, scale


def _fit_table_state(
    *,
    authority: DelimitedTableAuthority,
    rows: Mapping[str, _TableRow],
    train_ids: Sequence[str],
    include_response: bool,
) -> dict[str, JsonValue]:
    columns = list(authority.feature_columns)
    if include_response:
        columns.append(authority.response_column)
    state: dict[str, JsonValue] = {
        "schema_version": "tabu.eval-table-fitted-state.v1",
        "fit_partition": "train",
        "families": {},
    }
    families: dict[str, JsonValue] = {}
    for column in columns:
        raw_values = [
            (
                rows[row_id].response
                if column is authority.response_column
                else rows[row_id].features[column.family_id]
            )
            for row_id in train_ids
        ]
        observed = [value for value in raw_values if value is not None]
        if column.kind is TargetKind.NUMERIC:
            mean, scale = _mean_scale(
                [float(value) for value in observed],
                family_id=column.family_id,
            )
            families[column.family_id] = {
                "kind": "numeric",
                "mean": mean,
                "scale": scale,
                "observed_count": len(observed),
            }
        else:
            domain = sorted({str(value) for value in observed})
            if not domain:
                raise RealEvalDataError(
                    f"train-only categorical family {column.family_id!r} has no support"
                )
            families[column.family_id] = {
                "kind": "categorical",
                "domain": domain,
                "observed_count": len(observed),
            }
    state["families"] = families
    return state


def _statistics_train_ids(
    *,
    scenario: ScenarioSpec,
    authority: DelimitedTableAuthority,
    selected: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Resolve the declared train-only statistics/codebook fitting scope.

    The v0 materializer intentionally fitted on the selected train examples.
    Base v1 keeps the same train-only boundary but uses the complete declared
    train partition so rare categorical values in the held-out selection are
    not silently turned into an artificial unknown-category failure.
    """

    if scenario.statistics_fit_scope == "selected_train":
        return tuple(selected["train"])
    if scenario.statistics_fit_scope == "full_train_partition":
        return tuple(authority.split.partitions["train"])
    raise RealEvalDataError("unsupported train-only statistics fit scope")


def _validate_train_only_domains(
    *,
    fitted_state: Mapping[str, JsonValue],
    rows: Mapping[str, _TableRow],
    selected: Mapping[str, Sequence[str]],
    authority: DelimitedTableAuthority,
    include_response: bool,
) -> None:
    columns = list(authority.feature_columns)
    if include_response:
        columns.append(authority.response_column)
    raw_families = fitted_state["families"]
    if not isinstance(raw_families, Mapping):
        raise RealEvalDataError("internal fitted-state family map is malformed")
    for column in columns:
        if column.kind is not TargetKind.CATEGORICAL:
            continue
        raw_family = raw_families[column.family_id]
        if not isinstance(raw_family, Mapping) or not isinstance(raw_family.get("domain"), list):
            raise RealEvalDataError("internal categorical fitted state is malformed")
        domain = set(str(value) for value in raw_family["domain"])
        for partition in _PARTITIONS:
            for row_id in selected[partition]:
                value = (
                    rows[row_id].response
                    if column is authority.response_column
                    else rows[row_id].features[column.family_id]
                )
                if value is not None and str(value) not in domain:
                    raise RealEvalDataError(
                        f"{partition} contains unseen category in train-only family "
                        f"{column.family_id!r}"
                    )


def _feature_specs_from_state(
    *,
    columns: Sequence[ColumnAuthority],
    fitted_state: Mapping[str, JsonValue],
    response_family: str | None = None,
) -> list[dict[str, JsonValue]]:
    raw_families = fitted_state.get("families")
    if not isinstance(raw_families, Mapping):
        raise RealEvalDataError("fitted state has no family map")
    specs: list[dict[str, JsonValue]] = []
    for column in columns:
        family = raw_families.get(column.family_id)
        if not isinstance(family, Mapping):
            raise RealEvalDataError("fitted state omits a declared family")
        if column.kind is TargetKind.NUMERIC:
            domain: list[str] = []
            codebook_id = None
        else:
            raw_domain = family.get("domain")
            if not isinstance(raw_domain, list) or not raw_domain:
                raise RealEvalDataError("categorical fitted state has no train-only domain")
            domain = [str(value) for value in raw_domain]
            codebook_id = (
                f"eval-{column.family_id}-{canonical_hash({'domain': domain})[:16]}"
            )
        specs.append(
            {
                "name": column.family_id,
                "kind": column.kind.value,
                "domain": domain,
                "codebook_id": codebook_id,
                "role": (
                    FeatureRole.RESPONSE.value
                    if column.family_id == response_family
                    else FeatureRole.PREDICTOR.value
                ),
            }
        )
    return specs


def _example_id(*, dataset_id: str, partition: str, row_id: str) -> str:
    digest = canonical_hash(
        {
            "schema": "tabu.eval-example-id.v1",
            "dataset_id": dataset_id,
            "partition": partition,
            "row_id": row_id,
        }
    )
    return f"{dataset_id}-{partition}-{digest[:20]}"


def _masked_cell_example_id(
    *, dataset_id: str, partition: str, row_id: str, family_id: str, mask_seed: int
) -> str:
    digest = canonical_hash(
        {
            "schema": "tabu.eval-masked-cell-example-id.v1",
            "dataset_id": dataset_id,
            "partition": partition,
            "row_id": row_id,
            "family_id": family_id,
            "mask_seed": mask_seed,
        }
    )
    return f"{dataset_id}-{partition}-cell-{digest[:20]}"


def _preparation(
    *,
    scenario: ScenarioSpec,
    fitted_state: Mapping[str, JsonValue],
    authority_sha256: str,
    split_authority_sha256: str,
    checkpoint_projection: Mapping[str, JsonValue],
    execution: Mapping[str, JsonValue] | None = None,
) -> PreparationContract:
    preprocessing: dict[str, JsonValue] = {
        "schema_version": "tabu.eval-real-preprocessing.v1",
        "fit_partition": "train",
        "implementation_sha256": _IMPLEMENTATION_SHA256,
        "fitted_state_sha256": canonical_hash(fitted_state),
        "fitted_state": dict(fitted_state),
        "source_authority_sha256": authority_sha256,
        "split_authority_sha256": split_authority_sha256,
        "checkpoint_projection": dict(checkpoint_projection),
    }
    if execution is not None:
        preprocessing["execution"] = dict(execution)
    return PreparationContract(
        preprocessing=preprocessing,
        selection=scenario.selection.model_dump(mode="python"),
        mask=(
            scenario.mask.model_dump(mode="python")
            if scenario.mask is not None
            else {"kind": "none"}
        ),
    )


def _build_prepared(
    *,
    scenario: ScenarioSpec,
    source_material: SourceMaterial,
    preparation: PreparationContract,
    partitions: Mapping[str, Sequence[PreparedExample]],
    topology_checks: Sequence[TopologyCheckCase] = (),
) -> PreparedScenario:
    train = tuple(partitions["train"])
    validation = tuple(partitions["validation"])
    test = tuple(partitions["test"])
    checks = tuple(topology_checks)
    binding = DatasetSnapshotBinding(
        dataset_id=scenario.dataset.dataset_id,
        source_sha256=source_material.raw_sha256,
        split_sha256=PreparedScenario.split_sha256_for(
            train=train,
            validation=validation,
            test=test,
        ),
        recipe_sha256=PreparedScenario.recipe_sha256_for(
            preparation=preparation,
            topology_checks=checks,
        ),
        truth_sidecar_sha256=PreparedScenario.truth_sidecar_sha256_for(test=test),
        partition_counts={
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
    )
    return PreparedScenario(
        scenario_id=scenario.scenario_id,
        binding=binding,
        source_material=source_material,
        preparation=preparation,
        train=train,
        validation=validation,
        test=test,
        topology_checks=checks,
    )


def _check_target_support(
    partitions: Mapping[str, Sequence[PreparedExample]],
    *,
    fitted_state: Mapping[str, JsonValue] | None = None,
) -> None:
    numeric_families = {
        item.target_family
        for item in partitions["train"]
        if item.target_kind is TargetKind.NUMERIC
    }
    categorical_support: dict[str, set[str]] = {}
    for item in partitions["train"]:
        if item.target_kind is TargetKind.CATEGORICAL:
            categorical_support.setdefault(item.target_family, set()).add(str(item.target))
    if fitted_state is not None:
        raw_families = fitted_state.get("families")
        if not isinstance(raw_families, Mapping):
            raise RealEvalDataError("fitted state has no family map for target support")
        for family_id, raw_family in raw_families.items():
            if not isinstance(raw_family, Mapping):
                continue
            raw_domain = raw_family.get("domain")
            if isinstance(raw_domain, list):
                categorical_support[str(family_id)] = {str(value) for value in raw_domain}
    for partition in ("validation", "test"):
        for item in partitions[partition]:
            if item.target_kind is TargetKind.NUMERIC:
                if item.target_family not in numeric_families:
                    raise RealEvalDataError(
                        f"{partition} numeric target family has no train-only scoring scale"
                    )
            elif str(item.target) not in categorical_support.get(item.target_family, set()):
                raise RealEvalDataError(
                    f"{partition} categorical target lacks complete train-only NLL support"
                )


def materialize_table_supervised(
    *,
    scenario: ScenarioSpec,
    source: RawSource,
    authority: DelimitedTableAuthority,
) -> PreparedScenario:
    """Materialize the Adult or Diabetes supervised v0 scenario offline."""

    _validate_scenario(scenario)
    if scenario.task not in {
        TaskKind.SUPERVISED_CLASSIFICATION,
        TaskKind.SUPERVISED_REGRESSION,
    }:
        raise RealEvalDataError("supervised table materializer received another task")
    content = _read_source(source)
    source_sha256 = _sha256_bytes(content)
    if source_sha256 != authority.split.source_sha256:
        raise RealEvalDataError("split authority does not bind the retained source bytes")
    rows = _parse_delimited_table(content, authority)
    _bind_split(
        scenario=scenario,
        split=authority.split,
        source_sha256=source_sha256,
        row_ids=set(rows),
    )
    selected = _select_partition_rows(scenario=scenario, rows=rows, split=authority.split)
    fitted_state = _fit_table_state(
        authority=authority,
        rows=rows,
        train_ids=_statistics_train_ids(
            scenario=scenario,
            authority=authority,
            selected=selected,
        ),
        include_response=True,
    )
    _validate_train_only_domains(
        fitted_state=fitted_state,
        rows=rows,
        selected=selected,
        authority=authority,
        include_response=True,
    )
    expected_kind = (
        TargetKind.CATEGORICAL
        if scenario.task is TaskKind.SUPERVISED_CLASSIFICATION
        else TargetKind.NUMERIC
    )
    if authority.response_column.kind is not expected_kind:
        raise RealEvalDataError("response kind differs from the supervised scenario")

    partitions: dict[str, tuple[PreparedExample, ...]] = {}
    for partition in _PARTITIONS:
        prepared_rows = []
        for row_id in selected[partition]:
            row = rows[row_id]
            prepared_rows.append(
                PreparedExample(
                    example_id=_example_id(
                        dataset_id=scenario.dataset.dataset_id,
                        partition=partition,
                        row_id=row_id,
                    ),
                    target_kind=authority.response_column.kind,
                    target_family=authority.response_column.family_id,
                    features={
                        family: value
                        for family, value in row.features.items()
                        if value is not None
                    },
                    target=row.response,
                    context={
                        "row_id": row_id,
                        "naturally_missing_features": sorted(
                            family for family, value in row.features.items() if value is None
                        ),
                    },
                )
            )
        partitions[partition] = tuple(prepared_rows)
    _check_target_support(partitions, fitted_state=fitted_state)

    columns = (*authority.feature_columns, authority.response_column)
    feature_specs = _feature_specs_from_state(
        columns=columns,
        fitted_state=fitted_state,
        response_family=authority.response_column.family_id,
    )
    checkpoint_projection = {
        "schema_version": CHECKPOINT_PROJECTION_SCHEMA,
        "kind": "table",
        "mode": "supervised",
        "dataset_id": scenario.dataset.dataset_id,
        "feature_specs": feature_specs,
        "response_family": authority.response_column.family_id,
    }
    preparation = _preparation(
        scenario=scenario,
        fitted_state=fitted_state,
        authority_sha256=authority.content_hash,
        split_authority_sha256=authority.split.content_hash,
        checkpoint_projection=checkpoint_projection,
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id=scenario.dataset.dataset_id,
        content=content,
        media_type="text/csv" if authority.delimiter == "," else "text/tab-separated-values",
    )
    return _build_prepared(
        scenario=scenario,
        source_material=source_material,
        preparation=preparation,
        partitions=partitions,
    )


def _mask_rank(
    *, mask_seed: int, dataset_id: str, partition: str, row_id: str, family_id: str
) -> tuple[str, str, str]:
    return (
        canonical_hash(
            {
                "schema": "tabu.eval-cell-mask-rank.v1",
                "mask_seed": mask_seed,
                "dataset_id": dataset_id,
                "partition": partition,
                "row_id": row_id,
                "family_id": family_id,
            }
        ),
        row_id,
        family_id,
    )


def _required_train_keys(
    *, rows: Mapping[str, _TableRow], row_ids: Sequence[str], columns: Sequence[ColumnAuthority]
) -> dict[tuple[str, str], tuple[str, ...]]:
    candidates: dict[tuple[str, str], list[str]] = {}
    for column in columns:
        if column.kind is TargetKind.NUMERIC:
            key_values = [(column.family_id, "__numeric__")]
        else:
            domain = sorted(
                {
                    str(rows[row_id].features[column.family_id])
                    for row_id in row_ids
                    if rows[row_id].features[column.family_id] is not None
                }
            )
            key_values = [(column.family_id, value) for value in domain]
        for key in key_values:
            candidates[key] = [
                row_id
                for row_id in row_ids
                if rows[row_id].features[column.family_id] is not None
                and (
                    column.kind is TargetKind.NUMERIC
                    or str(rows[row_id].features[column.family_id]) == key[1]
                )
            ]
    return {key: tuple(values) for key, values in candidates.items()}


def _deterministic_support_cover(
    *,
    rows: Mapping[str, _TableRow],
    row_ids: Sequence[str],
    columns: Sequence[ColumnAuthority],
    mask_seed: int,
    dataset_id: str,
) -> dict[str, str]:
    """Assign distinct train rows to every target family/category when possible."""

    candidates = _required_train_keys(rows=rows, row_ids=row_ids, columns=columns)
    ordered_keys = sorted(candidates, key=lambda key: (len(candidates[key]), key))
    row_to_key: dict[str, tuple[str, str]] = {}

    def assign(key: tuple[str, str], seen: set[str]) -> bool:
        ranked_rows = sorted(
            candidates[key],
            key=lambda row_id: _mask_rank(
                mask_seed=mask_seed,
                dataset_id=dataset_id,
                partition="train",
                row_id=row_id,
                family_id=key[0],
            ),
        )
        for row_id in ranked_rows:
            if row_id in seen:
                continue
            seen.add(row_id)
            previous = row_to_key.get(row_id)
            if previous is None or assign(previous, seen):
                row_to_key[row_id] = key
                return True
        return False

    for key in ordered_keys:
        if not candidates[key] or not assign(key, set()):
            raise RealEvalDataError(
                "train rows cannot provide one deterministic target for every family/category"
            )
    return {row_id: key[0] for row_id, key in row_to_key.items()}


def _partition_mask(
    *,
    scenario: ScenarioSpec,
    rows: Mapping[str, _TableRow],
    row_ids: Sequence[str],
    columns: Sequence[ColumnAuthority],
    partition: str,
    mask_authority: CompletionMaskAuthority,
) -> dict[str, tuple[str, ...]]:
    if scenario.mask is None:
        raise RealEvalDataError("completion scenario lacks a mask contract")
    eligible_by_row = {
        row_id: tuple(
            column.family_id
            for column in columns
            if rows[row_id].features[column.family_id] is not None
        )
        for row_id in row_ids
    }
    if any(not families for families in eligible_by_row.values()):
        raise RealEvalDataError("every selected completion row needs an observed feature")
    eligible = [
        (row_id, family_id)
        for row_id in row_ids
        for family_id in eligible_by_row[row_id]
    ]
    quota = math.floor(scenario.mask.fraction * len(eligible))
    if quota <= 0:
        raise RealEvalDataError("frozen mask fraction selects no observed completion cells")

    primary: dict[str, str]
    if partition == "train":
        primary = _deterministic_support_cover(
            rows=rows,
            row_ids=row_ids,
            columns=columns,
            mask_seed=mask_authority.mask_seed,
            dataset_id=scenario.dataset.dataset_id,
        )
    else:
        primary = {}
    for row_id in row_ids:
        if row_id not in primary:
            primary[row_id] = min(
                eligible_by_row[row_id],
                key=lambda family_id: _mask_rank(
                    mask_seed=mask_authority.mask_seed,
                    dataset_id=scenario.dataset.dataset_id,
                    partition=partition,
                    row_id=row_id,
                    family_id=family_id,
                ),
            )
    chosen = {(row_id, family_id) for row_id, family_id in primary.items()}
    if len(chosen) > quota:
        raise RealEvalDataError(
            "15% train mask budget cannot enumerate every train-only target support"
        )
    remaining = sorted(
        (cell for cell in eligible if cell not in chosen),
        key=lambda cell: _mask_rank(
            mask_seed=mask_authority.mask_seed,
            dataset_id=scenario.dataset.dataset_id,
            partition=partition,
            row_id=cell[0],
            family_id=cell[1],
        ),
    )
    chosen.update(remaining[: quota - len(chosen)])
    masks = {
        row_id: tuple(sorted(family for candidate, family in chosen if candidate == row_id))
        for row_id in row_ids
    }
    if sum(len(values) for values in masks.values()) != quota:
        raise RealEvalDataError("internal completion mask quota drift")
    return masks


def materialize_table_completion(
    *,
    scenario: ScenarioSpec,
    source: RawSource,
    authority: DelimitedTableAuthority,
    mask_authority: CompletionMaskAuthority,
) -> PreparedScenario:
    """Materialize Adult or Diabetes artificial-cell completion offline."""

    _validate_scenario(scenario)
    if scenario.task is not TaskKind.TABLE_COMPLETION:
        raise RealEvalDataError("completion table materializer received another task")
    if scenario.mask is None or not math.isclose(
        scenario.mask.fraction, 0.15, rel_tol=0.0, abs_tol=0.0
    ):
        raise RealEvalDataError("v0 table completion requires the frozen 15% mask")
    content = _read_source(source)
    source_sha256 = _sha256_bytes(content)
    if source_sha256 != authority.split.source_sha256:
        raise RealEvalDataError("split authority does not bind the retained source bytes")
    rows = _parse_delimited_table(content, authority)
    _bind_split(
        scenario=scenario,
        split=authority.split,
        source_sha256=source_sha256,
        row_ids=set(rows),
    )
    selected = _select_partition_rows(scenario=scenario, rows=rows, split=authority.split)
    fitted_state = _fit_table_state(
        authority=authority,
        rows=rows,
        train_ids=_statistics_train_ids(
            scenario=scenario,
            authority=authority,
            selected=selected,
        ),
        include_response=True,
    )
    _validate_train_only_domains(
        fitted_state=fitted_state,
        rows=rows,
        selected=selected,
        authority=authority,
        # Completion response values never enter an episode.  Its schema and
        # codebook/statistics are fitted from train above, but validation/test
        # response values must not influence materialization acceptance.
        include_response=False,
    )
    by_family = {column.family_id: column for column in authority.feature_columns}
    partitions: dict[str, tuple[PreparedExample, ...]] = {}
    mask_manifest: dict[str, JsonValue] = {
        "schema_version": MASK_EXECUTION_SCHEMA,
        "mask_seed": mask_authority.mask_seed,
        "partitions": {},
    }
    for partition in _PARTITIONS:
        masks = _partition_mask(
            scenario=scenario,
            rows=rows,
            row_ids=selected[partition],
            columns=authority.feature_columns,
            partition=partition,
            mask_authority=mask_authority,
        )
        mask_manifest["partitions"][partition] = {
            row_id: {"masked_families": list(masks[row_id])}
            for row_id in selected[partition]
        }
        examples: list[PreparedExample] = []
        for row_id in selected[partition]:
            row = rows[row_id]
            visible_features = {
                family: value
                for family, value in row.features.items()
                if value is not None and family not in set(masks[row_id])
            }
            context = {
                "row_id": row_id,
                "mask_seed": mask_authority.mask_seed,
                "artificially_masked_features": list(masks[row_id]),
                "naturally_missing_features": sorted(
                    family for family, value in row.features.items() if value is None
                ),
            }
            for family_id in masks[row_id]:
                column = by_family[family_id]
                target = row.features[family_id]
                if target is None:
                    raise RealEvalDataError(
                        "natural missing cell cannot become artificial-mask truth"
                    )
                examples.append(
                    PreparedExample(
                        example_id=_masked_cell_example_id(
                            dataset_id=scenario.dataset.dataset_id,
                            partition=partition,
                            row_id=row_id,
                            family_id=family_id,
                            mask_seed=mask_authority.mask_seed,
                        ),
                        target_kind=column.kind,
                        target_family=family_id,
                        features=visible_features,
                        target=target,
                        context=context,
                    )
                )
        partitions[partition] = tuple(examples)
    _check_target_support(partitions, fitted_state=fitted_state)

    feature_specs = _feature_specs_from_state(
        columns=(*authority.feature_columns, authority.response_column),
        fitted_state=fitted_state,
        response_family=authority.response_column.family_id,
    )
    checkpoint_projection = {
        "schema_version": CHECKPOINT_PROJECTION_SCHEMA,
        "kind": "table",
        "mode": "completion",
        "dataset_id": scenario.dataset.dataset_id,
        "feature_specs": feature_specs,
        "response_family": authority.response_column.family_id,
        "response_visibility": "schema_only_natural_missing",
    }
    execution = {
        "schema_version": MASK_EXECUTION_SCHEMA,
        "mask_authority_sha256": mask_authority.content_hash,
        "mask_seed": mask_authority.mask_seed,
        "mask_manifest_sha256": canonical_hash(mask_manifest),
        "target_enumeration": "all-masked-cells-v1",
        "selected_row_counts": {
            partition: len(selected[partition]) for partition in _PARTITIONS
        },
        "masked_target_counts": {
            partition: len(partitions[partition]) for partition in _PARTITIONS
        },
    }
    preparation = _preparation(
        scenario=scenario,
        fitted_state=fitted_state,
        authority_sha256=authority.content_hash,
        split_authority_sha256=authority.split.content_hash,
        checkpoint_projection=checkpoint_projection,
        execution=execution,
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id=scenario.dataset.dataset_id,
        content=content,
        media_type="text/csv" if authority.delimiter == "," else "text/tab-separated-values",
    )
    return _build_prepared(
        scenario=scenario,
        source_material=source_material,
        preparation=preparation,
        partitions=partitions,
    )


class _KarateNode:
    __slots__ = ("club", "features", "node_id")

    def __init__(self, *, node_id: str, features: Mapping[str, object], club: str) -> None:
        self.node_id = node_id
        self.features = dict(features)
        self.club = club


def _strict_json(content: bytes, *, source_name: str) -> object:
    try:
        text = content.decode("utf-8", errors="strict")
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealEvalDataError(
            f"{source_name} retained source must be strict UTF-8 JSON"
        ) from error


def _parse_karate(
    content: bytes, authority: KarateAuthority
) -> tuple[dict[str, _KarateNode], tuple[tuple[str, str], ...]]:
    payload = _strict_json(content, source_name="Karate")
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "dataset_id",
        "source_version",
        "nodes",
        "edges",
    }:
        raise RealEvalDataError("Karate JSON has missing or unknown top-level fields")
    if payload["schema_version"] != authority.format:
        raise RealEvalDataError("Karate JSON schema differs from authority")
    if payload["dataset_id"] != authority.split.dataset_id:
        raise RealEvalDataError("Karate JSON dataset id differs from authority")
    if payload["source_version"] != authority.split.source_version:
        raise RealEvalDataError("Karate JSON source version differs from authority")
    raw_nodes = payload["nodes"]
    raw_edges = payload["edges"]
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise RealEvalDataError("Karate JSON nodes and edges must be lists")
    nodes: dict[str, _KarateNode] = {}
    feature_by_name = {item.source_name: item for item in authority.feature_columns}
    for raw_node in raw_nodes:
        if not isinstance(raw_node, dict) or set(raw_node) != {"node_id", "club", "features"}:
            raise RealEvalDataError("Karate node has missing or unknown fields")
        node_id = raw_node["node_id"]
        club = raw_node["club"]
        features = raw_node["features"]
        if not isinstance(node_id, str) or not node_id or node_id in nodes:
            raise RealEvalDataError("Karate node ids must be unique non-empty strings")
        if club not in set(authority.club_domain):
            raise RealEvalDataError("Karate club value lies outside the frozen domain")
        if not isinstance(features, dict) or set(features) != set(feature_by_name):
            raise RealEvalDataError("Karate node features differ from the authority schema")
        parsed = {
            feature_by_name[name].family_id: _parse_value(
                value,
                column=feature_by_name[name],
            )
            for name, value in features.items()
        }
        nodes[node_id] = _KarateNode(node_id=node_id, features=parsed, club=club)
    if len(nodes) != authority.expected_node_count:
        raise RealEvalDataError("Karate node count differs from the frozen contract")
    edges: set[tuple[str, str]] = set()
    for raw_edge in raw_edges:
        if (
            not isinstance(raw_edge, list)
            or len(raw_edge) != 2
            or any(not isinstance(value, str) for value in raw_edge)
        ):
            raise RealEvalDataError("Karate edges must be two-string JSON lists")
        left, right = raw_edge
        if left == right or left not in nodes or right not in nodes:
            raise RealEvalDataError("Karate edge has an invalid endpoint")
        edge = tuple(sorted((left, right)))
        if edge in edges:
            raise RealEvalDataError("Karate topology contains a duplicate undirected edge")
        edges.add(edge)
    if len(edges) != authority.expected_edge_count:
        raise RealEvalDataError("Karate edge count differs from the frozen contract")
    ordered_edges = tuple(sorted(edges))
    topology_sha256 = canonical_hash(
        {
            "schema": "tabu.eval-karate-topology.v1",
            "node_ids": sorted(nodes),
            "edges": ordered_edges,
        }
    )
    if topology_sha256 != authority.topology_sha256:
        raise RealEvalDataError("Karate topology differs from the authority hash")
    return nodes, ordered_edges


def _neighbor_ids(node_id: str, edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    values = {
        right if left == node_id else left
        for left, right in edges
        if left == node_id or right == node_id
    }
    return tuple(sorted(values))


def _toggle_edge(
    edges: Sequence[tuple[str, str]], edge: tuple[str, str]
) -> tuple[tuple[str, str], ...]:
    canonical = tuple(sorted(edge))
    result = set(edges)
    if canonical in result:
        result.remove(canonical)
    else:
        result.add(canonical)
    return tuple(sorted(result))


def _graph_flat_payload(
    *,
    node: _KarateNode,
    edges: Sequence[tuple[str, str]],
    train_clubs: Mapping[str, str],
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    neighbors = _neighbor_ids(node.node_id, edges)
    return (
        {family: value for family, value in node.features.items() if value is not None},
        {
            "node_id": node.node_id,
            "neighbor_ids": list(neighbors),
            "neighbor_labels": [train_clubs[item] for item in neighbors if item in train_clubs],
            "naturally_missing_features": sorted(
                family for family, value in node.features.items() if value is None
            ),
        },
    )


def materialize_karate(
    *,
    scenario: ScenarioSpec,
    source: RawSource,
    authority: KarateAuthority,
) -> PreparedScenario:
    """Materialize the frozen 34-node Zachary Karate graph offline."""

    _validate_scenario(scenario)
    if scenario.task is not TaskKind.GRAPH_COMPLETION:
        raise RealEvalDataError("Karate materializer received another task")
    if set(scenario.topology_contract_checks) != {
        "topology_perturbation_pass",
        "locality_contract_pass",
    }:
        raise RealEvalDataError("Karate suite lacks the two frozen topology checks")
    if scenario.selection.method != "all" or scenario.selection.partition_limits != {
        "train": 20,
        "validation": 7,
        "test": 7,
    }:
        raise RealEvalDataError("Karate suite lacks its exact 20/7/7 all-node split")
    content = _read_source(source)
    source_sha256 = _sha256_bytes(content)
    if source_sha256 != authority.split.source_sha256:
        raise RealEvalDataError("split authority does not bind the retained source bytes")
    nodes, edges = _parse_karate(content, authority)
    _bind_split(
        scenario=scenario,
        split=authority.split,
        source_sha256=source_sha256,
        row_ids=set(nodes),
    )
    selected = {
        partition: tuple(
            sorted(
                authority.split.partitions[partition],
                key=lambda node_id: _stable_id_key(node_id, authority.split.stable_id_kind),
            )
        )
        for partition in _PARTITIONS
    }
    if {key: len(value) for key, value in selected.items()} != scenario.selection.partition_limits:
        raise RealEvalDataError("Karate split authority does not provide exact 20/7/7 counts")
    train_clubs = {node_id: nodes[node_id].club for node_id in selected["train"]}
    if set(train_clubs.values()) != set(authority.club_domain):
        raise RealEvalDataError("Karate train partition must support both club categories")

    # Reuse the table fitted-state schema for node features, but derive it from
    # train nodes only and add the train-only response codebook explicitly.
    pseudo_rows = {
        node_id: _TableRow(
            row_id=node_id,
            features=nodes[node_id].features,
            response=nodes[node_id].club,
        )
        for node_id in nodes
    }
    pseudo_authority = DelimitedTableAuthority(
        delimiter=",",
        field_whitespace="preserve",
        header=(
            "node_id",
            *(item.source_name for item in authority.feature_columns),
            "club",
        ),
        row_id_column="node_id",
        feature_columns=authority.feature_columns,
        response_column=ColumnAuthority(
            source_name="club",
            family_id=authority.club_family_id,
            kind=TargetKind.CATEGORICAL,
        ),
        split=authority.split,
    )
    fitted_state = _fit_table_state(
        authority=pseudo_authority,
        rows=pseudo_rows,
        train_ids=selected["train"],
        include_response=True,
    )
    _validate_train_only_domains(
        fitted_state=fitted_state,
        rows=pseudo_rows,
        selected=selected,
        authority=pseudo_authority,
        include_response=True,
    )

    partitions: dict[str, tuple[PreparedExample, ...]] = {}
    by_partition_id: dict[str, PreparedExample] = {}
    for partition in _PARTITIONS:
        examples: list[PreparedExample] = []
        for node_id in selected[partition]:
            features, context = _graph_flat_payload(
                node=nodes[node_id], edges=edges, train_clubs=train_clubs
            )
            example = PreparedExample(
                example_id=_example_id(
                    dataset_id=scenario.dataset.dataset_id,
                    partition=partition,
                    row_id=node_id,
                ),
                target_kind=TargetKind.CATEGORICAL,
                target_family=authority.club_family_id,
                features=features,
                target=nodes[node_id].club,
                context=context,
            )
            examples.append(example)
            by_partition_id[f"{partition}:{node_id}"] = example
        partitions[partition] = tuple(examples)
    _check_target_support(partitions)

    perturbations = authority.perturbations
    base_node_id = perturbations.base_node_id
    if base_node_id not in set(selected["test"]):
        raise RealEvalDataError("graph perturbation base node must lie in test")
    topology_edge = tuple(sorted(perturbations.topology_toggle_edge))
    locality_edge = tuple(sorted(perturbations.locality_toggle_edge))
    if any(endpoint not in nodes for edge in (topology_edge, locality_edge) for endpoint in edge):
        raise RealEvalDataError("graph perturbation edge has an unknown endpoint")
    if base_node_id not in topology_edge:
        raise RealEvalDataError("topology perturbation must toggle an edge incident to the base")
    topology_other = topology_edge[0] if topology_edge[1] == base_node_id else topology_edge[1]
    if topology_other not in train_clubs:
        raise RealEvalDataError(
            "topology perturbation must add/remove a train-labeled one-hop neighbor"
        )
    if base_node_id in locality_edge:
        raise RealEvalDataError("locality perturbation cannot touch the base node")

    base = by_partition_id[f"test:{base_node_id}"]
    topology_edges = _toggle_edge(edges, topology_edge)
    locality_edges = _toggle_edge(edges, locality_edge)
    topology_features, topology_context = _graph_flat_payload(
        node=nodes[base_node_id], edges=topology_edges, train_clubs=train_clubs
    )
    locality_features, locality_context = _graph_flat_payload(
        node=nodes[base_node_id], edges=locality_edges, train_clubs=train_clubs
    )
    if (topology_features, topology_context) == (base.features, base.context):
        raise RealEvalDataError(
            "topology perturbation does not change evaluator-visible local input"
        )
    if (locality_features, locality_context) != (base.features, base.context):
        raise RealEvalDataError("locality perturbation changes the base node's one-hop input")
    checks = (
        TopologyCheckCase(
            check_id="topology_perturbation_pass",
            base_example_id=base.example_id,
            perturbed_example=BlindExample(
                example_id=f"{base.example_id}-topology-perturbed",
                target_kind=base.target_kind,
                target_family=base.target_family,
                features=topology_features,
                context=topology_context,
            ),
            expected_relation="different",
        ),
        TopologyCheckCase(
            check_id="locality_contract_pass",
            base_example_id=base.example_id,
            perturbed_example=BlindExample(
                example_id=f"{base.example_id}-locality-perturbed",
                target_kind=base.target_kind,
                target_family=base.target_family,
                features=locality_features,
                context=locality_context,
            ),
            expected_relation="equal",
        ),
    )

    feature_specs = _feature_specs_from_state(
        columns=(
            *authority.feature_columns,
            pseudo_authority.response_column,
        ),
        fitted_state=fitted_state,
        response_family=authority.club_family_id,
    )
    node_features = {
        node_id: {
            family: value
            for family, value in nodes[node_id].features.items()
            if value is not None
        }
        for node_id in sorted(nodes)
    }
    naturally_missing = {
        node_id: sorted(
            family for family, value in nodes[node_id].features.items() if value is None
        )
        for node_id in sorted(nodes)
    }
    checkpoint_projection = {
        "schema_version": CHECKPOINT_PROJECTION_SCHEMA,
        "kind": "graph",
        "dataset_id": scenario.dataset.dataset_id,
        "feature_specs": feature_specs,
        "node_ids": list(
            sorted(
                nodes,
                key=lambda value: _stable_id_key(value, authority.split.stable_id_kind),
            )
        ),
        "node_features": node_features,
        "naturally_missing": naturally_missing,
        "edges": [list(edge) for edge in edges],
        "direction": "undirected",
        "response_family": authority.club_family_id,
        "perturbation_edges": {
            "topology_perturbation_pass": list(topology_edge),
            "locality_contract_pass": list(locality_edge),
        },
    }
    preparation = _preparation(
        scenario=scenario,
        fitted_state=fitted_state,
        authority_sha256=authority.content_hash,
        split_authority_sha256=authority.split.content_hash,
        checkpoint_projection=checkpoint_projection,
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id=scenario.dataset.dataset_id,
        content=content,
        media_type="application/json",
    )
    return _build_prepared(
        scenario=scenario,
        source_material=source_material,
        preparation=preparation,
        partitions=partitions,
        topology_checks=checks,
    )


class _Rating:
    __slots__ = ("item_id", "rating", "timestamp", "user_id")

    def __init__(self, *, user_id: str, item_id: str, rating: float, timestamp: int) -> None:
        self.user_id = user_id
        self.item_id = item_id
        self.rating = rating
        self.timestamp = timestamp

    @property
    def interaction_id(self) -> str:
        return f"{self.user_id}:{self.item_id}"


def _read_zip_member(archive: zipfile.ZipFile, name: str) -> bytes:
    infos = [item for item in archive.infolist() if item.filename == name]
    if len(infos) != 1:
        raise RealEvalDataError(f"MovieLens archive needs exactly one member {name!r}")
    info = infos[0]
    if info.flag_bits & 0x1:
        raise RealEvalDataError("encrypted MovieLens archive members are forbidden")
    if info.file_size > 32 * 1024 * 1024:
        raise RealEvalDataError("MovieLens split member exceeds the safe retained size")
    return archive.read(info)


def _parse_ratings(content: bytes, *, member: str) -> dict[str, _Rating]:
    try:
        text = content.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise RealEvalDataError(f"MovieLens member {member!r} must be ASCII") from error
    ratings: dict[str, _Rating] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 4 or any(not value.isdigit() for value in fields):
            raise RealEvalDataError(
                f"MovieLens member {member!r} has malformed row {line_number}"
            )
        user_id, item_id, rating_text, timestamp_text = fields
        if int(user_id) <= 0 or int(item_id) <= 0:
            raise RealEvalDataError("MovieLens user and item ids must be positive")
        rating_value = int(rating_text)
        if rating_value < 1 or rating_value > 5:
            raise RealEvalDataError("MovieLens ratings must lie in the official 1..5 scale")
        rating = _Rating(
            user_id=user_id,
            item_id=item_id,
            rating=float(rating_value),
            timestamp=int(timestamp_text),
        )
        if rating.interaction_id in ratings:
            raise RealEvalDataError("MovieLens split contains duplicate user-item interactions")
        ratings[rating.interaction_id] = rating
    if not ratings:
        raise RealEvalDataError(f"MovieLens member {member!r} is empty")
    return ratings


def _parse_movielens(
    content: bytes, authority: MovieLensAuthority
) -> tuple[dict[str, _Rating], dict[str, _Rating], dict[str, _Rating]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
            names = [item.filename for item in archive.infolist()]
            if len(names) != len(set(names)):
                raise RealEvalDataError("MovieLens archive contains duplicate member names")
            if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
                raise RealEvalDataError("MovieLens archive exceeds the safe uncompressed size")
            base = _parse_ratings(
                _read_zip_member(archive, authority.base_member),
                member=authority.base_member,
            )
            test = _parse_ratings(
                _read_zip_member(archive, authority.test_member),
                member=authority.test_member,
            )
    except zipfile.BadZipFile as error:
        raise RealEvalDataError("MovieLens retained source is not a valid ZIP archive") from error
    if set(base) & set(test):
        raise RealEvalDataError("MovieLens official base/test interactions must be disjoint")
    if len(base) + len(test) != authority.expected_interactions:
        raise RealEvalDataError("MovieLens interaction count differs from the official contract")
    all_ratings = (*base.values(), *test.values())
    if len({item.user_id for item in all_ratings}) != authority.expected_users:
        raise RealEvalDataError("MovieLens user count differs from the official contract")
    if len({item.item_id for item in all_ratings}) != authority.expected_items:
        raise RealEvalDataError("MovieLens item count differs from the official contract")
    validation_ids = set(authority.validation_interaction_ids)
    if not validation_ids < set(base):
        raise RealEvalDataError("MovieLens validation carve must be a proper subset of base")
    validation = {key: base[key] for key in validation_ids}
    train = {key: value for key, value in base.items() if key not in validation_ids}
    return train, validation, test


def _support_selection(
    *, train: Mapping[str, _Rating], users: int, items: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    user_counts = Counter(item.user_id for item in train.values())
    item_counts = Counter(item.item_id for item in train.values())
    if len(user_counts) < users or len(item_counts) < items:
        raise RealEvalDataError("MovieLens train split lacks the frozen 64x128 support")
    selected_users = tuple(
        sorted(user_counts, key=lambda value: (-user_counts[value], int(value), value))[:users]
    )
    selected_items = tuple(
        sorted(item_counts, key=lambda value: (-item_counts[value], int(value), value))[:items]
    )
    return selected_users, selected_items


def materialize_movielens(
    *,
    scenario: ScenarioSpec,
    source: RawSource,
    authority: MovieLensAuthority,
) -> PreparedScenario:
    """Materialize MovieLens-100K 64x128 interaction completion offline."""

    _validate_scenario(scenario)
    if scenario.task is not TaskKind.RECSYS_COMPLETION:
        raise RealEvalDataError("MovieLens materializer received another task")
    if (
        scenario.selection.method != "support_desc_stable_id"
        or scenario.selection.users != 64
        or scenario.selection.items != 128
    ):
        raise RealEvalDataError("MovieLens suite lacks the frozen 64x128 train-side selection")
    content = _read_source(source)
    source_sha256 = _sha256_bytes(content)
    if source_sha256 != authority.source_sha256:
        raise RealEvalDataError("MovieLens authority does not bind retained ZIP bytes")
    if scenario.dataset.dataset_id != authority.dataset_id:
        raise RealEvalDataError("MovieLens authority dataset differs from scenario")
    if scenario.dataset.source_version != authority.source_version:
        raise RealEvalDataError("MovieLens authority source version differs from scenario")
    train, validation, test = _parse_movielens(content, authority)
    selected_users, selected_items = _support_selection(
        train=train,
        users=scenario.selection.users,
        items=scenario.selection.items,
    )
    user_set = set(selected_users)
    item_set = set(selected_items)
    raw_partitions = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    filtered: dict[str, tuple[_Rating, ...]] = {}
    for partition in _PARTITIONS:
        values = tuple(
            sorted(
                (
                    item
                    for item in raw_partitions[partition].values()
                    if item.user_id in user_set and item.item_id in item_set
                ),
                key=lambda item: (int(item.user_id), int(item.item_id)),
            )
        )
        if not values:
            raise RealEvalDataError(
                f"MovieLens selected 64x128 snapshot has no {partition} interactions"
            )
        filtered[partition] = values
    train_ratings = [item.rating for item in filtered["train"]]
    mean, scale = _mean_scale(train_ratings, family_id="rating")
    fitted_state: dict[str, JsonValue] = {
        "schema_version": "tabu.eval-movielens-fitted-state.v1",
        "fit_partition": "train",
        "rating": {"kind": "numeric", "mean": mean, "scale": scale},
        "selected_users": list(selected_users),
        "selected_items": list(selected_items),
        "user_support": {
            value: sum(item.user_id == value for item in train.values())
            for value in selected_users
        },
        "item_support": {
            value: sum(item.item_id == value for item in train.values())
            for value in selected_items
        },
    }
    partitions: dict[str, tuple[PreparedExample, ...]] = {}
    for partition in _PARTITIONS:
        partitions[partition] = tuple(
            PreparedExample(
                example_id=_example_id(
                    dataset_id=scenario.dataset.dataset_id,
                    partition=partition,
                    row_id=item.interaction_id,
                ),
                target_kind=TargetKind.NUMERIC,
                target_family="rating",
                features={},
                target=item.rating,
                context={"user_id": item.user_id, "item_id": item.item_id},
            )
            for item in filtered[partition]
        )
    _check_target_support(partitions)
    checkpoint_projection = {
        "schema_version": CHECKPOINT_PROJECTION_SCHEMA,
        "kind": "recsys",
        "dataset_id": scenario.dataset.dataset_id,
        "selected_users": list(selected_users),
        "selected_items": list(selected_items),
        "rating_family": "rating",
    }
    split_authority_sha256 = canonical_hash(
        {
            "schema_version": "tabu.eval-movielens-split-authority.v1",
            "base_member": authority.base_member,
            "test_member": authority.test_member,
            "validation_interaction_ids": authority.validation_interaction_ids,
            "validation_origin": authority.validation_origin,
        }
    )
    preparation = _preparation(
        scenario=scenario,
        fitted_state=fitted_state,
        authority_sha256=authority.content_hash,
        split_authority_sha256=split_authority_sha256,
        checkpoint_projection=checkpoint_projection,
    )
    source_material = SourceMaterial.from_bytes(
        dataset_id=scenario.dataset.dataset_id,
        content=content,
        media_type="application/zip",
    )
    return _build_prepared(
        scenario=scenario,
        source_material=source_material,
        preparation=preparation,
        partitions=partitions,
    )


def _partition_example(
    prepared: PreparedScenario, example_id: str
) -> tuple[str, PreparedExample]:
    for partition in _PARTITIONS:
        for item in getattr(prepared, partition):
            if item.example_id == example_id:
                return partition, item
    raise RealEvalDataError("checkpoint projection example id is outside PreparedScenario")


def _projection(prepared: PreparedScenario) -> Mapping[str, JsonValue]:
    value = prepared.preparation.preprocessing.get("checkpoint_projection")
    if not isinstance(value, Mapping):
        raise RealEvalDataError("PreparedScenario has no checkpoint projection contract")
    if value.get("schema_version") != CHECKPOINT_PROJECTION_SCHEMA:
        raise RealEvalDataError("unsupported checkpoint projection schema")
    return value


def _feature_specs(value: object) -> tuple[FeatureSpec, ...]:
    if not isinstance(value, list) or not value:
        raise RealEvalDataError("checkpoint projection has no feature specs")
    try:
        return tuple(
            FeatureSpec(
                name=str(item["name"]),
                kind=str(item["kind"]),
                domain=tuple(str(member) for member in item["domain"]),
                codebook_id=(
                    None if item["codebook_id"] is None else str(item["codebook_id"])
                ),
                role=str(item["role"]),
            )
            for item in value
            if isinstance(item, Mapping)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RealEvalDataError("checkpoint projection feature specs are malformed") from error


def _encoded_value(value: object, spec: FeatureSpec) -> float:
    if spec.kind is FeatureKind.NUMERIC:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise RealEvalDataError(f"numeric episode feature {spec.name!r} is malformed")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise RealEvalDataError(f"numeric episode feature {spec.name!r} is not finite")
        return numeric
    try:
        return float(spec.domain.index(str(value)))
    except ValueError as error:
        raise RealEvalDataError(
            f"categorical episode feature {spec.name!r} lies outside train-only codebook"
        ) from error


def _table_episode(
    *,
    prepared: PreparedScenario,
    partition: str,
    query: PreparedExample,
    projection: Mapping[str, JsonValue],
) -> EvidenceEpisode:
    specs = _feature_specs(projection.get("feature_specs"))
    feature_names = tuple(spec.name for spec in specs)
    mode = projection.get("mode")
    if mode not in {"supervised", "completion"}:
        raise RealEvalDataError("checkpoint table projection mode is malformed")
    response_family = projection.get("response_family")
    if not isinstance(response_family, str) or not response_family:
        raise RealEvalDataError("checkpoint table projection has no response family")
    response_columns = tuple(
        index for index, spec in enumerate(specs) if spec.role is FeatureRole.RESPONSE
    )
    if response_columns != (len(specs) - 1,) or specs[-1].name != response_family:
        raise RealEvalDataError(
            "checkpoint table response schema must be the single final feature"
        )
    if mode == "completion" and (
        projection.get("response_visibility") != "schema_only_natural_missing"
    ):
        raise RealEvalDataError(
            "completion response schema must remain natural-missing and value-free"
        )

    support_order: list[str] = []
    support_by_row: dict[str, dict[str, object]] = {}
    for item in prepared.train:
        raw_row_id = item.context.get("row_id")
        raw_masked = item.context.get("artificially_masked_features", [])
        raw_natural = item.context.get("naturally_missing_features", [])
        if (
            not isinstance(raw_row_id, str)
            or not isinstance(raw_masked, list)
            or not isinstance(raw_natural, list)
        ):
            raise RealEvalDataError("checkpoint table support context is malformed")
        if mode == "completion" and (
            response_family in item.features
            or item.target_family == response_family
            or response_family in raw_masked
        ):
            raise RealEvalDataError(
                "completion response values and masks are forbidden from model evidence"
            )
        signature = canonical_hash(
            {
                "schema": "tabu.eval-table-support-row.v1",
                "features": item.features,
                "artificially_masked_features": raw_masked,
                "naturally_missing_features": raw_natural,
                "mask_seed": item.context.get("mask_seed"),
            }
        )
        retained = support_by_row.get(raw_row_id)
        if retained is None:
            retained = {
                "features": dict(item.features),
                "masked": tuple(str(value) for value in raw_masked),
                "natural": tuple(str(value) for value in raw_natural),
                "signature": signature,
                "targets": {},
            }
            support_by_row[raw_row_id] = retained
            support_order.append(raw_row_id)
        elif retained["signature"] != signature:
            raise RealEvalDataError(
                "expanded completion targets disagree on their original train row"
            )
        targets = retained["targets"]
        if not isinstance(targets, dict) or item.target_family in targets:
            raise RealEvalDataError("train row contains a duplicate masked-cell target")
        targets[item.target_family] = item.target

    for retained in support_by_row.values():
        targets = retained["targets"]
        masked = retained["masked"]
        if not isinstance(targets, dict) or not isinstance(masked, tuple):
            raise RealEvalDataError("checkpoint table support aggregation is malformed")
        if mode == "completion" and set(targets) != set(masked):
            raise RealEvalDataError(
                "checkpoint train support does not enumerate every masked-cell truth"
            )
        if mode == "supervised" and len(targets) != 1:
            raise RealEvalDataError("supervised checkpoint support needs one response per row")

    execution = prepared.preparation.preprocessing.get("execution")
    if mode == "completion":
        if not isinstance(execution, Mapping):
            raise RealEvalDataError("completion projection has no target-enumeration execution")
        selected_counts = execution.get("selected_row_counts")
        if not isinstance(selected_counts, Mapping) or selected_counts.get("train") != len(
            support_by_row
        ):
            raise RealEvalDataError("checkpoint train support omits or duplicates selected rows")

    query_row_id = query.context.get("row_id")
    query_masked = query.context.get("artificially_masked_features", [])
    query_natural = query.context.get("naturally_missing_features", [])
    if (
        not isinstance(query_row_id, str)
        or not isinstance(query_masked, list)
        or not isinstance(query_natural, list)
    ):
        raise RealEvalDataError("checkpoint table query context is malformed")
    if mode == "completion" and (
        response_family in query.features
        or query.target_family == response_family
        or response_family in query_masked
    ):
        raise RealEvalDataError(
            "completion response values and masks are forbidden from model evidence"
        )
    row_ids = (
        *(f"train:{row_id}" for row_id in support_order),
        f"{partition}:{query_row_id}",
    )
    values: list[list[float]] = []
    origins: list[list[str]] = []
    roles: list[list[int]] = []
    rows: list[tuple[Mapping[str, object], Mapping[str, object], set[str], set[str], bool]] = []
    for row_id in support_order:
        retained = support_by_row[row_id]
        raw_features = retained["features"]
        raw_targets = retained["targets"]
        raw_masked = retained["masked"]
        raw_natural = retained["natural"]
        if not isinstance(raw_features, Mapping) or not isinstance(raw_targets, Mapping):
            raise RealEvalDataError("checkpoint table support aggregation is malformed")
        rows.append(
            (
                raw_features,
                raw_targets,
                set(str(value) for value in raw_masked),
                set(str(value) for value in raw_natural),
                False,
            )
        )
    rows.append(
        (
            query.features,
            {},
            set(str(value) for value in query_masked),
            set(str(value) for value in query_natural),
            True,
        )
    )
    for features, targets, masked, natural, is_query in rows:
        row_values: list[float] = []
        row_origins: list[str] = []
        row_roles: list[int] = []
        for spec in specs:
            if is_query and spec.name == query.target_family:
                row_values.append(0.0)
                row_origins.append(
                    (
                        OriginState.QUERY
                        if mode == "supervised"
                        else OriginState.ARTIFICIAL_MASK
                    ).value
                )
                row_roles.append(int(ForwardRole.RECEIVER | ForwardRole.TARGET))
                continue
            raw_value = features.get(spec.name)
            if not is_query and spec.name in targets:
                raw_value = targets[spec.name]
            if raw_value is None:
                row_values.append(0.0)
                row_origins.append(
                    OriginState.ARTIFICIAL_MASK.value
                    if spec.name in masked
                    else OriginState.NATURAL_MISSING.value
                )
                row_roles.append(int(ForwardRole.RECEIVER))
            else:
                row_values.append(_encoded_value(raw_value, spec))
                row_origins.append(OriginState.OBSERVED.value)
                row_roles.append(int(ForwardRole.RECEIVER | ForwardRole.SOURCE))
            if spec.name in natural and raw_value is not None:
                raise RealEvalDataError(
                    "natural-missing projection state contradicts visible value"
                )
        values.append(row_values)
        origins.append(row_origins)
        roles.append(row_roles)
    return EvidenceEpisode(
        episode_id=f"episode-{query.example_id}",
        dataset_id=prepared.binding.dataset_id,
        source_partition=partition,
        fit_partition="train",
        row_ids=row_ids,
        feature_names=feature_names,
        feature_specs=specs,
        forward_values=torch.tensor(values, dtype=torch.float32),
        origin_states=origins,
        forward_roles=roles,
        metadata={
            "projection_schema": CHECKPOINT_PROJECTION_SCHEMA,
            **(
                {"response_visibility": "schema_only_natural_missing"}
                if mode == "completion"
                else {}
            ),
        },
    )


def _adjacency(
    *, node_ids: Sequence[str], edges: Sequence[Sequence[str]]
) -> list[list[bool]]:
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    matrix = [[False for _ in node_ids] for _ in node_ids]
    for edge in edges:
        if len(edge) != 2 or edge[0] not in index or edge[1] not in index:
            raise RealEvalDataError("checkpoint graph projection contains an invalid edge")
        left = index[edge[0]]
        right = index[edge[1]]
        matrix[left][right] = True
        matrix[right][left] = True
    return matrix


def _graph_episode(
    *,
    prepared: PreparedScenario,
    partition: str,
    query: PreparedExample,
    projection: Mapping[str, JsonValue],
    override_edges: Sequence[Sequence[str]] | None = None,
) -> EvidenceEpisode:
    specs = _feature_specs(projection.get("feature_specs"))
    raw_node_ids = projection.get("node_ids")
    raw_node_features = projection.get("node_features")
    raw_missing = projection.get("naturally_missing")
    raw_edges = projection.get("edges")
    response_family = projection.get("response_family")
    if (
        not isinstance(raw_node_ids, list)
        or not all(isinstance(value, str) for value in raw_node_ids)
        or not isinstance(raw_node_features, Mapping)
        or not isinstance(raw_missing, Mapping)
        or not isinstance(raw_edges, list)
        or not isinstance(response_family, str)
    ):
        raise RealEvalDataError("checkpoint graph projection is malformed")
    node_ids = tuple(raw_node_ids)
    train_targets = {str(item.context["node_id"]): item.target for item in prepared.train}
    query_node_id = str(query.context["node_id"])
    values: list[list[float]] = []
    origins: list[list[str]] = []
    roles: list[list[int]] = []
    for node_id in node_ids:
        raw_features = raw_node_features.get(node_id)
        natural = raw_missing.get(node_id)
        if not isinstance(raw_features, Mapping) or not isinstance(natural, list):
            raise RealEvalDataError("checkpoint graph node projection is malformed")
        row_values: list[float] = []
        row_origins: list[str] = []
        row_roles: list[int] = []
        for spec in specs:
            if spec.name == response_family:
                if node_id == query_node_id:
                    row_values.append(0.0)
                    row_origins.append(OriginState.ARTIFICIAL_MASK.value)
                    row_roles.append(int(ForwardRole.RECEIVER | ForwardRole.TARGET))
                elif node_id in train_targets:
                    row_values.append(_encoded_value(train_targets[node_id], spec))
                    row_origins.append(OriginState.OBSERVED.value)
                    row_roles.append(int(ForwardRole.RECEIVER | ForwardRole.SOURCE))
                else:
                    row_values.append(0.0)
                    row_origins.append(OriginState.NATURAL_MISSING.value)
                    row_roles.append(int(ForwardRole.RECEIVER))
                continue
            raw_value = raw_features.get(spec.name)
            if raw_value is None:
                if spec.name not in set(str(value) for value in natural):
                    raise RealEvalDataError("graph projection omits an undeclared node feature")
                row_values.append(0.0)
                row_origins.append(OriginState.NATURAL_MISSING.value)
                row_roles.append(int(ForwardRole.RECEIVER))
            else:
                row_values.append(_encoded_value(raw_value, spec))
                row_origins.append(OriginState.OBSERVED.value)
                row_roles.append(int(ForwardRole.RECEIVER | ForwardRole.SOURCE))
        values.append(row_values)
        origins.append(row_origins)
        roles.append(row_roles)
    edges = raw_edges if override_edges is None else list(override_edges)
    topology = GraphTopology(
        node_ids=node_ids,
        adjacency=torch.tensor(_adjacency(node_ids=node_ids, edges=edges), dtype=torch.bool),
        direction="undirected",
    )
    return EvidenceEpisode(
        episode_id=f"episode-{query.example_id}",
        dataset_id=prepared.binding.dataset_id,
        source_partition=partition,
        fit_partition="train",
        row_ids=node_ids,
        feature_names=tuple(spec.name for spec in specs),
        feature_specs=specs,
        forward_values=torch.tensor(values, dtype=torch.float32),
        origin_states=origins,
        forward_roles=roles,
        graph_topology=topology,
        metadata={"projection_schema": CHECKPOINT_PROJECTION_SCHEMA},
    )


def _recsys_episode(
    *,
    prepared: PreparedScenario,
    partition: str,
    query: PreparedExample,
    projection: Mapping[str, JsonValue],
) -> EvidenceEpisode:
    raw_users = projection.get("selected_users")
    raw_items = projection.get("selected_items")
    if (
        not isinstance(raw_users, list)
        or not all(isinstance(value, str) for value in raw_users)
        or not isinstance(raw_items, list)
        or not all(isinstance(value, str) for value in raw_items)
    ):
        raise RealEvalDataError("checkpoint recsys projection is malformed")
    users = tuple(raw_users)
    items = tuple(raw_items)
    user_index = {value: index for index, value in enumerate(users)}
    item_index = {value: index for index, value in enumerate(items)}
    values = [[0.0 for _ in items] for _ in users]
    origins = [[OriginState.NATURAL_MISSING.value for _ in items] for _ in users]
    roles = [[int(ForwardRole.RECEIVER) for _ in items] for _ in users]
    for support in prepared.train:
        user_id = str(support.context["user_id"])
        item_id = str(support.context["item_id"])
        if user_id not in user_index or item_id not in item_index:
            raise RealEvalDataError("train interaction lies outside checkpoint 64x128 projection")
        row = user_index[user_id]
        column = item_index[item_id]
        values[row][column] = float(support.target)
        origins[row][column] = OriginState.OBSERVED.value
        roles[row][column] = int(ForwardRole.RECEIVER | ForwardRole.SOURCE)
    query_user = str(query.context["user_id"])
    query_item = str(query.context["item_id"])
    if query_user not in user_index or query_item not in item_index:
        raise RealEvalDataError("query interaction lies outside checkpoint 64x128 projection")
    row = user_index[query_user]
    column = item_index[query_item]
    values[row][column] = 0.0
    origins[row][column] = OriginState.ARTIFICIAL_MASK.value
    roles[row][column] = int(ForwardRole.RECEIVER | ForwardRole.TARGET)
    specs = tuple(
        FeatureSpec(name=f"item-{item_id}", role=FeatureRole.RESPONSE)
        for item_id in items
    )
    return EvidenceEpisode(
        episode_id=f"episode-{query.example_id}",
        dataset_id=prepared.binding.dataset_id,
        source_partition=partition,
        fit_partition="train",
        row_ids=tuple(f"user-{user_id}" for user_id in users),
        feature_names=tuple(spec.name for spec in specs),
        feature_specs=specs,
        forward_values=torch.tensor(values, dtype=torch.float32),
        origin_states=origins,
        forward_roles=roles,
        metadata={"projection_schema": CHECKPOINT_PROJECTION_SCHEMA},
    )


def _episode_payload(episode: EvidenceEpisode) -> dict[str, JsonValue]:
    topology = episode.graph_topology
    return {
        "schema_version": EPISODE_PAYLOAD_SCHEMA,
        "episode_id": episode.episode_id,
        "dataset_id": episode.dataset_id,
        "source_partition": episode.source_partition,
        "fit_partition": episode.fit_partition,
        "row_ids": list(episode.row_ids),
        "feature_specs": [
            {
                "name": spec.name,
                "kind": spec.kind.value,
                "domain": list(spec.domain),
                "codebook_id": spec.codebook_id,
                "role": spec.role.value,
            }
            for spec in episode.feature_specs
        ],
        "forward_values": episode.forward_values.tolist(),
        "origin_states": episode.origin_states.tolist(),
        "forward_roles": episode.forward_roles.tolist(),
        "graph_topology": (
            None
            if topology is None
            else {
                "node_ids": list(topology.node_ids),
                "adjacency": topology.adjacency.tolist(),
                "direction": topology.direction.value,
            }
        ),
        "metadata": dict(episode.metadata),
    }


def _blind_from_episode(
    *, query: PreparedExample, episode: EvidenceEpisode, readout_row_id: str, readout_feature: str
) -> BlindExample:
    return BlindExample(
        example_id=query.example_id,
        target_kind=query.target_kind,
        target_family=query.target_family,
        features={EPISODE_PAYLOAD_KEY: _episode_payload(episode)},
        context={
            READOUT_SELECTOR_KEY: {
                "schema_version": READOUT_SELECTOR_SCHEMA,
                "row_id": readout_row_id,
                "feature_name": readout_feature,
            }
        },
    )


def checkpoint_blind_example(
    prepared: PreparedScenario, *, example_id: str
) -> BlindExample:
    """Project one validation/test example to a truth-free checkpoint payload.

    The helper is evaluator-side: it may inspect ``PreparedScenario`` truth to
    reconstruct train support, but the selected validation/test target is never
    copied into the returned object.  Train examples are intentionally rejected
    as evaluation queries so their dual support/readout role cannot be ambiguous.
    """

    # Revalidate hashes before using a potentially deserialized or mutated object.
    try:
        prepared = PreparedScenario.model_validate(prepared.model_dump(mode="python"))
    except ValueError as error:
        raise RealEvalDataError("PreparedScenario failed canonical hash revalidation") from error
    partition, query = _partition_example(prepared, example_id)
    if partition == "train":
        raise RealEvalDataError("checkpoint evaluation query must be validation or test")
    projection = _projection(prepared)
    kind = projection.get("kind")
    if kind == "table":
        episode = _table_episode(
            prepared=prepared,
            partition=partition,
            query=query,
            projection=projection,
        )
        readout_row = f"{partition}:{query.context['row_id']}"
        readout_feature = query.target_family
    elif kind == "graph":
        episode = _graph_episode(
            prepared=prepared,
            partition=partition,
            query=query,
            projection=projection,
        )
        readout_row = str(query.context["node_id"])
        readout_feature = query.target_family
    elif kind == "recsys":
        episode = _recsys_episode(
            prepared=prepared,
            partition=partition,
            query=query,
            projection=projection,
        )
        readout_row = f"user-{query.context['user_id']}"
        readout_feature = f"item-{query.context['item_id']}"
    else:
        raise RealEvalDataError("unknown checkpoint projection kind")
    return _blind_from_episode(
        query=query,
        episode=episode,
        readout_row_id=readout_row,
        readout_feature=readout_feature,
    )


def checkpoint_topology_cases(prepared: PreparedScenario) -> tuple[TopologyCheckCase, ...]:
    """Build graph topology pairs with the same explicit checkpoint payload."""

    prepared = PreparedScenario.model_validate(prepared.model_dump(mode="python"))
    projection = _projection(prepared)
    if projection.get("kind") != "graph":
        raise RealEvalDataError("topology checkpoint projection requires graph data")
    raw_edges = projection.get("edges")
    raw_perturbations = projection.get("perturbation_edges")
    if not isinstance(raw_edges, list) or not isinstance(raw_perturbations, Mapping):
        raise RealEvalDataError("graph checkpoint perturbation contract is malformed")
    retained: list[TopologyCheckCase] = []
    for case in prepared.topology_checks:
        partition, query = _partition_example(prepared, case.base_example_id)
        raw_edge = raw_perturbations.get(case.check_id)
        if not isinstance(raw_edge, list) or len(raw_edge) != 2:
            raise RealEvalDataError("graph checkpoint perturbation edge is malformed")
        toggled = _toggle_edge(
            tuple(tuple(str(value) for value in edge) for edge in raw_edges),
            (str(raw_edge[0]), str(raw_edge[1])),
        )
        episode = _graph_episode(
            prepared=prepared,
            partition=partition,
            query=query,
            projection=projection,
            override_edges=toggled,
        )
        blind = _blind_from_episode(
            query=query,
            episode=episode,
            readout_row_id=str(query.context["node_id"]),
            readout_feature=query.target_family,
        ).model_copy(update={"example_id": case.perturbed_example.example_id})
        retained.append(
            TopologyCheckCase(
                check_id=case.check_id,
                base_example_id=case.base_example_id,
                perturbed_example=blind,
                expected_relation=case.expected_relation,
            )
        )
    return tuple(retained)


__all__ = [
    "CHECKPOINT_PROJECTION_SCHEMA",
    "MATERIALIZER_SCHEMA",
    "ColumnAuthority",
    "CompletionMaskAuthority",
    "DelimitedTableAuthority",
    "GraphPerturbationAuthority",
    "KarateAuthority",
    "MovieLensAuthority",
    "RealEvalDataError",
    "SplitAuthority",
    "checkpoint_blind_example",
    "checkpoint_topology_cases",
    "materialize_karate",
    "materialize_movielens",
    "materialize_table_completion",
    "materialize_table_supervised",
]
