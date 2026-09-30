"""Consume frozen legacy tables without importing Torch or owning data supply.

Hashes prove identity at their stated level, not data quality or unseen-world
generalization. Source manifests and data stay in place; no registry is written.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _digest(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        c not in "0123456789abcdef" for c in value
    ):
        raise ValueError("expected a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class Provenance:
    """Reference an upstream record; missing world identity remains missing."""

    manifest_path: Path
    manifest_sha256: str
    record_id: str
    generation_seed: int | None = None
    split_seed: int | None = None
    world_id: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).resolve())
        _digest(self.manifest_sha256)
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("record_id must be nonempty")
        for seed in (self.generation_seed, self.split_seed):
            if seed is not None and (type(seed) is not int or seed < 0):
                raise ValueError("provenance seeds must be nonnegative integers or None")
        if self.world_id is not None and (
            not isinstance(self.world_id, str) or not self.world_id
        ):
            raise ValueError("world_id must be nonempty or None")

    def verify(self, *, expected_data_sha256=None) -> bool:
        """Check bytes and, for a supported source schema, the actual record.

        True means the claimed world ID is bound to a parsed legacy record.
        Generic provenance references remain caller-declared metadata and cannot
        certify world separation merely by assigning different strings.
        """
        raw = self.manifest_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest_sha256:
            raise ValueError("source manifest digest mismatch")
        source = json.loads(raw)
        if source.get("schema") != "tabu.tar.diverse-fit-corpus.1":
            return False
        records = [r for r in source.get("records", []) if r.get("dataset") == self.record_id]
        if len(records) != 1:
            raise ValueError("source record identity must resolve exactly once")
        record = records[0]
        if (self.generation_seed != record.get("request", {}).get("seed")
                or self.split_seed != source.get("split_seed")
                or self.world_id != record.get("realized", {}).get("world_hash")):
            raise ValueError("provenance differs from the source manifest record")
        if expected_data_sha256 is not None and record.get("data_sha256") != expected_data_sha256:
            raise ValueError("dataset digest is not bound to the source record")
        return self.world_id is not None

    def identity(self):
        return {k: v for k, v in asdict(self).items() if k != "manifest_path"}


@dataclass(frozen=True)
class BoundTable:
    """A metadata-only handle; raw labels never appear in its manifest record."""

    table_id: str
    cohort: str
    kind: str
    path: Path
    sha256: str
    content_sha256: str
    split_sha256: str
    schema: str
    width: int
    splits: tuple[tuple[str, tuple[int, ...]], ...]
    provenance: Provenance

    def rows(self, partition: str) -> tuple[int, ...]:
        return dict(self.splits)[partition]

    def identity(self) -> dict:
        result = asdict(self)
        del result["path"]
        result["provenance"] = self.provenance.identity()
        return result

    def verify(self):
        """Rebind before consumption; fail if either data or provenance drifted."""
        rebound = bind_table(
            self.path, expected_sha256=self.sha256, table_id=self.table_id,
            cohort=self.cohort, kind=self.kind, provenance=self.provenance,
        )
        if rebound != self:
            raise ValueError("bound table metadata mismatch")


def bind_table(
    path: Path | str, *, expected_sha256: str, table_id: str, cohort: str,
    kind: str, provenance: Provenance,
) -> BoundTable:
    """Bind existing typed-fit JSON bytes and row splits, without transforming data.

    Full type/domain and episode feasibility validation belongs to the consumer.
    Content identity ignores row order, split assignment and cosmetic metadata,
    retaining column order. It is not an arbitrary-transformation duplicate test.
    """
    if any(not isinstance(v, str) or not v.strip() for v in (table_id, cohort)):
        raise ValueError("table_id and cohort must be nonempty strings")
    if kind not in ("synthetic", "real"):
        raise ValueError("kind must be synthetic or real")
    _digest(expected_sha256)
    provenance.verify(expected_data_sha256=expected_sha256)
    path = Path(path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"dataset digest mismatch: {table_id}")
    payload = json.loads(raw)
    if payload.get("schema") != "tabu.tar.typed-fit-table.1":
        raise ValueError("only the existing typed-fit-table.1 format is supported")
    values = payload.get("values")
    if (not isinstance(values, list) or not values or not isinstance(values[0], list)
            or not values[0]):
        raise ValueError("values must be a nonempty rectangular numeric matrix")
    width = len(values[0])
    for row in values:
        if not isinstance(row, list) or len(row) != width or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in row
        ):
            raise ValueError("values must be finite rectangular numeric data")
    if not isinstance(payload.get("features"), list) or len(payload["features"]) != width:
        raise ValueError("features must declare every column")
    splits = payload.get("splits")
    if (not isinstance(splits, dict) or not {"train", "test"} <= splits.keys()
            or splits.keys() - {"train", "validation", "test"}):
        raise ValueError("explicit train/test splits and optional validation required")
    seen, normalized = set(), []
    for name in ("train", "validation", "test"):
        ids = splits.get(name, [])
        if (not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(values)
                                           for i in ids)
                or len(set(ids)) != len(ids) or seen.intersection(ids)):
            raise ValueError("split addresses must be unique, disjoint and in bounds")
        seen.update(ids)
        normalized.append((name, tuple(ids)))
    if seen != set(range(len(values))) or len(splits["train"]) < 3:
        raise ValueError("splits must cover all rows and include at least three training rows")
    # Normalize integral JSON numbers, then sort whole rows (multiplicity kept).
    rows = sorted(json.dumps([int(v) if v == int(v) else v for v in row],
                            separators=(",", ":")) for row in values)
    return BoundTable(
        table_id, cohort, kind, path, expected_sha256, _hash(rows), _hash(normalized),
        payload["schema"], width, tuple(normalized), provenance,
    )


def bind_legacy_corpus(manifest_path: Path | str, *, cohort: str) -> tuple[BoundTable, ...]:
    """Bind a recorded diverse-fit corpus in place, checking every table SHA.

    Legacy names are reused between old120/new120, so consumer IDs are namespaced
    by cohort. Source record names and generation/split seeds remain unchanged.
    """
    manifest_path = Path(manifest_path).resolve()
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "tabu.tar.diverse-fit-corpus.1":
        raise ValueError("unsupported corpus schema")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("corpus records must be nonempty")
    tables, seen = [], set()
    for record in records:
        name = record["dataset"]
        if (not isinstance(name, str) or not name or Path(name).name != name
                or name in (".", "..") or name in seen):
            raise ValueError("source record names must be unique path components")
        seen.add(name)
        provenance = Provenance(
            manifest_path, hashlib.sha256(raw).hexdigest(), name,
            record.get("request", {}).get("seed"), manifest.get("split_seed"),
            record.get("realized", {}).get("world_hash"),
        )
        tables.append(bind_table(
            manifest_path.parent / "data" / f"{name}.json",
            expected_sha256=record["data_sha256"], table_id=f"{cohort}/{name}",
            cohort=cohort, kind="synthetic", provenance=provenance,
        ))
    return tuple(tables)


def require_disjoint(previous, incoming, *, level="table") -> dict:
    """Reject renamed/re-split copies and known world overlap, without seed inference.

    Table level checks exact bytes and row-order-invariant content. World level
    additionally requires source-declared world IDs for every table. Missing
    identities are reported at table level, and fail closed at world level.
    """
    previous, incoming = tuple(previous), tuple(incoming)
    if not previous or not incoming or level not in ("table", "world"):
        raise ValueError("two nonempty collections and table/world level are required")
    for field in ("sha256", "content_sha256"):
        if {getattr(t, field) for t in previous} & {getattr(t, field) for t in incoming}:
            raise ValueError(f"old/new overlap at {field}")
    old_worlds = {t.provenance.world_id for t in previous} - {None}
    new_worlds = {t.provenance.world_id for t in incoming} - {None}
    if old_worlds & new_worlds:
        raise ValueError("old/new overlap at source-declared world_id")
    unknown = [t.table_id for t in (*previous, *incoming)
               if not t.provenance.verify(expected_data_sha256=t.sha256)]
    if level == "world" and unknown:
        raise ValueError("world separation cannot be established with missing world identities")
    return {"level": level, "old_tables": len(previous), "new_tables": len(incoming),
            "unknown_world_ids": unknown, "world_separation_established": not unknown}


@dataclass(frozen=True)
class StageSelection:
    name: str
    question: str
    tables: tuple[BoundTable, ...]
    fit_mode: str
    previous: tuple[BoundTable, ...] = ()
    separation_level: str | None = None
    expected_count: int | None = None
    kind_requirement: str | None = None
    require_reserved: bool = False

    def manifest(self) -> dict:
        """Portable metadata record, never a successful training receipt."""
        body = {"schema": "tabu.curriculum.selection.v1", "stage": self.name,
                "question": self.question,
                "fit_mode": self.fit_mode, "status": "planned",
                "tables": [t.identity() for t in self.tables],
                "previous": [t.identity() for t in self.previous],
                "separation_level": self.separation_level,
                "expected_count": self.expected_count, "kind_requirement": self.kind_requirement,
                "require_reserved": self.require_reserved}
        return {**body, "sha256": _hash(body)}


def select_stage(name: str, tables, *, question: str, previous=(), fit_mode="shared",
                 expected_count=None, kind=None, separation_level=None,
                 require_reserved=False) -> StageSelection:
    """Compose a named capability probe, without a fixed stage order or size.

    Actual scheduling, budgets, replay exposure and checkpoint lineage remain in
    a separately identified execution manifest. Counts, domains, separation level
    and fitting mode are declared policies, not universal learning laws.
    """
    tables, previous = tuple(tables), tuple(previous)
    if any(not isinstance(v, str) or not v.strip() for v in (name, question)) or not tables:
        raise ValueError("stage name, question and table selection must be nonempty")
    if expected_count is not None and (
        type(expected_count) is not int or expected_count < 1 or len(tables) != expected_count
    ):
        raise ValueError("table count differs from the declared policy")
    if fit_mode not in ("shared", "independent"):
        raise ValueError("fit_mode must declare shared or independent parameters")
    if len({t.table_id for t in tables}) != len(tables):
        raise ValueError("stage table IDs must be unique")
    if len({t.content_sha256 for t in tables}) != len(tables):
        raise ValueError("stage cannot inflate table count with duplicate content")
    if kind is not None and (kind not in ("real", "synthetic") or
                             any(t.kind != kind for t in tables)):
        raise ValueError(f"{name} requires {kind} provenance labels")
    if type(require_reserved) is not bool:
        raise ValueError("require_reserved must be boolean")
    if require_reserved and any(not t.rows("test") for t in tables):
        raise ValueError("this policy requires explicit reserved test rows")
    if separation_level is not None:
        require_disjoint(previous, tables, level=separation_level)
    return StageSelection(name, question, tables, fit_mode, previous, separation_level,
                          expected_count, kind, require_reserved)


def reference_stage(policy: str, tables, *, name=None, **overrides) -> StageSelection:
    """Revisable V4/V5-derived defaults; use select_stage for any other probe.

    For example reference_stage('fit8', tables, name='fit16', expected_count=16)
    changes the empirical table budget explicitly. No policy installs a quality
    threshold, transfers weights, starts training, or automatically advances.
    """
    defaults = {
        "fit8": {"question": "Can the core mechanism fit the small fixed table panel?",
                 "expected_count": 8, "kind": "synthetic"},
        "multi120": {"question": "Can one parameter state jointly fit the fixed table panel?",
                     "expected_count": 120, "kind": "synthetic"},
        "continual120": {"question": "Can new tables be learned while retaining prior fit?",
                         "expected_count": 120, "kind": "synthetic",
                         "separation_level": "table"},
        "real12": {"question": "How do known-table fit and reserved-row R-squared differ?",
                   "expected_count": 12, "kind": "real", "require_reserved": True},
    }
    if policy not in defaults:
        raise ValueError("unknown reference policy; use select_stage for a custom probe")
    return select_stage(name or policy, tables, **(defaults[policy] | overrides))
