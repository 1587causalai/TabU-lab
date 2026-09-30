"""Pure, checkpointable 120-table normal-cycle / loss-ranked replay state.

Normal updates use the existing schedule with ``normal_cursor``. Once all 120
normal losses exist, freeze one ranking. V1 uses top6 then top24 (30 extras).
V2 uses three top2 passes, two top6 passes and one top24 pass (42 extras).
V2 retains inherited cumulative extras and records their per-table baseline.
Extra episode indices are cumulative and use a versioned RNG namespace.

Every transition returns owned JSON-safe containers. No RNG, tensors, files or
training side effects belong here. ``cycle_index`` counts completed normal
cycles over the whole stage, including cycles before a boundary migration.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Iterable

_TABLES = 120
V1 = "normal120_top5_top20_v1"
V2 = "normal120_p99x3_p95x2_p80x1_v2"
_PASSES = {
    V1: ((6, 1, "extra_top6"), (24, 1, "extra_top24")),
    V2: ((2, 3, "extra_top2"), (6, 2, "extra_top6"), (24, 1, "extra_top24")),
}
_FIELDS = {
    "normal_cursor", "extra_updates", "cycle_index", "partial", "queue",
    "queue_cursor", "extra_by_table",
}


def extras_per_cycle(kind: str) -> int:
    if not isinstance(kind, str) or kind not in _PASSES:
        raise ValueError("unsupported loss_replay kind")
    return sum(count * repeats for count, repeats, _ in _PASSES[kind])


def _names(table_names: Iterable[str]) -> tuple[str, ...]:
    try:
        names = tuple(table_names)
    except TypeError as error:
        raise ValueError("table_names must contain exactly 120 table names") from error
    if (len(names) != _TABLES
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != _TABLES):
        raise ValueError("table_names must contain exactly 120 distinct nonempty names")
    return tuple(sorted(names))


def _integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _loss(value) -> float:
    if type(value) not in (int, float):
        raise ValueError("normal loss must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("normal loss must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError("normal loss must be a finite number")
    return result


def _ranked_queue(partial: dict, kind: str = V1) -> list[dict]:
    ranked = sorted(partial, key=lambda table: (-partial[table], table))
    return [
        {"table": table, "update_kind": update_kind}
        for count, repeats, update_kind in _PASSES[kind]
        for _ in range(repeats)
        for table in ranked[:count]
    ]


def validate_state(state: dict, table_names: Iterable[str]) -> None:
    """Reject malformed or internally inconsistent recovery state.

    The caller supplies the manifest's table registry on checkpoint loading.
    Historical rankings are not retained, so completed-cycle table counts are
    checked for necessary bounds, while the pending cycle is checked exactly.
    """
    names = _names(table_names)
    expected = set(names)
    if not isinstance(state, dict):
        raise ValueError("loss replay state fields do not match the contract")
    kind = state.get("kind", V1)
    per_cycle = extras_per_cycle(kind)
    fields = _FIELDS | {"kind", "base_extra_by_table"} if kind == V2 else _FIELDS
    if set(state) != fields:
        raise ValueError("loss replay state fields do not match the contract")
    normal = _integer(state["normal_cursor"], "normal_cursor")
    extras = _integer(state["extra_updates"], "extra_updates")
    cycle = _integer(state["cycle_index"], "cycle_index")
    cursor = _integer(state["queue_cursor"], "queue_cursor")
    if cycle != normal // _TABLES:
        raise ValueError("cycle_index does not match normal_cursor")
    partial, queue, counts = state["partial"], state["queue"], state["extra_by_table"]
    if not isinstance(partial, dict) or not set(partial) <= expected:
        raise ValueError("partial losses contain an unknown table or invalid mapping")
    for value in partial.values():
        _loss(value)
    if not isinstance(counts, dict) or set(counts) != expected:
        raise ValueError("extra_by_table must contain the manifest's exact 120 tables")
    for value in counts.values():
        _integer(value, "per-table extra count")
    if sum(counts.values()) != extras:
        raise ValueError("extra_updates does not equal per-table extra counts")
    base = dict.fromkeys(names, 0)
    if kind == V2:
        base = state["base_extra_by_table"]
        if not isinstance(base, dict) or set(base) != expected:
            raise ValueError("base_extra_by_table must contain the manifest's exact 120 tables")
        for name, value in base.items():
            if _integer(value, "base per-table extra count") > counts[name]:
                raise ValueError("per-table extras cannot precede their inherited baseline")
    new_extras = extras - sum(base.values())
    if not isinstance(queue, list):
        raise ValueError("queue must be a JSON list")
    if queue:
        if normal % _TABLES or set(partial) != expected or not cycle:
            raise ValueError("pending replay requires one complete normal cycle")
        if len(queue) != per_cycle or cursor >= per_cycle:
            raise ValueError(f"pending queue must have {per_cycle} entries "
                             f"and cursor below {per_cycle}")
        if queue != _ranked_queue(partial, kind):
            raise ValueError("queue does not match the frozen loss ranking and replay passes")
        if new_extras % per_cycle != cursor:
            raise ValueError("extra_updates does not match pending queue_cursor")
    else:
        if cursor or new_extras % per_cycle:
            raise ValueError("an empty queue cannot have partially consumed extras")
        if len(partial) != normal % _TABLES:
            raise ValueError("partial loss count does not match normal_cursor")

    completed = (new_extras - cursor) // per_cycle
    if completed + bool(queue) > cycle:
        raise ValueError("extra cycles exceed completed normal cycles")
    consumed = Counter(entry["table"] for entry in queue[:cursor])
    history = [counts[name] - base[name] - consumed[name] for name in names]
    maximum = 6 if kind == V2 else 2
    if any(count < 0 or count > maximum * completed for count in history):
        raise ValueError("per-table extras disagree with consumed queue or cycle bounds")
    if kind == V1:
        # Each completed replay cycle has exactly six double-selected tables.
        if not (sum(max(count - completed, 0) for count in history)
                <= 6 * completed <= sum(count // 2 for count in history)):
            raise ValueError("per-table extra counts cannot support "
                             "six double selections per cycle")
    else:
        # Necessary historical bounds; the current frozen queue is checked exactly.
        # A single cycle contributes [6,6,3,3,3,3,1*18,0*96]. Cumulative counts
        # cannot concentrate more mass than repeated identical rankings would.
        profile = [6] * 2 + [3] * 4 + [1] * 18 + [0] * 96
        actual_prefix = maximum_prefix = 0
        for actual, allowed in zip(sorted(history, reverse=True), profile, strict=True):
            actual_prefix += actual
            maximum_prefix += allowed * completed
            if actual_prefix > maximum_prefix:
                raise ValueError("per-table extras exceed v2 ranked cycle bounds")
        if (sum(count // 6 for count in history) < 2 * completed
                or sum(count // 3 for count in history) < 8 * completed):
            raise ValueError("per-table extras cannot support v2 top2/top6 multiplicities")


def new_state(table_names: Iterable[str], start_normal_cursor: int = 0, *,
              kind: str = V1, base_extra_by_table: dict | None = None) -> dict:
    """Start at a normal boundary; v2 may inherit cumulative per-table extras.

    The caller must independently verify the parent finished its replay queue.
    This constructor cannot establish parent lineage from counters alone.
    """
    names = _names(table_names)
    extras_per_cycle(kind)
    if kind == V1 and base_extra_by_table is not None:
        raise ValueError("v1 cannot inherit an extra baseline")
    start = _integer(start_normal_cursor, "start_normal_cursor")
    if start % _TABLES:
        raise ValueError("loss replay migration requires a 120-update normal boundary")
    state = {
        "normal_cursor": start,
        "extra_updates": 0,
        "cycle_index": start // _TABLES,
        "partial": {},
        "queue": [],
        "queue_cursor": 0,
        "extra_by_table": dict.fromkeys(names, 0),
    }
    if kind == V2:
        base = (dict.fromkeys(names, 0) if base_extra_by_table is None
                else copy.deepcopy(base_extra_by_table))
        if not isinstance(base, dict) or set(base) != set(names):
            raise ValueError("base_extra_by_table must contain the manifest's exact 120 tables")
        for value in base.values():
            _integer(value, "base per-table extra count")
        state.update(kind=V2, base_extra_by_table=base,
                     extra_by_table=copy.deepcopy(base), extra_updates=sum(base.values()))
    validate_state(state, names)
    return state


def _validate_owned_registry(state: dict) -> None:
    if not isinstance(state, dict) or not isinstance(state.get("extra_by_table"), dict):
        raise ValueError("loss replay state lacks its table registry")
    validate_state(state, state["extra_by_table"])


def next_extra(state: dict) -> dict | None:
    """Return the next replay address, without consuming or mutating it."""
    _validate_owned_registry(state)
    if not state["queue"]:
        return None
    entry = state["queue"][state["queue_cursor"]]
    return {
        **entry,
        "table_episode_index": state["extra_by_table"][entry["table"]],
    }


def record_normal(state: dict, table: str, loss: float) -> dict:
    """Record one successful normal update and freeze ranking at step 120."""
    _validate_owned_registry(state)
    if state["queue"]:
        raise ValueError("finish the pending replay queue before another normal update")
    if not isinstance(table, str) or table not in state["extra_by_table"]:
        raise ValueError("normal update names an unknown table")
    if table in state["partial"]:
        raise ValueError("duplicate normal table within one 120-table cycle")
    value = _loss(loss)
    result = copy.deepcopy(state)
    result["partial"][table] = value
    result["normal_cursor"] += 1
    if len(result["partial"]) == _TABLES:
        result["cycle_index"] += 1
        result["queue"] = _ranked_queue(result["partial"], result.get("kind", V1))
    return result


def record_extra(state: dict) -> dict:
    """Consume one successfully trained replay entry; never rescore its rank."""
    _validate_owned_registry(state)
    if not state["queue"]:
        raise ValueError("no pending extra update to record")
    result = copy.deepcopy(state)
    entry = result["queue"][result["queue_cursor"]]
    result["extra_by_table"][entry["table"]] += 1
    result["extra_updates"] += 1
    result["queue_cursor"] += 1
    if result["queue_cursor"] == extras_per_cycle(result.get("kind", V1)):
        result["partial"] = {}
        result["queue"] = []
        result["queue_cursor"] = 0
    return result


__all__ = ["V1", "V2", "extras_per_cycle", "new_state", "next_extra", "record_extra",
           "record_normal", "validate_state"]
