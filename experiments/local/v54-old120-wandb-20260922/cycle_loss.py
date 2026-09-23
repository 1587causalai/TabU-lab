"""Exact 120-table cycle loss from every raw training-journal update.

``initial_state(table_names, base_update=...)`` creates JSON-safe state.
``advance(state, rows)`` returns ``(new_state, complete_cycle_events)`` without
mutating either input. The caller commits state and events in one transaction.
An invalid row rejects the entire supplied batch, including earlier full cycles
in that batch. No W&B dependency, training imports, or filesystem access.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

CYCLE_SIZE = 120
SCHEMA = "tabu.old120.cycle-loss.v1"
REPLAY_SCHEMA = "tabu.old120.replay-cycle-loss.v2"
WEIGHTED_REPLAY_SCHEMA = "tabu.old120.replay-cycle-loss.v3"
LEGACY_REPLAY_KIND = "normal120_top5_top20_v1"
WEIGHTED_REPLAY_KIND = "normal120_p99x3_p95x2_p80x1_v2"
# (journal kind, selected tables, complete passes), all from one frozen rank.
REPLAY_GROUPS = {
    LEGACY_REPLAY_KIND: (("extra_top6", 6, 1), ("extra_top24", 24, 1)),
    WEIGHTED_REPLAY_KIND: (("extra_top2", 2, 3), ("extra_top6", 6, 2), ("extra_top24", 24, 1)),
}


def _integer(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _loss(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("loss must be a finite JSON number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("loss must be a finite JSON number") from error
    if not math.isfinite(result):
        raise ValueError("loss must be finite")
    return result


def _tables(table_names: Iterable[str]) -> list[str]:
    names = list(table_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("table names must be non-empty strings")
    if len(names) != CYCLE_SIZE or len(set(names)) != CYCLE_SIZE:
        raise ValueError("a cycle must declare exactly 120 distinct tables")
    return sorted(names)


def _linear_quantile(ordered_losses: list[float], q: float) -> float:
    """Linear interpolation at zero-based position ``(n - 1) * q``."""
    position = (len(ordered_losses) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    # Weighted endpoints avoid overflow in (upper_value - lower_value).
    return math.fsum(((1. - fraction) * ordered_losses[lower],
                      fraction * ordered_losses[upper]))


def initial_state(table_names: Iterable[str], *, base_update: int = 0) -> dict:
    """Start at an audited balanced-cycle boundary, with zero new updates.

    Fresh Small and H4 use ``base_update=0``; the Nano extension uses 122880.
    ``base_update`` identifies global old120 updates already consumed before
    this observer's declared experiment/extension. It is not extra exposure.
    """
    return {"schema": SCHEMA, "base_update": _integer(base_update, "base_update"),
            "table_names": _tables(table_names), "new_updates": 0, "partial": []}


def _validated_copy(state: Mapping[str, Any]) -> dict:
    if not isinstance(state, Mapping) or state.get("schema") != SCHEMA:
        raise ValueError("unsupported cycle-loss state schema")
    try:
        result = initial_state(state["table_names"], base_update=state["base_update"])
        processed = _integer(state["new_updates"], "new_updates")
        partial = state["partial"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete cycle-loss state") from error
    if not isinstance(partial, list) or len(partial) != processed % CYCLE_SIZE:
        raise ValueError("partial cycle length disagrees with processed updates")
    allowed = set(result["table_names"])
    seen = set()
    for entry in partial:
        if not isinstance(entry, Mapping) or "table" not in entry or "loss" not in entry:
            raise ValueError("malformed partial cycle entry")
        table = entry["table"]
        if not isinstance(table, str) or table not in allowed or table in seen:
            raise ValueError("partial cycle contains an unknown or repeated table")
        seen.add(table)
        result["partial"].append({"table": table, "loss": _loss(entry["loss"])})
    result["new_updates"] = processed
    return result


def advance(state: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """Consume all journal rows, returning new JSON-safe state and cycle events.

    Each row needs ``update`` (global integer), ``table``, and ``loss``. Extra
    journal fields are ignored. Updates must be consecutive from
    ``base_update + new_updates + 1``. Within each consecutive 120 new updates,
    every declared table must occur exactly once; row ordering may vary.

    Events have 1-based ``cycle_index``, ``end_new_update``,
    ``end_global_update``, ``mean_loss`` = ``math.fsum(losses) / 120``,
    ``median_loss`` (ordinary sample median), and ``p05_loss`` / ``p95_loss``
    (linear interpolation of sorted raw losses at ``(120 - 1) * q``), plus
    ``table_count`` = 120. An incomplete cycle emits no event and persists its
    raw losses in ``partial``; no average of sampled or rolling means is used.
    """
    result = _validated_copy(state)
    allowed = set(result["table_names"])
    seen = {entry["table"] for entry in result["partial"]}
    events = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("journal row must be a mapping")
        try:
            update = _integer(row["update"], "update")
            table = row["table"]
            loss = _loss(row["loss"])
        except (KeyError, TypeError) as error:
            raise ValueError("journal row lacks update, table, or loss") from error
        if update != result["base_update"] + result["new_updates"] + 1:
            raise ValueError("journal update gap, duplicate, or out-of-order row")
        if not isinstance(table, str) or table not in allowed:
            raise ValueError("journal table is outside the declared 120-table set")
        if table in seen:
            raise ValueError("table repeated within a 120-update cycle")
        seen.add(table)
        result["partial"].append({"table": table, "loss": loss})
        result["new_updates"] += 1
        if len(result["partial"]) == CYCLE_SIZE:
            if seen != allowed:
                raise ValueError("completed cycle does not contain every declared table")
            try:
                mean_loss = math.fsum(entry["loss"] for entry in result["partial"]) / CYCLE_SIZE
            except OverflowError as error:
                raise ValueError("cycle loss sum is not finite") from error
            if not math.isfinite(mean_loss):
                raise ValueError("cycle mean loss is not finite")
            ordered_losses = sorted(entry["loss"] for entry in result["partial"])
            events.append({"cycle_index": result["new_updates"] // CYCLE_SIZE,
                           "end_new_update": result["new_updates"],
                           "end_global_update": update, "mean_loss": mean_loss,
                           "median_loss": _linear_quantile(ordered_losses, .5),
                           "p05_loss": _linear_quantile(ordered_losses, .05),
                           "p95_loss": _linear_quantile(ordered_losses, .95),
                           "table_count": CYCLE_SIZE})
            result["partial"] = []
            seen = set()
    return result, events


def initial_replay_state(table_names: Iterable[str], base_update: int = 0,
                         base_normal_update: int = 0, *,
                         policy_kind: str = LEGACY_REPLAY_KIND) -> dict:
    """Start a new 120-normal + top6 + top24 continuation at a normal boundary.

    A new continuation begins with normal training, not retroactive extras for
    the parent's last cycle. Checkpoint resumption within this continuation
    must restore the full returned state, including any pending extra phase.
    """
    if policy_kind == WEIGHTED_REPLAY_KIND:
        return _initial_weighted_replay_state(table_names, base_update, base_normal_update)
    if policy_kind != LEGACY_REPLAY_KIND:
        raise ValueError("unsupported replay policy kind")
    total = _integer(base_update, "base_update")
    normal = _integer(base_normal_update, "base_normal_update")
    if normal % CYCLE_SIZE or normal > total:
        raise ValueError("replay base must be a complete normal-cycle boundary within total updates")
    return {"schema": REPLAY_SCHEMA, "base_update": total,
            "base_normal_update": normal, "table_names": _tables(table_names),
            "global_update": total, "normal_update": normal, "new_updates": 0,
            "partial": [], "phase": "normal", "phase_tables": [],
            "top6_tables": [], "top24_tables": [],
            "extra_updates": {"extra_top6": 0, "extra_top24": 0}}


def _replay_validated_copy(state: Mapping[str, Any]) -> dict:
    if not isinstance(state, Mapping) or state.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported replay cycle-loss state schema")
    try:
        result = initial_replay_state(state["table_names"], state["base_update"],
                                      state["base_normal_update"])
        normal_state = _validated_copy({"schema": SCHEMA,
            "base_update": state["base_normal_update"], "table_names": state["table_names"],
            "new_updates": state["new_updates"], "partial": state["partial"]})
        result.update(new_updates=normal_state["new_updates"], partial=normal_state["partial"],
                      global_update=_integer(state["global_update"], "global_update"),
                      normal_update=_integer(state["normal_update"], "normal_update"))
        extras = state["extra_updates"]
        if not isinstance(extras, Mapping) or set(extras) != {"extra_top6", "extra_top24"}:
            raise ValueError("invalid extra-update counters")
        result["extra_updates"] = {key: _integer(extras[key], key) for key in extras}
        phase = state["phase"]
        if phase not in ("normal", "extra_top6", "extra_top24"):
            raise ValueError("unknown replay phase")
        result["phase"] = phase
        allowed = set(result["table_names"])
        for key in ("phase_tables", "top6_tables", "top24_tables"):
            values = state[key]
            if (not isinstance(values, list)
                    or any(not isinstance(t, str) or t not in allowed for t in values)
                    or len(values) != len(set(values))):
                raise ValueError("invalid replay table selection")
            result[key] = list(values)
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete replay cycle-loss state") from error
    if result["normal_update"] != result["base_normal_update"] + result["new_updates"]:
        raise ValueError("normal counter disagrees with processed normal rows")
    if result["global_update"] != result["base_update"] + result["new_updates"] + sum(result["extra_updates"].values()):
        raise ValueError("optimizer counter disagrees with normal plus extra rows")
    cycles = result["new_updates"] // CYCLE_SIZE
    selected = len(result["phase_tables"])
    if phase == "normal":
        if result["phase_tables"] or result["top6_tables"] or result["top24_tables"]:
            raise ValueError("normal phase cannot retain pending extra selections")
        expected6, expected24 = cycles * 6, cycles * 24
    else:
        if not cycles or result["partial"] or result["new_updates"] % CYCLE_SIZE:
            raise ValueError("extra phase requires a completed normal cycle")
        if (len(result["top6_tables"]) != 6 or len(result["top24_tables"]) != 24
                or result["top6_tables"] != result["top24_tables"][:6]):
            raise ValueError("top6/top24 selections do not have the required nested sizes")
        selected_from = result["top6_tables"] if phase == "extra_top6" else result["top24_tables"]
        if not set(result["phase_tables"]) <= set(selected_from) or selected >= len(selected_from):
            raise ValueError("extra phase cursor disagrees with its selected tables")
        expected6 = (cycles - 1) * 6 + selected if phase == "extra_top6" else cycles * 6
        expected24 = (cycles - 1) * 24 + (selected if phase == "extra_top24" else 0)
    if result["extra_updates"] != {"extra_top6": expected6, "extra_top24": expected24}:
        raise ValueError("extra counters disagree with the replay phase")
    return result


def _advance_replay_legacy(state: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """Consume the full normal/extra stream; emit statistics for normal120 only.

    Required row fields: ``update`` (all optimizer updates), ``normal_update``
    (normal rows only), ``update_kind``, ``table`` and finite ``loss``. Total
    updates must increase by one on every row; normal updates only on normal
    rows. A normal block contains every table once. Its original losses select
    top6 and top24 using ``(-loss, table_id)``; the two extra blocks visit each
    corresponding selection once, with arbitrary within-block order. Thus the
    top6 tables correctly occur in both extra blocks. Extra losses never alter
    this selection or the normal-cycle distribution.

    Returned ``new_updates`` counts normal updates relative to this
    continuation; ``global_update`` is the actual optimizer cursor. Events use
    ``end_new_update`` for relative normal count, ``end_normal_update`` for
    absolute normal count, and ``end_global_update`` for actual total count.
    The 1-based ``cycle_index`` is relative to this continuation. State and
    event persistence must be one caller transaction; errors mutate no input.
    """
    result = _replay_validated_copy(state)
    allowed = set(result["table_names"])
    normal_seen = {entry["table"] for entry in result["partial"]}
    events = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("journal row must be a mapping")
        try:
            update = _integer(row["update"], "update")
            normal_update = _integer(row["normal_update"], "normal_update")
            kind, table, loss = row["update_kind"], row["table"], _loss(row["loss"])
        except (KeyError, TypeError) as error:
            raise ValueError("replay row lacks required counters, kind, table, or loss") from error
        if update != result["global_update"] + 1:
            raise ValueError("optimizer update gap, duplicate, or out-of-order row")
        if kind != result["phase"]:
            raise ValueError("unexpected normal/extra phase ordering")
        if normal_update != result["normal_update"] + int(kind == "normal"):
            raise ValueError("normal update must advance only on normal rows")
        if not isinstance(table, str) or table not in allowed:
            raise ValueError("journal table is outside the declared 120-table set")
        if kind == "normal":
            if table in normal_seen:
                raise ValueError("table repeated within a normal120 cycle")
            normal_seen.add(table)
            result["partial"].append({"table": table, "loss": loss})
            result["new_updates"] += 1
            if len(result["partial"]) == CYCLE_SIZE:
                if normal_seen != allowed:
                    raise ValueError("normal120 does not contain every table")
                losses = [entry["loss"] for entry in result["partial"]]
                try:
                    mean_loss = math.fsum(losses) / CYCLE_SIZE
                except OverflowError as error:
                    raise ValueError("normal cycle loss sum is not finite") from error
                if not math.isfinite(mean_loss):
                    raise ValueError("normal cycle mean loss is not finite")
                ordered = sorted(losses)
                ranking = sorted(result["partial"], key=lambda entry: (-entry["loss"], entry["table"]))
                result["top6_tables"] = [entry["table"] for entry in ranking[:6]]
                result["top24_tables"] = [entry["table"] for entry in ranking[:24]]
                events.append({"cycle_index": result["new_updates"] // CYCLE_SIZE,
                    "end_new_update": result["new_updates"], "end_normal_update": normal_update,
                    "end_global_update": update, "mean_loss": mean_loss,
                    "median_loss": _linear_quantile(ordered, .5),
                    "p05_loss": _linear_quantile(ordered, .05), "p95_loss": _linear_quantile(ordered, .95),
                    "table_count": CYCLE_SIZE, "normal_count": CYCLE_SIZE})
                result["partial"] = []
                normal_seen = set()
                result["phase"] = "extra_top6"
        else:
            selection = result["top6_tables"] if kind == "extra_top6" else result["top24_tables"]
            if table not in selection or table in result["phase_tables"]:
                raise ValueError("extra table is not selected or repeats within its extra block")
            result["phase_tables"].append(table)
            result["extra_updates"][kind] += 1
            if len(result["phase_tables"]) == len(selection):
                result["phase_tables"] = []
                if kind == "extra_top6":
                    result["phase"] = "extra_top24"
                else:
                    result["phase"] = "normal"
                    result["top6_tables"] = []
                    result["top24_tables"] = []
        result["global_update"] = update
        result["normal_update"] = normal_update
    return result, events


def _initial_weighted_replay_state(table_names, base_update, base_normal_update):
    total = _integer(base_update, "base_update")
    normal = _integer(base_normal_update, "base_normal_update")
    if normal % CYCLE_SIZE or normal > total:
        raise ValueError("weighted replay needs a complete normal boundary and drained parent extras")
    return {"schema": WEIGHTED_REPLAY_SCHEMA, "policy_kind": WEIGHTED_REPLAY_KIND,
            "base_update": total, "base_normal_update": normal, "base_extra_updates": total - normal,
            "table_names": _tables(table_names), "global_update": total, "normal_update": normal,
            "new_updates": 0, "partial": [], "phase": "normal", "phase_pass": 0,
            "phase_tables": [], "top24_tables": [],
            "extra_updates": {kind: 0 for kind, _, _ in REPLAY_GROUPS[WEIGHTED_REPLAY_KIND]},
            "cumulative_extra_updates": total - normal}


def _weighted_validated_copy(state):
    if not isinstance(state, Mapping) or state.get("schema") != WEIGHTED_REPLAY_SCHEMA:
        raise ValueError("unsupported weighted replay state schema")
    if state.get("policy_kind") != WEIGHTED_REPLAY_KIND:
        raise ValueError("weighted replay state policy mismatch")
    try:
        result = _initial_weighted_replay_state(state["table_names"], state["base_update"], state["base_normal_update"])
        partial = _validated_copy({"schema": SCHEMA, "base_update": state["base_normal_update"],
                                   "table_names": state["table_names"], "new_updates": state["new_updates"],
                                   "partial": state["partial"]})
        result.update(new_updates=partial["new_updates"], partial=partial["partial"],
                      global_update=_integer(state["global_update"], "global_update"),
                      normal_update=_integer(state["normal_update"], "normal_update"),
                      phase_pass=_integer(state["phase_pass"], "phase_pass"),
                      cumulative_extra_updates=_integer(state["cumulative_extra_updates"], "cumulative_extra_updates"))
        if _integer(state["base_extra_updates"], "base_extra_updates") != result["base_extra_updates"]:
            raise ValueError("inherited extra counter disagrees with initial actual minus normal")
        extras = state["extra_updates"]
        if not isinstance(extras, Mapping) or set(extras) != set(result["extra_updates"]):
            raise ValueError("weighted replay requires exactly top2/top6/top24 extra counters")
        result["extra_updates"] = {kind: _integer(extras[kind], kind) for kind in extras}
        allowed = set(result["table_names"])
        for key in ("phase_tables", "top24_tables"):
            names = state[key]
            if (not isinstance(names, list) or len(names) != len(set(names))
                    or any(not isinstance(name, str) or name not in allowed for name in names)):
                raise ValueError("invalid weighted replay table selection")
            result[key] = list(names)
        result["phase"] = state["phase"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete weighted replay state") from error
    extra_new = sum(result["extra_updates"].values())
    if (result["global_update"] != result["base_update"] + result["new_updates"] + extra_new
            or result["normal_update"] != result["base_normal_update"] + result["new_updates"]
            or result["cumulative_extra_updates"] != result["base_extra_updates"] + extra_new):
        raise ValueError("weighted replay total/normal/inherited extra counters disagree")
    groups = REPLAY_GROUPS[WEIGHTED_REPLAY_KIND]
    phase = result["phase"]
    cycles = result["new_updates"] // CYCLE_SIZE
    if phase == "normal":
        if result["phase_pass"] or result["phase_tables"] or result["top24_tables"]:
            raise ValueError("normal phase retains pending weighted extras")
        expected = {kind: cycles * count * passes for kind, count, passes in groups}
    else:
        kinds = [kind for kind, _, _ in groups]
        if (phase not in kinds or not cycles or result["partial"]
                or result["new_updates"] % CYCLE_SIZE or len(result["top24_tables"]) != 24):
            raise ValueError("extra phase requires one completed normal120 and frozen top24")
        group_index = kinds.index(phase)
        _, count, passes = groups[group_index]
        selected = len(result["phase_tables"])
        if (result["phase_pass"] >= passes or selected >= count
                or not set(result["phase_tables"]) <= set(result["top24_tables"][:count])):
            raise ValueError("invalid pass cursor or table in weighted extra group")
        expected = {}
        for i, (kind, size, repeats) in enumerate(groups):
            progress = size * repeats if i < group_index else result["phase_pass"] * size + selected if i == group_index else 0
            expected[kind] = (cycles - 1) * size * repeats + progress
    if result["extra_updates"] != expected:
        raise ValueError("weighted extra counters disagree with phase/pass cursor")
    return result


def _advance_weighted_replay(state, rows):
    result = _weighted_validated_copy(state)
    allowed = set(result["table_names"])
    normal_seen = {entry["table"] for entry in result["partial"]}
    events = []
    groups = REPLAY_GROUPS[WEIGHTED_REPLAY_KIND]
    kinds = [kind for kind, _, _ in groups]
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("weighted replay row must be a mapping")
        try:
            update = _integer(row["update"], "update")
            normal = _integer(row["normal_update"], "normal_update")
            extra = _integer(row["extra_updates"], "extra_updates")
            kind, table, loss = row["update_kind"], row["table"], _loss(row["loss"])
        except (KeyError, TypeError) as error:
            raise ValueError("weighted replay row lacks counters, kind, table, or loss") from error
        if (update != result["global_update"] + 1 or kind != result["phase"]
                or normal != result["normal_update"] + int(kind == "normal")
                or extra != update - normal):
            raise ValueError("weighted replay counter discontinuity or unexpected phase")
        if not isinstance(table, str) or table not in allowed:
            raise ValueError("weighted replay table is outside declared scope")
        if kind == "normal":
            if table in normal_seen:
                raise ValueError("duplicate table in normal120")
            result["partial"].append({"table": table, "loss": loss})
            result["new_updates"] += 1
            normal_seen.add(table)
            if len(result["partial"]) == CYCLE_SIZE:
                losses = sorted(entry["loss"] for entry in result["partial"])
                try:
                    mean = math.fsum(losses) / CYCLE_SIZE
                except OverflowError as error:
                    raise ValueError("weighted normal cycle loss sum is not finite") from error
                if not math.isfinite(mean):
                    raise ValueError("weighted normal cycle mean is not finite")
                result["top24_tables"] = [entry["table"] for entry in sorted(
                    result["partial"], key=lambda entry: (-entry["loss"], entry["table"]))[:24]]
                events.append({"cycle_index": result["new_updates"] // CYCLE_SIZE,
                    "end_new_update": result["new_updates"], "end_normal_update": normal,
                    "end_global_update": update, "mean_loss": mean,
                    "median_loss": _linear_quantile(losses, .5), "p05_loss": _linear_quantile(losses, .05),
                    "p95_loss": _linear_quantile(losses, .95), "p99_loss": _linear_quantile(losses, .99),
                    "table_count": CYCLE_SIZE, "normal_count": CYCLE_SIZE,
                    "new_extra_updates": extra - result["base_extra_updates"], "cumulative_extra_updates": extra})
                result["partial"] = []
                normal_seen = set()
                result["phase"] = kinds[0]
        else:
            index = kinds.index(kind)
            _, count, passes = groups[index]
            if table not in result["top24_tables"][:count] or table in result["phase_tables"]:
                raise ValueError("extra table is not selected or repeats within a complete pass")
            result["phase_tables"].append(table)
            result["extra_updates"][kind] += 1
            if len(result["phase_tables"]) == count:
                result["phase_tables"] = []
                result["phase_pass"] += 1
                if result["phase_pass"] == passes:
                    result["phase_pass"] = 0
                    result["phase"] = kinds[index + 1] if index + 1 < len(kinds) else "normal"
                    if result["phase"] == "normal":
                        result["top24_tables"] = []
        result["global_update"], result["normal_update"] = update, normal
        result["cumulative_extra_updates"] = extra
    return result, events


def advance_replay(state: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> tuple[dict, list[dict]]:
    """Dispatch immutable replay aggregation while retaining legacy state support.

    The 42-extra policy uses top2 for three complete passes, top6 for two, then
    top24 once. ``extra_updates`` in its state is NEW per-group counts;
    ``cumulative_extra_updates`` includes inherited extras. Journal counters are
    all cumulative. Normal distributions alone add linear P99 as a diagnostic.
    """
    if isinstance(state, Mapping) and state.get("schema") == WEIGHTED_REPLAY_SCHEMA:
        return _advance_weighted_replay(state, rows)
    return _advance_replay_legacy(state, rows)
