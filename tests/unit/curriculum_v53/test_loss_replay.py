"""Loss-ranked replay accounting, immutable transitions and recovery boundaries."""

import copy
import json
from collections import Counter

import pytest

from tabu_lab.curriculum_v53.loss_replay import (
    V1,
    V2,
    extras_per_cycle,
    new_state,
    next_extra,
    record_extra,
    record_normal,
    validate_state,
)

TABLES = tuple(f"table_{index:03d}" for index in range(120))


def _normal_cycle(state, *, tied=False):
    for index, table in reversed(list(enumerate(TABLES))):
        state = record_normal(state, table, 1.0 if tied else float(index))
    return state


def _finish_extras(state):
    addresses = []
    while (address := next_extra(state)) is not None:
        addresses.append(address)
        state = record_extra(state)
    return state, addresses


def test_120_plus_30_overlap_counts_and_boundary_migration():
    state = new_state(TABLES, start_normal_cursor=122880)
    assert state["cycle_index"] == 1024
    assert next_extra(state) is None
    state = _normal_cycle(state)
    assert state["normal_cursor"] == 123000
    assert state["cycle_index"] == 1025
    assert len(state["partial"]) == 120
    assert [entry["table"] for entry in state["queue"][:6]] == list(reversed(TABLES[-6:]))
    assert [entry["table"] for entry in state["queue"][6:]] == list(reversed(TABLES[-24:]))
    state, addresses = _finish_extras(state)
    counts = Counter(address["table"] for address in addresses)
    assert counts == Counter({table: 2 if table in TABLES[-6:] else 1
                              for table in TABLES[-24:]})
    assert Counter(1 + state["extra_by_table"][table] for table in TABLES) == {
        1: 96, 2: 18, 3: 6,
    }
    assert state["extra_updates"] == 30
    assert state["normal_cursor"] == 123000
    assert state["partial"] == {} and state["queue"] == [] and state["queue_cursor"] == 0
    validate_state(state, TABLES)


def test_equal_losses_use_table_id_and_repeated_tables_get_distinct_extra_indices():
    state = _normal_cycle(new_state(reversed(TABLES)), tied=True)
    state, first = _finish_extras(state)
    assert [item["table"] for item in first] == [*TABLES[:6], *TABLES[:24]]
    assert [item["update_kind"] for item in first] == ["extra_top6"] * 6 + ["extra_top24"] * 24
    assert [item["table_episode_index"] for item in first] == [0] * 6 + [1] * 6 + [0] * 18
    state = _normal_cycle(state, tied=True)
    state, second = _finish_extras(state)
    assert [item["table_episode_index"] for item in second] == [2] * 6 + [3] * 6 + [1] * 18
    assert state["extra_updates"] == 60
    validate_state(state, TABLES)


def test_json_recovery_at_every_normal_and_extra_boundary_matches_uninterrupted_state():
    original = new_state(TABLES, 120)
    restored = json.loads(json.dumps(original, allow_nan=False))
    extra_addresses = []
    for cycle in range(3):
        order = TABLES if cycle % 2 else tuple(reversed(TABLES))
        for index, table in enumerate(order):
            restored = json.loads(json.dumps(restored, allow_nan=False))
            validate_state(restored, TABLES)
            assert next_extra(restored) is None
            value = float((index * 13 + cycle) % 19)
            original = record_normal(original, table, value)
            restored = record_normal(restored, table, value)
            assert restored == original
        for position in range(30):
            restored = json.loads(json.dumps(restored, allow_nan=False))
            validate_state(restored, TABLES)
            assert next_extra(restored) == next_extra(original)
            extra_addresses.append(next_extra(restored))
            original = record_extra(original)
            restored = record_extra(restored)
            assert restored == original
            assert restored["queue_cursor"] == (position + 1) % 30
    assert restored["normal_cursor"] == 480 and restored["extra_updates"] == 90
    for table in TABLES:
        indices = [entry["table_episode_index"] for entry in extra_addresses
                   if entry["table"] == table]
        assert indices == list(range(len(indices)))


def test_transitions_and_next_address_do_not_mutate_or_alias_the_input():
    initial = new_state(TABLES)
    snapshot = copy.deepcopy(initial)
    changed = record_normal(initial, TABLES[0], 2.0)
    assert initial == snapshot
    changed["extra_by_table"][TABLES[0]] = 10
    assert initial == snapshot
    ready = _normal_cycle(initial)
    before = copy.deepcopy(ready)
    address = next_extra(ready)
    address["table"] = "changed locally"
    consumed = record_extra(ready)
    assert ready == before
    consumed["queue"][0]["table"] = "changed locally"
    consumed["partial"][TABLES[0]] = 100.0
    assert ready == before


@pytest.mark.parametrize("start", [-1, 1, 119, 121, 1.0, True])
def test_new_state_rejects_non_boundary_or_invalid_start(start):
    with pytest.raises(ValueError):
        new_state(TABLES, start)


@pytest.mark.parametrize("tables", [TABLES[:-1], (*TABLES, "extra"),
                                    (*TABLES[:-1], TABLES[0]), (*TABLES[:-1], ""),
                                    (*TABLES[:-1], None), None])
def test_exact_table_registry_is_required(tables):
    with pytest.raises(ValueError):
        new_state(tables)


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), -float("inf"),
                                  True, "1.0", None, 10**400])
def test_invalid_normal_loss_is_rejected_without_mutation(loss):
    state = new_state(TABLES)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="finite number"):
        record_normal(state, TABLES[0], loss)
    assert state == before


def test_illegal_transition_duplicate_and_unknown_table_are_rejected():
    state = record_normal(new_state(TABLES), TABLES[0], 0.0)
    with pytest.raises(ValueError, match="duplicate"):
        record_normal(state, TABLES[0], 1.0)
    with pytest.raises(ValueError, match="unknown table"):
        record_normal(state, "unknown", 1.0)
    with pytest.raises(ValueError, match="no pending"):
        record_extra(state)
    ready = _normal_cycle(new_state(TABLES))
    with pytest.raises(ValueError, match="finish the pending"):
        record_normal(ready, TABLES[0], 1.0)


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(unknown=1),
    lambda s: s.pop("normal_cursor"),
    lambda s: s.update(normal_cursor=-1),
    lambda s: s.update(normal_cursor=True),
    lambda s: s.update(cycle_index=99),
    lambda s: s.update(partial=[]),
    lambda s: s["partial"].update(unknown=0.0),
    lambda s: s["partial"].update({TABLES[0]: float("nan")}),
    lambda s: s["partial"].pop(TABLES[0]),
    lambda s: s.update(queue=[]),
    lambda s: s.update(queue=tuple(s["queue"])),
    lambda s: s["queue"].pop(),
    lambda s: s["queue"].reverse(),
    lambda s: s["queue"][0].update(update_kind="normal"),
    lambda s: s["queue"][0].update(table_episode_index=0),
    lambda s: s.update(queue_cursor=30),
    lambda s: s.update(queue_cursor=1),
    lambda s: s.update(extra_updates=1),
    lambda s: s["extra_by_table"].pop(TABLES[0]),
    lambda s: s["extra_by_table"].update({TABLES[0]: -1}),
    lambda s: s["extra_by_table"].update({TABLES[0]: True}),
])
def test_malformed_pending_recovery_state_is_rejected(mutate):
    state = _normal_cycle(new_state(TABLES))
    mutate(state)
    with pytest.raises(ValueError):
        validate_state(state, TABLES)


def test_completed_and_partial_recovery_count_invariants():
    state = new_state(TABLES)
    state["partial"][TABLES[0]] = 1.0
    with pytest.raises(ValueError, match="partial loss count"):
        validate_state(state, TABLES)
    state, _ = _finish_extras(_normal_cycle(new_state(TABLES)))
    # Thirty selections of thirty distinct tables cannot come from top6+top24.
    state["extra_by_table"] = {table: int(table in TABLES[:30]) for table in TABLES}
    with pytest.raises(ValueError, match="six double selections"):
        validate_state(state, TABLES)
    with pytest.raises(ValueError, match="exact 120 tables"):
        validate_state(new_state(TABLES), (*TABLES[:-1], "replacement"))


def test_consumed_queue_prefix_must_match_table_extra_counts():
    state = record_extra(_normal_cycle(new_state(TABLES)))
    selected = TABLES[-1]
    state["extra_by_table"][selected] = 0
    state["extra_by_table"][TABLES[0]] = 1
    with pytest.raises(ValueError, match="consumed queue"):
        validate_state(state, TABLES)


def test_v2_120_plus_42_additive_passes_inherit_counts_without_aliasing():
    parent, _ = _finish_extras(_normal_cycle(new_state(TABLES)))
    baseline = copy.deepcopy(parent["extra_by_table"])
    state = new_state(TABLES, 120, kind=V2, base_extra_by_table=baseline)
    baseline[TABLES[0]] = 999
    assert state["extra_updates"] == 30
    assert state["base_extra_by_table"] == parent["extra_by_table"]
    state = _normal_cycle(state)
    assert len(state["queue"]) == extras_per_cycle(V2) == 42
    state, addresses = _finish_extras(state)
    ranked = list(reversed(TABLES))
    assert [entry["table"] for entry in addresses] == (
        ranked[:2] * 3 + ranked[:6] * 2 + ranked[:24])
    assert [entry["update_kind"] for entry in addresses] == (
        ["extra_top2"] * 6 + ["extra_top6"] * 12 + ["extra_top24"] * 24)
    additions = Counter(entry["table"] for entry in addresses)
    assert additions == Counter({table: 6 if table in ranked[:2]
                                 else 3 if table in ranked[:6] else 1
                                 for table in ranked[:24]})
    assert Counter(additions.values()) == {6: 2, 3: 4, 1: 18}
    for table in TABLES:
        old = parent["extra_by_table"][table]
        assert [entry["table_episode_index"] for entry in addresses if entry["table"] == table] == (
            list(range(old, old + additions[table])))
    assert (state["normal_cursor"], state["extra_updates"]) == (240, 72)
    assert not state["queue"] and not state["partial"] and state["queue_cursor"] == 0
    assert parent["extra_updates"] == 30


def test_v2_json_recovery_every_boundary_ties_and_changing_rankings():
    state = new_state(TABLES, 120, kind=V2,
                      base_extra_by_table={name: i % 3 for i, name in enumerate(TABLES)})
    baseline = copy.deepcopy(state["base_extra_by_table"])
    for cycle in range(5):
        for index, table in enumerate(TABLES):
            before = copy.deepcopy(state)
            restored = json.loads(json.dumps(state, allow_nan=False))
            value = 1.0 if cycle == 0 else float((index * 13 + cycle * 17) % 121)
            expected = record_normal(state, table, value)
            assert state == before
            assert expected == record_normal(restored, table, value)
            state = expected
        if cycle == 0:
            assert [x["table"] for x in state["queue"]] == (
                list(TABLES[:2]) * 3 + list(TABLES[:6]) * 2 + list(TABLES[:24]))
        for position in range(42):
            before = copy.deepcopy(state)
            restored = json.loads(json.dumps(state, allow_nan=False))
            assert next_extra(state) == next_extra(restored)
            expected = record_extra(state)
            assert state == before
            assert expected == record_extra(restored)
            state = expected
            assert state["queue_cursor"] == (position + 1) % 42
        assert state["extra_updates"] == sum(baseline.values()) + (cycle + 1) * 42
        assert state["base_extra_by_table"] == baseline
        validate_state(state, TABLES)


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("kind"),
    lambda s: s.update(kind=V1),
    lambda s: s.update(kind="unsupported"),
    lambda s: s.pop("base_extra_by_table"),
    lambda s: s["base_extra_by_table"].pop(TABLES[0]),
    lambda s: s["base_extra_by_table"].update({TABLES[0]: -1}),
    lambda s: s["base_extra_by_table"].update({TABLES[0]: True}),
    lambda s: s["base_extra_by_table"].update({TABLES[0]: 1}),
    lambda s: s.update(queue_cursor=42),
    lambda s: s["queue"][0].update(update_kind="extra_top6"),
    lambda s: s["queue"].__setitem__(slice(0, 6), s["queue"][6:12]),
])
def test_v2_malformed_baseline_kind_or_pending_queue_rejected(mutate):
    state = _normal_cycle(new_state(TABLES, kind=V2))
    mutate(state)
    with pytest.raises(ValueError):
        validate_state(state, TABLES)


def test_v2_completed_counts_cannot_replace_heavy_groups_with_distinct_tables():
    state, _ = _finish_extras(_normal_cycle(new_state(TABLES, kind=V2)))
    state["extra_by_table"] = {name: int(i < 42) for i, name in enumerate(TABLES)}
    with pytest.raises(ValueError, match="multiplicities"):
        validate_state(state, TABLES)
    with pytest.raises(ValueError, match="v1 cannot inherit"):
        new_state(TABLES, kind=V1, base_extra_by_table=dict.fromkeys(TABLES, 0))
    with pytest.raises(ValueError, match="unsupported"):
        new_state(TABLES, kind="unknown")
    assert set(new_state(TABLES)) == {
        "normal_cursor", "extra_updates", "cycle_index", "partial", "queue",
        "queue_cursor", "extra_by_table"}
