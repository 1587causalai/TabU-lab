"""Focused CPU checks; run with unittest discovery in this directory."""
import copy
import json
import math
import unittest

from cycle_loss import (advance, initial_state, advance_replay, initial_replay_state,
                        WEIGHTED_REPLAY_KIND, REPLAY_GROUPS)


TABLES = [f"table_{index:03d}" for index in range(120)]


def rows(count=120, base=0):
    return [{"update": base + i + 1, "table": TABLES[i % 120], "loss": float(i + 1)}
            for i in range(count)]


class CycleLossTests(unittest.TestCase):
    def test_all_raw_points_and_cycle_boundaries(self):
        original = initial_state(TABLES)
        journal = rows(240)
        next_state, events = advance(original, journal)
        self.assertEqual(original["new_updates"], 0)
        self.assertEqual(original["partial"], [])
        self.assertEqual(next_state["new_updates"], 240)
        self.assertEqual(next_state["partial"], [])
        self.assertEqual(events, [
            {"cycle_index": 1, "end_new_update": 120, "end_global_update": 120,
             "mean_loss": 60.5, "median_loss": 60.5, "p05_loss": 6.95,
             "p95_loss": 114.05, "table_count": 120},
            {"cycle_index": 2, "end_new_update": 240, "end_global_update": 240,
             "mean_loss": 180.5, "median_loss": 180.5, "p05_loss": 126.95,
             "p95_loss": 234.05, "table_count": 120},
        ])
        # Sampling every 100th step would give 100 and 200, not either mean.
        self.assertNotEqual(events[0]["mean_loss"], journal[99]["loss"])

    def test_resume_and_partial_cycle_survive_json_roundtrip(self):
        base = 122880
        journal = rows(250, base)
        state, events = advance(initial_state(TABLES, base_update=base), journal[:47])
        self.assertEqual(events, [])
        self.assertEqual(len(state["partial"]), 47)
        state = json.loads(json.dumps(state, allow_nan=False))
        state, events = advance(state, journal[47:241])
        self.assertEqual([event["end_global_update"] for event in events], [123000, 123120])
        self.assertEqual([event["end_new_update"] for event in events], [120, 240])
        self.assertEqual([event["median_loss"] for event in events], [60.5, 180.5])
        self.assertAlmostEqual(events[0]["p05_loss"], 6.95)
        self.assertAlmostEqual(events[1]["p95_loss"], 234.05)
        self.assertEqual(len(state["partial"]), 1)
        state, events = advance(state, journal[241:])
        self.assertEqual(events, [])
        self.assertEqual(len(state["partial"]), 10)

    def test_fsum_and_arbitrary_table_order(self):
        journal = rows()
        for i, row in enumerate(journal):
            row["table"] = TABLES[-i - 1]
            row["loss"] = [1e16, 1., -1e16][i] if i < 3 else 0.
        _, events = advance(initial_state(TABLES), journal)
        self.assertEqual(events[0]["mean_loss"], math.fsum(row["loss"] for row in journal) / 120)
        self.assertEqual(events[0]["mean_loss"], 1. / 120)
        self.assertEqual(events[0]["median_loss"], 0.)
        self.assertEqual(events[0]["p05_loss"], 0.)
        self.assertEqual(events[0]["p95_loss"], 0.)

    def test_zero_to_119_linear_quantiles(self):
        journal = rows()
        for i, row in enumerate(journal):
            row["loss"] = float(i)
        _, events = advance(initial_state(TABLES), journal)
        self.assertEqual(events[0]["mean_loss"], 59.5)
        self.assertEqual(events[0]["median_loss"], 59.5)
        self.assertAlmostEqual(events[0]["p05_loss"], 5.95)
        self.assertAlmostEqual(events[0]["p95_loss"], 113.05)

    def test_extreme_outlier_changes_mean_without_forcing_quantiles(self):
        journal = rows()
        for row in journal:
            row["loss"] = 0.
        journal[-1]["loss"] = 1e300
        _, events = advance(initial_state(TABLES), journal)
        self.assertEqual(events[0]["mean_loss"], 1e300 / 120)
        self.assertEqual(events[0]["median_loss"], 0.)
        self.assertEqual(events[0]["p05_loss"], 0.)
        self.assertEqual(events[0]["p95_loss"], 0.)

    def test_invalid_batch_does_not_mutate_input(self):
        state, _ = advance(initial_state(TABLES), rows(70))
        saved = copy.deepcopy(state)
        for bad in ({"update": 72, "table": TABLES[70], "loss": 1.},
                    {"update": 70, "table": TABLES[70], "loss": 1.},
                    {"update": 71, "table": TABLES[0], "loss": 1.},
                    {"update": 71, "table": "missing_table", "loss": 1.},
                    {"update": 71., "table": TABLES[70], "loss": 1.}):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    advance(state, [bad])
                self.assertEqual(state, saved)

    def test_missing_or_duplicate_table_cannot_complete_cycle(self):
        journal = rows()
        state, events = advance(initial_state(TABLES), journal[:119])
        self.assertEqual(events, [])
        self.assertEqual(len(state["partial"]), 119)
        journal[-1]["table"] = TABLES[0]
        with self.assertRaises(ValueError):
            advance(state, [journal[-1]])

    def test_nonfinite_and_invalid_loss_rejected(self):
        for loss in (float("nan"), float("inf"), -float("inf"), True, "1.2", None):
            with self.subTest(loss=loss):
                journal = rows()
                journal[110]["loss"] = loss
                state = initial_state(TABLES)
                with self.assertRaises(ValueError):
                    advance(state, journal)
                self.assertEqual(state["new_updates"], 0)

    def test_late_error_rejects_even_earlier_complete_cycles(self):
        journal = rows(130)
        journal[-1]["update"] += 1
        state = initial_state(TABLES)
        with self.assertRaises(ValueError):
            advance(state, journal)
        self.assertEqual(state, initial_state(TABLES))

    def test_state_validation_and_empty_poll(self):
        with self.assertRaises(ValueError):
            initial_state(TABLES[:-1])
        with self.assertRaises(ValueError):
            initial_state(TABLES[:-1] + [TABLES[0]])
        with self.assertRaises(ValueError):
            initial_state(TABLES, base_update=True)
        state, _ = advance(initial_state(TABLES), rows(3))
        unchanged, events = advance(state, [])
        self.assertEqual(unchanged, state)
        self.assertIsNot(unchanged, state)
        self.assertEqual(events, [])
        bad = copy.deepcopy(state)
        bad["new_updates"] += 1
        with self.assertRaises(ValueError):
            advance(bad, [])


def replay_rows(cycles=1, *, base=0, normal_base=0, tied=False):
    result = []
    total, normal = base, normal_base
    losses = {table: 1. if tied else float(i) for i, table in enumerate(TABLES)}
    ranking = sorted(TABLES, key=lambda table: (-losses[table], table))
    for _ in range(cycles):
        for table in reversed(TABLES):
            total += 1
            normal += 1
            result.append({"update": total, "normal_update": normal,
                           "update_kind": "normal", "table": table, "loss": losses[table]})
        for kind, selection in (("extra_top6", ranking[:6]), ("extra_top24", ranking[:24])):
            for table in reversed(selection):
                total += 1
                result.append({"update": total, "normal_update": normal,
                               "update_kind": kind, "table": table, "loss": 1e9})
    return result


class ReplayCycleLossTests(unittest.TestCase):
    def test_normal_only_distribution_and_nested_extra_counts(self):
        original = initial_replay_state(TABLES)
        state, events = advance_replay(original, replay_rows(2))
        self.assertEqual(original, initial_replay_state(TABLES))
        self.assertEqual(state["global_update"], 300)
        self.assertEqual(state["normal_update"], 240)
        self.assertEqual(state["new_updates"], 240)
        self.assertEqual(state["extra_updates"], {"extra_top6": 12, "extra_top24": 48})
        self.assertEqual(state["phase"], "normal")
        self.assertEqual(state["partial"], [])
        self.assertEqual([e["cycle_index"] for e in events], [1, 2])
        self.assertEqual([e["end_new_update"] for e in events], [120, 240])
        self.assertEqual([e["end_normal_update"] for e in events], [120, 240])
        self.assertEqual([e["end_global_update"] for e in events], [120, 270])
        for event in events:
            self.assertEqual(event["mean_loss"], 59.5)
            self.assertEqual(event["median_loss"], 59.5)
            self.assertAlmostEqual(event["p05_loss"], 5.95)
            self.assertAlmostEqual(event["p95_loss"], 113.05)
            self.assertEqual(event["table_count"], 120)
        # Two extra phases select the top6 twice without entering the mean.
        first = replay_rows()
        for table in TABLES[-6:]:
            self.assertEqual(sum(r["table"] == table for r in first), 3)

    def test_nano_offset_partial_normal_and_extra_resume(self):
        base = 122880
        journal = replay_rows(2, base=base, normal_base=base)
        state = initial_replay_state(TABLES, base, base)
        all_events = []
        previous = 0
        for boundary in (57, 123, 139, 271, 300):
            state, events = advance_replay(state, journal[previous:boundary])
            all_events.extend(events)
            state = json.loads(json.dumps(state, allow_nan=False))
            if boundary == 123:
                self.assertEqual(state["phase"], "extra_top6")
                self.assertEqual(len(state["phase_tables"]), 3)
            if boundary == 139:
                self.assertEqual(state["phase"], "extra_top24")
                self.assertEqual(len(state["phase_tables"]), 13)
            previous = boundary
        self.assertEqual([e["end_global_update"] for e in all_events], [123000, 123150])
        self.assertEqual([e["end_normal_update"] for e in all_events], [123000, 123120])
        self.assertEqual([e["end_new_update"] for e in all_events], [120, 240])
        full_state, full_events = advance_replay(initial_replay_state(TABLES, base, base), journal)
        self.assertEqual(state, full_state)
        self.assertEqual(all_events, full_events)

    def test_tied_losses_use_table_id_order_and_frozen_selection(self):
        journal = replay_rows(tied=True)
        state, events = advance_replay(initial_replay_state(TABLES), journal[:120])
        self.assertEqual(state["top6_tables"], TABLES[:6])
        self.assertEqual(state["top24_tables"], TABLES[:24])
        self.assertEqual(events[0]["mean_loss"], 1.)
        state, events = advance_replay(state, journal[120:])
        self.assertEqual(events, [])
        self.assertEqual(state["phase"], "normal")

    def test_phase_selection_and_counter_violations_reject_batch(self):
        journal = replay_rows()
        alterations = [
            (0, {"update_kind": "extra_top6"}),
            (120, {"update_kind": "extra_top24"}),
            (120, {"normal_update": 121}),
            (120, {"table": TABLES[0]}),
            (121, {"table": journal[120]["table"]}),
            (126, {"update_kind": "normal", "normal_update": 121}),
            (149, {"table": TABLES[0]}),
            (140, {"update": 140}),
            (140, {"loss": float("inf")}),
            (10, {"normal_update": 10}),
            (10, {"table": journal[9]["table"]}),
        ]
        for index, change in alterations:
            with self.subTest(index=index, change=change):
                mutated = copy.deepcopy(journal)
                mutated[index].update(change)
                initial = initial_replay_state(TABLES)
                with self.assertRaises(ValueError):
                    advance_replay(initial, mutated)
                self.assertEqual(initial, initial_replay_state(TABLES))

    def test_boundaries_state_integrity_and_empty_poll(self):
        with self.assertRaises(ValueError):
            initial_replay_state(TABLES, 122881, 122881)
        with self.assertRaises(ValueError):
            initial_replay_state(TABLES, 0, 120)
        state, _ = advance_replay(initial_replay_state(TABLES), replay_rows()[:121])
        same, events = advance_replay(state, [])
        self.assertEqual(same, state)
        self.assertIsNot(same, state)
        self.assertEqual(events, [])
        for change in ({"normal_update": 121}, {"global_update": 120},
                       {"phase": "normal"}, {"top6_tables": TABLES[:6]},
                       {"extra_updates": {"extra_top6": 2, "extra_top24": 0}}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    advance_replay(state | change, [])


def weighted_replay_rows(cycles=2, *, base=120600, normal_base=120000, tied=False):
    result = []
    total, normal = base, normal_base
    losses = {table: 1. if tied else float(i) for i, table in enumerate(TABLES)}
    ranking = sorted(TABLES, key=lambda table: (-losses[table], table))
    for _ in range(cycles):
        for table in TABLES:
            total += 1
            normal += 1
            result.append({"update": total, "normal_update": normal, "extra_updates": total - normal,
                           "update_kind": "normal", "table": table, "loss": losses[table]})
        for kind, count, passes in REPLAY_GROUPS[WEIGHTED_REPLAY_KIND]:
            for _pass in range(passes):
                for table in reversed(ranking[:count]):
                    total += 1
                    result.append({"update": total, "normal_update": normal, "extra_updates": total - normal,
                                   "update_kind": kind, "table": table, "loss": 1e9})
    return result


class WeightedReplayCycleLossTests(unittest.TestCase):
    def initial(self):
        return initial_replay_state(TABLES, 120600, 120000, policy_kind=WEIGHTED_REPLAY_KIND)

    def test_42_extras_inherited_counter_and_normal_only_p99(self):
        rows = weighted_replay_rows()
        state, events = advance_replay(self.initial(), rows)
        self.assertEqual(state["global_update"], 120924)
        self.assertEqual(state["normal_update"], 120240)
        self.assertEqual(state["base_extra_updates"], 600)
        self.assertEqual(state["cumulative_extra_updates"], 684)
        self.assertEqual(state["extra_updates"], {"extra_top2": 12, "extra_top6": 24, "extra_top24": 48})
        self.assertEqual([e["end_global_update"] for e in events], [120720, 120882])
        self.assertEqual([e["end_new_update"] for e in events], [120, 240])
        self.assertEqual([e["new_extra_updates"] for e in events], [0, 42])
        self.assertEqual([e["cumulative_extra_updates"] for e in events], [600, 642])
        for event in events:
            self.assertEqual(event["mean_loss"], 59.5)
            self.assertEqual(event["median_loss"], 59.5)
            self.assertAlmostEqual(event["p05_loss"], 5.95)
            self.assertAlmostEqual(event["p95_loss"], 113.05)
            self.assertAlmostEqual(event["p99_loss"], 117.81)
        first_extra = [r for r in rows[:162] if r["update_kind"] != "normal"]
        for i, table in enumerate(TABLES):
            expected = 6 if i >= 118 else 3 if i >= 114 else 1 if i >= 96 else 0
            self.assertEqual(sum(r["table"] == table for r in first_extra), expected)

    def test_partial_pass_resume_and_json_roundtrip(self):
        rows = weighted_replay_rows()
        state, events, previous = self.initial(), [], 0
        for boundary in (57, 121, 123, 125, 129, 133, 138, 161, 162, 283, 324):
            state, emitted = advance_replay(state, rows[previous:boundary])
            events.extend(emitted)
            state = json.loads(json.dumps(state, allow_nan=False))
            if boundary == 123:
                self.assertEqual(state["phase"], "extra_top2")
                self.assertEqual(state["phase_pass"], 1)
                self.assertEqual(len(state["phase_tables"]), 1)
            if boundary == 133:
                self.assertEqual(state["phase"], "extra_top6")
                self.assertEqual(state["phase_pass"], 1)
            previous = boundary
        complete, expected_events = advance_replay(self.initial(), rows)
        self.assertEqual(state, complete)
        self.assertEqual(events, expected_events)

    def test_tied_rank_and_no_reranking_from_extra_losses(self):
        rows = weighted_replay_rows(cycles=1, tied=True)
        state, events = advance_replay(self.initial(), rows[:120])
        self.assertEqual(state["top24_tables"], TABLES[:24])
        self.assertEqual(events[0]["p99_loss"], 1.)
        state, _ = advance_replay(state, rows[120:])
        self.assertEqual(state["phase"], "normal")
        self.assertEqual(state["cumulative_extra_updates"], 642)

    def test_bad_pass_counter_and_inherited_extra_reject_whole_batch(self):
        rows = weighted_replay_rows(cycles=1)
        changes = [(121, {"table": rows[120]["table"]}),
                   (122, {"update_kind": "extra_top6"}),
                   (126, {"update_kind": "extra_top24"}),
                   (132, {"table": TABLES[0]}),
                   (120, {"extra_updates": 1}),
                   (120, {"normal_update": 120121}),
                   (161, {"loss": float("nan")}),
                   (161, {"update": rows[161]["update"] + 1})]
        for index, change in changes:
            with self.subTest(index=index, change=change):
                bad = copy.deepcopy(rows)
                bad[index].update(change)
                state = self.initial()
                with self.assertRaises(ValueError):
                    advance_replay(state, bad)
                self.assertEqual(state, self.initial())
        state, _ = advance_replay(self.initial(), rows[:123])
        for change in ({"phase_pass": 3}, {"base_extra_updates": 0},
                       {"cumulative_extra_updates": 3}, {"top24_tables": TABLES[:24]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                advance_replay(state | change, [])
        with self.assertRaises(ValueError):
            initial_replay_state(TABLES, 120600, 120000, policy_kind="unknown")


if __name__ == "__main__":
    unittest.main()
