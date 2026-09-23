"""CPU-only integration checks against the real SQLite outbox; no SSH/W&B I/O."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from wandb_mirror import Outbox, REMOTE, connect
from cycle_loss import WEIGHTED_REPLAY_KIND
from test_cycle_loss import weighted_replay_rows


TABLES = [f"table_{i:03d}" for i in range(120)]
REFERENCES = {table: {"r2": 0., "nmse": 1.} for table in TABLES}


def config(base=15120, replay=True):
    result = {"run_id": "synthetic-only", "identity_sha256": "identity", "source_sha256": "source",
              "remote_output": "/not-a-remote-path", "initial_update": base,
              "base_counts": {table: base // 120 for table in TABLES},
              "base_total_seconds": 100., "evaluate_every": 15360,
              "new_max_updates": 300 if replay else 240,
              "global_max_updates": base + (300 if replay else 240)}
    if replay:
        result.update(initial_normal_update=base, loss_replay={"kind": "normal120_top5_top20_v1",
                      "normal_max_updates": base + 240, "start_normal_cursor": base})
    return result


def journal(cfg, cycles=2):
    result = []
    total = normal = cfg["initial_update"]
    for _ in range(cycles):
        groups = [("normal", TABLES)]
        if "loss_replay" in cfg:
            groups += [("extra_top6", list(reversed(TABLES[-6:]))),
                       ("extra_top24", list(reversed(TABLES[-24:])))]
        for kind, names in groups:
            for table in names:
                total += 1
                normal += int(kind == "normal")
                row = {"update": total, "table": table,
                       "loss": float(TABLES.index(table)) if kind == "normal" else 1e9,
                       "gradient_norm": 1., "seconds": .25,
                       "total_seconds": cfg["base_total_seconds"] + (total - cfg["initial_update"]) * .25}
                if "loss_replay" in cfg:
                    row.update(update_kind=kind, normal_update=normal, extra_updates=total - normal)
                result.append(row)
    return result


def evaluation(update, name=None, r2=.3):
    return {"file": name or f"evaluation-{update}.json", "update": update, "complete": True,
            "tables": [{"table": table, "metrics": {"query_numeric_r2": r2,
                       "query_numeric_normalized_mse": 1. - r2}, "query_exposures": 3,
                       "unique_query_cells": 3} for table in TABLES]}


def snapshot(rows, offset=0, evaluations=(), terminal=None):
    return {"identity_sha256": "identity", "source_sha256": "source", "tables": TABLES,
            "updates": rows, "offset": offset, "evaluations": list(evaluations), "terminal": terminal}


def events(box, prefix):
    return [json.loads(row[0]) for row in box.db.execute(
        "SELECT payload FROM events WHERE event_id LIKE ? ORDER BY seq", (prefix + "%",))]


class ReplayMirrorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "outbox.sqlite3"
        self.boxes = []

    def tearDown(self):
        for box in self.boxes:
            box.db.close()
        self.temp.cleanup()

    def box(self, cfg):
        box = Outbox(self.path, cfg)
        self.boxes.append(box)
        return box

    def poll(self, box, cfg, rows, offset, evaluations=(), terminal=None):
        point = snapshot(rows, offset, evaluations, terminal)
        box.ingest(point, cfg, REFERENCES)
        box.cycle_request(cfg)
        box.ingest_cycles(point, cfg, REFERENCES)

    def test_replay_resume_partial_extra_and_normal_only_cycles(self):
        cfg = config(base=122880)
        rows = journal(cfg)
        box = self.box(cfg)
        self.poll(box, cfg, rows[:57], 570)
        self.assertEqual(box.cycle_status(cfg)["cycle_partial_steps"], 57)
        self.poll(box, cfg, rows[57:123], 1230)
        self.assertEqual(box.cycle_status(cfg)["cycle_partial_steps"], 0)
        self.assertEqual(box.cycle_status(cfg)["cycle_next_phase"], "extra_top6")
        self.assertEqual(box.cycle_status(cfg)["cycle_complete_count"], 1)
        box.db.close()
        self.boxes.remove(box)
        box = self.box(cfg)
        self.assertEqual(box.request(cfg)["offset"], 1230)
        self.assertEqual(box.cycle_request(cfg)["offset"], 1230)
        self.poll(box, cfg, rows[123:139], 1390)
        self.assertEqual(box.cycle_status(cfg)["cycle_next_phase"], "extra_top24")
        self.poll(box, cfg, rows[139:], 3000)
        self.assertEqual(box.cycle_status(cfg)["cycle_complete_count"], 2)
        self.assertEqual(box.cycle_status(cfg)["cycle_partial_steps"], 0)
        self.assertEqual(box.get("normal_update"), 123120)
        self.assertEqual(sum(box.get("extra_counts").values()), 60)
        means = events(box, "cycle_loss:")
        self.assertEqual([e["train_cycle/update"] for e in means], [120, 240])
        self.assertEqual([e["train_cycle/global_update"] for e in means], [123000, 123150])
        self.assertEqual([e["train_cycle/normal_update"] for e in means], [123000, 123120])
        self.assertEqual([e["train_cycle/new_actual_update"] for e in means], [120, 270])
        self.assertEqual([e["train_cycle/mean_loss"] for e in means], [59.5, 59.5])
        distributions = events(box, "cycle_distribution:")
        self.assertEqual([e["train_cycle/median_loss"] for e in distributions], [59.5, 59.5])
        self.assertAlmostEqual(distributions[0]["train_cycle/p05_loss"], 5.95)
        self.assertAlmostEqual(distributions[1]["train_cycle/p95_loss"], 113.05)
        extra = events(box, "replay:")
        self.assertEqual(len(extra), 60)
        self.assertEqual(extra[0]["replay/update"], 121)
        self.assertEqual(extra[-1]["replay/update"], 300)
        self.assertEqual(extra[0]["replay/group"], "extra_top6")
        self.assertEqual(extra[-1]["replay/group"], "extra_top24")
        self.assertEqual(extra[-1]["replay/new_normal_update"], 240)
        raw = events(box, "train:")
        self.assertEqual([e["train/update"] for e in raw], [100, 200, 300])
        self.assertEqual([e["train/update_kind"] for e in raw], ["normal", "normal", "extra_top24"])
        saved = box.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        self.poll(box, cfg, [], 3000)
        self.assertEqual(box.db.execute("SELECT COUNT(*) FROM events").fetchone()[0], saved)

    def test_late_fixed_actual_boundary_has_exact_normal_extra_exposure(self):
        cfg = config()  # actual 15360 is 240 new steps, only 210 normal steps
        rows = journal(cfg)
        box = self.box(cfg)
        self.poll(box, cfg, rows, 3000, [evaluation(cfg["initial_update"], r2=.5)])
        terminal = {"outcome": "completed", "update": rows[-1]["update"],
                    "total_seconds": 175., "normal_update": 15360, "extra_updates": 60}
        self.poll(box, cfg, [], 3000, [evaluation(15360), evaluation(15420)], terminal)
        fixed = events(box, "evaluation:")
        periodic = fixed[1]
        self.assertEqual(periodic["fixed/update"], 240)
        self.assertEqual(periodic["fixed/normal_update"], 15330)
        self.assertEqual(periodic["fixed/new_normal_update"], 210)
        self.assertEqual(periodic["fixed/new_extra_updates"], 30)
        self.assertEqual(periodic["fixed/pre_evaluation_seconds"], 60.)
        self.assertEqual(periodic["fixed/table_119/cumulative_extra_exposure"], 2)
        self.assertEqual(periodic["fixed/table_119/new_normal_exposure"], 1)
        self.assertEqual(periodic["fixed/table_119/new_exposure"], 3)
        self.assertEqual(periodic["fixed/table_000/new_normal_exposure"], 2)
        self.assertEqual(periodic["fixed/table_000/reference_r2"], 0.)
        self.assertEqual(periodic["fixed/table_000/best_r2"], .5)
        final = fixed[2]
        self.assertEqual(final["fixed/new_normal_update"], 240)
        self.assertEqual(final["fixed/new_extra_updates"], 60)
        self.assertEqual(final["fixed/table_119/new_extra_exposure"], 4)
        self.assertTrue(box.cycles_caught_up_to_terminal(cfg))
        self.assertEqual(events(box, "terminal")[0]["terminal/new_normal_update"], 240)
        self.poll(box, cfg, [], 3000, [evaluation(15360, "duplicate-name.json")], terminal)
        self.assertEqual(len(events(box, "evaluation:")), 3)

    def test_replay_invalid_stream_rolls_back_events_exposure_and_cursors(self):
        cfg = config()
        rows = journal(cfg)
        box = self.box(cfg)
        self.poll(box, cfg, rows[:57], 570)
        before = list(box.db.iterdump())
        # Valid normal cycle followed by an extra loss outside the selected set.
        bad = copy.deepcopy(rows[57:123])
        bad[-1]["table"] = TABLES[0]
        with self.assertRaises(ValueError):
            box.ingest(snapshot(bad, 1230), cfg, REFERENCES)
        self.assertEqual(list(box.db.iterdump()), before)
        # A late scalar counter error also rolls back earlier SQL writes.
        bad = copy.deepcopy(rows[57:123])
        bad[-1]["extra_updates"] += 1
        with self.assertRaises(ValueError):
            box.ingest(snapshot(bad, 1230), cfg, REFERENCES)
        self.assertEqual(list(box.db.iterdump()), before)
        bad = copy.deepcopy(rows[57:123])
        bad[-1]["normal_update"] += 1
        with self.assertRaises(ValueError):
            box.ingest_cycles(snapshot(bad, 1230), cfg, REFERENCES)
        self.assertEqual(list(box.db.iterdump()), before)

    def test_replay_identity_and_budget_guard(self):
        cfg = config()
        box = self.box(cfg)
        explicit_zero = {**cfg, "base_extra_counts": {table: 0 for table in TABLES}}
        second = self.box(explicit_zero)
        self.assertEqual(box.get("identity"), second.get("identity"))
        for changes in ({"initial_normal_update": cfg["initial_update"] - 1},
                        {"new_max_updates": 240},
                        {"global_max_updates": cfg["global_max_updates"] + 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.box({**cfg, **changes})
        changed = copy.deepcopy(cfg)
        changed["loss_replay"]["normal_max_updates"] += 120
        changed["new_max_updates"] += 150
        changed["global_max_updates"] += 150
        with self.assertRaises(ValueError):
            self.box(changed)
        with self.assertRaises(ValueError):
            box.cycle_request({**cfg, "loss_replay": {"kind": "unknown"}})

    def test_v1_cycle_and_old_database_identity_remain_compatible(self):
        cfg = config(base=122880, replay=False)
        box = self.box(cfg)
        rows = journal(cfg)
        self.poll(box, cfg, rows[:57], 570)
        self.poll(box, cfg, rows[57:], 2400)
        self.assertEqual(box.get("cycle_distribution_schema"), "median-p05-p95-linear-v1")
        self.assertEqual(box.cycle_status(cfg)["cycle_complete_count"], 2)
        self.assertEqual(box.cycle_status(cfg)["cycle_partial_steps"], 0)
        self.assertEqual(events(box, "replay:"), [])
        self.assertEqual([e["train_cycle/global_update"] for e in events(box, "cycle_loss:")], [123000, 123120])
        self.assertNotIn("train_cycle/normal_update", events(box, "cycle_loss:")[0])
        self.assertEqual(set(box.get("identity")), {"run_id", "identity_sha256", "remote_output",
                         "initial_update", "base_counts", "base_total_seconds", "evaluate_every"})
        second = self.box(cfg)
        self.assertEqual(second.request(cfg)["last_update"], 123120)

    def test_remote_reader_preserves_replay_fields_and_partial_line(self):
        cfg = config()
        rows = journal(cfg)[:2]
        root = Path(self.temp.name) / "synthetic-run"
        root.mkdir()
        (root / "resolved.json").write_text(json.dumps({"identity": {"sha256": "identity",
            "source": {"sha256": "source"}, "datasets": {t: {} for t in TABLES}}, "spec": {"model": {"heads": 4}}}))
        complete = "".join(json.dumps(row) + "\n" for row in rows)
        (root / "updates.jsonl").write_text(complete + '{"update":')
        request = {"output": str(root), "offset": 0, "last_update": cfg["initial_update"],
                   "seen_evaluations": [], "cycles_only": True}
        result = subprocess.run([sys.executable, "-c", REMOTE], input=json.dumps(request) + "\n",
                                text=True, capture_output=True, check=True)
        received = json.loads(result.stdout)
        self.assertEqual(received["updates"], rows)
        self.assertEqual(received["offset"], len(complete.encode()))
        self.assertEqual(received["declared_model"], {"heads": 4})

    def test_cloud_contract_uses_explicit_axes_without_online_sdk(self):
        cfg = config()
        cfg["parent_checkpoint_sha256"] = "parent-checkpoint"
        captured, defined, config_updates = {}, [], {}

        class FakeConfig:
            def update(self, values, **kwargs):
                config_updates.update(values)

        run = types.SimpleNamespace(config=FakeConfig(), define_metric=lambda *a, **kw: defined.append((a, kw)))

        def init(**kwargs):
            captured.update(kwargs)
            return run

        fake = types.SimpleNamespace(init=init, Settings=lambda **kwargs: kwargs)
        with patch.dict(sys.modules, {"wandb": fake}):
            self.assertIs(connect(cfg, Path(self.temp.name)), run)
        self.assertIn((("replay/*",), {"step_metric": "replay/update"}), defined)
        self.assertIn((("train_cycle/*",), {"step_metric": "train_cycle/update"}), defined)
        self.assertEqual(captured["config"]["new_normal_max_updates"], 240)
        self.assertEqual(captured["config"]["new_actual_max_updates"], 300)
        self.assertEqual(captured["config"]["parent_checkpoint_sha256"], "parent-checkpoint")
        self.assertIn("NORMAL", config_updates["cycle_loss_definition"])
        self.assertIn("excluded", config_updates["cycle_loss_definition"])


def weighted_config():
    cfg = config(base=15120)
    extras = {table: 2 if i >= 114 else 1 if i >= 96 else 0 for i, table in enumerate(TABLES)}
    cfg.update(initial_update=15150, initial_extra_updates=30,
               base_extra_counts=extras, base_counts={t: 126 + extras[t] for t in TABLES},
               new_max_updates=324, global_max_updates=15474,
               loss_replay={"kind": WEIGHTED_REPLAY_KIND, "normal_max_updates": 15360,
                            "start_normal_cursor": 15120, "start_extra_updates": 30})
    return cfg


def weighted_journal(cfg):
    rows = weighted_replay_rows(base=cfg["initial_update"], normal_base=cfg["initial_normal_update"])
    for i, row in enumerate(rows, 1):
        row.update(gradient_norm=1., seconds=.25, total_seconds=cfg["base_total_seconds"] + i * .25)
    return rows


class WeightedReplayMirrorTests(unittest.TestCase):
    setUp = ReplayMirrorTests.setUp
    tearDown = ReplayMirrorTests.tearDown
    box = ReplayMirrorTests.box
    poll = ReplayMirrorTests.poll

    def test_weighted_counter_resume_and_late_fixed_exposure(self):
        cfg = weighted_config()
        rows = weighted_journal(cfg)
        box = self.box(cfg)
        self.poll(box, cfg, rows[:123], 1230)
        self.assertEqual(box.cycle_status(cfg)["cycle_new_extra_updates"], 3)
        self.assertEqual(box.cycle_status(cfg)["cycle_cumulative_extra_updates"], 33)
        self.assertEqual(box.cycle_status(cfg)["cycle_extra_pass_index"], 1)
        box.db.close(); self.boxes.remove(box)
        box = self.box(cfg)
        terminal = {"outcome": "completed", "update": 15474, "total_seconds": 181.,
                    "normal_update": 15360, "extra_updates": 114}
        self.poll(box, cfg, rows[123:], 3240, terminal=terminal)
        self.poll(box, cfg, [], 3240, evaluations=[evaluation(15360), evaluation(15474)])
        means = events(box, "cycle_loss:")
        self.assertEqual([e["train_cycle/update"] for e in means], [120, 240])
        self.assertEqual([e["train_cycle/global_update"] for e in means], [15270, 15432])
        self.assertEqual([e["train_cycle/cumulative_extra_updates"] for e in means], [30, 72])
        self.assertEqual([e["train_cycle/new_extra_updates"] for e in means], [0, 42])
        self.assertAlmostEqual(events(box, "cycle_distribution:")[0]["train_cycle/p99_loss"], 117.81)
        self.assertEqual(len(events(box, "replay:")), 84)
        first = events(box, "replay:")[0]
        self.assertEqual(first["replay/group"], "extra_top2")
        self.assertEqual(first["replay/extra_updates"], 31)
        self.assertEqual(first["replay/new_extra_updates"], 1)
        fixed = events(box, "evaluation:")
        self.assertEqual(fixed[0]["fixed/new_normal_update"], 168)
        self.assertEqual(fixed[0]["fixed/new_extra_updates"], 42)
        self.assertEqual(fixed[0]["fixed/extra_updates"], 72)
        self.assertEqual(fixed[1]["fixed/table_119/new_extra_exposure"], 12)
        self.assertEqual(fixed[1]["fixed/table_119/cumulative_extra_exposure"], 14)
        self.assertEqual(fixed[1]["fixed/table_119/new_normal_exposure"], 2)
        self.assertEqual(fixed[1]["fixed/table_119/cumulative_normal_exposure"], 128)
        self.assertEqual(box.cycle_status(cfg)["cycle_new_extra_updates"], 84)
        self.assertEqual(box.cycle_status(cfg)["cycle_cumulative_extra_updates"], 114)
        self.assertEqual(events(box, "terminal")[0]["terminal/new_extra_updates"], 84)

    def test_weighted_budget_initial_extra_and_invalid_pass_guard(self):
        cfg = weighted_config()
        for change in ({"initial_extra_updates": 0}, {"new_max_updates": 300},
                       {"base_extra_counts": {t: 0 for t in TABLES}},
                       {"loss_replay": {k: v for k, v in cfg["loss_replay"].items() if k != "start_extra_updates"}},
                       {"loss_replay": cfg["loss_replay"] | {"start_extra_updates": 0}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.box(cfg | change)
        box = self.box(cfg)
        rows = weighted_journal(cfg)
        self.poll(box, cfg, rows[:123], 1230)
        before = list(box.db.iterdump())
        rows[123]["table"] = rows[122]["table"]
        with self.assertRaises(ValueError):
            box.ingest(snapshot(rows[123:], 3240), cfg, REFERENCES)
        self.assertEqual(list(box.db.iterdump()), before)

    def test_weighted_cloud_metadata_additive_groups_and_p99(self):
        cfg = weighted_config()
        captured, updates = {}, {}
        class FakeConfig:
            def update(self, values, **kwargs):
                updates.update(values)
        run = types.SimpleNamespace(config=FakeConfig(), define_metric=lambda *a, **kw: None)
        def init(**kwargs):
            captured.update(kwargs)
            return run
        with patch.dict(sys.modules, {"wandb": types.SimpleNamespace(init=init, Settings=lambda **kw: kw)}):
            connect(cfg, Path(self.temp.name))
        metadata = captured["config"]
        self.assertEqual(metadata["initial_extra_updates"], 30)
        self.assertEqual(metadata["extra_updates_per_cycle"], 42)
        self.assertEqual(metadata["actual_updates_per_cycle"], 162)
        self.assertEqual(metadata["new_extra_max_updates"], 84)
        self.assertIn("highest2 tables receive6 extras", metadata["loss_replay_strategy"])
        self.assertIn("train_cycle/p99_loss", updates["cycle_distribution_metrics"])
        self.assertIs(metadata["best_includes_initial_boundary_evaluation"], False)
        with patch.dict(sys.modules, {"wandb": types.SimpleNamespace(init=init, Settings=lambda **kw: kw)}):
            connect(cfg | {"best_includes_initial_boundary_evaluation": True}, Path(self.temp.name))
        self.assertIs(captured["config"]["best_includes_initial_boundary_evaluation"], True)


if __name__ == "__main__":
    unittest.main()
