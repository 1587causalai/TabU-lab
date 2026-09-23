"""Terminal telemetry regression checks: pure policy, real reader, fake SDK."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from terminal_lifecycle import classify_terminal, finish_telemetry
from wandb_mirror import REMOTE, main


def stopped(**changes):
    result = {"outcome": "interrupted", "update": 150, "durable_update": 150,
              "total_seconds": 10.,
              "checkpoint_sha256": "a" * 64, "checkpoint_verified": True,
              "terminal_identity_matches": True, "error_type": None, "error_present": False}
    result.update(changes)
    return result


class TerminalLifecycleTests(unittest.TestCase):
    def test_saved_boundary_stops_finish_without_claiming_budget_completion(self):
        for outcome in ("stopped", "interrupted"):
            with self.subTest(outcome=outcome):
                terminal = stopped(outcome=outcome)
                before = copy.deepcopy(terminal)
                result = classify_terminal(terminal, {"global_max_updates": 150})
                self.assertEqual(result["exit_code"], 0)
                self.assertEqual(result["summary"]["training_outcome"], outcome)
                self.assertEqual(result["summary"]["telemetry/lifecycle"], "closed_after_stop")
                self.assertTrue(result["summary"]["training_intentionally_stopped"])
                self.assertFalse(result["summary"]["training_budget_completed"])
                self.assertEqual(terminal, before)

    def test_exception_and_unverified_stops_remain_failed(self):
        variants = [{"error_type": "KeyboardInterrupt", "error": ""}, {"error": "failure"},
                    {"error_present": True}, {"durable_update": 149}, {"durable_update": None},
                    {"durable_update": True}, {"update": True}, {"update": -1},
                    {"checkpoint_verified": False}, {"checkpoint_verified": 1},
                    {"checkpoint_sha256": "invalid"}, {"terminal_identity_matches": False},
                    {"terminal_identity_matches": None}]
        for change in variants:
            with self.subTest(change=change):
                result = classify_terminal(stopped(**change), {"global_max_updates": 300})
                self.assertEqual(result["exit_code"], 1)
                self.assertFalse(result["summary"]["training_budget_completed"])
                self.assertFalse(result["summary"]["training_intentionally_stopped"])

    def test_completed_outcome_and_budget_are_independent(self):
        terminal = stopped(outcome="completed", normal_update=120)
        config = {"global_max_updates": 150, "loss_replay": {"normal_max_updates": 120}}
        result = classify_terminal(terminal, config)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["summary"]["training_budget_completed"])
        self.assertFalse(result["summary"]["training_intentionally_stopped"])
        for change in ({"normal_update": 119}, {"update": 149, "durable_update": 149}, {"normal_update": None}):
            with self.subTest(change=change):
                result = classify_terminal(terminal | change, config)
                self.assertEqual(result["exit_code"], 0)
                self.assertFalse(result["summary"]["training_budget_completed"])
        self.assertEqual(classify_terminal(terminal | {"error_present": True}, config)["exit_code"], 1)
        self.assertEqual(classify_terminal(terminal | {"durable_update": 149}, config)["exit_code"], 1)

    def test_budget_exhausted_inconclusive_and_unknown_never_become_completed(self):
        for outcome in ("budget_exhausted", "inconclusive", "gate_failed", "failed", "started", "other", None):
            with self.subTest(outcome=outcome):
                result = classify_terminal(stopped(outcome=outcome), {"global_max_updates": 150})
                self.assertEqual(result["exit_code"], 1)
                self.assertFalse(result["summary"]["training_budget_completed"])
                self.assertNotEqual(result["summary"]["training_outcome"], "completed")

    def test_finish_preserves_original_outcome_without_uploading_error_text(self):
        finishes = []
        run = types.SimpleNamespace(summary={}, finish=lambda **kw: finishes.append(kw))
        decision = finish_telemetry(run, stopped(error="PRIVATE ERROR TEXT", unrelated_metadata={"private": "not uploaded"}), {})
        self.assertEqual(finishes, [{"exit_code": 1}])
        self.assertEqual(run.summary["terminal/outcome"], "interrupted")
        self.assertNotIn("PRIVATE ERROR TEXT", json.dumps(run.summary))
        self.assertNotIn("unrelated_metadata", json.dumps(run.summary))
        self.assertEqual(run.summary["telemetry/lifecycle"], decision["summary"]["telemetry/lifecycle"])


class TerminalReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "run"
        self.root.mkdir()
        (self.root / "resolved.json").write_text(json.dumps({"identity": {"sha256": "id", "source": {"sha256": "source"}, "datasets": {}}, "spec": {"model": {}}}))
        self.payload = b"small synthetic checkpoint bytes; no torch needed"
        self.digest = hashlib.sha256(self.payload).hexdigest()
        self.generation = self.root / "checkpoints" / (self.digest + ".pt")
        self.generation.parent.mkdir()
        self.generation.write_bytes(self.payload)
        (self.root / "checkpoint-progress.pt").symlink_to(self.generation.relative_to(self.root))
        self.receipt = {"outcome": "interrupted", "update": 150, "durable_update": 150,
                        "checkpoint": "checkpoint-progress.pt", "checkpoint_sha256": self.digest,
                        "identity": {"sha256": "id"}, "error_type": None}

    def tearDown(self):
        self.temp.cleanup()

    def read(self, **changes):
        (self.root / "terminal.json").write_text(json.dumps(self.receipt | changes))
        request = {"output": str(self.root), "offset": 0, "last_update": 150, "seen_evaluations": []}
        result = subprocess.run([sys.executable, "-c", REMOTE], input=json.dumps(request) + "\n", text=True, capture_output=True, check=True)
        return json.loads(result.stdout)["terminal"]

    def test_real_reader_hashes_committed_generation_and_redacts_errors(self):
        terminal = self.read()
        self.assertTrue(terminal["checkpoint_verified"])
        self.assertTrue(terminal["terminal_identity_matches"])
        self.assertEqual(classify_terminal(terminal, {})["exit_code"], 0)
        # Sidecars can be ahead of the pointer after a failed save; do not trust them.
        (self.root / "checkpoint-progress.json").write_text('{"sha256":"stale-sidecar"}')
        self.assertTrue(self.read()["checkpoint_verified"])
        terminal = self.read(error="DO NOT UPLOAD THIS")
        self.assertTrue(terminal["error_present"])
        self.assertNotIn("DO NOT UPLOAD THIS", json.dumps(terminal))
        self.assertEqual(classify_terminal(terminal, {})["exit_code"], 1)

    def test_missing_empty_changed_hash_and_external_checkpoint_are_rejected(self):
        for contents in (b"", b"changed bytes"):
            self.generation.write_bytes(contents)
            self.assertFalse(self.read()["checkpoint_verified"])
        self.generation.unlink()
        self.assertFalse(self.read()["checkpoint_verified"])
        outside = Path(self.temp.name) / (self.digest + ".pt")
        outside.write_bytes(self.payload)
        pointer = self.root / "checkpoint-progress.pt"
        pointer.unlink()
        pointer.symlink_to(outside)
        self.assertFalse(self.read()["checkpoint_verified"])

    def test_terminal_identity_and_digest_claim_must_match(self):
        self.assertFalse(self.read(checkpoint_sha256="bad")["checkpoint_verified"])
        terminal = self.read(identity={"sha256": "another-run"})
        self.assertFalse(terminal["terminal_identity_matches"])
        self.assertEqual(classify_terminal(terminal, {})["exit_code"], 1)
        self.assertFalse(self.read(checkpoint=str(self.generation))["checkpoint_verified"])


class MainCloseIntegrationTests(unittest.TestCase):
    def test_main_persists_closure_after_drained_outbox_and_successful_sdk_finish(self):
        with tempfile.TemporaryDirectory() as temporary:
            panel = Path(temporary)
            tables = [f"table_{i:03d}" for i in range(120)]
            config = {"run_id": "offline-test", "entity": "none", "project": "none", "host": "unused", "remote_output": "/unused",
                      "identity_sha256": "id", "source_sha256": "source", "initial_update": 0,
                      "base_counts": {t: 0 for t in tables}, "base_total_seconds": 0, "evaluate_every": 15360,
                      "heads": 4, "backbone_layers": 2, "unit_layers": 0, "device": "mps", "dtype": "float32",
                      "model_label": "synthetic", "initialization": "none", "new_max_updates": 120}
            cfgpath = panel / "config.json"
            cfgpath.write_text(json.dumps(config))
            (panel / "naive-query-reference.json").write_text(json.dumps({"records": [{"table": t, "metrics": {}} for t in tables]}))
            terminal = stopped(update=0, durable_update=0)
            snapshot = {"identity_sha256": "id", "source_sha256": "source", "tables": tables, "offset": 0,
                        "updates": [], "evaluations": [], "terminal": terminal}
            delivered, finishes = [], []
            run = types.SimpleNamespace(id="offline-test", url="offline://test", step=0, summary={},
                log=lambda data, **kw: delivered.append((data, kw)), finish=lambda **kw: finishes.append(kw))
            with patch.object(sys, "argv", ["mirror", "--root", str(panel), "--config", str(cfgpath)]), patch("wandb_mirror.read_remote", return_value=snapshot), patch("wandb_mirror.connect", return_value=run):
                main()
            self.assertEqual(finishes, [{"exit_code": 0}])
            self.assertEqual(delivered[0][0]["terminal/outcome"], "interrupted")
            self.assertFalse(run.summary["training_budget_completed"])
            status = json.loads((panel / "wandb-mirror-status.json").read_text())
            self.assertEqual(status["terminal"]["outcome"], "interrupted")
            self.assertEqual(status["telemetry_closure"]["summary"]["telemetry/lifecycle"], "closed_after_stop")
            with sqlite3.connect(panel / "wandb-mirror.sqlite3") as db:
                self.assertTrue(json.loads(db.execute("select value from meta where key='finished'").fetchone()[0]))
                self.assertEqual(json.loads(db.execute("select value from meta where key='telemetry_closure'").fetchone()[0])["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
