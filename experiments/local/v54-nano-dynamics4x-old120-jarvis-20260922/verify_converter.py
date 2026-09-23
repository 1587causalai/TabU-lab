"""Bounded CPU-only conversion checks using synthetic, never-trained weights.

No real checkpoint, remote host, CUDA/MPS execution, optimizer step or training
runner is used. A temporary full-shape Nano fixture exercises the real model,
checkpoint serializer and fresh-state validator; a six-row forward compares
the target against four calls to the same original two-layer backbone.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PANEL = Path(__file__).resolve().parent
REPO = PANEL.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(PANEL))

import torch
from tabu_lab.curriculum_v53 import artifacts, runner
from tabu_lab.curriculum_v53.data import build_episode
from tabu_lab.curriculum_v53.factory import make_model
from tabu_lab.curriculum_v53.protocol import load_v54_plan
from tabu_lab.restoration_optimizers import adamw

import convert_dynamics4x as converter


def fixture_plan(root, *, layers, seed):
    raw = json.dumps({
        "features": [{"kind": "numeric"}, {"kind": "numeric"}],
        "values": [[float(i), float(i * i + i)] for i in range(8)],
        "splits": {"train": list(range(6)), "validation": [6], "test": [7]},
    }).encode()
    (root / "table.json").write_bytes(raw)
    spec = {
        "schema": "tabu.curriculum.v54.v1", "experiment_id": f"synthetic-conversion-l{layers}",
        "seeds": dict(model=seed, order=2, masks=3, codes=4, windows=5, evaluation=6),
        "model": {"size": "nano", "backbone": {"layers": layers}, "center_chunk_size": 4},
        "tables": [{"id": f"fixture-{i:03}", "kind": "synthetic", "cohort": "old120",
                    "path": "table.json", "sha256": hashlib.sha256(raw).hexdigest()}
                   for i in range(120)],
        "probes": [],
        "stages": [{
            "name": "fit", "question": "Synthetic converter validation only; never run training",
            "max_updates": converter.ACTUAL_MAX_UPDATES, "max_seconds": 120,
            "sampling": [{"cohort": "old120", "episodes": 120}],
            "recipe": {"synthetic": {"kind": "supervised_row", "fraction": .25}},
            "optimizer": "adamw", "evaluate_every": 162, "checkpoint_every": 162,
            "probes": [], "loss_replay": {
                "kind": "normal120_p99x3_p95x2_p80x1_v2",
                "normal_max_updates": converter.NORMAL_MAX_UPDATES,
                "start_normal_cursor": 0, "start_extra_updates": 0,
            },
        }],
    }
    path = root / f"fixture-l{layers}.json"
    path.write_text(json.dumps(spec))
    return load_v54_plan(path)


class ConverterChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="tabu-dynamics4x-cpu-")
        cls.root = Path(cls.temporary.name)
        cls.runtime = runner.configure_runtime("cpu")
        cls.parent_plan = fixture_plan(cls.root, layers=2, seed=77)
        cls.plan = fixture_plan(cls.root, layers=8, seed=19)
        runner._seed_model(cls.parent_plan)
        parent_model = make_model(cls.parent_plan).float()
        optimizer = adamw(parent_model, cls.parent_plan.optimizer)
        # Synthetic nonempty moments verify that conversion discards the parent
        # optimizer. No backward or optimizer step is performed.
        parameter = next(parent_model.parameters())
        optimizer.state[parameter] = {
            "step": torch.tensor(7.), "exp_avg": torch.full_like(parameter, .01),
            "exp_avg_sq": torch.full_like(parameter, .02),
        }
        state = runner._new_state(cls.parent_plan.spec["stages"])
        state.update(update=7, cursor=7, phase="train", total_seconds=12., stage_seconds=[12.],
                     exposure={"fixture-000": {"updates": 7}})
        cls.parent_path = cls.root / "checkpoint-fixture-parent.pt"
        cls.parent_sha = artifacts.save_checkpoint(
            cls.parent_path, plan=cls.parent_plan, model=parent_model, optimizer=optimizer,
            state=state, runtime={**cls.runtime, "dtype": "float32", "synthetic_fixture": True},
            lineage=[{"mode": "synthetic_fixture_never_trained"}],
        )
        cls.parent, _ = artifacts.load_checkpoint(cls.parent_path)
        cls.receipt = converter.convert(cls.plan, cls.parent_path, cls.root / "converted",
                                        device="cpu", parent_sha256=cls.parent_sha)
        cls.result, _ = artifacts.load_checkpoint(cls.receipt["checkpoint"])
        cls.observed = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def new_target(self):
        model = make_model(self.plan).double().eval()
        model.load_state_dict(self.result["model"], strict=True)
        return model

    def test_exact_mapping_and_non_backbone_preservation(self):
        self.assertEqual(self.receipt["backbone_layer_sequence"], [0, 1, 0, 1, 0, 1, 0, 1])
        self.assertEqual(set(self.result["model"]), set(self.new_target().state_dict()))
        for name, value in self.result["model"].items():
            source = converter._parent_key(name)
            self.assertTrue(torch.equal(value, self.parent["model"][source].to(value.dtype)), name)
            if not name.startswith("backbone."):
                self.assertEqual(name, source)
        self.assertEqual(self.receipt["config_changes"], {"backbone.layers": {"before": 2, "after": 8}})
        self.assertEqual(len(self.new_target().unit_blocks), 0)

    def test_independent_copies_can_change_without_mutating_siblings_or_parent(self):
        model = self.new_target()
        converter._independent_storage(model.named_parameters(remove_duplicate=False))
        sibling = model.backbone.layers[2].row.out.weight.detach().clone()
        parent = self.parent["model"]["backbone.layers.0.row.out.weight"].clone()
        with torch.no_grad():
            model.backbone.layers[0].row.out.weight[0, 0].add_(1.)
        self.assertTrue(torch.equal(sibling, model.backbone.layers[2].row.out.weight))
        self.assertTrue(torch.equal(parent, self.parent["model"]["backbone.layers.0.row.out.weight"]))
        self.assertFalse(torch.equal(model.backbone.layers[0].row.out.weight, sibling))

    def test_complete_forward_matches_four_original_backbone_passes(self):
        class FourPasses(torch.nn.Module):
            def __init__(self, original):
                super().__init__()
                self.original = original

            def forward(self, h, visible, query):
                for _ in range(4):
                    h = self.original(h, visible, query)
                return h

        target = self.new_target()
        original = make_model(self.parent_plan).double().eval()
        original.load_state_dict(self.parent["model"], strict=True)
        reference = copy.deepcopy(target)
        reference.backbone = FourPasses(original.backbone)
        inputs, request, _truth, info = build_episode(
            self.plan.tables[0], {"kind": "supervised_row", "fraction": .25}, 0,
            self.plan.spec["seeds"], "cpu", evaluation=True, partition="train",
            epsilon=self.plan.config.epsilon, codec_version=self.plan.config.codec_version,
        )
        with torch.no_grad():
            actual, expected = target(inputs, request), reference(inputs, request)
        comparisons = {"carriers": (actual.carriers, expected.carriers),
                       "units": (actual.units, expected.units)}
        self.assertEqual(len(actual.columns), len(expected.columns))
        for left, right in zip(actual.columns, expected.columns, strict=True):
            self.assertEqual(left.result.status, "ok")
            self.assertEqual(left.column, right.column)
            self.assertTrue(torch.equal(left.target_indices, right.target_indices))
            comparisons[f"encoding/{left.column}"] = (left.result.encoding, right.result.encoding)
            comparisons[f"decoded/{left.column}"] = (left.decoded, right.decoded)
        errors = {}
        for name, (left, right) in comparisons.items():
            self.assertTrue(bool(torch.isfinite(left).all() and torch.isfinite(right).all()))
            errors[name] = float((left - right).abs().max())
            self.assertTrue(torch.allclose(left, right, rtol=1e-12, atol=1e-12), name)
        self.observed["four_pass_forward"] = {"max_abs_errors": errors, "query_count": info["query_count"]}

    def test_fresh_optimizer_rng_cursors_and_complete_resume_state(self):
        self.assertTrue(self.parent["optimizer"]["state"])
        self.assertFalse(self.result["optimizer"]["state"])
        state = self.result["state"]
        self.assertEqual((state["update"], state["cursor"], state["exposure"]), (0, 0, {}))
        self.assertEqual(state["loss_replay"]["normal_cursor"], 0)
        self.assertEqual(state["loss_replay"]["extra_updates"], 0)
        self.assertFalse(state["loss_replay"]["queue"])
        runner._validate_resume(self.result, self.plan, self.runtime)
        runner._seed_model(self.plan)
        fresh = artifacts.rng_state()
        self.assertTrue(torch.equal(fresh["torch"], self.result["rng"]["torch"]))
        self.assertEqual(fresh["python"], self.result["rng"]["python"])
        self.assertFalse(torch.equal(self.parent["rng"]["torch"], self.result["rng"]["torch"]))
        self.assertEqual(self.result["lineage"][0]["mode"], "synthetic_fixture_never_trained")
        self.assertFalse(self.result["lineage"][-1]["function_preservation_claimed"])

    def test_parent_immutable_and_serialized_storage_independent(self):
        self.assertEqual(artifacts.sha256(self.parent_path), self.parent_sha)
        self.assertEqual(converter._independent_storage(self.result["model"].items()),
                         len(self.result["model"]))

    def test_reject_contract_source_codec_and_budget_drift(self):
        changes = [
            lambda p: p["model_config"]["backbone"].update(heads=8),
            lambda p: p["model_config"].update(codec_version="unit_gaussian_composition_v1"),
            lambda p: p["identity"]["source"].update(sha256="wrong-source"),
            lambda p: p.update(purpose="qualification"),
        ]
        for change in changes:
            with self.subTest(change=change):
                payload = copy.deepcopy(self.parent)
                change(payload)
                with self.assertRaises(ValueError):
                    converter._validate_contract(self.plan, payload)
        altered = copy.deepcopy(self.plan)
        altered.spec["stages"][0]["max_updates"] += 1
        with self.assertRaisesRegex(ValueError, "authorized budget"):
            converter._validate_contract(altered, self.parent)

    def test_reject_target_changes_other_than_depth(self):
        from dataclasses import replace
        from tabu_lab.models.restoration_v54 import V54Config
        changed = V54Config(size="nano", backbone={"layers": 8, "heads": 8}, center_chunk_size=4)
        with self.assertRaisesRegex(ValueError, "only backbone.layers"):
            converter._validate_contract(replace(self.plan, config=changed), self.parent)

    def test_reject_missing_parent_tensor_and_bad_codec_buffer(self):
        for mode in ("missing", "codec"):
            payload = copy.deepcopy(self.parent)
            if mode == "missing":
                payload["model"].pop("encoder.unit_seed")
            else:
                payload["model"]["_codec_signature"].add_(1)
            with self.subTest(mode=mode), patch.object(artifacts, "load_checkpoint", return_value=(payload, self.parent_sha)):
                with self.assertRaisesRegex(ValueError, "parent state keys|codec signature"):
                    converter.convert(self.plan, self.parent_path, self.root / f"invalid-{mode}", device="cpu")

    def test_reject_wrong_digest_overwrite_device_and_alias(self):
        with self.assertRaisesRegex(ValueError, "pinned checkpoint"):
            converter.convert(self.plan, self.parent_path, self.root / "wrong-sha", device="cpu", parent_sha256="0" * 64)
        with self.assertRaises(FileExistsError):
            converter.convert(self.plan, self.parent_path, self.root / "converted", device="cpu")
        with self.assertRaisesRegex(ValueError, "qualified CUDA"):
            converter.convert(self.plan, self.parent_path, self.root / "bad-device", device="mps")
        value = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "storage alias"):
            converter._independent_storage([("first", value[:2]), ("second", value[2:])])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new JSON verification receipt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("verification receipt already exists")
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConverterChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic full-shape CPU conversion; six-row complete forward only. No real checkpoint, remote access, training runner, optimizer step, CUDA or MPS execution.",
        "passed": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [str(test) for test, _ in result.failures],
        "errors": [str(test) for test, _ in result.errors],
        "torch": str(torch.__version__), "device": "cpu", "target_dtype": "float64",
        "source_parent_dtype": "float32",
        "converter_sha256": artifacts.sha256(PANEL / "convert_dynamics4x.py"),
        "verification_script_sha256": artifacts.sha256(__file__),
    }
    if hasattr(ConverterChecks, "receipt"):
        conversion = ConverterChecks.receipt
        receipt["conversion"] = {key: conversion[key] for key in (
            "backbone_layer_sequence", "config_changes", "parent_state_entries", "target_state_entries",
            "non_backbone_state_entries", "parent_parameter_count", "target_parameter_count",
            "independent_parameter_count", "parent_checkpoint_unchanged", "optimizer_fresh",
            "fresh_update", "fresh_normal_update", "fresh_extra_updates", "normal_max_updates", "actual_max_updates",
        )}
        receipt["source_identity"] = conversion["new_identity"]["source"]
        receipt["forward"] = ConverterChecks.observed.get("four_pass_forward")
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "passed": receipt["passed"], "tests_run": result.testsRun}))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
