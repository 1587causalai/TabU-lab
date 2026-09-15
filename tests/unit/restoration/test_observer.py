"""Observation is optional, filtered, resumable, and isolated from fit receipts."""

from __future__ import annotations

import json

import pytest

from tabu_lab.observers.restoration import create_restoration_observer


class FakeRun:
    url = "https://wandb.ai/example/restoration/runs/stable-run"

    def __init__(self):
        self.logs = []
        self.definitions = []
        self.summary = {}
        self.finished = 0
        self.raise_on_log = False

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, payload):
        if self.raise_on_log:
            raise RuntimeError("secret credentials must never appear in warnings")
        self.logs.append(payload)

    def finish(self):
        self.finished += 1


class FakeSDK:
    def __init__(self):
        self.calls = []
        self.settings = []
        self.run = FakeRun()

    def Settings(self, **kwargs):
        self.settings.append(kwargs)
        return kwargs

    def init(self, **kwargs):
        self.calls.append(kwargs)
        return self.run


@pytest.fixture
def enabled(monkeypatch):
    for key in (
        "WANDB_PROJECT", "TABU_LAB_WANDB_PROJECT", "WANDB_NAME", "WANDB_ENTITY",
        "WANDB_GROUP", "WANDB_RUN_GROUP", "WANDB_RUN_ID", "WANDB_MODE", "WANDB_API_KEY",
        "TABU_LAB_WANDB_TABLE_METRICS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TABU_LAB_OBSERVER", "wandb")
    monkeypatch.setenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", "1")


def test_default_disabled_makes_no_sdk_calls(monkeypatch):
    monkeypatch.delenv("TABU_LAB_OBSERVER", raising=False)
    fake = FakeSDK()
    observer = create_restoration_observer({}, {}, wandb=fake)
    observer({"event": "update", "loss": 1.0})
    observer.close()
    assert fake.calls == []
    assert not observer.active
    assert observer.run_url is None


def test_online_requires_disclosure_gate_but_offline_does_not(enabled, monkeypatch):
    monkeypatch.delenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE")
    fake = FakeSDK()
    with pytest.warns(UserWarning, match="TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1"):
        observer = create_restoration_observer({}, {}, wandb=fake)
    assert not observer.active and not fake.calls
    monkeypatch.setenv("WANDB_MODE", "offline")
    observer = create_restoration_observer({}, {}, wandb=fake)
    assert observer.active
    assert fake.calls[0]["mode"] == "offline"


def test_metadata_and_events_are_allowlisted_and_sdk_auth_is_accepted(enabled, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_ID", "stable-run")
    monkeypatch.setenv("WANDB_ENTITY", "example")
    fake = FakeSDK()
    observer = create_restoration_observer(wandb=fake)
    assert fake.calls == []  # Lazy until the immutable run identity is known.
    observer({
        "event": "phase", "stage": "start", "update": 0,
        "config": {
            "max_rounds": 10, "model": {"encoder": {"width": 128, "token": "secret"}},
            "optimizer": {"kind": "adamw", "betas": [0.9, 0.95]},
            "hostname": "secret-host", "corpus": "/private/corpus", "api_key": "secret",
        },
        "identity": {
            "preregistration_sha256": "a" * 64,
            "source": {"sha256": "b" * 64, "files": {"/private/code.py": "secret"}},
            "corpus_manifest_sha256": "/private/path", "device": "cuda:0", "dtype": "float64",
            "dataset_sha256": {"sensitive-name": "secret"},
        },
    })
    assert observer.active and observer.run_url == fake.run.url
    assert fake.calls[0]["project"] == "restoration"
    assert fake.calls[0]["entity"] == "example"
    assert fake.calls[0]["id"] == "stable-run"
    assert fake.calls[0]["resume"] == "allow"
    assert fake.calls[0]["config"] == {
        "max_rounds": 10, "model/encoder/width": 128, "optimizer/kind": "adamw",
        "optimizer/betas": [0.9, 0.95], "preregistration_sha256": "a" * 64,
        "source_sha256": "b" * 64, "device": "cuda:0", "dtype": "float64",
    }
    assert fake.settings[0] == {
        "disable_git": True, "disable_code": True, "console": "off", "silent": True,
        "x_disable_stats": True, "x_disable_meta": True,
    }
    observer({"event": "update", "update": 5, "round": 1, "loss": 2.0,
              "gradient_norm": float("inf"), "update_seconds": 0.3,
              "elapsed_seconds": 4.5, "table": "/private/path", "api_key": "secret",
              "mask": {"query_coverage": 1 / 3, "raw_values": [1, 2]}})
    assert fake.run.logs[-1] == {
        "update": 5, "round": 1, "elapsed_seconds": 4.5, "train/loss": 2.0,
        "train/update_seconds": 0.3, "train/coverage/query_coverage": 1 / 3,
    }
    encoded = json.dumps(fake.calls + fake.run.logs)
    assert "secret" not in encoded and "/private" not in encoded
    observer.close()
    observer.close()
    assert fake.run.finished == 1


def test_evaluation_progress_and_terminal_summary_have_numeric_metrics(enabled):
    fake = FakeSDK()
    observer = create_restoration_observer({}, {}, wandb=fake, run_id="fixed-id")
    observer({"event": "evaluation_progress", "stage": "round_2", "update": 240,
              "completed_episodes": 120, "total_episodes": 960})
    assert fake.run.logs[-1] == {"phase": "round_2", "update": 240,
                                "completed_episodes": 120, "total_episodes": 960}
    values = {"encoding_mse": 0.2, "numeric_mse": 3.0, "count": 4,
              "discrete_accuracy": None, "raw_values": [10, 20], "secret": "private"}
    observer({"event": "summary", "metrics": {
        "outcome": "completed", "status": "local_unissued", "update": 240,
        "elapsed_seconds": 100.0, "checkpoint": "/private/checkpoint.pt",
        "final": {
            "by_state": {"query": values},
            "by_type": {"numeric": {"query": values}},
            "by_source": {"scm_mixed_v1": {"query": values},
                          "/private/source": {"query": values}},
            "by_table": {"scm_mixed_v1_000": {"query": values}},
            "coverage": {"query_fraction": 1 / 3, "query_cells": 80},
        },
    }})
    summary = fake.run.summary
    assert summary["outcome"] == "completed"
    assert summary["update"] == 240
    assert summary["evaluation/final/by_state/query/encoding_mse"] == 0.2
    assert summary["evaluation/final/by_type/numeric/query/numeric_mse"] == 3.0
    assert summary["evaluation/final/by_source/scm_mixed_v1/query/count"] == 4
    assert not any("by_table" in name for name in summary)
    assert "private" not in json.dumps(summary)


def test_per_table_metrics_require_opt_in_and_safe_public_name(enabled, monkeypatch):
    monkeypatch.setenv("TABU_LAB_WANDB_TABLE_METRICS", "1")
    fake = FakeSDK()
    observer = create_restoration_observer({}, {}, wandb=fake)
    observer({"event": "summary", "metrics": {
        "by_table": {"scm_mixed_v1_000": {"query": {"count": 4}},
                     "/private/key": {"query": {"count": 8}}},
    }})
    assert fake.run.logs[-1] == {"evaluation/by_table/scm_mixed_v1_000/query/count": 4}


def test_partial_evaluation_is_distinct_from_complete_evaluation(enabled, monkeypatch):
    monkeypatch.setenv("WANDB_RUN_GROUP", "small128-budget")
    fake = FakeSDK()
    observer = create_restoration_observer(
        {"evaluate_every_rounds": 10, "final_reserve_seconds": 300}, {}, wandb=fake,
    )
    observer({"event": "phase", "stage": "final", "metrics": {
        "complete": False, "completed_episodes": 40, "expected_episodes": 960,
        "stop_reason": "wall_limit", "loss": 2.0,
    }})
    assert fake.calls[0]["group"] == "small128-budget"
    assert fake.calls[0]["config"] == {
        "evaluate_every_rounds": 10, "final_reserve_seconds": 300,
    }
    assert fake.run.logs[-1] == {
        "phase": "final", "evaluation/complete": False,
        "evaluation/completed_episodes": 40, "evaluation/expected_episodes": 960,
        "evaluation/stop_reason": "wall_limit", "evaluation/loss": 2.0,
    }


@pytest.mark.parametrize("stage", ["round-0032", "initial_complete", "round-0032_complete",
                                   "final_complete"])
def test_runner_stage_names_are_mirrored(enabled, stage):
    fake = FakeSDK()
    observer = create_restoration_observer(wandb=fake)
    observer({"event": "phase", "stage": "start", "model_parameters": 123})
    observer({"event": "phase", "stage": stage})
    assert fake.calls[0]["config"]["model_parameters"] == 123
    assert fake.run.logs[-1]["phase"] == stage


def test_tracker_failure_is_isolated_and_redacted_and_close_still_finishes(enabled):
    fake = FakeSDK()
    observer = create_restoration_observer({}, {}, wandb=fake)
    fake.run.raise_on_log = True
    with pytest.warns(UserWarning, match="local receipts continue") as caught:
        observer({"event": "update", "loss": 1.0})
    assert "secret" not in str(caught[0].message)
    assert observer.error_type == "RuntimeError" and not observer.active
    observer({"event": "update", "loss": 0.5})
    observer.close()
    assert fake.run.finished == 1


def test_init_failure_is_isolated_and_resuming_uses_same_run_id(enabled):
    class BrokenSDK(FakeSDK):
        def init(self, **kwargs):
            raise RuntimeError("private authentication failure")

    with pytest.warns(UserWarning, match="RuntimeError") as caught:
        observer = create_restoration_observer({}, {}, wandb=BrokenSDK(), run_id="same")
    assert not observer.active
    assert "private" not in str(caught[0].message)
    fake = FakeSDK()
    first = create_restoration_observer({}, {}, wandb=fake, run_id="same")
    first.close()
    second = create_restoration_observer({}, {}, wandb=fake, run_id="same")
    second({"event": "update", "update": 5, "loss": 1.0})
    assert [call["id"] for call in fake.calls] == ["same", "same"]
    assert [call["resume"] for call in fake.calls] == ["allow", "allow"]
    # The model update is a custom axis, not a W&B history step that may rewind.
    assert fake.run.logs[-1]["update"] == 5
