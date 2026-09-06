"""Tests for the optional observation mirrors (Phase 1B).

These run without the ``wandb`` dependency installed: the ``WandbObserver`` is
exercised against an injected fake ``wandb`` module, and the default
``NullObserver`` / factory degrade paths are covered directly.
"""

from __future__ import annotations

import pytest

from tabu_lab.observers import (
    NullObserver,
    WandbObserver,
    _coarse_observer_config,
    get_observer,
)


def test_null_observer_is_a_noop():
    observer = NullObserver()
    # None of these may raise, even with junk payloads.
    observer.log_step({"step": 0, "loss": 1.0})
    observer.log_summary({"loss_ratio": 0.9, "verdict": "pass"})
    observer.close()


def test_coarse_config_allowlist_strips_forbidden_fields():
    payload = {
        "host_class": "mps-host",
        "torch_version": "2.5.0",
        "hostname": "secret-host",
        "username": "nobody",
        "home_directory": "/Users/nobody",
        "WANDB_API_KEY": "sk-secret",
    }
    config = _coarse_observer_config(payload)
    assert config == {"host_class": "mps-host", "torch_version": "2.5.0"}
    assert "hostname" not in config
    assert "username" not in config
    assert "home_directory" not in config
    assert "WANDB_API_KEY" not in config


def test_get_observer_defaults_to_null(monkeypatch):
    monkeypatch.delenv("TABU_LAB_OBSERVER", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    observer = get_observer(
        run_id="run-1",
        attempt_id="attempt-1",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )
    assert isinstance(observer, NullObserver)


def test_get_observer_requests_wandb_but_missing_degrades_to_null(monkeypatch):
    monkeypatch.setenv("TABU_LAB_OBSERVER", "wandb")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    # Ensure the real wandb is unavailable so we exercise the missing-dep path.
    monkeypatch.setitem(__import__("sys").modules, "wandb", None)
    observer = get_observer(
        run_id="run-1",
        attempt_id="attempt-1",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )
    assert isinstance(observer, NullObserver)


class _FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.log_calls = []
        self.finished = False
        self.raise_on_log = False

    class Settings:
        def __init__(self, disable_git=True, silent=True):
            self.disable_git = disable_git
            self.silent = silent

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return object()

    def log(self, metrics):
        if self.raise_on_log:
            raise RuntimeError("boom")
        self.log_calls.append(metrics)

    def finish(self):
        self.finished = True


@pytest.mark.parametrize("mode", [None, "online"])
def test_get_observer_hosted_mode_requires_explicit_host_disclosure_opt_in(
    monkeypatch, mode
):
    fake = _FakeWandb()
    monkeypatch.setenv("TABU_LAB_OBSERVER", "wandb")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-key")
    monkeypatch.delenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", raising=False)
    if mode is None:
        monkeypatch.delenv("WANDB_MODE", raising=False)
    else:
        monkeypatch.setenv("WANDB_MODE", mode)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    with pytest.warns(UserWarning, match="TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1"):
        observer = get_observer(
            run_id="run-1",
            attempt_id="attempt-1",
            experiment_id="exp-1",
            contract_id="tabuf",
            seed=0,
            stage="F0",
            environment_payload={"host_class": "cpu-host"},
        )

    assert isinstance(observer, NullObserver)
    assert fake.init_calls == []


def test_get_observer_hosted_mode_accepts_exact_host_disclosure_opt_in(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setenv("TABU_LAB_OBSERVER", "wandb")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-key")
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", "1")
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    observer = get_observer(
        run_id="run-1",
        attempt_id="attempt-1",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )

    assert isinstance(observer, WandbObserver)
    assert observer._alive is True
    assert fake.init_calls[0]["mode"] == "online"
    observer.close()


def test_get_observer_offline_mode_does_not_require_host_disclosure_opt_in(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setenv("TABU_LAB_OBSERVER", "wandb")
    monkeypatch.setenv("WANDB_API_KEY", "test-only-key")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake)

    observer = get_observer(
        run_id="run-1",
        attempt_id="attempt-1",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )

    assert isinstance(observer, WandbObserver)
    assert observer._alive is True
    assert fake.init_calls[0]["mode"] == "offline"
    observer.close()


def test_direct_wandb_observer_hosted_mode_is_fail_closed(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", raising=False)

    with pytest.warns(UserWarning, match="TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1"):
        observer = WandbObserver(
            wandb=fake,
            run_id="run-1",
            attempt_id="attempt-9",
            experiment_id="exp-1",
            contract_id="tabuf",
            seed=0,
            stage="F0",
            environment_payload={"host_class": "cpu-host"},
        )

    assert observer._alive is False
    assert fake.init_calls == []


def test_wandb_observer_forwards_only_coarse_config_and_typed_ids(monkeypatch):
    monkeypatch.setenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", "1")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    fake = _FakeWandb()
    observer = WandbObserver(
        wandb=fake,
        run_id="run-1",
        attempt_id="attempt-9",
        experiment_id="exp-1",
        contract_id="tabu.unit_row",
        seed=3,
        stage="F0",
        environment_payload={
            "host_class": "mps-host",
            "torch_version": "2.5.0",
            "hostname": "should-not-leak",
        },
    )
    assert observer._alive is True
    call = fake.init_calls[0]
    assert call["project"] == "tabu-lab"
    assert call["name"] == "attempt-9"
    assert call["tags"] == [
        "run:run-1",
        "attempt:attempt-9",
        "experiment:exp-1",
        "contract:tabu.unit_row",
        "seed:3",
        "stage:F0",
    ]
    assert call["config"] == {"host_class": "mps-host", "torch_version": "2.5.0"}
    assert call["settings"].disable_git is True
    assert call["settings"].silent is True
    observer.close()
    assert fake.finished is True


def test_wandb_observer_flattens_gradient_norms_and_swallows_errors(monkeypatch):
    monkeypatch.setenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", "1")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    fake = _FakeWandb()
    observer = WandbObserver(
        wandb=fake,
        run_id="run-1",
        attempt_id="attempt-9",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )
    observer.log_step(
        {
            "step": 10,
            "loss": 0.5,
            "gradient_norm": 1.2,
            "elapsed_seconds": 3.0,
            "gradient_norms": {"row": 0.9, "feature": 0.4},
        }
    )
    assert fake.log_calls[-1] == {
        "step": 10,
        "loss": 0.5,
        "gradient_norm": 1.2,
        "elapsed_seconds": 3.0,
        "gradient_norm/row": 0.9,
        "gradient_norm/feature": 0.4,
    }
    # A broken log must disable further logging without raising.
    fake.raise_on_log = True
    observer.log_step({"step": 20, "loss": 0.4})
    assert observer._alive is False
    observer.close()  # must not raise


def test_wandb_observer_init_failure_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE", "1")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _BrokenWandb:
        class Settings:
            def __init__(self, **_kw):
                pass

        def init(self, **_kw):
            raise RuntimeError("cannot reach server")

    observer = WandbObserver(
        wandb=_BrokenWandb(),
        run_id="run-1",
        attempt_id="attempt-9",
        experiment_id="exp-1",
        contract_id="tabuf",
        seed=0,
        stage="F0",
        environment_payload={"host_class": "cpu-host"},
    )
    assert observer._alive is False
    # Log/close must be safe no-ops after a failed init.
    observer.log_step({"step": 1, "loss": 0.5})
    observer.close()
