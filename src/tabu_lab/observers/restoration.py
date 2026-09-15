"""Optional, allow-listed W&B mirror for restoration joint-fit events.

Local receipts remain authoritative. The observer accepts SDK-managed auth
(including netrc), never uploads inputs/checkpoints, and isolates tracker errors
from training. Set WANDB_RUN_ID to reuse the same mirror across continuation.
"""

from __future__ import annotations

import math
import os
import re
import uuid
import warnings
from collections.abc import Mapping
from typing import Any

from tabu_lab.observers import (
    _host_disclosure_opted_in,
    _wandb_mode,
    _wandb_mode_requires_host_disclosure_opt_in,
    _warn_host_disclosure_not_allowed,
)

_NUMERIC_CONFIG = frozenset({
    "table_count", "query_count", "evaluation_masks", "max_rounds", "max_seconds",
    "evaluate_every_rounds", "checkpoint_every_round", "final_reserve_seconds",
    "model_parameters", "bandwidth", "ridge", "width", "mlp_hidden", "epsilon",
    "layers", "heads", "ff_width", "slots", "tau_presence", "reference_mass", "norm_eps",
    "learning_rate", "weight_decay", "eps", "grad_clip", "model", "order", "masks", "codes",
})
_CONFIG_GROUPS = frozenset({"model", "encoder", "backbone", "optimizer", "seeds"})
_CONFIG_ENUMS = {
    "readout": {"ll", "nw"}, "kind": {"direct", "inducing", "adamw"},
    "category_map": {"identity128", "rotary32", "mlp32", "mlp256"},
}
_IDENTITY_ENUMS = {"device": {"cpu", "cuda:0"}, "dtype": {"float64", "float32"}}
_HASH_KEYS = (
    "preregistration_sha256", "corpus_preregistration_sha256", "corpus_manifest_sha256",
)
_COUNTERS = (
    "update", "round", "elapsed_seconds", "completed_episodes", "total_episodes",
    "model_parameters", "cursor", "peak_allocated_bytes", "training_round",
)
_TRAIN_METRICS = (
    "loss", "gradient_norm", "update_seconds", "train_seconds", "learning_rate",
    "peak_allocated_bytes", "checkpoint_seconds", "preparation_seconds",
)
_STATE_METRICS = (
    "count", "encoding_mse", "numeric_mse", "discrete_accuracy", "numeric_count",
    "discrete_count",
)
_COVERAGE_METRICS = (
    "query_cells", "total_cells", "protected_discrete_cells", "unmaskable_discrete_classes",
    "singleton_discrete_classes", "query_fraction",
)
_SOURCES = frozenset({"scm_mixed_v1", "discoscm", "scm_numeric_v0", "sklearn_synthetic"})
_STATUSES = frozenset({
    "started", "completed", "segment_completed", "wall_limit", "interrupted", "failed",
    "budget_exhausted", "local_unissued",
})


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _config(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if key in _CONFIG_GROUPS and isinstance(value, Mapping):
            result.update(_config(value, f"{name}/"))
        elif (key in _NUMERIC_CONFIG and _finite(value)) or (
            key in _CONFIG_ENUMS and isinstance(value, str) and value in _CONFIG_ENUMS[key]
        ):
            result[name] = value
        elif (key == "betas" and isinstance(value, list | tuple) and len(value) == 2
              and all(_finite(item) for item in value)):
            result[name] = list(value)
    return result


def _metadata(config: Mapping, identity: Mapping) -> dict[str, Any]:
    result = _config(config)
    for key, allowed in _IDENTITY_ENUMS.items():
        if isinstance(identity.get(key), str) and identity[key] in allowed:
            result[key] = identity[key]
    for key in _HASH_KEYS:
        value = identity.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            result[key] = value
    source = identity.get("source", {})
    if isinstance(source, Mapping):
        digest = source.get("sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            result["source_sha256"] = digest
    return result


def _copy_numeric(result: dict, payload: Mapping, names, prefix="") -> None:
    for name in names:
        value = payload.get(name)
        if _finite(value):
            result[f"{prefix}{name}"] = value


def _evaluation(payload: Mapping, prefix: str, *, table_metrics: bool) -> dict:
    result: dict[str, Any] = {}
    _copy_numeric(result, payload, ("loss", "completed_episodes", "expected_episodes"), prefix)
    if type(payload.get("complete")) is bool:
        result[f"{prefix}complete"] = payload["complete"]
    if payload.get("stop_reason") == "wall_limit":
        result[f"{prefix}stop_reason"] = "wall_limit"
    groups = {"by_state": payload.get("by_state", {})}
    for key, allowed in (("by_type", {"numeric", "nominal", "ordinal"}),
                         ("by_source", _SOURCES)):
        values = payload.get(key, {})
        if isinstance(values, Mapping):
            groups.update({f"{key}/{name}": item for name, item in values.items()
                           if name in allowed})
    tables = payload.get("by_table", {})
    if table_metrics and isinstance(tables, Mapping):
        groups.update({f"by_table/{name}": item for name, item in tables.items()
                       if isinstance(name, str) and re.fullmatch(
                           r"(?:scm_mixed_v1|discoscm|scm_numeric_v0|sklearn_synthetic)_\d{3}",
                           name)})
    for name, states in groups.items():
        if not isinstance(states, Mapping):
            continue
        for state in ("retained", "query"):
            item = states.get(state, {})
            if isinstance(item, Mapping):
                _copy_numeric(result, item, _STATE_METRICS, f"{prefix}{name}/{state}/")
    coverage = payload.get("coverage", {})
    if isinstance(coverage, Mapping):
        _copy_numeric(result, coverage, _COVERAGE_METRICS, f"{prefix}coverage/")
    return result


class RestorationObserver:
    """Callable event sink; backend failure disables only this mirror."""

    def __init__(self, *, wandb=None, config=None, identity=None, run_id=None):
        self._wandb = wandb
        self._run = None
        self._closed = False
        self.error_type: str | None = None
        self.run_id = run_id or os.environ.get("WANDB_RUN_ID") or uuid.uuid4().hex[:12]
        self._table_metrics = os.environ.get("TABU_LAB_WANDB_TABLE_METRICS") == "1"
        if wandb is not None and (config is not None or identity is not None):
            self._start(config or {}, identity or {})

    @property
    def active(self) -> bool:
        return self._run is not None and not self._closed and self.error_type is None

    @property
    def run_url(self) -> str | None:
        return getattr(self._run, "url", None)

    def _fail(self, error: Exception) -> None:
        # Exception messages may contain SDK credentials or filesystem paths.
        self.error_type = type(error).__name__
        warnings.warn("Restoration W&B mirror unavailable; local receipts continue "
                      f"({self.error_type}).", stacklevel=3)

    def _start(self, config: Mapping, identity: Mapping) -> None:
        try:
            settings = self._wandb.Settings(
                disable_git=True, disable_code=True, console="off", silent=True,
                x_disable_stats=True, x_disable_meta=True,
            )
            kwargs: dict[str, Any] = {
                "project": os.environ.get("WANDB_PROJECT")
                or os.environ.get("TABU_LAB_WANDB_PROJECT", "restoration"),
                "id": self.run_id, "resume": "allow", "config": _metadata(config, identity),
                "settings": settings,
            }
            for key in ("entity", "name", "group", "mode"):
                value = os.environ.get(f"WANDB_{key.upper()}")
                if key == "group":
                    value = os.environ.get("WANDB_RUN_GROUP") or value
                if value:
                    kwargs[key] = value
            self._run = self._wandb.init(**kwargs)
            if self._run is None:
                raise RuntimeError("W&B init returned no run")
            self._run.define_metric("update")
            for pattern in ("train/*", "evaluation/*", "progress/*"):
                self._run.define_metric(pattern, step_metric="update")
        except Exception as error:
            self._fail(error)

    def __call__(self, event: Mapping[str, Any]) -> None:
        self.emit(event)

    def emit(self, event: Mapping[str, Any]) -> None:
        if self._wandb is None or self._closed or self.error_type is not None:
            return
        try:
            kind = event.get("event")
            if kind not in {"phase", "evaluation_progress", "update", "summary"}:
                return
            if self._run is None:
                config = dict(event.get("config", {}))
                if _finite(event.get("model_parameters")):
                    config["model_parameters"] = event["model_parameters"]
                self._start(config, event.get("identity", {}))
            if not self.active:
                return
            result: dict[str, Any] = {}
            _copy_numeric(result, event, _COUNTERS)
            stage = event.get("stage")
            if isinstance(stage, str) and re.fullmatch(
                r"(?:start|initial|final|train|training|round[-_]\d+|completed|interrupted)"
                r"(?:_complete)?", stage
            ):
                result["phase"] = stage
            if kind == "update":
                _copy_numeric(result, event, _TRAIN_METRICS, "train/")
                mask = event.get("mask", {})
                if isinstance(mask, Mapping):
                    _copy_numeric(result, mask, (*_COVERAGE_METRICS, "query_coverage"),
                                  "train/coverage/")
            metrics = event.get("metrics", {})
            if isinstance(metrics, Mapping):
                result.update(_evaluation(metrics, "evaluation/",
                                          table_metrics=self._table_metrics))
                for phase in ("initial", "final"):
                    values = metrics.get(phase, {})
                    if isinstance(values, Mapping):
                        result.update(_evaluation(values, f"evaluation/{phase}/",
                                                  table_metrics=self._table_metrics))
                if kind == "summary":
                    _copy_numeric(result, metrics, _COUNTERS)
                    if type(metrics.get("budget_exhausted")) is bool:
                        result["budget_exhausted"] = metrics["budget_exhausted"]
                    for key in ("outcome", "status"):
                        value = metrics.get(key)
                        if isinstance(value, str) and value in _STATUSES:
                            result[key] = value
            if result:
                self._run.log(result)
                if kind == "summary":
                    self._run.summary.update(result)
        except Exception as error:
            self._fail(error)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._run is not None:
                self._run.finish()
        except Exception as error:
            self._fail(error)
        finally:
            self._closed = True


def create_restoration_observer(config=None, identity=None, *, run_id=None, wandb=None):
    """Return a default-off event sink, optionally initialized with public metadata.

    When metadata is omitted initialization waits for the runner's start event.
    SDK authentication is used as-is: this function never prompts for a key or
    calls login. Launchers should check ``active`` after initialization when live
    observation is a requirement for launching their experiment.
    """
    if os.environ.get("TABU_LAB_OBSERVER", "").strip().lower() != "wandb":
        return RestorationObserver(run_id=run_id)
    mode = _wandb_mode()
    if _wandb_mode_requires_host_disclosure_opt_in(mode) and not _host_disclosure_opted_in():
        _warn_host_disclosure_not_allowed(mode)
        return RestorationObserver(run_id=run_id)
    if wandb is None:
        try:
            import wandb
        except Exception as error:
            warnings.warn("Restoration W&B mirror unavailable: optional SDK import failed "
                          f"({type(error).__name__}).",
                          stacklevel=2)
            return RestorationObserver(run_id=run_id)
    return RestorationObserver(wandb=wandb, config=config, identity=identity, run_id=run_id)


__all__ = ["RestorationObserver", "create_restoration_observer"]
