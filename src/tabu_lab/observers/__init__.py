"""Optional observation adapter for TAR local fitting.

The existing program-training telemetry remains separate and unchanged. This
adapter keeps the established TAR observer protocol and explicit opt-ins. It
emits allow-listed metrics and coarse metadata, degrades to a null observer on
tracker failure, and never changes receipts or model maturity. The W&B backend
may collect host metadata independently of the allow-listed payload.
"""

from __future__ import annotations

import abc
import os
import warnings
from collections.abc import Mapping
from typing import Any

_COARSE_ENV_KEYS = (
    "host_class",
    "os_family",
    "architecture",
    "python_version",
    "torch_version",
    "device",
    "accelerator",
    "cuda_version",
    "cudnn_version",
    "deterministic_algorithms",
    "dependency_lock_hash",
    "container_image_hash",
)
_HOST_DISCLOSURE_OPT_IN = "TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE"
_LOCAL_ONLY_WANDB_MODES = frozenset({"disabled", "dryrun", "offline"})


def _wandb_mode() -> str:
    """Return the normalized W&B mode; an unset mode is hosted/online."""

    return os.environ.get("WANDB_MODE", "").strip().lower()


def _wandb_mode_requires_host_disclosure_opt_in(mode: str) -> bool:
    """Whether ``mode`` can contact a backend that independently records host."""

    return mode not in _LOCAL_ONLY_WANDB_MODES


def _host_disclosure_opted_in() -> bool:
    """Require the exact documented value; loose truthy parsing is intentionally avoided."""

    return os.environ.get(_HOST_DISCLOSURE_OPT_IN) == "1"


def _warn_host_disclosure_not_allowed(mode: str) -> None:
    mode_label = mode or "online (default)"
    warnings.warn(
        "TabU-lab observer: W&B mode "
        f"{mode_label!r} may independently collect the machine hostname; set "
        f"{_HOST_DISCLOSURE_OPT_IN}=1 only after accepting that disclosure risk. "
        "Observation disabled.",
        stacklevel=3,
    )


def _coarse_observer_config(environment_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build a W&B ``config`` from an allow-list of coarse environment fields only.

    Never forwards hostname, username, private paths, tokens, or any key outside
    the allow-list, even if the payload contained them.
    """

    config: dict[str, Any] = {}
    for key in _COARSE_ENV_KEYS:
        if key in environment_payload:
            value = environment_payload[key]
            if value is None:
                continue
            config[key] = value
    return config


class Observer(abc.ABC):
    """A derived mirror of a single seed attempt's metrics."""

    @property
    def run_url(self) -> str | None:
        return None

    @abc.abstractmethod
    def log_step(self, record: Mapping[str, Any]) -> None: ...

    @abc.abstractmethod
    def log_summary(self, summary: Mapping[str, Any]) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...


class NullObserver(Observer):
    """Default observer: disabled, zero side effects, never raises."""

    def log_step(self, record: Mapping[str, Any]) -> None:
        return None

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class WandbObserver(Observer):
    """Optional W&B mirror. Inject ``wandb`` so it is testable without the dep.

    All public methods are fail-closed: any exception disables further logging
    and is swallowed, never propagated into the run.
    """

    def __init__(
        self,
        *,
        wandb: Any,
        run_id: str,
        attempt_id: str,
        experiment_id: str,
        contract_id: str,
        seed: int,
        stage: str,
        environment_payload: Mapping[str, Any],
    ) -> None:
        self._wandb = wandb
        self._alive = False
        mode = _wandb_mode()
        if _wandb_mode_requires_host_disclosure_opt_in(mode) and not (_host_disclosure_opted_in()):
            _warn_host_disclosure_not_allowed(mode)
            return
        tags = [
            f"run:{run_id}",
            f"attempt:{attempt_id}",
            f"experiment:{experiment_id}",
            f"contract:{contract_id}",
            f"seed:{seed}",
            f"stage:{stage}",
        ]
        config = _coarse_observer_config(environment_payload)
        project = os.environ.get("TABU_LAB_WANDB_PROJECT", "tabu-lab")
        settings = wandb.Settings(disable_git=True, silent=True)
        os.environ.setdefault("WANDB_DISABLE_SYSTEM_LOGGING", "true")
        init_kwargs: dict[str, Any] = {
            "project": project,
            "name": attempt_id,
            "group": experiment_id,
            "tags": tags,
            "config": config,
            "settings": settings,
        }
        mode = os.environ.get("WANDB_MODE")
        if mode:
            init_kwargs["mode"] = mode
        try:
            self._run = wandb.init(**init_kwargs)
            self._alive = True
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(
                f"TabU-lab observer: W&B init failed; observation disabled ({exc!r})",
                stacklevel=2,
            )
            self._alive = False

    @property
    def run_url(self) -> str | None:
        return getattr(getattr(self, "_run", None), "url", None)

    def log_step(self, record: Mapping[str, Any]) -> None:
        if not self._alive:
            return
        try:
            metrics: dict[str, Any] = {
                "step": record.get("step"),
                "loss": record.get("loss"),
                "gradient_norm": record.get("gradient_norm"),
                "elapsed_seconds": record.get("elapsed_seconds"),
            }
            for key in (
                "learning_rate",
                "episode_index",
                "context_rows",
                "query_rows",
                "unique_codebooks",
                "train_seconds",
                "updates_per_second",
                "fit_loss",
                "fit_accuracy",
                "fit_nll",
                "fit_normalized_mse",
                "fit_support_max_weight_mean",
                "fit_support_entropy_mean",
            ):
                value = record.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[key] = value
            for group, norm in (record.get("gradient_norms") or {}).items():
                metrics[f"gradient_norm/{group}"] = norm
            self._wandb.log(metrics)
        except Exception:  # pragma: no cover - defensive
            self._alive = False

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        if not self._alive:
            return
        try:
            metrics = {
                "summary/initial_objective": summary.get("initial_objective"),
                "summary/final_objective": summary.get("final_objective"),
                "summary/loss_ratio": summary.get("loss_ratio"),
                "summary/steps": summary.get("steps"),
                "summary/verdict": str(summary.get("verdict")),
            }
            for phase in ("initial_fit", "final_fit", "initial_holdout", "final_holdout"):
                for key in ("loss", "mse", "rmse", "normalized_mse", "nll", "accuracy"):
                    value = (summary.get(phase) or {}).get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metrics[f"evaluation/{phase}/{key}"] = value
            self._wandb.log(metrics)
        except Exception:  # pragma: no cover - defensive
            self._alive = False

    def close(self) -> None:
        if not self._alive:
            return
        try:
            self._wandb.finish()
        except Exception:  # pragma: no cover - defensive
            pass
        finally:
            self._alive = False


def get_observer(
    *,
    run_id: str,
    attempt_id: str,
    experiment_id: str,
    contract_id: str,
    seed: int,
    stage: str,
    environment_payload: Mapping[str, Any],
) -> Observer:
    """Return the observer selected by ``TABU_LAB_OBSERVER`` (default: disabled).

    Returns a :class:`NullObserver` unless the variable is exactly ``wandb`` and
    the ``wandb`` package and ``WANDB_API_KEY`` are both available. Hosted or
    online modes additionally require
    ``TABU_LAB_ALLOW_WANDB_HOST_DISCLOSURE=1`` because the W&B backend may
    collect host metadata independently of the allow-listed TabU payload. Any
    missing prerequisite degrades to the null observer with a warning; the run
    is never blocked.
    """

    kind = os.environ.get("TABU_LAB_OBSERVER", "").strip().lower()
    if kind != "wandb":
        return NullObserver()
    try:
        import wandb  # lazy import, optional dependency
    except Exception:
        warnings.warn(
            "TABU_LAB_OBSERVER=wandb but 'wandb' is not installed; observation disabled.",
            stacklevel=2,
        )
        return NullObserver()
    if not os.environ.get("WANDB_API_KEY"):
        warnings.warn(
            "TABU_LAB_OBSERVER=wandb but WANDB_API_KEY is unset; observation disabled.",
            stacklevel=2,
        )
        return NullObserver()
    mode = _wandb_mode()
    if _wandb_mode_requires_host_disclosure_opt_in(mode) and not (_host_disclosure_opted_in()):
        _warn_host_disclosure_not_allowed(mode)
        return NullObserver()
    return WandbObserver(
        wandb=wandb,
        run_id=run_id,
        attempt_id=attempt_id,
        experiment_id=experiment_id,
        contract_id=contract_id,
        seed=seed,
        stage=stage,
        environment_payload=environment_payload,
    )


__all__ = [
    "NullObserver",
    "Observer",
    "WandbObserver",
    "get_observer",
]
