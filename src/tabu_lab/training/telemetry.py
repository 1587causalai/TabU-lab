"""Truth-safe training telemetry with an optional passive W&B mirror.

Telemetry is deliberately outside :class:`ProgramSnapshot`: changing a chart,
backend, or logging cadence cannot change model/data/training semantics.  The
local JSONL stream is authoritative for the observer; W&B is only a projection.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from tabu_lab.contracts import EvidenceEpisode, PredictionBundle, canonical_hash, canonical_json

from .trainer import TrainStep

TelemetryMode = Literal["disabled", "local", "wandb"]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TelemetryProtocol:
    protocol_id: str
    version: str
    step_interval: int
    metric_groups: tuple[str, ...]
    protocol_hash: str

    @property
    def ref(self) -> str:
        return f"{self.protocol_id}@{self.version}"


def load_telemetry_protocol(path: str | Path) -> TelemetryProtocol:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("telemetry protocol root must be a mapping")
    expected = {
        "schema_version",
        "protocol_id",
        "version",
        "step_interval",
        "metric_groups",
        "description",
    }
    if set(payload) != expected:
        raise ValueError("telemetry protocol has missing or unknown fields")
    if payload["schema_version"] != "tabu.telemetry-protocol.v1":
        raise ValueError("unsupported telemetry protocol schema")
    protocol_id = payload["protocol_id"]
    version = payload["version"]
    step_interval = payload["step_interval"]
    groups = payload["metric_groups"]
    if not isinstance(protocol_id, str) or not protocol_id.strip():
        raise ValueError("telemetry protocol_id must be non-empty")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("telemetry version must be non-empty")
    if type(step_interval) is not int or step_interval <= 0:
        raise ValueError("telemetry step_interval must be a positive integer")
    if (
        not isinstance(groups, list)
        or not groups
        or any(not isinstance(group, str) or not group.strip() for group in groups)
        or len(groups) != len(set(groups))
    ):
        raise ValueError("telemetry metric_groups must be unique non-empty strings")
    semantic_payload = dict(payload)
    semantic_payload.pop("description")
    return TelemetryProtocol(
        protocol_id=protocol_id,
        version=version,
        step_interval=step_interval,
        metric_groups=tuple(groups),
        protocol_hash=canonical_hash(semantic_payload),
    )


def _finite_scalar(value: Any) -> float | None:
    try:
        scalar = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _masked_mean(value: Any, mask: Any) -> float | None:
    if value is None or mask is None:
        return None
    selected = value.detach()[mask.detach().to(device=value.device, dtype=bool)]
    if selected.numel() == 0:
        return None
    return _finite_scalar(selected.float().mean())


def collect_training_metrics(
    step: TrainStep,
    evidence: EvidenceEpisode,
    *,
    generator_ref: str,
    learning_rate: float,
) -> dict[str, float | int | str]:
    """Project truth-free inputs and loss-boundary scalars into one record."""

    prediction: PredictionBundle = step.prediction
    metrics: dict[str, float | int | str] = {
        "step": step.step,
        "data/generator_ref": generator_ref,
        "data/dataset_id": evidence.dataset_id,
        "data/rows": len(evidence.row_ids),
        "data/features": len(evidence.feature_names),
        "data/targets": int(evidence.target_mask.sum().item()),
        "optim/learning_rate": float(learning_rate),
        "optim/gradient_norm": float(step.gradient_norm),
    }
    for name, value in step.gradient_norms.items():
        metrics[f"optim/gradient_norm/{name}"] = float(value)
        metrics[f"optim/zero_gradient/{name}"] = float(value == 0.0)
    total = _finite_scalar(step.loss.total)
    if total is not None:
        metrics["loss/total"] = total
    for name, value in step.loss.components.items():
        scalar = _finite_scalar(value)
        if scalar is not None:
            metrics[f"loss/{name}"] = scalar
    for name, value in step.loss.counts.items():
        metrics[f"coverage/{name}"] = value

    target_mask = prediction.auxiliaries.get("target_mask")
    support = prediction.auxiliaries.get("support_available")
    if target_mask is not None and support is not None:
        target_count = int(target_mask.sum().item())
        supported = int((target_mask & support).sum().item())
        metrics["support/coverage"] = supported / max(target_count, 1)
        metrics["support/abstention_rate"] = (target_count - supported) / max(
            target_count, 1
        )
    for source, target in (
        ("routing_effective_support_size", "support/effective_size"),
        ("routing_entropy", "support/routing_entropy"),
        ("routing_max_weight", "support/top1_mass"),
    ):
        value = prediction.auxiliaries.get(source)
        mean = _masked_mean(value, target_mask)
        if mean is not None:
            metrics[target] = mean

    metadata: Mapping[str, Any] = evidence.metadata
    response_family = metadata.get("response_value_family", "numeric")
    metrics["data/response_family"] = str(response_family)
    metrics["data/is_numeric_response"] = float(response_family == "numeric")
    metrics["data/is_categorical_response"] = float(response_family == "categorical")
    for source, target in (
        ("response_class_count", "data/response_class_count"),
        ("width", "data/world_width"),
        ("context_rows", "data/context_rows"),
        ("query_rows", "data/query_rows"),
        ("routing_pairs", "data/routing_pairs"),
    ):
        value = metadata.get(source)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            metrics[target] = float(value)
    for source in ("family", "predictor_regime", "noise_level", "scale_band"):
        value = metadata.get(source)
        if isinstance(value, str):
            metrics[f"data/{source}"] = value
    return metrics


@dataclass(frozen=True, slots=True)
class TelemetryResult:
    metrics_path: Path
    receipt_path: Path
    status: str
    wandb_run_id: str | None
    wandb_url: str | None


class TrainingTelemetry:
    """Non-authoritative observer that never controls the training loop."""

    def __init__(
        self,
        *,
        output_root: str | Path,
        protocol: TelemetryProtocol,
        mode: TelemetryMode,
        run_identity_hash: str,
        snapshot_hash: str,
        program_ref: str,
        wandb_project: str = "tabu-pretraining",
        wandb_entity: str | None = None,
        wandb_run_name: str | None = None,
    ) -> None:
        if mode not in {"disabled", "local", "wandb"}:
            raise ValueError("unsupported telemetry mode")
        if mode == "disabled":
            raise ValueError("disabled telemetry must not construct an observer")
        self.output_root = Path(output_root)
        self.protocol = protocol
        self.mode = mode
        self.run_identity_hash = run_identity_hash
        self.snapshot_hash = snapshot_hash
        self.program_ref = program_ref
        self.metrics_path = self.output_root / "training-metrics.jsonl"
        self.receipt_path = self.output_root / "telemetry-receipt.json"
        self._stream = self.metrics_path.open("x", encoding="utf-8")
        self._wandb: Any = None
        self._wandb_run: Any = None
        self._wandb_error: str | None = None
        self._logged_steps = 0
        if mode == "wandb":
            try:
                import wandb

                self._wandb = wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    id=run_identity_hash[:32],
                    resume="allow",
                    config={
                        "program_ref": program_ref,
                        "snapshot_hash": snapshot_hash,
                        "run_identity_hash": run_identity_hash,
                        "telemetry_protocol_ref": protocol.ref,
                        "telemetry_protocol_hash": protocol.protocol_hash,
                    },
                    reinit=True,
                )
            except Exception as exc:  # pragma: no cover - external observer boundary
                self._wandb_error = f"{type(exc).__name__}: {exc}"
                self._wandb = None
                self._wandb_run = None

    def log(self, metrics: Mapping[str, float | int | str]) -> None:
        step = int(metrics["step"])
        if step % self.protocol.step_interval != 0:
            return
        payload = {
            "schema_version": "tabu.training-metric-record.v1",
            "protocol_ref": self.protocol.ref,
            "run_identity_hash": self.run_identity_hash,
            **dict(metrics),
        }
        self._stream.write(canonical_json(payload) + "\n")
        self._stream.flush()
        self._logged_steps += 1
        if self._wandb_run is not None:
            try:
                wandb_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if key != "step" and isinstance(value, int | float)
                }
                self._wandb_run.log(wandb_metrics, step=step)
            except Exception as exc:  # pragma: no cover - external observer boundary
                self._wandb_error = f"{type(exc).__name__}: {exc}"
                self._wandb_run = None

    def close(self) -> TelemetryResult:
        wandb_run_id = None
        wandb_url = None
        if self._wandb_run is not None:
            wandb_run_id = str(getattr(self._wandb_run, "id", "")) or None
            wandb_url = str(getattr(self._wandb_run, "url", "")) or None
            try:
                self._wandb_run.finish()
            except Exception as exc:  # pragma: no cover - external observer boundary
                self._wandb_error = f"{type(exc).__name__}: {exc}"
        self._stream.close()
        status = "ok" if self._wandb_error is None else "local_only_degraded"
        receipt = {
            "schema_version": "tabu.telemetry-receipt.v1",
            "observer_semantics": "non_authoritative_projection",
            "status": status,
            "mode": self.mode,
            "program_ref": self.program_ref,
            "snapshot_hash": self.snapshot_hash,
            "run_identity_hash": self.run_identity_hash,
            "protocol_ref": self.protocol.ref,
            "protocol_hash": self.protocol.protocol_hash,
            "logged_steps": self._logged_steps,
            "metrics_file": self.metrics_path.name,
            "metrics_sha256": _file_sha256(self.metrics_path),
            "wandb_run_id": wandb_run_id,
            "wandb_url": wandb_url,
            "wandb_error": self._wandb_error,
        }
        receipt["receipt_hash"] = canonical_hash(receipt)
        self.receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        return TelemetryResult(
            metrics_path=self.metrics_path,
            receipt_path=self.receipt_path,
            status=status,
            wandb_run_id=wandb_run_id,
            wandb_url=wandb_url,
        )


__all__ = [
    "TelemetryMode",
    "TelemetryProtocol",
    "TelemetryResult",
    "TrainingTelemetry",
    "collect_training_metrics",
    "load_telemetry_protocol",
]
