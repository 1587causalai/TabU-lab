"""Public training surface for dense reference models."""

from .objective import MixedObjective, NumericObjective, Objective
from .telemetry import (
    TelemetryProtocol,
    TelemetryResult,
    TrainingTelemetry,
    collect_training_metrics,
    load_telemetry_protocol,
)
from .trainer import Trainer, TrainStep, train, train_model

__all__ = [
    "MixedObjective",
    "NumericObjective",
    "Objective",
    "TelemetryProtocol",
    "TelemetryResult",
    "TrainStep",
    "Trainer",
    "TrainingTelemetry",
    "collect_training_metrics",
    "load_telemetry_protocol",
    "train",
    "train_model",
]
