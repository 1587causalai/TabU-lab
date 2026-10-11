"""Opt-in V8 matrix completion prototype; never selected by V7 factories."""

from .migration import from_v7, from_v7_checkpoint, load_weights, save_weights
from .model import (
    CompletionEpisode,
    CompletionOutput,
    V8CompletionModel,
    V8Config,
    completion_loss,
    prepare_completion,
)
from .precision import configure_precision

__all__ = [
    "CompletionEpisode",
    "CompletionOutput",
    "V8CompletionModel",
    "V8Config",
    "completion_loss",
    "configure_precision",
    "from_v7",
    "from_v7_checkpoint",
    "load_weights",
    "prepare_completion",
    "save_weights",
]
