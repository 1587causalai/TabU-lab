"""Opt-in V8 matrix completion prototype; never selected by V7 factories."""

from .memory import configure_memory
from .migration import (
    from_v7,
    from_v7_checkpoint,
    from_v8,
    from_v81,
    load_weights,
    save_weights,
    with_supervised_attention,
)
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
    "configure_memory",
    "configure_precision",
    "from_v7",
    "from_v7_checkpoint",
    "from_v8",
    "from_v81",
    "load_weights",
    "prepare_completion",
    "save_weights",
    "with_supervised_attention",
]
