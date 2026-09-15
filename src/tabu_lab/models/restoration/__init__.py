"""Five-step table restoration; implementation checks are not capability claims."""

from .answers import CategoricalAnswers, NumericAnswers
from .backbone import BackboneConfig
from .contracts import (
    ColumnSchema,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    make_episode,
)
from .encoding import EncoderConfig
from .losses import encoding_mse
from .model import RestorationConfig, RestorationModel, RestorationOutput
from .readout import EncodedRestoration, RestorationReadout, unit_kernel_logits
from .training import LossConfig, batch_loss, score_episode

__all__ = [
    "BackboneConfig",
    "CategoricalAnswers",
    "ColumnSchema",
    "EncodedRestoration",
    "EncoderConfig",
    "LossConfig",
    "NumericAnswers",
    "RestorationConfig",
    "RestorationInput",
    "RestorationModel",
    "RestorationOutput",
    "RestorationReadout",
    "RestorationRequest",
    "TruthSidecar",
    "batch_loss",
    "encoding_mse",
    "make_episode",
    "score_episode",
    "unit_kernel_logits",
]
