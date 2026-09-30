"""TabU-TAR: typed additive readout, fixed inducing, independent model identity."""

from .checkpoint import load_checkpoint, save_checkpoint
from .config import TARConfig
from .inference import predict_joint_supervised, predict_supervised
from .model import TabUTARModel
from .training import TARTrainer, TARTrainingConfig, score
from .types import TAREpisode, TARFeature, TAROutput, TARPrediction

__all__ = [
    "TARConfig",
    "TAREpisode",
    "TARFeature",
    "TAROutput",
    "TARPrediction",
    "TARTrainer",
    "TARTrainingConfig",
    "TabUTARModel",
    "load_checkpoint",
    "predict_joint_supervised",
    "predict_supervised",
    "save_checkpoint",
    "score",
]
