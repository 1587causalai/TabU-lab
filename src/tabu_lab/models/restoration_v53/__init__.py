"""V5.3 reference implementation; distinct from historical Restoration/TAR."""

from ..restoration.backbone import BackboneConfig
from ..restoration.contracts import (
    ColumnSchema,
    RestorationInput,
    RestorationRequest,
    TruthSidecar,
    make_episode,
)
from .answers import AffineOrdinalAnswers, GaussianNominalAnswers, ZScoreAnswers
from .encoding import AffineNumericAnswers
from .model import V53Config, V53Model
from .readout import FeatureSlopeProvider
from .training import (
    V53LossConfig,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)

__all__ = [
    "AffineNumericAnswers", "AffineOrdinalAnswers", "BackboneConfig", "ColumnSchema",
    "FeatureSlopeProvider", "GaussianNominalAnswers",
    "RestorationInput", "RestorationRequest", "TruthSidecar", "V53Config", "V53LossConfig",
    "V53Model", "ZScoreAnswers", "make_episode", "prepare_episode", "score_episode",
    "score_prepared_episode",
]
