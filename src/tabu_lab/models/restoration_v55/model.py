"""V5.5 entry point over the shared restoration execution engine."""

from __future__ import annotations

from ..restoration_v53.model import V53Model
from ..restoration_v53.readout import FeatureSlopeProvider
from .config import V55Config


class V55Model(V53Model):
    def __init__(
        self, config: V55Config | None = None, *, feature_slope: FeatureSlopeProvider | None = None
    ):
        if config is not None and not isinstance(config, V55Config):
            raise TypeError(
                "V55Model requires V55Config; historical versions use their own entry point"
            )
        super().__init__(config if config is not None else V55Config(), feature_slope=feature_slope)
