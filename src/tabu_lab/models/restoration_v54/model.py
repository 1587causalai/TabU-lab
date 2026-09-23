"""V5.4 entry point over the shared restoration execution engine."""

from __future__ import annotations

from ..restoration_v53.model import V53Model
from ..restoration_v53.readout import FeatureSlopeProvider
from .config import V54Config


class V54Model(V53Model):
    def __init__(
        self, config: V54Config | None = None, *, feature_slope: FeatureSlopeProvider | None = None
    ):
        if config is not None and not isinstance(config, V54Config):
            raise TypeError(
                "V54Model requires V54Config; legacy configs keep their own entry point"
            )
        super().__init__(config if config is not None else V54Config(), feature_slope=feature_slope)
