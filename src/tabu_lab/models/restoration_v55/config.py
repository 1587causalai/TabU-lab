"""V5.5 default configuration, with historical ordinal lines as explicit candidates."""

from __future__ import annotations

from dataclasses import dataclass

from ..restoration_v53.codec_versions import V55_CODEC_VERSIONS, V55_DEFAULT_CODEC_VERSION
from ..restoration_v54.config import V54_SIZES, V54Config

V55_SIZES = V54_SIZES


@dataclass(frozen=True)
class V55Config(V54Config):
    """Keep V5.4 sizes; default to Unit depth zero and the full ordinal code."""

    codec_version: str = V55_DEFAULT_CODEC_VERSION
    unit_layers: int | None = 0

    def _allowed_codec_versions(self) -> tuple[str, ...]:
        return V55_CODEC_VERSIONS

    def __post_init__(self):
        # A missing or explicit null Unit depth keeps V5.5's zero-depth default
        # for every size; positive depth is an explicit comparison setting.
        if self.unit_layers is None:
            object.__setattr__(self, "unit_layers", 0)
        super().__post_init__()
