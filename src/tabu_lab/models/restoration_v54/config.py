"""V5.4 composition identity and resolved, explicitly overridable size presets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..restoration.backbone import BackboneConfig
from ..restoration_v53.codec_versions import (
    COMPOSITION_CODEC_VERSIONS,
    DEFAULT_COMPOSITION_CODEC_VERSION,
)
from ..restoration_v53.model import V53Config

# (backbone layers, width, heads, FF width, Unit layers, inducing slots).
# TabU_V5p4_Unified_Composition.tex, appendix "Model sizes".
_SIZE_FIELDS = {
    "nano": (2, 128, 4, 256, 0, 256),
    "small": (3, 128, 8, 256, 3, 256),
    "medium": (6, 192, 8, 384, 3, 256),
    "standard": (12, 256, 8, 512, 6, 256),
    "large": (24, 384, 12, 768, 6, 256),
}
V54_SIZES = tuple(_SIZE_FIELDS)


@dataclass(frozen=True)
class V54Config(V53Config):
    """Resolve a named size, then apply explicit fields without renaming it.

    A partial ``backbone`` mapping overrides only its supplied fields. A
    ``BackboneConfig`` is already fully explicit and is retained as provided.
    After construction, ``backbone`` and ``unit_layers`` are always resolved;
    inherited ``as_dict`` persists those actual values alongside ``size``.
    Thus Small with ``unit_layers=0`` remains an explicit Small ablation.
    """

    backbone: BackboneConfig | Mapping[str, Any] | None = None
    unit_layers: int | None = None
    codec_version: str = DEFAULT_COMPOSITION_CODEC_VERSION
    size: str = "small"
    subtokens: int = 1

    def __post_init__(self):
        if not isinstance(self.size, str) or self.size.lower() not in _SIZE_FIELDS:
            raise ValueError(f"V5.4 size must be one of {V54_SIZES}")
        size = self.size.lower()
        object.__setattr__(self, "size", size)
        layers, width, heads, ff_width, unit_layers, slots = _SIZE_FIELDS[size]
        if self.backbone is None or isinstance(self.backbone, Mapping):
            fields = dict(layers=layers, width=width, heads=heads, ff_width=ff_width, slots=slots)
            if self.backbone is not None:
                fields.update(self.backbone)
            object.__setattr__(self, "backbone", BackboneConfig(**fields))
        elif not isinstance(self.backbone, BackboneConfig):
            raise TypeError("V5.4 backbone must be a BackboneConfig or field mapping")
        if self.unit_layers is None:
            object.__setattr__(self, "unit_layers", unit_layers)
        if type(self.subtokens) is not int or self.subtokens != 1:
            raise ValueError("V5.4 supports only subtokens=1; K>1 is not yet defined")
        if self.codec_version not in COMPOSITION_CODEC_VERSIONS:
            raise ValueError("V5.4 requires an explicit composition codec identity")
        super().__post_init__()

    @classmethod
    def from_dict(cls, values):
        """Accept a preset, a partial override, or a fully resolved saved config."""
        return cls(**dict(values))

    @classmethod
    def from_size(cls, size: str, **overrides):
        return cls.from_dict({"size": size, **overrides})
