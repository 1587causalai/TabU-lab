"""Persisted codec identities; existing numbers and realizations never change."""

# V5.3's default is the raw constant-weight compositional codec: 128/4
# numeric/ordinal bases and 128/8 nominal identities.  Unit-Gaussian remains
# a fully supported, explicitly selected comparison mode.
DEFAULT_CODEC_VERSION = "constant_weight_v1"
CODEC_IDS = {
    "legacy_v53": 0,
    "unit_gaussian_v1": 1,  # Historical shared-origin ordinal line.
    "unit_gaussian_v2": 2,  # Category identity plus shared rank direction.
    "constant_weight_v1": 3,  # Raw 128/4 bases and 128/8 nominal identities.
}
CODEC_VERSIONS = tuple(CODEC_IDS)
