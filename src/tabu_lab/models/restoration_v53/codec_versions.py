"""Persisted codec identities; existing numbers and realizations never change."""

DEFAULT_CODEC_VERSION = "unit_gaussian_v2"
CODEC_IDS = {
    "legacy_v53": 0,
    "unit_gaussian_v1": 1,  # Historical shared-origin ordinal line.
    "unit_gaussian_v2": 2,  # Category identity plus shared rank direction.
    "constant_weight_v1": 3,  # Raw 128/4 bases and 128/8 nominal identities.
}
CODEC_VERSIONS = tuple(CODEC_IDS)
