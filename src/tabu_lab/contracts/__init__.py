"""Stable L0 contracts shared by every TabU-lab model adapter."""

from .bundles import (
    EvaluationBundle,
    ForwardTrace,
    LossBundle,
    PredictionBundle,
    PredictionEntry,
    PredictionKind,
    PredictionStatus,
    TraceEvent,
)
from .canonical import (
    CanonicalizationError,
    canonical_hash,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    require_sha256,
    to_canonical_data,
)
from .dataset import EpisodeRecipe, RawDataset, SplitManifest, SplitView
from .episode import EvidenceEpisode, TruthSidecar, assert_truth_free
from .features import FeatureKind, FeatureRole, FeatureSpec
from .roles import (
    ForwardRole,
    OriginState,
    decode_forward_roles,
    decode_origin_states,
    encode_forward_roles,
    encode_origin_states,
    forward_role_code,
    forward_role_mask,
    origin_code,
    origin_mask,
    origin_value_mask,
)
from .topology import GraphDirection, GraphTopology

__all__ = [
    "CanonicalizationError",
    "EpisodeRecipe",
    "EvaluationBundle",
    "EvidenceEpisode",
    "FeatureKind",
    "FeatureRole",
    "FeatureSpec",
    "ForwardRole",
    "ForwardTrace",
    "GraphDirection",
    "GraphTopology",
    "LossBundle",
    "OriginState",
    "PredictionBundle",
    "PredictionEntry",
    "PredictionKind",
    "PredictionStatus",
    "RawDataset",
    "SplitManifest",
    "SplitView",
    "TraceEvent",
    "TruthSidecar",
    "assert_truth_free",
    "canonical_hash",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_sha256",
    "decode_forward_roles",
    "decode_origin_states",
    "encode_forward_roles",
    "encode_origin_states",
    "forward_role_code",
    "forward_role_mask",
    "origin_code",
    "origin_mask",
    "origin_value_mask",
    "require_sha256",
    "to_canonical_data",
]
