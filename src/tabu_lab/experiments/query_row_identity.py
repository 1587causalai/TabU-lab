"""Shared TabUR@0.2 identity projections without experiment dependencies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_READOUT_SCHEMA = "tabu.query-row-readout.v1"
_READOUT_MODES = frozenset({"homogeneous", "anchored", "free"})


def require_query_row_readout_identity(
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a validated TabUR@0.2 readout identity without applying defaults."""

    if model_identity.get("model_id") != "tabu.query.row":
        raise ValueError("TabUR checkpoint model_id must be tabu.query.row")
    if model_identity.get("contract_version") != "0.2.0":
        raise ValueError("TabUR checkpoint contract_version must be 0.2.0")
    readout = model_identity.get("row_readout")
    if not isinstance(readout, Mapping):
        raise ValueError("TabUR checkpoint model_identity.row_readout is required")
    required = {
        "schema_version",
        "mode",
        "beta",
        "anchored_gamma_initial",
        "axis_transform_normalization",
        "row_token_count",
        "global_w_rows",
    }
    missing = sorted(required - set(readout))
    if missing:
        raise ValueError(f"TabUR checkpoint row_readout is missing fields: {missing}")
    unexpected = sorted(set(readout) - required)
    if unexpected:
        raise ValueError(f"TabUR checkpoint row_readout has unexpected fields: {unexpected}")
    if readout["schema_version"] != _READOUT_SCHEMA:
        raise ValueError("TabUR checkpoint row_readout schema_version is unsupported")
    mode = readout["mode"]
    if mode not in _READOUT_MODES:
        raise ValueError(f"TabUR checkpoint row_readout mode is unsupported: {mode!r}")
    gamma = readout["anchored_gamma_initial"]
    if isinstance(gamma, bool) or not isinstance(gamma, int | float):
        raise ValueError("TabUR checkpoint anchored_gamma_initial must be numeric")
    if not math.isfinite(float(gamma)):
        raise ValueError("TabUR checkpoint anchored_gamma_initial must be finite")
    if mode != "anchored" and float(gamma) != 1.0e-2:
        raise ValueError(
            "TabUR homogeneous/free identity requires canonical anchored_gamma_initial=0.01"
        )
    beta = readout["beta"]
    if isinstance(beta, bool) or not isinstance(beta, int | float):
        raise ValueError("TabUR checkpoint row_readout beta must be numeric")
    expected_beta = 0.0 if mode == "free" else 1.0
    if float(beta) != expected_beta:
        raise ValueError("TabUR checkpoint row_readout beta does not match its mode")
    if readout["axis_transform_normalization"] != "exact_spectral_norm_v1":
        raise ValueError("TabUR checkpoint axis_transform_normalization is unsupported")
    for field in ("row_token_count", "global_w_rows"):
        value = readout[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"TabUR checkpoint row_readout {field} must be positive")
    token_count = readout["row_token_count"]
    if model_identity.get("row_token_count") != token_count:
        raise ValueError("TabUR checkpoint row_readout K disagrees with row_token_count")
    reference = model_identity.get("reference_config")
    if not isinstance(reference, Mapping):
        raise ValueError("TabUR checkpoint reference_config is required")
    if reference.get("matched_slots") != token_count:
        raise ValueError("TabUR checkpoint row_readout K disagrees with matched_slots")
    if readout["global_w_rows"] != token_count:
        raise ValueError("TabUR checkpoint row_readout K disagrees with global W rows")
    return dict(readout)


def query_row_result_identity(model_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Project the identity fields every comparable TabUR result must carry."""

    readout = require_query_row_readout_identity(model_identity)
    variant_ref = model_identity.get("variant_ref")
    if not isinstance(variant_ref, Mapping):
        raise ValueError("TabUR result identity requires variant_ref")
    for field, identity_field in (
        ("contract_id", "model_id"),
        ("contract_version", "contract_version"),
        ("profile_id", "profile_id"),
    ):
        if variant_ref.get(field) != model_identity.get(identity_field):
            raise ValueError(f"TabUR result identity variant_ref mismatch at {field}")
    model_spec_hash = variant_ref.get("model_spec_hash")
    variant_hash = model_identity.get("variant_hash")
    for field, value in (
        ("model_spec_hash", model_spec_hash),
        ("variant_hash", variant_hash),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"TabUR result identity {field} must be a lowercase SHA-256")
    return {
        "model_id": str(model_identity["model_id"]),
        "contract_version": str(model_identity["contract_version"]),
        "profile_id": str(model_identity["profile_id"]),
        "model_spec_hash": str(model_spec_hash),
        "variant_hash": str(variant_hash),
        "row_readout_mode": str(readout["mode"]),
        "row_readout_identity": readout,
    }


__all__ = ["query_row_result_identity", "require_query_row_readout_identity"]
