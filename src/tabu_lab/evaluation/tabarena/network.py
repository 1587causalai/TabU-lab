"""ModelSpec-faithful inference, with explicitly verified historical overlays."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import torch

from tabu_lab.models.restoration_v7.config import V7Config
from tabu_lab.models.restoration_v7.model import InheritedV6Dynamics, V7Model

# These complete frozen manifests were checked against their on-disk files.
# The historical writers recorded their digest and architecture in checkpoints;
# their V7Config did not contain query_source, although the wrapper enabled it.
# An architecture name or an arbitrary opt-in flag alone is not source evidence.
_LEGACY_OVERLAY = "donor-source-cyclic-v1"
_LEGACY_SOURCE_MANIFESTS = {
    "6c535942e2072a6480f357e133ec48a583d9ca4ab21f0c336c0ec335618623ec":
        "v7-cyclic618-mini-20261002",
    "342dde3bf9cbfaa28314373079fe03d01a62d028026934fda66acb83993216ba":
        "v7-cyclic2000-20261002",
}


class QuerySourceDynamics(InheritedV6Dynamics):
    """Historical wrapper; installed only after explicit source verification."""

    def forward(self, h, visible, query):
        return super().forward(h, visible | query, query)


class CyclicV7(V7Model):
    """Compatibility class name; its graph follows the stored ModelSpec exactly.

    In particular, query_source=False is a real receiver-only configuration,
    not evidence that a historical wrapper should be inferred.
    """

    def __init__(self, config):
        super().__init__(config)
        for kind in ("numeric", "nominal", "ordinal"):
            self.query_seed(kind).requires_grad_(False)


def _verify_legacy_overlay(saved, config, legacy_overlay):
    if legacy_overlay not in (None, _LEGACY_OVERLAY):
        raise ValueError(f"legacy_overlay must be None or {_LEGACY_OVERLAY!r}")
    recorded_source = "query_source" in saved["config"]
    if legacy_overlay is None:
        if saved.get("architecture") == _LEGACY_OVERLAY and not recorded_source:
            raise ValueError(
                "Historical checkpoint records an external Query-source graph but no "
                "query_source setting; select legacy_overlay='donor-source-cyclic-v1' "
                "explicitly after checking its source identity"
            )
        return None
    if recorded_source:
        raise ValueError("legacy_overlay cannot override an explicit query_source setting")
    source = _LEGACY_SOURCE_MANIFESTS.get(saved.get("source_manifest_sha256"))
    if saved.get("architecture") != _LEGACY_OVERLAY or source is None:
        raise ValueError(
            "legacy_overlay requires a verified historical architecture/source manifest"
        )
    if (
        config.model_version != "v7"
        or config.query_init != "donor"
        or not config.share_rounds
        or config.rounds not in (1, 4)
        or config.unit_layers <= 0
        or not config.coupling_bias
        or config.numeric_preprocessing != "legacy"
        or config.unit_source_policy != "legacy_cell_sources"
        or config.backbone.row_slots != 0
    ):
        raise ValueError("legacy_overlay configuration differs from the verified historical graph")
    return source


def execution_dtype(device):
    return torch.float32 if torch.device(device).type == "mps" else torch.float64


def load_network(
    checkpoint_path, checkpoint_sha256, device="cpu", *, trusted=False, legacy_overlay=None
):
    """Load pinned weights without reinterpreting False as missing metadata.

    An external historical graph needs both explicit selection and a recognized
    checkpoint source identity. This is frozen inference, not strict resume.
    """
    if not checkpoint_path or not checkpoint_sha256:
        raise ValueError("checkpoint_path and checkpoint_sha256 must be explicitly pinned")
    path = Path(checkpoint_path)
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != checkpoint_sha256:
        raise ValueError(f"Checkpoint SHA256 mismatch: {path.name}")
    # Training receipts can contain NumPy RNG state. Unrestricted pickle loading
    # requires an explicit opt-in for a checkpoint the caller trusts.
    try:
        saved = torch.load(path, map_location="cpu", weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        raise ValueError(
            "Checkpoint could not be loaded safely. For your own trusted training file, "
            "--trusted-checkpoint permits full state loading."
        ) from exc
    config = V7Config.from_dict(saved["config"])
    if saved.get("model_version", config.model_version) != config.model_version:
        raise ValueError("Checkpoint model_version differs from its stored ModelSpec")
    legacy_source = _verify_legacy_overlay(saved, config, legacy_overlay)
    with torch.random.fork_rng(devices=[]):
        model = CyclicV7(config).to(dtype=execution_dtype(device))
        if legacy_source is not None:
            for block in model.rounds:
                block.backbone = QuerySourceDynamics(
                    config.backbone, config.unit_layers,
                    unit_source_policy="legacy_cell_sources",
                ).to(dtype=execution_dtype(device))
    model.load_state_dict(saved["model"], strict=True)
    model.to(device=device, dtype=execution_dtype(device)).eval().requires_grad_(False)
    model.checkpoint_sha256 = actual
    model.inference_protocol = {
        "graph": "model-spec" if legacy_source is None else _LEGACY_OVERLAY,
        "legacy_overlay": legacy_overlay,
        "verified_source": legacy_source,
        "source_manifest_sha256": saved.get("source_manifest_sha256"),
        "checkpoint_architecture": saved.get("architecture"),
        "recorded_query_source": saved["config"].get("query_source"),
        "effective_query_source": config.query_source if legacy_source is None else True,
    }
    return model
