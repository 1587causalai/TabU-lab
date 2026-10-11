"""Audited weights-only imports; no optimizer or training resume implication."""

import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from ..restoration_v7.config import V7Config
from ..restoration_v7.model import V7Model
from ..restoration_v7.runner import CHECKPOINT_SCHEMA, manifest_digest
from .codebook import FEATURE64_MODE, LEARNED_MODES
from .model import V8CompletionModel, V8Config


def from_v8(donor: V8CompletionModel, *, codebook_mode=FEATURE64_MODE):
    """Keep every V8 weight for current 64D mode; initialize only the feature pool.

    Explicit historical 32D mode retains its old backbone-only migration.
    No optimizer state is implicitly migrated.
    """
    if donor.config.codebook_mode != "inherited" or donor.config.readout != "gaussian_ll":
        raise ValueError("V8.1 migration requires an inherited-code Gaussian LL V8 donor")
    if codebook_mode not in LEARNED_MODES:
        raise ValueError("unknown learned feature mode")
    _finite_state(donor.state_dict())
    target = V8CompletionModel(
        replace(donor.config, codebook_mode=codebook_mode)
    ).to(next(donor.parameters()))
    if codebook_mode == FEATURE64_MODE:
        state = target.state_dict()
        state.update(donor.state_dict())
        target.load_state_dict(state, strict=True)
        copied = list(donor.state_dict())
    else:
        target.backbone.load_state_dict(donor.backbone.state_dict(), strict=True)
        with torch.no_grad():
            target.unit_seed.copy_(donor.unit_seed)
            target.feature_seed.copy_(donor.feature_seed)
        copied = [f"backbone.{key}" for key in donor.backbone.state_dict()]
        copied += ["unit_seed", "feature_seed"]
    for key in copied:
        if not torch.equal(target.state_dict()[key], donor.state_dict()[key]):
            raise ValueError(f"weight copy mismatch: {key}")
    target.migration_receipt = {
        "mode": "weights_only",
        "donor_version": "v8",
        "copied_keys": copied,
        "new_keys": sorted(set(target.state_dict()) - set(copied)),
        "donor_config": donor.config.as_dict(),
        "omitted_donor_components": (
            ["optimizer", "rng"] if codebook_mode == FEATURE64_MODE
            else ["value_interface", "query_seed", "optimizer", "rng"]
        ),
    }
    return target


def _finite_state(state):
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("checkpoint contains nonfinite model weights")


def from_v7(
    donor: V7Model, *, round_index=None, rounds=1, copy_value_interface=True, readout="gaussian_ll"
):
    """Copy the COMPLETE selected Token Dynamics and its Unit/Feature seeds.

    Compatible per-cell phi/lift can also be copied. A row encoder is never
    copied; the four V8 table blocks and shared Query seed are always new.
    Unshared donors require an explicit selected round, rather than dropping
    other trained rounds silently. This preserves weights, not old predictions.
    """
    if round_index is None:
        if not donor.config.share_rounds and donor.config.rounds > 1:
            raise ValueError("unshared donor requires an explicit round_index")
        round_index = 0
    if type(round_index) is not int or not 0 <= round_index < donor.config.rounds:
        raise ValueError("donor round_index is out of bounds")
    source = donor.round_module(round_index)
    if copy_value_interface and not (hasattr(source, "phi") and hasattr(source, "lift")):
        raise ValueError(
            "donor has no compatible phi/lift; explicitly disable value-interface copy"
        )
    _finite_state(donor.state_dict())
    parameter = next(donor.parameters())
    target = V8CompletionModel(V8Config(donor.config, rounds, readout)).to(parameter)
    copied = []
    modules = [("backbone", target.backbone, source.backbone)]
    if copy_value_interface:
        modules += [
            ("encoder.phi", target.encoder.phi, source.phi),
            ("lift", target.lift, source.lift),
        ]
    for name, destination, origin in modules:
        destination.load_state_dict(origin.state_dict(), strict=True)
        for key, value in destination.state_dict().items():
            if not torch.equal(value, origin.state_dict()[key]):
                raise ValueError(f"weight copy mismatch: {name}.{key}")
            copied.append(f"{name}.{key}")
    with torch.no_grad():
        target.unit_seed.copy_(source.unit_seed)
        target.feature_seed.copy_(source.feature_seed)
    copied += ["unit_seed", "feature_seed"]
    target.migration_receipt = {
        "mode": "weights_only",
        "donor_version": donor.config.model_version,
        "donor_round": round_index,
        "donor_value_encoder": donor.config.value_encoder,
        "copied_keys": copied,
        "new_keys": sorted(set(target.state_dict()) - set(copied)),
        "omitted_donor_components": [
            "query_initialization",
            "row_encoder_if_any",
            "other_unshared_rounds_if_any",
            "optimizer",
            "rng",
        ],
    }
    return target


def from_v7_checkpoint(
    path,
    *,
    round_index=None,
    rounds=1,
    copy_value_interface=True,
    device="cpu",
    readout="gaussian_ll",
):
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("architecture") != "restoration_v7":
        raise ValueError("unsupported V7 checkpoint schema")
    config = V7Config.from_dict(state["config"])
    if state.get("model_version", config.model_version) != config.model_version:
        raise ValueError("donor version disagrees with its architecture spec")
    if not isinstance(state.get("manifest"), dict) or state.get(
        "manifest_sha256"
    ) != manifest_digest(state["manifest"]):
        raise ValueError("donor manifest digest mismatch")
    _finite_state(state["model"])
    dtypes = {value.dtype for value in state["model"].values() if value.is_floating_point()}
    if len(dtypes) != 1 or not dtypes <= {torch.float32, torch.float64}:
        raise ValueError("donor requires uniform FP32 or FP64 parameters")
    donor = V7Model(config).to(device=device, dtype=next(iter(dtypes)))
    donor.load_state_dict(state["model"], strict=True)
    model = from_v7(
        donor,
        round_index=round_index,
        rounds=rounds,
        copy_value_interface=copy_value_interface,
        readout=readout,
    )
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    model.migration_receipt.update(donor_sha256=digest, donor_step=state["step"])
    return model


def save_weights(model, path):
    """Write once, portable V8 weights plus architecture and migration receipt."""
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    _finite_state(state)
    with Path(path).open("xb") as handle:
        torch.save(
            {
                "schema": (
                    "completion_v81_weights_v1" if model.config.model_version == "v8.1"
                    else "completion_v8_weights_v1"
                ),
                "config": model.config.as_dict(),
                "model": state,
                "migration": model.migration_receipt,
                "execution_precision": model.execution_precision,
            },
            handle,
        )


def load_weights(path, *, device="cpu", precision=None, precision_reason=None):
    """Preserve weight dtype; FP32 weights default to current FP32 execution.

    For historical execution explicitly pass ``precision="reference"`` or the
    recorded mixed mode. The saved receipt is provenance, not an implicit override.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("schema") not in ("completion_v8_weights_v1", "completion_v81_weights_v1"):
        raise ValueError("unsupported V8 weights schema")
    config = V8Config.from_dict(state["config"])
    expected = (
        "completion_v81_weights_v1" if config.model_version == "v8.1"
        else "completion_v8_weights_v1"
    )
    if state["schema"] != expected:
        raise ValueError("weights schema disagrees with codebook mode")
    _finite_state(state["model"])
    dtypes = {value.dtype for value in state["model"].values() if value.is_floating_point()}
    if len(dtypes) != 1 or not dtypes <= {torch.float32, torch.float64}:
        raise ValueError("V8 weights require uniform FP32 or FP64 parameters")
    model = V8CompletionModel(config).to(
        device=device, dtype=next(iter(dtypes))
    )
    model.load_state_dict(state["model"], strict=True)
    model.migration_receipt = state["migration"]
    if precision is not None:
        from .precision import configure_precision

        configure_precision(model, mode=precision, reason=precision_reason)
    return model
