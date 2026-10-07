"""Explicit V6 initialization, without changing the default V7 architecture.

An exact Query token reconstruction is not a training-stability guarantee:
an ill-conditioned inherited projection can require a very large code seed.
"""

import hashlib
import io
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch

from ..restoration_v53.codec_versions import CODEC_IDS
from .config import DualStreamConfig, V7Config
from .model import V7Model


def from_v6_checkpoint(parent, *, device="cpu", dtype=torch.float64, value_map="coupling", seed=0):
    """Select the original codec and inherit all compatible trainable tensors.

    This is weights-only initialization. V7 typed seeds remain independent,
    and target broadcasting is not silently added. New phi starts at identity.
    The least-squares seed is an experimental control. Its norm amplification
    and projection condition number are recorded; an amplified preimage can
    become unstable on the very first shared-projection optimizer update.
    """
    old = parent["model_config"]
    if old.get("codec_version") != "constant_weight_composition_v2":
        raise ValueError("this migration requires the exact V6 composition-v2 codec")
    if old.get("numeric_scaling", "zscore") != "zscore" or old.get("regression_width") is not None:
        raise ValueError("unsupported V6 scaling or projected regression migration")
    if old.get("slope_source", "shared_ll") != "shared_ll":
        raise ValueError("only shared-LL V6 checkpoints are supported")
    weights = parent["model"]
    signature = weights.get("_codec_signature")
    expected_signature = torch.tensor([CODEC_IDS[old["codec_version"]], 1], dtype=torch.long)
    if (
        not isinstance(signature, torch.Tensor)
        or signature.dtype != torch.long
        or not torch.equal(signature.detach().cpu(), expected_signature)
    ):
        raise ValueError("parent checkpoint codec/scaling signature differs from its config")
    # Validate identity before constructing a model or changing the caller's RNG.
    config = V7Config.legacy(
        code_dim=128,
        codec=old["codec_version"],
        backbone=old["backbone"],
        unit_layers=old["unit_layers"],
        value_map=value_map,
        bandwidth=old["bandwidth"],
        ridge=old["ridge"],
        epsilon=old["epsilon"],
        center_chunk_size=old.get("center_chunk_size", 128),
    )
    torch.manual_seed(seed)
    model = V7Model(config).to(device=device, dtype=dtype)
    transferred = {}
    for k, v in weights.items():
        if v.is_floating_point() and not bool(torch.isfinite(v).all()):
            raise ValueError("nonfinite parent tensor: " + k)
        if k.startswith("backbone."):
            dest = (
                "rounds.0.backbone."
                + ("axial." if config.unit_layers else "")
                + k.removeprefix("backbone.")
            )
        elif k.startswith("unit_blocks."):
            dest = "rounds.0.backbone." + k
        elif k == "encoder.projection.weight":
            dest = "rounds.0.lift.weight"
        elif k in ("encoder.unit_seed", "encoder.feature_seed"):
            dest = "rounds.0." + k.removeprefix("encoder.")
        elif k in ("encoder.cell_seed", "_codec_signature"):
            continue
        else:
            raise ValueError("unhandled parent tensor: " + k)
        transferred[dest] = v
    expected = {
        k
        for k in model.state_dict()
        if ".backbone." in k or k.endswith(("lift.weight", "unit_seed", "feature_seed"))
    }
    if set(transferred) != expected:
        raise ValueError("incomplete compatible V6 tensor transfer")
    status = model.load_state_dict(transferred, strict=False)
    if status.unexpected_keys or set(status.missing_keys) != set(model.state_dict()) - expected:
        raise ValueError("unexpected migration keys")
    for k, v in transferred.items():
        if not torch.equal(model.state_dict()[k], v.to(device=device, dtype=dtype)):
            raise ValueError("inexact compatible tensor transfer: " + k)
    # Solve after dtype conversion; rectangular projections may not span cell_seed.
    w = model.rounds[0].lift.weight.detach().cpu().double()
    target = weights["encoder.cell_seed"].to(dtype=dtype).cpu().double()
    code = torch.linalg.lstsq(w, target, driver="gelsd").solution.to(device=device, dtype=dtype)
    singular_values = torch.linalg.svdvals(w)
    smallest = float(singular_values[-1])
    condition = float(singular_values[0]) / smallest if smallest > 0 else None
    code_norm = float(code.detach().cpu().double().norm())
    token_norm = float(target.norm())
    amplification = code_norm / token_norm if token_norm > 0 else None
    # Diagnostic threshold, not a qualification criterion for smaller seeds.
    amplified = code_norm > max(10.0, 10.0 * token_norm)
    if amplified:
        warnings.warn(
            "V6 Query seed preimage is strongly amplified; exact initial token "
            "reconstruction does not imply stable training with a trainable "
            "shared projection. Inspect transfer_receipt before using this control.",
            RuntimeWarning,
            stacklevel=2,
        )
    with torch.no_grad():
        for kind in ("numeric", "nominal", "ordinal"):
            model.query_seed(kind).copy_(code)
        reconstructed = model.rounds[0].lift(code)
        error = float((reconstructed.cpu().double() - target).abs().max())
    model.transfer_receipt = {
        "copied_tensors": sorted(transferred),
        "copied_tensor_count": len(transferred),
        "all_copied_tensors_exact_after_dtype_conversion": True,
        "query_seed_method": (
            "least-squares preimage of inherited cell_seed under inherited projection"
        ),
        "query_seed_max_token_error": error,
        "query_seed_exact_within_1e_5": error <= 1e-5,
        "query_seed_code_norm": code_norm,
        "inherited_query_token_norm": token_norm,
        "query_seed_norm_amplification": amplification,
        "projection_condition_number": condition,
        "amplified_query_preimage": amplified,
        "training_stability_qualified": False,
        "qualification_scope": "tensor transfer and initial token reconstruction only",
        "new_parameters": [k for k in model.state_dict() if ".phi." in k],
        "not_inherited": ["_codec_signature (new config records codec identity)"],
        "target_broadcast": False,
        "optimizer_resume": False,
    }
    return model


_DUAL_STREAM_EXCLUDED = (
    ("phi.", "old per-Cell phi; replaced by the row dual-stream encoder"),
    ("lift.", "old entry lift W_up; Z enters token dynamics directly"),
    ("query_seed_", "code-space Query seeds lie outside the specified transfer set"),
)


def _dual_stream_policy(key: str) -> str | None:
    """Return the exclusion reason for a donor key, or None if transferable."""
    name = key.split(".", 2)[2] if key.startswith("rounds.") else key
    for prefix, reason in _DUAL_STREAM_EXCLUDED:
        if name.startswith(prefix):
            return reason
    return None


def _transferable(key: str) -> bool:
    if not key.startswith("rounds."):
        return False
    name = key.split(".", 2)[2]
    return name.startswith("backbone.") or name in ("unit_seed", "feature_seed")


def row_dual_stream_from_checkpoint(
    path,
    *,
    device="cpu",
    dtype=torch.float64,
    seed=0,
    dual_stream: DualStreamConfig | None = None,
    expected_sha256: str | None = None,
):
    """Weights-only donor transfer into a new ``row_dual_stream`` model.

    The donor must be a ``phi_lift`` V7/V7.3 runner checkpoint. Its ModelSpec
    is the backbone authority: layers, heads, FFN, slots, Unit layers, rounds,
    sharing and LL settings are inherited unchanged; only the value encoder
    switches. Token dynamics and Unit/Feature seeds are copied exactly (after
    dtype conversion); ``phi``/``lift`` are not reused; encoder parameters are
    new. The donor optimizer, RNG and sampler state are deliberately ignored:
    this is not a training resume. Every tensor is classified and the target
    is loaded with ``strict=True``. A successful transfer proves weight
    reuse only, not that the new encoding is adapted to the donor dynamics.
    """
    from .runner import CHECKPOINT_SCHEMA

    path = Path(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("donor checkpoint SHA-256 differs from the declared receipt")
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("architecture") != "restoration_v7":
        raise ValueError("unsupported donor checkpoint schema")
    donor_config = V7Config.from_dict(state["config"])
    if state.get("model_version", donor_config.model_version) != donor_config.model_version:
        raise ValueError("donor model_version differs from its stored ModelSpec")
    if state.get("share_rounds", donor_config.share_rounds) != donor_config.share_rounds:
        raise ValueError("donor share_rounds differs from its stored ModelSpec")
    if donor_config.value_encoder != "phi_lift":
        raise ValueError("donor must be a phi_lift V7 checkpoint")
    weights = state["model"]
    for key, value in weights.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite donor tensor: " + key)
    source_dtypes = {v.dtype for v in weights.values() if v.is_floating_point()}
    if len(source_dtypes) != 1:
        raise ValueError("donor floating tensors must share one dtype")
    # Prove that the donor metadata reconstructs the donor graph exactly.
    with torch.random.fork_rng(devices=[]):
        donor_model = V7Model(donor_config).to(dtype=next(iter(source_dtypes)))
        donor_model.load_state_dict(weights, strict=True)
    del donor_model
    backbone = donor_config.backbone
    dual_stream = dual_stream or DualStreamConfig(
        tau_presence=backbone.tau_presence,
        reference_mass=backbone.reference_mass,
        norm_eps=backbone.norm_eps,
    )
    target_config = replace(donor_config, value_encoder="row_dual_stream", dual_stream=dual_stream)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = V7Model(target_config)
    model = model.to(device=device, dtype=dtype)
    initial = model.state_dict()
    copied, not_copied, shape_mismatched = {}, [], []
    for key, value in weights.items():
        reason = _dual_stream_policy(key)
        if reason is not None:
            not_copied.append(dict(key=key, shape=list(value.shape), reason=reason))
            continue
        if not _transferable(key) or key not in initial:
            raise ValueError("unclassified donor tensor: " + key)
        if tuple(initial[key].shape) != tuple(value.shape):
            shape_mismatched.append(
                dict(key=key, donor=list(value.shape), target=list(initial[key].shape))
            )
            continue
        copied[key] = value
    if shape_mismatched:
        raise ValueError(f"donor/target shape mismatch: {shape_mismatched}")
    new = [k for k in initial if k not in weights]
    unexpected_new = [k for k in new if ".encoder." not in k]
    if unexpected_new:
        raise ValueError(f"target tensors neither copied nor new encoder: {unexpected_new}")
    kept_fresh = [k for k in initial if k not in copied and k in weights]
    full = {
        k: (copied[k].to(device=device, dtype=dtype) if k in copied else v)
        for k, v in initial.items()
    }
    model.load_state_dict(full, strict=True)
    loaded = model.state_dict()
    tensors = []
    for key, value in copied.items():
        target = loaded[key].detach().cpu()
        if not torch.equal(target, value.to(dtype=dtype)):
            raise ValueError("inexact compatible tensor transfer: " + key)
        error = float((target.double() - value.double()).abs().max()) if value.numel() else 0.0
        tensors.append(
            dict(
                key=key,
                shape=list(value.shape),
                source_dtype=str(value.dtype),
                target_dtype=str(loaded[key].dtype),
                exact_after_conversion=True,
                max_abs_conversion_error=error,
            )
        )
    model.transfer_receipt = {
        "source_path": str(path),
        "source_sha256": digest,
        "source_step": state.get("step"),
        "source_execution": state.get("execution"),
        "source_manifest_sha256": state.get("manifest_sha256"),
        "donor_config": donor_config.as_dict(),
        "target_config": target_config.as_dict(),
        "target_device": str(next(model.parameters()).device),
        "target_dtype": str(dtype),
        "source_dtypes": dict(Counter(str(v.dtype) for v in weights.values())),
        "seed": seed,
        "copied_tensors": tensors,
        "copied_tensor_count": len(tensors),
        "copied_parameter_count": sum(int(v.numel()) for v in copied.values()),
        "not_copied": not_copied,
        "kept_fresh_in_target": kept_fresh,
        "new_tensors": [dict(key=k, shape=list(initial[k].shape)) for k in new],
        "new_parameter_count": sum(int(initial[k].numel()) for k in new),
        "shape_mismatched": shape_mismatched,
        "load_strict": True,
        "optimizer_resume": False,
        "ignored_donor_state": sorted(
            k for k in ("optimizer", "torch_cpu_rng", "accelerator_rng") if k in state
        ),
        "target_broadcast": False,
        "qualification_scope": (
            "weight transfer only; it does not show that the new row encoding is "
            "semantically adapted to the donor token dynamics"
        ),
    }
    return model
