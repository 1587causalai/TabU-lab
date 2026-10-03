"""Explicit V6 initialization, without changing the default V7 architecture.

An exact Query token reconstruction is not a training-stability guarantee:
an ill-conditioned inherited projection can require a very large code seed.
"""

import warnings

import torch

from .config import V7Config
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
    config = V7Config(
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
    weights = parent["model"]
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
