"""Named TAR size presets; the mathematical implementation and defaults stay unchanged."""

from __future__ import annotations

DEFAULT_VALIDATION_SIZE = "small"

# Standard deliberately delegates to TARConfig defaults, avoiding a second default source.
_SIZE_OVERRIDES = {
    "small": dict(blocks=3, width=96, ff_width=192),
    "small-128": dict(blocks=3, width=128, ff_width=256),
    "medium": dict(blocks=6, width=192, ff_width=384),
    "standard": {},
}


def config_for_size(size="standard", *, initialization_seed=None):
    from dataclasses import replace

    from tabu_lab.models.tar import TARConfig

    if size not in _SIZE_OVERRIDES:
        raise ValueError(f"unknown TAR size: {size!r}; choose small, small-128, medium or standard")
    cfg = TARConfig()
    fields = dict(_SIZE_OVERRIDES[size])
    if initialization_seed is not None:
        fields["initialization_seed"] = initialization_seed
    return replace(cfg, **fields)


def list_sizes():
    return [
        dict(
            model_size=size,
            parameter_count=(cfg := config_for_size(size)).parameter_count,
            blocks=cfg.blocks,
            width=cfg.width,
            ff_width=cfg.ff_width,
            heads=cfg.heads,
            semantic_slots=cfg.semantic_slots,
            inducing_slots=cfg.inducing_slots,
        )
        for size in _SIZE_OVERRIDES
    ]


def inspect_size(size):
    from tabu_lab.models.tar import TabUTARModel
    from tabu_lab.models.tar.checkpoint import source_digest

    cfg = config_for_size(size)
    model = TabUTARModel(cfg, device="meta")
    count = sum(p.numel() for p in model.parameters())
    if count != cfg.parameter_count:
        raise AssertionError("parameter count differs from configuration formula")
    return dict(
        model_id=model.model_id,
        model_size=size,
        status="local_unissued",
        parameter_count=count,
        parameter_tensors=len(list(model.parameters())),
        config=cfg.as_dict(),
        source_sha256=source_digest(),
    )
