"""Stable descriptions of concrete reference-model compositions."""

from __future__ import annotations

from typing import Any

from .contracts import ModelCompositionDescriptor


def _component_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def describe_model(
    model: Any, *, contract_id: str | None = None, contract_version: str | None = None
) -> ModelCompositionDescriptor:
    """Describe only declared component seams and semantic configuration.

    The function is intentionally conservative: it never serializes tensors or
    arbitrary object reprs into an identity.
    """

    inferred_id = contract_id or str(
        getattr(model, "model_id", getattr(model, "contract_id", "unknown"))
    )
    inferred_version = contract_version or str(getattr(model, "contract_version", "0.0.0"))
    roles = ("symbolizer", "tokenizer", "dynamics", "readout", "objective", "loss")
    components: dict[str, str] = {}
    modules: dict[str, str] = {}
    for role in roles:
        value = getattr(model, role, None)
        if value is None:
            continue
        components[role] = _component_name(value)
        modules[role] = type(value).__module__

    config = getattr(model, "config", None)
    configuration: dict[str, Any] = {}
    if config is not None:
        # Keep the complete reference configuration in the composition
        # identity.  The compact fields below are retained for readable
        # projections, but a partial projection would allow two semantically
        # different Base checkpoints to collide.
        configuration["reference_config"] = {
            key: getattr(value, "value", value)
            for key, value in getattr(config, "__dict__", {}).items()
        }
        for key in (
            "block_kind",
            "geometry_normalization",
            "routing_bandwidth",
            "numeric_terminal",
            "d_model",
            "n_heads",
            "d_ff",
            "n_blocks",
            "inducing_slots",
            "matched_slots",
            "max_features",
            "dropout",
            "presence_tau",
            "denominator_epsilon",
        ):
            if hasattr(config, key):
                value = getattr(config, key)
                configuration[key] = getattr(value, "value", value)
    for key in (
        "readout_geometry",
        "recommendation_address_plan",
        "label_address_plan",
        "numeric_terminal",
        "profile",
        "label_broadcast",
        "label_broadcast_tau",
        "variant_ref",
    ):
        if hasattr(model, key):
            value = getattr(model, key)
            if key == "variant_ref" and value is not None:
                value = value.as_dict() if hasattr(value, "as_dict") else value
            configuration[key] = getattr(value, "value", value)
    for role in ("readout", "label_readout"):
        component = getattr(model, role, None)
        if component is not None and hasattr(component, "numeric_terminal"):
            configuration[f"{role}.numeric_terminal"] = getattr(
                component.numeric_terminal, "value", component.numeric_terminal
            )
        if component is not None and hasattr(component, "ll_ridge"):
            configuration[f"{role}.ll_ridge"] = component.ll_ridge
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        tokenizer_version = getattr(model, "tokenizer_version", None)
        if tokenizer_version is None and inferred_id == "tabu.cell.base":
            tokenizer_version = "cell-tokenizer.v1"
        if tokenizer_version is not None:
            configuration["tokenizer_version"] = tokenizer_version
    readout = getattr(model, "readout", None)
    if readout is not None:
        terminal = getattr(readout, "numeric_terminal", None)
        configuration["terminal"] = getattr(terminal, "value", terminal)
        configuration["ll_ridge"] = getattr(readout, "ll_ridge", None)
    if config is not None:
        configuration["bandwidth"] = getattr(config, "routing_bandwidth", None)
    return ModelCompositionDescriptor(
        contract_id=inferred_id,
        contract_version=inferred_version,
        components=components or {"model": _component_name(model)},
        configuration=configuration,
        implementation_modules=modules,
    )


__all__ = ["describe_model"]
