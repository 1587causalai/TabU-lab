"""Dense reference primitives for TabU-family models."""

from .coupling import AffineCoupling, CouplingValueMap, inverse_perturbation_gain
from .mab import MAB, MABAttention, MABOutput
from .oattention import OAttention, OAttentionOutput, o_inject, presence_gate
from .omab import OMAB, OMABOutput
from .routing import (
    CategoricalReadoutOutput,
    NumericReadoutOutput,
    RoutingOutput,
    SameColumnNumericLocalLinear,
    SameColumnNumericNW,
    categorical_from_routing,
    masked_rbf_weights,
)

__all__ = [
    "MAB",
    "OMAB",
    "AffineCoupling",
    "CategoricalReadoutOutput",
    "CouplingValueMap",
    "MABAttention",
    "MABOutput",
    "NumericReadoutOutput",
    "OAttention",
    "OAttentionOutput",
    "OMABOutput",
    "RoutingOutput",
    "SameColumnNumericLocalLinear",
    "SameColumnNumericNW",
    "categorical_from_routing",
    "inverse_perturbation_gain",
    "masked_rbf_weights",
    "o_inject",
    "presence_gate",
]
