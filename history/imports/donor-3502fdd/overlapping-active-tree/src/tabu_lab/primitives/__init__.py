"""Dense reference primitives for TabU-family models."""

from .mab import MAB, MABAttention, MABOutput
from .oattention import OAttention, OAttentionOutput, o_inject, presence_gate
from .omab import OMAB, OMABOutput
from .routing import (
    BilinearNumericLocalLinear,
    BilinearNumericNW,
    CategoricalReadoutOutput,
    GlobalUserItemNumericLocalLinear,
    GlobalUserItemNumericNW,
    NumericReadoutOutput,
    RoutingOutput,
    SameColumnCategoricalNW,
    SameColumnNumericLocalLinear,
    SameColumnNumericNW,
    categorical_from_routing,
    masked_rbf_weights,
)

__all__ = [
    "MAB",
    "OMAB",
    "BilinearNumericLocalLinear",
    "BilinearNumericNW",
    "CategoricalReadoutOutput",
    "GlobalUserItemNumericLocalLinear",
    "GlobalUserItemNumericNW",
    "MABAttention",
    "MABOutput",
    "NumericReadoutOutput",
    "OAttention",
    "OAttentionOutput",
    "OMABOutput",
    "RoutingOutput",
    "SameColumnCategoricalNW",
    "SameColumnNumericLocalLinear",
    "SameColumnNumericNW",
    "categorical_from_routing",
    "masked_rbf_weights",
    "o_inject",
    "presence_gate",
]
