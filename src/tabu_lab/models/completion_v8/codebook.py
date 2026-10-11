"""V8.1: a learned feature vocabulary and non-trainable baseline directions.

Preparation stores indices and zero-origin codec templates, never a graph or
learned feature vectors. Binding reads the current bank once per forward.
The explicit input code_seed determines all draws, including partial recoding.
"""

from dataclasses import dataclass, replace

import torch
from torch import nn

from ..restoration._dtype import solve_dtype
from ..restoration._validation import finite
from ..restoration_v7.codec import G64Codec, _column_codec, select_softlog

POOL_SIZE = 10_000
CODE_DIM = 32
LEARNED_MODE = "learned_pool_v1"


def make_feature_bank():
    bank = nn.Embedding(POOL_SIZE, CODE_DIM)
    with torch.no_grad():
        # Embedding initializes iid N(0,1); unit length is initialization only.
        bank.weight.div_(bank.weight.norm(dim=-1, keepdim=True))
    return bank


@dataclass(frozen=True)
class FeatureCodebookPlan:
    indices: tuple[int, ...]
    template: G64Codec

    @property
    def dim(self):
        return self.template.dim

    @property
    def family(self):
        return self.template.family

    @property
    def numeric_preprocessing(self):
        return self.template.numeric_preprocessing

    def bind(self, bank: nn.Embedding):
        """Fresh differentiable gathers; this codec is one forward's snapshot."""
        if tuple(bank.weight.shape) != (POOL_SIZE, CODE_DIM):
            raise ValueError("feature bank must be [10000,32]")
        indices = torch.tensor(self.indices, device=bank.weight.device, dtype=torch.long)
        features = bank(indices)
        finite(features, "V8.1 feature codes")
        columns = []
        for base, template in zip(features, self.template.columns, strict=True):
            direction = None if template.direction is None else template.direction.to(base)
            if direction is not None:
                finite(direction, "V8.1 random direction")
                if not bool(direction.square().sum() > 0):
                    raise ValueError("zero numeric direction")
            columns.append(replace(
                template,
                base=base,
                direction=direction,
                codes=None if template.codes is None else base + template.codes.to(base),
                categories=(
                    None if template.categories is None else template.categories.to(base.device)
                ),
                rank_positions=(
                    None if template.rank_positions is None
                    else template.rank_positions.to(base.device)
                ),
            ))
        return replace(self.template, columns=tuple(columns))


def prepare_feature_plan(inputs, spec, *, recode_columns=None, reference_seed=0):
    """Preserve baseline draws/statistics at p=32; only the origin is learned.

    _column_codec reuses the constant-weight-8 or unit-Gaussian sampling rule
    without calling the public C64/8 constructor (which must remain 64D-only).
    Its random origin is discarded. Nominal candidates remain visible-only;
    ordinal candidates use the declared domain. No 10000-category limit is added.
    """
    if spec.codec not in ("C64/8", "G64"):
        raise ValueError("learned feature pool requires C64/8 or G64 baseline directions")
    generator = torch.Generator().manual_seed(inputs.code_seed)
    indices = tuple(torch.randint(POOL_SIZE, (len(inputs.schema),), generator=generator).tolist())
    dtype = solve_dtype(inputs.visible)
    epsilon = spec.epsilon if spec.numeric_preprocessing == "legacy" else 0.01
    columns = []
    for a, schema in enumerate(inputs.schema):
        column = _column_codec(
            schema, inputs.values[a][inputs.visible[:, a]], CODE_DIM, epsilon,
            generator, inputs.visible.device, dtype, spec.codec,
        )
        # Keep only random offsets/directions and fitted statistics in the plan.
        column = replace(
            column,
            codes=None if column.codes is None else (column.codes - column.base).detach(),
            base=torch.zeros_like(column.base),
            numeric_encoding=(
                "asinh" if schema.kind == "numeric"
                and spec.numeric_preprocessing == "standard_asinh_v1" else "affine"
            ),
        )
        columns.append(column)
    random_family = "C32/8" if spec.codec == "C64/8" else "G32"
    template = select_softlog(
        G64Codec(tuple(columns), CODE_DIM, f"{LEARNED_MODE}:{random_family}",
                 spec.numeric_preprocessing),
        inputs,
    )
    plan = FeatureCodebookPlan(indices, template)
    if recode_columns is None:
        return plan
    selected = tuple(recode_columns)
    if any(type(a) is not int or not 0 <= a < len(columns) for a in selected):
        raise ValueError("recode_columns must contain valid column indices")
    if len(selected) != len(set(selected)):
        raise ValueError("recode_columns must not repeat a column")
    if type(reference_seed) is not int:
        raise ValueError("reference_seed must be an integer")
    reference = prepare_feature_plan(replace(inputs, code_seed=reference_seed), spec)
    chosen = set(selected)
    return FeatureCodebookPlan(
        tuple(plan.indices[a] if a in chosen else reference.indices[a]
              for a in range(len(columns))),
        replace(template, columns=tuple(
            column if a in chosen else reference.template.columns[a]
            for a, column in enumerate(columns)
        )),
    )
