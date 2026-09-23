"""One version dispatch for all shared curriculum execution paths."""

from __future__ import annotations

from .protocol import SCHEMA, V54_SCHEMA


def artifact_schema(schema: str, kind: str) -> str:
    if schema not in (SCHEMA, V54_SCHEMA):
        raise ValueError("unsupported curriculum schema")
    return f"{schema.removesuffix('.v1')}.{kind}.v1"


def make_model(plan):
    if plan.spec["schema"] == V54_SCHEMA:
        from tabu_lab.models.restoration_v54 import V54Model

        return V54Model(plan.config)
    if plan.spec["schema"] == SCHEMA:
        from tabu_lab.models.restoration_v53 import V53Model

        return V53Model(plan.config)
    raise ValueError("unsupported curriculum schema")
