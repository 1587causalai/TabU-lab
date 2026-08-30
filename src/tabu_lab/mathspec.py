"""Typed mathematical content embedded in a ModelSpec.

The model registry owns semantic identity and implementation boundaries.  This module
adds a deliberately small, structured vocabulary for the mathematical projection of a
contract.  Equations remain authored LaTeX, while their order, names, meanings, and
invariants become machine-checkable data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ID = r"^[a-z][a-z0-9_.-]*$"


class MathSymbol(BaseModel):
    """One named symbol in the contract's notation table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    symbol: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    domain: str | None = Field(
        default=None,
        min_length=1,
        description="Optional raw LaTeX domain expression.",
    )


class MathEquation(BaseModel):
    """One display equation with a stable reference and plain-language meaning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    latex: str = Field(min_length=1)
    meaning: str = Field(min_length=1)


class MathStep(BaseModel):
    """An ordered mathematical stage in the generated model explanation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    title: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    equations: tuple[MathEquation, ...] = ()
    invariants: tuple[str, ...] = ()


class MathInvariant(BaseModel):
    """A falsifiable statement attached to the mathematical projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID)
    statement: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class Mathematics(BaseModel):
    """Structured mathematical narrative for one model contract.

    The renderer intentionally accepts raw LaTeX only in ``symbol``, ``domain``,
    and ``latex``. Human-facing prose is escaped before it enters the generated
    TeX document.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"]
    abstract: str = Field(min_length=1)
    unit_semantics: str = Field(min_length=1)
    notation: tuple[MathSymbol, ...] = Field(min_length=1)
    steps: tuple[MathStep, ...] = Field(min_length=1)
    invariants: tuple[MathInvariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_stable_ids(self) -> Mathematics:
        """Prevent ambiguous references in generated TeX and future receipts."""

        namespaces = {
            "notation": [item.id for item in self.notation],
            "steps": [item.id for item in self.steps],
            "equations": [equation.id for step in self.steps for equation in step.equations],
            "invariants": [item.id for item in self.invariants],
        }
        for namespace, identifiers in namespaces.items():
            duplicates = sorted(
                {identifier for identifier in identifiers if identifiers.count(identifier) > 1}
            )
            if duplicates:
                raise ValueError(f"{namespace} ids must be unique: {', '.join(duplicates)}")
        return self
