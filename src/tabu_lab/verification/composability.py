"""Discriminating Stage-2 checks for TabUBase component substitution.

This module observes the already-public model and prediction contracts.  It
does not add a new component injection path or change ``tabu.cell.base@0.2.0``
identity.  Its job is narrower: prove that one declared built-in component axis
can change while the other axes and the public forward interface stay fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from tabu_lab.contracts import PredictionBundle, canonical_hash
from tabu_lab.models.table_cell import TabUCellBaseModel

_COMPONENT_AXES = ("tokenizer", "dynamics", "readout")


class SubstitutionStatus(StrEnum):
    """Result of one bounded, one-axis component substitution check."""

    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class TabUBaseComposition:
    """Stable semantic names for the components used by one built model."""

    contract_id: str
    contract_version: str
    profile_id: str
    tokenizer: str
    dynamics: str
    readout: str
    supervision_route: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def composition_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def changed_axes(self, other: TabUBaseComposition) -> tuple[str, ...]:
        """Return changed component axes after checking the comparison boundary."""

        if (self.contract_id, self.contract_version) != (
            other.contract_id,
            other.contract_version,
        ):
            raise ValueError("component substitution requires the same model contract")
        if self.profile_id != other.profile_id:
            raise ValueError("component substitution requires the same evidence profile")
        if self.supervision_route != other.supervision_route:
            raise ValueError("component substitution cannot change the supervision route")
        return tuple(
            axis for axis in _COMPONENT_AXES if getattr(self, axis) != getattr(other, axis)
        )


@dataclass(frozen=True, slots=True)
class PredictionEntrySignature:
    """Shape-and-type surface of one public prediction entry."""

    name: str
    kind: str
    status: str
    value_shape: tuple[int, ...] | None
    support_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "value_shape": self.value_shape,
            "support_shape": self.support_shape,
        }


@dataclass(frozen=True, slots=True)
class ForwardInterfaceSignature:
    """Public output schema, intentionally excluding numerical predictions."""

    contract_version: str
    entries: tuple[PredictionEntrySignature, ...]
    auxiliaries: tuple[tuple[str, tuple[int, ...]], ...]
    trace_emitted: bool

    @classmethod
    def from_prediction(cls, prediction: PredictionBundle) -> ForwardInterfaceSignature:
        if not isinstance(prediction, PredictionBundle):
            raise TypeError("prediction must be a PredictionBundle")
        entries = tuple(
            PredictionEntrySignature(
                name=name,
                kind=entry.kind.value,
                status=entry.status.value,
                value_shape=None if entry.values is None else tuple(entry.values.shape),
                support_shape=tuple(entry.support_ids.shape),
            )
            for name, entry in sorted(prediction.entries.items())
        )
        auxiliaries = tuple(
            (name, tuple(value.shape)) for name, value in sorted(prediction.auxiliaries.items())
        )
        return cls(
            contract_version=prediction.contract_version,
            entries=entries,
            auxiliaries=auxiliaries,
            trace_emitted=prediction.trace is not None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "entries": [entry.as_dict() for entry in self.entries],
            "auxiliaries": [{"name": name, "shape": shape} for name, shape in self.auxiliaries],
            "trace_emitted": self.trace_emitted,
        }

    @property
    def interface_hash(self) -> str:
        return canonical_hash(self.as_dict())


@dataclass(frozen=True, slots=True)
class SubstitutionAssessment:
    """Local Stage-2 result for exactly one expected component-axis change."""

    expected_axis: str
    changed_axes: tuple[str, ...]
    interface_stable: bool
    variant_identity_changed: bool
    reference_composition_hash: str
    candidate_composition_hash: str
    reference_variant_hash: str
    candidate_variant_hash: str
    status: SubstitutionStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_axis": self.expected_axis,
            "changed_axes": self.changed_axes,
            "interface_stable": self.interface_stable,
            "variant_identity_changed": self.variant_identity_changed,
            "reference_composition_hash": self.reference_composition_hash,
            "candidate_composition_hash": self.candidate_composition_hash,
            "reference_variant_hash": self.reference_variant_hash,
            "candidate_variant_hash": self.candidate_variant_hash,
            "status": self.status.value,
        }

    @property
    def assessment_hash(self) -> str:
        return canonical_hash(self.as_dict())


def inspect_tabu_base_composition(model: TabUCellBaseModel) -> TabUBaseComposition:
    """Read component-plan identity from an already-built canonical model."""

    if not isinstance(model, TabUCellBaseModel):
        raise TypeError("model must be a TabUCellBaseModel")
    plan = getattr(model.dynamics, "plan", None)
    if plan is None or not callable(getattr(plan, "resolved_name", None)):
        raise ValueError("TabUBase dynamics must expose a resolvable plan")
    tokenizer = model.tokenizer_metadata.get("tokenizer_version")
    terminal = getattr(model.readout, "numeric_terminal", None)
    if not isinstance(tokenizer, str) or not isinstance(terminal, str):
        raise ValueError("TabUBase components must expose stable semantic names")
    return TabUBaseComposition(
        contract_id=model.model_id,
        contract_version=model.variant_ref.contract_version,
        profile_id=model.profile.value,
        tokenizer=tokenizer,
        dynamics=plan.resolved_name(model.config.block_kind),
        readout=f"same_column.{terminal}",
        supervision_route=("label_broadcast.v1" if model.label_broadcast else "none"),
    )


def assess_tabu_base_substitution(
    *,
    reference_model: TabUCellBaseModel,
    candidate_model: TabUCellBaseModel,
    reference_prediction: PredictionBundle,
    candidate_prediction: PredictionBundle,
    expected_axis: str,
) -> SubstitutionAssessment:
    """Check one semantic component change without comparing prediction quality."""

    if expected_axis not in _COMPONENT_AXES:
        raise ValueError(f"expected_axis must be one of {_COMPONENT_AXES!r}")
    reference = inspect_tabu_base_composition(reference_model)
    candidate = inspect_tabu_base_composition(candidate_model)
    changed_axes = reference.changed_axes(candidate)
    reference_interface = ForwardInterfaceSignature.from_prediction(reference_prediction)
    candidate_interface = ForwardInterfaceSignature.from_prediction(candidate_prediction)
    interface_stable = reference_interface == candidate_interface
    reference_variant_hash = reference_model.variant_ref.semantic_hash
    candidate_variant_hash = candidate_model.variant_ref.semantic_hash
    variant_identity_changed = reference_variant_hash != candidate_variant_hash
    passed = changed_axes == (expected_axis,) and interface_stable and variant_identity_changed
    return SubstitutionAssessment(
        expected_axis=expected_axis,
        changed_axes=changed_axes,
        interface_stable=interface_stable,
        variant_identity_changed=variant_identity_changed,
        reference_composition_hash=reference.composition_hash,
        candidate_composition_hash=candidate.composition_hash,
        reference_variant_hash=reference_variant_hash,
        candidate_variant_hash=candidate_variant_hash,
        status=SubstitutionStatus.PASS if passed else SubstitutionStatus.FAIL,
    )


__all__ = [
    "ForwardInterfaceSignature",
    "SubstitutionAssessment",
    "SubstitutionStatus",
    "TabUBaseComposition",
    "assess_tabu_base_substitution",
    "inspect_tabu_base_composition",
]
