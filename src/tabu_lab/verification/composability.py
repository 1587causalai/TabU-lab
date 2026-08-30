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
from tabu_lab.models.component_contract import (
    TabUBaseComposition,
    inspect_tabu_base_composition,
)
from tabu_lab.models.table_cell import TabUCellBaseModel

_COMPONENT_AXES = ("tokenizer", "dynamics", "readout")
_TOKENIZER_IDENTITY_KEYS = (
    "tokenizer_version",
    "nominal_tokenizer",
    "nominal_codebook_size",
    "nominal_codebook_seed",
    "nominal_codebook_hash",
)


class SubstitutionStatus(StrEnum):
    """Result of one bounded, one-axis component substitution check."""

    PASS = "pass"
    FAIL = "fail"


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
    predictions_bound: bool
    input_evidence_matched: bool
    components_declared: bool
    non_target_config_stable: bool
    variant_identity_changed: bool
    reference_composition_hash: str
    candidate_composition_hash: str
    reference_variant_hash: str
    candidate_variant_hash: str
    reference_non_target_hash: str
    candidate_non_target_hash: str
    status: SubstitutionStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "expected_axis": self.expected_axis,
            "changed_axes": self.changed_axes,
            "interface_stable": self.interface_stable,
            "predictions_bound": self.predictions_bound,
            "input_evidence_matched": self.input_evidence_matched,
            "components_declared": self.components_declared,
            "non_target_config_stable": self.non_target_config_stable,
            "variant_identity_changed": self.variant_identity_changed,
            "reference_composition_hash": self.reference_composition_hash,
            "candidate_composition_hash": self.candidate_composition_hash,
            "reference_variant_hash": self.reference_variant_hash,
            "candidate_variant_hash": self.candidate_variant_hash,
            "reference_non_target_hash": self.reference_non_target_hash,
            "candidate_non_target_hash": self.candidate_non_target_hash,
            "status": self.status.value,
        }

    @property
    def assessment_hash(self) -> str:
        return canonical_hash(self.as_dict())


def _non_target_identity(model: TabUCellBaseModel, expected_axis: str) -> dict[str, Any]:
    """Normalize checkpoint identity by removing only the declared target axis."""

    identity = dict(model.checkpoint_identity())
    identity.pop("variant_hash")
    identity.pop("variant_ref")
    if expected_axis == "tokenizer":
        for key in _TOKENIZER_IDENTITY_KEYS:
            identity.pop(key, None)
    elif expected_axis == "dynamics":
        reference_config = dict(identity["reference_config"])
        reference_config.pop("block_kind")
        identity["reference_config"] = reference_config
    else:
        identity.pop("terminal")
        identity.pop("ll_ridge")
    return identity


def _prediction_is_bound_to_model(
    prediction: PredictionBundle,
    model: TabUCellBaseModel,
) -> bool:
    """Require an emitted trace and exact semantic variant identity."""

    trace = prediction.trace
    return bool(
        prediction.model_id == model.model_id
        and prediction.contract_version == model.contract_version
        and prediction.metadata.get("variant_hash") == model.variant_ref.semantic_hash
        and prediction.metadata.get("variant_ref") == model.variant_ref.as_dict()
        and prediction.metadata.get("profile_id") == model.profile.value
        and trace is not None
        and trace.model_id == model.model_id
        and trace.metadata.get("variant_hash") == model.variant_ref.semantic_hash
        and trace.metadata.get("profile_id") == model.profile.value
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
    predictions_bound = _prediction_is_bound_to_model(
        reference_prediction,
        reference_model,
    ) and _prediction_is_bound_to_model(candidate_prediction, candidate_model)
    input_evidence_matched = bool(
        reference_prediction.episode_id == candidate_prediction.episode_id
        and reference_prediction.trace is not None
        and candidate_prediction.trace is not None
        and reference_prediction.trace.input_hash == candidate_prediction.trace.input_hash
    )
    components_declared = (
        reference.declaration_status == "model_spec_declared"
        and candidate.declaration_status == "model_spec_declared"
    )
    reference_variant_hash = reference_model.variant_ref.semantic_hash
    candidate_variant_hash = candidate_model.variant_ref.semantic_hash
    variant_identity_changed = reference_variant_hash != candidate_variant_hash
    reference_non_target_hash = canonical_hash(_non_target_identity(reference_model, expected_axis))
    candidate_non_target_hash = canonical_hash(_non_target_identity(candidate_model, expected_axis))
    non_target_config_stable = reference_non_target_hash == candidate_non_target_hash
    passed = (
        changed_axes == (expected_axis,)
        and interface_stable
        and predictions_bound
        and input_evidence_matched
        and components_declared
        and non_target_config_stable
        and variant_identity_changed
    )
    return SubstitutionAssessment(
        expected_axis=expected_axis,
        changed_axes=changed_axes,
        interface_stable=interface_stable,
        predictions_bound=predictions_bound,
        input_evidence_matched=input_evidence_matched,
        components_declared=components_declared,
        non_target_config_stable=non_target_config_stable,
        variant_identity_changed=variant_identity_changed,
        reference_composition_hash=reference.composition_hash,
        candidate_composition_hash=candidate.composition_hash,
        reference_variant_hash=reference_variant_hash,
        candidate_variant_hash=candidate_variant_hash,
        reference_non_target_hash=reference_non_target_hash,
        candidate_non_target_hash=candidate_non_target_hash,
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
