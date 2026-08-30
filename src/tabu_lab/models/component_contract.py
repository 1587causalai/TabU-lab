"""Resolved component identity for the canonical TabUBase contract.

The ModelSpec remains the semantic authority.  This module does not introduce
an independent component configuration or a second registry; it resolves the
choices already permitted by ``tabu.cell.base@0.2.0`` and verifies that the
built runtime uses the corresponding concrete components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tabu_lab.contracts import canonical_hash
from tabu_lab.primitives import SameColumnNumericLocalLinear, SameColumnNumericNW

from .components import CellTokenizer
from .dynamics import CellUnitDynamics
from .readouts import PairUnitReadout

_CONTRACT_ID = "tabu.cell.base"
_CONTRACT_VERSION = "0.2.0"
_COMPONENT_AXES = ("tokenizer", "dynamics", "readout")
_TOKENIZER_IDS = {
    CellTokenizer.EPISODE_RANDOM_SPHERE_V1: "cell-tokenizer.v1",
    CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2: "cell-tokenizer.v2",
}
_NUMERIC_TERMINALS = frozenset({"local_linear", "nadaraya_watson"})
_NUMERIC_TERMINAL_TYPES = {
    "local_linear": SameColumnNumericLocalLinear,
    "nadaraya_watson": SameColumnNumericNW,
}


@dataclass(frozen=True, slots=True)
class TabUBaseComposition:
    """Content-addressed binding from ModelSpec to one built composition."""

    schema_version: str
    contract_id: str
    contract_version: str
    model_spec_hash: str
    profile_id: str
    unit_semantics: str
    tokenizer: str
    dynamics: str
    readout: str
    supervision_route: str
    truth_boundary: str
    declaration_status: str
    registry_composition_hash: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.model_spec_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_spec_hash
        ):
            raise ValueError("model_spec_hash must be a lowercase SHA-256")

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def composition_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def changed_axes(self, other: TabUBaseComposition) -> tuple[str, ...]:
        """Return changed component axes inside one fixed semantic boundary."""

        fixed_fields = (
            "schema_version",
            "contract_id",
            "contract_version",
            "model_spec_hash",
            "profile_id",
            "unit_semantics",
            "supervision_route",
            "truth_boundary",
        )
        if any(getattr(self, field) != getattr(other, field) for field in fixed_fields):
            raise ValueError("component substitution crossed the TabUBase contract boundary")
        return tuple(
            axis for axis in _COMPONENT_AXES if getattr(self, axis) != getattr(other, axis)
        )


def _require_contract_semantics(spec: Any) -> None:
    """Reject a registered contract that no longer states TabUBase semantics."""

    if (spec.contract_id, spec.contract_version) != (_CONTRACT_ID, _CONTRACT_VERSION):
        raise ValueError("component binding requires tabu.cell.base@0.2.0")
    carrier = spec.carrier
    if (
        carrier.get("family") != "tabu.table_cell_as_unit"
        or carrier.get("unit") != "cell"
        or carrier.get("shape") != "N x M x d"
        or carrier.get("specials") != "none"
    ):
        raise ValueError("ModelSpec no longer declares the canonical cell-Unit carrier")
    if spec.dynamics.get("family") != "cell_unit_three_omab":
        raise ValueError("ModelSpec no longer declares the canonical TabUBase dynamics")
    if spec.readout.get("support") != "same-feature visible cells":
        raise ValueError("ModelSpec no longer declares same-feature visible support")
    if spec.loss.get("truth_entry") != "Step 5 only":
        raise ValueError("ModelSpec no longer keeps truth at the Step-5 sidecar boundary")


def resolve_tabu_base_composition(spec: Any, model: Any) -> TabUBaseComposition:
    """Validate and resolve one ModelSpec → builder → components binding."""

    from tabu_lab.registry import ModelSpec, model_spec_identity_payload

    if not isinstance(spec, ModelSpec):
        raise TypeError("spec must be a typed ModelSpec")
    _require_contract_semantics(spec)
    if getattr(model, "model_id", None) != spec.contract_id:
        raise ValueError("built model id does not match the ModelSpec")
    if getattr(model, "contract_version", None) != spec.contract_version:
        raise ValueError("built model version does not match the ModelSpec")
    model_spec_hash = canonical_hash(model_spec_identity_payload(spec))
    if getattr(model, "model_spec_hash", None) != model_spec_hash:
        raise ValueError("built model does not carry the exact ModelSpec identity")

    if not isinstance(getattr(model, "tokenizer", None), CellTokenizer):
        raise TypeError("TabUBase tokenizer must be CellTokenizer")
    nominal_tokenizer = model.tokenizer.nominal_tokenizer
    try:
        tokenizer_id = _TOKENIZER_IDS[nominal_tokenizer]
    except KeyError as exc:
        raise ValueError("TabUBase selected an undeclared tokenizer") from exc
    if model.tokenizer_metadata.get("tokenizer_version") != tokenizer_id:
        raise ValueError("tokenizer metadata disagrees with the concrete tokenizer")
    if nominal_tokenizer == CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2:
        alternatives = {alternative.id: alternative for alternative in spec.alternatives}
        if "nominal_source_scoped_frozen_codebook_v2" not in alternatives:
            raise ValueError("ModelSpec does not declare the v2 tokenizer alternative")

    if not isinstance(getattr(model, "dynamics", None), CellUnitDynamics):
        raise TypeError("TabUBase dynamics must be CellUnitDynamics")
    plan = model.dynamics.plan
    if plan.name != spec.dynamics["family"]:
        raise ValueError("runtime dynamics plan disagrees with the ModelSpec")
    dynamics_id = plan.resolved_name(model.config.block_kind)
    declaration_status = (
        "model_spec_declared"
        if dynamics_id == spec.dynamics["family"]
        else "code_only_non_o_ablation"
    )

    if not isinstance(getattr(model, "readout", None), PairUnitReadout):
        raise TypeError("TabUBase readout must be PairUnitReadout")
    numeric_terminal = model.readout.numeric_terminal
    if numeric_terminal not in _NUMERIC_TERMINALS:
        raise ValueError("TabUBase selected an undeclared numeric terminal")
    expected_terminal_type = _NUMERIC_TERMINAL_TYPES[numeric_terminal]
    if not isinstance(model.readout.terminal, expected_terminal_type):
        raise TypeError("numeric terminal label disagrees with the concrete terminal")

    profiles = spec.experimental_defaults.get("profiles", {})
    profile_id = model.profile.value
    if profile_id not in profiles:
        raise ValueError("built model selected a profile absent from the ModelSpec")
    declared_broadcast = profiles[profile_id].get("label_broadcast")
    if type(declared_broadcast) is not bool or declared_broadcast != model.label_broadcast:
        raise ValueError("runtime supervision route disagrees with the ModelSpec profile")

    resolved_components = getattr(model, "component_composition", None)
    if resolved_components is not None:
        from .component_registry import CANONICAL_COMPONENTS, ComponentRegistry

        component_registry = getattr(model, "component_registry", None)
        if not isinstance(component_registry, ComponentRegistry):
            raise TypeError("resolved composition requires its ComponentRegistry")
        component_registry.assert_extends(CANONICAL_COMPONENTS)
        component_registry.validate_runtime(
            resolved_components,
            {
                "tokenizer": model.tokenizer,
                "dynamics": model.dynamics,
                "readout": model.readout,
            },
        )
        manifest = resolved_components.manifest
        tokenizer_id = {
            "tabu.tokenizer.cell@1.0.0": "cell-tokenizer.v1",
            "tabu.tokenizer.cell@2.0.0": "cell-tokenizer.v2",
        }.get(manifest.tokenizer.spec_ref, manifest.tokenizer.spec_ref)
        dynamics_id = {
            "tabu.dynamics.cell-unit-three-omab@1.0.0": "cell_unit_three_omab",
        }.get(manifest.dynamics.spec_ref, manifest.dynamics.spec_ref)
        numeric_readout_id = {
            "tabu.readout.same-column-local-linear@1.0.0": "same_column.local_linear",
            "tabu.readout.same-column-nadaraya-watson@1.0.0": (
                "same_column.nadaraya_watson"
            ),
        }.get(manifest.readout.spec_ref, manifest.readout.spec_ref)
        authoritative = all(
            component_registry.is_authoritative(ref, CANONICAL_COMPONENTS)
            for ref in (manifest.tokenizer, manifest.dynamics, manifest.readout)
            if ref.spec_ref
        )
        if resolved_components.experimental_axes:
            declaration_status = "component_spec_experimental"
        elif authoritative:
            declaration_status = "model_spec_declared"
        else:
            declaration_status = "component_registry_untrusted"
    else:
        numeric_readout_id = f"same_column.{numeric_terminal}"

    registry_composition_hash = (
        resolved_components.composition_hash
        if resolved_components is not None
        else canonical_hash(
            {
                "schema": "tabu.legacy-derived-components.v1",
                "tokenizer": tokenizer_id,
                "dynamics": dynamics_id,
                "readout": numeric_readout_id,
            }
        )
    )

    return TabUBaseComposition(
        schema_version="tabu.tabubase-components.v1",
        contract_id=spec.contract_id,
        contract_version=spec.contract_version,
        model_spec_hash=model_spec_hash,
        profile_id=profile_id,
        unit_semantics="table_cell_as_unit",
        tokenizer=tokenizer_id,
        dynamics=dynamics_id,
        readout=numeric_readout_id,
        supervision_route="label_broadcast.v1" if model.label_broadcast else "none",
        truth_boundary="loss_sidecar_step_5_only",
        declaration_status=declaration_status,
        registry_composition_hash=registry_composition_hash,
    )


def inspect_tabu_base_composition(model: Any) -> TabUBaseComposition:
    """Resolve a built model against its exact packaged ModelSpec."""

    from tabu_lab.registry import get_model_spec

    model_id = getattr(model, "model_id", None)
    contract_version = getattr(model, "contract_version", None)
    if model_id != _CONTRACT_ID or contract_version != _CONTRACT_VERSION:
        raise TypeError("model must be a TabUCellBaseModel for tabu.cell.base@0.2.0")
    spec = get_model_spec(model_id, contract_version)
    return resolve_tabu_base_composition(spec, model)


__all__ = [
    "TabUBaseComposition",
    "inspect_tabu_base_composition",
    "resolve_tabu_base_composition",
]
