from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
import torch

from tabu_lab.contracts import FeatureKind, FeatureSpec
from tabu_lab.models import (
    CANONICAL_COMPONENTS,
    ComponentMaturity,
    ComponentRef,
    ComponentRegistry,
    ComponentRole,
    ComponentSpec,
    TabUBaseComponentManifest,
    TabUCellBaseModel,
    build_model,
    canonical_tabu_base_manifest,
    factory_dependency_hash,
    implementation_source_identity,
)
from tabu_lab.models.readouts import PairUnitReadout
from tabu_lab.models.types import DenseModelInput, ReferenceConfig
from tabu_lab.verification import (
    TabUBaseVerificationStage,
    TabUBaseVerificationStatus,
    inspect_tabu_base_composition,
    verify_tabu_base_component_extension,
)


class ExperimentalLocalLinearReadout(PairUnitReadout):
    """Minimal test extension with the existing readout interface."""


class BehaviorChangedReadout(ExperimentalLocalLinearReadout):
    """Distinct global target used to prove factory dependency rebinding."""


_FACTORY_RUNTIME = ExperimentalLocalLinearReadout


def _experimental_readout_factory(
    config: ReferenceConfig,
    options: Mapping[str, Any],
) -> ExperimentalLocalLinearReadout:
    return ExperimentalLocalLinearReadout(
        config,
        numeric_terminal=str(options["numeric_terminal"]),
    )


def _wrong_readout_factory(
    config: ReferenceConfig,
    options: Mapping[str, Any],
) -> PairUnitReadout:
    return PairUnitReadout(config, numeric_terminal=str(options["numeric_terminal"]))


def _global_runtime_factory(
    config: ReferenceConfig,
    options: Mapping[str, Any],
) -> ExperimentalLocalLinearReadout:
    return _FACTORY_RUNTIME(config, numeric_terminal=str(options["numeric_terminal"]))


def _indirect_readout_helper(
    config: ReferenceConfig,
    options: Mapping[str, Any],
) -> ExperimentalLocalLinearReadout:
    return ExperimentalLocalLinearReadout(
        config,
        numeric_terminal=str(options["numeric_terminal"]),
    )


def _indirect_helper_factory(
    config: ReferenceConfig,
    options: Mapping[str, Any],
) -> ExperimentalLocalLinearReadout:
    return _indirect_readout_helper(config, options)


def _config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=4,
    )


def _fixture() -> DenseModelInput:
    values = torch.tensor([[[0.0, 0.0], [2.0, 1.0], [3.0, 0.0]]])
    visible = torch.tensor([[[False, True], [True, True], [True, True]]])
    return DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=~visible,
        natural_missing_mask=torch.zeros_like(visible),
        feature_specs=(
            FeatureSpec(name="numeric"),
            FeatureSpec(
                name="category",
                kind=FeatureKind.CATEGORICAL,
                domain=("red", "blue"),
                codebook_id="component-extension.colors.v1",
            ),
        ),
        episode_id="tabubase-component-extension",
    )


def _extension_spec(
    factory=_experimental_readout_factory,
) -> ComponentSpec:
    implementation_ref, implementation_sha256 = implementation_source_identity(
        ExperimentalLocalLinearReadout
    )
    factory_ref, factory_sha256 = implementation_source_identity(factory)
    factory_dependency_sha256 = factory_dependency_hash(factory)
    return ComponentSpec(
        component_id="research.test.local-linear-readout",
        component_version="1.0.0",
        role=ComponentRole.READOUT,
        interface_id="tabu.cell-readout.v1",
        implementation_ref=implementation_ref,
        implementation_sha256=implementation_sha256,
        factory_ref=factory_ref,
        factory_sha256=factory_sha256,
        factory_dependency_sha256=factory_dependency_sha256,
        maturity=ComponentMaturity.EXPERIMENTAL,
        fixed_config={"numeric_terminal": "local_linear"},
    )


def _extension_registry_and_manifest():
    registry = CANONICAL_COMPONENTS.fork()
    spec = _extension_spec()
    registry.register(
        spec,
        _experimental_readout_factory,
        ExperimentalLocalLinearReadout,
    )
    manifest = replace(
        canonical_tabu_base_manifest(),
        readout=ComponentRef(spec.component_id, spec.component_version),
    )
    return registry, manifest


def test_registered_one_axis_extension_is_identity_bound_and_local_only() -> None:
    registry, manifest = _extension_registry_and_manifest()
    torch.manual_seed(1729)
    reference = build_model(
        "tabu.cell.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
    ).eval()
    torch.manual_seed(1729)
    candidate = build_model(
        "tabu.cell.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
        component_manifest=manifest,
        component_registry=registry,
    ).eval()
    candidate.load_state_dict(reference.state_dict())
    assert isinstance(candidate.readout, ExperimentalLocalLinearReadout)
    assert reference.variant_ref.semantic_hash != candidate.variant_ref.semantic_hash
    identity = candidate.checkpoint_identity()
    assert (
        identity["component_composition_hash"]
        == candidate.component_composition.composition_hash
    )
    assert identity["experimental_component_axes"] == ("readout",)

    fixture = _fixture()
    with torch.no_grad():
        reference_prediction = reference._forward_dense(fixture)
        candidate_prediction = candidate._forward_dense(fixture)
    assert (
        candidate_prediction.metadata["component_composition_hash"]
        == candidate.component_composition.composition_hash
    )
    evidence = verify_tabu_base_component_extension(
        reference_model=reference,
        candidate_model=candidate,
        reference_prediction=reference_prediction,
        candidate_prediction=candidate_prediction,
        expected_axis="readout",
    )
    assert evidence.stage is TabUBaseVerificationStage.COMPONENT_EXTENSION
    assert evidence.status is TabUBaseVerificationStatus.PASS
    assert evidence.evidence_status == "local_unissued"


def test_default_build_retains_legacy_identity_surface() -> None:
    model = build_model("tabu.cell.base", profile="completion.artificial_mask.v1")
    identity = model.checkpoint_identity()
    assert model.component_composition is None
    assert "component_composition_hash" not in identity
    assert "component_manifest_hash" not in identity

    explicit = build_model(
        "tabu.cell.base",
        profile="completion.artificial_mask.v1",
        component_manifest=canonical_tabu_base_manifest(),
    )
    explicit_identity = explicit.checkpoint_identity()
    assert explicit.component_composition is not None
    assert "component_composition_hash" in explicit_identity
    assert model.variant_ref.semantic_hash != explicit.variant_ref.semantic_hash


def test_registry_and_manifest_fail_closed() -> None:
    registry, manifest = _extension_registry_and_manifest()
    spec = _extension_spec()
    with pytest.raises(ValueError, match="cannot replace registered"):
        registry.register(spec, _experimental_readout_factory, ExperimentalLocalLinearReadout)

    canonical_ref = ComponentRef("tabu.readout.same-column-local-linear", "1.0.0")
    canonical_spec = registry.get(canonical_ref, expected_role=ComponentRole.READOUT)
    with pytest.raises(ValueError, match="experimental maturity"):
        registry.register(canonical_spec, _wrong_readout_factory, PairUnitReadout)

    unknown = replace(
        manifest,
        readout=ComponentRef("research.test.unknown-readout", "1.0.0"),
    )
    with pytest.raises(KeyError, match="unknown component"):
        build_model(
            "tabu.cell.base",
            profile="completion.artificial_mask.v1",
            component_manifest=unknown,
            component_registry=registry,
        )

    wrong_role = TabUBaseComponentManifest(
        tokenizer=manifest.tokenizer,
        dynamics=manifest.dynamics,
        readout=manifest.tokenizer,
    )
    with pytest.raises(ValueError, match="role"):
        build_model(
            "tabu.cell.base",
            profile="completion.artificial_mask.v1",
            component_manifest=wrong_role,
            component_registry=registry,
        )


def test_component_spec_rejects_wrong_source_identity_and_dual_authority() -> None:
    spec = replace(_extension_spec(), implementation_sha256="0" * 64)
    registry = CANONICAL_COMPONENTS.fork()
    with pytest.raises(ValueError, match="source identity"):
        registry.register(spec, _experimental_readout_factory, ExperimentalLocalLinearReadout)

    wrong_factory_registry = CANONICAL_COMPONENTS.fork()
    valid_spec = _extension_spec(factory=_wrong_readout_factory)
    wrong_factory_registry.register(
        valid_spec,
        _wrong_readout_factory,
        ExperimentalLocalLinearReadout,
    )
    with pytest.raises(TypeError, match="wrong runtime type"):
        wrong_factory_registry.build(
            ComponentRef(valid_spec.component_id, valid_spec.component_version),
            expected_role=ComponentRole.READOUT,
            config=_config(),
        )

    registry, manifest = _extension_registry_and_manifest()
    with pytest.raises(TypeError, match="only component-selection authority"):
        build_model(
            "tabu.cell.base",
            profile="completion.artificial_mask.v1",
            numeric_terminal="local_linear",
            component_manifest=manifest,
            component_registry=registry,
        )


def test_fresh_registry_cannot_forge_canonical_authority() -> None:
    authority_ref = ComponentRef("tabu.readout.same-column-local-linear", "1.0.0")
    canonical_spec = CANONICAL_COMPONENTS.get(
        authority_ref,
        expected_role=ComponentRole.READOUT,
    )
    fresh = ComponentRegistry()
    spoofed = replace(
        canonical_spec,
        fixed_config={"numeric_terminal": "nadaraya_watson"},
    )
    with pytest.raises(ValueError, match="experimental maturity"):
        fresh.register(spoofed, _wrong_readout_factory, PairUnitReadout)

    extension_spec = _extension_spec()
    fresh.register(
        extension_spec,
        _experimental_readout_factory,
        ExperimentalLocalLinearReadout,
    )
    manifest = replace(
        canonical_tabu_base_manifest(),
        readout=ComponentRef(extension_spec.component_id, extension_spec.component_version),
    )
    with pytest.raises(ValueError, match="canonical anchor"):
        build_model(
            "tabu.cell.base",
            profile="completion.artificial_mask.v1",
            component_manifest=manifest,
            component_registry=fresh,
        )


def test_runtime_inspection_rebinds_actual_module_to_registered_spec() -> None:
    registry, manifest = _extension_registry_and_manifest()
    candidate = build_model(
        "tabu.cell.base",
        config=_config(),
        profile="completion.artificial_mask.v1",
        component_manifest=manifest,
        component_registry=registry,
    )
    forged = TabUCellBaseModel(
        _config(),
        profile="completion.artificial_mask.v1",
        _component_tokenizer=candidate.tokenizer,
        _component_dynamics=candidate.dynamics,
        _component_readout=PairUnitReadout(_config(), numeric_terminal="local_linear"),
        _component_composition=candidate.component_composition,
        _component_registry=registry,
    )

    with pytest.raises(TypeError, match="runtime readout type"):
        inspect_tabu_base_composition(forged)


def test_component_ref_config_is_recursively_copied_and_frozen() -> None:
    supplied = {"nested": {"values": [1, 2]}}
    ref = ComponentRef("research.test.deep-config", "1.0.0", config=supplied)
    supplied["nested"]["values"].append(3)

    assert ref.config["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        ref.config["nested"]["values"][0] = 9


def test_factory_global_dependency_is_rechecked_before_build() -> None:
    global _FACTORY_RUNTIME

    registry = CANONICAL_COMPONENTS.fork()
    spec = _extension_spec(factory=_global_runtime_factory)
    registry.register(spec, _global_runtime_factory, ExperimentalLocalLinearReadout)
    ref = ComponentRef(spec.component_id, spec.component_version)
    built = registry.build(ref, expected_role=ComponentRole.READOUT, config=_config())
    assert type(built) is ExperimentalLocalLinearReadout

    _FACTORY_RUNTIME = BehaviorChangedReadout
    try:
        with pytest.raises(ValueError, match="factory identity drifted"):
            registry.build(ref, expected_role=ComponentRole.READOUT, config=_config())
    finally:
        _FACTORY_RUNTIME = ExperimentalLocalLinearReadout


def test_factory_indirect_helper_dependency_is_rejected() -> None:
    with pytest.raises(ValueError, match="indirect helper function"):
        _extension_spec(factory=_indirect_helper_factory)
