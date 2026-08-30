from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pydantic import BaseModel

from tabu_lab.contracts import (
    FeatureKind,
    FeatureRole,
    FeatureSpec,
    TruthSidecar,
    assert_truth_free,
    canonical_hash,
)
from tabu_lab.models import (
    MODEL_BUILDERS,
    TabUCellBaseModel,
    TabUCellBaseProfile,
    build_from_spec,
    build_model,
)
from tabu_lab.models.components import CellTokenizer, Symbolizer
from tabu_lab.models.table_cell import _label_broadcast
from tabu_lab.models.types import DenseModelInput, DynamicsBlockKind, ReferenceConfig
from tabu_lab.registry import (
    BuildStatus,
    ModelVersionNotFoundError,
    get_model_spec,
    model_spec_identity_payload,
    validate_registry_source_parity,
)
from tabu_lab.registry import build_model as build_contract


def _completion_input() -> DenseModelInput:
    values = torch.tensor([[[1.0, 0.0, 3.0], [2.0, 4.0, 0.0], [5.0, 6.0, 7.0]]])
    visible = values != 0
    target = ~visible
    return DenseModelInput(values, visible, target, torch.zeros_like(target))


def _supervised_input() -> DenseModelInput:
    values = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 4.0, 1.0], [5.0, 6.0, 0.0]]])
    visible = torch.tensor([[[True, True, False], [True, True, True], [True, True, False]]])
    query = ~visible
    return DenseModelInput(
        values,
        visible,
        query,
        torch.zeros_like(query),
        artificial_target_mask=torch.zeros_like(query),
        query_target_mask=query,
        feature_specs=(
            FeatureSpec(name="x0"),
            FeatureSpec(name="x1"),
            FeatureSpec(name="y", role=FeatureRole.RESPONSE),
        ),
    )


def _small_config(*, block_kind: DynamicsBlockKind = DynamicsBlockKind.OMAB) -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=8,
        block_kind=block_kind,
    )


def test_v020_profiles_are_explicit_and_identity_bound() -> None:
    completion = build_model(
        "tabu.cell.base", profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1
    )
    supervised = build_model(
        "tabu.cell.base", profile=TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
    )
    assert completion.profile is TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1
    assert supervised.profile is TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1
    assert completion.variant_ref.contract_version == "0.2.0"
    assert completion.variant_ref.semantic_hash != supervised.variant_ref.semantic_hash
    identity = supervised.checkpoint_identity()
    assert identity["profile_id"] == "supervised.label_broadcast.v1"
    assert identity["tokenizer_version"] == "cell-tokenizer.v1"
    assert identity["label_broadcast"] is True
    assert identity["reference_config"]["block_kind"] == "omab"
    assert identity["terminal"] == supervised.readout.numeric_terminal
    assert identity["bandwidth"] == supervised.config.routing_bandwidth
    nw = build_model(
        "tabu.cell.base",
        profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
        numeric_terminal="nadaraya_watson",
    )
    assert completion.variant_ref.semantic_hash != nw.variant_ref.semantic_hash
    with pytest.raises(ValueError, match="derived"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            label_broadcast=True,
        )
    with pytest.raises(TypeError, match="variant_ref"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            variant_ref=replace(
                completion.variant_ref,
                source_identity="forged-approved-receipt",
            ),
        )
    with pytest.raises(ValueError, match="canonical label_broadcast_tau"):
        build_model(
            "tabu.cell.base",
            profile=TabUCellBaseProfile.COMPLETION_ARTIFICIAL_MASK_V1,
            label_broadcast_tau=9.0,
        )
    supervised_custom_tau = build_model(
        "tabu.cell.base",
        profile=TabUCellBaseProfile.SUPERVISED_LABEL_BROADCAST_V1,
        label_broadcast_tau=9.0,
    )
    assert supervised.variant_ref.semantic_hash != supervised_custom_tau.variant_ref.semantic_hash
    with pytest.raises(ValueError, match="unexpected fields"):
        completion.validate_checkpoint_identity({**completion.checkpoint_identity(), "extra": True})


def test_registry_build_is_typed_versioned_and_yaml_bound() -> None:
    spec = get_model_spec("tabu.cell.base", "0.2.0")
    repository_root = Path(__file__).resolve().parents[2]
    public_manifest = repository_root / "specs" / "model-factory-source-manifest.json"
    packaged_manifest = repository_root / "src" / "tabu_lab" / "specs" / public_manifest.name
    assert public_manifest.read_bytes() == packaged_manifest.read_bytes()
    manifest = json.loads(public_manifest.read_text())
    source = manifest["contracts"]["tabu.cell.base"]
    closure_payload = json.dumps(
        source["semantic_source_closure"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    assert source["entrypoint_sha256"] == spec.upstream.sha256
    assert hashlib.sha256(closure_payload).hexdigest() == source["semantic_source_tree_sha256"]
    assert source["semantic_source_tree_sha256"] == spec.upstream.semantic_source_tree_sha256
    result = build_contract(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )
    assert result.status is BuildStatus.READY
    assert result.executable
    assert isinstance(result.model, TabUCellBaseModel)
    assert result.model.model_spec_hash == canonical_hash(model_spec_identity_payload(spec))
    exact = build_from_spec(
        spec,
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )
    assert exact.model_spec_hash == canonical_hash(model_spec_identity_payload(spec))
    with pytest.raises(TypeError, match="typed ModelSpec"):
        build_from_spec(
            spec.model_dump(mode="json"),
            config=_small_config(),
            profile="completion.artificial_mask.v1",
        )
    with pytest.raises(ModelVersionNotFoundError):
        build_from_spec(
            spec.model_copy(update={"contract_version": "999.0.0"}),
            config=_small_config(),
            profile="completion.artificial_mask.v1",
        )
    tampered = spec.model_copy(
        update={"upstream": spec.upstream.model_copy(update={"sha256": "0" * 64})}
    )
    with pytest.raises(ValueError, match="exactly match"):
        build_from_spec(
            tampered,
            config=_small_config(),
            profile="completion.artificial_mask.v1",
        )
    missing_profile = build_contract("tabu.cell.base", config=_small_config())
    assert missing_profile.status is BuildStatus.BUILD_ERROR
    assert not missing_profile.executable
    assert "explicit profile" in (missing_profile.detail or "")
    validate_registry_source_parity()
    with pytest.raises(ModelVersionNotFoundError):
        get_model_spec("tabu.cell.base", "0.1.0")
    with pytest.raises(TypeError, match="unknown table-cell base builder options"):
        build_model(
            "tabu.cell.base",
            profile="supervised.label_broadcast.v1",
            label_columns=(2,),
        )
    with pytest.raises(ValueError, match="canonical model builder cannot be replaced"):
        MODEL_BUILDERS.register("tabu.cell.base", lambda **_: object(), replace=True)
    exact_after_rejected_override = build_from_spec(
        spec,
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )
    assert isinstance(exact_after_rejected_override, TabUCellBaseModel)


def test_source_scoped_codebook_v2_has_distinct_checkpoint_identity() -> None:
    v1 = build_model("tabu.cell.base", profile="supervised.label_broadcast.v1")
    v2 = build_model(
        "tabu.cell.base",
        profile="supervised.label_broadcast.v1",
        nominal_tokenizer="source_scoped_frozen_codebook.v2",
        nominal_codebook_seed=1729,
    )
    v1_identity = v1.checkpoint_identity()
    v2_identity = v2.checkpoint_identity()
    assert v1_identity["tokenizer_version"] == "cell-tokenizer.v1"
    assert "nominal_codebook_hash" not in v1_identity
    assert v2_identity["tokenizer_version"] == "cell-tokenizer.v2"
    assert v2_identity["nominal_codebook_size"] == 100
    assert len(v2_identity["nominal_codebook_hash"]) == 64
    assert v1.variant_ref.semantic_hash != v2.variant_ref.semantic_hash
    with pytest.raises(ValueError, match="tokenizer_version"):
        v2.validate_checkpoint_identity(v1_identity)
    for option, value in (
        ("nominal_codebook_size", 2),
        ("nominal_codebook_seed", 999),
    ):
        with pytest.raises(ValueError, match="only configurable"):
            build_model(
                "tabu.cell.base",
                profile="supervised.label_broadcast.v1",
                **{option: value},
            )


def test_reference_config_max_features_is_enforced() -> None:
    too_small = build_model(
        "tabu.cell.base",
        config=replace(_small_config(), max_features=2),
        profile="completion.artificial_mask.v1",
    )
    with pytest.raises(ValueError, match="exceeds max_features=2"):
        too_small._forward_dense(_completion_input())
    exact = build_model(
        "tabu.cell.base",
        config=replace(_small_config(), max_features=3),
        profile="completion.artificial_mask.v1",
    )
    exact._forward_dense(_completion_input())


def test_supervised_profile_has_one_response_and_no_self_support() -> None:
    model = build_model("tabu.cell.base", profile="supervised.label_broadcast.v1")
    prediction = model._forward_dense(_supervised_input())
    assert prediction.metadata["profile_id"] == "supervised.label_broadcast.v1"
    # Query row 0 is never allowed to support its own target.
    assert float(prediction.outputs["support_weights"][0, 0, 2, 2].detach()) == 0.0
    with pytest.raises(ValueError, match="exactly one declared response"):
        bad = _supervised_input()
        bad = DenseModelInput(
            bad.values,
            bad.visible_mask,
            bad.target_mask,
            bad.natural_missing_mask,
            artificial_target_mask=bad.artificial_target_mask,
            query_target_mask=bad.query_target_mask,
            feature_specs=(
                FeatureSpec(name="x0"),
                FeatureSpec(name="x1", role=FeatureRole.RESPONSE),
                FeatureSpec(name="y", role=FeatureRole.RESPONSE),
            ),
        )
        model._forward_dense(bad)


def test_profile_target_origins_fail_closed() -> None:
    completion = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )
    with pytest.raises(ValueError, match="rejects query target"):
        completion._forward_dense(_supervised_input())
    with pytest.raises(ValueError, match="zero response columns"):
        completion._forward_dense(
            replace(
                _completion_input(),
                feature_specs=(
                    FeatureSpec(name="x0"),
                    FeatureSpec(name="x1"),
                    FeatureSpec(name="y", role=FeatureRole.RESPONSE),
                ),
            )
        )

    supervised = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    with pytest.raises(ValueError, match="rejects artificial-mask"):
        supervised._forward_dense(_completion_input())
    with pytest.raises(ValueError, match="exactly one declared response"):
        supervised._forward_dense(replace(_supervised_input(), feature_specs=()))

    visible = torch.tensor([[[True, False, False], [True, True, True]]])
    query = ~visible
    misaligned = DenseModelInput(
        values=torch.tensor([[[1.0, 0.0, 0.0], [2.0, 3.0, 4.0]]]),
        visible_mask=visible,
        target_mask=query,
        natural_missing_mask=torch.zeros_like(query),
        artificial_target_mask=torch.zeros_like(query),
        query_target_mask=query,
        unsupported_target_mask=torch.zeros_like(query),
        feature_specs=(
            FeatureSpec(name="x0"),
            FeatureSpec(name="x1"),
            FeatureSpec(name="y", role=FeatureRole.RESPONSE),
        ),
    )
    with pytest.raises(ValueError, match="single declared response"):
        supervised._forward_dense(misaligned)


def test_supervised_column_axis_excludes_query_rows_but_row_axis_keeps_predictors() -> None:
    model = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    observed: dict[str, torch.Tensor] = {}

    def capture_masks(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        kwargs: dict[str, torch.Tensor],
    ) -> None:
        observed["column"] = kwargs["column_source_mask"].detach().clone()
        observed["row"] = kwargs["row_source_mask"].detach().clone()

    handle = model.dynamics.blocks[0].register_forward_pre_hook(
        capture_masks,
        with_kwargs=True,
    )
    try:
        inputs = _supervised_input()
        model._forward_dense(inputs)
    finally:
        handle.remove()

    query_rows = inputs.query_target_mask.any(dim=2, keepdim=True)
    assert torch.equal(observed["column"], inputs.visible_mask & ~query_rows)
    assert torch.equal(observed["row"], inputs.visible_mask)
    assert model.dynamics.blocks[0].exclude_row_self is False


def test_label_broadcast_is_finite_for_large_finite_receivers() -> None:
    visible = torch.tensor([[[True, False]]])
    query = ~visible
    inputs = DenseModelInput(
        values=torch.tensor([[[1.0e20, 0.0]]]),
        visible_mask=visible,
        target_mask=query,
        natural_missing_mask=torch.zeros_like(query),
        artificial_target_mask=torch.zeros_like(query),
        query_target_mask=query,
        feature_specs=(
            FeatureSpec(name="x"),
            FeatureSpec(name="y", role=FeatureRole.RESPONSE),
        ),
    )
    cells = torch.tensor([[[[1.0e20, 0.0], [1.0, 0.0]]]])
    broadcast = _label_broadcast(cells, inputs, enabled=True)
    assert torch.isfinite(broadcast).all()


@pytest.mark.parametrize("name", ["presence_tau", "denominator_epsilon", "routing_bandwidth"])
def test_reference_config_rejects_non_finite_kernel_scalars(name: str) -> None:
    with pytest.raises(ValueError, match="finite positive scalar"):
        ReferenceConfig(**{name: float("nan")})
    with pytest.raises(ValueError, match="finite positive scalar"):
        ReferenceConfig(**{name: float("inf")})


def test_completion_profile_trace_contains_profile_and_tokenizer_version() -> None:
    model = build_model("tabu.cell.base", profile="completion.artificial_mask.v1")
    prediction = model._forward_dense(_completion_input())
    assert prediction.metadata["profile_id"] == "completion.artificial_mask.v1"
    assert prediction.metadata["variant_ref"]["contract_version"] == "0.2.0"
    tokenizer = next(event for event in prediction.trace.events if event.name == "tokenizer")
    assert tokenizer.metadata["tokenizer_version"] == "cell-tokenizer.v1"
    assert tuple(event.name for event in prediction.trace.events) == (
        "symbolizer",
        "tokenizer",
        "dynamics_plan",
        "readout",
        "prediction_boundary",
    )
    dynamics = next(event for event in prediction.trace.events if event.name == "dynamics_plan")
    assert dynamics.metadata["shape"] == (1, 3, 3, model.config.d_model)
    assert prediction.metadata["dynamics_plan"] == "cell_unit_three_omab"
    assert dynamics.metadata["plan"] == get_model_spec("tabu.cell.base").dynamics["family"]
    assert prediction.metadata["unit"] == "cell"


def test_truth_free_query_marker_drives_label_broadcast_without_label_payload() -> None:
    inputs = _supervised_input()
    model = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    symbols = model.symbolizer(inputs)
    tokens = model.tokenizer(symbols)
    broadcast = _label_broadcast(tokens.cells, inputs, enabled=True)
    assert not torch.equal(broadcast[0, 0, 0], tokens.cells[0, 0, 0])
    assert torch.equal(broadcast[0, 0, 2], tokens.cells[0, 0, 2])
    assert inputs.values[0, 0, 2].item() == 0.0

    prediction = model._forward_dense(inputs)
    assert prediction.metadata["query_marker"] == "unified"
    assert prediction.auxiliaries["query_target_mask"].any()


def test_cell_tokenizer_has_typed_numeric_nominal_and_exact_null_branches() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [0.0, 0.0]]])
    visible = torch.tensor([[[True, True], [True, True], [True, True], [False, False]]])
    natural = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=natural,
        natural_missing_mask=natural,
        artificial_target_mask=natural,
        query_target_mask=torch.zeros_like(natural),
        unsupported_target_mask=torch.zeros_like(natural),
        feature_specs=(
            FeatureSpec(name="n", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b", "c"),
                codebook_id="tabubase-tokenizer-contract",
            ),
        ),
        episode_id="tabubase-tokenizer-contract",
    )
    tokenizer = CellTokenizer(_small_config())
    symbols = Symbolizer()(inputs)
    first = tokenizer(symbols).cells
    second = tokenizer(symbols).cells
    assert torch.equal(first, second)
    assert torch.allclose(first[0, :3, 1].norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.equal(first[0, 3], torch.zeros_like(first[0, 3]))
    assert not torch.equal(first[0, 0, 1], first[0, 1, 1])


def test_cell_base_keeps_numeric_and_nominal_predictions_typed() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
    visible = torch.tensor([[[True, True], [True, False], [True, True]]])
    target = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        artificial_target_mask=target,
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=torch.zeros_like(target),
        feature_specs=(
            FeatureSpec(name="n", kind=FeatureKind.NUMERIC),
            FeatureSpec(
                name="c",
                kind=FeatureKind.CATEGORICAL,
                domain=("a", "b"),
                codebook_id="tabubase-mixed-contract",
            ),
        ),
        episode_id="tabubase-mixed-contract",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )._forward_dense(inputs)
    assert prediction.entries["numeric"].status.value == "ok"
    assert prediction.entries["categorical"].status.value == "ok"
    assert prediction.outputs["categorical"].shape == (1, 3, 2)
    assert torch.isfinite(prediction.auxiliaries["categorical_log_probabilities"]).all()
    assert not bool(prediction.auxiliaries["numeric_raw_prediction"][:, :, 1].any())


def test_numeric_prediction_uses_context_standardized_scale_with_raw_projection() -> None:
    def episode(multiplier: float, shift: float) -> DenseModelInput:
        visible = torch.tensor([[[True], [True], [True], [False]]])
        target = ~visible
        raw = torch.tensor([1.0, 2.0, 3.0]) * multiplier + shift
        values = torch.zeros(1, 4, 1)
        values[0, :3, 0] = raw
        return DenseModelInput(
            values,
            visible,
            target,
            torch.zeros_like(target),
            artificial_target_mask=target,
            query_target_mask=torch.zeros_like(target),
            unsupported_target_mask=torch.zeros_like(target),
            feature_specs=(FeatureSpec(name="x", kind=FeatureKind.NUMERIC),),
        )

    torch.manual_seed(37)
    model = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
        numeric_terminal="nadaraya_watson",
    ).eval()
    first = model._forward_dense(episode(1.0, 0.0))
    affine = model._forward_dense(episode(10.0, 5.0))
    large = model._forward_dense(episode(1.0e20, 0.0))
    target_index = (0, 3, 0)

    first_standardized = first.outputs["numeric"][target_index]
    affine_standardized = affine.outputs["numeric"][target_index]
    assert torch.allclose(first_standardized, affine_standardized, atol=2.0e-5)
    assert first.entries["numeric"].metadata["value_space"] == "context_standardized"
    assert first.auxiliaries["numeric_context_count"].item() == 3
    assert torch.allclose(
        affine.auxiliaries["numeric_context_mean"],
        first.auxiliaries["numeric_context_mean"] * 10.0 + 5.0,
    )
    assert torch.allclose(
        affine.auxiliaries["numeric_context_std"],
        first.auxiliaries["numeric_context_std"] * 10.0,
        atol=2.0e-6,
    )
    first_raw = first.auxiliaries["numeric_raw_prediction"][target_index]
    affine_raw = affine.auxiliaries["numeric_raw_prediction"][target_index]
    assert torch.allclose(affine_raw, first_raw * 10.0 + 5.0, atol=2.0e-4)
    assert torch.isfinite(large.outputs["numeric"]).all()
    assert torch.isfinite(large.auxiliaries["numeric_context_mean"]).all()
    assert torch.isfinite(large.auxiliaries["numeric_context_std"]).all()
    assert torch.isfinite(large.auxiliaries["numeric_raw_prediction"]).all()
    assert torch.allclose(
        large.outputs["numeric"][target_index],
        first_standardized,
        atol=2.0e-5,
    )


@pytest.mark.parametrize("block_kind", [DynamicsBlockKind.OMAB, DynamicsBlockKind.MAB])
def test_natural_missing_is_typed_no_support_for_both_dynamics_blocks(
    block_kind: DynamicsBlockKind,
) -> None:
    values = torch.tensor([[[0.0], [2.0]]])
    visible = torch.tensor([[[False], [True]]])
    target = ~visible
    inputs = DenseModelInput(
        values=values,
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.tensor([[[True], [False]]]),
        artificial_target_mask=torch.zeros_like(target),
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=target,
        episode_id=f"tabubase-null-{block_kind.value}",
    )
    prediction = build_model(
        "tabu.cell.base",
        config=_small_config(block_kind=block_kind),
        profile="completion.artificial_mask.v1",
    )._forward_dense(inputs)
    assert prediction.metadata["status"] == "unsupported"
    assert prediction.entries["numeric"].status.value == "unsupported"
    assert prediction.auxiliaries["support_available"][0, 0, 0].item() is False


def test_unsupported_numeric_target_has_no_raw_auxiliary_prediction() -> None:
    visible = torch.tensor([[[False], [True], [True]]])
    target = ~visible
    inputs = DenseModelInput(
        values=torch.tensor([[[0.0], [2.0], [3.0]]]),
        visible_mask=visible,
        target_mask=target,
        natural_missing_mask=torch.zeros_like(target),
        artificial_target_mask=torch.zeros_like(target),
        query_target_mask=torch.zeros_like(target),
        unsupported_target_mask=target,
    )
    prediction = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    )._forward_dense(inputs)
    assert prediction.entries["numeric"].status.value == "unsupported"
    assert prediction.auxiliaries["numeric_raw_prediction"][0, 0, 0].item() == 0.0


def test_truth_enters_only_after_forward_and_hidden_payload_is_inert() -> None:
    visible = torch.tensor([[True, False], [True, True], [True, True]])
    target = ~visible
    common = {
        "visible_mask": visible,
        "target_mask": target,
        "natural_missing_mask": torch.zeros_like(target),
        "episode_id": "tabubase-truth-boundary",
    }
    clean = SimpleNamespace(
        forward_values=torch.tensor([[1.0, 0.0], [2.0, 4.0], [3.0, 7.0]]),
        **common,
    )
    poisoned = SimpleNamespace(
        forward_values=torch.tensor([[1.0, 999.0], [2.0, 4.0], [3.0, 7.0]]),
        **common,
    )
    clean_dense = DenseModelInput.from_any(clean)
    poisoned_dense = DenseModelInput.from_any(poisoned)
    assert torch.equal(clean_dense.values, poisoned_dense.values)

    torch.manual_seed(5)
    model = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    ).eval()
    first = model._forward_dense(clean_dense)
    second = model._forward_dense(poisoned_dense)
    assert torch.equal(first.outputs["numeric"], second.outputs["numeric"])
    assert first.prediction_hash == second.prediction_hash
    boundary = next(event for event in first.trace.events if event.name == "prediction_boundary")
    assert boundary.metadata["truth_not_available"] is True
    assert boundary.metadata["supervision_boundary"] == "sidecar_only"

    def sidecar(value: float) -> TruthSidecar:
        truth_values = torch.zeros(3, 2)
        truth_values[0, 1] = value
        return TruthSidecar(
            episode_id="tabubase-truth-boundary",
            recipe_hash="a" * 64,
            row_ids=("r0", "r1", "r2"),
            feature_names=("x", "y"),
            target_values=truth_values,
            target_mask=target,
        )

    def numeric_loss(truth: TruthSidecar) -> torch.Tensor:
        prediction = first.outputs["numeric"]
        standardized_truth = (
            truth.target_values - first.auxiliaries["numeric_context_mean"]
        ) / first.auxiliaries["numeric_context_scale"]
        residual = prediction[truth.target_mask] - standardized_truth[truth.target_mask]
        return residual.square().mean()

    assert numeric_loss(sidecar(5.0)).item() != numeric_loss(sidecar(50.0)).item()
    assert first.prediction_hash == second.prediction_hash


def test_fixed_seed_checkpoint_roundtrip_is_deterministic() -> None:
    torch.manual_seed(1729)
    first = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    ).eval()
    torch.manual_seed(1729)
    second = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="completion.artificial_mask.v1",
    ).eval()
    second.load_state_dict(first.state_dict())

    fixture = _completion_input()
    with torch.no_grad():
        left = first._forward_dense(fixture)
        right = second._forward_dense(fixture)
    assert left.outputs.keys() == right.outputs.keys()
    for name in left.outputs:
        assert torch.equal(left.outputs[name], right.outputs[name]), name
    assert left.prediction_hash == right.prediction_hash
    assert first.checkpoint_identity() == second.checkpoint_identity()

    supervised = build_model(
        "tabu.cell.base",
        config=_small_config(),
        profile="supervised.label_broadcast.v1",
    )
    assert first.variant_ref.semantic_hash != supervised.variant_ref.semantic_hash


def test_truth_free_guard_recurses_into_pydantic_models() -> None:
    class WrappedTruth(BaseModel):
        target_values: list[float]

    with pytest.raises(ValueError, match="truth-bearing key"):
        assert_truth_free({"opaque": WrappedTruth(target_values=[999.0])})
