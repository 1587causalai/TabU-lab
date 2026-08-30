from __future__ import annotations

import math

import pytest
import torch
from pydantic import ValidationError

from tabu_lab.contracts import EvidenceEpisode, ForwardRole, OriginState
from tabu_lab.experiments import (
    DynamicsSemanticConfig,
    FitDevice,
    ModelSemanticConfig,
    ReferenceBackendConfig,
)
from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.experiments.runner import _reference_config
from tabu_lab.models import DynamicsBlockKind, ReferenceConfig, build_model
from tabu_lab.primitives import MAB, OMAB, MABAttention


def _config(*, block_kind: DynamicsBlockKind | str = DynamicsBlockKind.OMAB) -> ReferenceConfig:
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


def _episode() -> EvidenceEpisode:
    source = ForwardRole.RECEIVER | ForwardRole.SOURCE
    target = ForwardRole.RECEIVER | ForwardRole.TARGET
    return EvidenceEpisode(
        episode_id="mab-omab-variant",
        dataset_id="variant-test",
        source_partition="test",
        fit_partition="train",
        row_ids=("r0", "r1", "r2"),
        feature_names=("x", "y"),
        forward_values=torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
        origin_states=(
            (OriginState.ARTIFICIAL_MASK, OriginState.OBSERVED),
            (OriginState.OBSERVED, OriginState.OBSERVED),
            (OriginState.OBSERVED, OriginState.OBSERVED),
        ),
        forward_roles=(
            (target, source),
            (source, source),
            (source, source),
        ),
    )


def test_reference_config_has_typed_global_block_kind() -> None:
    assert _config(block_kind="mab").block_kind is DynamicsBlockKind.MAB
    assert _config().block_kind is DynamicsBlockKind.OMAB
    with pytest.raises(ValueError, match="block_kind"):
        _config(block_kind="not-a-block")


def test_semantic_config_rejects_unknown_block_kind_and_preserves_legacy_hash() -> None:
    legacy = ModelSemanticConfig(reference=ReferenceBackendConfig())
    legacy_payload = legacy.model_dump(mode="python")
    from tabu_lab.contracts import canonical_hash

    assert "dynamics" not in legacy_payload
    assert legacy.content_hash == canonical_hash(legacy_payload)

    explicit = ModelSemanticConfig(
        reference=ReferenceBackendConfig(),
        dynamics=DynamicsSemanticConfig(block_kind="mab"),
    )
    assert explicit.content_hash != legacy.content_hash
    with pytest.raises(ValidationError, match="block_kind"):
        ModelSemanticConfig.model_validate(
            {"reference": {}, "dynamics": {"block_kind": "not-a-block"}}
        )


def test_runner_passes_semantic_block_kind_into_runtime_config() -> None:
    spec = build_f0_preregistration("tabuf", device=FitDevice.CPU)
    mab_semantic = spec.semantic.model_copy(
        update={"dynamics": DynamicsSemanticConfig(block_kind="mab")}
    )
    mab_spec = spec.model_copy(update={"semantic": mab_semantic})
    assert _reference_config(mab_spec).block_kind is DynamicsBlockKind.MAB


def test_explicit_pair_variants_bind_semantic_and_run_identity_inputs() -> None:
    omab_spec = build_f0_preregistration(
        "tabuf",
        device=FitDevice.CPU,
        block_kind="omab",
        experiment_id="F0-paired-omab",
    )
    mab_spec = build_f0_preregistration(
        "tabuf",
        device=FitDevice.CPU,
        block_kind="mab",
        experiment_id="F0-paired-mab",
    )

    assert omab_spec.semantic.model_dump(mode="json")["dynamics"] == {
        "block_kind": "omab"
    }
    assert mab_spec.semantic.model_dump(mode="json")["dynamics"] == {
        "block_kind": "mab"
    }
    assert omab_spec.semantic.content_hash != mab_spec.semantic.content_hash
    assert omab_spec.spec_hash != mab_spec.spec_hash


def test_mab_and_omab_are_parameter_isomorphic_and_strictly_loadable() -> None:
    torch.manual_seed(17)
    omab = OMAB(8, 2, 16)
    torch.manual_seed(17)
    mab = MAB(8, 2, 16)

    assert tuple(omab.state_dict()) == tuple(mab.state_dict())
    assert sum(parameter.numel() for parameter in omab.parameters()) == sum(
        parameter.numel() for parameter in mab.parameters()
    )
    for name, value in omab.state_dict().items():
        assert torch.equal(value, mab.state_dict()[name]), name
    result = mab.load_state_dict(omab.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_mab_attention_matches_masked_softmax_reference() -> None:
    torch.manual_seed(23)
    attention = MABAttention(4, 2).eval()
    receivers = torch.randn(1, 2, 4)
    sources = torch.randn(1, 3, 4)
    source_mask = torch.tensor([[True, False, True]])
    actual = attention(receivers, sources, source_mask=source_mask)

    q = attention.q_proj(receivers).view(1, 2, 2, 2).transpose(1, 2)
    k = attention.k_proj(sources).view(1, 3, 2, 2).transpose(1, 2)
    v = attention.v_proj(sources).view(1, 3, 2, 2).transpose(1, 2)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2)
    weights = torch.softmax(
        scores.masked_fill(~source_mask[:, None, None, :], -torch.inf), dim=-1
    )
    attended = torch.matmul(weights, v).transpose(1, 2).reshape(1, 2, 4)
    expected = attention.out_proj(attended)

    assert torch.allclose(actual.weights, weights)
    assert torch.allclose(actual.output, expected)
    assert actual.support_available.all()


def test_mab_removes_omab_zero_receiver_closure() -> None:
    torch.manual_seed(31)
    mab = MAB(8, 2, 16).eval()
    receivers = torch.zeros(1, 2, 8)
    sources = torch.randn(1, 3, 8)
    output = mab(receivers, sources)
    assert not torch.equal(output.state, torch.zeros_like(output.state))


@pytest.mark.parametrize(
    "model_id",
    (
        "tabuf",
        "tabufl",
        "tabul",
        "tabu4rec",
        "tabu4graph",
        "tabu.unit_row",
        "tabu.unit_pair",
    ),
)
def test_global_block_kind_selects_one_variant_in_every_dynamics_path(model_id: str) -> None:
    model = build_model(model_id, config=_config(block_kind="mab"))
    blocks = tuple(module for module in model.modules() if isinstance(module, (MAB, OMAB)))
    assert blocks
    assert all(type(module) is MAB for module in blocks)

    omab_model = build_model(model_id, config=_config(block_kind="omab"))
    omab_blocks = tuple(
        module for module in omab_model.modules() if isinstance(module, (MAB, OMAB))
    )
    assert omab_blocks
    assert all(type(module) is OMAB for module in omab_blocks)


@pytest.mark.parametrize(
    "model_id",
    (
        "tabuf",
        "tabufl",
        "tabul",
        "tabu4rec",
        "tabu4graph",
        "tabu.unit_row",
        "tabu.unit_pair",
    ),
)
def test_full_model_variants_are_seedwise_parameter_isomorphic(model_id: str) -> None:
    torch.manual_seed(97)
    omab = build_model(model_id, config=_config(block_kind="omab"))
    torch.manual_seed(97)
    mab = build_model(model_id, config=_config(block_kind="mab"))

    assert tuple(omab.state_dict()) == tuple(mab.state_dict())
    for name, value in omab.state_dict().items():
        assert torch.equal(value, mab.state_dict()[name]), (model_id, name)


def test_forward_trace_records_non_o_ablation_variant() -> None:
    model = build_model("tabuf", config=_config(block_kind="mab")).eval()
    prediction = model(_episode())
    assert prediction.trace.metadata["block_kind"] == "mab"
    assert prediction.trace.metadata["variant_role"] == "non_o_ablation"
    assert prediction.metadata["block_kind"] == "mab"
    assert prediction.metadata["variant_role"] == "non_o_ablation"
