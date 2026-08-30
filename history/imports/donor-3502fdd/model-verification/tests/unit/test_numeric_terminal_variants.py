from __future__ import annotations

import torch

from tabu_lab.experiments import (
    FitDevice,
    ModelSemanticConfig,
    NumericTerminal,
    ReferenceBackendConfig,
)
from tabu_lab.experiments.preregistration import build_f0_preregistration
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.primitives import (
    BilinearNumericLocalLinear,
    BilinearNumericNW,
    GlobalUserItemNumericNW,
    SameColumnNumericLocalLinear,
    SameColumnNumericNW,
)


def _config() -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=8,
    )


def test_local_linear_and_nw_share_support_contract_and_have_distinct_terminals() -> None:
    coordinates = torch.arange(6.0).view(1, 6, 1, 1)
    values = (2.0 * torch.arange(6.0) + 3.0).view(1, 6, 1)
    visible = torch.ones_like(values, dtype=torch.bool)

    local_linear = SameColumnNumericLocalLinear()(coordinates, values, visible)
    nw = SameColumnNumericNW()(coordinates, values, visible)

    assert local_linear.support_available.equal(nw.support_available)
    assert local_linear.routing.support_mask.equal(nw.routing.support_mask)
    assert local_linear.routing.support_count.equal(nw.routing.support_count)
    assert local_linear.routing.weights.shape == nw.routing.weights.shape
    assert torch.allclose(local_linear.values[:, 1:-1, 0], values[:, 1:-1, 0], atol=2.0e-3)


def test_bilinear_local_linear_preserves_two_active_arms_and_gradients() -> None:
    coordinates = torch.tensor(
        [[[[0.0], [0.2]], [[0.4], [0.6]], [[0.8], [1.0]]]],
        requires_grad=True,
    )
    values = torch.tensor([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
    visible = torch.ones_like(values, dtype=torch.bool)
    output = BilinearNumericLocalLinear()(coordinates, values, visible)
    output.values.square().mean().backward()

    assert output.support_available.all()
    assert torch.allclose(
        output.routing.weights[..., :3].sum(dim=-1),
        torch.full((1, 3, 2), 0.5),
        atol=1.0e-5,
    )
    assert torch.allclose(
        output.routing.weights[..., 3:].sum(dim=-1),
        torch.full((1, 3, 2), 0.5),
        atol=1.0e-5,
    )


def test_numeric_terminal_is_a_semantic_axis_and_fails_closed_for_unversioned_ll() -> None:
    nw = ModelSemanticConfig(reference=ReferenceBackendConfig())
    ll = nw.model_copy(update={"numeric_terminal": NumericTerminal.LOCAL_LINEAR})
    assert nw.content_hash != ll.content_hash

    try:
        build_f0_preregistration(
            "tabuf",
            device=FitDevice.CPU,
            numeric_terminal=NumericTerminal.LOCAL_LINEAR,
        )
    except ValueError as exc:
        assert "versioned" in str(exc)
    else:  # pragma: no cover - defensive assertion for the fail-closed contract
        raise AssertionError("unversioned local-linear preregistration must be rejected")

    versioned = build_f0_preregistration(
        "tabuf",
        device=FitDevice.CPU,
        numeric_terminal=NumericTerminal.LOCAL_LINEAR,
        experiment_id="F0-paired-tabuf-local-linear-v2",
        supersedes_experiment_ids=("F0-001-tabuf-v1",),
        revision_rationale="Version the independent local-linear numeric terminal axis.",
    )
    assert versioned.semantic.numeric_terminal is NumericTerminal.LOCAL_LINEAR


def test_all_executable_model_factories_accept_the_numeric_terminal_axis() -> None:
    model_ids = (
        "tabuf",
        "tabufl",
        "tabul",
        "tabu4rec",
        "tabu4graph",
        "tabu.unit_row",
        "tabu.unit_pair",
    )
    for model_id in model_ids:
        kwargs: dict[str, object] = {
            "config": _config(),
            "numeric_terminal": "local_linear",
        }
        if model_id in {"tabuf", "tabufl", "tabul", "tabu4rec"}:
            kwargs["readout_geometry"] = "matched_uf"
        if model_id in {"tabufl", "tabul"}:
            kwargs["label_columns"] = (6, 7)
            kwargs["label_address_plan"] = "predictor_unit_linked_per_label_v2"
        if model_id == "tabu4rec":
            kwargs["recommendation_address_plan"] = "matched_uf"
        if model_id == "tabu4graph":
            kwargs["target_feature"] = 0
            kwargs["unit_receiver_plan"] = "same_row_visible_cells"

        model = build_model(model_id, **kwargs)
        terminals = tuple(
            module
            for module in model.modules()
            if hasattr(module, "numeric_terminal")
        )
        assert terminals, model_id
        if model_id == "tabu4rec":
            # The current mainline is the parameterized matched score; LL/NW
            # remain available only through the explicit empirical appendix
            # address plan.
            assert all(module.numeric_terminal == "parameterized_matching" for module in terminals)
        else:
            assert all(module.numeric_terminal == "local_linear" for module in terminals)


def test_unit_as_cell_factory_defaults_to_contract_local_linear_terminal() -> None:
    model = build_model("tabu.unit_pair", config=_config())

    assert model.readout.numeric_terminal == "local_linear"
    assert model.readout.numeric_terminal_trace == "local_linear"
    assert model.readout.projection.bias is None


def test_nw_factory_defaults_and_local_linear_pair_use_distinct_terminal_types() -> None:
    nw = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
    )
    ll = build_model(
        "tabu4rec",
        config=_config(),
        recommendation_address_plan="axis_address_bootstrap_v1",
        rec_axis_summary_dim=2,
        rec_matched_residual_scale=0.1,
        numeric_terminal="local_linear",
    )
    assert isinstance(nw.readout.terminal, BilinearNumericNW)
    assert isinstance(ll.readout.terminal, BilinearNumericLocalLinear)


def test_rec_axis_address_variant_preserves_explicit_readout_width_and_scale() -> None:
    spec = build_f0_preregistration(
        "tabu4rec",
        device=FitDevice.CPU,
        fixture_version="v2",
        experiment_id="F0-test-tabu4rec-wide-axis",
        supersedes_experiment_ids=("F0-014-tabu4rec-axis-address-v2",),
        revision_rationale="test explicit Rec readout variant",
        rec_axis_summary_dim=8,
        rec_matched_residual_scale=1.0,
    )

    assert spec.semantic.rec_axis_summary_dim == 8
    assert spec.semantic.rec_matched_residual_scale == 1.0


def test_global_user_item_router_uses_one_joint_denominator() -> None:
    coordinates = torch.tensor([[[[0.0], [1.0], [2.0]], [[0.5], [1.5], [2.5]]]])
    values = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    visible = torch.ones_like(values, dtype=torch.bool)

    output = GlobalUserItemNumericNW()(coordinates, values, visible)

    assert output.routing.support_count.tolist() == [[[3, 3, 3], [3, 3, 3]]]
    assert torch.allclose(output.routing.weights.sum(dim=-1), torch.ones_like(values))
    # The target cell itself is excluded, while both same-row and same-column
    # supports remain in one shared support ledger.
    assert output.routing.support_mask[0, 0, 0].sum().item() == 3
