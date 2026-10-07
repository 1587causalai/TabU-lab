"""Unit source roles, branch initialization, and attention failure boundaries."""

import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from tabu_lab.models.restoration.backbone import OMAB, AxialBackbone, BackboneConfig
from tabu_lab.models.restoration_v7.config import V7Config
from tabu_lab.models.restoration_v7.model import InheritedV6Dynamics, V7Model
from tabu_lab.primitives.coupling import CouplingValueMap


@pytest.fixture(autouse=True)
def small_runtime():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        yield
    torch.set_num_threads(previous)


def config(**overrides):
    values = dict(
        backbone=BackboneConfig(width=64, layers=1, heads=2, ff_width=96, slots=3),
        coupling_blocks=1,
        coupling_hidden=(8,),
        unit_layers=1,
        query_init="donor",
        query_source=True,
    )
    return V7Config.v73(**(values | overrides))


def carriers():
    # Row 1 has facts, row 2 only a Query, and row 3 no active Cell.
    visible = torch.tensor([[True, False], [False, False], [False, False]])
    query = torch.tensor([[False, False], [True, False], [False, False]])
    h = torch.randn(4, 3, 64, dtype=torch.float64)
    h[:3, :2][~(visible | query)] = 0
    h[3, 2] = 0
    return h, visible, query


@pytest.mark.parametrize("policy", ["observed", "legacy_cell_sources"])
@pytest.mark.parametrize("query_source", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_unit_sources_respect_the_versioned_role_policy(policy, query_source, checkpointing):
    cfg = config(
        unit_source_policy=policy,
        query_source=query_source,
        gradient_checkpointing=checkpointing,
    )
    model = V7Model(cfg).double()
    block = model.rounds[0]
    h, visible, query = carriers()
    seen = []
    handle = block.backbone.unit_blocks[0].register_forward_pre_hook(
        lambda module, args: seen.append(args[2].clone())
    )
    result = model.run_backbone(block, h, visible, query)
    handle.remove()
    expected_mask = torch.tensor(
        [True, policy == "legacy_cell_sources" and query_source, False]
    )
    assert len(seen) == 1
    assert torch.equal(seen[0], expected_mask)

    source_cells = visible | query if query_source else visible
    axial = block.backbone.axial(h, source_cells, query)
    units = axial[:3, 2]
    expected_units = block.backbone.unit_blocks[0](units, units, expected_mask)
    torch.testing.assert_close(result[:3, 2], expected_units, rtol=0, atol=0)
    assert torch.equal(result[:, :2], axial[:, :2])


def test_legacy_constructor_and_state_keys_remain_compatible():
    cfg = config(unit_source_policy="legacy_cell_sources")
    model = V7Model(cfg).double()
    historical = InheritedV6Dynamics(cfg.backbone, cfg.unit_layers).double()
    historical.load_state_dict(model.rounds[0].backbone.state_dict(), strict=True)
    assert historical.unit_source_policy == "legacy_cell_sources"
    h, visible, query = carriers()
    expected = historical(h, visible | query, query)
    actual = model.run_backbone(model.rounds[0], h, visible, query)
    assert torch.equal(actual, expected)

    observed = V7Model(replace(cfg, unit_source_policy="observed")).double()
    observed.load_state_dict(model.state_dict(), strict=True)
    assert list(observed.state_dict()) == list(model.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(observed.state_dict()[name], value)
    changed = observed.run_backbone(observed.rounds[0], h, visible, query)
    assert not torch.equal(changed[:3, 2], actual[:3, 2])


@pytest.mark.parametrize("unit_layers", [0, 1, 2])
@pytest.mark.parametrize("row_slots", [0, 3])
def test_v73_optional_unit_stack_keeps_spec_initialization(unit_layers, row_slots):
    cfg = config(unit_layers=unit_layers)
    cfg = replace(cfg, backbone=replace(cfg.backbone, row_slots=row_slots))
    model = V7Model(cfg)
    backbone = model.rounds[0].backbone
    axial = backbone.axial if unit_layers else backbone
    for layer in axial.layers:
        for seed in (layer.slot_seed, layer.row_slot_seed):
            if seed is not None:
                torch.testing.assert_close(seed.norm(dim=-1), torch.ones(len(seed)))
    for module in backbone.modules():
        if not isinstance(module, OMAB):
            continue
        for linear in (module.q, module.k, module.v, module.out, module.ff[0], module.ff[2]):
            largest = float(linear.weight.detach().abs().max())
            xavier_bound = math.sqrt(6 / (linear.in_features + linear.out_features))
            assert largest <= xavier_bound
            # With this fixed seed, Xavier samples exceed nn.Linear's default
            # bound. This catches an accidentally reconstructed, uninitialized branch.
            assert largest > 1 / math.sqrt(linear.in_features)


def historical_initialization(cfg):
    """Frozen V7 construction order from af7e5f51, independent of V7Round."""
    model = nn.Module()
    rounds = []
    for _ in range(1 if cfg.share_rounds else cfg.rounds):
        block = nn.Module()
        block.phi = CouplingValueMap(
            cfg.code_dim,
            n_blocks=cfg.coupling_blocks,
            hidden=cfg.coupling_hidden,
            alpha=cfg.coupling_alpha,
            scale=cfg.coupling_scale,
            bias=True,
        )
        block.lift = nn.Linear(cfg.code_dim, cfg.backbone.width, bias=False)
        nn.init.orthogonal_(block.lift.weight)
        for name in ("unit_seed", "feature_seed"):
            value = torch.randn(cfg.backbone.width)
            setattr(block, name, nn.Parameter(value / value.norm()))
        block.backbone = AxialBackbone(cfg.backbone)
        for module in block.backbone.modules():
            if isinstance(module, OMAB):
                linears = (module.q, module.k, module.v, module.out, module.ff[0], module.ff[2])
                for linear in linears:
                    nn.init.xavier_uniform_(linear.weight)
        for layer in block.backbone.layers:
            with torch.no_grad():
                seeds = torch.randn_like(layer.slot_seed)
                layer.slot_seed.copy_(seeds / seeds.norm(dim=-1, keepdim=True))
        if cfg.unit_layers:
            block.backbone = InheritedV6Dynamics(cfg.backbone, cfg.unit_layers)
        rounds.append(block)
    model.rounds = nn.ModuleList(rounds)
    for kind in ("numeric", "nominal", "ordinal"):
        seed = nn.Parameter(torch.randn(cfg.code_dim) / math.sqrt(cfg.code_dim))
        setattr(model, f"query_seed_{kind}", seed)
    return model


@pytest.mark.parametrize("unit_layers", [0, 1, 2])
@pytest.mark.parametrize("share_rounds", [False, True])
def test_v7_initialization_preserves_historical_weights_and_rng(unit_layers, share_rounds):
    cfg = V7Config.for_version(
        "v7",
        backbone=BackboneConfig(width=64, layers=1, heads=2, ff_width=96, slots=3),
        coupling_blocks=1,
        coupling_hidden=(8,),
        unit_layers=unit_layers,
        rounds=2,
        share_rounds=share_rounds,
        coupling_bias=True,
        numeric_preprocessing="legacy",
    )
    torch.manual_seed(119)
    reference = historical_initialization(cfg)
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(119)
    current = V7Model(cfg)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert list(current.state_dict()) == list(reference.state_dict())
    for name, expected in reference.state_dict().items():
        assert torch.equal(current.state_dict()[name], expected), name


def overflow_block(*, strict_finite_content=True):
    cfg = BackboneConfig(width=2, layers=1, heads=1, ff_width=2, slots=2)
    module = OMAB(cfg, strict_finite_content=strict_finite_content).double()
    with torch.no_grad():
        for projection in (module.q, module.v, module.out):
            projection.weight.copy_(torch.eye(2))
        module.k.weight.copy_(-torch.eye(2))
        module.ff[0].weight.zero_()
        module.ff[2].weight.zero_()
    return module


def test_shared_operator_default_retains_historical_failure_boundary():
    module = overflow_block(strict_finite_content=False)
    receiver = torch.tensor([[1e200, 0.0]], dtype=torch.float64)
    # af7e5f51 allowed a negative-infinite content score to become zero mass.
    # The V7.3 content guard must not silently change this shared default.
    output = module(receiver, receiver, torch.tensor([True]))
    assert torch.isfinite(output).all()
    assert not OMAB(module.config).strict_finite_content


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_attention_numerical_policy_is_versioned(version, unit_layers):
    cfg = V7Config.for_version(
        version, backbone=BackboneConfig(width=64, layers=1, heads=2, ff_width=64, slots=3),
        coupling_blocks=1, coupling_hidden=(8,), unit_layers=unit_layers,
    )
    model = V7Model(cfg)
    modules = [module for module in model.modules() if isinstance(module, OMAB)]
    assert modules
    assert all(module.strict_finite_content == (version == "v7.3") for module in modules)


@pytest.mark.parametrize("sign", [-1, 1])
def test_attention_rejects_active_content_overflow_in_both_directions(sign):
    module = overflow_block()
    receiver = torch.tensor([[1e200, 0.0]], dtype=torch.float64)
    source = sign * receiver
    with pytest.raises(FloatingPointError, match="OMAB attention content"):
        module(receiver, source, torch.tensor([True]))


@pytest.mark.parametrize("zero_presence", [False, True])
def test_attention_does_not_check_deleted_source_scores(zero_presence):
    module = overflow_block()
    with torch.no_grad():
        module.q.weight.mul_(1e250)
    receiver = torch.tensor([[1e100, 0.0]], dtype=torch.float64, requires_grad=True)
    # Even an overflowing receiver projection cannot create an attention edge
    # to an ineligible or exact-zero-presence source.
    source = torch.zeros_like(receiver) if zero_presence else torch.full_like(receiver, torch.nan)
    source.requires_grad_()
    output = module(receiver, source, torch.tensor([zero_presence]))
    assert torch.equal(output, receiver)
    output.sum().backward()
    for parameter in (receiver, source, *module.parameters()):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_empty_source_q_deletion_preserves_local_ffn_and_other_batches():
    module = overflow_block()
    with torch.no_grad():
        module.ff[0].weight.copy_(torch.eye(2))
        module.ff[2].weight.copy_(torch.eye(2))
    receivers = torch.tensor([[[0.2, 0.7]], [[0.1, -0.3]]], dtype=torch.float64)
    sources = torch.tensor([[[0.0, 0.0]], [[0.4, -0.3]]], dtype=torch.float64)
    mask = torch.ones(2, 1, dtype=torch.bool)
    expected = torch.stack([module(receivers[i], sources[i], mask[i]) for i in range(2)])
    actual = module.batched(receivers, sources, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0], receivers[0])


def test_masked_nonfinite_payload_matches_physical_source_deletion():
    module = overflow_block()
    receiver = torch.tensor([[0.2, 0.7]], dtype=torch.float64)
    source = torch.tensor([[0.4, -0.3]], dtype=torch.float64)
    expected = module(receiver, source, torch.tensor([True]))
    padded = torch.cat((source, torch.full_like(source, torch.nan)))
    actual = module(receiver, padded, torch.tensor([True, False]))
    assert torch.equal(actual, expected)
