from __future__ import annotations

import math

import pytest
import torch

from tabu_lab.primitives import AffineCoupling, CouplingValueMap, inverse_perturbation_gain


def _perturb_parameters(module: torch.nn.Module, seed: int = 0, std: float = 0.3) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * std)


def test_default_map_is_exact_identity_at_initialisation() -> None:
    value_map = CouplingValueMap()
    x = torch.randn(5, 7, 64)

    assert torch.equal(value_map(x), x)
    assert torch.equal(value_map.inverse(x), x)


def test_hidden_layers_start_with_zero_bias() -> None:
    block = CouplingValueMap().blocks[0]

    for layer in block.net:
        if isinstance(layer, torch.nn.Linear):
            assert torch.count_nonzero(layer.bias) == 0


def test_default_parameter_count_matches_design() -> None:
    value_map = CouplingValueMap()

    assert sum(p.numel() for p in value_map.parameters()) == 4 * 28992


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_float64_round_trip_with_moderate_parameters(scale: bool, seed: int) -> None:
    # Invertibility does not guarantee conditioning for arbitrary large shift
    # networks. Exercise nonidentity maps with reproducible, moderate parameters.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        value_map = CouplingValueMap(scale=scale).double()
        _perturb_parameters(value_map, seed=seed, std=0.05)
        x = torch.randn(256, 64, dtype=torch.float64)
        y = value_map(x)
        assert not torch.allclose(y, x)
        torch.testing.assert_close(value_map.inverse(y), x, rtol=0.0, atol=1e-10)
        torch.testing.assert_close(value_map(value_map.inverse(x)), x, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize("scale", [True, False])
def test_float32_round_trip_after_moderate_perturbation(scale: bool) -> None:
    value_map = CouplingValueMap(scale=scale)
    _perturb_parameters(value_map, std=0.05)
    x = torch.randn(256, 64)

    y = value_map(x)

    assert not torch.allclose(y, x)
    assert torch.allclose(value_map.inverse(y), x, atol=1.0e-5)


def test_additive_mode_has_no_scale_outputs() -> None:
    block = AffineCoupling(8, range(4, 8), hidden=(16,), scale=False)

    assert block.net[-1].out_features == 4


def test_scale_multiplier_stays_within_clamp() -> None:
    alpha = 0.5
    block = AffineCoupling(4, [2, 3], hidden=(8,), alpha=alpha)
    with torch.no_grad():
        block.net[-1].bias[:2] = 1.0e3
        block.net[-1].bias[2:] = 0.0
    x = torch.randn(10, 4)

    ratio = block(x)[:, 2:] / x[:, 2:]

    assert torch.allclose(ratio, torch.full_like(ratio, math.exp(alpha)), rtol=1.0e-5)
    assert torch.equal(block(x)[:, :2], x[:, :2])


def test_map_bends_a_numeric_affine_code_line() -> None:
    value_map = CouplingValueMap()
    _perturb_parameters(value_map, seed=1)
    q, b = torch.randn(64), torch.randn(64)
    z = torch.linspace(-2.0, 2.0, 5).unsqueeze(-1)

    e = value_map(q + z * b)
    midpoint_gap = e[2] - 0.5 * (e[1] + e[3])

    assert midpoint_gap.norm() > 1.0e-3


def test_inverse_perturbation_gain_is_one_at_identity_and_finite_after_perturbation() -> None:
    value_map = CouplingValueMap()
    e = torch.randn(32, 64)
    delta = 1.0e-3 * torch.randn(32, 64)

    gain = inverse_perturbation_gain(value_map, e, delta)
    assert torch.allclose(gain, torch.ones(32), rtol=0.0, atol=1.0e-4)

    _perturb_parameters(value_map, seed=2)
    gain = inverse_perturbation_gain(value_map, e, delta)
    assert torch.isfinite(gain).all()
    assert (gain > 0).all()


def test_gradients_reach_every_block_through_forward_and_inverse() -> None:
    value_map = CouplingValueMap()
    x = torch.randn(16, 64, requires_grad=True)

    value_map.inverse(value_map(x) + 0.1).square().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for block in value_map.blocks:
        last = block.net[-1]
        assert last.weight.grad is not None and last.weight.grad.abs().sum() > 0


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError):
        AffineCoupling(4, [0, 1, 2, 3])
    with pytest.raises(ValueError):
        AffineCoupling(4, [0, 0])
    with pytest.raises(ValueError):
        AffineCoupling(4, [2, 3], alpha=0.0)
    with pytest.raises(ValueError):
        CouplingValueMap()(torch.randn(3, 32))


@pytest.mark.parametrize("scale", ["false", "true", 0, 1, None])
def test_public_coupling_constructor_rejects_non_boolean_scale(scale):
    with pytest.raises(ValueError, match="scale must be a boolean"):
        AffineCoupling(4, [2, 3], scale=scale)
    with pytest.raises(ValueError, match="scale must be a boolean"):
        CouplingValueMap(scale=scale)
