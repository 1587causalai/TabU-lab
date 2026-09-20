import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.curriculum_v53.protocol import _model
from tabu_lab.models.restoration.backbone import BackboneConfig as HistoricalBackboneConfig
from tabu_lab.models.restoration_v53 import BackboneConfig, V53Config, V53Model
from tabu_lab.models.restoration_v53.encoding import AffineValueEncoder
from tabu_lab.models.restoration_v53.geometry import QRIsometry
from tabu_lab.restoration_optimizers import OptimizerConfig, adamw, switch_to_muon

from .test_model import config, deterministic_cpu  # noqa: F401


def test_defaults_propagate_to_curriculum_without_changing_historical_backbone():
    assert HistoricalBackboneConfig().tau_presence == 1.
    for cfg in (V53Config(), _model({}), _model({"backbone": {"slots": 4}})):
        assert cfg.backbone.tau_presence == 1e-6
        assert cfg.input_projection == "isometric_qr"
        assert V53Config.from_dict(cfg.as_dict()) == cfg
    assert _model({"backbone": {"tau_presence": 0.03}}).backbone.tau_presence == .03
    for key in ("input_projection", "tau_presence"):
        values = V53Config().as_dict()
        del (values if key == "input_projection" else values["backbone"])[key]
        with pytest.raises(ValueError, match="lacks input geometry identity"):
            V53Config.from_dict(values)


@pytest.mark.parametrize("width", [128, 160])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_gram_survives_plain_optimizer_updates_and_dtype_conversion(width, dtype):
    encoder = AffineValueEncoder(width).to(dtype=dtype)
    values, target = torch.randn(7, 128, dtype=dtype), torch.randn(7, width, dtype=dtype)
    values[0] = 0
    values[0, :8] = 1  # Raw nominal C code; energy eight must survive.
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2, weight_decay=.5)
    original = encoder.projection.weight.detach().clone()
    for _ in range(6):
        optimizer.zero_grad()
        loss = (encoder.projection(values) - target).square().mean()
        loss.backward()
        raw = encoder.projection.parametrizations.weight.original
        assert torch.isfinite(raw.grad).all() and raw.grad.norm() > 0
        optimizer.step()
        encoder.validate_projection()
        mapped = encoder.projection(values)
        # Compare after scaling by each vector norm: unscaled near-zero dot
        # products amplify ordinary FP32 accumulation error over 128 terms.
        norms = values.norm(dim=-1)
        scale = norms[:, None] * norms[None, :]
        tolerance = 32 * torch.finfo(dtype).eps
        torch.testing.assert_close((mapped @ mapped.T) / scale, (values @ values.T) / scale,
                                   atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(mapped[0].square().sum(), values.new_tensor(8.))
    assert not torch.allclose(original, encoder.projection.weight)


@pytest.mark.parametrize("width", [128, 160])
def test_legacy_projection_keeps_exact_seed_sequence_and_extra_scale(width):
    torch.manual_seed(33)
    old = AffineValueEncoder(width, input_projection="legacy_scaled").double()
    torch.manual_seed(33)
    new = AffineValueEncoder(width).double()
    # Float32 QR initialization followed by a double QR can differ at roundoff.
    torch.testing.assert_close(new.projection.weight, old.projection.weight * 8,
                               atol=2e-6, rtol=2e-6)
    for name in ("cell_seed", "unit_seed", "feature_seed"):
        assert torch.equal(getattr(old, name), getattr(new, name))


def test_qr_parameterization_derivative_and_degenerate_coordinates():
    mapping = QRIsometry()
    raw = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mapping, (raw,))
    for bad in (torch.zeros_like(raw), raw.detach()[:, :1].expand(-1, 3).clone()):
        with pytest.raises(FloatingPointError, match="rank deficient"):
            mapping(bad)
    with pytest.raises(FloatingPointError, match="finite"):
        mapping(torch.full_like(raw, float("nan")))


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_optimizer_partition_and_checkpoint_continue_on_isometric_weights(kind):
    if kind == "muon" and not hasattr(torch.optim, "Muon"):
        pytest.skip("native Muon unavailable")
    cfg = config()
    model = V53Model(cfg).double()
    opt_cfg = OptimizerConfig(1e-3, .01, (.9, .95), 1e-8, 1.)

    def make_opt(net):
        opt = adamw(net, opt_cfg)
        return switch_to_muon(opt, net, opt_cfg)[0] if kind == "muon" else opt

    optimizer = make_opt(model)
    values, target = torch.randn(9, 128).double(), torch.randn(9, 128).double()

    def step(net, opt):
        opt.zero_grad()
        (net.encoder.projection(values) - target).square().mean().backward()
        opt.step()
        net.encoder.validate_projection()

    step(model, optimizer)
    clone = V53Model(V53Config.from_dict(cfg.as_dict())).double()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other = make_opt(clone)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(clone, other)
    for key, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])
    torch.testing.assert_close(model.encoder.projection.weight.T @ model.encoder.projection.weight,
                               torch.eye(128, dtype=torch.float64))


def test_checkpoint_geometry_identity_fails_closed_even_without_strict_loading():
    old_cfg = replace(config(), input_projection="legacy_scaled",
                      backbone=replace(config().backbone, tau_presence=1.))
    old = V53Model(old_cfg).double()
    state = copy.deepcopy(old.state_dict())
    del state["_geometry_signature"]
    with pytest.raises(RuntimeError, match="lacks input geometry identity"):
        V53Model(config()).double().load_state_dict(state, strict=False)
    old.load_state_dict(state)  # Explicit historical interpretation.
    for cfg in (config(), replace(old_cfg, backbone=BackboneConfig(tau_presence=1e-6))):
        with pytest.raises(RuntimeError, match="input geometry identity does not match"):
            V53Model(cfg).double().load_state_dict(old.state_dict(), strict=False)
    model = V53Model(config()).double()
    invalid = copy.deepcopy(model.state_dict())
    invalid["encoder.projection.parametrizations.weight.original"].zero_()
    with pytest.raises(FloatingPointError, match="rank deficient"):
        model.load_state_dict(invalid)


def test_presence_default_half_radius_and_unit_readout():
    from tabu_lab.models.restoration.backbone import OMAB

    attention = OMAB(BackboneConfig(layers=1, slots=4)).double()
    # Stable log presence is the exact existing operator, with only tau changed.
    method = attention.log_presence
    values = torch.zeros(3, 128, dtype=torch.float64)
    values[1, 0], values[2, 0] = 1e-3, 1.
    actual = method(values).exp()
    torch.testing.assert_close(actual, torch.tensor([0., .5, 1 / (1 + 1e-6)], dtype=torch.float64))
