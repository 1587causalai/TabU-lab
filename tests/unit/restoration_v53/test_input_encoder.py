import copy

import pytest
import torch

from tabu_lab.curriculum_v53.protocol import _model
from tabu_lab.models.restoration_v53 import BackboneConfig, V53Config, V53Model
from tabu_lab.models.restoration_v53.encoding import AffineValueEncoder
from tabu_lab.restoration_optimizers import OptimizerConfig, adamw, switch_to_muon

from .test_model import config, deterministic_cpu  # noqa: F401


def test_presence_default_and_config_roundtrip():
    for cfg in (V53Config(), _model({}), _model({"backbone": {"slots": 4}})):
        assert cfg.backbone.tau_presence == 1.
        assert V53Config.from_dict(cfg.as_dict()) == cfg
    assert _model({"backbone": {"tau_presence": 0.03}}).backbone.tau_presence == .03


@pytest.mark.parametrize("width", [128, 160])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_input_layer_learns_amplitude(width, dtype, monkeypatch):
    encoder = AffineValueEncoder(width).to(dtype=dtype)
    assert not torch.nn.utils.parametrize.is_parametrized(encoder.projection)
    original = encoder.projection.weight.detach().clone()
    torch.testing.assert_close(original.T @ original, torch.eye(128, dtype=dtype) / 64,
                               atol=2e-6, rtol=2e-6)
    values = torch.eye(128, dtype=dtype)
    target = 2 * encoder.projection(values).detach()
    initial_loss = (encoder.projection(values) - target).square().sum() / width
    optimizer = torch.optim.SGD(encoder.parameters(), lr=8.)

    def forbidden_qr(*args, **kwargs):
        raise AssertionError("default projection must not use QR after initialization")

    monkeypatch.setattr(torch.linalg, "qr", forbidden_qr)
    for _ in range(8):
        optimizer.zero_grad()
        loss = (encoder.projection(values) - target).square().sum() / width
        loss.backward()
        assert torch.isfinite(encoder.projection.weight.grad).all()
        optimizer.step()
    assert (encoder.projection(values) - target).square().sum() / width < initial_loss / 2
    assert encoder.projection.weight.norm() > 1.25 * original.norm()
    assert not torch.allclose(encoder.projection.weight.T @ encoder.projection.weight,
                              torch.eye(128, dtype=dtype) / 64, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_optimizer_partition_and_checkpoint_continue_on_projection_weights(kind):
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

    step(model, optimizer)
    clone = V53Model(V53Config.from_dict(cfg.as_dict())).double()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    other = make_opt(clone)
    other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    step(model, optimizer)
    step(clone, other)
    for key, value in model.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])


def test_presence_default_half_radius_and_unit_readout():
    from tabu_lab.models.restoration.backbone import OMAB

    attention = OMAB(BackboneConfig(layers=1, slots=4)).double()
    # Stable log presence is the exact existing operator, with only tau changed.
    method = attention.log_presence
    values = torch.zeros(3, 128, dtype=torch.float64)
    values[1, 0], values[2, 0] = 1e-3, 1.
    actual = method(values).exp()
    torch.testing.assert_close(
        actual, torch.tensor([0., 1e-6 / (1 + 1e-6), .5], dtype=torch.float64)
    )
