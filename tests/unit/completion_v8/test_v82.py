"""V8.2 algebra, role boundaries, precision, migration and memory qualification."""

import copy
import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    configure_memory,
    from_v81,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.completion_v8.row_attention import MEAN_LINEAR, RAW_LINEAR, PresenceLinearOMAB
from tabu_lab.models.restoration.backbone import OMAB, BackboneConfig

from .test_completion import data, spec
from .test_completion import runtime as runtime


def config(**kwargs):
    return V8Config(spec(**kwargs), codebook_mode="learned_feature64_v1", row_attention=MEAN_LINEAR)


def operator(dtype=torch.float64, mode=RAW_LINEAR):
    op = PresenceLinearOMAB(BackboneConfig(width=8, heads=2, ff_width=12), mode=mode).to(dtype)
    op._accumulation_dtype_override = dtype
    return op


def pairwise(op, x, y, mask):
    """Independent explicit sum; direct squared-norm gates for bounded test inputs."""
    y = y.masked_fill(~mask[..., None], 0)
    q = F.linear(x, op.q.weight).reshape(x.shape[0], x.shape[1], 2, 4)
    k = F.linear(y, op.k.weight).reshape(y.shape[0], y.shape[1], 2, 4)
    v = F.linear(y, op.v.weight).reshape(y.shape[0], y.shape[1], 2, 4)
    sx = F.linear(x, op.attention_presence.weight).square().sum(-1)
    sy = F.linear(y, op.attention_presence.weight).square().sum(-1)
    px = sx / (op.config.tau_presence + sx)
    py = sy / (op.config.tau_presence + sy)
    if op.mode == MEAN_LINEAR:
        q = q / (q.square().sum(-1, keepdim=True) + op.config.norm_eps).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + op.config.norm_eps).sqrt()
        py = py / mask.sum(-1, keepdim=True).clamp_min(1)
    weights = torch.einsum("brhd,bshd->brsh", q, k) / math.sqrt(4)
    mixed = torch.einsum("brsh,bs,bshd->brhd", weights, py, v).flatten(-2)
    return F.linear(px[..., None] * mixed, op.out.weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", [RAW_LINEAR, MEAN_LINEAR])
def test_explicit_pairwise_outputs_and_all_gradients(dtype, mode):
    op = operator(dtype, mode)
    ref = copy.deepcopy(op)
    x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
    y = torch.randn(2, 5, 8, dtype=dtype, requires_grad=True)
    xr, yr = x.detach().clone().requires_grad_(), y.detach().clone().requires_grad_()
    mask = torch.tensor([[True, False, True, True, False], [False] * 5])
    actual = op.attention_update(x, y, mask)[0]
    expected = pairwise(ref, xr, yr, mask)
    tolerance = 4e-6 if dtype == torch.float32 else 2e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for a, b in [(x.grad, xr.grad), (y.grad, yr.grad)]:
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    for name, value in op.named_parameters():
        other = dict(ref.named_parameters())[name]
        if value.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(value.grad, other.grad, rtol=tolerance, atol=tolerance)
    assert torch.equal(y.grad[~mask], torch.zeros_like(y.grad[~mask]))


@pytest.mark.parametrize("mode", [RAW_LINEAR, MEAN_LINEAR])
def test_physical_deletion_empty_zero_receiver_and_permutations(mode):
    op = operator(mode=mode)
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    x[:, 1] = 0
    y = torch.randn(2, 5, 8, dtype=torch.float64)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[:, 2] = False
    y[:, 2] = torch.nan
    out = op.attention_update(x, y, mask)[0]
    torch.testing.assert_close(
        out, op.attention_update(x, y[:, [0, 1, 3, 4]], mask[:, [0, 1, 3, 4]])[0]
    )
    assert torch.count_nonzero(out[:, 1]) == 0
    perm = torch.tensor([3, 0, 4, 2, 1])
    torch.testing.assert_close(out, op.attention_update(x, y[:, perm], mask[:, perm])[0])
    rp = torch.tensor([2, 0, 1])
    torch.testing.assert_close(out[:, rp], op.attention_update(x[:, rp], y, mask)[0])
    for source, eligible in [(y, torch.zeros_like(mask)), (y[:, :0], mask[:, :0])]:
        empty = op.attention_update(x, source, eligible)[0]
        assert torch.equal(empty, torch.zeros_like(empty))
    with torch.no_grad():
        op.attention_presence.weight.zero_()
    assert torch.count_nonzero(op.attention_update(x, y, mask)[0]) == 0


@pytest.mark.parametrize("mode", [RAW_LINEAR, MEAN_LINEAR])
def test_common_source_presence_attenuation_is_not_normalized_away(mode):
    op = operator(mode=mode)
    with torch.no_grad():
        op.attention_presence.weight.zero_()
        op.attention_presence.weight[0, 0] = 1
        op.attention_presence.weight[1, 1] = 1
        for projection in (op.q, op.k, op.v):
            projection.weight[:, 0] = 0  # Change source gate without changing Q/K/V.
    x = torch.randn(1, 3, 8, dtype=torch.float64)
    x[..., 0], x[..., 1] = 0, 1
    y = torch.randn(1, 5, 8, dtype=torch.float64)
    y[..., 0], y[..., 1] = math.sqrt(0.25 / 0.75), 0
    mask = torch.ones(1, 5, dtype=torch.bool)
    before = op.attention_update(x, y, mask)[0]
    y[..., 0] = math.sqrt(0.125 / 0.875)
    after = op.attention_update(x, y, mask)[0]
    torch.testing.assert_close(after, before * 0.5, rtol=2e-12, atol=2e-12)


def test_mean_is_invariant_to_repeating_all_sources():
    op = operator(mode=MEAN_LINEAR)
    x, y = torch.randn(1, 3, 8).double(), torch.randn(1, 5, 8).double()
    mask = torch.ones(1, 5, dtype=torch.bool)
    original = op.attention_update(x, y, mask)[0]
    repeated = op.attention_update(x, y.repeat(1, 4, 1), mask.repeat(1, 4))[0]
    torch.testing.assert_close(original, repeated, rtol=2e-12, atol=2e-12)


def test_nonfinite_eligible_content_is_not_silently_deleted():
    op = operator(torch.float32)
    x = torch.ones(1, 2, 8)
    y = torch.full((1, 3, 8), 1e30)
    with pytest.raises(FloatingPointError, match=r"V8\.2"):
        op.attention_update(x, y, torch.ones(1, 3, dtype=torch.bool))
    out = op.attention_update(x, y, torch.zeros(1, 3, dtype=torch.bool))[0]
    assert torch.count_nonzero(out) == 0


def test_only_row_operator_changes_no_new_tensors_or_rng_draws():
    torch.manual_seed(9)
    old = V8CompletionModel(replace(config(), row_attention="inherited"))
    old_rng = torch.get_rng_state()
    torch.manual_seed(9)
    new = V8CompletionModel(config())
    assert torch.equal(old_rng, torch.get_rng_state())
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    axial = new.backbone.axial
    assert isinstance(axial.layers[0].row, PresenceLinearOMAB)
    assert type(axial.layers[0].column) is OMAB
    assert type(axial.layers[0].collect) is OMAB
    assert all(type(x) is OMAB for x in new.backbone.unit_blocks)
    assert all(
        m.accumulation_dtype(torch.ones(1, 128)) == torch.float32
        for m in new.modules()
        if isinstance(m, OMAB)
    )
    assert new._ll_compute_dtype == torch.float32


def test_versions_and_rejected_combinations():
    assert V8Config.from_dict(config().as_dict()) == config()
    assert config().model_version == "v8.2"
    legacy = V8Config(spec())
    assert legacy.model_version == "v8"
    assert "row_attention" not in legacy.as_dict()
    assert V8Config.from_dict(legacy.as_dict()) == legacy
    with pytest.raises(ValueError, match=r"V8\.2"):
        V8Config(spec(), row_attention="presence_linear_v1")
    with pytest.raises(ValueError, match="row_slots"):
        config(backbone=replace(spec().backbone, row_slots=2))
    with pytest.raises(ValueError, match="row_attention"):
        replace(config(), row_attention="typo")


def test_migration_exact_weights_save_reload_and_schema_guard(tmp_path):
    donor = V8CompletionModel(replace(config(), row_attention="inherited")).eval()
    old_state = {k: v.clone() for k, v in donor.state_dict().items()}
    new = from_v81(donor)
    assert not new.training
    assert new.migration_receipt["new_keys"] == []
    for key, value in new.state_dict().items():
        assert torch.equal(value, old_state[key])
    assert type(donor.backbone.axial.layers[0].row) is OMAB
    inputs, _ = data(dtype=torch.float32)
    ep = prepare_completion(inputs, new.config)
    expected = new(ep, decode=False).completed_codes.detach()
    path = tmp_path / "v82.pt"
    save_weights(new, path)
    state = torch.load(path, weights_only=True)
    assert state["schema"] == "completion_v82_weights_v1"
    restored = load_weights(path)
    assert torch.equal(expected, restored(ep, decode=False).completed_codes)
    assert restored.config.model_version == "v8.2"
    state["schema"] = "completion_v81_weights_v1"
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="schema disagrees"):
        load_weights(tmp_path / "bad.pt")
    with pytest.raises(ValueError, match=r"64D V8\.1"):
        from_v81(new)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_full_completion_two_steps_truth_isolation_and_reload(tmp_path, device):
    model = V8CompletionModel(config()).to(device)
    inputs, truth = data(dtype=torch.float32, device=device)
    changed, _ = data(dtype=torch.float32, device=device, changed=True)
    ep = prepare_completion(inputs, model.config)
    alt = prepare_completion(changed, model.config)
    with torch.no_grad():
        assert torch.equal(model(ep).completed_codes, model(alt).completed_codes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = completion_loss(model(ep, decode=False), ep, truth)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert model.feature_codebook.weight.grad.abs().sum() > 0
        assert model.backbone.axial.layers[0].row.attention_presence.weight.grad.abs().sum() > 0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    path = tmp_path / "terminal.pt"
    save_weights(model, path)
    with torch.no_grad():
        restored = load_weights(path, device=device)
        assert torch.equal(model(ep).completed_codes, restored(ep).completed_codes)


@pytest.mark.parametrize("linear", [False, True])
def test_recomputation_and_batch_chunks_match_full_outputs_and_gradients(linear):
    cfg = config()
    if not linear:
        cfg = replace(cfg, row_attention="inherited")
    base = V8CompletionModel(cfg).double()
    bounded = copy.deepcopy(base)
    configure_memory(bounded, attention_mib=1, checkpoint_activations=True)
    # Force every OMAB to exercise independent-batch chunking on small inputs.
    for module in bounded.modules():
        if isinstance(module, OMAB):
            module._bounded_bytes = 1
    inputs, truth = data()
    ep = prepare_completion(inputs, cfg)
    a, b = base(ep, decode=False), bounded(ep, decode=False)
    torch.testing.assert_close(a.completed_codes, b.completed_codes, rtol=2e-10, atol=2e-10)
    completion_loss(a, ep, truth).backward()
    completion_loss(b, ep, truth).backward()
    for (name, p), (other, q) in zip(
        base.named_parameters(), bounded.named_parameters(), strict=True
    ):
        assert name == other
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-8, atol=2e-10, msg=name)
    with pytest.raises(ValueError, match="already installed"):
        configure_memory(bounded)
