"""Communication, Null closure, reversibility and serialized protocol boundaries."""

import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    configure_memory,
    load_weights,
    prepare_completion,
    save_weights,
    with_supervised_attention,
)
from tabu_lab.models.completion_v8.supervised import SUPPORT_ROWS_NULL, supervised_dynamics
from tabu_lab.models.restoration.backbone import OMAB
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode

from .test_completion import runtime as runtime
from .test_v82 import config, operator, pairwise


def cfg(**kwargs):
    return replace(config(**kwargs), attention_policy=SUPPORT_ROWS_NULL)


def episode_data(device="cpu", changed=False):
    schema = tuple(ColumnSchema(x, "numeric") for x in ("a", "b", "y"))
    x = torch.arange(7, device=device, dtype=torch.float32) / 4
    values = (x, x.square(), x * 2 + 1)
    observed = torch.ones(7, 3, device=device, dtype=torch.bool)
    observed[1, 1] = False
    query = torch.zeros_like(observed)
    query[4:, 2] = True
    if changed:
        values = (*values[:2], torch.where(query[:, 2], values[2] + 100, values[2]))
    inputs, _, truth = make_episode(schema, values, observed, query, code_seed=7)
    return inputs, truth


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_encoder_null_inverse_and_query_row_isolation(dtype):
    encoder = V8CompletionModel(cfg()).to(dtype).encoder
    x = torch.randn(6, 4, 64, dtype=dtype, requires_grad=True)
    valid = torch.ones(6, 4, dtype=torch.bool)
    valid[0, 1] = valid[4, 2] = False
    support = torch.tensor([True, True, True, False, False, False])
    encoded = encoder(x, valid, support_rows=support)
    altered = x.detach().clone()
    altered[3] += torch.randn_like(altered[3]) * 5
    other = encoder(altered, valid, support_rows=support)
    keep = torch.arange(6) != 3
    assert torch.equal(encoded[keep], other[keep])
    assert not torch.equal(encoded[3], other[3])
    # Same isolation must hold when inverse starts from arbitrary embeddings.
    inv = encoder.inverse(x, valid, support_rows=support)
    inv_other = encoder.inverse(altered, valid, support_rows=support)
    assert torch.equal(inv[keep], inv_other[keep])
    restored = encoder.inverse(encoded, valid, support_rows=support)
    tol = 8e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(restored[valid], x[valid], rtol=tol, atol=tol)
    assert torch.count_nonzero(encoded[~valid]) == 0
    assert torch.count_nonzero(restored[~valid]) == 0
    encoded[keep].square().sum().backward()
    assert torch.count_nonzero(x.grad[3]) == 0
    assert torch.isfinite(x.grad).all()
    # No reset in coupling blocks: their own zero dynamics must stay closed.
    for block in encoder.blocks:
        zero = torch.zeros(6, 4, 32, dtype=dtype)
        u, v = block(zero, zero, valid, support)
        assert torch.count_nonzero(u) == torch.count_nonzero(v) == 0


def test_backbone_query_isolation_and_natural_null_closure():
    base = cfg()
    model = V8CompletionModel(replace(base, inherited=replace(
        base.inherited, backbone=replace(base.inherited.backbone, layers=2)
    )))
    h = torch.randn(7, 5, 128, requires_grad=True)
    with torch.no_grad():
        h[1, 1] = h[4, 2] = h[-1, -1] = 0
    support = torch.tensor([True, True, True, False, False, False])
    out = supervised_dynamics(model.backbone, h, support)
    changed = h.detach().clone()
    changed[3] += torch.randn_like(changed[3]) * 5
    alt = supervised_dynamics(model.backbone, changed, support)
    keep = torch.arange(7) != 3
    assert torch.equal(out[keep], alt[keep])
    assert not torch.equal(out[3], alt[3])
    assert torch.count_nonzero(out[[1, 4, 6], [1, 2, 4]]) == 0
    out[keep].square().sum().backward()
    assert torch.count_nonzero(h.grad[3]) == 0
    assert torch.isfinite(h.grad).all()
    # Query features cannot wake a column with zero Support evidence.
    layer = model.backbone.axial.layers[0]
    if layer.collect is not None:
        captured = []
        original = layer.column.batched
        def inspect(receivers, sources, eligible):
            assert eligible is None
            captured.append(sources.detach().clone())
            return original(receivers, sources, eligible)
        layer.column.batched = inspect
        empty = h.detach().clone()
        empty[:3, 0] = 0
        layer(empty, support)
        assert torch.count_nonzero(captured[0][0]) == 0


def test_same_query_row_cells_communicate_and_support_is_readable():
    model = V8CompletionModel(cfg())
    x = torch.randn(5, 3, 32, requires_grad=True)
    valid = torch.ones(5, 3, dtype=torch.bool)
    support = torch.tensor([True, True, True, False, False])
    intra, inter = model.encoder.blocks[:2]
    intra.attention_update(x, valid, support)[4, 0].square().sum().backward()
    assert x.grad[4, 2].abs().sum() > 0  # Query target is a source within its row.
    x.grad = None
    inter.attention_update(x, valid, support)[4, 0].square().sum().backward()
    assert x.grad[:3, 0].abs().sum() > 0  # Query receiver reads Support.
    assert torch.count_nonzero(x.grad[3]) == 0


def test_mask_free_mean_counts_all_column_addresses_and_null_has_zero_mass():
    op = operator(mode="presence_linear_mean_v1")
    x = torch.randn(2, 3, 8, dtype=torch.float64)
    y = torch.randn(2, 5, 8, dtype=torch.float64)
    y[:, 1] = 0
    actual = op.attention_update(x, y, None)[0]
    expected = pairwise(op, x, y, torch.ones(2, 5, dtype=torch.bool))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    removed = op.attention_update(x, y[:, [0, 2, 3, 4]], None)[0]
    torch.testing.assert_close(actual, removed * 4 / 5)
    assert torch.count_nonzero(op.attention_update(x, torch.zeros_like(y), None)[0]) == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"
))])
def test_fp32_loss_backward_reload_hidden_truth_and_no_cell_masks(tmp_path, device):
    model = V8CompletionModel(cfg()).to(device)
    inputs, truth = episode_data(device)
    ep = prepare_completion(inputs, model.config)
    changed = prepare_completion(episode_data(device, changed=True)[0], model.config)
    for module in model.modules():
        if isinstance(module, OMAB):
            original = module.attention_update
            def inspect(x, y, eligible, original=original):
                assert eligible is None  # Includes collect, read, rows and Unit stack.
                assert x.dtype == y.dtype == torch.float32
                return original(x, y, eligible)
            module.attention_update = inspect
    out = model(ep, decode=False)
    assert torch.equal(out.completed_codes, model(changed, decode=False).completed_codes)
    loss = completion_loss(out, ep, truth)
    loss.backward()
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.feature_codebook.weight.grad.abs().sum() > 0
    assert model.query_seed.grad.abs().sum() > 0
    assert model._ll_compute_dtype == torch.float32
    path = tmp_path / "new.pt"
    save_weights(model, path)
    loaded = load_weights(path, device=device)
    assert torch.equal(out.completed_codes, loaded(ep, decode=False).completed_codes)
    # The historical Cell toggle has no influence on the new policy.
    loaded.config = replace(loaded.config, inherited=replace(
        loaded.config.inherited, query_source=not loaded.config.inherited.query_source
    ))
    assert torch.equal(out.completed_codes, loaded(ep, decode=False).completed_codes)


@pytest.mark.parametrize("force_chunks", [False, True])
def test_recomputation_chunks_outputs_and_gradients(force_chunks):
    # FP64 is solely an algebraic reference for forced tiny batch chunks.
    # FP32 tests the actual recomputation graph with unchanged batch shape.
    dtype = torch.float64 if force_chunks else torch.float32
    model = V8CompletionModel(cfg()).to(dtype)
    other = copy.deepcopy(model)
    configure_memory(other, attention_mib=1, checkpoint_activations=True)
    for module in other.modules():
        if force_chunks and isinstance(module, OMAB):
            module._bounded_bytes = 1
    inputs, truth = episode_data()
    ep = prepare_completion(inputs, model.config)
    a, b = model(ep, decode=False), other(ep, decode=False)
    torch.testing.assert_close(a.completed_codes, b.completed_codes, rtol=1e-5, atol=1e-6)
    completion_loss(a, ep, truth).backward()
    completion_loss(b, ep, truth).backward()
    for (name, p), (other_name, q) in zip(
        model.named_parameters(), other.named_parameters(), strict=True
    ):
        assert name == other_name
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-3, atol=2e-6, msg=name)


@pytest.mark.parametrize("policy", ["legacy_cell_sources", SUPPORT_ROWS_NULL])
@pytest.mark.parametrize("checkpoint_activations", [False, True])
def test_optional_unit_stack_actually_recomputes(policy, checkpoint_activations):
    model = V8CompletionModel(replace(config(), attention_policy=policy))
    calls = []
    hook = model.backbone.unit_blocks[0].ff[0].register_forward_pre_hook(
        lambda *_: calls.append(None)
    )
    configure_memory(model, attention_mib=128, checkpoint_activations=checkpoint_activations)
    inputs, truth = episode_data()
    ep = prepare_completion(inputs, model.config)
    try:
        out = model(ep, decode=False)
        assert len(calls) == 1
        completion_loss(out, ep, truth).backward()
        # Output/gradient equivalence alone cannot detect a bypassed wrapper.
        assert len(calls) == (2 if checkpoint_activations else 1)
    finally:
        hook.remove()


@pytest.mark.parametrize("policy", ["legacy_cell_sources", SUPPORT_ROWS_NULL])
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS unavailable"
))])
def test_deepcopy_after_recomputation_uses_clone_weights_and_gradients(policy, device):
    original = V8CompletionModel(replace(config(), attention_policy=policy)).to(device)
    configure_memory(original, checkpoint_activations=True)
    clone = copy.deepcopy(original)
    inputs, truth = episode_data(device)
    ep = prepare_completion(inputs, clone.config)
    with torch.no_grad():
        before = clone(ep, decode=False).completed_codes.clone()
        for p in (*clone.encoder.parameters(), *clone.backbone.parameters()):
            p.add_(0.005)
        after = clone(ep, decode=False).completed_codes
        assert not torch.equal(before, after)
        assert torch.equal(before, original(ep, decode=False).completed_codes)

    # Rebuilding from the clone's weights is independent of its method wrappers.
    reference = V8CompletionModel(clone.config).to(device)
    reference.load_state_dict(clone.state_dict())
    with torch.no_grad():
        assert torch.equal(after, reference(ep, decode=False).completed_codes)
    actual, expected = clone(ep, decode=False), reference(ep, decode=False)
    torch.testing.assert_close(actual.completed_codes, expected.completed_codes)
    completion_loss(actual, ep, truth).backward()
    completion_loss(expected, ep, truth).backward()
    assert all(p.grad is None for p in original.parameters())
    for (name, p), (other_name, q) in zip(
        clone.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == other_name
        if q.grad is None:
            assert p.grad is None
        else:
            assert p.grad is not None and torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-3, atol=2e-6, msg=name)


def test_migration_preserves_weights_and_legacy_serialization(tmp_path):
    donor = V8CompletionModel(config()).eval()
    assert "attention_policy" not in donor.config.as_dict()
    assert V8Config.from_dict(donor.config.as_dict()) == donor.config
    rng = torch.get_rng_state().clone()
    model = with_supervised_attention(donor)
    assert torch.equal(rng, torch.get_rng_state())
    assert not model.training
    assert model.config.attention_policy == SUPPORT_ROWS_NULL
    assert model.migration_receipt["new_keys"] == []
    for key, value in donor.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    path = tmp_path / "supervised.pt"
    save_weights(model, path)
    assert load_weights(path).config == model.config
    with pytest.raises(ValueError, match="already enabled"):
        with_supervised_attention(model)
    with pytest.raises(ValueError, match="attention_policy"):
        replace(model.config, attention_policy="unknown")


def test_multitarget_and_unassigned_rows_are_rejected():
    from .test_completion import data
    with pytest.raises(ValueError, match="exactly one"):
        prepare_completion(data(dtype=torch.float32)[0], cfg())
    inputs, _ = episode_data()
    visible = inputs.visible.clone()
    visible[0, 2] = False
    with pytest.raises(ValueError, match="each supervised row"):
        prepare_completion(replace(inputs, visible=visible), cfg())
