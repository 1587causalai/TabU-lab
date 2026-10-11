"""Feature-only learning, random baseline codes, temporal binding and old compatibility."""

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    from_v8,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode

from .test_completion import data, spec
from .test_completion import runtime as runtime


def config(**kwargs):
    return V8Config(spec(**kwargs), codebook_mode="learned_pool_v1")


def mixed(device="cpu", dtype=torch.float32):
    schema = (
        ColumnSchema("number", "numeric"),
        ColumnSchema("class", "nominal", 3),
        ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)),
    )
    values = (
        torch.linspace(-2, 3, 8, device=device, dtype=dtype),
        torch.arange(8, device=device) % 3,
        torch.arange(8, device=device) % 3,
    )
    visible = torch.ones(8, 3, device=device, dtype=torch.bool)
    query = torch.zeros_like(visible)
    query[-1] = True
    inputs, _, truth = make_episode(schema, values, visible, query, code_seed=17)
    return inputs, truth


def test_feature_only_parameters_and_exact_declared_dimensions():
    model = V8CompletionModel(config())
    bank = model.feature_codebook.weight
    assert bank.shape == (10000, 32) and bank.numel() == 320000
    torch.testing.assert_close(bank.norm(dim=-1), torch.ones(10000))
    assert not any("numeric_codebook" in n or "token_codebook" in n
                   for n, _ in model.named_parameters())
    assert model.query_seed.shape == (32,) and model.lift.weight.shape == (128, 32)
    assert model.encoder.blocks[0].attention.q.weight.shape == (32, 16)
    assert model.encoder.blocks[0].attention.v.weight.shape == (16, 16)
    assert model.encoder.blocks[0].offn.config.ff_width == 64
    assert model.config.inherited.code_dim == 64  # donor identity is not rewritten
    assert model.config.codec_family == "learned_pool_v1:C32/8"
    with pytest.raises(ValueError, match="gaussian_ll"):
        V8Config(spec(), readout="signed_linear_attention", codebook_mode="learned_pool_v1")


@pytest.mark.parametrize("family", ["C64/8", "G64"])
def test_baseline_directions_stay_constants_and_decode_with_learned_origin(family):
    model = V8CompletionModel(config(codec=family)).double()
    inputs, _ = mixed(dtype=torch.float64)
    episode = prepare_completion(inputs, model.config)
    assert episode.observed is None
    with torch.no_grad():
        model.feature_codebook.weight.mul_(7).add_(3)
    codec = episode.codec.bind(model.feature_codebook)
    for a, column in enumerate(codec.columns):
        template = episode.codec.template.columns[a]
        assert not template.base.requires_grad
        if template.direction is not None:
            assert not template.direction.requires_grad
            expected = 8.0 if family == "C64/8" else 1.0
            torch.testing.assert_close(template.direction.square().sum(),
                                       template.base.new_tensor(expected))
        if template.codes is not None:
            assert not template.codes.requires_grad
        rows = inputs.visible[:, a]
        x = inputs.values[a][rows]
        torch.testing.assert_close(column.decode(column.encode(x)), x, atol=1e-10, rtol=1e-10)
        if column.kind != "numeric":
            # Feature updates are a common translation, not learned class directions.
            torch.testing.assert_close(
                column.codes[1:] - column.codes[:1],
                template.codes[1:] - template.codes[:1],
            )


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("mps", marks=pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="MPS unavailable"
    )),
])
def test_two_optimizer_steps_refresh_bank_without_stale_graph(device):
    model = V8CompletionModel(config()).to(device)
    inputs, truth = mixed(device)
    episode = prepare_completion(inputs, model.config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    before = model.feature_codebook.weight.detach().clone()
    old_directions = [c.codes.clone() if c.codes is not None else c.direction.clone()
                      for c in episode.codec.template.columns]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(episode)
        loss = completion_loss(output, episode, truth)
        loss.backward()
        grad = model.feature_codebook.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert float(grad.abs().sum()) > 0
        unused = torch.ones(10000, dtype=torch.bool, device=device)
        unused[list(episode.codec.indices)] = False
        assert torch.count_nonzero(grad[unused]) == 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    assert not torch.equal(before, model.feature_codebook.weight)
    for column, old in zip(episode.codec.template.columns, old_directions, strict=True):
        assert torch.equal(column.codes if column.codes is not None else column.direction, old)
    rebound = model(episode)
    assert not torch.equal(output.codec.columns[0].base, rebound.codec.columns[0].base)
    assert model.execution_precision["gaussian_ll"] == "float32"


def test_hidden_truth_isolation_and_same_forward_reference():
    model = V8CompletionModel(config())
    inputs, truth = data(torch.float32)
    changed, changed_truth = data(torch.float32, changed=True)
    episode = prepare_completion(inputs, model.config)
    other = prepare_completion(changed, model.config)
    output, altered = model(episode), model(other)
    assert torch.equal(output.completed_codes, altered.completed_codes)
    assert not torch.equal(completion_loss(output, episode, truth),
                           completion_loss(altered, other, changed_truth))
    old_loss = completion_loss(output, episode, truth).detach()
    # The scoring snapshot is independent of a later bank lookup.
    with torch.no_grad():
        model.feature_codebook.weight.add_(0.5)
    assert torch.equal(old_loss, completion_loss(output, episode, truth).detach())
    with pytest.raises(ValueError, match="same forward"):
        completion_loss(output, other, truth)


def test_feature_gradient_matches_finite_difference_with_frozen_reference():
    model = V8CompletionModel(config()).double()
    inputs, truth = mixed(dtype=torch.float64)
    episode = prepare_completion(inputs, model.config)
    output = model(episode, decode=False)
    mask = episode.inputs.query
    reference = torch.zeros_like(output.completed_codes)
    with torch.no_grad():
        for a, column in enumerate(output.codec.columns):
            reference[mask[:, a], a] = column.encode(truth.values[a][mask[:, a]])
    reference = reference[mask]
    completion_loss(output, episode, truth).backward()
    bank = model.feature_codebook.weight
    index = episode.codec.indices[0]
    direction = torch.randn_like(bank[index])
    direction /= direction.norm()
    analytic = (bank.grad[index] * direction).sum()
    original = bank[index].detach().clone()
    values = []
    with torch.no_grad():
        for sign in (1, -1):
            bank[index].copy_(original + sign * 1e-6 * direction)
            prediction = model(episode, decode=False).states[-1]
            values.append((prediction - reference).square().mean())
        bank[index].copy_(original)
    numerical = (values[0] - values[1]) / 2e-6
    torch.testing.assert_close(analytic, numerical, atol=2e-7, rtol=3e-4)


def test_partial_recoding_keeps_reference_indices_and_random_directions():
    cfg = config()
    inputs, _ = mixed()
    reference = prepare_completion(replace(inputs, code_seed=0), cfg)
    full = prepare_completion(inputs, cfg)
    partial = prepare_completion(inputs, cfg, recode_columns=[1], reference_seed=0)
    for a in range(3):
        expected = full if a == 1 else reference
        assert partial.codec.indices[a] == expected.codec.indices[a]
        actual = partial.codec.template.columns[a]
        target = expected.codec.template.columns[a]
        for name in ("direction", "codes"):
            left, right = getattr(actual, name), getattr(target, name)
            assert left is right is None or torch.equal(left, right)
    with pytest.raises(ValueError, match="valid column"):
        prepare_completion(inputs, cfg, recode_columns=[3])


def test_large_ordinal_domain_is_not_limited_by_feature_vocabulary():
    schema = (ColumnSchema("rank", "ordinal", 10001),)
    inputs, _, _ = make_episode(
        schema, (torch.tensor([0, 1, 2]),), torch.ones(3, 1, dtype=torch.bool),
        torch.tensor([[False], [False], [True]]), code_seed=3,
    )
    plan = prepare_completion(inputs, config()).codec
    assert plan.template.columns[0].codes.shape == (10001, 32)


def test_saved_weights_replay_and_schema_mismatch_rejected(tmp_path):
    model = V8CompletionModel(config())
    inputs, truth = mixed()
    episode = prepare_completion(inputs, model.config, recode_columns=[2])
    completion_loss(model(episode), episode, truth).backward()
    torch.optim.SGD(model.parameters(), lr=1e-4).step()
    path = tmp_path / "v81.pt"
    save_weights(model, path)
    state = torch.load(path, weights_only=True)
    assert state["schema"] == "completion_v81_weights_v1"
    restored = load_weights(path)
    rebuilt = prepare_completion(inputs, restored.config, recode_columns=[2])
    assert episode.codec.indices == rebuilt.codec.indices
    assert torch.equal(model(episode).completed_codes, restored(rebuilt).completed_codes)
    state["schema"] = "completion_v8_weights_v1"
    torch.save(state, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="schema disagrees"):
        load_weights(tmp_path / "bad.pt")
    legacy = V8Config(spec())
    assert "codebook_mode" not in legacy.as_dict()
    assert V8Config.from_dict(legacy.as_dict()) == legacy


def test_migration_copies_only_backbone_and_compatible_seeds():
    donor = V8CompletionModel(V8Config(spec())).double()
    target = from_v8(donor)
    assert target.config.model_version == "v8.1"
    assert target.query_seed.shape == (32,)
    assert all(torch.equal(target.state_dict()[k], donor.state_dict()[k])
               for k in target.migration_receipt["copied_keys"])
    assert "feature_codebook.weight" in target.migration_receipt["new_keys"]
    assert all(k.startswith("backbone.") or k in ("unit_seed", "feature_seed")
               for k in target.migration_receipt["copied_keys"])


def test_32d_inverse_and_recomputation_preserve_gradients():
    from torch.utils.checkpoint import checkpoint

    model = V8CompletionModel(config()).double()
    inputs, truth = mixed(dtype=torch.float64)
    episode = prepare_completion(inputs, model.config)
    raw = torch.randn(4, 3, 32, dtype=torch.float64)
    valid = torch.rand(4, 3) > 0.2
    torch.testing.assert_close(model.encoder.inverse(model.encoder(raw, valid), valid)[valid],
                               raw[valid], atol=1e-10, rtol=1e-10)
    plain = model(episode, decode=False)
    completion_loss(plain, episode, truth).backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    forward, inverse = model.encoder.forward, model.encoder.inverse
    model.encoder.forward = lambda *args: checkpoint(forward, *args, use_reentrant=False)
    model.encoder.inverse = lambda *args: checkpoint(inverse, *args, use_reentrant=False)
    recomputed = model(episode, decode=False)
    completion_loss(recomputed, episode, truth).backward()
    assert torch.equal(plain.completed_codes, recomputed.completed_codes)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.grad, gradients[n], atol=1e-10, rtol=1e-10)
