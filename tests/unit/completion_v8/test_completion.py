"""Integration contracts, including leakage, real inverse and inherited dynamics."""

import pytest
import torch

from tabu_lab.models.completion_v8 import (
    V8CompletionModel,
    V8Config,
    completion_loss,
    from_v7,
    from_v7_checkpoint,
    load_weights,
    prepare_completion,
    save_weights,
)
from tabu_lab.models.completion_v8.encoder import SetAttention, stable_normalize
from tabu_lab.models.restoration.backbone import BackboneConfig
from tabu_lab.models.restoration.contracts import (
    ColumnSchema,
    RestorationInput,
    TruthSidecar,
    make_episode,
)
from tabu_lab.models.restoration_v7 import V7Config, V7Model


@pytest.fixture(autouse=True)
def runtime():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        yield
    torch.set_num_threads(old)


def spec(**kwargs):
    return V7Config.v73(
        **(
            dict(
                backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
                unit_layers=1,
                coupling_blocks=1,
                coupling_hidden=(16,),
                center_chunk_size=4,
            )
            | kwargs
        )
    )


def data(dtype=torch.float64, device="cpu", changed=False):
    schema = (
        ColumnSchema("a", "numeric"),
        ColumnSchema("b", "nominal", 3),
        ColumnSchema("c", "numeric"),
    )
    x = torch.arange(7, dtype=dtype, device=device) / 4
    values = [x.clone(), torch.arange(7, device=device) % 3, x.square()]
    query = torch.zeros(7, 3, dtype=torch.bool, device=device)
    query[4, 0] = query[5, 1] = query[6, 2] = query[6, 0] = True
    observed = torch.ones_like(query)
    observed[1, 1] = False
    if changed:
        values[0][query[:, 0]] += 20
        values[1][query[:, 1]] = 2
        values[2][query[:, 2]] -= 20
    inputs, _, truth = make_episode(schema, tuple(values), observed, query, code_seed=7)
    return inputs, truth


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_whole_table_inverse_and_null(dtype):
    model = V8CompletionModel(V8Config(spec())).to(dtype)
    codes = torch.randn(6, 4, 64, dtype=dtype, requires_grad=True)
    valid = torch.rand(6, 4) > 0.25
    encoded = model.encoder(codes, valid)
    restored = model.encoder.inverse(encoded, valid)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(restored[valid], codes[valid], atol=tolerance, rtol=tolerance)
    assert torch.equal(restored[~valid], torch.zeros_like(restored[~valid]))
    restored.square().sum().backward()
    assert torch.isfinite(codes.grad).all()
    torch.testing.assert_close(
        codes.grad[valid], 2 * codes.detach()[valid], atol=tolerance * 20, rtol=tolerance * 20
    )


def test_axis_contract_and_permutation():
    model = V8CompletionModel(V8Config(spec())).double()
    x = torch.randn(5, 3, 32, dtype=torch.float64)
    valid = torch.ones(5, 3, dtype=torch.bool)
    first, second = model.encoder.blocks[:2]
    torch.testing.assert_close(first.attention_update(x, valid), first.attention(x, valid))
    torch.testing.assert_close(
        second.attention_update(x, valid),
        second.attention(x.transpose(0, 1), valid.T).transpose(0, 1),
    )
    c = torch.randn(5, 3, 64, dtype=torch.float64)
    rp, cp = torch.tensor([3, 1, 0, 4, 2]), torch.tensor([2, 0, 1])
    torch.testing.assert_close(
        model.encoder(c[rp][:, cp], valid[rp][:, cp]), model.encoder(c, valid)[rp][:, cp]
    )


def test_attention_matches_explicit_pairwise_formula():
    attention = SetAttention().double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    valid = torch.rand(2, 7) > 0.3
    raw = torch.where(valid[..., None], x, 0)
    b = torch.nn.functional.layer_norm(raw, (32,), eps=1e-5)
    q, k, v = [torch.nn.functional.silu(p(b)) for p in (attention.q, attention.k, attention.v)]
    q, k = [z / (z.square().sum(-1, keepdim=True) + 1e-6).sqrt() for z in (q, k)]
    pairwise = (q @ k.transpose(-1, -2)).masked_fill(~valid[:, None], 0)
    h = pairwise @ v / 32**0.5
    h = h / (h.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    mass = attention.presence_projection(raw).square().sum(-1, keepdim=True)
    expected = attention.out(mass / (1 + mass) * attention.rms_gain * h)
    expected = torch.where(valid[..., None], expected, 0)
    torch.testing.assert_close(attention(x, valid), expected, atol=1e-12, rtol=1e-10)
    assert torch.equal(attention(x, torch.zeros_like(valid)), torch.zeros_like(x))


@pytest.mark.parametrize("mean_square", [False, True])
def test_normalization_does_not_silently_collapse_at_large_magnitude(mean_square):
    x = torch.randn(4, 32) * 1e30
    reference = stable_normalize(x.double(), mean_square=mean_square)
    actual = stable_normalize(x, mean_square=mean_square)
    assert torch.isfinite(actual).all() and float(actual.norm()) > 1
    torch.testing.assert_close(actual.double(), reference, atol=3e-7, rtol=3e-6)
    zero = torch.zeros(2, 32, requires_grad=True)
    stable_normalize(zero, mean_square=mean_square).sum().backward()
    assert torch.isfinite(zero.grad).all()
    torch.testing.assert_close(zero.grad, torch.full_like(zero, 1000.0))


def test_extreme_qk_remain_nonzero_and_match_fp64():
    attention = SetAttention()
    x = torch.randn(2, 7, 32)
    valid = torch.ones(2, 7, dtype=torch.bool)
    with torch.no_grad():
        attention.q.weight.mul_(1e20)
        attention.k.weight.mul_(1e20)
        actual = attention(x, valid)
        expected = attention.double()(x.double(), valid)
    assert torch.isfinite(actual).all() and float(actual.norm()) > 1
    torch.testing.assert_close(actual.double(), expected, atol=3e-5, rtol=3e-4)


@pytest.mark.parametrize("rounds", [1, 2])
def test_completion_backward_and_hidden_truth_isolation(rounds):
    cfg = V8Config(spec(), rounds)
    model = V8CompletionModel(cfg).double()
    inputs, truth = data()
    changed, changed_truth = data(changed=True)
    episode, altered = prepare_completion(inputs, cfg), prepare_completion(changed, cfg)
    out, other = model(episode), model(altered)
    assert torch.equal(out.completed_codes, other.completed_codes)
    assert torch.equal(out.completed_codes[inputs.visible], episode.observed[inputs.visible])
    assert torch.equal(
        out.completed_codes[~(inputs.visible | inputs.query)],
        torch.zeros(1, 64, dtype=torch.float64),
    )
    loss = completion_loss(out, episode, truth)
    assert not torch.equal(loss, completion_loss(other, altered, changed_truth))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    for prefix in ("encoder.phi", "encoder.blocks", "lift", "backbone", "query_seed"):
        assert (
            sum(
                float(p.grad.abs().sum())
                for n, p in model.named_parameters()
                if n.startswith(prefix)
            )
            > 0
        ), prefix
    before = model.query_seed.detach().clone()
    torch.optim.SGD(model.parameters(), lr=1e-5).step()
    assert not torch.equal(before, model.query_seed)
    assert torch.isfinite(completion_loss(model(episode), episode, truth))


def test_all_predictions_are_assembled_before_one_inverse():
    model = V8CompletionModel(V8Config(spec())).double()
    episode = prepare_completion(data()[0], model.config)
    seen = []
    original = model.encoder.inverse

    def inspect(table, valid):
        seen.append(table.detach().clone())
        return original(table, valid)

    model.encoder.inverse = inspect
    model(episode)
    assert len(seen) == 1 and seen[0].shape == (7, 3, 64)
    codes = episode.observed.clone()
    codes[episode.inputs.query] = model.query_seed
    e = model.encoder(codes, episode.inputs.visible | episode.inputs.query)
    assert torch.equal(seen[0][episode.inputs.visible], e.detach()[episode.inputs.visible])
    for a in range(3):
        assert not torch.equal(
            seen[0][episode.inputs.query[:, a], a], e.detach()[episode.inputs.query[:, a], a]
        )


@pytest.mark.parametrize("query_source", [False, True])
def test_dynamics_migration_exact_weights_and_outputs(query_source, tmp_path):
    donor = V7Model(spec(query_source=query_source)).double()
    model = from_v7(donor)
    before = donor.state_dict()
    for key, value in model.backbone.state_dict().items():
        assert torch.equal(value, donor.rounds[0].backbone.state_dict()[key])
        assert value.data_ptr() != donor.rounds[0].backbone.state_dict()[key].data_ptr()
    inputs, _ = data()
    h = torch.randn(8, 4, 128, dtype=torch.float64)
    sources = inputs.visible | inputs.query if query_source else inputs.visible
    assert torch.equal(
        donor.run_backbone(donor.rounds[0], h, inputs.visible, inputs.query),
        model.backbone(h, sources, inputs.query),
    )
    assert all(torch.equal(value, donor.state_dict()[key]) for key, value in before.items())
    episode = prepare_completion(inputs, model.config)
    path = tmp_path / "weights.pt"
    save_weights(model, path)
    restored = load_weights(path)
    assert torch.equal(model(episode).completed_codes, restored(episode).completed_codes)
    with pytest.raises(FileExistsError):
        save_weights(model, path)


def test_checkpoint_import_and_corruption_rejected(tmp_path):
    from tabu_lab.models.restoration_v7.runner import checkpoint_state

    donor = V7Model(spec()).double()
    state = checkpoint_state(
        donor, torch.optim.Adam(donor.parameters()), step=1, manifest={"test": True}
    )
    path = tmp_path / "donor.pt"
    torch.save(state, path)
    migrated = from_v7_checkpoint(path)
    assert migrated.migration_receipt["donor_step"] == 1
    assert len(migrated.migration_receipt["donor_sha256"]) == 64
    state["manifest"]["test"] = False
    torch.save(state, path)
    with pytest.raises(ValueError, match="digest"):
        from_v7_checkpoint(path)


def test_unshared_donor_requires_selection():
    donor = V7Model(spec(rounds=2, share_rounds=False)).double()
    with pytest.raises(ValueError, match="explicit round"):
        from_v7(donor)
    assert from_v7(donor, round_index=1).migration_receipt["donor_round"] == 1


def test_one_support_and_missing_truth_boundary():
    cfg = V8Config(spec())
    schema = (ColumnSchema("x", "numeric"),)
    visible = torch.tensor([[True], [False]])
    query = ~visible
    inputs = RestorationInput(
        schema, (torch.tensor([3.0, float("nan")], dtype=torch.float64),), visible, query
    )
    episode = prepare_completion(inputs, cfg)
    model = V8CompletionModel(cfg).double()
    out = model(episode)
    assert torch.isfinite(out.completed_codes).all()
    truth = TruthSidecar((torch.tensor([3.0, 4.0]),), torch.tensor([[0], [-1]]))
    with pytest.raises(ValueError, match="held-out observed"):
        completion_loss(out, episode, truth)
    with pytest.raises(ValueError, match="no-support"):
        prepare_completion(
            RestorationInput(schema, inputs.values, torch.zeros_like(visible), query), cfg
        )


def test_natural_missing_query_is_completed_but_excluded_from_loss():
    model = V8CompletionModel(V8Config(spec())).double()
    inputs, truth = data()
    episode = prepare_completion(inputs, model.config)
    output = model(episode)
    states = truth.states.clone()
    states[6, 0] = -1
    values = list(truth.values)
    values[0] = values[0].clone()
    values[0][6] = float("nan")
    missing_truth = TruthSidecar(tuple(values), states)
    actual = completion_loss(output, episode, missing_truth)
    mask = inputs.query & (states == 1)
    expected = completion_loss(output, episode, truth, score_mask=mask)
    assert torch.equal(actual, expected)
    assert torch.isfinite(output.completed_codes[6, 0]).all()
    with pytest.raises(ValueError, match="held-out observed"):
        completion_loss(output, episode, missing_truth, score_mask=inputs.query)


def test_full_completion_gradient_matches_directional_finite_difference():
    model = V8CompletionModel(V8Config(spec())).double()
    inputs, truth = data()
    episode = prepare_completion(inputs, model.config)
    completion_loss(model(episode, decode=False), episode, truth).backward()
    parameters = dict(model.named_parameters())
    names = [
        "query_seed",
        "encoder.blocks.0.attention.out.weight",
        next(n for n in parameters if n.startswith("encoder.phi")),
        next(n for n in parameters if n.startswith("backbone") and n.endswith("q.weight")),
    ]
    for name in names:
        p = parameters[name]
        direction = torch.randn_like(p)
        direction /= direction.norm()
        analytic = (p.grad * direction).sum()
        original = p.detach().clone()
        values = []
        epsilon = 1e-6
        with torch.no_grad():
            for sign in (1, -1):
                p.copy_(original + sign * epsilon * direction)
                values.append(completion_loss(model(episode, decode=False), episode, truth))
            p.copy_(original)
        numerical = (values[0] - values[1]) / (2 * epsilon)
        torch.testing.assert_close(analytic, numerical, atol=2e-7, rtol=2e-4)


def test_donor_row_encoders_are_not_silently_copied():
    from tabu_lab.models.restoration_v7.config import DualStreamConfig, RowReversible64Config

    donor = V7Model(
        spec(value_encoder="row_reversible64", row_reversible64=RowReversible64Config())
    ).double()
    target = from_v7(donor)
    assert not any("row_encoder" in k for k in target.state_dict())
    assert all(
        torch.equal(v, donor.rounds[0].phi.state_dict()[k])
        for k, v in target.encoder.phi.state_dict().items()
    )
    dual = V7Model(spec(value_encoder="row_dual_stream", dual_stream=DualStreamConfig())).double()
    with pytest.raises(ValueError, match="no compatible phi/lift"):
        from_v7(dual)
    migrated = from_v7(dual, copy_value_interface=False)
    assert all(
        torch.equal(v, dual.rounds[0].backbone.state_dict()[k])
        for k, v in migrated.backbone.state_dict().items()
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_fp32_end_to_end():
    model = V8CompletionModel(V8Config(spec())).to("mps")
    inputs, truth = data(torch.float32, "mps")
    episode = prepare_completion(inputs, model.config)
    out = model(episode)
    loss = completion_loss(out, episode, truth)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
