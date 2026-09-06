from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tabu_lab.models import build_model
from tabu_lab.models.tar import TabUTARModel, TARConfig, TAREpisode, score
from tabu_lab.models.tar.attention import TAROMAB, presence
from tabu_lab.models.tar.checkpoint import load_checkpoint, save_checkpoint
from tabu_lab.models.tar.data import synthetic_episode
from tabu_lab.models.tar.inference import predict_supervised
from tabu_lab.models.tar.terminal import gaussian_weights, local_linear
from tabu_lab.models.tar.training import TARTrainer, TARTrainingConfig
from tabu_lab.models.tar.verification import mixed_fixture


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture
def cfg():
    return TARConfig(width=16, heads=4, ff_width=32, blocks=2, semantic_slots=3, inducing_slots=8)


@pytest.mark.parametrize(
    "inducing,blocks,count", [(True, 15, 54071520), (False, 15, 35593440), (False, 23, 54516960)]
)
def test_exact_default_and_baseline_budget(inducing, blocks, count):
    config = TARConfig(inducing_enabled=inducing, blocks=blocks)
    model = build_model("tabu.tar", config=config, device="meta")
    assert sum(p.numel() for p in model.parameters()) == config.parameter_count == count
    assert ("layers.0.inducing" in dict(model.named_parameters())) == inducing


def test_defaults_match_design():
    source = (
        Path(__file__).resolve().parents[4] / "latex/model-factory/TabU-TAR/design/defaults.json"
    )
    # The package also works without the parent research checkout.
    if source.exists():
        assert TARConfig.from_design_defaults(json.loads(source.read_text())) == TARConfig()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(width=7, heads=2),
        dict(inducing_slots=0),
        dict(blocks=True),
        dict(lambda_feature=-1),
        dict(ll_ridge=0),
        dict(minimum_support=1),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        TARConfig(**kwargs)


@pytest.mark.parametrize("inducing", [True, False])
def test_forward_backward_null_and_fixed_slots(cfg, inducing):
    model = TabUTARModel(replace(cfg, inducing_enabled=inducing))
    episode, truth = mixed_fixture()
    slots = []
    hooks = []
    if inducing:
        for layer in model.layers:
            hooks.append(
                layer.collect.register_forward_pre_hook(
                    lambda m, args: slots.append(args[0].shape[-2])
                )
            )
    output = model(episode)
    loss = score(output, truth)
    loss.backward()
    for hook in hooks:
        hook.remove()
    assert output.carriers.shape == (8, 6, 16)
    assert (bool(slots) and all(n == 8 for n in slots)) if inducing else slots == []
    assert torch.count_nonzero(output.carriers[5:, 3:]) == 0
    assert torch.count_nonzero(output.carriers[4, 2]) == 0
    for parameter in model.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    assert model.response_cell.grad.norm() > 0
    for pred in output.predictions:
        assert pred.status == "ok" and pred.support_count >= 2
        if pred.probabilities is not None:
            torch.testing.assert_close(
                pred.probabilities.sum(), torch.tensor(1.0, dtype=torch.float64)
            )
            assert (pred.probabilities > 0).all()


def test_truth_erasure_and_fail_closed_input(cfg):
    episode, truth = mixed_fixture()
    raw = episode.values.clone()
    raw[~episode.visible] = float("nan")
    erased = TAREpisode.from_table(
        raw, episode.visible, episode.queries, episode.features, codebook_seed=7
    )
    model = TabUTARModel(cfg)
    torch.testing.assert_close(model(erased).responses, model(episode).responses, atol=0, rtol=0)
    with pytest.raises(ValueError, match="physically erased"):
        model(replace(episode, values=raw))
    overlap = episode.visible | episode.queries
    with pytest.raises(ValueError, match="overlap"):
        model(replace(episode, visible=overlap))
    output = model(episode)
    changed = {a: y + 0.1 if a[1] == 0 else y for a, y in truth.items()}
    assert score(output, truth) != score(output, changed)


def test_permutation_equivariance(cfg):
    ep, _ = mixed_fixture()
    model = TabUTARModel(cfg, dtype=torch.float64)
    base = model(ep)
    rows = torch.tensor([4, 0, 2, 1, 3])
    cols = torch.tensor([2, 0, 1])
    reordered = TAREpisode.from_table(
        ep.values[rows][:, cols],
        ep.visible[rows][:, cols],
        ep.queries[rows][:, cols],
        [ep.features[i] for i in cols],
        codebook_seed=ep.codebook_seed,
        codebooks=base.codebooks,
        codebook_classes=base.codebook_classes,
    )
    out = model(reordered)
    torch.testing.assert_close(out.responses, base.responses[rows][:, cols], atol=1e-12, rtol=1e-12)
    original = {p.address: p for p in base.predictions}
    for pred in out.predictions:
        r, a = pred.address
        expected = original[int(rows[r]), int(cols[a])]
        torch.testing.assert_close(pred.value, expected.value, atol=1e-12, rtol=1e-12)


def test_independent_gates_feature_translation_and_gradient(cfg):
    ep, truth = mixed_fixture()
    reference = TabUTARModel(cfg, dtype=torch.float64)
    outputs = {}
    for f, u in ((0, 0), (0, 1), (1, 0), (1, 1)):
        model = TabUTARModel(replace(cfg, lambda_feature=f, lambda_unit=u), dtype=torch.float64)
        model.load_state_dict(reference.state_dict())
        outputs[f, u] = model(ep)
    for u in (0, 1):
        for a, b in zip(outputs[0, u].predictions, outputs[1, u].predictions, strict=True):
            torch.testing.assert_close(a.weights, b.weights, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(a.value, b.value, atol=1e-13, rtol=1e-13)
    assert not torch.allclose(outputs[0, 0].responses, outputs[0, 1].responses)
    loss = score(reference(ep), truth)
    loss.backward()
    assert reference.feature_query.grad.abs().max() < 1e-12
    assert reference.response_feature.grad.abs().max() < 1e-12
    assert reference.unit_query.grad.norm() > 1e-9


def test_chunking_same_global_softmax_and_gradients(cfg):
    torch.manual_seed(32)
    full = TAROMAB(cfg, dtype=torch.float64)
    chunked = TAROMAB(replace(cfg, source_chunk=2, receiver_chunk_rows=2), dtype=torch.float64)
    chunked.load_state_dict(full.state_dict())
    x = torch.randn(5, 16, dtype=torch.float64, requires_grad=True)
    s = torch.randn(7, 16, dtype=torch.float64, requires_grad=True)
    xc = x.detach().clone().requires_grad_()
    sc = s.detach().clone().requires_grad_()
    y = full(x, s)
    yc = chunked(xc, sc)
    torch.testing.assert_close(y, yc, atol=1e-12, rtol=1e-12)
    y.square().sum().backward()
    yc.square().sum().backward()
    for a, b in [(x.grad, xc.grad), (s.grad, sc.grad)] + [
        (p.grad, q.grad) for p, q in zip(full.parameters(), chunked.parameters(), strict=True)
    ]:
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_presence_small_nonzero_and_masked_sources(cfg):
    x = torch.full((2, 16), 3e-8, requires_grad=True)
    mass = presence(x)
    assert (mass > 0).all()
    mass.sum().backward()
    assert (x.grad != 0).all()
    op = TAROMAB(cfg)
    receiver = torch.ones(2, 16)
    sources = torch.ones(3, 16)
    masked = op(receiver, sources, source_mask=torch.zeros(3, dtype=torch.bool))
    empty = op(receiver, sources[:0])
    torch.testing.assert_close(masked, empty, rtol=0, atol=0)
    assert torch.count_nonzero(op(torch.zeros(2, 16), sources)) == 0
    assert (
        torch.count_nonzero(op(receiver, sources, null_mask=torch.ones(2, dtype=torch.bool))) == 0
    )


def test_empty_column_suppresses_inducing_seed_residual(cfg):
    ep, _ = mixed_fixture()
    v = ep.visible.clone()
    v[:, 1] = False
    ep = TAREpisode.from_table(ep.values, v, ep.queries, ep.features)
    model = TabUTARModel(cfg)
    seen = []
    hook = model.layers[0].read.register_forward_pre_hook(
        lambda m, args: seen.append(args[1].detach().clone())
    )
    out = model(ep)
    hook.remove()
    assert torch.count_nonzero(torch.cat(seen, dim=0)[1]) == 0
    assert [p.status for p in out.predictions if p.address[1] == 1] == ["insufficient-support"]
    with pytest.raises(ValueError, match="unsupported"):
        score(out, {p.address: 0 for p in out.predictions})


def test_ll_matches_augmented_system_and_is_differentiable():
    torch.manual_seed(21)
    delta = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn(5, dtype=torch.float64)
    alpha = gaussian_weights(delta, 1.0)
    eta = local_linear(delta, alpha, y, 0.01)
    X = torch.cat((torch.ones(2, 5, 1, dtype=torch.float64), delta), dim=-1)
    penalty = torch.diag(torch.tensor([0.0, 0.01, 0.01, 0.01], dtype=torch.float64))
    normal = X.transpose(-1, -2) @ (alpha[..., None] * X) + penalty
    rhs = X.transpose(-1, -2) @ (alpha * y)[..., None]
    expected = torch.linalg.solve(normal, rhs)[:, 0, 0]
    torch.testing.assert_close(eta, expected, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradcheck(
        lambda d: local_linear(d, gaussian_weights(d, 1.0), y, 0.01), (delta,)
    )


def test_supervised_test_rows_are_independent(cfg):
    ep, _ = mixed_fixture()
    model = TabUTARModel(cfg)
    train = ep.values[:3]
    visible = torch.ones_like(train, dtype=torch.bool)
    test = torch.tensor([[3.0, 1.0, 1.0], [5.0, 0.0, 2.0]])
    testmask = torch.ones_like(test, dtype=torch.bool)
    a = predict_supervised(model, train, visible, test, testmask, ep.features, response_column=0)
    test[1] = torch.tensor([999.0, 2.0, 0.0])
    test[0, 0] = -999.0
    b = predict_supervised(model, train, visible, test, testmask, ep.features, response_column=0)
    torch.testing.assert_close(a[0].value, b[0].value, rtol=0, atol=0)
    assert model.training


def test_synthetic_world_and_masks_are_reproducible():
    ep, truth = synthetic_episode(4, rows=(16,), columns=(4,))
    ep2, truth2 = synthetic_episode(4, rows=(16,), columns=(4,))
    assert truth == truth2
    assert torch.equal(ep.values, ep2.values)
    assert torch.equal(ep.queries, ep2.queries)
    assert torch.count_nonzero(ep.values[~ep.visible]) == 0
    for a in ep.queries.nonzero()[:, 1].unique():
        assert ep.visible[:, a].sum() >= 2
    other, _ = synthetic_episode(4, split_id=1, rows=(16,), columns=(4,))
    assert not torch.equal(ep.values, other.values)


def test_checkpoint_exact_optimizer_resume_and_identity(cfg, tmp_path):
    ep, truth = mixed_fixture()
    model = TabUTARModel(cfg)
    trainer = TARTrainer(
        model, TARTrainingConfig(effective_episode_batch=1, warmup_steps=1, optimizer_steps=4)
    )
    trainer.train_step([(ep, truth)])
    path = save_checkpoint(model, tmp_path / "state", trainer=trainer)
    loaded, resumed = load_checkpoint(path, expected_config=cfg, restore_trainer=True)
    assert resumed.step == 1
    trainer.train_step([(ep, truth)])
    resumed.train_step([(ep, truth)])
    for a, b in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="config does not match"):
        load_checkpoint(path, expected_config=replace(cfg, lambda_unit=0))
    with pytest.raises(FileExistsError):
        save_checkpoint(model, path)
    weights = path / "weights.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(ValueError, match="weight digest"):
        load_checkpoint(path)


def test_query_insertion_cannot_become_factual_source(cfg):
    ep, _ = mixed_fixture()
    q = ep.queries.clone()
    q[4, 2] = True
    extended = replace(ep, queries=q)
    model = TabUTARModel(cfg, dtype=torch.float64)
    base = model(ep)
    out = model(extended)
    mask = torch.ones(base.carriers.shape[:2], dtype=torch.bool)
    mask[4, 2] = False
    torch.testing.assert_close(base.carriers[mask], out.carriers[mask], rtol=1e-12, atol=1e-12)
    predictions = {p.address: p for p in out.predictions}
    for pred in base.predictions:
        torch.testing.assert_close(
            pred.value, predictions[pred.address].value, rtol=1e-12, atol=1e-12
        )


def test_source_value_and_query_roles_are_distinct(cfg):
    ep, _ = mixed_fixture()
    model = TabUTARModel(cfg)
    base = model(ep)
    changed = ep.values.clone()
    changed[0, 0] += 2
    output = model(replace(ep, values=changed))
    assert not torch.allclose(base.responses, output.responses)
    # Complete model chunking also keeps the terminal and both axis stages equivalent.
    chunked = TabUTARModel(
        replace(cfg, source_chunk=1, receiver_chunk_rows=1, terminal_query_chunk=1)
    )
    chunked.load_state_dict(model.state_dict())
    other = chunked(ep)
    for a, b in zip(base.predictions, other.predictions, strict=True):
        torch.testing.assert_close(a.value, b.value, atol=2e-6, rtol=2e-6)


def test_presence_overflow_limit():
    large = torch.full((1, 4), 1e200, dtype=torch.float64, requires_grad=True)
    result = presence(large)
    torch.testing.assert_close(result, torch.ones(1, dtype=torch.float64), rtol=0, atol=0)
    result.sum().backward()
    assert torch.isfinite(large.grad).all()


def test_cli_exposes_tar_and_preserves_legacy_command():
    from tabu_lab.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["tar", "inspect"]).tar_command == "inspect"
    assert parser.parse_args(["tar", "verify", "--full"]).full
    assert (
        parser.parse_args(["tabur", "optimize", "--prereg", "x", "--output-root", "y"]).command
        == "tabur"
    )


def test_objective_is_sum_of_branch_means(cfg):
    ep, truth = mixed_fixture()
    out = TabUTARModel(cfg)(ep)
    numeric = []
    discrete = []
    for p in out.predictions:
        y = truth[p.address]
        if p.probabilities is None:
            numeric.append(0.5 * ((p.value - y) / p.scale) ** 2)
        else:
            discrete.append(-torch.log(p.probabilities[int(y)]))
    torch.testing.assert_close(
        score(out, truth), torch.stack(numeric).mean() + torch.stack(discrete).mean()
    )


def test_initialization_is_reproducible_and_preserves_caller_rng(cfg):
    before = torch.get_rng_state().clone()
    a = TabUTARModel(cfg)
    assert torch.equal(before, torch.get_rng_state())
    b = TabUTARModel(cfg)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for seed in (a.cell_query, a.unit_query, a.feature_query, a.layers[0].inducing):
        torch.testing.assert_close(
            seed.norm(dim=-1), torch.ones(seed.shape[:-1]), rtol=1e-6, atol=1e-6
        )
    j = cfg.fourier_frequencies
    torch.testing.assert_close(
        a.continuous.T @ a.continuous, torch.eye(2 * j) / j, atol=1e-6, rtol=1e-6
    )
