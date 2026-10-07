"""Row dual-stream value embedding: algebra, context, LL/readback and migration."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration import BackboneConfig
from tabu_lab.models.restoration.backbone import OFFN, OMAB, OAttention
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import (
    DualStreamConfig,
    RowDualStreamEncoder,
    V7Config,
    V7Model,
    V7Task,
    checkpoint_state,
    dual_stream_round,
    load_checkpoint,
    make_optimizer,
    prepare_auxiliary_reconstruction,
    prepare_episode,
    prepare_joint_episode,
    reference_values,
    row_dual_stream_from_checkpoint,
    save_checkpoint,
    score_joint,
    score_rounds,
    train_step,
)
from tabu_lab.models.restoration_v7.dual_stream import DualStreamPart, episode_parts

MPS = torch.backends.mps.is_available()


@pytest.fixture(autouse=True)
def small_runtime():
    old = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(2)
        # Earlier tests may enable strict determinism globally. MPS index_put
        # backward has no deterministic implementation; these checks assert
        # finite gradients and tolerance-based parity, not bitwise training.
        torch.use_deterministic_algorithms(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(old)


def config(**overrides):
    values = dict(
        backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
        rounds=2,
        center_chunk_size=4,
        value_encoder="row_dual_stream",
        dual_stream=DualStreamConfig(ff_width=32),
    )
    return V7Config.v73(**(values | overrides))


def task(*, single=False, device="cpu", dtype=torch.float64, changed_truth=False):
    schema = (
        ColumnSchema("x", "numeric"),
        ColumnSchema("c", "nominal", 2),
        ColumnSchema("y", "numeric"),
        ColumnSchema("z", "numeric"),
    )
    x = torch.arange(9, dtype=dtype, device=device) / 3
    values = [x.clone(), torch.arange(9, device=device) % 2, x.square() + x, (x - 1).sin()]
    present = torch.ones(9, 4, dtype=torch.bool, device=device)
    present[1, 3] = present[4, 0] = False  # structural Null Cells
    mask = torch.zeros(9, 4, dtype=torch.bool, device=device)
    mask[6:, 2] = True
    if not single:
        mask[5:7, 0] = True
        mask[7:, 1] = True
    if changed_truth:
        for column in (0, 2):
            values[column][mask[:, column]] += 1000
        values[1][mask[:, 1]] = 1 - values[1][mask[:, 1]]
    inputs, _, truth = make_episode(schema, tuple(values), present, mask, code_seed=9)
    return V7Task(inputs, truth, 4)


def encoder(code_dim=64, heads=4, ff_width=32, dtype=torch.float64, device="cpu"):
    module = RowDualStreamEncoder(
        DualStreamConfig(heads=heads, ff_width=ff_width), code_dim, strict_finite_content=True
    )
    return module.to(device=device, dtype=dtype)


def masks(n, m, device="cpu"):
    active = torch.ones(n, m, dtype=torch.bool, device=device)
    for r, c in ((0, 1), (2, 0), (3, 2)):
        if r < n and c < m:
            active[r, c] = False  # structural Null Cells
    return active, active.clone()


def codes_for(active, dim, dtype=torch.float64):
    codes = torch.randn(*active.shape, dim, dtype=dtype, device=active.device)
    return torch.where(active[..., None], codes, torch.zeros_like(codes))


# Extracted operators ---------------------------------------------------------


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_omab_is_exactly_residual_attention_update_then_residual_offn(strict, dtype):
    cfg = BackboneConfig(width=16, heads=2, ff_width=24)
    omab = OMAB(cfg, strict_finite_content=strict).to(dtype)
    for p in omab.parameters():
        torch.nn.init.normal_(p, std=0.3)
    attention = OAttention(cfg, strict_finite_content=strict).to(dtype)
    ffn = OFFN(cfg).to(dtype)
    weights = omab.state_dict()
    attention.load_state_dict({k: v for k, v in weights.items() if k in attention.state_dict()})
    ffn.load_state_dict({k: v for k, v in weights.items() if k in ffn.state_dict()})
    assert set(attention.state_dict()) | set(ffn.state_dict()) == set(omab.state_dict())
    assert not set(attention.state_dict()) & set(ffn.state_dict())
    x = torch.randn(3, 5, 16, dtype=dtype)
    eligible = torch.rand(3, 5) > 0.3
    full = omab.batched(x, x, eligible)
    residual = x + attention(x, x, eligible)
    assert torch.equal(full, residual + ffn(residual))


def test_omab_parameter_names_and_order_are_unchanged():
    names = [k for k, _ in OMAB(BackboneConfig(width=8, heads=2, ff_width=8)).named_parameters()]
    assert names == [  # recorded from the unmodified baseline snapshot
        "norm_scale", "attention_presence.weight", "ff_presence.weight", "q.weight",
        "k.weight", "v.weight", "out.weight", "ff.0.weight", "ff.2.weight",
    ]


# Config ----------------------------------------------------------------------


def test_default_modelspec_dict_is_unchanged_and_dual_stream_round_trips():
    for old in (V7Config(), V7Config.v73()):
        assert "value_encoder" not in old.as_dict() and "dual_stream" not in old.as_dict()
        assert V7Config.from_dict(old.as_dict()) == old
    cfg = config()
    values = cfg.as_dict()
    assert values["value_encoder"] == "row_dual_stream"
    assert values["dual_stream"] == DualStreamConfig(ff_width=32).as_dict()
    assert V7Config.from_dict(values) == cfg
    assert values["dual_stream"]["block_parameters"] == "independent"
    assert values["dual_stream"]["readback"] == "mean"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(value_encoder="phi_lift"), "require value_encoder"),
        (dict(dual_stream=None), "explicit DualStreamConfig"),
        (dict(value_encoder="other"), "value_encoder must"),
        (dict(backbone=BackboneConfig(width=256, heads=4, ff_width=64, slots=3)), "2 \\* code_dim"),
        (dict(query_source=False), "query_source"),
        (dict(value_map="identity"), "no phi"),
        (dict(dual_stream=DualStreamConfig(heads=3)), "divide code_dim"),
        (dict(codec="G64", code_dim=32), "code_dim 64"),
    ],
)
def test_invalid_dual_stream_configs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        config(**overrides)


@pytest.mark.parametrize(
    "values",
    [
        dict(block_parameters="shared"),
        dict(readback="u_only"),
        dict(assembly="sequential"),
        dict(blocks=0),
        dict(heads=True),
        dict(ff_width=1.5),
        dict(norm_eps=0.0),
        dict(reference_mass=float("inf")),
        dict(tau_presence=True),
    ],
)
def test_dual_stream_config_is_strict(values):
    with pytest.raises(ValueError):
        DualStreamConfig(**values)
    with pytest.raises((ValueError, TypeError)):
        stored = DualStreamConfig().as_dict() | values
        V7Config.from_dict(config().as_dict() | {"dual_stream": stored})


def test_dual_stream_round_has_no_phi_lift_dropout_or_entry_projection():
    model = V7Model(config())
    block = model.rounds[0]
    assert not hasattr(block, "phi") and not hasattr(block, "lift")
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert len(block.encoder.attention) == len(block.encoder.ffn) == 4
    ids = {id(p) for layer in block.encoder.attention for p in layer.parameters()}
    assert len(ids) == 4 * 5  # four independent attention blocks
    assert all(isinstance(m, OAttention) for m in block.encoder.attention)
    assert all(m.strict_finite_content for m in block.encoder.attention)
    assert len(model.rounds) == 1  # recovery modules keep share_rounds
    assert len(V7Model(config(share_rounds=False)).rounds) == 2


# Inverse ---------------------------------------------------------------------


def _round_trip(device, dtype, atol):
    torch.manual_seed(3)
    module = encoder(device=device, dtype=dtype)
    active, sources = masks(5, 4, device)
    # Arbitrary, nonzero, unequal branches, including nonzero Null coordinates.
    state = torch.randn(5, 4, 128, dtype=dtype, device=device)
    with torch.no_grad():
        forward = module.forward_state(state, active, sources)
        return _measure_round_trip(module, state, forward, active, sources, device, dtype, atol)


@torch.no_grad()
def _measure_round_trip(module, state, forward, active, sources, device, dtype, atol):
    u0, v0 = module.inverse(forward, active, sources)
    back = torch.cat((u0, v0), -1)
    assert back.device.type == device and forward.device.type == device
    assert not torch.equal(forward[active], state[active])
    assert torch.equal(forward[~active], state[~active])  # Null rows are identity
    state_error = float((back - state).abs().max())
    codes = codes_for(active, 64, dtype)
    encoded = module(codes, active, sources)
    assert torch.equal(encoded[~active], torch.zeros_like(encoded[~active]))
    code_error = float((module.readback(encoded, active, sources) - codes).abs().max())
    assert state_error <= atol and code_error <= atol, (state_error, code_error)
    return state_error, code_error


def test_cpu_fp64_round_trip_with_mixed_masks_and_arbitrary_branches():
    _round_trip("cpu", torch.float64, 1e-12)


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_mps_fp32_round_trip_with_mixed_masks_and_arbitrary_branches():
    _round_trip("mps", torch.float32, 5e-5)


def test_coupling_matches_written_block_recurrence():
    module = encoder()
    active, sources = masks(4, 3)
    codes = codes_for(active, 64)
    u = v = codes
    for attention, ffn in zip(module.attention, module.ffn, strict=True):
        a = attention.attention_update(v, v, sources)[0]
        u = u + torch.where(active[..., None], a, 0)
        f = ffn.local_update(u)
        v = v + torch.where(active[..., None], f, 0)
    assert torch.equal(module(codes, active, sources), torch.cat((u, v), -1))


def test_left_inverse_and_mean_readback_for_unequal_branches():
    module = encoder()
    active, sources = masks(4, 3)
    a, b = codes_for(active, 64), codes_for(active, 64)
    state = module.forward_state(torch.cat((a, b), -1), active, sources)
    u0, v0 = module.inverse(state, active, sources)
    assert float((u0 - v0).detach()[active].abs().min()) > 0
    torch.testing.assert_close(u0, a, rtol=0, atol=1e-12)
    torch.testing.assert_close(v0, b, rtol=0, atol=1e-12)
    mean = module.readback(state, active, sources)
    torch.testing.assert_close(mean, (a + b) / 2, rtol=0, atol=1e-12)


def test_nonzero_null_codes_are_rejected():
    module = encoder()
    active, sources = masks(3, 3)
    codes = torch.randn(3, 3, 64, dtype=torch.float64)
    with pytest.raises(ValueError, match="Null"):
        module(codes, active, sources)
    with pytest.raises(ValueError, match="active"):
        module(codes_for(active, 64), active, sources | ~active)


def test_encoder_is_deterministic_between_train_and_eval_modes():
    module = encoder()
    active, sources = masks(4, 3)
    codes = codes_for(active, 64)
    first = module.train()(codes, active, sources)
    assert torch.equal(first, module.eval()(codes, active, sources))


# Row context -----------------------------------------------------------------


def test_row_context_reaches_other_cells_and_never_other_rows():
    module = encoder()
    active, sources = masks(5, 4)
    codes = codes_for(active, 64)
    base = module(codes, active, sources).detach()
    changed = codes.clone()
    changed[1, 2] += torch.randn(64, dtype=torch.float64)
    after = module(changed, active, sources).detach()
    for column in (0, 3):  # other active Cells of the same row
        assert float((after[1, column] - base[1, column]).abs().max()) > 1e-6
    other_rows = torch.arange(5) != 1
    assert torch.equal(after[other_rows], base[other_rows])


def test_non_source_cells_do_not_influence_their_row():
    module = encoder()
    active, _ = masks(3, 4)
    sources = active.clone()
    sources[1, 2] = False
    codes = codes_for(active, 64)
    changed = codes.clone()
    changed[1, 2] *= 3
    base, after = module(codes, active, sources), module(changed, active, sources)
    assert torch.equal(after[1, [0, 1, 3]], base[1, [0, 1, 3]])


# Recovery --------------------------------------------------------------------


def _episode(item, cfg, single):
    if single:
        return prepare_episode(
            item.inputs,
            donor_seed=item.donor_seed,
            codec=cfg.codec,
            numeric_preprocessing=cfg.numeric_preprocessing,
        )
    return prepare_joint_episode(item.inputs, donor_seed=item.donor_seed, config=cfg)


def _score(output, episode, item, cfg, single):
    if single:
        return score_rounds(output, episode, reference_values(episode, item.truth), cfg)
    return score_joint(output, episode, item.truth, cfg)


@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("query_init", ["donor", "seed"])
def test_forward_loss_backward_reach_encoder_dynamics_and_seeds(single, query_init):
    cfg = config(query_init=query_init)
    model = V7Model(cfg).double()
    item = task(single=single)
    episode = _episode(item, cfg, single)
    output = model(episode)
    assert len(output.states) == cfg.rounds and output.decoded is not None
    assert all(s.shape[-1] == 64 for s in output.states)
    score = _score(output, episode, item, cfg, single)
    score.loss.backward()
    for name, p in model.named_parameters():
        if name.startswith("query_seed"):
            continue
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name
        # Feature carriers are never AxialBackbone sources (as on phi_lift),
        # so feature_seed legitimately receives an exact-zero gradient here.
        if ".encoder." in name or name.endswith("unit_seed"):
            assert float(p.grad.abs().sum()) > 0, name


def test_single_column_episode_is_the_one_column_joint_case():
    cfg = config()
    model = V7Model(cfg).double()
    item = task(single=True)
    single, joint = _episode(item, cfg, True), _episode(item, cfg, False)
    first, second = model(single), model(joint)
    for a, b in zip(first.states, second.states, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(
        _score(first, single, item, cfg, True).loss, _score(second, joint, item, cfg, False).loss
    )


def test_column_order_and_hidden_truth_do_not_change_forward():
    cfg = config()
    model = V7Model(cfg).double()
    item = task()
    episode = _episode(item, cfg, False)
    reordered = replace(episode, columns=tuple(reversed(episode.columns)))
    hidden = _episode(task(changed_truth=True), cfg, False)
    assert torch.equal(hidden.observed, episode.observed)
    base = model(episode)
    for other in (model(reordered), model(hidden)):
        for a, b in zip(base.states, other.states, strict=True):
            assert torch.equal(a, b)


def _round_inputs(model, episode):
    parts, rows, cols, donors = episode_parts(episode)
    observed = episode.observed.to(torch.float64)
    return parts, rows, cols, observed, observed[donors, cols]


def test_query_only_assembly_and_writeback_preserve_non_query_state():
    cfg = config()
    model = V7Model(cfg).double()
    episode = _episode(task(), cfg, False)
    parts, rows, cols, observed, current = _round_inputs(model, episode)
    inputs = episode.inputs
    result = dual_stream_round(
        model, model.rounds[0], observed, current, rows, cols, parts, inputs.visible, inputs.query
    )
    query = inputs.query
    assert torch.equal(result.assembled[~query], result.encoded[~query])
    assert torch.equal(result.assembled[rows, cols], result.predicted)
    assert not torch.equal(result.predicted, result.encoded[rows, cols])
    # Whole-row readback moves visible candidates; the writeback ignores them.
    active = inputs.visible | query
    readback = model.rounds[0].encoder.readback(result.assembled, active, active)
    visible = inputs.visible
    assert float((readback[visible] - observed[visible]).detach().abs().max()) > 1e-8
    torch.testing.assert_close(result.state, readback[rows, cols], rtol=0, atol=0)
    assert torch.equal(episode.observed.to(torch.float64), observed)


def test_assembly_is_simultaneous_not_sequential():
    """A column-by-column write would make later columns read updated Queries."""
    cfg = config()
    model = V7Model(cfg).double()
    episode = _episode(task(), cfg, False)
    parts, rows, cols, observed, current = _round_inputs(model, episode)
    vis, query = episode.inputs.visible, episode.inputs.query
    block = model.rounds[0]
    full = dual_stream_round(model, block, observed, current, rows, cols, parts, vis, query)
    for part in parts:
        alone = dual_stream_round(
            model, model.rounds[0], observed, current, rows, cols, (part,), vis, query
        )
        # Same Z/h snapshot gives the same column LL prediction ...
        torch.testing.assert_close(
            alone.predicted[part.positions], full.predicted[part.positions], rtol=0, atol=0
        )
        assert torch.equal(alone.encoded, full.encoded)


# Gradients -------------------------------------------------------------------


def test_encoder_and_readback_gradcheck_fp64():
    module = encoder(code_dim=8, heads=2, ff_width=12)
    active, sources = masks(3, 4)
    codes = codes_for(active, 8).requires_grad_()

    def encode(c):
        return module(torch.where(active[..., None], c, 0), active, sources)

    assert torch.autograd.gradcheck(encode, (codes,), eps=1e-6, atol=1e-7)
    state = torch.randn(3, 4, 16, dtype=torch.float64)
    state = torch.where(active[..., None], state, 0).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda s: module.readback(s, active, sources), (state,), eps=1e-6, atol=1e-7
    )
    # Parameter path: directional derivative of the readback w.r.t. all weights.
    def readback_of_scaled_encoding():
        return module.readback(module(codes.detach(), active, sources) * 1.1, active, sources)

    _parameter_directional_check(module, readback_of_scaled_encoding)


def _parameter_directional_check(module, function, eps=1e-6):
    params = [p for p in module.parameters() if p.requires_grad]
    directions = [torch.randn_like(p) for p in params]
    weights = None

    def objective():
        nonlocal weights
        value = function()
        if weights is None:
            weights = torch.randn_like(value)
        return (value * weights).sum()

    loss = objective()
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    analytic = sum(
        float((g * d).sum()) for g, d in zip(grads, directions, strict=True) if g is not None
    )
    with torch.no_grad():
        for p, d in zip(params, directions, strict=True):
            p.add_(eps * d)
        plus = float(objective())
        for p, d in zip(params, directions, strict=True):
            p.sub_(2 * eps * d)
        minus = float(objective())
        for p, d in zip(params, directions, strict=True):
            p.add_(eps * d)
    numeric = (plus - minus) / (2 * eps)
    assert analytic != 0
    assert abs(analytic - numeric) <= 1e-6 * max(1.0, abs(numeric)), (analytic, numeric)
    return analytic, numeric


def test_support_responses_depend_on_other_query_cells_in_their_row():
    cfg = config()
    model = V7Model(cfg).double()
    episode = _episode(task(), cfg, False)
    parts, rows, cols, observed, current = _round_inputs(model, episode)
    vis, query = episode.inputs.visible, episode.inputs.query
    part_c = next(p for p in parts if p.column == 1)
    # Row 5 is a visible support of column 1 and holds a Query in column 0.
    q_pos = int(((rows == 5) & (cols == 0)).nonzero())
    assert 5 in part_c.supports.tolist()
    current = current.clone().requires_grad_()
    result = dual_stream_round(
        model, model.rounds[0], observed, current, rows, cols, parts, vis, query
    )
    responses = result.encoded[part_c.supports, 1]
    (jac,) = torch.autograd.grad(
        responses[part_c.supports.tolist().index(5)].sum(), current, retain_graph=True
    )
    assert float(jac[q_pos].abs().sum()) > 0
    assert float(jac[torch.arange(len(rows)) != q_pos].abs().sum()) == 0  # only row 5's Query
    (no_query,) = torch.autograd.grad(responses[part_c.supports.tolist().index(0)].sum(), current)
    assert float(no_query.abs().sum()) == 0  # row 0 holds no Query
    # Finite difference of column 1's LL prediction along that Query cell.
    direction = torch.zeros_like(current)
    direction[q_pos] = torch.randn(64, dtype=torch.float64)
    weights = torch.randn(len(part_c.positions), 128, dtype=torch.float64)

    def prediction(state):
        block = model.rounds[0]
        out = dual_stream_round(model, block, observed, state, rows, cols, parts, vis, query)
        return (out.predicted[part_c.positions] * weights).sum()

    (grad,) = torch.autograd.grad(prediction(current), current)
    analytic = float((grad * direction).sum())
    eps = 1e-6
    with torch.no_grad():
        plus = prediction(current + eps * direction)
        minus = prediction(current - eps * direction)
    numeric = float(plus - minus) / (2 * eps)
    assert analytic != 0
    assert abs(analytic - numeric) <= 1e-6 * max(1.0, abs(numeric)), (analytic, numeric)


def test_full_round_and_two_round_gradcheck_through_ll_and_readback():
    cfg = config()
    model = V7Model(cfg).double()
    episode = _episode(task(), cfg, False)
    parts, rows, cols, observed, current = _round_inputs(model, episode)
    vis, query = episode.inputs.visible, episode.inputs.query

    def one(state):
        return dual_stream_round(
            model, model.rounds[0], observed, state, rows, cols, parts, vis, query
        ).state

    def two(state):
        return one(one(state))

    start = current.clone().requires_grad_()
    assert torch.autograd.gradcheck(one, (start,), eps=1e-6, atol=1e-6, fast_mode=True)
    assert torch.autograd.gradcheck(two, (start,), eps=1e-6, atol=1e-6, fast_mode=True)
    (cross,) = torch.autograd.grad(two(start).sum(), start)
    assert float(cross.abs().sum()) > 0


def test_multi_round_bptt_parameter_directional_derivative():
    cfg = config(round_loss_rho=0.0)  # only the final round is scored
    model = V7Model(cfg).double()
    item = task()
    episode = _episode(item, cfg, False)
    _parameter_directional_check(
        model, lambda: score_joint(model(episode, decode=False), episode, item.truth, cfg).loss
    )


# Balanced reconstruction -----------------------------------------------------


@pytest.mark.parametrize("single", [True, False])
def test_balanced_reconstruction_uses_real_ll_readback_with_gradients(single):
    cfg = config(loss_mode="balanced_reconstruction")
    model = V7Model(cfg).double()
    item = task(single=single)
    episode = prepare_auxiliary_reconstruction(_episode(item, cfg, single), cfg, seed=5)
    output = model(episode, decode=False)
    plain = V7Model(config()).double()
    plain.load_state_dict(model.state_dict())
    reference_states = plain(_episode(item, config(), single), decode=False).states
    for a, b in zip(output.states, reference_states, strict=True):
        assert torch.equal(a, b)  # auxiliary branch never changes Query writeback
    for round_aux in output.auxiliary_states:
        for part in episode.auxiliary.columns:
            predicted = round_aux[part.column]
            observed = episode.observed[part.rows, part.column]
            assert predicted.shape == observed.shape
            assert float((predicted - observed).detach().abs().max()) > 1e-6
    score = _score(output, episode, item, cfg, single)
    aux_only = sum(
        (round_aux[p.column] - episode.observed[p.rows, p.column]).square().sum()
        for round_aux in output.auxiliary_states
        for p in episode.auxiliary.columns
    )
    encoder_parameters = list(model.rounds[0].encoder.parameters())
    grads = torch.autograd.grad(aux_only, encoder_parameters, retain_graph=True)
    assert all(bool(torch.isfinite(g).all()) for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    score.loss.backward()


def test_auxiliary_assembly_directional_derivative():
    cfg = config(loss_mode="balanced_reconstruction", rounds=1)
    model = V7Model(cfg).double()
    item = task()
    episode = prepare_auxiliary_reconstruction(_episode(item, cfg, False), cfg, seed=5)

    def aux():
        output = model(episode, decode=False)
        return torch.cat([output.auxiliary_states[0][p.column] for p in episode.auxiliary.columns])

    _parameter_directional_check(model, aux)


# Checkpoint, training and migration -----------------------------------------


def test_checkpoint_save_reload_rebuilds_the_graph(tmp_path):
    cfg = config()
    model = V7Model(cfg).double()
    optimizer = make_optimizer(model)
    item = task()
    train_step(model, optimizer, [item, task(single=True)])
    state = checkpoint_state(model, optimizer, step=1, manifest={"run": "unit"})
    save_checkpoint(tmp_path / "dual.pt", state)
    loaded_state = torch.load(tmp_path / "dual.pt", map_location="cpu", weights_only=True)
    rebuilt = V7Model(V7Config.from_dict(loaded_state["config"])).double()
    manifest = {"run": "unit"}
    optimizer = make_optimizer(rebuilt)
    step = load_checkpoint(tmp_path / "dual.pt", rebuilt, optimizer, manifest=manifest)
    assert step == 1
    episode = _episode(item, cfg, False)
    for a, b in zip(model(episode).states, rebuilt(episode).states, strict=True):
        assert torch.equal(a, b)
    other = V7Model(config(rounds=3)).double()
    with pytest.raises(ValueError, match="ModelSpec"):
        load_checkpoint(tmp_path / "dual.pt", other, manifest=manifest)


def test_micro_training_steps_are_finite_and_change_new_and_inherited_weights():
    model = V7Model(config()).double()
    optimizer = make_optimizer(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tasks = [task(), task(single=True)]
    records = [train_step(model, optimizer, tasks, grad_clip_norm=1.0) for _ in range(3)]
    assert all(torch.isfinite(torch.tensor(r.loss)) for r in records)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert any(".encoder." in k for k in changed) and any(".backbone." in k for k in changed)


def _donor(tmp_path, **overrides):
    values = dict(
        backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
        rounds=2,
        coupling_blocks=1,
        coupling_hidden=(16,),
        center_chunk_size=4,
    )
    donor = V7Model(V7Config.v73(**(values | overrides))).double()
    with torch.no_grad():
        for p in donor.parameters():
            p.add_(0.01 * torch.randn_like(p))
    state = checkpoint_state(donor, make_optimizer(donor), step=3, manifest={"donor": True})
    path = tmp_path / "donor.pt"
    save_checkpoint(path, state)
    return donor, path


@pytest.mark.parametrize("share_rounds", [True, False])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_migration_copies_compatible_tensors_exactly_and_lists_everything(
    tmp_path, share_rounds, unit_layers
):
    donor, path = _donor(tmp_path, share_rounds=share_rounds, unit_layers=unit_layers)
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rng = torch.get_rng_state()
    model = row_dual_stream_from_checkpoint(path, expected_sha256=digest)
    assert torch.equal(rng, torch.get_rng_state())  # caller RNG untouched
    receipt = model.transfer_receipt
    donor_state = donor.state_dict()
    copied = {t["key"] for t in receipt["copied_tensors"]}
    expected = {
        k for k in donor_state if ".backbone." in k or k.endswith(("unit_seed", "feature_seed"))
    }
    assert copied == expected
    for key in copied:
        assert torch.equal(model.state_dict()[key], donor_state[key])
    not_copied = {t["key"] for t in receipt["not_copied"]}
    excluded = (".phi.", ".lift.", "query_seed")
    assert not_copied == {k for k in donor_state if any(e in k for e in excluded)}
    new = {t["key"] for t in receipt["new_tensors"]}
    assert new == {k for k in model.state_dict() if ".encoder." in k}
    assert receipt["shape_mismatched"] == [] and receipt["optimizer_resume"] is False
    assert "optimizer" in receipt["ignored_donor_state"]
    target = V7Config.from_dict(receipt["target_config"])
    assert target == model.config and target.value_encoder == "row_dual_stream"
    assert replace(target, value_encoder="phi_lift", dual_stream=None) == donor.config
    assert copied | not_copied == set(donor_state)
    item = task()
    episode = _episode(item, target, False)
    score_joint(model(episode), episode, item.truth, target).loss.backward()


def test_migration_rejects_wrong_digest_and_dual_stream_donor(tmp_path):
    _, path = _donor(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        row_dual_stream_from_checkpoint(path, expected_sha256="0" * 64)
    dual = V7Model(config()).double()
    other = tmp_path / "dual.pt"
    save_checkpoint(other, checkpoint_state(dual, make_optimizer(dual), step=0, manifest={}))
    with pytest.raises(ValueError, match="phi_lift"):
        row_dual_stream_from_checkpoint(other)


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_mps_fp32_migration_forward_backward_and_step(tmp_path):
    donor, path = _donor(tmp_path)
    model = row_dual_stream_from_checkpoint(path, device="mps", dtype=torch.float32)
    receipt = model.transfer_receipt
    assert receipt["target_dtype"] == "torch.float32" and receipt["target_device"].startswith("mps")
    for entry in receipt["copied_tensors"]:
        expected = donor.state_dict()[entry["key"]].float()
        assert torch.equal(model.state_dict()[entry["key"]].cpu(), expected)
        assert entry["max_abs_conversion_error"] < 1e-6
    item = task(device="mps", dtype=torch.float32)
    optimizer = make_optimizer(model)
    record = train_step(model, optimizer, [item], grad_clip_norm=1.0)
    assert torch.isfinite(torch.tensor(record.loss))
    assert all(p.device.type == "mps" for p in model.parameters())


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_mps_fp32_joint_forward_matches_cpu_reference():
    cfg = config()
    model = V7Model(cfg).double()
    item = task()
    reference = model(_episode(item, cfg, False)).states[-1]
    mps = V7Model(cfg).to(device="mps", dtype=torch.float32)
    mps.load_state_dict(model.state_dict())
    mps_item = task(device="mps", dtype=torch.float32)
    state = mps(_episode(mps_item, cfg, False)).states[-1]
    assert state.device.type == "mps"
    torch.testing.assert_close(state.cpu().double(), reference, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(unit_layers=1),
        dict(
            backbone=BackboneConfig(
                width=128, layers=1, heads=4, ff_width=64, slots=3, row_slots=2
            )
        ),
        dict(share_rounds=False, gradient_checkpointing=True),
    ],
)
def test_dual_stream_composes_with_inherited_dynamics_options(overrides):
    cfg = config(**overrides)
    model = V7Model(cfg).double().train()
    item = task()
    episode = _episode(item, cfg, False)
    score = score_joint(model(episode), episode, item.truth, cfg)
    score.loss.backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for name, p in model.named_parameters()
        if ".encoder." in name
    )


def test_v7_version_dual_stream_keeps_its_operator_failure_boundary():
    legacy = V7Config.legacy(
        backbone=BackboneConfig(width=128, layers=1, heads=4, ff_width=64, slots=3),
        query_source=True,
        rounds=2,
        center_chunk_size=4,
        value_encoder="row_dual_stream",
        dual_stream=DualStreamConfig(ff_width=32),
    )
    model = V7Model(legacy).double()
    assert not any(m.strict_finite_content for m in model.rounds[0].encoder.attention)
    item = task()
    episode = _episode(item, legacy, False)
    score_joint(model(episode), episode, item.truth, legacy).loss.backward()


def _fit_config(tmp_path):
    from test_v7_masking_fit import fixture

    path, cfg, _ = fixture(tmp_path)
    cfg["model"]["backbone"]["width"] = 128
    return path, cfg


def _dual_overrides():
    dual_stream = DualStreamConfig(ff_width=16).as_dict()
    return {"value_encoder": "row_dual_stream", "dual_stream": dual_stream}


def test_fit_entry_refuses_value_encoder_switch_under_strict_initialization(tmp_path):
    import yaml

    from tabu_lab.models.restoration_v7.fit import run_fit

    path, cfg = _fit_config(tmp_path)
    parent_config = V7Config.legacy(**cfg["model"])
    parent_path = tmp_path / "parent.pt"
    state = V7Model(parent_config).double().state_dict()
    torch.save({"config": parent_config.as_dict(), "model": state}, parent_path)
    cfg.update(init_checkpoint=str(parent_path), steps=1, model=_dual_overrides())
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="row_dual_stream_from_checkpoint"):
        run_fit(path)
    assert not (tmp_path / "run").exists()


def test_fit_entry_runs_and_resumes_weights_of_an_explicit_dual_stream_model(tmp_path):
    import yaml

    from tabu_lab.models.restoration_v7.fit import run_fit

    path, cfg = _fit_config(tmp_path)
    cfg["model"] |= _dual_overrides()
    cfg["steps"] = 2
    path.write_text(yaml.safe_dump(cfg))
    result = run_fit(path, execute=True)
    saved = torch.load(result["checkpoint"], weights_only=True)
    assert saved["config"]["value_encoder"] == "row_dual_stream"
    assert any(".encoder." in k for k in saved["model"])
    child = dict(cfg, init_checkpoint=str(result["checkpoint"]), output_dir="child", model={})
    path.write_text(yaml.safe_dump(child))
    plan = run_fit(path)["manifest"]
    assert V7Config.from_dict(plan["model"]) == V7Config.from_dict(saved["config"])


def test_auxiliary_plan_with_query_only_config_is_refused():
    """Balanced mode with a query_only config must be refused, not re-targeted."""
    cfg = config()
    balanced = config(loss_mode="balanced_reconstruction")
    episode = prepare_auxiliary_reconstruction(_episode(task(), balanced, False), balanced, seed=1)
    with pytest.raises(ValueError, match="balanced_reconstruction"):
        V7Model(cfg).double()(episode)
    part = DualStreamPart(0, torch.arange(2), torch.arange(1), torch.arange(1))
    assert part.auxiliary_rows is None
