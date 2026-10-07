"""Balanced Query/visible loss, isolation, gradients and public runner checks."""

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    V7Task,
    checkpoint_state,
    evaluate_task,
    make_optimizer,
    prepare_auxiliary_reconstruction,
    prepare_episode,
    prepare_joint_episode,
    reference_values,
    score_joint,
    score_rounds,
    train_step,
)
from tabu_lab.models.restoration_v7.readout import column_shared_ll, evaluate_fitted_ll


@pytest.fixture(autouse=True)
def runtime():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        yield
    torch.set_num_threads(old)


def config(**kw):
    return V7Config.v73(
        **(
            dict(
                backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
                coupling_blocks=1,
                coupling_hidden=(16,),
                rounds=2,
                query_init="donor",
                query_source=True,
                center_chunk_size=4,
                loss_mode="balanced_reconstruction",
            )
            | kw
        )
    )


def task(single=False, dtype=torch.float64, device="cpu"):
    schema = (
        ColumnSchema("x", "numeric"),
        ColumnSchema("c", "nominal", 2),
        ColumnSchema("y", "numeric"),
    )
    x = torch.arange(9, device=device, dtype=dtype) / 3
    values = (x, torch.arange(9, device=device) % 2, x.square() + x)
    query = torch.zeros(9, 3, device=device, dtype=torch.bool)
    query[6:, 2] = True
    if not single:
        query[5:7, 0] = True
        query[7:, 1] = True
    inputs, _, truth = make_episode(schema, values, torch.ones_like(query), query, code_seed=9)
    return V7Task(inputs, truth, 4)


def episode(item, cfg, single=False):
    ep = (
        prepare_episode(
            item.inputs, donor_seed=item.donor_seed, numeric_preprocessing=cfg.numeric_preprocessing
        )
        if single
        else prepare_joint_episode(item.inputs, donor_seed=item.donor_seed, config=cfg)
    )
    return prepare_auxiliary_reconstruction(ep, cfg, seed=23)


def score(out, ep, item, cfg, single):
    return (
        score_rounds(out, ep, reference_values(ep, item.truth), cfg)
        if single
        else score_joint(out, ep, item.truth, cfg)
    )


@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_formula_plan_no_writeback_and_gradients(single, dtype):
    cfg = config(balanced_loss_scale=0.37)
    model = V7Model(cfg).to(dtype=dtype)
    item = task(single, dtype)
    ep = episode(item, cfg, single)
    again = episode(item, cfg, single)
    assert ep.auxiliary.as_dict() == again.auxiliary.as_dict()
    before = ep.observed.clone()
    out = model(ep, decode=False)
    baseline = model(replace(ep, auxiliary=None), decode=False)
    assert torch.equal(ep.observed, before)
    for a, b in zip(out.states, baseline.states, strict=True):
        assert torch.equal(a, b)  # auxiliary never enters the state recurrence
    expected = []
    for t, state in enumerate(out.states):
        loss = state.new_zeros(())
        for part in ep.auxiliary.columns:
            a = part.column
            qr = item.inputs.query[:, a].nonzero(as_tuple=True)[0]
            reference = ep.codec.columns[a].encode(item.truth.values[a][qr]).to(state)
            chi = 128 if item.inputs.schema[a].kind == "numeric" else 1
            qerr = (state[part.query_positions] - reference).square().sum() * chi / 64
            refaux = ep.observed[part.rows, a].to(state)
            predaux = out.auxiliary_states[t][a]
            assert not torch.equal(predaux, refaux)  # not clamped truth pretending to be prediction
            verr = (predaux - refaux).square().sum() * chi / 64
            nq, nv = len(qr), len(part.rows)
            w = nq / (nq + nv)
            assert (1 - w) * nq == pytest.approx(w * nv)
            loss = loss + (1 - w) * qerr + w * verr
        expected.append(loss * 0.37)
    result = score(out, ep, item, cfg, single)
    torch.testing.assert_close(result.round_losses, torch.stack(expected))
    torch.testing.assert_close(result.loss, (expected[0] + 2 * expected[1]) / 3)
    # Auxiliary output is an actual contributor to the loss and to model gradients.
    aux = next(iter(out.auxiliary_states[0].values()))
    aux.retain_grad()
    result.loss.backward()
    assert aux.grad is not None and bool(aux.grad.abs().sum() > 0)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.rounds[0].lift.weight.grad.abs().sum() > 0


def test_hidden_reference_is_only_consumed_by_scorer():
    cfg = config()
    model = V7Model(cfg).double()
    item = task()
    ep = episode(item, cfg)
    out = model(ep)
    values = list(item.truth.values)
    values[2] = values[2].clone()
    values[2][item.inputs.query[:, 2]] += 100
    changed = replace(item, truth=replace(item.truth, values=tuple(values)))
    after = model(ep)
    for a, b in zip(out.states, after.states, strict=True):
        assert torch.equal(a, b)
    for a, b in zip(out.auxiliary_states, after.auxiliary_states, strict=True):
        for col in a:
            assert torch.equal(a[col], b[col])
    assert (
        score_joint(out, ep, item.truth, cfg).loss != score_joint(out, ep, changed.truth, cfg).loss
    )


def test_missing_plan_extreme_ratios_and_bad_rows_fail_explicitly():
    cfg = config()
    item = task(True)
    model = V7Model(cfg).double()
    ep = prepare_episode(item.inputs, donor_seed=4, numeric_preprocessing=cfg.numeric_preprocessing)
    with pytest.raises(ValueError, match="fixed plan"):
        score_rounds(model(ep), ep, reference_values(ep, item.truth), cfg)
    with pytest.raises(ValueError, match="extreme auxiliary ratio"):
        prepare_auxiliary_reconstruction(
            ep, config(auxiliary_per_query=100, auxiliary_min_group_weight=0.4), seed=1
        )
    ep = prepare_auxiliary_reconstruction(ep, cfg, seed=1)
    part = ep.auxiliary.columns[0]
    bad = replace(part, rows=ep.query_rows)
    ep = replace(ep, auxiliary=replace(ep.auxiliary, columns=(bad,)))
    with pytest.raises(ValueError, match="must be visible"):
        model(ep)


@pytest.mark.parametrize("single", [True, False])
def test_public_train_step_records_plan_and_evaluation_stays_query_only(single):
    cfg = config()
    model = V7Model(cfg).double()
    item = task(single)
    record = train_step(model, make_optimizer(model), [item], grad_clip_norm=1)
    assert record.auxiliary_plans[0]["columns"]
    assert record.auxiliary_plans[0]["global_scale"] == 1
    assert all(c["n_query"] > 0 and c["n_aux"] > 0 for c in record.auxiliary_plans[0]["columns"])
    assert evaluate_task(model, item) is not None
    state = checkpoint_state(model, make_optimizer(model), step=1, manifest={})
    assert V7Config.from_dict(state["config"]) == cfg
    legacy = cfg.as_dict()
    for k in [
        "loss_mode",
        "auxiliary_per_query",
        "auxiliary_min_group_weight",
        "balanced_loss_scale",
    ]:
        legacy.pop(k)
    assert V7Config.from_dict(legacy).loss_mode == "query_only"


def test_single_and_joint_interface_match_for_balanced_loss():
    cfg = config()
    model = V7Model(cfg).double()
    item = task(True)
    a, b = episode(item, cfg, True), episode(item, cfg, False)
    assert a.auxiliary.as_dict() == b.auxiliary.as_dict()
    sa = score_rounds(model(a), a, reference_values(a, item.truth), cfg)
    sb = score_joint(model(b), b, item.truth, cfg)
    torch.testing.assert_close(sa.loss, sb.loss, atol=0, rtol=0)


def test_extra_readout_matches_direct_solve_and_backpropagates():
    units = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    features = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    responses = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    supports = torch.arange(5)
    qr = torch.tensor([6, 7])
    aux = torch.tensor([0, 2, 4])
    fit = column_shared_ll(units, supports, responses, features, qr, ridge=0.001, bandwidth=1)
    extra = evaluate_fitted_ll(
        units, supports, responses, features, aux, slope=fit.slope, bandwidth=1
    )
    direct = column_shared_ll(units, supports, responses, features, aux, ridge=0.001, bandwidth=1)
    torch.testing.assert_close(extra, direct.recovered, atol=0, rtol=0)
    extra.square().sum().backward()
    assert all(v.grad is not None and v.grad.abs().sum() > 0 for v in [units, features, responses])


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(loss_mode="unknown"),
        dict(auxiliary_per_query=0),
        dict(auxiliary_per_query=float("nan")),
        dict(auxiliary_min_group_weight=0),
        dict(auxiliary_min_group_weight=0.6),
        dict(balanced_loss_scale=float("inf")),
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.fixture
def mps_algorithm_mode():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # MPS index accumulation does not implement strict deterministic backward.
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("single", [True, False])
def test_mps_fp32_balanced_update_with_checkpointing(single, mps_algorithm_mode):
    cfg = config(gradient_checkpointing=True)
    model = V7Model(cfg).to(device="mps", dtype=torch.float32)
    item = task(single, dtype=torch.float32, device="mps")
    before = model.rounds[0].lift.weight.detach().clone()
    record = train_step(model, make_optimizer(model), [item], grad_clip_norm=1)
    assert torch.isfinite(torch.tensor(record.loss))
    assert record.auxiliary_plans[0]["columns"]
    assert not torch.equal(before, model.rounds[0].lift.weight)
    assert evaluate_task(model, item) is not None


def test_all_visible_default_and_explicit_subsampling_preserve_rng_and_masks():
    item = task()
    cfg = config()
    base = prepare_joint_episode(item.inputs, donor_seed=4, config=cfg)
    rng = torch.get_rng_state().clone()
    full = prepare_auxiliary_reconstruction(base, cfg, seed=31)
    assert torch.equal(rng, torch.get_rng_state())
    assert full.inputs is base.inputs
    for c in full.auxiliary.columns:
        expected = base.inputs.visible[:, c.column].nonzero(as_tuple=True)[0]
        assert torch.equal(c.rows, expected)
    subcfg = config(auxiliary_per_query=0.5)
    first = prepare_auxiliary_reconstruction(base, subcfg, seed=31)
    again = prepare_auxiliary_reconstruction(base, subcfg, seed=31)
    assert first.auxiliary.as_dict() == again.auxiliary.as_dict()
    for c in first.auxiliary.columns:
        assert len(c.rows) == (len(c.query_positions) + 1) // 2
