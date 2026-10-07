"""Joint recovery invariants and backward compatibility with single-column V7."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration import BackboneConfig
from tabu_lab.models.restoration.contracts import ColumnSchema, make_episode
from tabu_lab.models.restoration_v7 import (
    V7Config,
    V7Model,
    V7Task,
    evaluate_joint_task,
    evaluate_task,
    make_optimizer,
    prepare_episode,
    prepare_joint_episode,
    reference_values,
    score_joint,
    score_rounds,
    train_step,
)


@pytest.fixture(autouse=True)
def small_runtime():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(21)
        yield
    torch.set_num_threads(old)


def config(**overrides):
    values = dict(
        backbone=BackboneConfig(width=64, layers=1, heads=2, ff_width=64, slots=3),
        coupling_blocks=1,
        coupling_hidden=(16,),
        rounds=2,
        unit_layers=1,
        query_init="donor",
        query_source=True,
        center_chunk_size=4,
    )
    return V7Config(**(values | overrides))


def task(*, single=False, device="cpu", dtype=torch.float64, changed_truth=False):
    schema = (
        ColumnSchema("x", "numeric"),
        ColumnSchema("c", "nominal", 2),
        ColumnSchema("y", "numeric"),
    )
    x = torch.arange(9, dtype=dtype, device=device) / 3
    values = [x.clone(), torch.arange(9, device=device) % 2, x.square() + x]
    mask = torch.zeros(9, 3, dtype=torch.bool, device=device)
    mask[6:, 2] = True
    if not single:
        mask[5:7, 0] = True
        mask[7:, 1] = True
    if changed_truth:
        for column in (0, 2):
            values[column][mask[:, column]] += 1000
        values[1][mask[:, 1]] = 1 - values[1][mask[:, 1]]
    inputs, _, truth = make_episode(schema, tuple(values), torch.ones_like(mask), mask, code_seed=9)
    return V7Task(inputs, truth, 4)


@pytest.mark.parametrize("query_source", [False, True])
@pytest.mark.parametrize("query_init", ["seed", "donor"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("row_slots", [0, 3])
def test_one_column_joint_matches_legacy_states_loss_and_gradients(
    query_source, query_init, dtype, row_slots
):
    cfg = config(query_source=query_source, query_init=query_init)
    cfg = replace(cfg, backbone=replace(cfg.backbone, row_slots=row_slots))
    model = V7Model(cfg).to(dtype=dtype)
    item = task(single=True, dtype=dtype)
    old = prepare_episode(item.inputs, donor_seed=item.donor_seed)
    joint = prepare_joint_episode(item.inputs, donor_seed=item.donor_seed, config=cfg)
    old_output, joint_output = model(old), model(joint)
    for a, b in zip(old_output.states, joint_output.states, strict=True):
        assert torch.equal(a, b)
    first = score_rounds(old_output, old, reference_values(old, item.truth), cfg)
    second = score_joint(joint_output, joint, item.truth, cfg)
    assert torch.equal(first.loss, second.loss)
    first.loss.backward()
    grads = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    second.loss.backward()
    for k, p in model.named_parameters():
        assert (p.grad is None) == (grads[k] is None), k
        if p.grad is not None:
            torch.testing.assert_close(p.grad, grads[k], rtol=0, atol=0)


def test_joint_truth_isolation_row_equal_loss_and_shared_backbone():
    model = V7Model(config()).double()
    item, altered = task(), task(changed_truth=True)
    ep = prepare_joint_episode(item.inputs, donor_seed=4, config=model.config)
    other = prepare_joint_episode(altered.inputs, donor_seed=4, config=model.config)
    assert torch.equal(ep.observed, other.observed)
    # Four Query rows; row 6 has two Query cells, row 7 and 8 also have two.
    assert float(ep.cell_weights.sum()) == pytest.approx(1)
    for row in ep.query_rows.unique():
        assert float(ep.cell_weights[ep.query_rows == row].sum()) == pytest.approx(1 / 4)
    calls = []
    handle = model.rounds[0].backbone.register_forward_hook(lambda *args: calls.append(1))
    out = model(ep)
    handle.remove()
    assert len(calls) == model.config.rounds
    other_out = model(other)
    assert all(torch.equal(a, b) for a, b in zip(out.states, other_out.states, strict=True))
    assert (
        score_joint(out, ep, item.truth, model.config).loss
        != score_joint(other_out, other, altered.truth, model.config).loss
    )
    for part in ep.columns:
        assert bool(item.inputs.visible[part.donors, part.column].all())
    report = evaluate_task(model, item)
    assert set(report.columns) == {0, 1, 2}
    assert all(len(r.predictions) == 2 for r in report.columns.values())


@pytest.mark.parametrize("row_slots", [0, 3])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_joint_checkpointing_preserves_forward_and_backward(row_slots, unit_layers):
    cfg = config(unit_layers=unit_layers)
    cfg = replace(cfg, backbone=replace(cfg.backbone, row_slots=row_slots))
    plain = V7Model(cfg).double()
    recomputed = V7Model(replace(plain.config, gradient_checkpointing=True)).double()
    recomputed.load_state_dict(plain.state_dict())
    item = task()
    results = [
        train_step(m, make_optimizer(m), [item], grad_clip_norm=1) for m in (plain, recomputed)
    ]
    assert results[0] == results[1]
    for a, b in zip(plain.parameters(), recomputed.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("share_rounds", [False, True])
def test_joint_seed_and_unshared_rounds_train(share_rounds):
    model = V7Model(config(query_init="seed", share_rounds=share_rounds)).double()
    record = train_step(model, make_optimizer(model), [task(single=True), task()])
    assert record.loss > 0
    assert model.query_seed_numeric.grad is not None
    assert model.query_seed_nominal.grad is not None


@pytest.fixture
def mps_algorithm_mode():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    # MPS index accumulation has no strict deterministic implementation.
    # Restore the caller's process-wide mode after this backend check.
    torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("row_slots", [0, 3])
@pytest.mark.parametrize("unit_layers", [0, 1])
def test_mps_fp32_joint_real_update_and_evaluation(row_slots, unit_layers, mps_algorithm_mode):
    cfg = config(gradient_checkpointing=True, unit_layers=unit_layers)
    cfg = replace(cfg, backbone=replace(cfg.backbone, row_slots=row_slots))
    model = V7Model(cfg).to(device="mps", dtype=torch.float32)
    item = task(device="mps", dtype=torch.float32)
    record = train_step(model, make_optimizer(model), [item], grad_clip_norm=1)
    assert torch.isfinite(torch.tensor(record.loss))
    report = evaluate_joint_task(model, item)
    assert set(report.columns) == {0, 1, 2}


def test_legacy_native_checkpoint_without_new_config_fields_loads(tmp_path):
    from tabu_lab.models.restoration_v7 import checkpoint_state, load_checkpoint, save_checkpoint

    model = V7Model(config(query_source=False))
    state = checkpoint_state(model, make_optimizer(model), step=0, manifest={"legacy": True})
    state["config"].pop("query_source")
    state["config"].pop("gradient_checkpointing")
    state["config"]["backbone"].pop("row_slots")
    state.pop("accelerator_rng")
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, state)
    restored = V7Model(model.config)
    assert load_checkpoint(path, restored, manifest={"legacy": True}) == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
