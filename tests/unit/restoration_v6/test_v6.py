"""Supervision, leakage, parameter compatibility and backward contracts for V6."""

import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration.contracts import (
    ColumnSchema,
    RestorationInput,
    RestorationRequest,
    make_episode,
)
from tabu_lab.models.restoration_v6 import V6Model, score_training_episode
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v53.training import (
    prepare_training_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration_v55 import BackboneConfig, V55Config, V55Model


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260927)
        yield
    torch.set_num_threads(threads)


def model(mode="target_only"):
    config = V55Config(
        backbone=BackboneConfig(layers=1, ff_width=24, slots=4),
        unit_layers=0, regression_width=5, center_chunk_size=2,
    )
    return V6Model(config, supervision=mode).double()


def episode(target=0, *, missing=False):
    schema = (ColumnSchema("number", "numeric"), ColumnSchema("class", "nominal", 3),
              ColumnSchema("rank", "ordinal", 3, order=(2, 0, 1)))
    values = (torch.tensor([-2., 0., 1., 3., 2., -1., .5, 1.5, 2.5]).double(),
              torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2]),
              torch.tensor([0, 1, 2, 0, 1, 2, 2, 0, 1]))
    observed = torch.ones(9, 3, dtype=torch.bool)
    if missing:
        observed[7, (target + 1) % 3] = False
    query = torch.zeros_like(observed)
    query[6:, target] = True
    return make_episode(schema, values, observed, query, code_seed=29)


@pytest.mark.parametrize("target", [0, 1, 2])
@pytest.mark.parametrize("mode", ["target_only", "joint_all"])
def test_coverage_decode_and_finite_backward(target, mode):
    m = model(mode)
    inputs, _, truth = episode(target, missing=True)
    score = score_training_episode(m, inputs, truth)
    assert score.query_rows == 3
    assert score.scored_cells == (3 if mode == "target_only" else 8)
    assert bool(torch.isfinite(score.loss))
    score.loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    assert sum(float(g.square().sum()) for g in grads) > 0
    request = RestorationRequest(inputs.query.nonzero())
    out = m(inputs, request)
    column = out.columns[0]
    expected = out.facts[target].answers.decode(
        column.result.encoding / (2 if mode == "joint_all" else 1)
    )
    torch.testing.assert_close(column.decoded, expected)


def test_v55_weights_load_exactly_but_forward_is_different():
    old = V55Model(model().config).double()
    m = model()
    m.load_state_dict(copy.deepcopy(old.state_dict()), strict=True)
    assert dict(old.named_parameters()).keys() == dict(m.named_parameters()).keys()
    for key, value in m.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[key], rtol=0, atol=0)
    inputs, _, _ = episode()
    request = RestorationRequest(inputs.query.nonzero())
    assert not torch.allclose(m(inputs, request).carriers, old(inputs, request).carriers)


@pytest.mark.parametrize("mode", ["target_only", "joint_all"])
def test_query_truth_changes_loss_without_changing_forward(mode):
    m = model(mode)
    inputs, _, truth = episode()
    values = [v.clone() for v in truth.values]
    values[0][6:] += 100
    changed = replace(truth, values=tuple(values))
    first = score_training_episode(m, inputs, truth)
    second = score_training_episode(m, inputs, changed)
    torch.testing.assert_close(first.output.carriers, second.output.carriers, rtol=0, atol=0)
    for a, b in zip(first.output.columns, second.output.columns, strict=True):
        torch.testing.assert_close(a.result.encoding, b.result.encoding, rtol=0, atol=0)
    assert not torch.isclose(first.loss, second.loss)
    assert bool((inputs.values[0][6:] == 0).all())


def test_broadcast_uses_query_seed_preserves_null_and_auxiliary_tokens():
    m = model()
    inputs, _, _ = episode(missing=True)
    prepared = m.prepare(inputs, RestorationRequest(inputs.query.nonzero()))
    base = m.encoder.forward_prepared(inputs, prepared.features)
    captured = []
    handle = m.backbone.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        m.forward_prepared(prepared)
    finally:
        handle.remove()
    actual = captured[0]
    active = inputs.visible | inputs.query
    expected = base[:9, :3] + base[:9, 0, None]
    torch.testing.assert_close(actual[:9, :3][active], expected[active])
    assert bool((actual[:9, :3][~active] == 0).all())
    torch.testing.assert_close(actual[9], base[9])
    torch.testing.assert_close(actual[:9, 3], base[:9, 3])


def test_target_only_reuses_original_query_scorer():
    m = model()
    inputs, request, truth = episode()
    config = V53LossConfig()
    prepared = prepare_training_episode(m, inputs, request, truth, config)
    expected = score_prepared_episode(m, prepared, loss_config=config)
    actual = score_training_episode(m, inputs, truth, request=request, loss_config=config)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    torch.testing.assert_close(actual.per_cell, expected.per_target, rtol=0, atol=0)


@pytest.mark.parametrize("target", [0, 1, 2])
def test_joint_loss_is_row_mean_with_target_type_scaling(target):
    m = model("joint_all")
    inputs, _, truth = episode(target, missing=True)
    score = score_training_episode(m, inputs, truth)
    out = score.output
    row_losses = {6: [], 7: [], 8: []}
    for col in out.columns:
        rows = out.request.targets[col.target_indices, 0]
        expected = out.facts[col.column].answers.encode_targets(truth.values[col.column][rows])
        expected = expected + out.facts[target].answers.encode_targets(truth.values[target][rows])
        squared = (col.result.encoding - expected).square().sum(-1)
        for row, loss in zip(rows.tolist(), squared, strict=True):
            row_losses[row].append(loss)
    expected = torch.stack([torch.stack(v).mean() for v in row_losses.values()]).mean()
    if target != 0:
        expected = expected / 128
    torch.testing.assert_close(score.loss, expected)


def test_invalid_task_and_loss_modes_fail_closed():
    with pytest.raises(ValueError, match="supervision"):
        model("unknown")
    m = model()
    inputs, _, truth = episode()
    query = inputs.query.clone()
    query[6, 1] = True
    bad = RestorationInput(inputs.schema, inputs.values, inputs.visible & ~query, query)
    with pytest.raises(ValueError, match="exactly one"):
        score_training_episode(m, bad, truth)
    with pytest.raises(ValueError, match="Query-only"):
        score_training_episode(m, inputs, truth, loss_config=V53LossConfig(state_weights=None))
    with pytest.raises(ValueError, match="own Query-row"):
        score_training_episode(model("joint_all"), inputs, truth, loss_config=V53LossConfig())


def test_prepared_mutation_is_rejected():
    m = model()
    inputs, _, _ = episode()
    prepared = m.prepare(inputs, RestorationRequest(inputs.query.nonzero()))
    prepared.inputs.values[0].add_(1)
    with pytest.raises(ValueError, match="mutated"):
        m.forward_prepared(prepared)
