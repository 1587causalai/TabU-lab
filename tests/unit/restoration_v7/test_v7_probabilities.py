"""Fixed-code probability readout and full-query classification scoring."""

from dataclasses import replace
import json

import pytest
import torch

from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput, make_episode
from tabu_lab.models.restoration_v7 import (
    G64ColumnCodec,
    V7Config,
    V7Model,
    V7ProtocolError,
    V7Task,
    build_value_codec,
    evaluate_joint_task,
    evaluate_task,
    prepare_episode,
    prepare_joint_episode,
)
from tabu_lab.models.restoration_v7.evaluation import _trajectory
from tabu_lab.models.restoration_v7.model import V7Output


@pytest.fixture(autouse=True)
def bounded_cpu_runtime():
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        yield
    torch.set_num_threads(threads)


def discrete_column(kind, family="C64/8"):
    schema = (ColumnSchema("class", kind, 4, (2, 0, 3, 1) if kind == "ordinal" else None),)
    visible = torch.ones(3, 1, dtype=torch.bool)
    inputs = RestorationInput(schema, (torch.tensor([0, 2, 3]),), visible, ~visible, 18)
    return build_value_codec(inputs, codec=family).columns[0]


@pytest.mark.parametrize("kind", ["nominal", "ordinal"])
@pytest.mark.parametrize("family", ["C64/8", "G64"])
def test_complete_code_probabilities_and_fixed_candidate_order(kind, family):
    column = discrete_column(kind, family)
    # The return order follows the stored dictionary, even if not sorted.
    order = torch.arange(len(column.categories) - 1, -1, -1)
    column = replace(column, categories=column.categories[order], codes=column.codes[order])
    states = torch.cat((column.codes, column.codes.mean(0, keepdim=True)))
    temperature = .7
    distances = (states[:, None] - column.codes[None]).square().sum(-1)
    expected = (-distances / temperature).log_softmax(-1)
    logp = column.log_probabilities(states, temperature=temperature)
    probabilities = column.probabilities(states, temperature=temperature)
    torch.testing.assert_close(logp, expected)
    torch.testing.assert_close(probabilities, logp.exp())
    torch.testing.assert_close(probabilities.sum(-1), probabilities.new_ones(len(states)))
    assert torch.equal(column.categories[probabilities.argmax(-1)], column.decode(states))
    if kind == "ordinal":
        assert len(column.categories) == 4  # includes unobserved declared level 1
    else:
        assert set(column.categories.tolist()) == {0, 2, 3}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("temperature", [1e-310, 5e-324])
def test_tiny_positive_temperature_retains_nearest_candidate(dtype, temperature):
    column = G64ColumnCodec(
        "nominal", torch.zeros(2, dtype=dtype), None, torch.tensor([2, 0]),
        torch.tensor([[0., 0.], [1., 0.]], dtype=dtype), domain_size=3,
    )
    states = torch.tensor([[2., 0.], [.5, 0.]], dtype=dtype)
    logp = column.log_probabilities(states, temperature=temperature)
    assert torch.isneginf(logp[0, 0]) and logp[0, 1] == 0
    torch.testing.assert_close(logp.exp(), states.new_tensor([[0., 1.], [.5, .5]]))


@pytest.mark.parametrize("temperature", [0., -1., float("inf"), float("nan"), True])
def test_probability_temperature_validation(temperature):
    column = discrete_column("nominal")
    with pytest.raises(ValueError, match="temperature"):
        column.probabilities(column.codes, temperature=temperature)


def test_probability_domain_empty_inputs_single_candidate_and_numeric_rejection():
    column = discrete_column("nominal")
    empty = column.codes[:0]
    assert column.probabilities(empty).shape == (0, 3)
    assert column.log_probabilities(empty).shape == (0, 3)
    single = replace(column, categories=column.categories[:1], codes=column.codes[:1])
    assert torch.equal(single.probabilities(column.codes), torch.ones(3, 1, dtype=column.codes.dtype))
    assert torch.equal(single.log_probabilities(column.codes), torch.zeros(3, 1, dtype=column.codes.dtype))
    no_candidates = replace(column, categories=column.categories[:0], codes=empty)
    with pytest.raises(V7ProtocolError, match="no-support"):
        no_candidates.probabilities(column.codes)
    numeric = G64ColumnCodec("numeric", column.base, column.base, None, None)
    with pytest.raises(ValueError, match="nominal or ordinal"):
        numeric.probabilities(column.codes)


def test_nonfinite_distance_is_a_numerical_failure():
    column = discrete_column("nominal")
    with pytest.raises(FloatingPointError, match="class code distances"):
        column.probabilities(torch.full((1, 64), 1e200, dtype=torch.float64))


def model_and_task(*, joint=False, missing_answer=False):
    schema = (
        ColumnSchema("x", "numeric"),
        ColumnSchema("nominal", "nominal", 3),
        ColumnSchema("ordinal", "ordinal", 4, (2, 0, 3, 1)),
    )
    values = (
        torch.tensor([-2., -1., 0., 1., 2., 3.], dtype=torch.float64),
        torch.tensor([0, 2, 0, 2, 1 if missing_answer else 0, 2]),
        torch.tensor([0, 2, 0, 2, 3, 1]),
    )
    query = torch.zeros(6, 3, dtype=torch.bool)
    query[4:, 1] = True
    if joint:
        query[4:, 2] = True
    inputs, _, truth = make_episode(schema, values, torch.ones_like(query), query, code_seed=8)
    config = V7Config.v73(
        backbone=dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
        coupling_blocks=1, coupling_hidden=(12,), rounds=2,
        query_init="donor", query_source=True,
    )
    return V7Model(config).double(), V7Task(inputs, truth, donor_seed=3)


@pytest.mark.parametrize("joint", [False, True])
def test_evaluation_reports_final_probabilities_candidates_temperature_and_log_loss(joint):
    model, task = model_and_task(joint=joint)
    keys = tuple(model.state_dict())
    result = evaluate_task(model, task, probability_temperature=.4)
    episode = prepare_joint_episode(task.inputs, donor_seed=3, config=model.config, admission="inference")
    with torch.no_grad():
        states = model(episode, decode=False).states[-1]
    reports = result.columns if joint else {1: result}
    for part in episode.columns:
        report = reports[part.column]
        column = episode.codec.columns[part.column]
        logp = column.log_probabilities(states[part.positions], temperature=.4)
        truth = task.truth.values[part.column][part.rows]
        selected = logp.gather(1, column._lookup(truth)[:, None]).squeeze(1)
        assert report.status == "ok" and report.probability_temperature == .4
        assert torch.equal(report.candidates, column.categories)
        torch.testing.assert_close(report.log_probabilities, logp)
        torch.testing.assert_close(report.probabilities, logp.exp())
        assert torch.equal(report.candidates[report.probabilities.argmax(-1)], report.predictions[-1])
        assert report.metrics["answer_code_count"] == report.metrics["query_count"] == 2
        assert report.metrics["answer_code_coverage"] == 1.
        assert report.metrics["log_loss"] == pytest.approx(float(-selected.mean()))
        json.dumps(report.metrics, allow_nan=False)
    assert tuple(model.state_dict()) == keys
    # The explicit joint entry point shares the same output protocol.
    direct = evaluate_joint_task(model, task, probability_temperature=.4)
    torch.testing.assert_close(direct.columns[1].probabilities, reports[1].probabilities)


def test_missing_answer_does_not_shrink_probability_metric_denominator():
    model, task = model_and_task(missing_answer=True)
    report = evaluate_task(model, task)
    assert report.status == "no-answer-code" and report.code_losses is None
    assert report.candidates.tolist() == [0, 2]
    assert report.probabilities.shape == (2, 2)
    assert report.probability_temperature == 1.
    assert report.metrics["query_count"] == 2
    assert report.metrics["answer_code_count"] == 1
    assert report.metrics["answer_code_coverage"] == .5
    assert report.metrics["log_loss"] is None
    assert report.metrics["accuracy"] <= .5
    json.dumps(report.metrics, allow_nan=False)


def test_nonrepresentable_true_class_log_probability_fails_without_clipping():
    model, task = model_and_task()
    episode = prepare_episode(task.inputs, donor_seed=3, admission="inference",
                              numeric_preprocessing=model.config.numeric_preprocessing)
    column = episode.codec.columns[1]
    # Both predictions select class 0; the second truth is class 2.
    states = column.codes[:1].expand(2, -1)
    output = V7Output(states, (states,), (), column.decode(states))
    with pytest.raises(FloatingPointError, match="true-class log probabilities"):
        _trajectory(episode, output, task, model.config, 1e-310)


def test_log_loss_uses_log_softmax_when_true_probability_underflows():
    model, task = model_and_task()
    episode = prepare_episode(task.inputs, donor_seed=3, admission="inference",
                              numeric_preprocessing=model.config.numeric_preprocessing)
    column = episode.codec.columns[1]
    states = column.codes[:1].expand(2, -1)
    output = V7Output(states, (states,), (), column.decode(states))
    report = _trajectory(episode, output, task, model.config, .001)
    assert report.probabilities[1, 1] == 0
    assert torch.isfinite(report.log_probabilities).all()
    expected = float((column.codes[0] - column.codes[1]).square().sum()) / .001 / 2
    assert report.metrics["log_loss"] == pytest.approx(expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_probability_readout_tiny_temperature_and_joint_evaluation():
    column = discrete_column("nominal")
    column = replace(column, base=column.base.float().to("mps"),
                     codes=column.codes.float().to("mps"), categories=column.categories.to("mps"))
    # Exercise the host-FP64 fallback for an FP32-unrepresentable temperature.
    logp = column.log_probabilities(column.codes, temperature=1e-310)
    torch.testing.assert_close(logp.exp().cpu(), torch.eye(3))
    assert logp.device.type == "mps" and logp.dtype == torch.float32
    model, task = model_and_task(joint=True)
    inputs = RestorationInput(
        task.inputs.schema,
        tuple(v.float().to("mps") if v.is_floating_point() else v.to("mps") for v in task.inputs.values),
        task.inputs.visible.to("mps"), task.inputs.query.to("mps"), task.inputs.code_seed,
    )
    truth = replace(
        task.truth,
        values=tuple(v.float().to("mps") if v.is_floating_point() else v.to("mps") for v in task.truth.values),
        states=task.truth.states.to("mps"),
    )
    report = evaluate_joint_task(model.float().to("mps"), V7Task(inputs, truth, 3))
    for result in report.columns.values():
        assert result.probabilities.device.type == "mps"
        assert result.probabilities.dtype == torch.float32
        assert result.metrics["answer_code_coverage"] == 1.
        assert result.metrics["log_loss"] >= 0
