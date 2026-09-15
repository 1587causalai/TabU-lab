"""Prepared replay must preserve gradients, fixed truth boundaries and updates."""

import copy
from dataclasses import replace

import pytest
import torch

from tabu_lab.models.restoration import (
    LossConfig,
    RestorationModel,
    RestorationRequest,
    TruthSidecar,
    prepare_episode,
    score_episode,
    score_prepared_episode,
)
from tabu_lab.models.restoration.answers import CategoricalAnswers, NumericAnswers
from tabu_lab.models.restoration.end_to_end_checks import example_episode, small_config


@pytest.fixture(autouse=True)
def bounded_cpu():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(threads)


def assert_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("kind", ["direct", "inducing"])
@pytest.mark.parametrize("mapping", ["identity128", "rotary32", "mlp32", "mlp256"])
@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_replay_matches_fresh_through_optimizer_updates(kind, mapping, mode):
    episode = example_episode(damage=True)
    inputs, request, truth = episode
    request = RestorationRequest(request.targets.flip(0))
    episode = inputs, request, truth
    torch.manual_seed(91)
    fresh = RestorationModel(small_config(kind, mapping, mode)).double()
    cached = copy.deepcopy(fresh)
    prepared = prepare_episode(cached, *episode)
    opts = [torch.optim.AdamW(m.parameters(), lr=1e-4) for m in (fresh, cached)]
    for step in range(3):
        config = LossConfig() if step == 0 else LossConfig((0., 1., 2., 3.))
        a = score_episode(fresh, *episode, loss_config=config)
        b = score_prepared_episode(cached, prepared, config, decode=True, report=True)
        assert_tree(a.loss, b.loss)
        assert_tree(a.per_target, b.per_target)
        assert_tree(a.by_state, b.by_state)
        for x, y in zip(a.output.columns, b.output.columns, strict=True):
            assert_tree(x.result.encoding, y.result.encoding)
            assert_tree(x.decoded, y.decoded)
        a.loss.backward()
        b.loss.backward()
        assert_tree([p.grad for p in fresh.parameters()], [p.grad for p in cached.parameters()])
        for opt in opts:
            opt.step()
            opt.zero_grad(set_to_none=True)
        assert_tree(fresh.state_dict(), cached.state_dict())
        assert_tree(opts[0].state_dict(), opts[1].state_dict())
    # Rebuild prepared data after restoration into a different model instance.
    resumed = RestorationModel(cached.config).double()
    resumed.load_state_dict(cached.state_dict())
    resumed_plan = prepare_episode(resumed, *episode)
    assert_tree(score_prepared_episode(cached, prepared).loss,
                score_prepared_episode(resumed, resumed_plan).loss)


def test_training_replay_does_not_prepare_decode_or_report(monkeypatch):
    model = RestorationModel(small_config()).double()
    episode = example_episode(damage=True)
    expected = score_episode(model, *episode)
    expected.loss.backward()
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    prepared = prepare_episode(model, *episode)

    def forbidden(*args, **kwargs):
        raise AssertionError("fixed preparation, decoding or host report repeated")

    from tabu_lab.models.restoration import encoding, training
    from tabu_lab.models.restoration import model as model_module
    monkeypatch.setattr(model.encoder, "prepare", forbidden)
    monkeypatch.setattr(encoding, "prepare_features", forbidden)
    monkeypatch.setattr(model_module, "prepare_readout", forbidden)
    monkeypatch.setattr(training, "_preflight", forbidden)
    monkeypatch.setattr(NumericAnswers, "decode_batch", forbidden)
    monkeypatch.setattr(CategoricalAnswers, "decode", forbidden)
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    actual = score_prepared_episode(model, prepared)
    assert actual.by_state is None
    assert all(col.decoded is None for col in actual.output.columns)
    assert_tree(actual.loss, expected.loss)
    actual.loss.backward()
    assert_tree(gradients, [p.grad for p in model.parameters()])


def test_snapshot_owns_data_and_truth_never_reaches_visible_forward():
    model = RestorationModel(small_config()).double()
    inputs, request, truth = example_episode()
    prepared = prepare_episode(model, inputs, request, truth)
    original = score_prepared_episode(model, prepared, decode=True, report=True)
    altered = tuple(v + inputs.query[:, a] * (30 if a == 0 else 0)
                    for a, v in enumerate(truth.values))
    other = prepare_episode(model, inputs, request, TruthSidecar(altered, truth.states))
    changed = score_prepared_episode(model, other)
    for a, b in zip(original.output.columns, changed.output.columns, strict=True):
        assert_tree(a.result.encoding, b.result.encoding)
    assert original.loss != changed.loss
    inputs.values[0].add_(100)
    inputs.visible.logical_not_()
    request.targets.zero_()
    truth.values[0].add_(200)
    truth.states.zero_()
    assert_tree(score_prepared_episode(model, prepared).loss, original.loss)


@pytest.mark.parametrize("field", ["visible", "request", "coordinates", "truth", "indices"])
def test_mutation_of_prepared_tensors_is_rejected(field):
    model = RestorationModel(small_config()).double()
    prepared = prepare_episode(model, *example_episode())
    tensors = {
        "visible": prepared.visible.inputs.visible,
        "request": prepared.visible.request.targets,
        "coordinates": prepared.visible.features.groups[0][1],
        "truth": prepared.scoring.groups[0][1],
        "indices": prepared.visible.layout.packed_rows,
    }
    tensors[field].zero_()
    with pytest.raises(ValueError, match="mutated"):
        score_prepared_episode(model, prepared)


def test_changed_encoder_config_requires_repreparation():
    model = RestorationModel(small_config()).double()
    prepared = prepare_episode(model, *example_episode())
    model.encoder.config = replace(model.encoder.config, epsilon=0.01)
    with pytest.raises(ValueError, match="encoder changed"):
        score_prepared_episode(model, prepared)


def test_invalid_training_request_and_hidden_class_fail_before_forward(monkeypatch):
    model = RestorationModel(small_config()).double()
    inputs, request, truth = example_episode()
    monkeypatch.setattr(model.backbone, "forward", lambda *_: pytest.fail("neural forward"))
    with pytest.raises(ValueError, match="all original observations"):
        prepare_episode(model, inputs, RestorationRequest(request.targets[:-1]), truth)
    # Class 2 now appears only in hidden truth.
    inputs.values[1][inputs.values[1] == 2] = 1
    with pytest.raises(ValueError, match="no-answer-code"):
        prepare_episode(model, inputs, request, truth)


def test_empty_request_and_unsupported_inference_are_preparable():
    from tabu_lab.models.restoration import RestorationInput
    model = RestorationModel(small_config()).double()
    inputs, request, _ = example_episode()
    empty = RestorationRequest(request.targets[:0])
    prepared = model.prepare(inputs, empty)
    output = model.forward_prepared(prepared)
    assert not output.columns
    unsupported = RestorationInput(inputs.schema, inputs.values, inputs.visible & False,
                                   inputs.query | True, inputs.code_seed)
    output = model.forward_prepared(model.prepare(unsupported, request))
    assert all(col.result.status == "no-support" for col in output.columns)


def test_prepared_cli_records_check_and_refuses_overwrite(tmp_path, capsys):
    import json

    from tabu_lab.cli import main
    output = tmp_path / "benchmark.json"
    args = ["restoration", "prepared-benchmark", "--device", "cpu", "--output", str(output)]
    assert main(args) == 0
    result = json.loads(output.read_text())
    assert result["outcome"] == "passed"
    assert all(v == 0 for v in result["max_abs_differences"].values())
    assert all(v > 0 for v in result["median_seconds"].values())
    assert len(result["measurements"]) == 12
    assert len(result["source_sha256"]) >= 16
    with pytest.raises(FileExistsError):
        main(args)
    capsys.readouterr()
