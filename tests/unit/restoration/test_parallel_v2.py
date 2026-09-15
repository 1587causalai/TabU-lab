from dataclasses import replace
from pathlib import Path

import pytest
import torch

from tabu_lab.models.restoration import (
    LossConfig,
    RestorationInput,
    RestorationModel,
    RestorationRequest,
)
from tabu_lab.models.restoration.answers import NumericAnswers
from tabu_lab.models.restoration.end_to_end_checks import example_episode, small_config
from tabu_lab.restoration_pipeline_benchmark import check_pair, compare_tensors, make_pair


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_pipeline_comparator_rejects_matching_nonfinite_gradients(value):
    values = {"gradient": torch.tensor([value])}
    with pytest.raises(FloatingPointError, match="nonfinite"):
        compare_tensors(values, values)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 21])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batched_numeric_codec_matches_scalar_with_small_and_constant_support(n, dtype):
    values = torch.arange(n, dtype=dtype)
    columns = torch.stack((values, values * 0 + 7, values * 1e8))
    batched = NumericAnswers.from_visible_batch(columns, epsilon=1e-6)
    for row, actual in zip(columns, batched, strict=True):
        expected = NumericAnswers.from_visible(row, epsilon=1e-6)
        for field in ("encoded", "median", "scale"):
            left, right = getattr(actual, field), getattr(expected, field)
            if left is None:
                assert right is None
            else:
                torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numeric_batch_truth_is_detached_and_inverse_decode_gradients_match(dtype):
    supports = torch.tensor([[0., 1., 2.], [5., 5., 5.]], dtype=torch.float64)
    codecs = NumericAnswers.from_visible_batch(supports, epsilon=0.02)
    truth = torch.tensor([[1., 7.], [2., 9.]], dtype=torch.float64, requires_grad=True)
    batched = NumericAnswers.encode_targets_batch(codecs, truth)
    expected = torch.stack([c.encode_targets(truth[i]) for i, c in enumerate(codecs)])
    assert not batched.requires_grad
    torch.testing.assert_close(batched, expected, rtol=0, atol=0)
    predictions = batched.to(dtype).clone().requires_grad_()
    decoded = NumericAnswers.decode_batch(codecs, predictions)
    expected = torch.stack([c.decode(predictions[i]) for i, c in enumerate(codecs)])
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    actual_grad, = torch.autograd.grad(decoded.sum(), predictions, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), predictions)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_numeric_native_codec_wrong_answer_width_is_not_silently_truncated():
    inputs, request, _ = example_episode()
    model = RestorationModel(small_config()).double()
    facts = list(model.encoder.prepare(inputs))
    codec = facts[0].answers
    invalid = replace(codec, encoded=codec.encoded.expand(-1, 2))
    facts[0] = replace(facts[0], answers=invalid)
    with pytest.raises(ValueError, match="one answer coordinate"):
        model._forward_prepared(inputs, request, tuple(facts))


@pytest.mark.parametrize("backbone", ["direct", "inducing"])
@pytest.mark.parametrize("mapping", ["identity128", "rotary32", "mlp32", "mlp256"])
@pytest.mark.parametrize("readout", ["nw", "ll"])
def test_complete_pipeline_parity_with_frozen_previous_code(backbone, mapping, readout):
    root = Path(__file__).resolve().parents[3]
    pair = make_pair(small_config(backbone, mapping, readout), "cpu", root)
    episode = example_episode(damage=True)
    check_pair(pair, episode)
    check_pair(pair, episode, LossConfig((0, 1, 2, 3)))
    # Reordered, uneven requests including Null targets exercise padded gathers.
    inputs, request, _ = episode
    request = RestorationRequest(request.targets[torch.tensor([7, 0, 5, 3, 12])])
    results = []
    for model, _ in pair.values():
        output = model(inputs, request)
        results.append({f"column_{c.column}": c.result.encoding for c in output.columns})
    compare_tensors(*results)
    # Empty evidence remains legal for inference, without fabricated predictions.
    empty = RestorationInput(
        inputs.schema, inputs.values, torch.zeros_like(inputs.visible),
        torch.ones_like(inputs.query), inputs.code_seed,
    )
    for model, _ in pair.values():
        output = model(empty, request)
        assert all(c.result.status == "no-support" for c in output.columns)
        assert bool(torch.isfinite(output.carriers).all())
