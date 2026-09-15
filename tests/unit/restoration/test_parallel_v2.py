from pathlib import Path

import pytest
import torch

from tabu_lab.models.restoration import LossConfig, RestorationInput, RestorationRequest
from tabu_lab.models.restoration.answers import NumericAnswers
from tabu_lab.models.restoration.end_to_end_checks import example_episode, small_config
from tabu_lab.restoration_pipeline_benchmark import check_pair, compare_tensors, make_pair


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


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
