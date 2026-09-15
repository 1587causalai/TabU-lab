"""Scorer truth encoding, fixed visible codebooks, and per-target MSE."""

from __future__ import annotations

import inspect

import pytest
import torch

from tabu_lab.models.restoration import (
    CategoricalAnswers,
    NumericAnswers,
    RestorationReadout,
    encoding_mse,
)


def visible_book(width):
    book = torch.zeros(2, width, dtype=torch.float64)
    book[0, :8] = 1
    book[1, -8:] = 1
    return book


@pytest.mark.parametrize("width", [9, 32, 128])
def test_visible_realization_reused_for_supports_truth_and_exact_zero_loss(width):
    book = visible_book(width).requires_grad_()
    codec = CategoricalAnswers.from_visible(
        torch.tensor([4, 2, 4]), torch.tensor([4, 2]), book, domain_size=6
    )
    # Caller order is different from the schema's class order.
    torch.testing.assert_close(codec.classes, torch.tensor([2, 4]))
    torch.testing.assert_close(codec.encoded, book[[0, 1, 0]])
    truth = codec.encode_targets(torch.tensor([2, 4]))
    torch.testing.assert_close(truth, book[[1, 0]])
    torch.testing.assert_close(codec.decode(truth), torch.tensor([2, 4]))
    assert not codec.encoded.requires_grad and not truth.requires_grad
    assert torch.equal(encoding_mse(truth, truth), torch.zeros(2, dtype=torch.float64))
    saved = codec.codebook.clone()
    with torch.no_grad():
        book.zero_()
    torch.testing.assert_close(codec.codebook, saved, rtol=0, atol=0)


@pytest.mark.parametrize("width", [32, 128])
def test_decoder_ties_use_schema_order_and_ignore_support_multiplicity(width):
    book = visible_book(width)
    predictions = torch.stack((book.mean(0), book[0], book[1])).requires_grad_()
    for labels in ([4, 2], [4, 4, 4, 2], [2, 2, 2, 4]):
        codec = CategoricalAnswers.from_visible(
            torch.tensor(labels), torch.tensor([4, 2]), book, domain_size=6
        )
        assert list(inspect.signature(codec.decode).parameters) == ["encoded"]
        decoded = codec.decode(predictions)
        torch.testing.assert_close(decoded, torch.tensor([2, 4, 2]))
        assert decoded.grad_fn is None and not decoded.requires_grad
        # Geometry is already expressed in the restored encoding; no second input.
        with pytest.raises(TypeError):
            codec.decode(predictions, torch.zeros(3, len(labels)))


@pytest.mark.parametrize("width", [32, 128])
@pytest.mark.parametrize("case", ["ll_extrapolation", "common_coordinates"])
def test_nearest_code_preserves_class_difference_at_large_finite_coordinates(width, case):
    book = torch.zeros(2, width, dtype=torch.float64)
    book[:, :7] = 1
    book[0, 7] = 1
    book[1, 8] = 1  # Two differing bits; both codes have squared norm eight.
    codec = CategoricalAnswers.from_visible(
        torch.tensor([0, 1]), torch.tensor([0, 1]), book, domain_size=2
    )
    if case == "ll_extrapolation":
        prediction = ((1 - 1e17) * book[0] + 1e17 * book[1])[None, :]
    else:
        prediction = book[1:2].clone()
        prediction[:, :7] = 1e17
        # Even full code dot products lose the one-coordinate difference here.
        full_products = prediction @ book.T
        assert full_products[0, 0] == full_products[0, 1]
    assert bool(prediction.isfinite().all())
    # Demonstrate the false tie in the former squared-distance computation.
    naive_distances = (prediction[:, None, :] - book).square().sum(-1)
    assert bool(naive_distances.isfinite().all())
    assert naive_distances[0, 0] == naive_distances[0, 1]
    # Shared coordinates cancel exactly; coordinate 8 is closer to code 1.
    assert prediction[0, 8] > prediction[0, 7]
    torch.testing.assert_close(codec.decode(prediction), torch.tensor([1]))


@pytest.mark.parametrize("width", [32, 128])
def test_nearest_code_compares_current_best_after_three_class_ll_extrapolation(width):
    book = torch.zeros(3, width, dtype=torch.float64)
    book[0, 7] = 1
    book[0, 9:16] = 1
    book[1, :8] = 1
    book[2, :7] = 1
    book[2, 8] = 1
    codec = CategoricalAnswers.from_visible(
        torch.tensor([2, 0, 1]), torch.arange(3), book, domain_size=3
    )
    result = RestorationReadout("ll", ridge=1)(
        torch.tensor([[0.0, -40.0, -40.0]], dtype=torch.float64),
        torch.arange(3), codec.encoded,
        support_cells=torch.tensor([[0.0], [-1.0], [1.0]], dtype=torch.float64),
        target_cells=torch.tensor([[1e33]], dtype=torch.float64),
    )
    assert result.status == "ok" and bool(result.encoding.isfinite().all())
    torch.testing.assert_close(result.coefficients.sum(-1), torch.ones(1, dtype=torch.float64))
    fixed_reference = result.encoding @ (book - book[:1]).T
    assert fixed_reference[0, 1] == fixed_reference[0, 2]  # Former false tie.
    assert ((book[2] - book[1]) * result.encoding[0]).sum() == 1
    torch.testing.assert_close(codec.decode(result.encoding), torch.tensor([2]))


def test_nonfinite_nearest_code_comparison_fails_explicitly():
    book = visible_book(32)
    codec = CategoricalAnswers.from_visible(
        torch.tensor([0, 1]), torch.tensor([0, 1]), book, domain_size=2
    )
    prediction = ((book[1] - book[0]) * torch.finfo(torch.float64).max)[None, :]
    assert bool(prediction.isfinite().all())
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(prediction)


@pytest.mark.parametrize("mode", ["nw", "ll"])
@pytest.mark.parametrize("width", [8, 32, 128])
def test_single_visible_class_restores_exact_code_with_zero_mse(mode, width):
    book = torch.zeros(1, width, dtype=torch.float64)
    book[:, :8] = 1
    codec = CategoricalAnswers.from_visible(
        torch.tensor([3]), torch.tensor([3]), book, domain_size=5
    )
    result = RestorationReadout(mode)(
        torch.zeros(2, 1), torch.tensor([0]), codec.encoded,
        support_cells=torch.zeros(1, 1), target_cells=torch.tensor([[20.0], [-5.0]]),
    )
    truth = codec.encode_targets(torch.tensor([3, 3]))
    assert torch.equal(encoding_mse(result.encoding, truth), torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(codec.decode(result.encoding), torch.tensor([3, 3]))


@pytest.mark.parametrize("width", [32, 128])
@pytest.mark.parametrize("unknown", [0, 3, 6, -1])
def test_unknown_truth_fails_whole_target_batch_without_extending_codebook(width, unknown):
    codec = CategoricalAnswers.from_visible(
        torch.tensor([2, 4]), torch.tensor([2, 4]), visible_book(width), domain_size=6
    )
    before = codec.codebook.clone()
    with pytest.raises(ValueError, match="no-answer-code"):
        codec.encode_targets(torch.tensor([2, unknown, 4]))
    torch.testing.assert_close(codec.codebook, before, rtol=0, atol=0)
    torch.testing.assert_close(codec.classes, torch.tensor([2, 4]))


@pytest.mark.parametrize("width", [32, 128])
def test_no_support_and_empty_target_shapes_are_explicit(width):
    empty = torch.empty(0, dtype=torch.long)
    codec = CategoricalAnswers.from_visible(empty, empty, torch.empty(0, width), domain_size=3)
    with pytest.raises(ValueError, match="no-support"):
        codec.decode(torch.zeros(1, width))
    with pytest.raises(ValueError, match="no-support"):
        codec.encode_targets(torch.tensor([1]))
    result = RestorationReadout("ll")(torch.zeros(1, 3), empty, codec.encoded)
    assert result.status == "no-support" and result.encoding is None
    codec = CategoricalAnswers.from_visible(
        torch.tensor([0, 1]), torch.tensor([0, 1]), visible_book(width), domain_size=3
    )
    assert codec.encode_targets(empty).shape == (0, width)
    assert codec.decode(torch.empty(0, width)).shape == (0,)
    assert encoding_mse(torch.empty(0, width), torch.empty(0, width)).shape == (0,)


def test_numeric_truth_uses_visible_statistics_and_detaches_truth():
    values = torch.tensor([1.0, 3.0, 5.0], requires_grad=True)
    codec = NumericAnswers.from_visible(values, epsilon=0.5)
    raw_truth = torch.tensor([-20.0, 60.0], requires_grad=True)
    truth = codec.encode_targets(raw_truth)
    # median 3, half-IQR 1: the floor 0.5 is inactive, scale is exactly 1.
    torch.testing.assert_close(truth[:, 0], raw_truth.double() - 3)
    assert truth.shape == (2, 1) and not truth.requires_grad
    assert not codec.encoded.requires_grad
    torch.testing.assert_close(codec.decode(truth), raw_truth.double())
    torch.testing.assert_close(codec.encode_targets(values), codec.encoded, rtol=0, atol=0)
    assert torch.equal(encoding_mse(truth, truth), torch.zeros(2, dtype=torch.float64))
    empty = NumericAnswers.from_visible(torch.empty(0), epsilon=0.5)
    with pytest.raises(ValueError, match="no-support"):
        empty.encode_targets(raw_truth)
    with pytest.raises(ValueError, match="no-support"):
        empty.decode(torch.zeros(1, 1))


def test_numeric_robust_coordinates_ignore_tail_moves():
    base = NumericAnswers.from_visible(torch.arange(8.0, 17.0), epsilon=1e-9)
    torch.testing.assert_close(base.median, base.median.new_tensor(12.0))
    torch.testing.assert_close(base.scale, base.scale.new_tensor(2.0))
    tailed = torch.arange(8.0, 17.0)
    tailed[-1] = 16000.0
    moved = NumericAnswers.from_visible(tailed, epsilon=1e-9)
    torch.testing.assert_close(moved.median, base.median, rtol=0, atol=0)
    torch.testing.assert_close(moved.scale, base.scale, rtol=0, atol=0)
    torch.testing.assert_close(moved.encoded[:-1], base.encoded[:-1], rtol=0, atol=0)
    pair = NumericAnswers.from_visible(torch.tensor([0.0, 10.0]), epsilon=1e-9)
    torch.testing.assert_close(pair.median, pair.median.new_tensor(5.0))
    torch.testing.assert_close(pair.scale, pair.scale.new_tensor(2.5))


def test_numeric_floor_covers_constant_and_single_support_columns():
    flat = NumericAnswers.from_visible(torch.ones(4), epsilon=0.02)
    assert float(flat.scale) == 0.02
    torch.testing.assert_close(flat.encoded, torch.zeros(4, 1, dtype=torch.float64))
    one = NumericAnswers.from_visible(torch.tensor([7.0]), epsilon=0.02)
    assert float(one.scale) == 0.02
    torch.testing.assert_close(one.decode(one.encoded), torch.tensor([7.0], dtype=torch.float64))


@pytest.mark.parametrize("width", [1, 32, 128])
def test_mse_means_only_answer_coordinates_with_detached_truth(width):
    prediction = torch.zeros(2, width, dtype=torch.float64, requires_grad=True)
    truth = torch.zeros_like(prediction)
    truth[0, 0] = 2
    truth[1] = 3
    truth.requires_grad_()
    losses = encoding_mse(prediction, truth)
    torch.testing.assert_close(losses, torch.tensor([4 / width, 9.0], dtype=torch.float64))
    losses.sum().backward()
    torch.testing.assert_close(prediction.grad, 2 * (prediction - truth) / width)
    assert truth.grad is None


@pytest.mark.parametrize("bad", ["short", "duplicate", "nonbinary", "weight", "nonfinite"])
def test_invalid_identity_codes_fail_explicitly(bad):
    book = visible_book(32)
    if bad == "short":
        book = torch.ones(2, 7)
    elif bad == "duplicate":
        book[1] = book[0]
    elif bad == "nonbinary":
        book[0, 0] = 0.5
    elif bad == "weight":
        book[0, 0] = 0
    else:
        book[0, 0] = float("nan")
    with pytest.raises((ValueError, FloatingPointError)):
        CategoricalAnswers.from_visible(
            torch.tensor([0, 1]), torch.tensor([0, 1]), book, domain_size=3
        )


@pytest.mark.parametrize("shape", [(2, 1), (1, 32), (2, 0)])
def test_mse_rejects_broadcast_or_zero_answer_width(shape):
    prediction = torch.zeros(2, 0) if shape == (2, 0) else torch.zeros(2, 32)
    with pytest.raises(ValueError, match="identical"):
        encoding_mse(prediction, torch.zeros(shape))


def test_invalid_target_and_prediction_tensors_fail_explicitly():
    codec = CategoricalAnswers.from_visible(
        torch.tensor([0, 1]), torch.tensor([0, 1]), visible_book(32), domain_size=3
    )
    with pytest.raises(ValueError, match="int64"):
        codec.encode_targets(torch.tensor([0.0]))
    with pytest.raises(ValueError, match="width"):
        codec.decode(torch.zeros(1, 128))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        codec.decode(torch.full((1, 32), float("nan")))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        encoding_mse(torch.zeros(1, 32), torch.full((1, 32), float("inf")))
