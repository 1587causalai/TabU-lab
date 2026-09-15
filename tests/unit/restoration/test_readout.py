"""Behavior and algebra checks for Step 4, without a backbone or training run."""

from __future__ import annotations

import pytest
import torch

from tabu_lab.models.restoration import (
    CategoricalAnswers,
    NumericAnswers,
    RestorationReadout,
    encoding_mse,
    unit_kernel_logits,
)
from tabu_lab.models.restoration.verification import (
    check_gradients,
    check_normal_equations,
    check_rotation_lift_mse,
    numeric_example,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def categorical(labels=(0, 1, 0), domain_size=3, width=128):
    book = torch.zeros(2, width, dtype=torch.float64)
    book[0, :8] = 1
    book[1, 8:16] = 1
    return CategoricalAnswers.from_visible(
        torch.tensor(labels), torch.tensor([0, 1]), book, domain_size=domain_size
    )


def test_hand_example_and_independent_normal_equations():
    assert numeric_example()["ll"] > 5  # Positive ridge still permits extrapolation.
    check_normal_equations()


def test_finite_differences_for_both_modes_and_answer_types():
    check_gradients()


@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_shared_unit_kernel_column_supports_self_support_and_multiplicity(mode):
    units = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64)
    logits = unit_kernel_logits(units, units, bandwidth=1)
    expected = -(units[:, None, :] - units[None, :, :]).square().sum(-1) / 2
    torch.testing.assert_close(logits, expected)
    # Different visible addresses per column, including one repeated answer value.
    for rows in (torch.tensor([0, 2]), torch.tensor([0, 1, 2])):
        answers = torch.tensor([[1.0], [1.0], [4.0]], dtype=torch.float64)[rows]
        result = RestorationReadout(mode)(
            logits, rows, answers, support_cells=units[rows], target_cells=units
        )
        torch.testing.assert_close(result.log_weights.exp(), expected[:, rows].softmax(-1))
        assert result.support_count == len(rows)
        assert result.log_weights[0, 0].isfinite()  # Visible self-support retained.


@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_support_permutation_and_target_selection(mode):
    gen = torch.Generator().manual_seed(12)
    units = torch.randn(5, 3, generator=gen, dtype=torch.float64)
    cells = torch.randn(5, 3, generator=gen, dtype=torch.float64)
    answers = torch.randn(3, 2, generator=gen, dtype=torch.float64)
    rows = torch.tensor([0, 2, 4])
    readout = RestorationReadout(mode, ridge=0.2)

    def predict(targets, sources, values, content):
        return readout(
            unit_kernel_logits(targets, sources, bandwidth=1),
            rows,
            values,
            support_cells=content,
            target_cells=cells[:2],
        ).encoding

    original = predict(units[:2], units, answers, cells[rows])
    # Reorder source rows and their facts together; target addresses stay fixed.
    permutation = torch.tensor([4, 1, 0, 3, 2])
    reordered = predict(units[:2], units[permutation], answers[[2, 0, 1]], cells[[4, 0, 2]])
    torch.testing.assert_close(reordered, original)
    full = readout(
        unit_kernel_logits(units, units, bandwidth=1),
        rows,
        answers,
        support_cells=cells[rows],
        target_cells=cells,
    ).encoding
    torch.testing.assert_close(full[:2], original)


@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_cell_translation_and_expected_direct_gradient_paths(mode):
    gen = torch.Generator().manual_seed(32)
    tu, su, tc, sc = [
        torch.randn(*shape, generator=gen, dtype=torch.float64).requires_grad_()
        for shape in ((1, 2), (3, 2), (1, 2), (3, 2))
    ]
    answers = torch.tensor([[0.0], [3.0], [-2.0]], dtype=torch.float64)
    readout = RestorationReadout(mode, ridge=0.2)
    logits = unit_kernel_logits(tu, su, bandwidth=1.2)
    result = readout(logits, torch.arange(3), answers, support_cells=sc, target_cells=tc)
    shifted = readout(logits, torch.arange(3), answers, support_cells=sc + 10, target_cells=tc + 10)
    torch.testing.assert_close(shifted.encoding, result.encoding)
    grads = torch.autograd.grad(result.encoding.sum(), (tu, su, tc, sc), allow_unused=True)
    assert all(g is not None and g.norm() > 1e-6 for g in grads[:2])
    if mode == "ll":
        assert all(g is not None and g.norm() > 1e-6 for g in grads[2:])
    else:
        assert grads[2:] == (None, None)
        changed = readout(
            logits, torch.arange(3), answers, support_cells=sc * 100, target_cells=tc * -3
        )
        torch.testing.assert_close(changed.encoding, result.encoding, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_empty_and_single_support(mode):
    readout = RestorationReadout(mode)
    empty = readout(torch.zeros(2, 3), torch.empty(0, dtype=torch.long), torch.empty(0, 1))
    assert empty.status == "no-support" and empty.support_count == 0 and empty.encoding is None
    result = readout(
        torch.zeros(2, 3),
        torch.tensor([1]),
        torch.tensor([[7.0]]),
        support_cells=torch.ones(1, 3),
        target_cells=torch.zeros(2, 3),
    )
    torch.testing.assert_close(result.encoding, torch.full((2, 1), 7.0, dtype=torch.float64))


def test_collinear_content_and_constant_numeric_answers():
    numeric = NumericAnswers.from_visible(torch.tensor([0.0, 0.0, 0.0]), sigma_min=0.02)
    assert numeric.scale == 0.02
    result = RestorationReadout("ll")(
        torch.zeros(2, 3),
        torch.arange(3),
        numeric.encoded,
        support_cells=torch.ones(3, 128),
        target_cells=torch.ones(2, 128),
    )
    torch.testing.assert_close(numeric.decode(result.encoding), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("mode", ["nw", "ll"])
@pytest.mark.parametrize("width", [32, 128])
def test_categorical_readout_decodes_nearest_visible_code(mode, width):
    codec = categorical(width=width)
    result = RestorationReadout(mode, ridge=0.1)(
        torch.zeros(1, 3),
        torch.arange(3),
        codec.encoded,
        support_cells=torch.tensor([[0.0], [1.0], [2.0]]),
        target_cells=torch.tensor([[3.0]]),
    )
    distances = (result.encoding[:, None, :] - codec.codebook).square().sum(-1)
    torch.testing.assert_close(codec.decode(result.encoding), codec.classes[distances.argmin(-1)])


@pytest.mark.parametrize("width", [32, 128])
def test_same_ll_answer_with_different_support_weights_decodes_identically(width):
    codec = categorical((0, 1), width=width)
    # With covariance=ridge=3/16, coefficient_1=(target_cell + weight_1)/2.
    # Both rows restore code 1 exactly while their geometric weights differ.
    weights = torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64)
    result = RestorationReadout("ll", ridge=3 / 16)(
        weights.log(),
        torch.arange(2),
        codec.encoded,
        support_cells=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
        target_cells=torch.tensor([[1.75], [1.25]], dtype=torch.float64),
    )
    assert not torch.equal(result.log_weights[0], result.log_weights[1])
    torch.testing.assert_close(result.encoding, codec.codebook[1:2].expand(2, -1))
    torch.testing.assert_close(codec.decode(result.encoding), torch.tensor([1, 1]))


def test_rotation_lift_preserves_normalized_mse_and_gradients():
    check_rotation_lift_mse()


@pytest.mark.parametrize("width", [1, 32, 128])
@pytest.mark.parametrize("mode", ["nw", "ll"])
def test_mse_gradients_reach_geometry_and_ll_cells_but_not_fixed_answers(mode, width):
    codec = (
        NumericAnswers.from_visible(torch.tensor([1.0, -0.5, 2.0]), sigma_min=0.01)
        if width == 1
        else categorical(width=width)
    )
    truth = codec.encode_targets(torch.tensor([0.2]) if width == 1 else torch.tensor([1]))
    truth.requires_grad_()
    answers = codec.encoded.clone().requires_grad_()
    support = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[0.0, -0.2, 0.1]], dtype=torch.float64, requires_grad=True)
    result = RestorationReadout(mode, ridge=0.5)(
        logits, torch.arange(3), answers, support_cells=support, target_cells=target
    )
    if width != 1:
        decoded = codec.decode(result.encoding)
        assert decoded.dtype == torch.long and not decoded.requires_grad
    grads = torch.autograd.grad(
        encoding_mse(result.encoding, truth).sum(),
        (logits, target, support, answers, truth),
        allow_unused=True,
    )
    assert grads[0].isfinite().all() and grads[0].norm() > 1e-6
    if mode == "ll":
        assert all(g is not None and g.isfinite().all() and g.norm() > 1e-6 for g in grads[1:3])
    else:
        assert grads[1:3] == (None, None)
    assert grads[3:] == (None, None)


@pytest.mark.parametrize("kwargs", [{"mode": "mixed"}, {"mode": "ll", "ridge": 0}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        RestorationReadout(**kwargs)


def test_rejects_invalid_supports_missing_ll_cells_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="distinct"):
        RestorationReadout("nw")(torch.zeros(1, 2), torch.tensor([0, 0]), torch.ones(2, 1))
    with pytest.raises(ValueError, match="requires"):
        RestorationReadout("ll")(torch.zeros(1, 2), torch.arange(2), torch.ones(2, 1))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        unit_kernel_logits(torch.tensor([[float("nan")]]), torch.zeros(2, 1), bandwidth=1)
    with pytest.raises(ValueError, match="positive"):
        NumericAnswers.from_visible(torch.ones(2), sigma_min=0)


def test_codebook_must_be_same_visible_identity_space():
    codec = categorical()
    with pytest.raises(ValueError, match="visible class set"):
        CategoricalAnswers.from_visible(
            torch.tensor([0]), codec.classes, codec.codebook, domain_size=3
        )
    with pytest.raises(ValueError, match="different identity"):
        CategoricalAnswers.from_visible(
            codec.labels, codec.classes, codec.codebook[[0, 0]], domain_size=3
        )
    with pytest.raises(ValueError, match="binary"):
        CategoricalAnswers.from_visible(
            codec.labels, codec.classes, codec.codebook / 8**0.5, domain_size=3
        )


@pytest.mark.parametrize("width", [1, 128])
def test_coincident_supports_large_common_translation_preserves_nw_limit(width):
    codec = NumericAnswers.from_visible(torch.tensor([1.0, 2.0, 5.0]), sigma_min=0.1)
    logits = torch.tensor([[0.0, -1.0, -2.0]], dtype=torch.float64)
    rows = torch.arange(3)
    expected = RestorationReadout("nw")(logits, rows, codec.encoded).encoding
    for offset in (0.0, 1e6):
        result = RestorationReadout("ll")(
            logits,
            rows,
            codec.encoded,
            support_cells=torch.full((3, width), offset, dtype=torch.float64),
            target_cells=torch.full((1, width), offset - 1e6, dtype=torch.float64),
        )
        torch.testing.assert_close(result.encoding, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(result.coefficients.sum(-1), torch.ones(1, dtype=torch.float64))


@pytest.mark.parametrize("scale", [1e-200, 1e160])
def test_kernel_scales_before_distance_for_representable_logits_and_gradients(scale):
    target = torch.tensor([[0.0]], dtype=torch.float64, requires_grad=True)
    source = torch.tensor([[scale]], dtype=torch.float64, requires_grad=True)
    logits = unit_kernel_logits(target, source, bandwidth=scale)
    torch.testing.assert_close(logits, torch.tensor([[-0.5]], dtype=torch.float64))
    target_grad, source_grad = torch.autograd.grad(logits.sum(), (target, source))
    torch.testing.assert_close(target_grad * scale, torch.ones_like(target_grad))
    torch.testing.assert_close(source_grad * scale, -torch.ones_like(source_grad))


def test_kernel_preserves_nearby_differences_in_large_span_under_source_permutation():
    target = torch.tensor([[1e16 + 2]], dtype=torch.float64)
    sources = torch.tensor([[0.0], [1e16], [1e16 + 6]], dtype=torch.float64)
    order = torch.tensor([1, 0, 2])
    original = unit_kernel_logits(target, sources, bandwidth=3)
    reordered = unit_kernel_logits(target, sources[order], bandwidth=3)[:, order]
    torch.testing.assert_close(original, reordered, rtol=0, atol=0)
    torch.testing.assert_close(
        original[:, 1:], torch.tensor([[-2 / 9, -8 / 9]], dtype=torch.float64)
    )


def test_kernel_query_chunk_size_preserves_predictions_and_gradients():
    gen = torch.Generator().manual_seed(51)
    target = torch.randn(5, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    source = torch.randn(4, 3, generator=gen, dtype=torch.float64, requires_grad=True)
    full = unit_kernel_logits(target, source, bandwidth=1.3, query_chunk_size=10)
    chunks = unit_kernel_logits(target, source, bandwidth=1.3, query_chunk_size=2)
    torch.testing.assert_close(chunks, full, rtol=0, atol=0)
    full_grad = torch.autograd.grad(full.sum(), (target, source))
    chunk_grad = torch.autograd.grad(chunks.sum(), (target, source))
    for left, right in zip(full_grad, chunk_grad, strict=True):
        torch.testing.assert_close(left, right)


def test_cli_exposes_component_verification_without_changing_existing_commands():
    from tabu_lab.cli import build_parser

    parser = build_parser()
    command = parser.parse_args(["restoration", "verify"])
    assert command.handler.__name__ == "_run_restoration_verify"
    assert parser.parse_args(["tar", "verify", "--smoke"]).handler.__name__ == "_run_tar"
