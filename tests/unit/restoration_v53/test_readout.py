import pytest
import torch

from tabu_lab.models.restoration_v53.readout import (
    evaluate_column,
    normalized_weights,
    shared_slope,
)


@pytest.fixture(autouse=True)
def fixed_rng():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9)
        yield
    torch.set_num_threads(threads)


def example():
    units = torch.randn(5, 2, dtype=torch.float64)
    cells = torch.randn(5, 3, dtype=torch.float64)
    rows = torch.tensor([0, 1, 3])
    answers = torch.randn(3, 4, dtype=torch.float64)
    return units, cells, rows, answers


def fit(units, cells, rows, answers, chunk=2):
    return shared_slope(units, rows, cells[rows], answers, ridge=0.17,
                        bandwidth=1.3, center_chunk_size=chunk)


def joint_least_squares(units, cells, rows, answers):
    """Independent oracle: explicit intercept per center and a single slope.

    Solve the ORIGINAL weighted least-squares design with ridge rows appended,
    without centering or forming the implementation's sufficient statistics.
    """
    n, d = cells.shape
    weights = normalized_weights(units, units[rows], 1.3)
    design, response = [], []
    for r in range(n):
        for s, support in enumerate(rows):
            intercepts = torch.eye(n, dtype=torch.float64)[r]
            row = torch.cat((intercepts, cells[support] - cells[r]))
            root_weight = (weights[r, s] / n).sqrt()
            design.append(root_weight * row)
            response.append(root_weight * answers[s])
    regularizer = torch.cat((torch.zeros(d, n), torch.eye(d)), -1).double() * 0.17**0.5
    design = torch.cat((torch.stack(design), regularizer))
    response = torch.cat((torch.stack(response), answers.new_zeros(d, answers.shape[1])))
    solved = torch.linalg.lstsq(design, response).solution
    return solved[n:].T, solved[:n]


def test_shared_solve_matches_original_joint_objective():
    units, cells, rows, answers = example()
    slope = fit(units, cells, rows, answers)
    expected_slope, expected_predictions = joint_least_squares(units, cells, rows, answers)
    result = evaluate_column(units, cells, rows, answers, torch.arange(5), slope,
                             bandwidth=1.3, chunk_size=2)
    torch.testing.assert_close(slope, expected_slope, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(result.encoding, expected_predictions, atol=1e-12, rtol=1e-11)


def test_chunk_support_permutation_and_coordinate_translation():
    units, cells, rows, answers = example()
    expected = fit(units, cells, rows, answers)
    for chunk in (1, 3, 20):
        torch.testing.assert_close(fit(units, cells, rows, answers, chunk), expected)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(fit(units, cells, rows[perm], answers[perm]), expected)
    torch.testing.assert_close(fit(units, cells + 100, rows, answers - 50), expected)


def test_constant_and_single_support_have_zero_slope():
    units, cells, rows, answers = example()
    cases = ((rows, answers[:1].expand_as(answers)), (rows[:1], answers[:1]))
    for selected_rows, selected_answers in cases:
        slope = fit(units, cells, selected_rows, selected_answers)
        assert torch.count_nonzero(slope) == 0
        result = evaluate_column(
            units, cells, selected_rows, selected_answers, torch.arange(5), slope,
            bandwidth=1.3, chunk_size=2,
        )
        torch.testing.assert_close(result.encoding, selected_answers[:1].expand(5, -1))


@pytest.mark.parametrize("chunk", [1, 2, 32])
def test_concentrated_kernel_preserves_small_local_variance_at_large_span(chunk):
    # The former second-moment subtraction rounded BOTH moments to zero here,
    # returning a finite B=0 and Query prediction near zero instead of ~0.4813.
    # A two-point population has analytic variance p*(1-p)*span**2, so this
    # oracle does not repeat the implementation's centered matrix arithmetic.
    units = torch.tensor([[0.0], [9.0], [0.0]], dtype=torch.float64)
    span, ridge = 1e8, 1e-3
    cells = torch.tensor([[0.0], [span], [span / 2]], dtype=torch.float64)
    rows = torch.tensor([0, 1])
    answers = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    p = torch.sigmoid(torch.tensor(-9.0**2 / 2, dtype=torch.float64))
    variance = p * (1 - p)
    expected_slope = span * variance / (span**2 * variance + ridge)
    expected_prediction = p + expected_slope * span * (0.5 - p)
    slope = shared_slope(
        units, rows, cells[rows], answers, ridge=ridge, bandwidth=1.0,
        center_chunk_size=chunk,
    )
    result = evaluate_column(
        units, cells, rows, answers, torch.tensor([2]), slope,
        bandwidth=1.0, chunk_size=chunk,
    )
    torch.testing.assert_close(slope.squeeze(), expected_slope, atol=0, rtol=1e-12)
    torch.testing.assert_close(result.encoding.squeeze(), expected_prediction, atol=0, rtol=1e-12)


def test_gradients_include_unrequested_nonsupport_centers():
    units, cells, rows, answers = example()
    units.requires_grad_()
    cells.requires_grad_()
    answers.requires_grad_()
    slope = fit(units, cells, rows, answers)
    result = evaluate_column(units, cells, rows, answers, torch.tensor([0]), slope,
                             bandwidth=1.3, chunk_size=2)
    result.encoding.square().sum().backward()
    assert units.grad[4].norm() > 1e-8  # Neither target nor support; enters the shared fit.
    assert answers.grad is None
    assert torch.isfinite(units.grad).all() and torch.isfinite(cells.grad).all()


def test_shared_slope_autograd_matches_finite_differences():
    units = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    cells = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    rows = torch.tensor([0, 1])
    answers = torch.tensor([[1.0], [3.0]], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda u, c: fit(u, c, rows, answers), (units, cells))


def test_overflow_is_not_an_empty_source():
    units, cells, rows, answers = example()
    units[0, 0] = 1e308
    with pytest.raises(FloatingPointError, match="numerical-failure"):
        fit(units, cells, rows, answers)
