from __future__ import annotations

import math

import numpy as np

from tabu_lab.experiments.tabubase_real_metrics import (
    classification_metrics,
    regression_metrics,
)


def test_classification_metrics_share_one_probability_matrix() -> None:
    truth = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.8, 0.2], [0.25, 0.75], [0.1, 0.9]],
        dtype=np.float64,
    )
    metrics = classification_metrics(truth, probabilities, classes=2)

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == 1.0
    expected_log_loss = -float(np.log([0.9, 0.8, 0.75, 0.9]).mean())
    assert math.isclose(metrics["log_loss"], expected_log_loss)
    assert math.isclose(metrics["normalized_nll"], expected_log_loss / math.log(2))


def test_regression_metrics_report_raw_scaled_and_r2() -> None:
    truth = np.asarray([1.0, 2.0, 3.0])
    predicted = np.asarray([1.0, 2.0, 4.0])
    metrics = regression_metrics(truth, predicted, target_scale=2.0)

    assert math.isclose(metrics["rmse"], math.sqrt(1.0 / 3.0))
    assert math.isclose(metrics["mae"], 1.0 / 3.0)
    assert math.isclose(metrics["scaled_rmse"], math.sqrt(1.0 / 3.0) / 2.0)
    assert math.isclose(metrics["scaled_mae"], 1.0 / 6.0)
    assert math.isclose(metrics["r2"], 0.5)
