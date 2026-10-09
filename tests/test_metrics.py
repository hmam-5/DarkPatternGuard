"""Tests for evaluation metrics."""

import numpy as np

from eval.metrics import (
    bootstrap_macro_f1,
    expected_calibration_error,
    macro_f1,
)


def test_macro_f1_is_computed_over_all_classes() -> None:
    true = np.array(["Urgency", "Scarcity", "Social Proof", "Normal"])
    predicted = np.array(["Urgency", "Scarcity", "Normal", "Normal"])
    assert macro_f1(true, predicted) == 2 / 3


def test_bootstrap_ci_is_reproducible_and_bounded() -> None:
    true = np.array(["Urgency", "Scarcity", "Normal", "Normal"])
    predicted = np.array(["Urgency", "Normal", "Normal", "Normal"])
    first = bootstrap_macro_f1(true, predicted, seed=1729, resamples=1_000)
    second = bootstrap_macro_f1(true, predicted, seed=1729, resamples=1_000)
    assert first == second
    assert 0.0 <= first["lower"] <= first["upper"] <= 1.0


def test_perfect_calibration_has_zero_ece() -> None:
    true = np.array(["Urgency", "Scarcity", "Normal"])
    probabilities = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    assert expected_calibration_error(true, probabilities) == 0.0
