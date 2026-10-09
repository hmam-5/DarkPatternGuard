"""Metrics, confidence intervals, calibration, and prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

LABELS = ("Urgency", "Scarcity", "Social Proof", "Normal")
PROBABILITY_COLUMNS = (
    "prob_Urgency",
    "prob_Scarcity",
    "prob_Social_Proof",
    "prob_Normal",
)


def macro_f1(y_true: list[str] | np.ndarray, y_pred: list[str] | np.ndarray) -> float:
    """Return macro-F1 over the fixed four-class label set."""
    return float(
        f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    )


def expected_calibration_error(
    y_true: list[str] | np.ndarray,
    probabilities: np.ndarray,
    labels: tuple[str, ...] = LABELS,
    bins: int = 10,
) -> float:
    """Compute top-label expected calibration error with equal-width bins."""
    true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability columns must match the label count")
    confidence = probabilities.max(axis=1)
    predicted = np.asarray(labels)[probabilities.argmax(axis=1)]
    correct = predicted == true
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (confidence > lower) & (confidence <= upper)
        if lower == 0:
            selected = (confidence >= lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(
                correct[selected].mean() - confidence[selected].mean()
            )
    return float(error)


def compute_metrics(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    probabilities: np.ndarray,
    seed: int = 1729,
    bootstrap_resamples: int = 1_000,
) -> dict[str, Any]:
    """Compute classification, calibration, and bootstrap metrics."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    report = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    interval = bootstrap_macro_f1(y_true, y_pred, seed, bootstrap_resamples)
    return {
        "macro_f1": macro_f1(y_true, y_pred),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
        "per_class": report,
        "ece": expected_calibration_error(y_true, probabilities),
        "macro_f1_bootstrap_95_ci": interval,
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }


def bootstrap_macro_f1(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    seed: int,
    resamples: int = 1_000,
) -> dict[str, float]:
    """Return percentile 95% CI for macro-F1 using deterministic resampling."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    true = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(true), len(true))
        scores[index] = macro_f1(true[sample], predicted[sample])
    return {
        "lower": float(np.percentile(scores, 2.5)),
        "upper": float(np.percentile(scores, 97.5)),
    }


def paired_bootstrap_test(
    y_true: list[str] | np.ndarray,
    predictions_a: list[str] | np.ndarray,
    predictions_b: list[str] | np.ndarray,
    seed: int = 1729,
    resamples: int = 1_000,
) -> dict[str, float]:
    """Test paired macro-F1 differences using shared bootstrap indices."""
    true = np.asarray(y_true)
    first = np.asarray(predictions_a)
    second = np.asarray(predictions_b)
    if not (len(true) == len(first) == len(second)):
        raise ValueError("Paired inputs must have equal lengths")
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        sample = rng.integers(0, len(true), len(true))
        differences[index] = macro_f1(true[sample], first[sample]) - macro_f1(
            true[sample], second[sample]
        )
    return {
        "observed_difference": macro_f1(true, first) - macro_f1(true, second),
        "lower": float(np.percentile(differences, 2.5)),
        "upper": float(np.percentile(differences, 97.5)),
        "p_value": float(2 * min((differences <= 0).mean(), (differences >= 0).mean())),
    }


def save_confusion_matrix(
    y_true: list[str] | np.ndarray,
    y_pred: list[str] | np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a labeled confusion matrix PNG."""
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_prediction_artifacts(
    output_dir: Path,
    identifiers: list[str] | np.ndarray,
    y_true: list[str] | np.ndarray,
    probabilities: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Write standardized predictions, metrics, and confusion matrix."""
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    frame = {
        "id": identifiers,
        "true_label": y_true,
        **{
            column: probabilities[:, index]
            for index, column in enumerate(PROBABILITY_COLUMNS)
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(frame).to_csv(output_dir / "preds.csv", index=False)
    metrics = compute_metrics(y_true, predicted, probabilities, seed=seed)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    save_confusion_matrix(
        y_true, predicted, output_dir / "confusion_matrix.png", output_dir.parent.name
    )
    return metrics
