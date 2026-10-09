"""Validation-fitted probability ensembles over persisted model predictions."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from eval.metrics import LABELS, PROBABILITY_COLUMNS, save_prediction_artifacts
from models.neural_common import configured_seeds


def _probabilities(frame: pd.DataFrame) -> np.ndarray:
    return frame.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    scaled = np.exp(logits / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, labels: Iterable[str]) -> float:
    """Fit one temperature using validation probabilities and labels only."""
    target = np.asarray([LABELS.index(label) for label in labels])

    def objective(log_temperature: float) -> float:
        scaled = _temperature_scale(probabilities, np.exp(log_temperature))
        return float(
            -np.log(np.clip(scaled[np.arange(len(target)), target], 1e-8, 1)).mean()
        )

    result = minimize(objective, np.array([0.0]), method="Nelder-Mead")
    return float(np.exp(result.x[0]))


def calibrate_probabilities(
    probabilities: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_labels: Iterable[str],
) -> np.ndarray:
    """Calibrate test probabilities with a temperature fitted on validation."""
    temperature = fit_temperature(validation_probabilities, validation_labels)
    return _temperature_scale(probabilities, temperature)


def soft_voting(probabilities: list[np.ndarray]) -> np.ndarray:
    """Average calibrated model probabilities."""
    return np.mean(probabilities, axis=0)


def optimize_weights(
    validation_probabilities: list[np.ndarray],
    validation_labels: Iterable[str],
) -> np.ndarray:
    """Optimize non-negative weighted voting weights on validation only."""
    target = np.asarray([LABELS.index(label) for label in validation_labels])

    def objective(raw: np.ndarray) -> float:
        weights = np.exp(raw - raw.max())
        weights /= weights.sum()
        combined = sum(
            weight * probs
            for weight, probs in zip(weights, validation_probabilities, strict=True)
        )
        return float(
            -np.log(np.clip(combined[np.arange(len(target)), target], 1e-8, 1)).mean()
        )

    result = minimize(
        objective,
        np.zeros(len(validation_probabilities)),
        method="SLSQP",
        bounds=[(-5.0, 5.0)] * len(validation_probabilities),
    )
    raw = result.x
    weights = np.exp(raw - raw.max())
    return weights / weights.sum()


def stacking(
    validation_probabilities: list[np.ndarray],
    validation_labels: Iterable[str],
    test_probabilities: list[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Train a logistic-regression meta-learner on validation probabilities."""
    x_val = np.hstack(validation_probabilities)
    x_test = np.hstack(test_probabilities)
    target = np.asarray([LABELS.index(label) for label in validation_labels])
    meta = LogisticRegression(max_iter=500, random_state=seed, multi_class="auto")
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cross_val_predict(meta, x_val, target, cv=folds, method="predict_proba")
    meta.fit(x_val, target)
    output = meta.predict_proba(x_test)
    aligned = np.zeros((len(output), len(LABELS)))
    for index, label in enumerate(meta.classes_):
        aligned[:, label] = output[:, index]
    return aligned


def _model_dirs(results_dir: Path, seed: int) -> list[Path]:
    excluded = {
        "ensemble_soft_voting",
        "ensemble_weighted_average",
        "ensemble_stacking",
    }
    return sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir()
        and path.name not in excluded
        and (path / f"seed{seed}" / "preds.csv").is_file()
    )


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"id", "true_label", *PROBABILITY_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns: {sorted(missing)}")
    return frame


def run_ensembles(
    results_dir: Path = Path("results"),
    methods: tuple[str, ...] = (
        "soft_voting",
        "weighted_average",
        "stacking",
        "classical_only",
        "neural_only",
        "without_llm",
        "leave_one_model_out",
    ),
) -> None:
    """Generate ensemble artifacts from saved predictions only."""
    for seed in configured_seeds():
        model_dirs = _model_dirs(results_dir, seed)
        if not model_dirs:
            raise FileNotFoundError(f"No saved model predictions found for seed {seed}")
        test_frames = [
            _load_predictions(path / f"seed{seed}" / "preds.csv") for path in model_dirs
        ]
        validation_frames = []
        for path in model_dirs:
            validation_path = path / f"seed{seed}" / "val_preds.csv"
            if not validation_path.is_file():
                raise FileNotFoundError(
                    "Validation predictions required for calibration/stacking: "
                    f"{validation_path}"
                )
            validation_frames.append(_load_predictions(validation_path))
        validation_labels = validation_frames[0]["true_label"].tolist()
        test_probabilities = [
            calibrate_probabilities(
                _probabilities(test),
                _probabilities(validation),
                validation["true_label"],
            )
            for test, validation in zip(test_frames, validation_frames, strict=True)
        ]
        validation_probabilities = [
            _probabilities(validation) for validation in validation_frames
        ]
        for method in methods:
            if method == "leave_one_model_out":
                for excluded, excluded_path in enumerate(model_dirs):
                    selected = [
                        index for index in range(len(model_dirs)) if index != excluded
                    ]
                    if not selected:
                        continue
                    save_prediction_artifacts(
                        results_dir
                        / f"ensemble_leave_one_out_{excluded_path.name}"
                        / f"seed{seed}",
                        test_frames[0]["id"].to_numpy(),
                        test_frames[0]["true_label"].to_numpy(),
                        soft_voting([test_probabilities[index] for index in selected]),
                        seed,
                    )
                continue
            selected = list(range(len(model_dirs)))
            if method == "classical_only":
                selected = [
                    index
                    for index, path in enumerate(model_dirs)
                    if any(
                        token in path.name.lower()
                        for token in ("naive", "logistic", "svm", "lightgbm")
                    )
                ]
            elif method == "neural_only":
                selected = [
                    index
                    for index, path in enumerate(model_dirs)
                    if any(
                        token in path.name.lower()
                        for token in ("cnn", "lstm", "transformer", "finetune")
                    )
                ]
            elif method == "without_llm":
                selected = [
                    index
                    for index, path in enumerate(model_dirs)
                    if "llm" not in path.name.lower()
                ]
            if method in {"classical_only", "neural_only", "without_llm"}:
                if not selected:
                    continue
                method_probabilities = [test_probabilities[index] for index in selected]
                probabilities = soft_voting(method_probabilities)
                save_prediction_artifacts(
                    results_dir / f"ensemble_{method}" / f"seed{seed}",
                    test_frames[0]["id"].to_numpy(),
                    test_frames[0]["true_label"].to_numpy(),
                    probabilities,
                    seed,
                )
                continue
            if method == "soft_voting":
                probabilities = soft_voting(test_probabilities)
            elif method == "weighted_average":
                weights = optimize_weights(validation_probabilities, validation_labels)
                probabilities = sum(
                    weight * probs
                    for weight, probs in zip(weights, test_probabilities, strict=True)
                )
            elif method == "stacking":
                probabilities = stacking(
                    validation_probabilities,
                    validation_labels,
                    test_probabilities,
                    seed,
                )
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            save_prediction_artifacts(
                results_dir / f"ensemble_{method}" / f"seed{seed}",
                test_frames[0]["id"].to_numpy(),
                test_frames[0]["true_label"].to_numpy(),
                probabilities,
                seed,
            )


def main() -> None:
    """Run requested persisted-prediction ensembles."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run_ensembles(args.results_dir)


if __name__ == "__main__":
    main()
