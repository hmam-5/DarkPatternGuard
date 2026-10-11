from __future__ import annotations

import numpy as np
import pandas as pd

import ensemble.ensemble as ensemble
from eval.metrics import LABELS, PROBABILITY_COLUMNS


def _write_predictions(path, identifiers, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = np.full((len(labels), len(LABELS)), 0.01)
    for row, label in enumerate(labels):
        probabilities[row, LABELS.index(label)] = 0.97
    pd.DataFrame(
        {
            "id": identifiers,
            "true_label": labels,
            **{
                column: probabilities[:, index]
                for index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    ).to_csv(path, index=False)


def test_ensembles_save_validation_and_test_predictions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ensemble, "configured_seeds", lambda: [1729])
    results_dir = tmp_path / "results"
    labels = list(LABELS) * 3
    identifiers = [str(index) for index in range(len(labels))]

    for model in ("naive_bayes", "cnn"):
        seed_dir = results_dir / model / "seed1729"
        _write_predictions(seed_dir / "preds.csv", identifiers, labels)
        _write_predictions(seed_dir / "val_preds.csv", identifiers, labels)

    ensemble.run_ensembles(results_dir)

    assert [path.name for path in ensemble._model_dirs(results_dir, 1729)] == [
        "cnn",
        "naive_bayes",
    ]
    output_dirs = list(results_dir.glob("ensemble_*/seed1729"))
    assert output_dirs
    for output_dir in output_dirs:
        validation = pd.read_csv(output_dir / "val_preds.csv")
        test = pd.read_csv(output_dir / "preds.csv")
        assert validation["id"].astype(str).tolist() == identifiers
        assert test["id"].astype(str).tolist() == identifiers
