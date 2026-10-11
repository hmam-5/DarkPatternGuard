from __future__ import annotations

import numpy as np
import pandas as pd

import eval.benchmark as benchmark
from eval.metrics import LABELS, PROBABILITY_COLUMNS


def _prediction_frame(
    identifiers: list[str], labels: list[str], predictions: list[str]
) -> pd.DataFrame:
    probabilities = np.full((len(labels), len(LABELS)), 0.01)
    for row, label in enumerate(predictions):
        probabilities[row, LABELS.index(label)] = 0.97
    return pd.DataFrame(
        {
            "id": identifiers,
            "true_label": labels,
            **{
                column: probabilities[:, index]
                for index, column in enumerate(PROBABILITY_COLUMNS)
            },
        }
    )


def test_error_report_aligns_recovered_examples_by_id(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "configured_seeds", lambda: [1729, 2718])
    test_dir = tmp_path / "data" / "cleaned"
    test_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "text": [
                "Only three seats remain | book now",
                "Just one item left",
                "Customers love this product",
                "Standard cotton shirt",
            ]
        }
    ).to_csv(test_dir / "test.csv", index=False)

    results_dir = tmp_path / "results"
    true_labels = list(LABELS)
    identifiers = [str(index) for index in range(len(LABELS))]
    for seed, bert_predictions in (
        (1729, ["Normal", *true_labels[1:]]),
        (2718, ["Normal", "Urgency", *true_labels[2:]]),
    ):
        bert_path = results_dir / "finetune_bert" / f"seed{seed}"
        bert_path.mkdir(parents=True)
        _prediction_frame(identifiers, true_labels, bert_predictions).to_csv(
            bert_path / "preds.csv", index=False
        )

        reverse_order = list(reversed(range(len(LABELS))))
        deberta_path = results_dir / "finetune_deberta" / f"seed{seed}"
        deberta_path.mkdir(parents=True)
        _prediction_frame(
            [identifiers[index] for index in reverse_order],
            [true_labels[index] for index in reverse_order],
            [true_labels[index] for index in reverse_order],
        ).to_csv(deberta_path / "preds.csv", index=False)

    benchmark.write_error_analysis(results_dir)

    report = (results_dir / "errors.md").read_text(encoding="utf-8")
    assert (
        "| DeBERTa | 0 | Only three seats remain \\| book now | "
        "Urgency | Normal | Urgency | 1729, 2718 |"
    ) in report
    assert (
        "| DeBERTa | 1 | Just one item left | Scarcity | Urgency | " "Scarcity | 2718 |"
    ) in report
