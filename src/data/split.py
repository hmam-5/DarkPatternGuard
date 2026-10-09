"""Create reproducible stratified train, validation, and test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from data.clean import load_config_seed


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count) for label, count in frame["label"].value_counts().items()
    }


def _stratified_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, ...]:
    """Split 80/10/10, with a deterministic fallback for tiny classes."""
    try:
        train, remainder = train_test_split(
            frame, test_size=0.2, random_state=seed, stratify=frame["label"]
        )
        val, test = train_test_split(
            remainder, test_size=0.5, random_state=seed, stratify=remainder["label"]
        )
    except ValueError:
        shuffled = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
        train_end = max(1, int(len(shuffled) * 0.8))
        val_end = train_end + max(1, int(len(shuffled) * 0.1))
        train, val, test = (
            shuffled.iloc[:train_end],
            shuffled.iloc[train_end:val_end],
            shuffled.iloc[val_end:],
        )
    return tuple(part.reset_index(drop=True) for part in (train, val, test))


def write_split_artifacts(
    cleaned_path: Path = Path("data/cleaned/cleaned.csv"),
    output_dir: Path = Path("data/cleaned"),
    seed: int | None = None,
) -> dict[str, Any]:
    """Write split CSVs, manifest, and balanced class weights."""
    frame = pd.read_csv(cleaned_path)
    if frame.empty or frame["label"].isna().any() or frame["text"].isna().any():
        raise ValueError("Cleaned data must contain non-empty text and labels")
    seed = load_config_seed() if seed is None else seed
    train, val, test = _stratified_split(frame, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_frame in (("train", train), ("val", val), ("test", test)):
        split_frame.to_csv(output_dir / f"{name}.csv", index=False)
    labels = sorted(frame["label"].unique())
    weights = {
        label: len(frame) / (len(labels) * int((frame["label"] == label).sum()))
        for label in labels
    }
    (output_dir / "class_weights.json").write_text(
        json.dumps(weights, indent=2, sort_keys=True), encoding="utf-8"
    )
    manifest = {
        "seed": seed,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "total_rows": len(frame),
        "splits": {
            "train": {"rows": len(train), "class_counts": _counts(train)},
            "val": {"rows": len(val), "class_counts": _counts(val)},
            "test": {"rows": len(test), "class_counts": _counts(test)},
        },
        "class_counts": _counts(frame),
        "class_weights_path": "class_weights.json",
    }
    (output_dir / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    card = [
        "# DarkPatternGuard data card",
        "",
        "This dataset is cleaned from project-owned web-scraped e-commerce JSON.",
        "No Kaggle or Hugging Face dataset is used.",
        "",
        f"- Total rows: {len(frame)}",
        f"- Seed: {seed}",
        "- Labels: " + ", ".join(labels),
        "",
        "## Class distribution",
        "",
        "| Label | Count | Weight |",
        "| --- | ---: | ---: |",
    ]
    card.extend(
        f"| {label} | {int((frame['label'] == label).sum())} | {weights[label]:.6f} |"
        for label in labels
    )
    card.extend(
        (
            "",
            "Synthetic neutral examples are marked by `is_synthetic=true` in the "
            "cleaned CSV when Normal was underrepresented.",
        )
    )
    (output_dir.parent / "DATA_CARD.md").write_text(
        "\n".join(card) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """Run the command-line splitting pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/cleaned/cleaned.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cleaned"))
    args = parser.parse_args()
    write_split_artifacts(args.input, args.output_dir)


if __name__ == "__main__":
    main()
