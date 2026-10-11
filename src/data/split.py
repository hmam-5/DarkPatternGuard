"""Create reproducible stratified train, validation, and test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from data.clean import load_config_seed


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count) for label, count in frame["label"].value_counts().items()
    }


def _stratified_split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, ...]:
    """Split 80/10/10, keeping source-page groups in a single partition."""
    if "source_id" in frame and frame["source_id"].notna().any():
        groups = frame["source_id"].astype("string")
        missing = groups.isna() | groups.str.strip().eq("")
        groups = groups.astype(object)
        groups.loc[missing] = [f"row-{index}" for index in frame.index[missing]]
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_indices, remainder_indices = next(
            outer.split(frame["text"], frame["label"], groups)
        )
        remainder = frame.iloc[remainder_indices].reset_index(drop=True)
        remainder_groups = groups.iloc[remainder_indices].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        val_indices, test_indices = next(
            inner.split(remainder["text"], remainder["label"], remainder_groups)
        )
        train = frame.iloc[train_indices]
        val = remainder.iloc[val_indices]
        test = remainder.iloc[test_indices]
        expected_labels = set(frame["label"])
        if any(set(part["label"]) != expected_labels for part in (train, val, test)):
            raise ValueError(
                "A source-group split omitted a class; review the group labels"
            )
        group_sets = [
            set(groups.iloc[train_indices].astype(str)),
            set(remainder_groups.iloc[val_indices].astype(str)),
            set(remainder_groups.iloc[test_indices].astype(str)),
        ]
        if (
            group_sets[0] & group_sets[1]
            or group_sets[0] & group_sets[2]
            or group_sets[1] & group_sets[2]
        ):
            raise RuntimeError("Source groups overlap across data partitions")
        return tuple(part.reset_index(drop=True) for part in (train, val, test))

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
        "grouped_by_source_id": bool(
            "source_id" in frame and frame["source_id"].notna().any()
        ),
        "unique_source_ids": (
            int(frame["source_id"].nunique(dropna=True))
            if "source_id" in frame
            else None
        ),
        "synthetic_rows": (
            int(frame["is_synthetic"].astype(bool).sum())
            if "is_synthetic" in frame
            else 0
        ),
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
        "The raw source is the `dataset/dataset.tsv` release from "
        "[yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern), "
        "pinned to commit `85ea9df31928d5ad6ea0b106316b441c201adbbe`. Its README "
        "describes the dark-pattern examples as originating from Mathur et al. "
        "(2019) and its Normal examples as non-dark e-commerce segments collected "
        "from the same sites. The repository includes an Apache-2.0 license.",
        "",
        f"- Total rows: {len(frame):,}",
        f"- Seed: {seed}",
        (
            "- Unique source IDs: "
            f"{frame['source_id'].nunique(dropna=True) if 'source_id' in frame else 0}"
        ),
        f"- Synthetic rows: {manifest['synthetic_rows']}",
        "- Split by source page to reduce page-level leakage.",
        "- Labels: " + ", ".join(labels),
        "",
        "## Grouped split sizes",
        "",
        "| Split | Rows |",
        "| --- | ---: |",
    ]
    card.extend(
        f"| {name.title()} | {manifest['splits'][name]['rows']:,} |"
        for name in ("train", "val", "test")
    )
    card.extend(
        (
            "",
            "## Class distribution",
            "",
            "| Label | Count | Weight |",
            "| --- | ---: | ---: |",
        )
    )
    card.extend(
        f"| {label} | {int((frame['label'] == label).sum())} | {weights[label]:.6f} |"
        for label in labels
    )
    card.extend(
        (
            "",
            "Only Urgency, Scarcity, Social Proof, and `Not Dark Pattern` records "
            "are retained; `Not Dark Pattern` maps to Normal. Other source "
            "categories are excluded. Original text and source-page/category "
            "metadata remain available in the raw JSON; the cleaner records the "
            "source ID and category in the cleaned CSV. No synthetic examples "
            "were added when genuine Normal examples met the balancing threshold.",
            "",
            "## Citation",
            "",
            "Yada et al., *Dark patterns in e-commerce: a dataset and its baseline "
            "evaluations* (2022). The dataset is derived in part from Mathur et al., "
            "*Dark Patterns at Scale* (2019).",
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
