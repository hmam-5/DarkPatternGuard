"""Load and clean project-owned raw e-commerce JSON data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

LABELS = ("Urgency", "Scarcity", "Social Proof", "Normal")
TEXT_FIELDS = ("text", "content", "description", "product_description", "sentence")
NEUTRAL_SAMPLES = (
    "add to cart",
    "100% cotton shirt",
    "free shipping on orders over $50",
    "blue ceramic coffee mug",
    "machine washable polyester jacket",
    "stainless steel water bottle",
    "women's cotton socks, size medium",
    "wireless keyboard with usb connection",
)


def load_config_seed(config_path: Path = Path("configs/base.yaml")) -> int:
    """Read the fixed integer seed from the small project YAML config."""
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("seed:"):
            return int(line.split(":", maxsplit=1)[1].strip())
    raise ValueError(f"Missing seed in {config_path}")


def _records_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "items"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return [payload]
    return []


def load_raw_records(raw_dir: Path) -> list[dict[str, Any]]:
    """Load JSON arrays, JSON objects, and JSONL files from a directory."""
    records: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json*")):
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.extend(_records_from_json(json.loads(line)))
        else:
            records.extend(
                _records_from_json(json.loads(path.read_text(encoding="utf-8")))
            )
    if not records:
        raise FileNotFoundError(f"No JSON records found in {raw_dir}")
    return records


def clean_text(value: Any) -> str:
    """Normalize one scraped text value into a deduplication-ready string."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: Any) -> str | None:
    """Map common label spellings to the canonical four-class label set."""
    normalized = clean_text(value).replace("_", " ")
    aliases = {
        "urgency": "Urgency",
        "scarcity": "Scarcity",
        "social proof": "Social Proof",
        "normal": "Normal",
    }
    return aliases.get(normalized)


def _text_from_record(record: dict[str, Any]) -> Any:
    for field in TEXT_FIELDS:
        if record.get(field) is not None:
            return record[field]
    return None


def _row(text: str, label: str, is_synthetic: bool = False) -> dict[str, Any]:
    return {
        "text": text,
        "label": label,
        "is_synthetic": is_synthetic,
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def inject_normal_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add neutral examples when Normal is missing or below 25% of other rows."""
    non_normal_count = sum(row["label"] != "Normal" for row in rows)
    normal_count = len(rows) - non_normal_count
    target = max(1, (non_normal_count + 3) // 4)
    if normal_count >= target:
        return rows
    existing = {row["text_hash"] for row in rows}
    additions: list[dict[str, Any]] = []
    for index in range(max(0, target - normal_count)):
        sample = NEUTRAL_SAMPLES[index % len(NEUTRAL_SAMPLES)]
        text = sample if index < len(NEUTRAL_SAMPLES) else f"{sample} ({index})"
        row = _row(text, "Normal", is_synthetic=True)
        if row["text_hash"] not in existing:
            additions.append(row)
            existing.add(row["text_hash"])
        if normal_count + len(additions) >= target:
            break
    return rows + additions


def clean_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Clean records, remove duplicates, and ensure a usable Normal class."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        text = clean_text(_text_from_record(record))
        label = normalize_label(record.get("pattern_type"))
        if not text or label is None:
            continue
        row = _row(text, label)
        if row["text_hash"] not in seen:
            rows.append(row)
            seen.add(row["text_hash"])
    rows = inject_normal_samples(rows)
    if not rows:
        raise ValueError("No valid labeled, non-empty records remained after cleaning")
    return pd.DataFrame(rows, columns=("text", "label", "is_synthetic", "text_hash"))


def clean_dataset(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/cleaned/cleaned.csv"),
) -> pd.DataFrame:
    """Load, clean, balance, and save the cleaned dataset."""
    frame = clean_records(load_raw_records(raw_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def main() -> None:
    """Run the command-line cleaning pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/cleaned/cleaned.csv"))
    args = parser.parse_args()
    clean_dataset(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
