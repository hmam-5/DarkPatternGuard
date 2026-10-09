"""Prefetch Hugging Face model snapshots into an ignored deployment cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_MODELS = (
    "bert-base-uncased",
    "microsoft/deberta-v3-base",
    "distilbert-base-uncased",
    "roberta-base",
)


def fetch_models(models: list[str], cache_dir: Path) -> None:
    """Download model snapshots without placing weights in the repository."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    for model_id in models:
        print(f"Fetching {model_id} into {cache_dir}")
        snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))


def main() -> None:
    """Parse model IDs and prefetch their snapshots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Hugging Face model ID; may be repeated.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models_cache"),
        help="Ignored local cache directory.",
    )
    args = parser.parse_args()
    fetch_models(args.models or list(DEFAULT_MODELS), args.cache_dir)


if __name__ == "__main__":
    main()
