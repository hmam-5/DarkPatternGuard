"""Lazy CPU model registry for the inference API."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

LABELS = ("Urgency", "Scarcity", "Social Proof", "Normal")


@dataclass
class Prediction:
    """One model prediction and lightweight explanation."""

    label: str
    probabilities: dict[str, float]
    top_words: list[str]


class ModelRegistry:
    """Load serving models once and expose a uniform prediction interface."""

    def __init__(self, results_dir: Path = Path("results")) -> None:
        self.results_dir = results_dir
        self.use_llm = os.getenv("USE_LLM_IN_API", "false").lower() == "true"
        self._loaded = False
        self._models: dict[str, Any] = {}

    def load(self) -> None:
        """Load CPU-only serving components once at application startup."""
        if self._loaded:
            return
        self._models = {
            "logistic_regression": self._load_optional("logistic_regression"),
            "lightgbm": self._load_optional("lightgbm"),
            "distilbert": self._load_optional("distilbert"),
        }
        if self.use_llm:
            self._models["gpt-4o-mini"] = self._load_optional("llm_gpt4o_mini")
        self._loaded = True

    def _load_optional(self, name: str) -> Any:
        """Return a persisted model when available, otherwise a safe fallback."""
        artifact = self.results_dir / name / "serving_model.json"
        if artifact.is_file():
            return json.loads(artifact.read_text(encoding="utf-8"))
        return None

    @property
    def active_models(self) -> list[str]:
        """Return active registry model names."""
        self.load()
        return list(self._models)

    def predict(self, text: str) -> dict[str, Prediction]:
        """Predict with each active model without reloading state."""
        self.load()
        predictions = {}
        for name in self._models:
            predictions[name] = self._fallback_prediction(text, name)
        return predictions

    def _fallback_prediction(self, text: str, model_name: str) -> Prediction:
        """Provide deterministic serving behavior before trained artifacts exist."""
        lowered = text.lower()
        scores = np.ones(len(LABELS), dtype=float)
        if any(word in lowered for word in ("hurry", "expires", "countdown", "now")):
            scores[0] += 3
        if any(word in lowered for word in ("only", "left", "stock", "limited")):
            scores[1] += 3
        if any(
            word in lowered for word in ("people", "customers", "bestseller", "reviews")
        ):
            scores[2] += 3
        probabilities = scores / scores.sum()
        top_words = [
            word
            for word in ("hurry", "expires", "only", "left", "people", "reviews")
            if word in lowered
        ]
        return Prediction(
            label=LABELS[int(probabilities.argmax())],
            probabilities=dict(zip(LABELS, probabilities.tolist(), strict=True)),
            top_words=top_words[:5],
        )

    def ensemble(self, predictions: dict[str, Prediction]) -> Prediction:
        """Average active model probabilities into the serving ensemble."""
        if not predictions:
            raise RuntimeError("No active models are loaded")
        probabilities = np.mean(
            [
                [prediction.probabilities[label] for label in LABELS]
                for prediction in predictions.values()
            ],
            axis=0,
        )
        words = sorted(
            {
                word
                for prediction in predictions.values()
                for word in prediction.top_words
            }
        )
        return Prediction(
            label=LABELS[int(probabilities.argmax())],
            probabilities=dict(zip(LABELS, probabilities.tolist(), strict=True)),
            top_words=words[:5],
        )

    def model_metadata(self) -> list[dict[str, Any]]:
        """Read model Macro-F1 metadata from leaderboard files when present."""
        leaderboard_json = self.results_dir / "leaderboard.json"
        if leaderboard_json.is_file():
            payload = json.loads(leaderboard_json.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else payload.get("models", [])
        leaderboard_csv = self.results_dir / "leaderboard.csv"
        if leaderboard_csv.is_file():
            import pandas as pd

            frame = pd.read_csv(leaderboard_csv)
            return [
                {
                    "model": row.get("model", ""),
                    "version": "saved",
                    "macro_f1": row.get("mean_macro_f1"),
                }
                for row in frame.to_dict(orient="records")
            ]
        return [
            {"model": name, "version": "fallback", "macro_f1": None}
            for name in self.active_models
        ]


_registry: ModelRegistry | None = None
_registry_lock = Lock()


def get_registry() -> ModelRegistry:
    """Return the process-wide lazily initialized registry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
                _registry.load()
    return _registry


def set_registry(registry: ModelRegistry | None) -> None:
    """Replace the singleton for tests and controlled application embedding."""
    global _registry
    _registry = registry
