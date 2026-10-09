"""FastAPI endpoint tests using a lightweight mocked registry."""

from fastapi.testclient import TestClient

from app.backend.main import app
from app.backend.models_registry import ModelRegistry, Prediction, set_registry


class MockRegistry(ModelRegistry):
    """Instant deterministic registry for API tests."""

    def __init__(self) -> None:
        super().__init__()
        self._models = {"mock_model": object()}
        self._loaded = True

    @property
    def active_models(self) -> list[str]:
        return ["mock_model"]

    def predict(self, text: str) -> dict[str, Prediction]:
        return {
            "mock_model": Prediction(
                label="Normal",
                probabilities={
                    "Urgency": 0.1,
                    "Scarcity": 0.1,
                    "Social Proof": 0.1,
                    "Normal": 0.7,
                },
                top_words=["cotton"],
            )
        }

    def model_metadata(self) -> list[dict]:
        return [{"model": "mock_model", "version": "test", "macro_f1": 0.5}]


def test_all_endpoints_and_validation() -> None:
    set_registry(MockRegistry())
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"text": "100% cotton shirt", "privacy_flag": True}
        )
        assert response.status_code == 200
        assert response.json()["label"] == "Normal"
        prediction_id = response.json()["prediction_id"]
        assert "per_model" in response.json()
        assert "X-Request-ID" in response.headers
        assert client.get("/history").json()["items"][0]["text"] is None
        feedback = client.post(
            "/feedback",
            json={
                "prediction_id": prediction_id,
                "user_label": "Normal",
                "comment": "Looks correct",
            },
        )
        assert feedback.status_code == 200
        assert client.get("/stats").json()["feedback"]["agreement_rate"] == 1.0

        batch = client.post(
            "/predict/batch", json={"texts": ["Add to cart", "Only one left"]}
        )
        assert batch.status_code == 200
        assert len(batch.json()["predictions"]) == 2

        assert client.get("/models").json()["models"][0]["model"] == "mock_model"
        assert client.get("/health").json()["status"] == "ok"
        assert "darkpatternguard_requests_total" in client.get("/metrics").text
        assert client.get("/docs").status_code == 200

        too_long = client.post("/predict", json={"text": "word " * 501})
        assert too_long.status_code == 422

        too_many = client.post("/predict/batch", json={"texts": ["x"] * 51})
        assert too_many.status_code == 422

    set_registry(None)
