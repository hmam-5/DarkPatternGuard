"""Database model tests using an isolated in-memory SQLite database."""

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.backend.models import Base, FeedbackRecord, PredictionRecord


def test_prediction_retrieval_and_feedback_linking() -> None:
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        prediction = PredictionRecord(
            text_hash="a" * 64,
            text=None,
            per_model_probs={"mock": {"Normal": 0.9}},
            final_label="Normal",
            confidence=0.9,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        feedback = FeedbackRecord(
            prediction_id=prediction.id,
            user_label="Normal",
            comment="confirmed",
        )
        session.add(feedback)
        session.commit()
        retrieved = session.scalar(
            select(PredictionRecord).where(PredictionRecord.id == prediction.id)
        )
        assert retrieved is not None
        assert retrieved.text is None
        assert retrieved.feedback[0].user_label == "Normal"
        assert feedback.prediction_id == prediction.id
