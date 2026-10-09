"""Create prediction, feedback, and model run tables."""

import sqlalchemy as sa

from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the initial database schema."""
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("per_model_probs", sa.JSON(), nullable=False),
        sa.Column("final_label", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
    )
    op.create_index("ix_predictions_text_hash", "predictions", ["text_hash"])
    op.create_index("ix_predictions_final_label", "predictions", ["final_label"])
    op.create_table(
        "feedback",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("prediction_id", sa.Integer(), nullable=False),
        sa.Column("user_label", sa.String(length=32), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["predictions.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_feedback_prediction_id", "feedback", ["prediction_id"])
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("version", sa.String(length=128), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_model_runs_name", "model_runs", ["name"])


def downgrade() -> None:
    """Drop the initial database schema."""
    op.drop_index("ix_model_runs_name", table_name="model_runs")
    op.drop_table("model_runs")
    op.drop_index("ix_feedback_prediction_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_predictions_final_label", table_name="predictions")
    op.drop_index("ix_predictions_text_hash", table_name="predictions")
    op.drop_table("predictions")
