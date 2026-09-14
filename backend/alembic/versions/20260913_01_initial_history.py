from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260903_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "health_records",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("source_disease", sa.String(length=32), nullable=False),
        sa.Column("height_cm", sa.Numeric(5, 2), nullable=False),
        sa.Column("weight_kg", sa.Numeric(5, 2), nullable=False),
        sa.Column("bmi", sa.Numeric(5, 2), nullable=False),
        sa.Column("systolic_bp", sa.Numeric(5, 1), nullable=True),
        sa.Column("diastolic_bp", sa.Numeric(5, 1), nullable=True),
        sa.Column("heart_rate", sa.Numeric(5, 1), nullable=True),
        sa.Column("glucose", sa.Numeric(6, 1), nullable=True),
        sa.Column("cholesterol", sa.Numeric(6, 1), nullable=True),
        sa.Column("request_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_health_records_user_id", "health_records", ["user_id"])
    op.create_index("ix_health_records_recorded_at", "health_records", ["recorded_at"])
    op.create_index("ix_health_records_user_recorded", "health_records", ["user_id", "recorded_at"])

    op.create_table(
        "predictions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("health_record_id", sa.Uuid(), nullable=False),
        sa.Column("disease", sa.String(length=32), nullable=False),
        sa.Column("probability", sa.Numeric(8, 7), nullable=False),
        sa.Column("classification", sa.String(length=20), nullable=False),
        sa.Column("threshold", sa.Numeric(8, 7), nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("result_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("probability >= 0 AND probability <= 1", name="ck_prediction_probability"),
        sa.CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_prediction_threshold"),
        sa.CheckConstraint("disease IN ('diabetes', 'heart_disease', 'hypertension', 'stroke')", name="ck_prediction_disease"),
        sa.CheckConstraint("classification IN ('elevated', 'not_elevated')", name="ck_prediction_classification"),
        sa.ForeignKeyConstraint(["health_record_id"], ["health_records.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("health_record_id"),
    )
    op.create_index("ix_predictions_user_id", "predictions", ["user_id"])
    op.create_index("ix_predictions_disease", "predictions", ["disease"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("ix_predictions_user_disease_created", "predictions", ["user_id", "disease", "created_at"])

def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("health_records")
    op.drop_table("users")
