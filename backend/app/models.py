import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .database import Base


JsonType = JSON().with_variant(JSONB, "postgresql")


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    health_records: Mapped[list["HealthRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class HealthRecord(Base):
    __tablename__ = "health_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    source_disease: Mapped[str] = mapped_column(String(32))
    height_cm: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    weight_kg: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    bmi: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    systolic_bp: Mapped[Decimal | None] = mapped_column(Numeric(5, 1))
    diastolic_bp: Mapped[Decimal | None] = mapped_column(Numeric(5, 1))
    heart_rate: Mapped[Decimal | None] = mapped_column(Numeric(5, 1))
    glucose: Mapped[Decimal | None] = mapped_column(Numeric(6, 1))
    cholesterol: Mapped[Decimal | None] = mapped_column(Numeric(6, 1))
    request_payload: Mapped[dict] = mapped_column(JsonType)

    user: Mapped[User] = relationship(back_populates="health_records")
    prediction: Mapped["Prediction"] = relationship(
        back_populates="health_record", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        Index("ix_health_records_user_recorded", "user_id", "recorded_at"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    health_record_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("health_records.id", ondelete="CASCADE"), unique=True
    )
    disease: Mapped[str] = mapped_column(String(32), index=True)
    probability: Mapped[Decimal] = mapped_column(Numeric(8, 7))
    classification: Mapped[str] = mapped_column(String(20))
    threshold: Mapped[Decimal] = mapped_column(Numeric(8, 7))
    model_name: Mapped[str] = mapped_column(String(50))
    model_version: Mapped[str] = mapped_column(String(50))
    result_payload: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    user: Mapped[User] = relationship(back_populates="predictions")
    health_record: Mapped[HealthRecord] = relationship(back_populates="prediction")

    __table_args__ = (
        Index("ix_predictions_user_disease_created", "user_id", "disease", "created_at"),
        CheckConstraint(
            "probability >= 0 AND probability <= 1",
            name="ck_prediction_probability",
        ),
        CheckConstraint(
            "threshold >= 0 AND threshold <= 1",
            name="ck_prediction_threshold",
        ),
        CheckConstraint(
            "disease IN ('diabetes', 'heart_disease', 'hypertension', 'stroke')",
            name="ck_prediction_disease",
        ),
        CheckConstraint(
            "classification IN ('elevated', 'not_elevated')",
            name="ck_prediction_classification",
        ),
    )
