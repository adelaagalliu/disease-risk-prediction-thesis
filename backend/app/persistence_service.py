from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from .feature_builder import calculate_bmi
from .models import HealthRecord, Prediction, User


def save_prediction(
    db: Session,
    user: User,
    disease: str,
    request: Any,
    response_payload: dict[str, Any],
) -> Prediction:
    request_payload = request.model_dump(mode="json")
    health_record = HealthRecord(
        user_id=user.id,
        source_disease=disease,
        height_cm=request.height_cm,
        weight_kg=request.weight_kg,
        bmi=calculate_bmi(request.height_cm, request.weight_kg),
        systolic_bp=getattr(request, "systolic_bp", None),
        diastolic_bp=getattr(request, "diastolic_bp", None),
        heart_rate=getattr(request, "heart_rate", None),
        glucose=getattr(request, "glucose", None),
        cholesterol=getattr(request, "cholesterol", None),
        request_payload=request_payload,
    )
    prediction = Prediction(
        user_id=user.id,
        health_record=health_record,
        disease=disease,
        probability=response_payload["probability"],
        classification=response_payload["classification"],
        threshold=response_payload["threshold"],
        model_name=response_payload["model"],
        model_version=response_payload["model_version"],
        result_payload=response_payload,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def utc_iso(value: datetime) -> str:
    return value.isoformat()
