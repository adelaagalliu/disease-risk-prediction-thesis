from contextlib import asynccontextmanager
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import create_access_token, get_current_user, hash_password, verify_password
from .database import get_db
from .explanation_service import explanation_service
from .feature_builder import build_features
from .model_registry import registry
from .models import HealthRecord, Prediction, User
from .persistence_service import save_prediction
from .recommendation_service import create_recommendations
from .schemas import (
    DiabetesRequest,
    Disease,
    HeartRequest,
    HypertensionRequest,
    LoginRequest,
    PredictionHistoryItem,
    PredictionResponse,
    RegisterRequest,
    StrokeRequest,
    TokenResponse,
    TrendMetric,
    TrendPoint,
    UserResponse,
)

REQUEST_SCHEMAS = {
    "diabetes": DiabetesRequest,
    "heart_disease": HeartRequest,
    "hypertension": HypertensionRequest,
    "stroke": StrokeRequest,
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()
    yield

app = FastAPI(
    title="Disease Risk Prediction API",
    version="1.1.0",
    lifespan=lifespan,
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/auth/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> User:
    user = User(email=payload.email, password_hash=hash_password(payload.password))
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=409,
            detail="Kjo adresë email është regjistruar më parë.",
        ) from exc
    db.refresh(user)
    return user

@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == payload.email))
    if user is None or not user.is_active or not verify_password(
        payload.password, user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Adresa email ose fjalëkalimi është i pasaktë.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_in = create_access_token(user.id)
    return TokenResponse(access_token=token, expires_in=expires_in)

@app.get("/users/me", response_model=UserResponse)
def current_user(user: User = Depends(get_current_user)) -> User:
    return user

@app.post("/predict/{disease}", response_model=PredictionResponse)
def predict(
    disease: Disease,
    payload: dict[str, Any],
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    schema = REQUEST_SCHEMAS[disease]
    try:
        request = schema.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_context=False),
        ) from exc

    model_data = registry.get(disease)
    features = build_features(disease, request, model_data["feature_cols"])
    probability = float(model_data["pipeline"].predict_proba(features)[0, 1])
    threshold = model_data["threshold"]
    top_factors, explanation_units = explanation_service.explain(
        disease,
        model_data,
        features,
        request,
    )
    recommendations = create_recommendations(
        disease,
        model_data,
        request,
        probability,
        top_factors,
    )
    response_payload = {
        "contract_version": "1.1.0",
        "disease": disease,
        "model": model_data["model_name"],
        "model_version": "1",
        "probability": round(probability, 6),
        "classification": (
            "elevated" if probability >= threshold else "not_elevated"
        ),
        "threshold": threshold,
        "top_factors": top_factors,
        "recommendations": recommendations,
        "explanation_units": explanation_units,
        "explanation_note": (
            "Vlerat SHAP tregojnë se si veçoritë e rritën ose e ulën rezultatin "
            "e modelit kundrejt vlerës bazë. Ato nuk tregojnë lidhje shkakësore."
        ),
        "warnings": [],
        "disclaimer": "Ky është një vlerësim kërkimor dhe jo një diagnozë mjekësore.",
    }
    prediction = save_prediction(db, user, disease, request, response_payload)
    return PredictionResponse(
        **response_payload,
        prediction_id=prediction.id,
        created_at=prediction.created_at,
    )

@app.get("/history/predictions", response_model=list[PredictionHistoryItem])
def prediction_history(
    disease: Disease | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[PredictionHistoryItem]:
    statement = (
        select(Prediction, HealthRecord)
        .join(HealthRecord, Prediction.health_record_id == HealthRecord.id)
        .where(Prediction.user_id == user.id)
        .order_by(Prediction.created_at.desc())
        .limit(limit)
    )
    if disease is not None:
        statement = statement.where(Prediction.disease == disease)

    return [
        PredictionHistoryItem(
            id=prediction.id,
            disease=prediction.disease,
            probability=float(prediction.probability),
            classification=prediction.classification,
            threshold=float(prediction.threshold),
            model=prediction.model_name,
            model_version=prediction.model_version,
            created_at=prediction.created_at,
            inputs=record.request_payload,
        )
        for prediction, record in db.execute(statement).all()
    ]

@app.get("/history/predictions/{prediction_id}", response_model=PredictionResponse)
def prediction_detail(
    prediction_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    prediction = db.scalar(
        select(Prediction).where(Prediction.id == prediction_id, Prediction.user_id == user.id)
    )
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")
    return PredictionResponse(
        **{**prediction.result_payload, "prediction_id": prediction.id, "created_at": prediction.created_at}
    )


@app.get("/history/trends/{metric}", response_model=list[TrendPoint])
def health_trend(
    metric: TrendMetric,
    limit: int = Query(default=250, ge=1, le=1000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[TrendPoint]:
    metric_column = getattr(HealthRecord, metric)
    statement = (
        select(HealthRecord)
        .where(HealthRecord.user_id == user.id, metric_column.is_not(None))
        .order_by(HealthRecord.recorded_at.desc())
        .limit(limit)
    )
    records = list(db.scalars(statement).all())
    return [
        TrendPoint(
            health_record_id=record.id,
            disease=record.source_disease,
            recorded_at=record.recorded_at,
            value=float(getattr(record, metric)),
        )
        for record in reversed(records)
    ]
