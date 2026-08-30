from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .explanation_service import explanation_service
from .feature_builder import build_features
from .model_registry import registry
from .recommendation_service import create_recommendations
from .schemas import (
    DiabetesRequest,
    HeartRequest,
    HypertensionRequest,
    PredictionResponse,
    StrokeRequest,
)

Disease = Literal["diabetes", "heart_disease", "hypertension", "stroke"]
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

@app.post("/predict/{disease}", response_model=PredictionResponse)
def predict(disease: Disease, payload: dict[str, Any]) -> PredictionResponse:
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

    warnings: list[str] = []
    if disease == "hypertension":
        warnings.append(
            "This output estimates a prevalent hypertension pattern, "
            "not future hypertension risk."
        )

    return PredictionResponse(
        contract_version="1.2.0",
        disease=disease,
        model=model_data["model_name"],
        model_version="1",
        probability=round(probability, 6),
        classification=(
            "elevated" if probability >= threshold else "not_elevated"
        ),
        threshold=threshold,
        top_factors=top_factors,
        recommendations=recommendations,
        explanation_units=explanation_units,
        explanation_note=(
            "SHAP values describe how features pushed this model output higher or "
            "lower relative to its baseline. They are not causal effects."
        ),
        warnings=warnings,
        disclaimer="Research estimate, not a medical diagnosis.",
    )
