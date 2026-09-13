from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "datasets_processed"
BACKGROUND_SIZE = 100
RANDOM_STATE = 44

TREE_MODELS = {"catboost", "lightgbm", "random_forest", "xgboost"}

FEATURE_LABELS = {
    "age_range": "Age group",
    "sex_binary": "Sex",
    "bmi": "Body mass index (BMI)",
    "bmi_obese": "BMI obesity indicator",
    "smoker": "Current smoking",
    "smoker_was_missing": "Unknown smoking status",
    "heavy_alcohol": "Heavy alcohol consumption",
    "physical_activity": "Physical activity",
    "fruits": "Regular fruit consumption",
    "veggies": "Regular vegetable consumption",
    "general_health": "Self-reported general health",
    "mental_health": "Poor mental-health days",
    "physical_health": "Poor physical-health days",
    "diff_walk": "Difficulty walking",
    "high_bp": "High blood pressure",
    "high_chol": "High cholesterol",
    "chol_check": "Cholesterol screening",
    "heart_disease": "Heart-disease history",
    "stroke": "Stroke history",
    "diabetes": "Diabetes history",
    "education": "Education category",
    "income": "Income category",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "pulse_pressure": "Pulse pressure",
    "mean_arterial_pressure": "Mean arterial pressure",
    "heart_rate": "Heart rate",
    "bp_meds": "Blood-pressure medication",
    "cholesterol": "Measured cholesterol",
    "glucose": "Measured glucose",
    "cigs_per_day": "Cigarettes per day",
    "ever_married": "Marital-history category",
    "residence_type": "Residence type",
    "age_bmi": "Combined age and BMI effect",
    "age_glucose": "Combined age and glucose effect",
    "glucose_bmi": "Combined glucose and BMI effect",
    "age_genhealth": "Combined age and general-health effect",
    "general_health_diffwalk": "Combined general-health and walking-difficulty effect",
    "bp_chol_both": "Combined blood-pressure and cholesterol indicator",
    "poor_health_days": "Combined poor-health days",
    "metabolic_risk_score": "Combined metabolic-risk indicators",
    "lifestyle_score": "Combined lifestyle indicators",
    "uncontrolled_bp": "High blood pressure without recorded medication",
    "tachycardia": "High heart-rate indicator",
    "heavy_smoker": "Heavy-smoking indicator",
    "hyperglycemia": "High-glucose indicator",
}

INCOME_LABELS = {
    "ALL": {
        "less_than_10000": "Nen 800 000 leke",
        "10000_to_14999": "800 000–1 200 000 leke",
        "15000_to_19999": "1 200 000–1 600 000 leke",
        "20000_to_24999": "1 600 000–2 000 000 leke",
        "25000_to_34999": "2 000 000–2 800 000 leke",
        "35000_to_49999": "2 800 000–4 000 000 leke",
        "50000_to_74999": "4 000 000–6 000 000 leke",
        "75000_or_more": "6 000 000 leke ose me shume",
    },
    "EUR": {
        "less_than_10000": "Nen 8 600 euro",
        "10000_to_14999": "8 600–12 900 euro",
        "15000_to_19999": "12 900–17 200 euro",
        "20000_to_24999": "17 200–21 600 euro",
        "25000_to_34999": "21 600–30 200 euro",
        "35000_to_49999": "30 200–43 100 euro",
        "50000_to_74999": "43 100–64 700 euro",
        "75000_or_more": "64 700 euro ose me shume",
    },
}

LIFESTYLE_MODIFIABLE_FEATURES = {
    "bmi",
    "bmi_obese",
    "smoker",
    "heavy_alcohol",
    "physical_activity",
    "fruits",
    "veggies",
    "cigs_per_day",
    "heavy_smoker",
    "lifestyle_score",
}
CLINICALLY_MODIFIABLE_FEATURES = {
    "general_health",
    "mental_health",
    "physical_health",
    "diff_walk",
    "high_bp",
    "high_chol",
    "systolic_bp",
    "diastolic_bp",
    "pulse_pressure",
    "mean_arterial_pressure",
    "heart_rate",
    "cholesterol",
    "glucose",
    "poor_health_days",
    "uncontrolled_bp",
    "tachycardia",
    "hyperglycemia",
    "metabolic_risk_score",
}
CARE_MODIFIABLE_FEATURES = {
    "chol_check",
    "bp_meds",
}
DERIVED_MODIFIABLE_FEATURES = {
    "glucose_bmi",
    "general_health_diffwalk",
    "bp_chol_both",
}
MODIFIABLE_FEATURES = (
    LIFESTYLE_MODIFIABLE_FEATURES
    | CLINICALLY_MODIFIABLE_FEATURES
    | CARE_MODIFIABLE_FEATURES
    | DERIVED_MODIFIABLE_FEATURES
)

def convert_numpy_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def display_feature_value(feature: str, request: Any, value: Any) -> Any:
    """Return a user-facing value instead of an encoded/interaction number."""
    bmi = round(request.weight_kg / ((request.height_cm / 100) ** 2), 2)
    if feature == "income":
        currency = getattr(request, "income_display_currency", "ALL")
        return INCOME_LABELS[currency][request.income]
    direct_values = {
        "age_range": f"{request.age} years",
        "sex_binary": request.sex,
        "bmi": bmi,
        "bmi_obese": f"BMI {bmi}",
        "smoker": request.smoking_status,
        "glucose": (
            request.glucose if hasattr(request, "glucose") else None
        ),
        "age_bmi": f"Age {request.age}; BMI {bmi}",
        "glucose_bmi": (
            f"Glucose {request.glucose:g} mg/dL; BMI {bmi}"
            if hasattr(request, "glucose") else None
        ),
        "age_glucose": (
            f"Age {request.age}; glucose {request.glucose:g} mg/dL"
            if hasattr(request, "glucose") else None
        ),
    }
    if feature in direct_values and direct_values[feature] is not None:
        return direct_values[feature]

    if hasattr(request, "general_health"):
        general_health_values = {
            "general_health": request.general_health.replace("_", " "),
            "age_genhealth": (
                f"Age {request.age}; general health: "
                f"{request.general_health.replace('_', ' ')}"
            ),
            "general_health_diffwalk": (
                f"General health: {request.general_health.replace('_', ' ')}; "
                f"difficulty walking: {'yes' if request.difficulty_walking else 'no'}"
            ),
            "poor_health_days": (
                f"{request.mental_health_days} mental-health and "
                f"{request.physical_health_days} physical-health days/month"
            ),
        }
        if feature in general_health_values:
            return general_health_values[feature]

    if feature == "bp_chol_both" and hasattr(request, "high_bp"):
        return (
            f"High blood pressure: {'yes' if request.high_bp else 'no'}; "
            f"high cholesterol: {'yes' if request.high_chol else 'no'}"
        )

    return convert_numpy_value(value)

class ExplanationService:
    def __init__(self) -> None:
        self._explainers: dict[str, tuple[Any, Any, list[str], str]] = {}

    def create_explainer(self, disease: str, model_data: dict[str, Any]):
        pipeline = model_data["pipeline"]
        feature_cols = model_data["feature_cols"]
        model_name = model_data["model_name"]
        preprocessor = pipeline.named_steps["preprocessing"]
        model = pipeline.named_steps["model"]

        train_path = PROCESSED_DATA_DIR / f"{disease}_train.csv"
        train_frame = pd.read_csv(train_path, usecols=feature_cols)
        background_raw = train_frame.sample(
            n=min(BACKGROUND_SIZE, len(train_frame)),
            random_state=RANDOM_STATE,
        )
        background = preprocessor.transform(background_raw)

        try:
            feature_names = [
                name.split("__", 1)[-1]
                for name in preprocessor.get_feature_names_out()
            ]
        except Exception:
            feature_names = list(feature_cols)

        if model_name in TREE_MODELS:
            try:
                explainer = shap.TreeExplainer(
                    model,
                    data=background,
                    model_output="probability",
                )
                units = "probability"
            except (NotImplementedError, ValueError):
                explainer = shap.TreeExplainer(
                    model,
                    feature_perturbation="tree_path_dependent",
                )
                units = "raw_model_output"
        elif model_name == "logreg":
            explainer = shap.LinearExplainer(model, background)
            units = "log_odds"
        else:
            raise ValueError(f"Unsupported model for SHAP: {model_name}")

        cached = (explainer, preprocessor, feature_names, units)
        self._explainers[disease] = cached
        return cached

    def explain(
        self,
        disease: str,
        model_data: dict[str, Any],
        features: pd.DataFrame,
        request: Any,
        limit: int = 10,
    ) -> tuple[list[dict[str, Any]], str]:
        cached = self._explainers.get(disease)
        if cached is None:
            cached = self.create_explainer(disease, model_data)
        explainer, preprocessor, feature_names, units = cached

        transformed = preprocessor.transform(features)
        explanation = explainer(transformed)
        values = np.asarray(explanation.values)
        if values.ndim == 3:
            values = values[:, :, 1]
        row_values = values[0]

        transformed_row = np.asarray(transformed)[0]
        factors = []
        for feature, value, shap_value in zip(
            feature_names,
            transformed_row,
            row_values,
        ):
            factors.append(
                {
                    "feature": feature,
                    "label": FEATURE_LABELS.get(
                        feature,
                        feature.replace("_", " ").capitalize(),
                    ),
                    "value": display_feature_value(feature, request, value),
                    "direction": (
                        "increases_estimate"
                        if float(shap_value) >= 0
                        else "decreases_estimate"
                    ),
                    "importance": abs(float(shap_value)),
                    "modifiable": feature in MODIFIABLE_FEATURES,
                }
            )

        factors.sort(key=lambda item: item["importance"], reverse=True)
        return factors[:limit], units

explanation_service = ExplanationService()
