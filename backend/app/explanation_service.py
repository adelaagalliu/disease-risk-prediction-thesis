from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "datasets_processed"
BACKGROUND_SIZE = 100
RANDOM_STATE = 44

FEATURE_LABELS = {
    "age_range": "Grupmosha",
    "sex_binary": "Gjinia",
    "bmi": "Indeksi i mases trupore (BMI)",
    "bmi_obese": "Tregues i obezitetit",
    "smoker": "Pirja aktuale e duhanit",
    "smoker_was_missing": "Statusi i panjohur i duhanpirjes",
    "heavy_alcohol": "Konsumimi i larte i alkoolit",
    "physical_activity": "Aktiviteti fizik",
    "fruits": "Konsumimi i rregullt i frutave",
    "veggies": "Konsumimi i rregullt i perimeve",
    "general_health": "Gjendja e pergjithshme shendetesore",
    "mental_health": "Ditet me shendet te dobet mendor",
    "physical_health": "Ditet me shendet te dobet fizik",
    "diff_walk": "Veshtiresia ne ecje",
    "high_bp": "Tensioni i larte i gjakut",
    "high_chol": "Kolesteroli i larte",
    "chol_check": "Kontrolli i kolesterolit",
    "heart_disease": "Semundje e zemres",
    "stroke": "Insulti cerebral",
    "diabetes": "Diabeti",
    "education": "Arsimi",
    "income": "Te ardhurat vjetore te familjes",
    "systolic_bp": "Tensioni sistolik",
    "diastolic_bp": "Tensioni diastolik",
    "pulse_pressure": "Presioni i pulsit",
    "mean_arterial_pressure": "Presioni mesatar arterial",
    "heart_rate": "Ritmi i zemres",
    "bp_meds": "Ilacet për tensionin",
    "cholesterol": "Kolesteroli i matur",
    "glucose": "Glukoza e matur",
    "cigs_per_day": "Cigaret në dite",
    "ever_married": "Gjendja martesore",
    "residence_type": "Lloji i vendbanimit",
    "age_bmi": "Efekti i kombinuar i moshes dhe BMI",
    "age_glucose": "Efekti i kombinuar i moshes dhe glukozes",
    "glucose_bmi": "Efekti i kombinuar i glukozes dhe BMI",
    "age_genhealth": "Efekti i kombinuar i moshes dhe shendetit",
    "general_health_diffwalk": "Efekti i kombinuar i shendetit dhe ecjes",
    "bp_chol_both": "Treguesi i kombinuar i tensionit dhe kolesterolit",
    "poor_health_days": "Ditet me shendet të dobet",
    "metabolic_risk_score": "Treguesit e kombinuar te rrezikut metabolik",
    "lifestyle_score": "Treguesit e kombinuar te stilit të jeteses",
    "uncontrolled_bp": "Tension i larte pa mjekim",
    "tachycardia": "Treguesi i ritmit te larte te zemres",
    "heavy_smoker": "Treguesi i duhanpirjes se larte",
    "hyperglycemia": "Treguesi i glukozes se larte",
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

# kthen vlerat NumPy ne vlera te perdorshme nga API
def convert_numpy_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value

# formaton vleren e vecorise per perdoruesin
def display_feature_value(feature: str, request: Any, value: Any) -> Any:
    bmi = round(request.weight_kg / ((request.height_cm / 100) ** 2), 2)
    if feature == "income":
            currency = getattr(request, "income_display_currency", "ALL")
            return INCOME_LABELS[currency][request.income]
    glucose = getattr(request, "glucose", None)
    direct_values = {
        "age_range": f"{request.age} vjeç",
        "sex_binary": request.sex,
        "bmi": bmi,
        "bmi_obese": f"BMI {bmi}",
        "smoker": request.smoking_status,
        "glucose": glucose,
        "age_bmi": f"Mosha {request.age}; BMI {bmi}",
        "glucose_bmi": (
            f"Glukoza {glucose:g} mg/dL; BMI {bmi}"
            if glucose is not None else None
        ),
        "age_glucose": (
            f"Mosha {request.age}; glukoza {glucose:g} mg/dL"
            if glucose is not None else None
        ),
    }
    if feature in direct_values and direct_values[feature] is not None:
        return direct_values[feature]

    if hasattr(request, "general_health"):
        general_health = request.general_health.replace("_", " ")
        general_health_values = {
            "general_health": general_health,
            "age_genhealth": (
                f"Mosha {request.age}; shendeti i pergjithshem: {general_health}"
            ),
            "general_health_diffwalk": (
                f"Shendeti i pergjithshem: {general_health}; "
                f"veshtirësi ne ecje: {'po' if request.difficulty_walking else 'jo'}"
            ),
            "poor_health_days": (
                f"{request.mental_health_days} dite me shendet te dobet mendor dhe "
                f"{request.physical_health_days} dite me shendet te dobet fizik ne muaj"
            ),
        }
        if feature in general_health_values:
            return general_health_values[feature]

    if feature == "bp_chol_both" and hasattr(request, "high_bp"):
        return (
            f"Tension i larte: {'po' if request.high_bp else 'jo'}; "
            f"kolesterol i larte: {'po' if request.high_chol else 'jo'}"
        )

    return convert_numpy_value(value)

class ExplanationService:
    # pergatit memorien e shpjeguesve SHAP
    def __init__(self) -> None:
        self._explainers: dict[str, tuple[Any, Any, list[str], str]] = {}

    # krijon dhe ruan shpjeguesin SHAP per semundjen
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

        if model_name != "logreg":
            raise ValueError(f"Modeli nuk mbeshtetet nga SHAP: {model_name}")

        explainer = shap.LinearExplainer(model, background)
        cached = (explainer, preprocessor, feature_names, "log_odds")
        self._explainers[disease] = cached
        return cached

    # kthen faktoret kryesor qe ndikojne në parashikim
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
        transformed_row = (
            transformed.toarray()[0]
            if hasattr(transformed, "toarray")
            else np.asarray(transformed)[0]
        )
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