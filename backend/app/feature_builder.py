from typing import Any
import pandas as pd

SEX_ENCODING = {"female": 0, "male": 1}

EDUCATION_ENCODING = {
    "never_attended_or_kindergarten": 1,
    "elementary": 2,
    "some_high_school": 3,
    "high_school_graduate": 4,
    "some_college_or_technical_school": 5,
    "college_graduate": 6,
}

INCOME_ENCODING = {
    "less_than_10000": 1,
    "10000_to_14999": 2,
    "15000_to_19999": 3,
    "20000_to_24999": 4,
    "25000_to_34999": 5,
    "35000_to_49999": 6,
    "50000_to_74999": 7,
    "75000_or_more": 8,
}

GENERAL_HEALTH_ENCODING = {
    "excellent": 1,
    "very_good": 2,
    "good": 3,
    "fair": 4,
    "poor": 5,
}

WORK_TYPE_MAPPING = {
    "private": "private",
    "self_employed": "self-employed",
    "government_job": "govt_job",
    "never_worked": "never_worked",
}

SMOKING_MAPPING = {
    "never": "never smoked",
    "former": "formerly smoked",
    "current": "smokes",
    "unknown": "unknown",
}

SMOKER_IMPUTATION_VALUES = {
    "diabetes": 0.4424915677220105,
    "heart_disease": 0.4431324429901778,
    "hypertension": 0.4946808510638298,
}

STROKE_WORK_TYPES = [
    "private",
    "self-employed",
    "govt_job",
    "never_worked",
]

def age_to_range(age: int) -> int:
    boundaries = (25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80)
    return 1 + sum(age >= boundary for boundary in boundaries)

def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    return round(weight_kg / ((height_cm / 100) ** 2), 2)

def encode_smoker(disease: str, smoking_status: str) -> tuple[float, int]:
    if smoking_status == "unknown":
        return SMOKER_IMPUTATION_VALUES[disease], 1
    return float(smoking_status == "current"), 0

def build_brfss_row(disease: str,data: Any) -> dict[str, Any]:
    smoker, smoker_was_missing = encode_smoker(disease,data.smoking_status)
    row = {
        "stroke": int(data.stroke),
        "sex_binary": SEX_ENCODING[data.sex],
        "age_range": age_to_range(data.age),
        "bmi": calculate_bmi(data.height_cm, data.weight_kg),
        "smoker": smoker,
        "heavy_alcohol": int(data.heavy_alcohol),
        "physical_activity": int(data.physical_activity),
        "education": EDUCATION_ENCODING[data.education],
        "income": INCOME_ENCODING[data.income],
        "high_bp": int(data.high_bp),
        "high_chol": int(data.high_chol),
        "chol_check": int(data.chol_check),
        "fruits": int(data.fruits),
        "veggies": int(data.veggies),
        "general_health": GENERAL_HEALTH_ENCODING[data.general_health],
        "mental_health": data.mental_health_days,
        "physical_health": data.physical_health_days,
        "diff_walk": int(data.difficulty_walking),
        "smoker_was_missing": smoker_was_missing,
    }

    if hasattr(data, "heart_disease"):
        row["heart_disease"] = int(data.heart_disease)

    if hasattr(data, "diabetes"):
        row["diabetes"] = int(data.diabetes)
    return row

def build_hypertension_row(data: Any) -> dict[str, Any]:
    smoker, smoker_was_missing = encode_smoker("hypertension", data.smoking_status,)
    return {
        "diabetes": int(data.diabetes),
        "sex_binary": SEX_ENCODING[data.sex],
        "age_range": age_to_range(data.age),
        "bmi": calculate_bmi(data.height_cm, data.weight_kg),
        "smoker": smoker,
        "cigs_per_day": data.cigarettes_per_day,
        "systolic_bp": data.systolic_bp,
        "diastolic_bp": data.diastolic_bp,
        "high_bp": int(data.systolic_bp >= 130 or data.diastolic_bp >= 80),
        "heart_rate": data.heart_rate,
        "bp_meds": int(data.bp_meds),
        "cholesterol": data.cholesterol,
        "high_chol": int(data.cholesterol > 200),
        "glucose": data.glucose,
        "smoker_was_missing": smoker_was_missing,
        "bp_meds_was_missing": 0,
    }

def build_stroke_row(data: Any) -> dict[str, Any]:
    return {
        "heart_disease": int(data.heart_disease),
        "hypertension": int(data.hypertension),
        "sex_binary": SEX_ENCODING[data.sex],
        "age_range": age_to_range(data.age),
        "bmi": calculate_bmi(data.height_cm, data.weight_kg),
        "glucose": data.glucose,
        "ever_married": int(data.ever_married),
        "residence_type": int(data.residence_type == "urban"),
        "smoking_status": SMOKING_MAPPING[data.smoking_status],
        "work_type": WORK_TYPE_MAPPING[data.work_type],
        "bmi_was_missing": 0,
    }

def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    def has(*columns: str) -> bool:
        return all(column in frame.columns for column in columns)

    if has("high_bp", "high_chol"):
        frame["bp_chol_both"] = (
            frame["high_bp"] * frame["high_chol"]
        )

    if has("bmi"):
        frame["bmi_obese"] = (frame["bmi"] >= 30).astype(int)

    if has("mental_health", "physical_health"):
        frame["poor_health_days"] = (
            frame["mental_health"] + frame["physical_health"]
        )

    if has("age_range", "bmi"):
        frame["age_bmi"] = frame["age_range"] * frame["bmi"]

    if has("general_health", "diff_walk"):
        frame["general_health_diffwalk"] = (
            frame["general_health"] * frame["diff_walk"]
        )

    if has("age_range", "general_health"):
        frame["age_genhealth"] = (
            frame["age_range"] * frame["general_health"]
        )

    if has("systolic_bp", "diastolic_bp"):
        frame["pulse_pressure"] = (
            frame["systolic_bp"] - frame["diastolic_bp"]
        )
        frame["mean_arterial_pressure"] = (
            frame["diastolic_bp"] + frame["pulse_pressure"] / 3
        )

    if has("high_bp", "bp_meds"):
        frame["uncontrolled_bp"] = (
            (frame["high_bp"] == 1)
            & (frame["bp_meds"] == 0)
        ).astype(int)

    if has("heart_rate"):
        frame["tachycardia"] = (frame["heart_rate"] > 100).astype(int)

    if has("cigs_per_day"):
        frame["heavy_smoker"] = (
            frame["cigs_per_day"] >= 20
        ).astype(int)

    if has("glucose"):
        frame["hyperglycemia"] = (
            frame["glucose"] >= 126
        ).astype(int)

    if has("glucose", "bmi"):
        frame["glucose_bmi"] = frame["glucose"] * frame["bmi"]

    if has("age_range", "glucose"):
        frame["age_glucose"] = (
            frame["age_range"] * frame["glucose"]
        )

    if has("high_bp", "high_chol", "bmi"):
        frame["metabolic_risk_score"] = (
            frame["high_bp"]
            + frame["high_chol"]
            + frame["bmi_obese"]
        )

        if "hyperglycemia" in frame.columns:
            frame["metabolic_risk_score"] += frame["hyperglycemia"]

    if has("work_type"):
        category = pd.Categorical(
            frame["work_type"],
            categories=STROKE_WORK_TYPES,
        )

        dummies = pd.get_dummies(
            category,
            prefix="work_type",
            dtype=int,
        )

        dummies.index = frame.index
        frame = pd.concat([frame, dummies], axis=1)

    if has("smoking_status"):
        frame["smoking_current"] = (
            frame["smoking_status"] == "smokes"
        ).astype(int)

        frame["smoking_former"] = (
            frame["smoking_status"] == "formerly smoked"
        ).astype(int)

        frame["smoking_unknown"] = (
            frame["smoking_status"] == "unknown"
        ).astype(int)

    lifestyle = [
        column
        for column in (
            "physical_activity",
            "fruits",
            "veggies",
        )
        if column in frame
    ]

    risks = [
        column
        for column in (
            "smoker",
            "heavy_alcohol",
        )
        if column in frame
    ]

    if lifestyle and risks:
        frame["lifestyle_score"] = (
            frame[lifestyle].sum(axis=1)
            - frame[risks].sum(axis=1)
        )

    return frame


def finish_feature_row(
    raw_row: dict[str, Any],
    feature_cols: list[str],
) -> pd.DataFrame:
    frame = add_engineered_features(pd.DataFrame([raw_row]))

    missing = [
        column
        for column in feature_cols
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(f"Feature builder did not produce required columns: {missing}")
    return frame.loc[:, feature_cols]

def build_features(disease: str, request: Any, feature_cols: list[str],) -> pd.DataFrame:
    if disease in {"diabetes", "heart_disease"}:
        raw = build_brfss_row(disease, request)
    elif disease == "hypertension":
        raw = build_hypertension_row(request)
    elif disease == "stroke":
        raw = build_stroke_row(request)
    else:
        raise ValueError(f"Unsupported disease: {disease}")

    return finish_feature_row(raw, feature_cols)