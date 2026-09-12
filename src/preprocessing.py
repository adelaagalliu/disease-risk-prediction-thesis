import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from config import (
    CONTINUOUS_COLS, GROUP_COL, ORDINAL_COLS, STROKE_WORK_TYPE_CATEGORIES,
)


def add_engineered_features(df):
    df = df.copy()

    def has(*cols):
        return all(c in df.columns for c in cols)

    if has("high_bp", "high_chol"):
        df["bp_chol_both"] = df["high_bp"] * df["high_chol"]

    if has("bmi"):
        df["bmi_obese"] = (df["bmi"] >= 30).astype(int)

    if has("mental_health", "physical_health"):
        df["poor_health_days"] = df["mental_health"] + df["physical_health"]

    if has("age_range", "bmi"):
        df["age_bmi"] = df["age_range"] * df["bmi"]

    if has("general_health", "diff_walk"):
        df["general_health_diffwalk"] = (
            df["general_health"] * df["diff_walk"]
        )

    if has("age_range", "general_health"):
        df["age_genhealth"] = (
            df["age_range"] * df["general_health"]
        )

    if has("systolic_bp", "diastolic_bp"):
        df["pulse_pressure"] = (
            df["systolic_bp"] - df["diastolic_bp"]
        )

        df["mean_arterial_pressure"] = (
            df["diastolic_bp"] + df["pulse_pressure"] / 3
        )

    if has("high_bp", "bp_meds"):
        df["uncontrolled_bp"] = (
            (df["high_bp"] == 1) & (df["bp_meds"] == 0)
        ).astype(int)

    if has("heart_rate"):
        df["tachycardia"] = (df["heart_rate"] > 100).astype(int)

    if has("cigs_per_day"):
        df["heavy_smoker"] = (df["cigs_per_day"] >= 20).astype(int)

    if has("glucose"):
        df["hyperglycemia"] = (df["glucose"] >= 126).astype(int)

    if has("glucose", "bmi"):
        df["glucose_bmi"] = df["glucose"] * df["bmi"]

    if has("age_range", "glucose"):
        df["age_glucose"] = df["age_range"] * df["glucose"]

    if has("high_bp", "high_chol", "bmi"):
        df["metabolic_risk_score"] = (
            df["high_bp"]
            + df["high_chol"]
            + df["bmi_obese"]
        )

        if "hyperglycemia" in df.columns:
            df["metabolic_risk_score"] += df["hyperglycemia"]

    if has("work_type"):
        cat = pd.Categorical(
            df["work_type"],
            categories=STROKE_WORK_TYPE_CATEGORIES
        )
        dummies = pd.get_dummies(
            cat,
            prefix="work_type",
            dtype=int
        )
        dummies.index = df.index
        df = pd.concat([df, dummies], axis=1)

    # Stroke uses its full smoking-history variable instead of the binary
    # smoker feature used by the other datasets. Never-smoked is the reference
    # group; current, former, and unknown are explicit indicator columns.
    if has("smoking_status"):
        df["smoking_current"] = (
            df["smoking_status"] == "smokes"
        ).astype(int)
        df["smoking_former"] = (
            df["smoking_status"] == "formerly smoked"
        ).astype(int)
        df["smoking_unknown"] = (
            df["smoking_status"] == "unknown"
        ).astype(int)

    lifestyle_cols = [
        c for c in ["physical_activity", "fruits", "veggies"]
        if c in df.columns
    ]
    risk_cols = [
        c for c in ["smoker", "heavy_alcohol"]
        if c in df.columns
    ]
    if lifestyle_cols and risk_cols:
        df["lifestyle_score"] = (
            df[lifestyle_cols].sum(axis=1) - df[risk_cols].sum(axis=1)
        )

    return df

def get_feature_columns(df, target: str, drop_cols: list[str]) -> list[str]:
    exclude = set(drop_cols) | {target, GROUP_COL}
    return [c for c in df.columns if c not in exclude]

def build_preprocessor(feature_cols: list[str], scale_continuous: bool) -> ColumnTransformer:
    """
    feature_cols: the actual columns present for this run
    scale_continuous: whether continuous and ordinal predictors are standardized.

    Ordinal columns (age_range, education, income, general_health) are
    numeric but not on a 0/1 scale like the binary columns -- leaving them
    on "passthrough" next to unscaled continuous features is exactly what
    was starving lbfgs of convergence. They're scaled together with the
    continuous columns for stable Logistic Regression optimization.
    """
    ordinal_feature = [c for c in ORDINAL_COLS if c in feature_cols]
    continuous_feature = [c for c in CONTINUOUS_COLS if c in feature_cols]
    scale_feature = continuous_feature + ordinal_feature
    passthrough_feature = [
        c for c in feature_cols
        if c not in scale_feature
    ]

    if scale_continuous and scale_feature:
        transformers = [
            ("scaled", StandardScaler(), scale_feature),
            ("passthrough", "passthrough", passthrough_feature),
        ]
    else:
        transformers = [("passthrough", "passthrough", feature_cols)]

    return ColumnTransformer(transformers, remainder="drop")
