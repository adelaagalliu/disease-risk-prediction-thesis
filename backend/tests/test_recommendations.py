import numpy as np

from app.recommendation_service import create_recommendations
from app.schemas import HeartRequest


class FakePipeline:
    def predict_proba(self, features):
        row = features.iloc[0]
        probability = (
            0.10
            + 0.20 * row["high_bp"]
            + 0.15 * row["smoker"]
            + 0.10 * (1 - row["physical_activity"])
            + 0.002 * (row["bmi"] - 20)
        )
        return np.array([[1 - probability, probability]])


def heart_request() -> HeartRequest:
    return HeartRequest(
        diabetes=False,
        stroke=False,
        sex="female",
        age=52,
        height_cm=170.0,
        weight_kg=100.0,
        smoking_status="current",
        heavy_alcohol=False,
        physical_activity=False,
        education="college_graduate",
        income="50000_to_74999",
        high_bp=True,
        high_chol=True,
        chol_check=True,
        fruits=True,
        veggies=True,
        general_health="fair",
        mental_health_days=10,
        physical_health_days=5,
        difficulty_walking=False,
    )


def test_recommendations_use_source_inputs_and_rank_actual_reductions():
    request = heart_request()
    feature_cols = [
        "bmi",
        "smoker",
        "physical_activity",
        "high_bp",
        "high_chol",
        "bp_chol_both",
    ]
    model_data = {"pipeline": FakePipeline(), "feature_cols": feature_cols}
    original_features = __import__(
        "app.feature_builder", fromlist=["build_features"]
    ).build_features("heart_disease", request, feature_cols)
    original_probability = FakePipeline().predict_proba(original_features)[0][1]
    top_factors = [
        {"feature": "bp_chol_both"},
        {"feature": "smoker"},
        {"feature": "physical_activity"},
        {"feature": "bmi"},
    ]

    recommendations = create_recommendations(
        "heart_disease",
        model_data,
        request,
        original_probability,
        top_factors,
    )

    assert 3 <= len(recommendations) <= 5
    reductions = [item["reduction_percentage_points"] for item in recommendations]
    assert reductions == sorted(reductions, reverse=True)
    assert all(item["new_probability"] < original_probability for item in recommendations)
    assert recommendations[0]["action"] == "manage_blood_pressure"
    assert recommendations[0]["changes"][0]["field"] == "high_bp"
    assert "bp_chol_both" in recommendations[0]["contributing_features"]


def test_non_contributing_and_non_improving_scenarios_are_omitted():
    request = heart_request()
    feature_cols = ["bmi", "smoker", "physical_activity", "high_bp", "high_chol"]
    model_data = {"pipeline": FakePipeline(), "feature_cols": feature_cols}
    original_features = __import__(
        "app.feature_builder", fromlist=["build_features"]
    ).build_features("heart_disease", request, feature_cols)
    original_probability = FakePipeline().predict_proba(original_features)[0][1]

    recommendations = create_recommendations(
        "heart_disease",
        model_data,
        request,
        original_probability,
        [{"feature": "high_chol"}],
    )

    # FakePipeline ignores high_chol, so this candidate produces no reduction.
    assert recommendations == []
