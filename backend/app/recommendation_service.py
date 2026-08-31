from dataclasses import dataclass
from typing import Any, Literal

from .feature_builder import build_features, calculate_bmi

ScenarioType = Literal["actionable", "health_management", "contextual"]

MIN_REDUCTION = 0.005
MAX_RECOMMENDATIONS = 5

GENERAL_HEALTH_IMPROVEMENT = {
    "poor": "fair",
    "fair": "good",
    "good": "very_good",
    "very_good": "excellent",
}

@dataclass(frozen=True)
class Candidate:
    action: str
    scenario_type: ScenarioType
    triggers: frozenset[str]
    updates: dict[str, Any]
    message_prefix: str
    units: dict[str, str]

def _reduced_health_days(current: int) -> int | None:
    if current >= 15:
        return current - 5
    if current >= 7:
        return current - 3
    if current >= 3:
        return current - 2
    return None

def _weight_target(height_cm: float, weight_kg: float) -> float | None:
    bmi = calculate_bmi(height_cm, weight_kg)
    if bmi >= 35:
        change = -5.0
    elif bmi >= 30:
        change = -3.0
    elif bmi >= 25:
        change = -2.0
    elif bmi < 18.5:
        change = 2.0
    else:
        return None
    target = round(weight_kg + change, 1)
    return target if 30 <= target <= 300 else None

def _candidate_interventions(disease: str, request: Any) -> list[Candidate]:
    candidates: list[Candidate] = []

    def add(
        action: str,
        scenario_type: ScenarioType,
        triggers: set[str],
        updates: dict[str, Any],
        message_prefix: str,
        units: dict[str, str] | None = None,
    ) -> None:
        candidates.append(
            Candidate(
                action=action,
                scenario_type=scenario_type,
                triggers=frozenset(triggers),
                updates=updates,
                message_prefix=message_prefix,
                units=units or {},
            )
        )

    weight_target = _weight_target(request.height_cm, request.weight_kg)
    if weight_target is not None:
        add(
            "modest_weight_change",
            "actionable",
            {"bmi", "bmi_obese", "age_bmi", "glucose_bmi", "metabolic_risk_score"},
            {"weight_kg": weight_target},
            f"With a modeled weight change from {request.weight_kg:g} kg to {weight_target:g} kg",
            {"weight_kg": "kg"},
        )

    if request.smoking_status == "current":
        updates: dict[str, Any] = {"smoking_status": "former"}
        if hasattr(request, "cigarettes_per_day"):
            updates["cigarettes_per_day"] = 0.0
        add(
            "stop_smoking",
            "actionable",
            {"smoker", "heavy_smoker", "cigs_per_day", "smoking_current", "lifestyle_score"},
            updates,
            "If you stopped smoking",
        )

        if disease == "hypertension" and request.cigarettes_per_day >= 5:
            reduced_cigarettes = round(request.cigarettes_per_day / 2, 1)
            add(
                "reduce_cigarettes",
                "actionable",
                {"cigs_per_day", "heavy_smoker"},
                {"cigarettes_per_day": reduced_cigarettes},
                ("With modeled cigarette use reduced from "
                 f"{request.cigarettes_per_day:g} to {reduced_cigarettes:g} per day"),
                {"cigarettes_per_day": "cigarettes/day"},
            )
    
    if disease in {"diabetes", "heart_disease"}:
        boolean_rules = (
            ("manage_blood_pressure", "health_management", "high_bp", False,
             {"high_bp", "bp_chol_both", "metabolic_risk_score"},
             "If the high-blood-pressure indicator changed to no"),
            ("manage_cholesterol", "health_management", "high_chol", False,
             {"high_chol", "bp_chol_both", "metabolic_risk_score"},
             "If the high-cholesterol indicator changed to no"),
            ("reduce_heavy_alcohol", "actionable", "heavy_alcohol", False,
             {"heavy_alcohol", "lifestyle_score"},
             "If heavy alcohol consumption were reduced"),
            ("increase_physical_activity", "actionable", "physical_activity", True,
             {"physical_activity", "lifestyle_score"},
             "If you became regularly physically active"),
            ("eat_fruit_regularly", "actionable", "fruits", True,
             {"fruits", "lifestyle_score"},
             "If you regularly ate fruit"),
            ("eat_vegetables_regularly", "actionable", "veggies", True,
             {"veggies", "lifestyle_score"},
             "If you regularly ate vegetables"),
        )
        for action, kind, field, target, triggers, wording in boolean_rules:
            if getattr(request, field) != target:
                add(action, kind, triggers, {field: target}, wording)

        improved_health = GENERAL_HEALTH_IMPROVEMENT.get(request.general_health)
        if improved_health:
            add(
                "improved_general_health",
                "contextual",
                {"general_health", "age_genhealth", "general_health_diffwalk"},
                {"general_health": improved_health},
                f"If your reported general health improved from {request.general_health.replace('_', ' ')} to {improved_health.replace('_', ' ')}",
            )

        for field, feature, label in (
            ("mental_health_days", "mental_health", "poor mental-health"),
            ("physical_health_days", "physical_health", "poor physical-health"),
        ):
            current = getattr(request, field)
            target = _reduced_health_days(current)
            if target is not None:
                reduction = current - target
                add(
                    f"fewer_{field}",
                    "contextual",
                    {feature, "poor_health_days"},
                    {field: target},
                    f"With {reduction} fewer {label} days per month",
                    {field: "days/month"},
                )
                
    if disease in {"hypertension", "stroke"} and request.glucose >= 126:
        target_glucose = max(110.0, request.glucose - 20.0)
        add(
            "manage_glucose",
            "health_management",
            {"glucose", "hyperglycemia", "glucose_bmi", "age_glucose"},
            {"glucose": target_glucose},
            "With a modeled reduction in measured glucose",
            {"glucose": "mg/dL"},
            )

    if disease == "hypertension":
        blood_pressure_updates: dict[str, Any] = {}
        if request.systolic_bp >= 130:
            blood_pressure_updates["systolic_bp"] = max(120.0, request.systolic_bp - 10.0)
        if request.diastolic_bp >= 80:
            blood_pressure_updates["diastolic_bp"] = max(80.0, request.diastolic_bp - 5.0)
        if "diastolic_bp" in blood_pressure_updates:
            resulting_systolic = blood_pressure_updates.get(
                "systolic_bp", request.systolic_bp
            )
            blood_pressure_updates["diastolic_bp"] = min(
                blood_pressure_updates["diastolic_bp"], resulting_systolic - 1.0
            )
        if blood_pressure_updates:
            add(
                "lower_blood_pressure",
                "health_management",
                {
                    "systolic_bp",
                    "diastolic_bp",
                    "high_bp",
                    "pulse_pressure",
                    "mean_arterial_pressure",
                    "uncontrolled_bp",
                },
                blood_pressure_updates,
                "With a modeled reduction in measured blood pressure",
                {
                    "systolic_bp": "mmHg",
                    "diastolic_bp": "mmHg",
                },
            )

        if request.cholesterol > 200:
            target_cholesterol = max(200.0, request.cholesterol - 20.0)
            add(
                "manage_cholesterol",
                "health_management",
                {"cholesterol", "high_chol"},
                {"cholesterol": target_cholesterol},
                "With a modeled reduction in measured cholesterol",
                {"cholesterol": "mg/dL"},
            )

        if request.heart_rate > 100:
            target_heart_rate = max(90.0, request.heart_rate - 10.0)
            add(
                "evaluate_high_heart_rate",
                "contextual",
                {"heart_rate", "tachycardia"},
                {"heart_rate": target_heart_rate},
                "With a modeled lower heart-rate measurement",
                {"heart_rate": "bpm"},
            )

    if disease == "stroke" and request.hypertension:
        add(
            "manage_hypertension",
            "health_management",
            {"hypertension"},
            {"hypertension": False},
            "If the hypertension indicator changed to no",
        )

    return candidates

def create_recommendations(
    disease: str,
    model_data: dict[str, Any],
    request: Any,
    original_probability: float,
    top_factors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    top_feature_names = {factor["feature"] for factor in top_factors}
    recommendations: list[dict[str, Any]] = []

    for candidate in _candidate_interventions(disease, request):
        contributing = sorted(candidate.triggers & top_feature_names)
        if not contributing:
            continue

        modified_request = request.model_copy(update=candidate.updates)
        features = build_features(disease, modified_request, model_data["feature_cols"])
        new_probability = float(model_data["pipeline"].predict_proba(features)[0, 1])
        reduction = original_probability - new_probability
        if reduction < MIN_REDUCTION:
            continue

        changes = [
            {
                "field": field,
                "from_value": getattr(request, field),
                "to_value": value,
                "unit": candidate.units.get(field),
            }
            for field, value in candidate.updates.items()
        ]
        recommendations.append(
            {
                "action": candidate.action,
                "scenario_type": candidate.scenario_type,
                "contributing_features": contributing,
                "changes": changes,
                "message": (
                    f"{candidate.message_prefix}, the model estimate would decrease "
                    f"from {original_probability * 100:.1f}% to {new_probability * 100:.1f}%."
                ),
                "new_probability": round(new_probability, 6),
                "reduction_percentage_points": round(reduction * 100, 2),
                "relative_reduction_percent": round(
                    reduction / original_probability * 100, 2
                ),
                "causal": False,
            }
        )

    recommendations.sort(
        key=lambda item: item["reduction_percentage_points"], reverse=True
    )
    return recommendations[:MAX_RECOMMENDATIONS]
