from app.recommendation_service import _candidate_interventions
from app.schemas import DiabetesRequest, HeartRequest, HypertensionRequest, StrokeRequest

def brfss_data():
    return {
        "stroke": False,
        "sex": "female",
        "age": 52,
        "height_cm": 170.0,
        "weight_kg": 100.0,
        "smoking_status": "current",
        "heavy_alcohol": True,
        "physical_activity": False,
        "education": "college_graduate",
        "income": "50000_to_74999",
        "high_bp": True,
        "high_chol": True,
        "chol_check": True,
        "fruits": False,
        "veggies": False,
        "general_health": "fair",
        "mental_health_days": 10,
        "physical_health_days": 5,
        "difficulty_walking": False,
    }

def actions(disease, request):
    return {
        item.action
        for item in _candidate_interventions(disease, request)
    }

def test_diabetes():
    request = DiabetesRequest(
        heart_disease=False,
        **brfss_data(),
    )
    result = actions("diabetes", request)
    assert "stop_smoking" in result
    assert "manage_blood_pressure" in result
    assert "increase_physical_activity" in result

def test_heart():
    request = HeartRequest(
        diabetes=False,
        **brfss_data(),
    )
    result = actions("heart_disease", request)
    assert "stop_smoking" in result
    assert "manage_cholesterol" in result
    assert "eat_vegetables_regularly" in result

def test_hypertension():
    request = HypertensionRequest(
        diabetes=False,
        sex="male",
        age=55,
        height_cm=175.0,
        weight_kg=90.0,
        smoking_status="current",
        cigarettes_per_day=20.0,
        systolic_bp=148.0,
        diastolic_bp=92.0,
        heart_rate=108.0,
        bp_meds=False,
        cholesterol=235.0,
        glucose=145.0,
    )
    result = actions("hypertension", request)
    assert "lower_blood_pressure" in result
    assert "manage_glucose" in result
    assert "reduce_cigarettes" in result


def test_stroke():
    request = StrokeRequest(
        heart_disease=False,
        hypertension=True,
        sex="female",
        age=60,
        height_cm=165.0,
        weight_kg=65.0,
        smoking_status="never",
        glucose=150.0,
        ever_married=True,
        work_type="private",
        residence_type="urban",
    )
    result = actions("stroke", request)
    assert result == {
        "manage_hypertension",
        "manage_glucose",
    }