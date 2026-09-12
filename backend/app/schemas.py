from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Sex = Literal["female", "male"]
Education = Literal[
    "never_attended_or_kindergarten",
    "elementary",
    "some_high_school",
    "high_school_graduate",
    "some_college_or_technical_school",
    "college_graduate",
]
Income = Literal[
    "less_than_10000",
    "10000_to_14999",
    "15000_to_19999",
    "20000_to_24999",
    "25000_to_34999",
    "35000_to_49999",
    "50000_to_74999",
    "75000_or_more",
]
GeneralHealth = Literal["excellent", "very_good", "good", "fair", "poor"]
SmokingStatus = Literal["never", "former", "current", "unknown"]

class StrictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        allow_inf_nan=False,
    )

class PatientProfileRequest(StrictRequest):
    sex: Sex
    age: int = Field(ge=18, le=120)
    height_cm: float = Field(ge=100, le=250)
    weight_kg: float = Field(ge=30, le=300)

class BrfssRequest(PatientProfileRequest):
    smoking_status: SmokingStatus
    heavy_alcohol: bool
    physical_activity: bool
    education: Education
    income: Income
    high_bp: bool
    high_chol: bool
    chol_check: bool
    fruits: bool
    veggies: bool
    general_health: GeneralHealth
    mental_health_days: int = Field(ge=0, le=30)
    physical_health_days: int = Field(ge=0, le=30)
    difficulty_walking: bool
    stroke: bool

class DiabetesRequest(BrfssRequest):
    heart_disease: bool

class HeartRequest(BrfssRequest):
    diabetes: bool

class HypertensionRequest(PatientProfileRequest):
    diabetes: bool
    smoking_status: SmokingStatus
    cigarettes_per_day: float = Field(ge=0, le=100)
    systolic_bp: float = Field(ge=70, le=260)
    diastolic_bp: float = Field(ge=40, le=160)
    heart_rate: float = Field(ge=30, le=220)
    bp_meds: bool
    cholesterol: float = Field(ge=80, le=700)
    glucose: float = Field(ge=30, le=700)

    @model_validator(mode="after")
    def validate_relationships(self) -> "HypertensionRequest":
        if self.diastolic_bp >= self.systolic_bp:
            raise ValueError("diastolic_bp must be lower than systolic_bp")

        if self.smoking_status == "current" and self.cigarettes_per_day <= 0:
            raise ValueError(
                "current smokers must provide cigarettes_per_day greater than 0"
            )

        if self.smoking_status != "current" and self.cigarettes_per_day != 0:
            raise ValueError(
                "cigarettes_per_day must be 0 for never or former smokers"
            )

        return self

class StrokeRequest(PatientProfileRequest):
    heart_disease: bool
    hypertension: bool
    smoking_status: SmokingStatus
    glucose: float = Field(ge=30, le=700)
    ever_married: bool
    work_type: Literal[
    "private",
    "self_employed",
    "government_job",
    "never_worked",
    ]
    residence_type: Literal["urban", "rural"]

class TopFactor(BaseModel):
    feature: str
    label: str
    value: Any
    direction: Literal["increases_estimate", "decreases_estimate"]
    importance: float = Field(ge=0)
    modifiable: bool

class RecommendationChange(BaseModel):
    field: str
    from_value: Any
    to_value: Any
    unit: str | None = None

class Recommendation(BaseModel):
    action: str
    scenario_type: Literal["actionable", "health_management", "contextual"]
    contributing_features: list[str]
    changes: list[RecommendationChange]
    message: str
    new_probability: float = Field(ge=0, le=1)
    reduction_percentage_points: float = Field(gt=0)
    relative_reduction_percent: float = Field(gt=0)
    causal: Literal[False] = False

class PredictionResponse(BaseModel):
    contract_version: Literal["1.1.0"]
    disease: Literal["diabetes", "heart_disease", "hypertension", "stroke"]
    model: Literal["logreg"]
    model_version: str
    probability: float = Field(ge=0, le=1)
    classification: Literal["elevated", "not_elevated"]
    threshold: float = Field(ge=0, le=1)
    top_factors: list[TopFactor]
    recommendations: list[Recommendation] = Field(max_length=5)
    explanation_units: Literal["probability", "raw_model_output", "log_odds"]
    explanation_note: str
    warnings: list[str]
    disclaimer: str
