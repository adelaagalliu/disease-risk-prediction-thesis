from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_ORIGINAL = ROOT / "data" / "datasets_original"
DATA_CLEANED = ROOT / "data" / "datasets_cleaned"
DATA_PROCESSED = ROOT / "data" / "datasets_processed"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_SELECTION_FILE = MODELS_DIR / "model_selection.json"

# Stable duplicate-profile identifier saved with every processed split. It is
# metadata for grouped CV/bootstrap resampling and must never be a predictor.
GROUP_COL = "_group_id"

DATASETS = {
    "diabetes": {
        "csv": DATA_CLEANED / "diabetes_clean.csv",
        "target": "diabetes",
        "drop_cols": [
            "sex",
            "diabetes_severity",
        ],
    },
    "heart_disease": {
        "csv": DATA_CLEANED / "heart_clean.csv",
        "target": "heart_disease",
        "drop_cols": [
            "sex",             
            "diabetes_severity",
        ],
    },
    "hypertension": {
        "csv": DATA_CLEANED / "hypertension_clean.csv",
        "target": "hypertension",
        "drop_cols": [
            "sex",   
            "age", 
        ],
    },
    "stroke": {
        "csv": DATA_CLEANED / "stroke_clean.csv",
        "target": "stroke",
        "drop_cols": [
            "sex",             
            "age",             
            "smoking_status",  
            "smoker", 
            "work_type",               ],
    },
}

# Fixed category list for one-hot encoding Stroke's work_type (see
# preprocessing.add_engineered_features). Must be fixed rather than derived
# per-split: "never_worked" is only ~0.1% of the dataset (5 rows total) and
# can easily be entirely absent from val or test, which would otherwise
# produce a different set of dummy columns per split.
STROKE_WORK_TYPE_CATEGORIES = ["private", "self-employed", "govt_job", "never_worked"]

ORDINAL_COLS = ["age_range", "education", "income", "general_health"]

BINARY_COLS = [
    "sex_binary", "smoker", "heavy_alcohol", "physical_activity",
    "high_bp", "high_chol", "chol_check", "fruits", "veggies", "diff_walk",
    "bp_meds", "ever_married", "residence_type",
    "bmi_was_missing", "bp_meds_was_missing", "smoker_was_missing",
    "smoking_current", "smoking_former", "smoking_unknown",
    "work_type_private", "work_type_self-employed", "work_type_govt_job", "work_type_never_worked",
]

CONTINUOUS_COLS = [
    "bmi", "mental_health", "physical_health",
    "poor_health_days", "age_bmi", "general_health_diffwalk",
    "age_genhealth", "metabolic_risk_score", "lifestyle_score",
    "cigs_per_day", "systolic_bp", "diastolic_bp", "heart_rate",
    "cholesterol", "glucose", "pulse_pressure", "mean_arterial_pressure",
    "glucose_bmi", "age_glucose",
]

COMORBIDITY_COLS = ["diabetes", "heart_disease", "stroke", "hypertension"]

RANDOM_STATE = 44
TEST_SIZE = 0.2
VAL_SIZE = 0.1 

MODEL_NAMES = ["logreg", "random_forest", "xgboost", "lightgbm", "catboost"]

SEARCH_N_ITER = 10
SEARCH_CV_FOLDS = 3

SEARCH_SCORING = "average_precision"

THRESHOLD_METRIC = "f1"

USE_FEATURE_ENGINEERING = True
N_BOOTSTRAP = 1000          
BOOTSTRAP_ALPHA = 0.05  
CALIBRATION_BINS = 10   

SHAP_BACKGROUND_SIZE = 100   # reference/background sample used by the explainer (from train)
SHAP_EXPLAIN_SIZE = 200      # how many test rows to compute SHAP values for (kept small: SHAP is expensive)
TOP_N_LOCAL_EXAMPLES = 3     # individual waterfall (single-prediction) plots saved per model

def load_split(name: str, split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / f"{name}_{split}.csv")
