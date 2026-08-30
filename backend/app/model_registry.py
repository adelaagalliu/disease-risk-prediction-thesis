from pathlib import Path
import joblib

MODEL_DIR = Path(
    r"C:\Users\User\Documents\diploma\models"
)

MODEL_CONFIG = {
    "diabetes": {
        "artifact": "diabetes_catboost.joblib",
        "model_name": "catboost",
        "threshold": 0.63,
    },
    "heart_disease": {
        "artifact": "heart_disease_lightgbm.joblib",
        "model_name": "lightgbm",
        "threshold": 0.73,
    },
    "hypertension": {
        "artifact": "hypertension_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.55,
    },
    "stroke": {
        "artifact": "stroke_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.76,
    },
}

class ModelRegistry:
    def __init__(self):
        self.models = {}

    def load_all(self):
        for disease, config in MODEL_CONFIG.items():
            saved = joblib.load(MODEL_DIR / config["artifact"])

            self.models[disease] = {
                "pipeline": saved["pipeline"],
                "feature_cols": saved["feature_cols"],
                "model_name": config["model_name"],
                "threshold": config["threshold"],
            }

    def get(self, disease: str):
        return self.models[disease]

registry = ModelRegistry()
