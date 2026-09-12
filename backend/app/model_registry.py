from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

MODEL_CONFIG = {
    "diabetes": {
        "artifact": "diabetes_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.21000000000000002,
    },
    "heart_disease": {
        "artifact": "heart_disease_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.17,
    },
    "hypertension": {
        "artifact": "hypertension_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.34,
    },
    "stroke": {
        "artifact": "stroke_logreg.joblib",
        "model_name": "logreg",
        "threshold": 0.19,
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
