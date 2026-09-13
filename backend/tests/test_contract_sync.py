import json
from pathlib import Path

from app.model_registry import MODEL_CONFIG

PROJECT_DIR = Path(__file__).resolve().parents[2]

def test_registry_matches_model_selection_and_validation_thresholds():
    selection = json.loads(
        (PROJECT_DIR / "models" / "model_selection.json").read_text()
    )
    metrics = json.loads((PROJECT_DIR / "results" / "metrics.json").read_text())

    for disease, config in MODEL_CONFIG.items():
        selected_model = selection[disease]["model"]
        selected_threshold = metrics[disease]["models"][selected_model]["val"][
            "tuned_threshold"
        ]["threshold"]
        assert config["model_name"] == selected_model
        assert config["artifact"] == f"{disease}_{selected_model}.joblib"
        assert config["threshold"] == selected_threshold

def test_prediction_contract_matches_registry():
    contract = json.loads(
        (PROJECT_DIR / "contracts" / "prediction-contracts.json").read_text()
    )

    assert contract["contract_version"] == "1.1.0"

    for disease, config in MODEL_CONFIG.items():
        disease_contract = contract["diseases"][disease]
        assert disease_contract["route"] == f"/predict/{disease}"
        assert disease_contract["model"]["name"] == config["model_name"]
        assert disease_contract["model"]["threshold"] == config["threshold"]
        assert disease_contract["response_example"]["model"] == config["model_name"]
        assert disease_contract["response_example"]["threshold"] == config["threshold"]
