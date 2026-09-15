import json
import logging
from typing import Any

import httpx

from .config import get_settings

logger = logging.getLogger(__name__)

DISEASE_LABELS = {
    "diabetes": "diabetin",
    "heart_disease": "sëmundjen e zemrës",
    "hypertension": "hipertensionin",
    "stroke": "insultin cerebral",
}


def _fallback_text(disease: str, scenario_type: str) -> str:
    label = DISEASE_LABELS.get(disease, "këtë rezultat")
    if scenario_type == "actionable":
        return (
            f"Ky simulim tregon si ndryshon vlerësimi për {label} kur modifikohet "
            "vetëm faktori i përshkruar dhe faktorët e tjerë mbahen të pandryshuar."
        )
    if scenario_type == "health_management":
        return (
            f"Ky është një skenar statistikor për {label}, i bazuar në një vlerë të "
            "modeluar shëndetësore; ai nuk zëvendëson matjen ose këshillën klinike."
        )
    return (
        f"Ky skenar kontekstual tregon sjelljen e modelit për {label}; ai nuk provon "
        "se ndryshimi i përshkruar do të shkaktojë uljen e vlerësimit."
    )


def _response_text(payload: dict[str, Any]) -> str:
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return content.get("text", "")
    raise ValueError("OpenAI response did not contain output text")


def add_ai_explanations(
    disease: str,
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add short language explanations without changing deterministic values."""
    enriched = [
        {
            **item,
            "ai_explanation": _fallback_text(disease, item["scenario_type"]),
            "ai_generated": False,
        }
        for item in recommendations
    ]
    settings = get_settings()
    if not enriched or not settings.openai_api_key:
        return enriched

    scenarios = [
        {
            "id": index,
            "disease": disease,
            "action": item["action"],
            "scenario_type": item["scenario_type"],
            "message": item["message"],
        }
        for index, item in enumerate(enriched)
    ]
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["id", "explanation"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    request_payload = {
        "model": settings.openai_model,
        "store": False,
        "max_output_tokens": 600,
        "instructions": (
            "Shkruaj në shqip standard një shpjegim të qartë me 1-2 fjali për secilin "
            "skenar statistikor. Mos ndrysho dhe mos përsërit shifrat. Mos jep diagnozë, "
            "urdhër mjekësor, garanci ose pretendim shkakësor. Shpjego vetëm se rezultati "
            "tregon sjelljen e modelit kur një faktor ndryshohet dhe të tjerët mbahen konstantë."
        ),
        "input": json.dumps(scenarios, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "scenario_explanations",
                "strict": True,
                "schema": schema,
            }
        },
    }

    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
            response.raise_for_status()
        generated = json.loads(_response_text(response.json()))["items"]
        by_id = {
            item["id"]: item["explanation"].strip()[:500]
            for item in generated
            if isinstance(item.get("id"), int) and item.get("explanation", "").strip()
        }
        for index, item in enumerate(enriched):
            if index in by_id:
                item["ai_explanation"] = by_id[index]
                item["ai_generated"] = True
    except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        logger.exception("AI explanation failed; deterministic fallback will be used")

    return enriched
