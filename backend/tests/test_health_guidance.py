import json
from types import SimpleNamespace

from app import ai_explanation_service
from app.health_guidance import GUIDANCE_CATALOG, build_health_guidance
from app.schemas import GeneralHealthGuidance

def test_catalog_uses_only_official_who_https_sources():
    assert GUIDANCE_CATALOG
    for item in GUIDANCE_CATALOG.values():
        assert item["source_url"].startswith("https://www.who.int/")

def test_action_specific_guidance_is_prioritized_and_deduplicated():
    guidance = build_health_guidance(
        "hypertension",
        [
            {"action": "stop_smoking"},
            {"action": "lower_blood_pressure"},
            {"action": "manage_blood_pressure"},
        ],
    )
    item_ids = [item["id"] for item in guidance["items"]]

    assert item_ids[0] == "tobacco"
    assert "blood_pressure" in item_ids
    assert len(item_ids) == len(set(item_ids))
    assert len(item_ids) <= 4

def test_guidance_matches_response_schema():
    guidance = build_health_guidance("diabetes", [])
    validated = GeneralHealthGuidance.model_validate(guidance)

    assert validated.items
    assert validated.ai_generated is False
    assert all(item.source_organization == "WHO" for item in validated.items)

def test_ai_summary_falls_back_without_api_key(monkeypatch):
    monkeypatch.setattr(
        ai_explanation_service,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key=None, openai_model="unused"),
    )
    guidance = build_health_guidance("stroke", [])
    result = ai_explanation_service.add_ai_guidance_summary("stroke", guidance)
    assert result == guidance
    assert result["ai_generated"] is False

def test_safety_filter_rejects_medical_directives_and_numbers():
    assert ai_explanation_service._is_safe_guidance_summary(
        "Ky është informacion i përgjithshëm edukativ për zakone që mbështesin shëndetin."
    )
    assert not ai_explanation_service._is_safe_guidance_summary(
        "Merr ilaçin çdo ditë sipas kësaj përgjigjeje."
    )
    assert not ai_explanation_service._is_safe_guidance_summary(
        "Kryej 30 minuta aktivitet fizik."
    )

def test_ai_receives_only_approved_catalog_material(monkeypatch):
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self):
            return None
        def json(self):
            content = json.dumps(
                {
                    "summary": (
                        "Ky është informacion i përgjithshëm edukativ për zakone "
                        "që mbështesin shëndetin dhe mirëqenien."
                    )
                },
                ensure_ascii=False,
            )
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": content}],
                    }
                ]
            }
    class FakeClient:
        def __init__(self, **_kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return None
        def post(self, _url, *, headers, json):
            captured["headers"] = headers
            captured["payload"] = json
            return FakeResponse()

    monkeypatch.setattr(
        ai_explanation_service,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test-key", openai_model="test-model"),
    )
    monkeypatch.setattr(ai_explanation_service.httpx, "Client", FakeClient)
    guidance = build_health_guidance("diabetes", [{"action": "manage_glucose"}])

    result = ai_explanation_service.add_ai_guidance_summary("diabetes", guidance)
    sent_input = json.loads(captured["payload"]["input"])

    assert result["ai_generated"] is True
    assert captured["payload"]["store"] is False
    assert set(sent_input) == {"disease", "items"}
    assert all(set(item) == {"id", "title", "approved_text"} for item in sent_input["items"])
    assert "probability" not in captured["payload"]["input"]
    assert "measurement" not in captured["payload"]["input"]
