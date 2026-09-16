from typing import Any


SOURCE_ORGANIZATION = "WHO"
SOURCE_REVIEW_DATE = "2026-09-16"
GUIDANCE_DISCLAIMER = (
    "Ky informacion është i përgjithshëm dhe edukativ. Nuk përbën diagnozë, "
    "rekomandim individual ose plan trajtimi dhe nuk zëvendëson konsultën me "
    "një profesionist të kualifikuar të kujdesit shëndetësor."
)


GUIDANCE_CATALOG: dict[str, dict[str, str]] = {
    "healthy_diet": {
        "title": "Ushqyerja e ekuilibruar",
        "text": (
            "Një mënyrë ushqyerjeje e larmishme dhe e ekuilibruar, me më pak "
            "kripë, sheqerna të lira dhe yndyrna të ngopura, mbështet shëndetin "
            "kardiometabolik."
        ),
        "source_title": "Healthy diet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
    },
    "physical_activity": {
        "title": "Aktiviteti fizik",
        "text": (
            "Aktiviteti fizik i rregullt, i përshtatur me gjendjen dhe mundësitë "
            "individuale, ndihmon në uljen e faktorëve të rrezikut për sëmundjet "
            "jo të transmetueshme."
        ),
        "source_title": "Physical activity",
        "source_url": (
            "https://www.who.int/health-topics/noncommunicable-diseases/physical-activity"
        ),
    },
    "tobacco": {
        "title": "Ndërprerja e duhanit",
        "text": (
            "Të gjitha format e duhanit janë të dëmshme. Mbështetja profesionale "
            "mund ta bëjë më të menaxhueshëm procesin e ndërprerjes."
        ),
        "source_title": "Tobacco and nicotine",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/tobacco",
    },
    "alcohol": {
        "title": "Kufizimi i alkoolit",
        "text": (
            "Reduktimi i konsumit të alkoolit zvogëlon ekspozimin ndaj një faktori "
            "që lidhet me shumë sëmundje dhe dëmtime."
        ),
        "source_title": "Alcohol",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/alcohol",
    },
    "weight_management": {
        "title": "Menaxhimi i peshës",
        "text": (
            "Menaxhimi gradual i peshës mbështetet te ushqyerja, aktiviteti fizik "
            "dhe, kur nevojitet, vlerësimi nga një profesionist shëndetësor."
        ),
        "source_title": "Obesity and overweight",
        "source_url": (
            "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight"
        ),
    },
    "diabetes_management": {
        "title": "Parandalimi dhe menaxhimi i diabetit",
        "text": (
            "Ushqyerja e shëndetshme, aktiviteti fizik, shmangia e duhanit dhe "
            "kontrollet e rekomanduara mund të ndihmojnë në parandalimin ose "
            "menaxhimin e diabetit të tipit 2."
        ),
        "source_title": "Diabetes",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    },
    "blood_pressure": {
        "title": "Monitorimi i tensionit arterial",
        "text": (
            "Tensioni i lartë identifikohet me matje. Vlerat e larta ose të "
            "përsëritura duhet të vlerësohen nga një profesionist shëndetësor."
        ),
        "source_title": "Hypertension",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    },
    "cardiovascular_health": {
        "title": "Shëndeti kardiovaskular",
        "text": (
            "Kontrolli i tensionit, glukozës dhe lipideve, së bashku me zakone të "
            "shëndetshme, është pjesë e parandalimit të sëmundjeve kardiovaskulare."
        ),
        "source_title": "Cardiovascular diseases (CVDs)",
        "source_url": (
            "https://www.who.int/news-room/fact-sheets/detail/"
            "cardiovascular-diseases-%28cvds%29"
        ),
    },
    "stroke_prevention": {
        "title": "Parandalimi i insultit cerebral",
        "text": (
            "Menaxhimi i tensionit dhe i faktorëve të tjerë kardiometabolikë, "
            "shmangia e duhanit dhe aktiviteti fizik ndihmojnë në uljen e rrezikut "
            "të insultit cerebral."
        ),
        "source_title": "Stroke",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/stroke",
    },
}


ACTION_GUIDANCE = {
    "modest_weight_change": ("weight_management", "healthy_diet", "physical_activity"),
    "stop_smoking": ("tobacco",),
    "reduce_cigarettes": ("tobacco",),
    "manage_blood_pressure": ("blood_pressure", "cardiovascular_health"),
    "lower_blood_pressure": ("blood_pressure", "cardiovascular_health"),
    "manage_hypertension": ("blood_pressure", "stroke_prevention"),
    "manage_cholesterol": ("cardiovascular_health", "healthy_diet"),
    "manage_glucose": ("diabetes_management", "healthy_diet"),
    "reduce_heavy_alcohol": ("alcohol",),
    "increase_physical_activity": ("physical_activity",),
    "eat_fruit_regularly": ("healthy_diet",),
    "eat_vegetables_regularly": ("healthy_diet",),
}


DISEASE_DEFAULTS = {
    "diabetes": ("diabetes_management", "healthy_diet", "physical_activity"),
    "heart_disease": (
        "cardiovascular_health",
        "healthy_diet",
        "physical_activity",
    ),
    "hypertension": ("blood_pressure", "healthy_diet", "physical_activity"),
    "stroke": ("stroke_prevention", "blood_pressure", "physical_activity"),
}


FALLBACK_SUMMARIES = {
    "diabetes": (
        "Këto udhëzime të përgjithshme përqendrohen te zakonet dhe kontrollet "
        "që mbështesin parandalimin dhe menaxhimin e faktorëve të diabetit."
    ),
    "heart_disease": (
        "Këto udhëzime të përgjithshme përqendrohen te faktorët e modifikueshëm "
        "që lidhen me shëndetin kardiovaskular."
    ),
    "hypertension": (
        "Këto udhëzime të përgjithshme përqendrohen te monitorimi dhe zakonet "
        "që mbështesin kontrollin e tensionit arterial."
    ),
    "stroke": (
        "Këto udhëzime të përgjithshme përqendrohen te faktorët që lidhen me "
        "parandalimin e insultit cerebral."
    ),
}


def _catalog_item(item_id: str) -> dict[str, str]:
    item = GUIDANCE_CATALOG[item_id]
    return {
        "id": item_id,
        **item,
        "source_organization": SOURCE_ORGANIZATION,
        "reviewed_at": SOURCE_REVIEW_DATE,
    }


def build_health_guidance(
    disease: str,
    recommendations: list[dict[str, Any]],
    max_items: int = 4,
) -> dict[str, Any]:
    """Build deterministic, source-backed guidance without interpreting patient data."""
    selected_ids: list[str] = []

    for recommendation in recommendations:
        for item_id in ACTION_GUIDANCE.get(recommendation.get("action", ""), ()):
            if item_id not in selected_ids:
                selected_ids.append(item_id)

    for item_id in DISEASE_DEFAULTS.get(disease, ()):
        if item_id not in selected_ids:
            selected_ids.append(item_id)

    return {
        "summary": FALLBACK_SUMMARIES.get(
            disease,
            "Këto janë udhëzime të përgjithshme të bazuara në burime shëndetësore.",
        ),
        "ai_generated": False,
        "items": [_catalog_item(item_id) for item_id in selected_ids[:max_items]],
        "disclaimer": GUIDANCE_DISCLAIMER,
    }
