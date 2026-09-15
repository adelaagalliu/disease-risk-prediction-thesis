from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app import main
from app.database import Base, get_db
from app.models import User
from app.persistence_service import save_prediction
from app.schemas import HypertensionRequest

@pytest.fixture
def client(monkeypatch) -> Generator[tuple[TestClient, sessionmaker], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    Base.metadata.create_all(engine)

    def override_db():
        with testing_session() as session:
            yield session

    main.app.dependency_overrides[get_db] = override_db
    monkeypatch.setattr(main.registry, "load_all", lambda: None)
    with TestClient(main.app) as test_client:
        yield test_client, testing_session
    main.app.dependency_overrides.clear()

def register_and_login(client: TestClient, email: str = "person@example.com") -> str:
    registration = client.post(
        "/auth/register",
        json={"email": email, "password": "Adela123?"},
    )
    assert registration.status_code == 201
    login = client.post(
        "/auth/login",
        json={"email": email, "password": "Adela123?"},
    )
    assert login.status_code == 200
    return login.json()["access_token"]

def test_registration_hashes_password_and_rejects_duplicate(client):
    test_client, testing_session = client
    register_and_login(test_client, "  Person@Example.com ")

    with testing_session() as db:
        user = db.scalar(select(User))
        assert user.email == "person@example.com"
        assert user.password_hash != "Adela123?"
        assert user.password_hash.startswith("$argon2")

    duplicate = test_client.post(
        "/auth/register",
        json={"email": "PERSON@example.com", "password": "anotherlongpassword1?"},
    )
    assert duplicate.status_code == 409

def test_login_and_me_require_valid_credentials(client):
    test_client, _ = client
    token = register_and_login(test_client)

    unauthenticated = test_client.get("/users/me")
    authenticated = test_client.get(
        "/users/me", headers={"Authorization": f"Bearer {token}"}
    )
    wrong_password = test_client.post(
        "/auth/login",
        json={"email": "person@example.com", "password": "thisisnotcorrect1?"},
    )

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.json()["email"] == "person@example.com"
    assert wrong_password.status_code == 401

def test_prediction_and_metric_history_are_scoped_to_current_user(client):
    test_client, testing_session = client
    first_token = register_and_login(test_client, "first@example.com")
    second_token = register_and_login(test_client, "second@example.com")

    request = HypertensionRequest(
        diabetes=False,
        sex="female",
        age=45,
        height_cm=165,
        weight_kg=68,
        smoking_status="never",
        cigarettes_per_day=0,
        systolic_bp=128,
        diastolic_bp=78,
        heart_rate=72,
        bp_meds=False,
        cholesterol=190,
        glucose=95,
    )
    response_payload = {
        "contract_version": "1.1.0",
        "disease": "hypertension",
        "model": "logreg",
        "model_version": "1",
        "probability": 0.25,
        "classification": "not_elevated",
        "threshold": 0.34,
        "top_factors": [],
        "recommendations": [],
        "explanation_units": "log_odds",
        "explanation_note": "Model influence, not causality.",
        "warnings": [],
        "disclaimer": "Research estimate, not a medical diagnosis.",
    }
    with testing_session() as db:
        first_user = db.scalar(select(User).where(User.email == "first@example.com"))
        save_prediction(
            db,
            first_user,
            "hypertension",
            request,
            response_payload,
        )

    first_history = test_client.get(
        "/history/predictions",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    first_bmi = test_client.get(
        "/history/trends/bmi",
        headers={"Authorization": f"Bearer {first_token}"},
    )
    second_history = test_client.get(
        "/history/predictions",
        headers={"Authorization": f"Bearer {second_token}"},
    )

    assert first_history.status_code == 200
    assert first_history.json()[0]["disease"] == "hypertension"
    assert first_history.json()[0]["inputs"]["weight_kg"] == 68.0
    assert first_bmi.status_code == 200
    assert first_bmi.json()[0]["value"] == pytest.approx(24.98)
    assert second_history.json() == []

def test_saved_result_detail_is_owned_and_complete(client):
    import uuid
    from app.models import Prediction

    test_client, testing_session = client
    first_token = register_and_login(test_client, "detail-first@example.com")
    second_token = register_and_login(test_client, "detail-second@example.com")
    request = HypertensionRequest(
        diabetes=False, sex="female", age=45, height_cm=165, weight_kg=68,
        smoking_status="never", cigarettes_per_day=0, systolic_bp=128,
        diastolic_bp=78, heart_rate=72, bp_meds=False, cholesterol=190, glucose=95,
    )
    payload = {
        "contract_version":"1.1.0", "disease":"hypertension", "model":"logreg",
        "model_version":"1", "probability":0.25, "classification":"not_elevated",
        "threshold":0.34, "top_factors":[], "recommendations":[],
        "explanation_units":"log_odds", "explanation_note":"Saved explanation",
        "warnings":[], "disclaimer":"Research only",
    }
    with testing_session() as db:
        user = db.scalar(select(User).where(User.email == "detail-first@example.com"))
        prediction = save_prediction(db, user, "hypertension", request, payload)
        prediction_id = str(prediction.id)
    endpoint = f"/history/predictions/{prediction_id}"
    headers = {"Authorization":f"Bearer {first_token}"}
    response = test_client.get(endpoint, headers=headers)
    assert response.status_code == 200
    assert response.json()["prediction_id"] == prediction_id
    assert response.json()["explanation_note"] == "Saved explanation"
    assert response.json()["created_at"]
    assert test_client.get(endpoint).status_code == 401
    assert test_client.get(endpoint, headers={"Authorization":f"Bearer {second_token}"}).status_code == 404
    assert test_client.get(f"/history/predictions/{uuid.uuid4()}", headers=headers).status_code == 404
    with testing_session() as db:
        assert len(db.scalars(select(Prediction)).all()) == 1
