"""
Basic test suite for VeriFact Pro.

Run with:
    pytest test_app.py -v

Note: these tests assume a model has already been trained (model/model.pkl
exists) — run `python model/train_model.py` first if predict-related tests
are skipped/failing.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import app as application


@pytest.fixture
def client():
    application.app.config["TESTING"] = True
    application.app.config["WTF_CSRF_ENABLED"] = False
    with application.app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def clean_db():
    with application.app.app_context():
        application.db.drop_all()
        application.db.create_all()
    yield


class TestPublicPages:
    """Every public page should load without requiring login."""

    @pytest.mark.parametrize("route", [
        "/", "/detect", "/compare", "/about", "/analytics",
        "/quiz", "/faq", "/login", "/signup",
    ])
    def test_page_loads(self, client, route):
        resp = client.get(route)
        assert resp.status_code == 200


class TestHealthAndInfo:
    def test_health_check(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"

    def test_model_info(self, client):
        resp = client.get("/api/model-info")
        assert resp.status_code == 200
        assert "endpoints" in resp.get_json()


class TestAuth:
    def test_signup_creates_user(self, client):
        resp = client.post("/signup", data={
            "username": "testuser1", "email": "t1@example.com", "password": "password123"
        }, follow_redirects=True)
        assert resp.status_code == 200
        assert b"testuser1" in resp.data or resp.request.path == "/dashboard"

    def test_duplicate_signup_rejected(self, client):
        client.post("/signup", data={
            "username": "dupe", "email": "dupe@example.com", "password": "password123"
        })
        resp = client.post("/signup", data={
            "username": "dupe", "email": "dupe@example.com", "password": "password123"
        })
        assert b"already registered" in resp.data

    def test_login_wrong_password_rejected(self, client):
        client.post("/signup", data={
            "username": "loginuser", "email": "login@example.com", "password": "correctpw"
        })
        client.get("/logout")
        resp = client.post("/login", data={"username": "loginuser", "password": "wrongpw"})
        assert b"Invalid" in resp.data

    def test_dashboard_requires_login(self, client):
        resp = client.get("/dashboard", follow_redirects=True)
        assert b"log in" in resp.data.lower() or resp.request.path == "/login"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "model", "model.pkl")),
    reason="Model not trained yet — run model/train_model.py first"
)
class TestPrediction:
    def test_predict_rejects_short_text(self, client):
        resp = client.post("/api/predict", json={"text": "too short"})
        assert resp.status_code == 400

    def test_predict_returns_verdict(self, client):
        resp = client.post("/api/predict", json={
            "text": "Officials confirmed the new policy will take effect after "
                     "a review by the relevant department this week."
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["prediction"] in ("REAL", "FAKE")
        assert 0 <= data["confidence"] <= 100

    def test_compare_needs_both_texts(self, client):
        resp = client.post("/api/compare", json={"text_a": "short", "text_b": "also short"})
        assert resp.status_code == 400
