from fastapi.testclient import TestClient

from app.api import create_app


def _register(client: TestClient, username: str) -> dict:
    response = client.post("/api/v1/auth/register", json={"username": username, "password": "password123", "display_name": username})
    assert response.status_code == 200, response.text
    return response.json()


def test_profile_defaults_and_user_isolation(application):
    with TestClient(create_app(application)) as client_a, TestClient(create_app(application)) as client_b:
        user_a = _register(client_a, "profile-a")
        user_b = _register(client_b, "profile-b")
        assert client_a.get("/api/v1/users/me/profile").json() == {
            "user_id": user_a["id"],
            "language": "zh-CN",
            "response_style": "balanced",
            "measurement_system": "metric",
            "preferred_output_format": None,
            "updated_at": client_a.get("/api/v1/users/me/profile").json()["updated_at"],
        }
        updated = client_a.patch("/api/v1/users/me/profile", json={"response_style": "concise", "measurement_system": "imperial"})
        assert updated.status_code == 200
        assert updated.json()["user_id"] == user_a["id"]
        assert updated.json()["response_style"] == "concise"
        assert client_b.get("/api/v1/users/me/profile").json()["user_id"] == user_b["id"]
        assert client_b.get("/api/v1/users/me/profile").json()["response_style"] == "balanced"


def test_profile_api_rejects_user_id_and_validates_formats(application):
    with TestClient(create_app(application)) as client:
        _register(client, "profile-validation")
        assert client.patch("/api/v1/users/me/profile", json={"user_id": "other"}).status_code == 422
        assert client.patch("/api/v1/users/me/profile", json={"preferred_output_format": "PDF"}).status_code == 400


def test_only_explicit_durable_preference_updates_profile(application):
    with TestClient(create_app(application)) as client:
        _register(client, "profile-extractor")
        first = client.post("/api/v1/ask", json={"message": "这次详细解释"})
        assert first.status_code == 200, first.text
        assert client.get("/api/v1/users/me/profile").json()["response_style"] == "balanced"
        durable = client.post("/api/v1/ask", json={"message": "以后都详细解释"})
        assert durable.status_code == 200, durable.text
        assert client.get("/api/v1/users/me/profile").json()["response_style"] == "detailed"
        one_off = client.post("/api/v1/ask", json={"message": "我今天想用 GeoJSON"})
        assert one_off.status_code == 200, one_off.text
        assert client.get("/api/v1/users/me/profile").json()["preferred_output_format"] is None
        default_format = client.post("/api/v1/ask", json={"message": "以后默认输出 GeoJSON"})
        assert default_format.status_code == 200, default_format.text
        assert client.get("/api/v1/users/me/profile").json()["preferred_output_format"] == "GeoJSON"
