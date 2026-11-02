from fastapi.testclient import TestClient

from app.main import app
from app.settings import settings


def configure_local_only_app(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "local-only.sqlite3")
    monkeypatch.setattr(settings, "biometric_key", "")
    monkeypatch.setattr(settings, "yunet_model_path", tmp_path / "missing-yunet.onnx")
    monkeypatch.setattr(settings, "sface_model_path", tmp_path / "missing-sface.onnx")


def test_local_dashboard_is_available_without_login(tmp_path, monkeypatch):
    configure_local_only_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/cameras").status_code == 200
        assert client.get("/api/people").status_code == 200
        assert client.post("/api/auth/login", json={"password": "unused"}).status_code == 404


def test_remote_clients_are_rejected(tmp_path, monkeypatch):
    configure_local_only_app(tmp_path, monkeypatch)
    with TestClient(app, client=("192.0.2.10", 50000)) as client:
        response = client.get("/api/cameras")
        assert response.status_code == 403
        assert response.json()["detail"] == (
            "This service is available only from this device."
        )
