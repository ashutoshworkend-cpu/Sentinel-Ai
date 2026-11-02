import sqlite3
from datetime import datetime, timezone

import numpy as np
import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import Detection, PersonCreate
from app.services.biometric_store import (
    BiometricConsentRequiredError,
    BiometricStore,
)
from app.services.face_recognition import FaceRecognitionService
from app.services.people_store import PeopleStore
from app.settings import settings


DETECTION = {
    "class_name": "person",
    "predicted_class": "person",
    "temporal_candidate": "person",
    "final_classification": "person",
    "confidence": 0.91,
    "classification_confidence": 0.91,
    "temporal_support": 0.9,
    "verification_status": "verified",
    "box": [10, 20, 100, 200],
}


def event_payload() -> dict:
    return {
        "camera_id": "camera-1",
        "camera_name": "Front door",
        "timestamp": "2025-01-01T10:00:00Z",
        "detections": [DETECTION],
    }


def local_access_headers(client: TestClient) -> dict[str, str]:
    del client
    return {}


def configure_test_app(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "people.sqlite3")
    monkeypatch.setattr(settings, "biometric_key", "")
    monkeypatch.setattr(settings, "yunet_model_path", tmp_path / "missing-yunet.onnx")
    monkeypatch.setattr(settings, "sface_model_path", tmp_path / "missing-sface.onnx")


def test_people_and_review_event_are_persistent_and_manual(
    tmp_path, monkeypatch
) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        person_response = client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "  Alex  ", "category": "resident"},
        )
        assert person_response.status_code == 201
        person = person_response.json()
        assert person["display_name"] == "Alex"
        assert person["category"] == "resident"

        event_response = client.post(
            "/api/review-events",
            headers=auth,
            json=event_payload(),
        )
        assert event_response.status_code == 201
        review_event = event_response.json()
        assert review_event["reviewed"] is False
        assert review_event["person_id"] is None

        rejected = client.patch(
            f"/api/review-events/{review_event['id']}",
            headers=auth,
            json={"reviewed": False, "person_id": person["id"]},
        )
        assert rejected.status_code == 422

        reviewed = client.patch(
            f"/api/review-events/{review_event['id']}",
            headers=auth,
            json={"reviewed": True, "person_id": person["id"]},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["person_name"] == "Alex"
        assert reviewed.json()["person_role"] == "resident"

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        assert client.get("/api/people", headers=auth).json()[0]["display_name"] == "Alex"
        persisted_events = client.get("/api/review-events", headers=auth).json()
        assert persisted_events[0]["id"] == review_event["id"]
        assert persisted_events[0]["reviewed"] is True


def test_local_only_access_and_person_management(tmp_path, monkeypatch) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/people").status_code == 200
        assert client.get("/api/cameras").status_code == 200
        assert client.get("/api/cameras/missing/stream.mjpg").status_code == 404
        assert client.post(
            "/api/auth/login",
            json={"password": "wrong"},
        ).status_code == 404
        auth = local_access_headers(client)
        assert client.get(
            "/api/cameras/missing/stream.mjpg",
        ).status_code == 404

        created = client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "Morgan", "category": "authorized_staff"},
        )
        assert created.status_code == 201
        person_id = created.json()["id"]
        assert client.get(f"/api/people/{person_id}", headers=auth).status_code == 200

        updated = client.put(
            f"/api/people/{person_id}",
            headers=auth,
            json={
                "display_name": "Morgan Lee",
                "category": "visitor",
                "active": False,
            },
        )
        assert updated.status_code == 200
        assert updated.json()["display_name"] == "Morgan Lee"
        assert updated.json()["category"] == "visitor"
        assert updated.json()["active"] is False
        assert client.delete(f"/api/people/{person_id}", headers=auth).status_code == 204
        assert client.get(f"/api/people/{person_id}", headers=auth).status_code == 404
        assert client.delete("/api/auth/session", headers=auth).status_code == 404
        assert client.get("/api/people", headers=auth).status_code == 200
        assert client.get("/api/cameras/missing/stream.mjpg").status_code == 404


def test_withdrawing_consent_scrubs_history_and_invalidates_live_cache(
    tmp_path, monkeypatch
) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        person = client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "Riley", "category": "resident"},
        ).json()
        payload = event_payload()
        payload["detections"] = [
            {
                **DETECTION,
                "recognized_person_id": person["id"],
                "recognized_name": "Riley",
                "recognized_category": "resident",
                "recognition_status": "VERIFIED_RESIDENT",
                "recognition_similarity": 0.81,
            }
        ]
        assert (
            client.post("/api/review-events", headers=auth, json=payload).status_code
            == 201
        )
        client.app.state.biometric_store.record_recognition(
            camera_id="camera-1",
            tracking_id=14,
            timestamp=datetime.now(timezone.utc),
            status="VERIFIED_RESIDENT",
            person_id=person["id"],
            similarity=0.81,
        )
        recognition = client.app.state.face_recognition
        prior_revision = recognition.profile_revision

        response = client.post(
            f"/api/people/{person['id']}/consent/withdraw",
            headers=auth,
        )

        assert response.status_code == 200
        assert response.json()["active"] is False
        assert recognition.profile_revision == prior_revision + 1
        detection = (
            client.get("/api/review-events", headers=auth).json()[0]["detections"][0]
        )
        assert detection["recognized_name"] is None
        assert detection["recognition_status"] == "UNKNOWN"
        event = client.get("/api/recognition/events", headers=auth).json()[0]
        assert event["person_id"] is None
        assert event["status"] == "UNKNOWN"
        assert event["similarity"] is None


def test_person_validation_and_missing_records(
    tmp_path, monkeypatch
) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        assert client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "  ", "category": "resident"},
        ).status_code == 422
        assert client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "Alex", "category": "unknown"},
        ).status_code == 422

        event_response = client.post(
            "/api/review-events",
            headers=auth,
            json=event_payload(),
        )
        event_id = event_response.json()["id"]
        assert client.patch(
            f"/api/review-events/{event_id}",
            headers=auth,
            json={"reviewed": True, "person_id": 999},
        ).status_code == 404
        assert client.patch(
            "/api/review-events/not-a-real-event",
            headers=auth,
            json={"reviewed": True, "person_id": None},
        ).status_code == 404
        assert client.delete("/api/people/999", headers=auth).status_code == 404


def test_deleting_person_removes_association_but_keeps_review_event(
    tmp_path, monkeypatch
) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        person = client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "Sam", "category": "visitor"},
        ).json()
        event_id = client.post(
            "/api/review-events",
            headers=auth,
            json=event_payload(),
        ).json()["id"]
        client.patch(
            f"/api/review-events/{event_id}",
            headers=auth,
            json={"reviewed": True, "person_id": person["id"]},
        )

        assert client.delete(
            f"/api/people/{person['id']}",
            headers=auth,
        ).status_code == 204
        remaining_event = client.get("/api/review-events", headers=auth).json()[0]
        assert remaining_event["reviewed"] is True
        assert remaining_event["person_id"] is None
        assert remaining_event["person_name"] is None


def test_enrollment_route_fails_closed_when_model_files_missing(
    tmp_path, monkeypatch
) -> None:
    configure_test_app(tmp_path, monkeypatch)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        auth = local_access_headers(client)
        person = client.post(
            "/api/people",
            headers=auth,
            json={"display_name": "Alex", "category": "resident"},
        ).json()
        response = client.post(
            f"/api/people/{person['id']}/enroll",
            headers=auth,
            data={"consent_confirmed": "true"},
            files=[
                ("photos", ("face1.jpg", b"one", "image/jpeg")),
                ("photos", ("face2.jpg", b"two", "image/jpeg")),
            ],
        )
        assert response.status_code == 503
        assert response.json()["detail"].startswith("Face recognition is not configured")
        status = client.get("/api/recognition/status", headers=auth)
        assert status.json() == {"configured": False, "status": "encryption_key_missing"}


def test_encrypted_enrollments_and_consent_withdrawal(tmp_path) -> None:
    database_path = tmp_path / "biometrics.sqlite3"
    people_store = PeopleStore(database_path)
    people_store.initialize()
    key = Fernet.generate_key().decode("ascii")
    biometric_store = BiometricStore(database_path, key)
    biometric_store.initialize()
    person = people_store.create_person(
        PersonCreate(display_name="Jordan", category="resident")
    )
    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.99, 0.1, 0.0], dtype=np.float32),
    ]

    with pytest.raises(BiometricConsentRequiredError):
        biometric_store.replace_embeddings(person["id"], embeddings, False)

    biometric_store.replace_embeddings(person["id"], embeddings, True)
    assert len(biometric_store.active_embeddings()) == 2
    with sqlite3.connect(database_path) as connection:
        encrypted = connection.execute(
            "SELECT encrypted_embedding FROM person_embeddings"
        ).fetchall()
    assert all(bytes(row[0]) != embeddings[0].tobytes() for row in encrypted)

    assert biometric_store.withdraw_consent(person["id"])
    assert biometric_store.active_embeddings() == []
    updated = people_store.get_person(person["id"])
    assert updated is not None
    assert updated["active"] is False
    assert updated["consented_at"] is None
    assert updated["enrollment_count"] == 0


def test_legacy_people_database_migrates_without_losing_records(tmp_path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE people (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO people (display_name, role, created_at) VALUES (?, ?, ?)",
            ("Taylor", "visitor", "2025-01-01T10:00:00+00:00"),
        )
    people_store = PeopleStore(database_path)
    people_store.initialize()
    biometric_store = BiometricStore(
        database_path,
        Fernet.generate_key().decode("ascii"),
    )
    biometric_store.initialize()

    people = people_store.list_people()
    assert len(people) == 1
    assert people[0]["display_name"] == "Taylor"
    assert people[0]["category"] == "visitor"
    assert people[0]["enrollment_count"] == 0


def test_deleting_profile_cascades_to_encrypted_samples(tmp_path) -> None:
    database_path = tmp_path / "delete-profile.sqlite3"
    people_store = PeopleStore(database_path)
    people_store.initialize()
    biometric_store = BiometricStore(
        database_path,
        Fernet.generate_key().decode("ascii"),
    )
    biometric_store.initialize()
    person = people_store.create_person(
        PersonCreate(display_name="Jamie", category="resident")
    )
    biometric_store.replace_embeddings(
        person["id"],
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.9, 0.1], dtype=np.float32),
        ],
        True,
    )

    assert people_store.delete_person(person["id"])
    with sqlite3.connect(database_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM person_embeddings"
        ).fetchone()[0]
    assert count == 0


def test_deleting_profile_scrubs_saved_recognition_identity(tmp_path) -> None:
    database_path = tmp_path / "scrub-profile.sqlite3"
    people_store = PeopleStore(database_path)
    people_store.initialize()
    biometric_store = BiometricStore(
        database_path,
        Fernet.generate_key().decode("ascii"),
    )
    biometric_store.initialize()
    person = people_store.create_person(
        PersonCreate(display_name="Jordan", category="resident")
    )
    detection = {
        **DETECTION,
        "recognized_person_id": person["id"],
        "recognized_name": "Jordan",
        "recognized_category": "resident",
        "recognition_status": "VERIFIED_RESIDENT",
        "recognition_similarity": 0.82,
    }
    people_store.create_review_event(
        camera_id="camera-1",
        camera_name="Entrance",
        timestamp=datetime.now(timezone.utc),
        detections=[
            Detection.model_validate(detection),
        ],
    )
    biometric_store.record_recognition(
        camera_id="camera-1",
        tracking_id=14,
        timestamp=datetime.now(timezone.utc),
        status="VERIFIED_RESIDENT",
        person_id=person["id"],
        similarity=0.82,
    )

    assert people_store.delete_person(person["id"])

    saved_detection = people_store.list_review_events()[0]["detections"][0]
    assert saved_detection["recognized_person_id"] is None
    assert saved_detection["recognized_name"] is None
    assert saved_detection["recognition_status"] == "UNKNOWN"
    event = biometric_store.list_recognition_events()[0]
    assert event["person_id"] is None
    assert event["person_name"] is None
    assert event["status"] == "UNKNOWN"
    assert event["similarity"] is None


class EnrollmentStoreStub:
    def __init__(self, records: list[dict]) -> None:
        self.records = records

    def active_embeddings(self) -> list[dict]:
        return self.records


def test_face_matching_returns_unknown_and_rejects_ambiguous_identity(tmp_path) -> None:
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    recognizer = FaceRecognitionService(
        store=EnrollmentStoreStub(
            [
                {
                    "person_id": 1,
                    "display_name": "Ari",
                    "category": "resident",
                    "embedding": embedding,
                },
                {
                    "person_id": 2,
                    "display_name": "Bea",
                    "category": "visitor",
                    "embedding": np.array([0.998, 0.063], dtype=np.float32),
                },
            ]
        ),
        detector_path=tmp_path / "yunet.onnx",
        recognizer_path=tmp_path / "sface.onnx",
        match_threshold=0.45,
        ambiguity_margin=0.05,
        minimum_face_size=80,
        minimum_face_score=0.8,
    )

    ambiguous = recognizer._match(embedding)
    assert ambiguous.status == "UNVERIFIED"
    assert ambiguous.display_name is None

    recognizer._store = EnrollmentStoreStub(
        [
            {
                "person_id": 1,
                "display_name": "Ari",
                "category": "resident",
                "embedding": np.array([0.0, 1.0], dtype=np.float32),
            }
        ]
    )
    unknown = recognizer._match(embedding)
    assert unknown.status == "UNKNOWN"
    assert unknown.display_name is None
    recognizer.shutdown()
