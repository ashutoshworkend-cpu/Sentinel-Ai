from concurrent.futures import Future
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace

from app.main import app
from app.classification import TrackMemory
from app.services.biometric_store import BiometricStore
from app.services.face_recognition import FaceRecognitionService, RecognitionOutcome
from app.services.video_processor import VideoProcessor, parse_camera_source
from app.settings import settings
from app.schemas import Detection


def ignore_event(event: dict) -> None:
    del event


def local_access_headers(client: TestClient) -> dict[str, str]:
    del client
    return {}


def test_parse_camera_source_converts_webcam_index() -> None:
    assert parse_camera_source(" 0 ") == 0


def test_parse_camera_source_preserves_stream_url() -> None:
    source = "rtsp://camera.local/live"
    assert parse_camera_source(source) == source


def test_person_labels_show_verified_role_and_safe_pending_states() -> None:
    def detection(
        status: str,
        name: str | None = None,
        category: str | None = None,
    ) -> Detection:
        return Detection(
            class_name="person",
            predicted_class="person",
            temporal_candidate="person",
            final_classification="person",
            confidence=0.9,
            classification_confidence=0.9,
            temporal_support=1.0,
            verification_status="verified",
            recognition_status=status,
            recognized_name=name,
            recognized_category=category,
            box=[1, 1, 10, 10],
        )

    assert VideoProcessor._person_label(
        detection("VERIFIED_RESIDENT", "Avery", "resident")
    )[0] == "Avery | Resident"
    assert VideoProcessor._person_label(
        detection("VERIFIED_VISITOR", "Jordan", "visitor")
    )[0] == "Jordan | Visitor"
    assert VideoProcessor._person_label(
        detection("VERIFIED_STAFF", "Morgan", "authorized_staff")
    )[0] == "Morgan | Staff"
    assert VideoProcessor._person_label(detection("IDENTIFYING"))[0] == "Identifying..."
    assert VideoProcessor._person_label(detection("FACE_NOT_VISIBLE"))[0] == (
        "Person | Face Not Visible"
    )
    assert VideoProcessor._person_label(detection("UNKNOWN", "Unverified"))[0] == (
        "Unknown Person"
    )
    assert VideoProcessor._person_label(detection("VERIFIED_RESIDENT"))[0] == (
        "Unknown Person"
    )


def test_expired_track_drops_pending_and_cached_identity(tmp_path) -> None:
    recognition = FaceRecognitionService(
        store=BiometricStore(
            tmp_path / "people.sqlite3",
            Fernet.generate_key().decode("ascii"),
        ),
        detector_path=tmp_path / "yunet.onnx",
        recognizer_path=tmp_path / "sface.onnx",
        match_threshold=0.45,
        ambiguity_margin=0.05,
        minimum_face_size=80,
        minimum_face_score=0.8,
    )
    processor = VideoProcessor(
        camera_id="camera-1",
        name="Entrance",
        source="0",
        on_event=ignore_event,
        recognition_service=recognition,
    )
    memory = TrackMemory()
    memory.observe(
        frame_index=0,
        box=[0, 0, 20, 40],
        predicted_class="person",
        confidence=0.9,
        verification_status="verified",
    )
    processor._track_memories[14] = memory
    future: Future[RecognitionOutcome] = Future()
    processor._recognition_futures[14] = (
        0,
        datetime.now(timezone.utc),
        future,
        recognition.profile_revision,
    )
    processor._recognition_results[14] = (
        0,
        RecognitionOutcome(
            status="VERIFIED_RESIDENT",
            person_id=4,
            display_name="Avery",
            category="resident",
        ),
    )
    processor._recognition_last_attempt[14] = 0
    processor._recognition_last_recorded[14] = ("VERIFIED_RESIDENT", 4)

    processor._extract_detections(
        result=SimpleNamespace(boxes=None),
        frame=np.zeros((80, 80, 3), dtype=np.uint8),
        frame_index=processor._classifier.max_missing_frames + 2,
        verifier=None,
        verification_cache=OrderedDict(),
    )
    future.set_result(
        RecognitionOutcome(
            status="VERIFIED_RESIDENT",
            person_id=4,
            display_name="Avery",
            category="resident",
        )
    )
    processor._collect_recognition_results(processor._classifier.max_missing_frames + 2)

    assert 14 not in processor._track_memories
    assert 14 not in processor._recognition_futures
    assert 14 not in processor._recognition_results
    assert 14 not in processor._recognition_last_attempt
    assert 14 not in processor._recognition_last_recorded
    recognition.shutdown()


def test_profile_change_clears_cached_track_identity(tmp_path) -> None:
    recognition = FaceRecognitionService(
        store=BiometricStore(
            tmp_path / "people.sqlite3",
            Fernet.generate_key().decode("ascii"),
        ),
        detector_path=tmp_path / "yunet.onnx",
        recognizer_path=tmp_path / "sface.onnx",
        match_threshold=0.45,
        ambiguity_margin=0.05,
        minimum_face_size=80,
        minimum_face_score=0.8,
    )
    processor = VideoProcessor(
        camera_id="camera-1",
        name="Entrance",
        source="0",
        on_event=ignore_event,
        recognition_service=recognition,
    )
    processor._recognition_results[14] = (
        10,
        RecognitionOutcome(
            status="VERIFIED_RESIDENT",
            person_id=4,
            display_name="Casey",
            category="resident",
        ),
    )
    processor._recognition_last_attempt[14] = 10
    processor._recognition_last_recorded[14] = ("VERIFIED_RESIDENT", 4)
    recognition.invalidate_profiles()

    processor._refresh_recognition_revision()

    assert processor._recognition_results == {}
    assert processor._recognition_last_attempt == {}
    assert processor._recognition_last_recorded == {}
    recognition.shutdown()


def test_health_and_camera_list(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "sentinel.sqlite3")
    monkeypatch.setattr(settings, "biometric_key", "")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/cameras", headers=local_access_headers(client)).json() == []


def test_add_camera_rejects_empty_source(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_path", tmp_path / "sentinel.sqlite3")
    monkeypatch.setattr(settings, "biometric_key", "")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        response = client.post(
            "/api/cameras",
            headers=local_access_headers(client),
            json={"name": "Entry", "source": ""},
        )
    assert response.status_code == 422
