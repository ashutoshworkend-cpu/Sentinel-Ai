import numpy as np
import pytest

from app.services.face_recognition import (
    FaceNotVisibleError,
    FaceQualityError,
    FaceRecognitionService,
)


class EnrollmentStoreStub:
    encryption_available = True

    def active_embeddings(self) -> list[dict]:
        return []


class DetectorStub:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        del image
        return 1, self.faces


class RecognizerStub:
    def alignCrop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        del face
        return image

    def feature(self, aligned: np.ndarray) -> np.ndarray:
        del aligned
        return np.array([[3.0, 4.0]], dtype=np.float32)


def face_row(size: int = 100) -> np.ndarray:
    return np.array(
        [[20, 20, size, size, 30, 30, 80, 30, 55, 55, 35, 90, 75, 90, 0.99]],
        dtype=np.float32,
    )


def service(tmp_path) -> FaceRecognitionService:
    return FaceRecognitionService(
        store=EnrollmentStoreStub(),
        detector_path=tmp_path / "yunet.onnx",
        recognizer_path=tmp_path / "sface.onnx",
        match_threshold=0.45,
        ambiguity_margin=0.05,
        minimum_face_size=80,
        minimum_face_score=0.8,
    )


def test_good_quality_face_produces_normalized_embedding(tmp_path) -> None:
    recognition = service(tmp_path)
    recognition._detector = DetectorStub(face_row())
    recognition._recognizer = RecognizerStub()
    image = np.random.default_rng(42).integers(
        low=50,
        high=200,
        size=(160, 160, 3),
        dtype=np.uint8,
    )

    embedding = recognition.enrollment_embedding(image)
    assert embedding.tolist() == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(embedding) == pytest.approx(1.0)
    recognition.shutdown()


def test_small_blurred_and_missing_faces_are_rejected(tmp_path) -> None:
    recognition = service(tmp_path)
    recognition._recognizer = RecognizerStub()
    image = np.full((160, 160, 3), 120, dtype=np.uint8)

    recognition._detector = DetectorStub(face_row(size=40))
    with pytest.raises(FaceQualityError, match="too small"):
        recognition.enrollment_embedding(image)

    recognition._detector = DetectorStub(face_row())
    with pytest.raises(FaceQualityError, match="dark, overexposed, or blurred"):
        recognition.enrollment_embedding(image)

    recognition._detector = DetectorStub(None)
    with pytest.raises(FaceNotVisibleError):
        recognition.enrollment_embedding(image)
    recognition.shutdown()


def test_live_missing_face_maps_to_face_not_visible(tmp_path) -> None:
    recognition = service(tmp_path)
    recognition._detector = DetectorStub(None)
    recognition._recognizer = RecognizerStub()

    outcome = recognition._recognize_safely(
        np.random.default_rng(4).integers(
            low=50,
            high=200,
            size=(160, 160, 3),
            dtype=np.uint8,
        )
    )
    assert outcome.status == "FACE_NOT_VISIBLE"
    assert outcome.display_name is None
    recognition.shutdown()
