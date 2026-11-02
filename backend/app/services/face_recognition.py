import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.schemas import RecognitionState
from app.services.biometric_store import BiometricStore

logger = logging.getLogger(__name__)


class FaceQualityError(ValueError):
    pass


class FaceNotVisibleError(FaceQualityError):
    pass


class FaceModelUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecognitionOutcome:
    status: RecognitionState
    person_id: int | None = None
    display_name: str | None = None
    category: str | None = None
    similarity: float | None = None


class FaceRecognitionService:
    def __init__(
        self,
        store: BiometricStore,
        detector_path: Path,
        recognizer_path: Path,
        match_threshold: float,
        ambiguity_margin: float,
        minimum_face_size: int,
        minimum_face_score: float,
        enabled: bool = True,
        device: str = "cpu",
    ) -> None:
        self._store = store
        self._detector_path = detector_path
        self._recognizer_path = recognizer_path
        self._match_threshold = match_threshold
        self._ambiguity_margin = ambiguity_margin
        self._minimum_face_size = minimum_face_size
        self._minimum_face_score = minimum_face_score
        self._enabled = enabled
        self._device = device
        self._model_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._pending_count = 0
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="face-recognition",
        )
        self._detector = None
        self._recognizer = None
        self._load_error = False
        self._profile_revision = 0

    @property
    def profile_revision(self) -> int:
        return self._profile_revision

    def invalidate_profiles(self) -> None:
        self._profile_revision += 1

    def status(self) -> dict[str, bool | str]:
        if not self._enabled:
            return {"configured": False, "status": "disabled"}
        if self._device != "cpu":
            return {"configured": False, "status": "unsupported_device"}
        if not self._store.encryption_available:
            return {"configured": False, "status": "encryption_key_missing"}
        if not self._detector_path.is_file() or not self._recognizer_path.is_file():
            return {"configured": False, "status": "model_files_missing"}
        if self._load_error:
            return {"configured": False, "status": "model_load_failed"}
        try:
            self._models()
        except FaceModelUnavailableError:
            return {"configured": False, "status": "model_load_failed"}
        return {"configured": True, "status": "ready"}

    def enrollment_embedding(self, image: np.ndarray) -> np.ndarray:
        return self._extract_embedding(image)

    def submit(self, person_crop: np.ndarray) -> Future[RecognitionOutcome] | None:
        if not self.status()["configured"]:
            return None
        with self._submit_lock:
            if self._pending_count >= 8:
                return None
            self._pending_count += 1
        try:
            future = self._executor.submit(self._recognize_safely, person_crop.copy())
            future.add_done_callback(self._decrement_pending)
            return future
        except Exception:
            self._decrement_pending(None)
            raise

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _recognize_safely(self, person_crop: np.ndarray) -> RecognitionOutcome:
        try:
            embedding = self._extract_embedding(person_crop)
            return self._match(embedding)
        except FaceNotVisibleError:
            return RecognitionOutcome(status="FACE_NOT_VISIBLE")
        except FaceQualityError:
            return RecognitionOutcome(status="UNVERIFIED")
        except FaceModelUnavailableError:
            logger.exception("Face recognition model is unavailable")
        except Exception:
            logger.exception("Face recognition failed")
        return RecognitionOutcome(status="UNVERIFIED")

    def _match(self, embedding: np.ndarray) -> RecognitionOutcome:
        records = self._store.active_embeddings()
        if not records:
            return RecognitionOutcome(status="UNKNOWN")

        per_person: dict[int, dict] = {}
        for record in records:
            similarity = float(np.dot(embedding, record["embedding"]))
            person_id = int(record["person_id"])
            previous = per_person.get(person_id)
            if previous is None or similarity > previous["similarity"]:
                per_person[person_id] = {
                    **record,
                    "similarity": similarity,
                }

        ranked = sorted(
            per_person.values(),
            key=lambda item: item["similarity"],
            reverse=True,
        )
        best = ranked[0]
        second_score = ranked[1]["similarity"] if len(ranked) > 1 else -1.0
        if best["similarity"] < self._match_threshold:
            return RecognitionOutcome(
                status="UNKNOWN",
                similarity=best["similarity"],
            )
        if best["similarity"] - second_score < self._ambiguity_margin:
            return RecognitionOutcome(
                status="UNVERIFIED",
                similarity=best["similarity"],
            )

        verified_status: RecognitionState = {
            "resident": "VERIFIED_RESIDENT",
            "visitor": "VERIFIED_VISITOR",
            "authorized_staff": "VERIFIED_STAFF",
        }[best["category"]]
        return RecognitionOutcome(
            status=verified_status,
            person_id=best["person_id"],
            display_name=best["display_name"],
            category=best["category"],
            similarity=best["similarity"],
        )

    def _extract_embedding(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            raise FaceQualityError("The image is empty or has an unsupported format.")

        detector, recognizer = self._models()
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            raise FaceNotVisibleError("No face was detected.")
        if len(faces) != 1:
            raise FaceQualityError("Use an image containing exactly one face.")

        face = faces[0]
        face_width, face_height = int(face[2]), int(face[3])
        if min(face_width, face_height) < self._minimum_face_size:
            raise FaceQualityError("The face is too small for enrollment or matching.")
        if float(face[14]) < self._minimum_face_score:
            raise FaceQualityError("The face detection is not sufficiently clear.")

        face_x, face_y = int(face[0]), int(face[1])
        image_height, image_width = image.shape[:2]
        face_x1, face_x2 = max(0, face_x), min(image_width, face_x + face_width)
        face_y1, face_y2 = max(0, face_y), min(image_height, face_y + face_height)
        face_region = image[face_y1:face_y2, face_x1:face_x2]
        if face_region.size == 0:
            raise FaceQualityError("The detected face is outside the person crop.")

        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if brightness < 25 or brightness > 230 or sharpness < 20:
            raise FaceQualityError(
                "The face image is too dark, overexposed, or blurred."
            )

        aligned = recognizer.alignCrop(image, face)
        features = np.asarray(recognizer.feature(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(features))
        if not np.isfinite(features).all() or norm <= 1e-8:
            raise FaceQualityError("Could not produce a valid face embedding.")
        return features / norm

    def _models(self):
        with self._model_lock:
            if self._detector is not None and self._recognizer is not None:
                return self._detector, self._recognizer
            if not self._enabled or not self._store.encryption_available:
                raise FaceModelUnavailableError(
                    "Recognition is not configured for this installation."
                )
            if self._device != "cpu":
                raise FaceModelUnavailableError(
                    "This OpenCV face-model adapter currently supports CPU inference only."
                )
            if not self._detector_path.is_file() or not self._recognizer_path.is_file():
                raise FaceModelUnavailableError(
                    "The local face model files are missing."
                )
            try:
                self._detector = cv2.FaceDetectorYN_create(
                    str(self._detector_path),
                    "",
                    (320, 320),
                    0.8,
                    0.3,
                    5000,
                )
                self._recognizer = cv2.FaceRecognizerSF_create(
                    str(self._recognizer_path),
                    "",
                )
            except Exception:
                self._load_error = True
                raise FaceModelUnavailableError(
                    "The configured local face models could not be loaded."
                ) from None
            return self._detector, self._recognizer

    def _decrement_pending(self, _: Future | None) -> None:
        with self._submit_lock:
            self._pending_count = max(0, self._pending_count - 1)
