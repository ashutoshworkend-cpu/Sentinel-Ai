import logging
import threading
from collections import OrderedDict
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Callable

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from app.classification import TemporalClassifier, TrackMemory, UNKNOWN_CLASS
from app.schemas import CameraStatus, Detection
from app.services.biometric_store import BiometricStore
from app.services.face_recognition import (
    FaceRecognitionService,
    RecognitionOutcome,
)
from app.settings import settings
from app.services.verification import ClipVerifier, VerificationResult
from app.services.zone_rules import ZoneRuleEngine
from app.services.zone_store import ZoneStore

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict], None]


def parse_camera_source(source: str) -> int | str:
    value = source.strip()
    if value.isdecimal():
        return int(value)
    return value


class VideoProcessor:
    def __init__(
        self,
        camera_id: str,
        name: str,
        source: str,
        on_event: EventCallback,
        recognition_service: FaceRecognitionService | None = None,
        biometric_store: BiometricStore | None = None,
        zone_store: ZoneStore | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.source_type = "webcam" if source.strip().isdecimal() else "network"
        self._on_event = on_event
        self._recognition_service = recognition_service
        self._biometric_store = biometric_store
        self._zone_store = zone_store
        self._zone_rules = ZoneRuleEngine()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._status = "stopped"
        self._error: str | None = None
        self._detections: list[Detection] = []
        self._timestamp: datetime | None = None
        self._jpeg: bytes | None = None
        self._model: YOLO | None = None
        self._classifier = TemporalClassifier(
            min_observations=settings.temporal_min_observations,
            window_size=settings.temporal_window,
            min_confidence=settings.min_class_confidence,
            min_support=settings.temporal_min_support,
        )
        self._track_memories: dict[int, TrackMemory] = {}
        self._recognition_futures: dict[
            int, tuple[int, datetime, Future[RecognitionOutcome], int]
        ] = {}
        self._recognition_results: dict[int, tuple[int, RecognitionOutcome]] = {}
        self._recognition_last_attempt: dict[int, int] = {}
        self._recognition_last_recorded: dict[int, tuple[str, int | None]] = {}
        self._recognition_revision = (
            recognition_service.profile_revision
            if recognition_service is not None
            else None
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            self._status = "stopped"

    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                id=self.camera_id,
                name=self.name,
                status=self._status,
                source_type=self.source_type,
                detections=list(self._detections),
                timestamp=self._timestamp,
                error=self._error,
            )

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def _set_error(self, message: str | None) -> None:
        with self._lock:
            self._error = message

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            self._model = YOLO(settings.model_path)
            verifier = (
                ClipVerifier(
                    model_name=settings.verifier_model,
                    min_score=settings.verifier_min_score,
                    min_margin=settings.verifier_min_margin,
                )
                if settings.verifier_model
                else None
            )
            verification_cache: OrderedDict[
                tuple[int, str], tuple[VerificationResult, int]
            ] = OrderedDict()
            frame_index = 0
            while not self._stop.is_set():
                with self._lock:
                    self._status = "connecting"
                    self._error = None

                capture = cv2.VideoCapture(parse_camera_source(self.source))
                if not capture.isOpened():
                    capture.release()
                    capture = None
                    self._zone_rules.pause(datetime.now(timezone.utc))
                    with self._lock:
                        self._status = "error"
                        self._error = "Unable to open the configured camera source."
                    self._stop.wait(settings.reconnect_delay)
                    continue

                with self._lock:
                    self._status = "online"
                self._zone_rules.resume(datetime.now(timezone.utc))
                self._classifier.reset()
                self._track_memories.clear()
                self._recognition_futures.clear()
                self._recognition_results.clear()
                self._recognition_last_attempt.clear()
                self._recognition_last_recorded.clear()
                verification_cache.clear()

                while not self._stop.is_set():
                    success, frame = capture.read()
                    if not success:
                        self._zone_rules.pause(datetime.now(timezone.utc))
                        with self._lock:
                            self._status = "offline"
                            self._error = "Camera frame read failed; reconnecting."
                        break

                    timestamp = datetime.now(timezone.utc)
                    try:
                        result = self._model.track(
                            frame,
                            persist=True,
                            tracker="bytetrack.yaml",
                            conf=settings.confidence,
                            iou=settings.nms_iou,
                            imgsz=settings.image_size,
                            verbose=False,
                        )[0]
                        detections = self._extract_detections(
                            result=result,
                            frame=frame,
                            frame_index=frame_index,
                            verifier=verifier,
                            verification_cache=verification_cache,
                        )
                        if self._zone_store is not None:
                            for rule_event in self._zone_rules.update(
                                detections=detections,
                                zones=self._zone_store.list_zones(self.camera_id),
                                timestamp=timestamp,
                                frame_width=frame.shape[1],
                                frame_height=frame.shape[0],
                            ):
                                self._on_event(
                                    {
                                        **rule_event,
                                        "camera_id": self.camera_id,
                                        "camera_name": self.name,
                                    }
                                )
                        self._collect_recognition_results(frame_index)
                        annotated = self._annotate_frame(frame, detections)
                        encoded, jpeg = cv2.imencode(".jpg", annotated)
                        if not encoded:
                            raise RuntimeError("Could not encode the annotated frame.")
                    except Exception as exc:
                        logger.exception("Inference failed for camera %s", self.camera_id)
                        self._set_error(f"Inference failed: {exc}")
                        self._stop.wait(settings.frame_interval)
                        continue

                    frame_index += 1
                    with self._lock:
                        self._detections = detections
                        self._timestamp = timestamp
                        self._jpeg = jpeg.tobytes()
                        self._error = None

                    self._on_event(
                        {
                            "type": "detections",
                            "camera_id": self.camera_id,
                            "camera_name": self.name,
                            "timestamp": timestamp.isoformat(),
                            "detections": [
                                detection.model_dump() for detection in detections
                            ],
                        }
                    )
                    self._stop.wait(settings.frame_interval)

                if capture:
                    capture.release()
                    capture = None
                if not self._stop.is_set():
                    self._stop.wait(settings.reconnect_delay)
        except Exception as exc:
            logger.exception("Camera worker failed for camera %s", self.camera_id)
            with self._lock:
                self._status = "error"
                self._error = f"Camera worker failed: {exc}"
        finally:
            if capture:
                capture.release()
            with self._lock:
                if self._stop.is_set():
                    self._status = "stopped"

    def _extract_detections(
        self,
        result: Results,
        frame: np.ndarray,
        frame_index: int,
        verifier: ClipVerifier | None,
        verification_cache: OrderedDict[
            tuple[int, str], tuple[VerificationResult, int]
        ],
    ) -> list[Detection]:
        self._refresh_recognition_revision()
        for key, (_, last_seen) in list(verification_cache.items()):
            if frame_index - last_seen > settings.verifier_cache_ttl_frames:
                del verification_cache[key]

        expired_tracks = [
            track_id
            for track_id, memory in self._track_memories.items()
            if memory.snapshot().last_seen_frame
            < frame_index - self._classifier.max_missing_frames
        ]
        for track_id in expired_tracks:
            del self._track_memories[track_id]
            self._recognition_futures.pop(track_id, None)
            self._recognition_results.pop(track_id, None)
            self._recognition_last_attempt.pop(track_id, None)
            self._recognition_last_recorded.pop(track_id, None)

        boxes = result.boxes
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            predicted_class = str(result.names[int(box.cls[0].item())])
            confidence = float(box.conf[0].item())
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].item())
            coordinates = [float(value) for value in box.xyxy[0].tolist()]
            classification = self._classifier.observe(
                track_id=track_id,
                predicted_class=predicted_class,
                confidence=confidence,
                frame_index=frame_index,
            )
            verification_score = None
            verification_alternative = None
            verification_status = classification.verification_status
            final_classification = classification.final_classification

            verification_target = final_classification or predicted_class
            requires_verification = (
                verification_target in settings.verifier_required_classes
            )
            should_verify = (
                track_id is not None
                and (
                    requires_verification
                    or verification_target in settings.verifier_trigger_classes
                    or confidence < 0.7
                    or (classification.classification_confidence or confidence) < 0.7
                    or classification.temporal_support < settings.temporal_min_support
                )
            )
            if should_verify and verifier is None:
                verification_status = "uncertain"
                final_classification = None
            elif should_verify:
                assert verifier is not None
                cache_key = (track_id, verification_target)
                cached = verification_cache.get(cache_key)
                if cached is None:
                    verification = verifier.verify(
                        frame=frame,
                        box=coordinates,
                        expected_class=verification_target,
                    )
                    verification_cache[cache_key] = (verification, frame_index)
                    verification_cache.move_to_end(cache_key)
                    while len(verification_cache) > 1024:
                        verification_cache.popitem(last=False)
                else:
                    verification = cached[0]
                    verification_cache.move_to_end(cache_key)

                if verification is not None:
                    verification_alternative = verification.alternative
                if verification is None or not verification.agrees:
                    verification_status = "uncertain"
                    final_classification = None
                elif final_classification is not None:
                    verification_status = "verified"
                    verification_score = verification.score
                else:
                    verification_status = "uncertain"
                    verification_score = verification.score

            recognition = self._recognition_results.get(track_id) if track_id is not None else None
            recognition_outcome = recognition[1] if recognition is not None else None
            recognition_configured = (
                predicted_class == "person"
                and track_id is not None
                and self._recognition_service is not None
                and self._recognition_service.status()["configured"]
            )
            recognition_status = (
                recognition_outcome.status
                if recognition_outcome is not None
                else (
                    "IDENTIFYING"
                    if predicted_class == "person"
                    and track_id is not None
                    and recognition_configured
                    else "UNVERIFIED"
                )
            )
            recognized_person_id = (
                recognition_outcome.person_id
                if recognition_outcome is not None
                else None
            )
            recognized_name = (
                recognition_outcome.display_name
                if recognition_outcome is not None
                else None
            )
            recognized_category = (
                recognition_outcome.category
                if recognition_outcome is not None
                else None
            )
            recognition_similarity = (
                recognition_outcome.similarity
                if recognition_outcome is not None
                else None
            )

            if (
                predicted_class == "person"
                and track_id is not None
                and recognition_configured
                and self._recognition_service is not None
                and track_id not in self._recognition_futures
            ):
                last_attempt = self._recognition_last_attempt.get(track_id)
                previous = self._recognition_results.get(track_id)
                result_needs_refresh = (
                    previous is None
                    or previous[1].status
                    not in {
                        "VERIFIED_RESIDENT",
                        "VERIFIED_VISITOR",
                        "VERIFIED_STAFF",
                    }
                    or frame_index - previous[0] >= settings.recognition_recheck_frames
                )
                interval_elapsed = (
                    last_attempt is None
                    or frame_index - last_attempt
                    >= settings.recognition_interval_frames
                )
                if result_needs_refresh and interval_elapsed:
                    image_height, image_width = frame.shape[:2]
                    x1, y1, x2, y2 = (
                        int(round(value)) for value in coordinates
                    )
                    x1, x2 = max(0, x1), min(image_width, x2)
                    y1, y2 = max(0, y1), min(image_height, y2)
                    if x2 > x1 and y2 > y1:
                        future = self._recognition_service.submit(frame[y1:y2, x1:x2])
                        if future is not None:
                            self._recognition_futures[track_id] = (
                                frame_index,
                                datetime.now(timezone.utc),
                                future,
                                self._recognition_service.profile_revision,
                            )
                            self._recognition_last_attempt[track_id] = frame_index

            x1, y1, x2, y2 = coordinates
            box_width, box_height = max(0.0, x2 - x1), max(0.0, y2 - y1)
            observations = []
            if min(box_width, box_height) > 0 and max(box_width, box_height) / min(
                box_width,
                box_height,
            ) >= 2.5:
                observations.append("elongated shape in the image")

            memory = None
            if track_id is not None:
                memory = self._track_memories.setdefault(
                    track_id,
                    TrackMemory(history_size=settings.temporal_window),
                ).observe(
                    frame_index=frame_index,
                    box=coordinates,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    verification_status=verification_status,
                )

            detections.append(
                Detection(
                    class_name=final_classification or UNKNOWN_CLASS,
                    predicted_class=predicted_class,
                    temporal_candidate=classification.temporal_candidate,
                    final_classification=final_classification,
                    confidence=confidence,
                    classification_confidence=classification.classification_confidence,
                    temporal_support=classification.temporal_support,
                    verification_status=verification_status,
                    verification_score=verification_score,
                    verification_alternative=verification_alternative,
                    observations=observations,
                    object_id=(
                        f"{self.camera_id}:{track_id}" if track_id is not None else None
                    ),
                    track_id=track_id,
                    first_seen_frame=(
                        memory.first_seen_frame if memory is not None else None
                    ),
                    last_seen_frame=(
                        memory.last_seen_frame if memory is not None else None
                    ),
                    track_observation_count=(
                        memory.observation_count if memory is not None else 0
                    ),
                    prediction_history=(
                        memory.prediction_history if memory is not None else []
                    ),
                    movement_pixels=(
                        memory.movement_pixels if memory is not None else None
                    ),
                    recognition_status=recognition_status,
                    recognized_person_id=recognized_person_id,
                    recognized_name=recognized_name,
                    recognized_category=recognized_category,
                    recognition_similarity=recognition_similarity,
                    box=coordinates,
                )
            )
        return detections

    def _collect_recognition_results(self, frame_index: int) -> None:
        for track_id, (_, timestamp, future, revision) in list(
            self._recognition_futures.items()
        ):
            if not future.done():
                continue
            del self._recognition_futures[track_id]
            if (
                self._recognition_service is None
                or revision != self._recognition_service.profile_revision
            ):
                continue
            if track_id not in self._track_memories:
                continue
            try:
                outcome = future.result()
            except Exception:
                logger.exception(
                    "Recognition worker failed for camera %s", self.camera_id
                )
                outcome = RecognitionOutcome(status="UNVERIFIED")
            self._recognition_results[track_id] = (frame_index, outcome)
            signature = (outcome.status, outcome.person_id)
            if (
                signature != self._recognition_last_recorded.get(track_id)
                and self._biometric_store is not None
            ):
                self._biometric_store.record_recognition(
                    camera_id=self.camera_id,
                    tracking_id=track_id,
                    timestamp=timestamp,
                    status=outcome.status,
                    person_id=outcome.person_id,
                    similarity=outcome.similarity,
                )
                self._recognition_last_recorded[track_id] = signature

    def _refresh_recognition_revision(self) -> None:
        if self._recognition_service is None:
            return
        revision = self._recognition_service.profile_revision
        if revision == self._recognition_revision:
            return
        self._recognition_revision = revision
        self._recognition_futures.clear()
        self._recognition_results.clear()
        self._recognition_last_attempt.clear()
        self._recognition_last_recorded.clear()

    @staticmethod
    def _person_label(
        detection: Detection,
    ) -> tuple[str, tuple[int, int, int]]:
        verified_categories = {
            "VERIFIED_RESIDENT": ("Resident", (72, 196, 128)),
            "VERIFIED_VISITOR": ("Visitor", (220, 175, 70)),
            "VERIFIED_STAFF": ("Staff", (190, 100, 220)),
        }
        verified = verified_categories.get(detection.recognition_status)
        if verified and detection.recognized_name:
            category, color = verified
            return f"{detection.recognized_name} | {category}", color
        if detection.recognition_status == "IDENTIFYING":
            return "Identifying...", (220, 190, 20)
        if detection.recognition_status == "FACE_NOT_VISIBLE":
            return "Person | Face Not Visible", (150, 150, 150)
        return "Unknown Person", (20, 170, 245)

    @staticmethod
    def _annotate_frame(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = (int(round(value)) for value in detection.box)
            if detection.predicted_class == "person":
                label, color = VideoProcessor._person_label(detection)
            else:
                color = (
                    (72, 196, 128)
                    if detection.verification_status == "verified"
                    else (60, 190, 220)
                    if detection.verification_status == "primary_only"
                    else (20, 170, 245)
                )
                if detection.class_name == UNKNOWN_CLASS:
                    label = f"unknown ({detection.predicted_class}?) {detection.confidence:.2f}"
                else:
                    label = f"{detection.class_name} {detection.confidence:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )
            text_x = max(0, min(x1, annotated.shape[1] - text_width - 8))
            text_y = max(text_height + baseline + 6, y1 - 8)
            cv2.rectangle(
                annotated,
                (text_x, text_y - text_height - baseline - 6),
                (text_x + text_width + 8, text_y + 2),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (text_x + 4, text_y - baseline - 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated
