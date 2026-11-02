from collections import Counter, deque
from dataclasses import dataclass
from math import hypot


UNKNOWN_CLASS = "unknown"


@dataclass(frozen=True)
class TrackMemorySnapshot:
    first_seen_frame: int
    last_seen_frame: int
    observation_count: int
    prediction_history: list[dict[str, str | float | int]]
    movement_pixels: float


class TrackMemory:
    def __init__(self, history_size: int = 12) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive.")
        self._history: deque[dict[str, str | float | int]] = deque(
            maxlen=history_size
        )
        self._first_seen_frame: int | None = None
        self._last_seen_frame: int | None = None
        self._first_center: tuple[float, float] | None = None
        self._last_center: tuple[float, float] | None = None
        self._observation_count = 0

    def observe(
        self,
        frame_index: int,
        box: list[float],
        predicted_class: str,
        confidence: float,
        verification_status: str,
    ) -> TrackMemorySnapshot:
        if len(box) != 4:
            raise ValueError("A tracking box must contain four coordinates.")
        x1, y1, x2, y2 = box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        if self._first_seen_frame is None:
            self._first_seen_frame = frame_index
            self._first_center = center

        self._last_seen_frame = frame_index
        self._last_center = center
        self._observation_count += 1
        self._history.append(
            {
                "frame_index": frame_index,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "verification_status": verification_status,
            }
        )
        return self.snapshot()

    def snapshot(self) -> TrackMemorySnapshot:
        if (
            self._first_seen_frame is None
            or self._last_seen_frame is None
            or self._first_center is None
            or self._last_center is None
        ):
            raise RuntimeError("Track memory has no observations.")
        movement = hypot(
            self._last_center[0] - self._first_center[0],
            self._last_center[1] - self._first_center[1],
        )
        return TrackMemorySnapshot(
            first_seen_frame=self._first_seen_frame,
            last_seen_frame=self._last_seen_frame,
            observation_count=self._observation_count,
            prediction_history=list(self._history),
            movement_pixels=movement,
        )


@dataclass(frozen=True)
class Classification:
    class_name: str
    predicted_class: str
    temporal_candidate: str
    final_classification: str | None
    classification_confidence: float | None
    temporal_support: float
    verification_status: str


class TemporalClassifier:
    def __init__(
        self,
        min_observations: int = 3,
        window_size: int = 12,
        min_confidence: float = 0.55,
        min_support: float = 0.65,
        max_missing_frames: int = 45,
    ) -> None:
        if min_observations < 1:
            raise ValueError("min_observations must be positive.")
        if window_size < min_observations:
            raise ValueError("window_size must be at least min_observations.")
        if not 0 <= min_confidence <= 1 or not 0 < min_support <= 1:
            raise ValueError(
                "min_confidence must be in [0, 1] and min_support in (0, 1]."
            )
        if max_missing_frames < 1:
            raise ValueError("max_missing_frames must be positive.")
        self.min_observations = min_observations
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_missing_frames = max_missing_frames
        self._histories: dict[int, deque[tuple[str, float]]] = {}
        self._last_seen: dict[int, int] = {}
        self._stable_labels: dict[int, str] = {}

    def reset(self) -> None:
        self._histories.clear()
        self._last_seen.clear()
        self._stable_labels.clear()

    def observe(
        self,
        track_id: int | None,
        predicted_class: str,
        confidence: float,
        frame_index: int,
    ) -> Classification:
        if track_id is None:
            return Classification(
                class_name=UNKNOWN_CLASS,
                predicted_class=predicted_class,
                temporal_candidate=predicted_class,
                final_classification=None,
                classification_confidence=None,
                temporal_support=0.0,
                verification_status="uncertain",
            )

        self._prune(frame_index)
        history = self._histories.setdefault(
            track_id,
            deque(maxlen=self.window_size),
        )
        history.append((predicted_class, confidence))
        self._last_seen[track_id] = frame_index

        scores: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        total_weight = 0.0
        for label, score in history:
            scores[label] += score
            counts[label] += 1
            total_weight += score

        winner = max(scores, key=scores.get)
        support = scores[winner] / total_weight if total_weight else 0.0
        mean_confidence = scores[winner] / counts[winner]
        enough_evidence = (
            counts[winner] >= self.min_observations
            and mean_confidence >= self.min_confidence
            and support >= self.min_support
        )

        if enough_evidence:
            self._stable_labels[track_id] = winner
        stable_label = self._stable_labels.get(track_id)
        if stable_label is None:
            return Classification(
                class_name=UNKNOWN_CLASS,
                predicted_class=predicted_class,
                temporal_candidate=winner,
                final_classification=None,
                classification_confidence=mean_confidence,
                temporal_support=support,
                verification_status="uncertain",
            )

        if stable_label not in scores:
            self._stable_labels.pop(track_id, None)
            return Classification(
                class_name=UNKNOWN_CLASS,
                predicted_class=predicted_class,
                temporal_candidate=winner,
                final_classification=None,
                classification_confidence=mean_confidence,
                temporal_support=support,
                verification_status="uncertain",
            )

        stable_score = scores[stable_label] / counts[stable_label]
        stable_support = scores[stable_label] / total_weight if total_weight else 0.0
        if stable_support < self.min_support:
            return Classification(
                class_name=UNKNOWN_CLASS,
                predicted_class=predicted_class,
                temporal_candidate=winner,
                final_classification=None,
                classification_confidence=stable_score,
                temporal_support=stable_support,
                verification_status="uncertain",
            )
        return Classification(
            class_name=stable_label,
            predicted_class=predicted_class,
            temporal_candidate=winner,
            final_classification=stable_label,
            classification_confidence=stable_score,
            temporal_support=stable_support,
            verification_status="primary_only",
        )

    def _prune(self, frame_index: int) -> None:
        expired = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if frame_index - last_seen > self.max_missing_frames
        ]
        for track_id in expired:
            self._last_seen.pop(track_id, None)
            self._histories.pop(track_id, None)
            self._stable_labels.pop(track_id, None)
