from dataclasses import dataclass

import cv2
import numpy as np

COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

VERIFICATION_LABELS = COCO_LABELS + (
    "broom",
    "mop",
    "bucket",
    "garden tool",
    "hand tool",
    "bag",
    "unidentified household object",
)


@dataclass(frozen=True)
class VerificationResult:
    agrees: bool
    score: float
    alternative: str


class ClipVerifier:
    def __init__(
        self,
        model_name: str,
        min_score: float,
        min_margin: float,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Secondary verification is configured, but Transformers is not "
                "installed. Install the verification extra: pip install -e "
                '".[verification]".'
            ) from exc

        self._min_score = min_score
        self._min_margin = min_margin
        self._classifier = pipeline(
            "zero-shot-image-classification",
            model=model_name,
            device=-1,
        )

    def verify(
        self,
        frame: np.ndarray,
        box: list[float],
        expected_class: str,
    ) -> VerificationResult:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Cannot verify an empty or out-of-frame object crop.")

        crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        predictions = self._classifier(
            crop,
            candidate_labels=list(VERIFICATION_LABELS),
            hypothesis_template="A photo of {}.",
        )
        if len(predictions) < 2:
            raise RuntimeError("The verifier returned fewer than two candidates.")

        best, second = predictions[0], predictions[1]
        best_label = str(best["label"])
        best_score = float(best["score"])
        margin = best_score - float(second["score"])
        return VerificationResult(
            agrees=(
                best_label == expected_class
                and best_score >= self._min_score
                and margin >= self._min_margin
            ),
            score=best_score,
            alternative=best_label,
        )
