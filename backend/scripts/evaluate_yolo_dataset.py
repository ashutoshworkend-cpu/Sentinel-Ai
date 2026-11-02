import argparse
import json
import statistics
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from ultralytics import YOLO

from app.classification import TemporalClassifier
from app.settings import settings
from app.services.video_processor import VideoProcessor
from app.services.verification import ClipVerifier, VerificationResult

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_samples(
    images: list[Path] | None,
    image_root: Path | None,
    video: Path | None,
    labels: Path,
    frame_step: int,
) -> Iterator[tuple[str, int, np.ndarray, Path]]:
    if images is not None:
        if image_root is None:
            raise ValueError("An image root is required for image datasets.")
        for frame_index, image_path in enumerate(images):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Could not decode image: {image_path}")
            label_path = labels / image_path.relative_to(image_root).with_suffix(".txt")
            yield str(image_path), frame_index, frame, label_path
        return

    if video is None:
        raise ValueError("Provide either an image directory or a video file.")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video}")
    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                label_path = labels / f"frame_{frame_index:06d}.txt"
                yield str(video), frame_index, frame, label_path
            frame_index += 1
    finally:
        capture.release()


def intersection_over_union(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0,
        second[3] - second[1],
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def read_yolo_labels(path: Path, width: int, height: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing label file {path}; create an empty file for a negative image."
        )
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{path}:{line_number}: expected class x_center y_center width height.")
        class_id_value, center_x, center_y, box_width, box_height = map(float, values)
        if not class_id_value.is_integer():
            raise ValueError(f"{path}:{line_number}: class id must be an integer.")
        if not all(
            0 <= value <= 1
            for value in (center_x, center_y, box_width, box_height)
        ):
            raise ValueError(f"{path}:{line_number}: YOLO box values must be normalized.")
        if box_width == 0 or box_height == 0:
            raise ValueError(f"{path}:{line_number}: box dimensions must be positive.")
        rows.append(
            {
                "class_id": int(class_id_value),
                "box": [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                ],
            }
        )
    return rows


def match_frame(
    ground_truth: list[dict],
    predictions: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    candidates = []
    for gt_index, gt in enumerate(ground_truth):
        for prediction_index, prediction in enumerate(predictions):
            overlap = intersection_over_union(gt["box"], prediction["box"])
            if overlap >= iou_threshold:
                candidates.append((overlap, gt_index, prediction_index))
    candidates.sort(reverse=True)

    matched_gt: set[int] = set()
    matched_predictions: set[int] = set()
    matches = []
    for _, gt_index, prediction_index in candidates:
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index))
    return matches, matched_gt, matched_predictions


def accepted_label(
    predicted_class: str,
    confidence: float,
    min_confidence: float,
    required_verification: set[str] | frozenset[str],
    verification_agrees: bool = False,
) -> str:
    if confidence < min_confidence:
        return "unknown"
    if predicted_class in required_verification and not verification_agrees:
        return "unknown"
    return predicted_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate detector classifications against YOLO-format labels."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Image directory.")
    source.add_argument("--video", type=Path, help="Labeled video sequence.")
    parser.add_argument("--labels", type=Path, required=True, help="YOLO label directory.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="For videos, infer/evaluate every Nth source frame.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        help="Optional JSON object mapping dataset class IDs to class names.",
    )
    parser.add_argument("--model", default=settings.model_path)
    parser.add_argument("--confidence", type=float, default=settings.confidence)
    parser.add_argument(
        "--label-confidence",
        type=float,
        default=settings.min_class_confidence,
        help="Lower-scoring detections count as unknown, not as the predicted class.",
    )
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--nms-iou", type=float, default=settings.nms_iou)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the configured CLIP verifier on required classes.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = None
    if args.images:
        if not args.images.is_dir():
            raise ValueError(f"Image directory does not exist: {args.images}")
        images = sorted(
            path
            for path in args.images.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in {args.images}.")
    if args.video and not args.video.is_file():
        raise ValueError(f"Video file does not exist: {args.video}")
    if not args.labels.is_dir():
        raise ValueError(f"Label directory does not exist: {args.labels}")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be positive.")
    if not 0 < args.match_iou <= 1:
        raise ValueError("--match-iou must be in (0, 1].")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be in [0, 1].")
    if not 0 <= args.label_confidence <= 1:
        raise ValueError("--label-confidence must be in [0, 1].")
    if args.image_size < 1:
        raise ValueError("--image-size must be positive.")
    if not 0 < args.nms_iou <= 1:
        raise ValueError("--nms-iou must be in (0, 1].")
    if args.verify and not settings.verifier_model:
        raise ValueError(
            "--verify requires SENTINEL_VERIFIER_MODEL to be configured."
        )

    model = YOLO(args.model)
    verifier = (
        ClipVerifier(
            model_name=settings.verifier_model,
            min_score=settings.verifier_min_score,
            min_margin=settings.verifier_min_margin,
        )
        if args.verify
        else None
    )
    video_processor = None
    verification_cache: OrderedDict[
        tuple[int, str], tuple[VerificationResult, int]
    ] = OrderedDict()
    if args.video:
        video_processor = VideoProcessor(
            camera_id="evaluation",
            name=args.video.name,
            source=str(args.video),
            on_event=lambda _: None,
        )
        video_processor._classifier = TemporalClassifier(
            min_observations=settings.temporal_min_observations,
            window_size=settings.temporal_window,
            min_confidence=args.label_confidence,
            min_support=settings.temporal_min_support,
        )
    model_class_names = {
        int(class_id): str(name) for class_id, name in model.names.items()
    }
    ground_truth_names = (
        {
            int(class_id): str(name)
            for class_id, name in json.loads(
                args.class_map.read_text(encoding="utf-8")
            ).items()
        }
        if args.class_map
        else model_class_names
    )
    confusion: Counter[tuple[str, str]] = Counter()
    ground_truth_count = 0
    prediction_count = 0
    correct_count = 0
    unknown_matched = 0
    false_positive_images = 0
    false_negative_count = 0
    latencies = []
    sample_count = 0

    for source_name, frame_index, frame, label_path in iter_samples(
        images=images,
        image_root=args.images,
        video=args.video,
        labels=args.labels,
        frame_step=args.frame_step,
    ):
        sample_count += 1
        height, width = frame.shape[:2]
        ground_truth = read_yolo_labels(
            label_path,
            width,
            height,
        )
        for annotation in ground_truth:
            if annotation["class_id"] not in ground_truth_names:
                raise ValueError(
                    f"{source_name}:{frame_index}: class id {annotation['class_id']} "
                    "is missing from the dataset class map."
                )

        started = time.perf_counter()
        if args.video:
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.confidence,
                iou=args.nms_iou,
                imgsz=args.image_size,
                verbose=False,
            )[0]
            if video_processor is None:
                raise RuntimeError("Video processor initialization failed.")
            detections = video_processor._extract_detections(
                result=result,
                frame=frame,
                frame_index=frame_index,
                verifier=verifier,
                verification_cache=verification_cache,
            )
            predictions = [
                {"class_name": detection.class_name, "box": detection.box}
                for detection in detections
            ]
        else:
            result = model.predict(
                frame,
                conf=args.confidence,
                iou=args.nms_iou,
                imgsz=args.image_size,
                verbose=False,
            )[0]
            predictions = []
            if result.boxes is not None:
                for box in result.boxes:
                    score = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    if class_id not in model_class_names:
                        raise ValueError(f"Model returned unknown class id {class_id}.")
                    predicted_class = model_class_names[class_id]
                    coordinates = [float(value) for value in box.xyxy[0].tolist()]
                    verification_agrees = False
                    if (
                        verifier is not None
                        and predicted_class in settings.verifier_required_classes
                        and score >= args.label_confidence
                    ):
                        verification_agrees = verifier.verify(
                            frame=frame,
                            box=coordinates,
                            expected_class=predicted_class,
                        ).agrees
                    predictions.append(
                        {
                            "class_name": accepted_label(
                                predicted_class=predicted_class,
                                confidence=score,
                                min_confidence=args.label_confidence,
                                required_verification=settings.verifier_required_classes,
                                verification_agrees=verification_agrees,
                            ),
                            "box": coordinates,
                        }
                    )
        latencies.append(time.perf_counter() - started)

        matches, matched_gt, matched_predictions = match_frame(
            ground_truth,
            predictions,
            args.match_iou,
        )
        ground_truth_count += len(ground_truth)
        prediction_count += len(predictions)
        frame_has_false_positive = False

        for gt_index, prediction_index in matches:
            actual = ground_truth_names[ground_truth[gt_index]["class_id"]]
            predicted = predictions[prediction_index]["class_name"]
            confusion[(actual, predicted)] += 1
            if actual == predicted:
                correct_count += 1
            else:
                frame_has_false_positive = True
                false_negative_count += 1
                if predicted == "unknown":
                    unknown_matched += 1
        for gt_index, annotation in enumerate(ground_truth):
            if gt_index not in matched_gt:
                actual = ground_truth_names[annotation["class_id"]]
                confusion[(actual, "background")] += 1
                false_negative_count += 1
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in matched_predictions:
                confusion[("background", prediction["class_name"])] += 1
                frame_has_false_positive = True
        if frame_has_false_positive:
            false_positive_images += 1

    if sample_count == 0:
        raise ValueError("No image or video frames were evaluated.")

    false_positive_count = prediction_count - correct_count
    precision = correct_count / prediction_count if prediction_count else 0.0
    recall = correct_count / ground_truth_count if ground_truth_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_count = sum(
        count
        for (actual, predicted), count in confusion.items()
        if actual != "background" and predicted != "background"
    )
    class_axis = sorted(
        {actual for actual, _ in confusion}
        | {predicted for _, predicted in confusion}
        | {"background"}
    )
    per_class = {}
    for label in class_axis:
        if label == "background":
            continue
        true_positive = confusion[(label, label)]
        false_positive = sum(
            count
            for (actual, predicted), count in confusion.items()
            if predicted == label and actual != label
        )
        false_negative = sum(
            count
            for (actual, predicted), count in confusion.items()
            if actual == label and predicted != label
        )
        class_precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        class_recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        per_class[label] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1": (
                2 * class_precision * class_recall / (class_precision + class_recall)
                if class_precision + class_recall
                else 0.0
            ),
            "true_positives": true_positive,
            "false_positives": false_positive,
            "false_negatives": false_negative,
        }

    report = {
        "model": args.model,
        "input": str(args.images or args.video),
        "images": sample_count if args.images else None,
        "video_sampled_frames": sample_count if args.video else None,
        "video_frame_step": args.frame_step if args.video else None,
        "iou_match_threshold": args.match_iou,
        "detection_confidence_threshold": args.confidence,
        "label_confidence_threshold": args.label_confidence,
        "image_size": args.image_size,
        "nms_iou_threshold": args.nms_iou,
        "secondary_verifier_enabled": verifier is not None,
        "ground_truth_objects": ground_truth_count,
        "predictions": prediction_count,
        "correct_classified_matches": correct_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": false_positive_count,
        "false_negatives": false_negative_count,
        "false_positive_image_rate": (
            false_positive_images / sample_count if sample_count else 0.0
        ),
        "false_negative_rate_objects": (
            false_negative_count / ground_truth_count if ground_truth_count else 0.0
        ),
        "unknown_classification_rate": (
            unknown_matched / matched_count if matched_count else 0.0
        ),
        "mean_inference_latency_ms": statistics.mean(latencies) * 1000,
        "per_class_metrics": per_class,
        "confusion_matrix": {
            "labels": class_axis,
            "rows": "ground truth class",
            "columns": "predicted class; background denotes unmatched objects",
            "counts": [
                [confusion[(actual, predicted)] for predicted in class_axis]
                for actual in class_axis
            ],
        },
        "metric_notes": (
            "A localization match is class-agnostic greedy IoU matching. "
            "Wrong-class matched boxes count as one FP and one FN for "
            "precision/recall. The image false-positive rate is the fraction "
            "of all images containing a wrong-class or unmatched prediction; "
            "it is not a negative-image-only statistical FPR."
        ),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
