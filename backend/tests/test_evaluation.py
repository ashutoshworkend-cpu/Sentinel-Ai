from pathlib import Path

import pytest

from scripts.evaluate_yolo_dataset import (
    accepted_label,
    intersection_over_union,
    match_frame,
    read_yolo_labels,
)


def test_iou_for_identical_boxes() -> None:
    assert intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]) == 1


def test_matching_uses_each_box_once() -> None:
    ground_truth = [
        {"box": [0, 0, 10, 10]},
        {"box": [20, 20, 30, 30]},
    ]
    predictions = [
        {"box": [0, 0, 10, 10]},
        {"box": [20, 20, 30, 30]},
        {"box": [1, 1, 9, 9]},
    ]

    matches, matched_gt, matched_predictions = match_frame(
        ground_truth,
        predictions,
        0.5,
    )

    assert len(matches) == 2
    assert matched_gt == {0, 1}
    assert matched_predictions == {0, 1}


def test_low_confidence_and_unverified_animals_become_unknown() -> None:
    assert accepted_label("bird", 0.4, 0.55, {"bird"}) == "unknown"
    assert accepted_label("bird", 0.9, 0.55, {"bird"}) == "unknown"
    assert accepted_label("bird", 0.9, 0.55, {"bird"}, True) == "bird"


def test_supported_non_animal_label_passes_confidence_gate() -> None:
    assert accepted_label("person", 0.8, 0.55, {"bird"}) == "person"


def test_missing_label_file_is_not_treated_as_negative(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="empty file for a negative image"):
        read_yolo_labels(tmp_path / "missing.txt", 100, 100)


def test_empty_label_file_represents_a_negative_image(tmp_path: Path) -> None:
    label_file = tmp_path / "negative.txt"
    label_file.write_text("", encoding="utf-8")

    assert read_yolo_labels(label_file, 100, 100) == []
