from app.classification import TemporalClassifier, TrackMemory, UNKNOWN_CLASS


def test_unstable_labels_remain_unknown() -> None:
    classifier = TemporalClassifier(min_observations=3, window_size=6)

    first = classifier.observe(14, "unknown", 0.42, 0)
    second = classifier.observe(14, "bird", 0.38, 1)
    third = classifier.observe(14, "unknown", 0.51, 2)

    assert first.class_name == UNKNOWN_CLASS
    assert second.class_name == UNKNOWN_CLASS
    assert third.class_name == UNKNOWN_CLASS
    assert third.final_classification is None


def test_consistent_track_predictions_stabilize() -> None:
    classifier = TemporalClassifier(min_observations=3, window_size=6)

    for frame_index in range(2):
        detection = classifier.observe(8, "person", 0.82, frame_index)
        assert detection.class_name == UNKNOWN_CLASS

    stable = classifier.observe(8, "person", 0.76, 2)
    assert stable.class_name == "person"
    assert stable.final_classification == "person"
    assert stable.verification_status == "primary_only"


def test_conflicting_predictions_do_not_flip_stable_identity() -> None:
    classifier = TemporalClassifier(min_observations=3, window_size=8)
    for frame_index in range(3):
        classifier.observe(4, "bird", 0.8, frame_index)

    first_conflict = classifier.observe(4, "broom", 0.99, 3)
    assert first_conflict.class_name == "bird"

    changed = classifier.observe(4, "broom", 0.99, 4)
    assert changed.class_name == UNKNOWN_CLASS
    assert changed.final_classification is None


def test_untracked_detections_are_not_presented_as_confirmed() -> None:
    detection = TemporalClassifier().observe(None, "bird", 0.91, 0)

    assert detection.class_name == UNKNOWN_CLASS
    assert detection.final_classification is None
    assert detection.verification_status == "uncertain"


def test_reset_discards_predictions_after_a_camera_reconnect() -> None:
    classifier = TemporalClassifier(min_observations=3, window_size=6)
    for frame_index in range(3):
        classifier.observe(3, "person", 0.9, frame_index)

    classifier.reset()
    after_reconnect = classifier.observe(3, "bird", 0.9, 3)

    assert after_reconnect.class_name == UNKNOWN_CLASS


def test_track_memory_keeps_a_bounded_history_and_movement() -> None:
    memory = TrackMemory(history_size=2)
    memory.observe(4, [0, 0, 10, 10], "unknown", 0.4, "uncertain")
    memory.observe(5, [3, 4, 13, 14], "bird", 0.6, "primary_only")
    snapshot = memory.observe(6, [6, 8, 16, 18], "bird", 0.7, "verified")

    assert snapshot.first_seen_frame == 4
    assert snapshot.last_seen_frame == 6
    assert snapshot.observation_count == 3
    assert len(snapshot.prediction_history) == 2
    assert snapshot.prediction_history[-1]["verification_status"] == "verified"
    assert snapshot.movement_pixels == 10
