from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from app.services.zone_rules import ZoneRuleEngine


ZONE = {
    "id": "front-door",
    "name": "Front door",
    "polygon": [
        {"x": 0.25, "y": 0.25},
        {"x": 0.75, "y": 0.25},
        {"x": 0.75, "y": 0.75},
        {"x": 0.25, "y": 0.75},
    ],
    "enabled": True,
    "loiter_seconds": 30,
    "event_cooldown_seconds": 30,
}


def detection(track_id: int, x: float, y: float, label: str = "UNKNOWN"):
    return SimpleNamespace(
        track_id=track_id,
        predicted_class="person",
        final_classification=label if label != "UNKNOWN" else None,
        box=[x - 5, y - 10, x + 5, y],
    )


def update(engine, timestamp, detections, zones=None):
    return engine.update(
        detections=detections,
        zones=zones or [ZONE],
        timestamp=timestamp,
        frame_width=100,
        frame_height=100,
    )


def test_entry_requires_track_age_and_two_inside_frames():
    engine = ZoneRuleEngine(min_track_age_seconds=1, entry_frames=2)
    start = datetime.now(timezone.utc)
    assert update(engine, start, [detection(1, 50, 40)]) == []
    assert update(engine, start + timedelta(seconds=0.5), [detection(1, 50, 40)]) == []
    events = update(engine, start + timedelta(seconds=1), [detection(1, 50, 40)])
    assert [event["event_type"] for event in events] == ["entry"]
    assert events[0]["object_label"] == "UNKNOWN"


def test_person_position_uses_bottom_center_and_exit_requires_two_frames():
    engine = ZoneRuleEngine(min_track_age_seconds=0, entry_frames=2, exit_frames=2)
    start = datetime.now(timezone.utc)
    narrow_zone = {
        **ZONE,
        "polygon": [
            {"x": 0.25, "y": 0.68},
            {"x": 0.75, "y": 0.68},
            {"x": 0.75, "y": 0.75},
            {"x": 0.25, "y": 0.75},
        ],
    }
    update(engine, start, [detection(1, 50, 70)], [narrow_zone])
    entry = update(
        engine, start + timedelta(milliseconds=10), [detection(1, 50, 70)], [narrow_zone]
    )
    assert [event["event_type"] for event in entry] == ["entry"]

    assert update(
        engine, start + timedelta(milliseconds=20), [detection(1, 50, 20)], [narrow_zone]
    ) == []
    exit_events = update(
        engine, start + timedelta(milliseconds=30), [detection(1, 50, 20)], [narrow_zone]
    )
    assert [event["event_type"] for event in exit_events] == ["exit"]
    assert update(
        engine, start + timedelta(milliseconds=40), [detection(1, 50, 20)], [narrow_zone]
    ) == []


def test_missing_frame_does_not_count_as_exit_and_changed_id_is_reassociated():
    engine = ZoneRuleEngine(
        min_track_age_seconds=0,
        entry_frames=2,
        exit_frames=2,
        track_loss_tolerance_seconds=2,
    )
    start = datetime.now(timezone.utc)
    update(engine, start, [detection(7, 50, 40)])
    update(engine, start + timedelta(milliseconds=10), [detection(7, 50, 40)])
    assert update(engine, start + timedelta(seconds=0.5), []) == []
    assert update(
        engine, start + timedelta(seconds=0.6), [detection(19, 50, 40)]
    ) == []


def test_loiter_emits_once_for_continuous_stay():
    engine = ZoneRuleEngine(
        min_track_age_seconds=0,
        entry_frames=2,
        default_loiter_seconds=1,
        default_cooldown_seconds=0,
    )
    start = datetime.now(timezone.utc)
    short_loiter_zone = {**ZONE, "loiter_seconds": 1}
    update(engine, start, [detection(1, 50, 40)], [short_loiter_zone])
    events = update(
        engine, start + timedelta(milliseconds=10), [detection(1, 50, 40)],
        [short_loiter_zone],
    )
    assert [event["event_type"] for event in events] == ["entry"]
    events = update(
        engine, start + timedelta(seconds=1.1), [detection(1, 50, 40)],
        [short_loiter_zone],
    )
    assert [event["event_type"] for event in events] == ["loitering"]
    assert update(
        engine, start + timedelta(seconds=2), [detection(1, 50, 40)],
        [short_loiter_zone],
    ) == []


def test_camera_pause_does_not_count_toward_track_age_or_loitering():
    engine = ZoneRuleEngine(
        min_track_age_seconds=0,
        entry_frames=2,
        default_loiter_seconds=1,
        default_cooldown_seconds=0,
    )
    short_loiter_zone = {**ZONE, "loiter_seconds": 1}
    start = datetime.now(timezone.utc)
    update(engine, start, [detection(1, 50, 40)], [short_loiter_zone])
    entered_at = start + timedelta(milliseconds=10)
    assert [
        event["event_type"]
        for event in update(engine, entered_at, [detection(1, 50, 40)], [short_loiter_zone])
    ] == ["entry"]

    engine.pause(entered_at)
    resumed_at = entered_at + timedelta(seconds=60)
    engine.resume(resumed_at)
    assert update(
        engine,
        resumed_at + timedelta(milliseconds=100),
        [detection(1, 50, 40)],
        [short_loiter_zone],
    ) == []
