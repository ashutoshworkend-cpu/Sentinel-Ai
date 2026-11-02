from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Any

from app.settings import settings


@dataclass
class ZonePresence:
    inside_frames: int = 0
    outside_frames: int = 0
    inside: bool = False
    entered_at: datetime | None = None
    entry_emitted: bool = False
    loiter_emitted: bool = False


@dataclass
class TrackedObject:
    canonical_id: int
    first_seen: datetime
    last_seen: datetime
    point: tuple[float, float]
    presences: dict[str, ZonePresence] = field(default_factory=dict)


class ZoneRuleEngine:
    def __init__(
        self,
        min_track_age_seconds: float = settings.zone_min_track_age_seconds,
        entry_frames: int = settings.zone_entry_frames,
        exit_frames: int = settings.zone_exit_frames,
        default_loiter_seconds: float = settings.zone_loiter_seconds,
        default_cooldown_seconds: float = settings.zone_event_cooldown_seconds,
        track_loss_tolerance_seconds: float = settings.zone_track_loss_tolerance_seconds,
        track_match_distance_fraction: float = settings.zone_track_match_distance_fraction,
    ) -> None:
        self.min_track_age_seconds = min_track_age_seconds
        self.entry_frames = entry_frames
        self.exit_frames = exit_frames
        self.default_loiter_seconds = default_loiter_seconds
        self.default_cooldown_seconds = default_cooldown_seconds
        self.track_loss_tolerance_seconds = track_loss_tolerance_seconds
        self.track_match_distance_fraction = track_match_distance_fraction
        self._tracks: dict[int, TrackedObject] = {}
        self._aliases: dict[int, int] = {}
        self._cooldowns: dict[tuple[str, int, str], datetime] = {}
        self._paused_at: datetime | None = None

    def reset(self) -> None:
        self._tracks.clear()
        self._aliases.clear()
        self._cooldowns.clear()
        self._paused_at = None

    def pause(self, timestamp: datetime) -> None:
        if self._paused_at is None:
            self._paused_at = timestamp

    def resume(self, timestamp: datetime) -> None:
        if self._paused_at is None:
            return
        paused_for = max(timedelta(0), timestamp - self._paused_at)
        for tracked in self._tracks.values():
            tracked.first_seen += paused_for
            tracked.last_seen = timestamp
            for presence in tracked.presences.values():
                if presence.entered_at is not None:
                    presence.entered_at += paused_for
        self._cooldowns = {
            key: last_emitted + paused_for
            for key, last_emitted in self._cooldowns.items()
        }
        self._paused_at = None

    def update(
        self,
        detections: list[Any],
        zones: list[dict[str, Any]],
        timestamp: datetime,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame dimensions must be positive.")
        enabled_zones = [zone for zone in zones if zone["enabled"]]
        enabled_zone_ids = {zone["id"] for zone in enabled_zones}
        for tracked in self._tracks.values():
            tracked.presences = {
                zone_id: presence
                for zone_id, presence in tracked.presences.items()
                if zone_id in enabled_zone_ids
            }
        current_ids: set[int] = set()
        events: list[dict[str, Any]] = []
        diagonal = math.hypot(frame_width, frame_height)

        for detection in detections:
            track_id = detection.track_id
            if track_id is None:
                continue
            point = self._reference_point(detection)
            canonical_id = self._canonical_id(
                track_id, point, timestamp, diagonal, current_ids
            )
            if canonical_id in current_ids:
                continue
            current_ids.add(canonical_id)
            tracked = self._tracks.get(canonical_id)
            if tracked is None:
                tracked = TrackedObject(canonical_id, timestamp, timestamp, point)
                self._tracks[canonical_id] = tracked
            else:
                tracked.last_seen = timestamp
                tracked.point = point

            age_seconds = (timestamp - tracked.first_seen).total_seconds()
            eligible = age_seconds >= self.min_track_age_seconds
            normalized_point = (point[0] / frame_width, point[1] / frame_height)
            for zone in enabled_zones:
                presence = tracked.presences.setdefault(zone["id"], ZonePresence())
                inside_now = self._contains(zone["polygon"], normalized_point)
                if inside_now:
                    presence.inside_frames += 1
                    presence.outside_frames = 0
                    if not presence.inside and presence.inside_frames >= self.entry_frames:
                        presence.inside = True
                        presence.entered_at = timestamp
                        presence.entry_emitted = False
                        presence.loiter_emitted = False
                else:
                    presence.outside_frames += 1
                    presence.inside_frames = 0
                    if presence.inside and presence.outside_frames >= self.exit_frames:
                        if eligible:
                            event = self._emit(
                                "exit", zone, tracked, detection, timestamp
                            )
                            if event:
                                events.append(event)
                        presence.inside = False
                        presence.entered_at = None
                        presence.entry_emitted = False
                        presence.loiter_emitted = False

                if eligible and presence.inside and not presence.entry_emitted:
                    event = self._emit("entry", zone, tracked, detection, timestamp)
                    presence.entry_emitted = True
                    if event:
                        events.append(event)

                if (
                    eligible
                    and presence.inside
                    and not presence.loiter_emitted
                    and presence.entered_at is not None
                    and (
                        timestamp - presence.entered_at
                    ).total_seconds()
                    >= zone.get("loiter_seconds", self.default_loiter_seconds)
                ):
                    event = self._emit(
                        "loitering", zone, tracked, detection, timestamp
                    )
                    presence.loiter_emitted = True
                    if event:
                        events.append(event)

        for track_id, tracked in self._tracks.items():
            if track_id not in current_ids:
                for presence in tracked.presences.values():
                    presence.inside_frames = 0
                    presence.outside_frames = 0
        self._expire_tracks(timestamp, current_ids)
        self._expire_cooldowns(timestamp, zones)
        return events

    def _canonical_id(
        self,
        track_id: int,
        point: tuple[float, float],
        timestamp: datetime,
        diagonal: float,
        current_ids: set[int],
    ) -> int:
        canonical = self._aliases.get(track_id, track_id)
        if canonical in self._tracks:
            tracked = self._tracks[canonical]
            if (
                timestamp - tracked.last_seen
            ).total_seconds() <= self.track_loss_tolerance_seconds:
                return canonical
            del self._tracks[canonical]
            self._aliases = {
                alias: target
                for alias, target in self._aliases.items()
                if target != canonical
            }
        candidates = [
            tracked
            for tracked in self._tracks.values()
            if tracked.canonical_id not in current_ids
            and 0 <= (timestamp - tracked.last_seen).total_seconds()
            <= self.track_loss_tolerance_seconds
            and math.dist(point, tracked.point)
            <= self.track_match_distance_fraction * diagonal
        ]
        if candidates:
            matched = min(
                candidates,
                key=lambda tracked: math.dist(point, tracked.point),
            )
            self._aliases[track_id] = matched.canonical_id
            return matched.canonical_id
        return track_id

    def _emit(
        self,
        event_type: str,
        zone: dict[str, Any],
        tracked: TrackedObject,
        detection: Any,
        timestamp: datetime,
    ) -> dict[str, Any] | None:
        key = (zone["id"], tracked.canonical_id, event_type)
        last_emitted = self._cooldowns.get(key)
        cooldown = zone.get(
            "event_cooldown_seconds", self.default_cooldown_seconds
        )
        if last_emitted is not None and (
            timestamp - last_emitted
        ).total_seconds() < cooldown:
            return None
        self._cooldowns[key] = timestamp
        return {
            "type": "zone_rule",
            "zone_id": zone["id"],
            "zone_name": zone["name"],
            "event_type": event_type,
            "track_id": detection.track_id,
            "object_label": detection.final_classification or "UNKNOWN",
            "timestamp": timestamp.isoformat(),
        }

    def _expire_tracks(
        self, timestamp: datetime, current_ids: set[int]
    ) -> None:
        expired = [
            track_id
            for track_id, tracked in self._tracks.items()
            if track_id not in current_ids
            and (timestamp - tracked.last_seen).total_seconds()
            > self.track_loss_tolerance_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]
            self._aliases = {
                alias: canonical
                for alias, canonical in self._aliases.items()
                if canonical != track_id
            }

    def _expire_cooldowns(
        self, timestamp: datetime, zones: list[dict[str, Any]]
    ) -> None:
        retention = max(
            self.default_cooldown_seconds,
            self.track_loss_tolerance_seconds,
            *(zone.get("event_cooldown_seconds", 0) for zone in zones),
        )
        self._cooldowns = {
            key: last_emitted
            for key, last_emitted in self._cooldowns.items()
            if (timestamp - last_emitted).total_seconds() <= retention
        }

    @staticmethod
    def _reference_point(detection: Any) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.box
        x = (x1 + x2) / 2
        y = y2 if detection.predicted_class == "person" else (y1 + y2) / 2
        return x, y

    @staticmethod
    def _contains(polygon: list[dict[str, float]], point: tuple[float, float]) -> bool:
        x, y = point
        inside = False
        previous = polygon[-1]
        for current in polygon:
            xi, yi = current["x"], current["y"]
            xj, yj = previous["x"], previous["y"]
            cross = (x - xi) * (yj - yi) - (y - yi) * (xj - xi)
            if (
                abs(cross) < 1e-9
                and min(xi, xj) <= x <= max(xi, xj)
                and min(yi, yj) <= y <= max(yi, yj)
            ):
                return True
            if (yi > y) != (yj > y) and x < (
                (xj - xi) * (y - yi) / (yj - yi) + xi
            ):
                inside = not inside
            previous = current
        return inside
