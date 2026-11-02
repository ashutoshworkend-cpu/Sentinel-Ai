import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from app.schemas import ZoneCreate, ZoneUpdate


class ZoneStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = threading.RLock()
        self._zones: dict[str, dict[str, dict[str, Any]]] = {}

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_zones (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    polygon_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    loiter_seconds REAL NOT NULL,
                    event_cooldown_seconds REAL NOT NULL
                )
                """
            )
            rows = connection.execute(
                """
                SELECT id, camera_id, name, polygon_json, enabled,
                       loiter_seconds, event_cooldown_seconds
                FROM detection_zones
                ORDER BY name COLLATE NOCASE, id
                """
            ).fetchall()
        with self._lock:
            self._zones = {}
            for row in rows:
                zone = {
                    "id": row[0],
                    "camera_id": row[1],
                    "name": row[2],
                    "polygon": json.loads(row[3]),
                    "enabled": bool(row[4]),
                    "loiter_seconds": row[5],
                    "event_cooldown_seconds": row[6],
                }
                self._zones.setdefault(zone["camera_id"], {})[zone["id"]] = zone

    def list_zones(self, camera_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(zone)
                for zone in sorted(
                    self._zones.get(camera_id, {}).values(),
                    key=lambda item: (item["name"].casefold(), item["id"]),
                )
            ]

    def create_zone(self, camera_id: str, zone: ZoneCreate) -> dict[str, Any]:
        zone_id = str(uuid.uuid4())
        record = self._serialize(zone_id, camera_id, zone)
        with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                self._write(connection, record)
            self._zones.setdefault(camera_id, {})[zone_id] = record
        return dict(record)

    def update_zone(
        self, camera_id: str, zone_id: str, zone: ZoneUpdate
    ) -> dict[str, Any] | None:
        record = self._serialize(zone_id, camera_id, zone)
        with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                cursor = connection.execute(
                    """
                    UPDATE detection_zones
                    SET name = ?, polygon_json = ?, enabled = ?,
                        loiter_seconds = ?, event_cooldown_seconds = ?
                    WHERE id = ? AND camera_id = ?
                    """,
                    (
                        record["name"],
                        json.dumps(record["polygon"]),
                        int(record["enabled"]),
                        record["loiter_seconds"],
                        record["event_cooldown_seconds"],
                        zone_id,
                        camera_id,
                    ),
                )
                if cursor.rowcount == 0:
                    return None
            self._zones.setdefault(camera_id, {})[zone_id] = record
        return dict(record)

    def delete_zone(self, camera_id: str, zone_id: str) -> bool:
        with self._lock:
            with sqlite3.connect(self.database_path) as connection:
                cursor = connection.execute(
                    "DELETE FROM detection_zones WHERE id = ? AND camera_id = ?",
                    (zone_id, camera_id),
                )
                if cursor.rowcount == 0:
                    return False
            self._zones.get(camera_id, {}).pop(zone_id, None)
        return True

    def delete_camera_zones(self, camera_id: str) -> None:
        with self._lock, sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "DELETE FROM detection_zones WHERE camera_id = ?", (camera_id,)
            )
            self._zones.pop(camera_id, None)

    @staticmethod
    def _serialize(zone_id: str, camera_id: str, zone: ZoneCreate) -> dict[str, Any]:
        polygon = [point.model_dump() for point in zone.polygon]
        if len({(point["x"], point["y"]) for point in polygon}) < 3:
            raise ValueError("A zone polygon must have at least three distinct points.")
        area = sum(
            polygon[index]["x"] * polygon[(index + 1) % len(polygon)]["y"]
            - polygon[(index + 1) % len(polygon)]["x"] * polygon[index]["y"]
            for index in range(len(polygon))
        )
        if abs(area) < 1e-9:
            raise ValueError("A zone polygon must enclose a non-zero area.")
        return {
            "id": zone_id,
            "camera_id": camera_id,
            "name": zone.name,
            "polygon": polygon,
            "enabled": zone.enabled,
            "loiter_seconds": zone.loiter_seconds,
            "event_cooldown_seconds": zone.event_cooldown_seconds,
        }

    @staticmethod
    def _write(connection: sqlite3.Connection, record: dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO detection_zones (
                id, camera_id, name, polygon_json, enabled,
                loiter_seconds, event_cooldown_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["camera_id"],
                record["name"],
                json.dumps(record["polygon"]),
                int(record["enabled"]),
                record["loiter_seconds"],
                record["event_cooldown_seconds"],
            ),
        )
