import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.schemas import Detection, PersonCreate


class PersonNotFoundError(Exception):
    pass


class ReviewEventNotFoundError(Exception):
    pass


class PeopleStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    camera_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    detections_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL
                );
                """
            )
            people_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(people)").fetchall()
            }
            if "category" not in people_columns:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN category TEXT NOT NULL DEFAULT 'resident'"
                )
                connection.execute(
                    """
                    UPDATE people SET category = CASE lower(trim(role))
                        WHEN 'visitor' THEN 'visitor'
                        WHEN 'authorized staff' THEN 'authorized_staff'
                        WHEN 'authorized_staff' THEN 'authorized_staff'
                        WHEN 'staff' THEN 'authorized_staff'
                        ELSE 'resident'
                    END
                    """
                )
            if "active" not in people_columns:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN active INTEGER NOT NULL DEFAULT 1"
                )
            if "consented_at" not in people_columns:
                connection.execute(
                    "ALTER TABLE people ADD COLUMN consented_at TEXT"
                )

    def list_people(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    people.id, people.display_name, people.category,
                    people.category AS role, people.active, people.consented_at,
                    people.created_at,
                    COUNT(person_embeddings.id) AS enrollment_count
                FROM people
                LEFT JOIN person_embeddings ON person_embeddings.person_id = people.id
                GROUP BY people.id
                ORDER BY people.display_name COLLATE NOCASE, people.id
                """
            ).fetchall()
        return [
            {**dict(row), "active": bool(row["active"])}
            for row in rows
        ]

    def create_person(self, person: PersonCreate) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO people (
                    display_name, role, created_at, category, active
                )
                VALUES (?, ?, ?, ?, 1)
                """,
                (
                    person.display_name,
                    person.category,
                    created_at,
                    person.category,
                ),
            )
            person_id = cursor.lastrowid
            row = connection.execute(
                """
                SELECT
                    people.id, people.display_name, people.category,
                    people.category AS role, people.active, people.consented_at,
                    people.created_at, 0 AS enrollment_count
                FROM people WHERE id = ?
                """,
                (person_id,),
            ).fetchone()
        return {**dict(row), "active": bool(row["active"])}

    def get_person(self, person_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = self._person_row(connection, person_id)
        return (
            {**dict(row), "active": bool(row["active"])}
            if row is not None
            else None
        )

    def update_person(
        self,
        person_id: int,
        display_name: str,
        category: str,
        active: bool,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE people
                SET display_name = ?, role = ?, category = ?, active = ?
                WHERE id = ?
                """,
                (display_name, category, category, int(active), person_id),
            )
            if cursor.rowcount == 0:
                return None
            row = self._person_row(connection, person_id)
        return {**dict(row), "active": bool(row["active"])}

    def set_consent(self, person_id: int, consented: bool) -> bool:
        consented_at = datetime.now(timezone.utc).isoformat() if consented else None
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE people SET consented_at = ? WHERE id = ?",
                (consented_at, person_id),
            )
        return cursor.rowcount > 0

    def scrub_recognition_identity(self, person_id: int) -> None:
        with self._connect() as connection:
            self._scrub_recognition_identity(connection, person_id)

    def delete_person(self, person_id: int) -> bool:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT id FROM people WHERE id = ?",
                (person_id,),
            ).fetchone()
            if exists is None:
                return False
            self._scrub_recognition_identity(connection, person_id)
            cursor = connection.execute("DELETE FROM people WHERE id = ?", (person_id,))
        return cursor.rowcount > 0

    def create_review_event(
        self,
        camera_id: str,
        camera_name: str,
        timestamp: datetime,
        detections: list[Detection],
    ) -> dict[str, Any]:
        event_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_events (
                    id, camera_id, camera_name, timestamp,
                    detections_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    camera_id,
                    camera_name,
                    timestamp.isoformat(),
                    json.dumps(
                        [detection.model_dump(mode="json") for detection in detections]
                    ),
                    created_at,
                ),
            )
            row = self._event_row(connection, event_id)
        return self._serialize_event(row)

    def list_review_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    review_events.*,
                    people.display_name AS person_name,
                    people.category AS person_role
                FROM review_events
                LEFT JOIN people ON people.id = review_events.person_id
                ORDER BY review_events.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._serialize_event(row) for row in rows]

    def review_event(
        self,
        event_id: str,
        reviewed: bool,
        person_id: int | None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM review_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise ReviewEventNotFoundError(event_id)

            if person_id is not None:
                person = connection.execute(
                    "SELECT id FROM people WHERE id = ?",
                    (person_id,),
                ).fetchone()
                if person is None:
                    raise PersonNotFoundError(person_id)

            reviewed_at = datetime.now(timezone.utc).isoformat() if reviewed else None
            connection.execute(
                """
                UPDATE review_events
                SET reviewed_at = ?, person_id = ?
                WHERE id = ?
                """,
                (reviewed_at, person_id if reviewed else None, event_id),
            )
            row = self._event_row(connection, event_id)
        return self._serialize_event(row)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _person_row(
        connection: sqlite3.Connection,
        person_id: int,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT
                people.id, people.display_name, people.category,
                people.category AS role, people.active, people.consented_at,
                people.created_at,
                COUNT(person_embeddings.id) AS enrollment_count
            FROM people
            LEFT JOIN person_embeddings ON person_embeddings.person_id = people.id
            WHERE people.id = ?
            GROUP BY people.id
            """,
            (person_id,),
        ).fetchone()

    @staticmethod
    def _scrub_recognition_identity(
        connection: sqlite3.Connection,
        person_id: int,
    ) -> None:
        review_rows = connection.execute(
            "SELECT id, detections_json FROM review_events"
        ).fetchall()
        for row in review_rows:
            detections = json.loads(row["detections_json"])
            changed = False
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                if detection.get("recognized_person_id") != person_id:
                    continue
                detection["recognized_person_id"] = None
                detection["recognized_name"] = None
                detection["recognized_category"] = None
                detection["recognition_similarity"] = None
                detection["recognition_status"] = "UNKNOWN"
                changed = True
            if changed:
                connection.execute(
                    "UPDATE review_events SET detections_json = ? WHERE id = ?",
                    (json.dumps(detections), row["id"]),
                )

        recognition_table = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'table' AND name = 'recognition_events'
            """
        ).fetchone()
        if recognition_table is not None:
            connection.execute(
                """
                UPDATE recognition_events
                SET person_id = NULL, status = 'UNKNOWN', similarity = NULL
                WHERE person_id = ?
                """,
                (person_id,),
            )

    @staticmethod
    def _event_row(connection: sqlite3.Connection, event_id: str) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT
                review_events.*,
                people.display_name AS person_name,
                people.category AS person_role
            FROM review_events
            LEFT JOIN people ON people.id = review_events.person_id
            WHERE review_events.id = ?
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            raise ReviewEventNotFoundError(event_id)
        return row

    @staticmethod
    def _serialize_event(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "camera_id": row["camera_id"],
            "camera_name": row["camera_name"],
            "timestamp": row["timestamp"],
            "detections": json.loads(row["detections_json"]),
            "created_at": row["created_at"],
            "reviewed": row["reviewed_at"] is not None,
            "reviewed_at": row["reviewed_at"],
            "person_id": row["person_id"],
            "person_name": row["person_name"],
            "person_role": row["person_role"],
        }
