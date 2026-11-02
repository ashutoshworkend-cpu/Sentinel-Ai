import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

from app.schemas import RecognitionState


class BiometricEncryptionError(Exception):
    pass


class BiometricPersonNotFoundError(Exception):
    pass


class BiometricConsentRequiredError(Exception):
    pass


class BiometricStore:
    def __init__(self, database_path: Path, encryption_key: str) -> None:
        self.database_path = database_path
        self._fernet: Fernet | None = None
        if encryption_key:
            try:
                self._fernet = Fernet(encryption_key.encode("ascii"))
            except (ValueError, UnicodeEncodeError) as exc:
                raise ValueError(
                    "SENTINEL_BIOMETRIC_KEY must be a Fernet URL-safe key."
                ) from exc

    @property
    def encryption_available(self) -> bool:
        return self._fernet is not None

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS person_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    encrypted_embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recognition_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    tracking_id INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
                    similarity REAL
                );
                """
            )

    def replace_embeddings(
        self,
        person_id: int,
        embeddings: list[np.ndarray],
        consent_confirmed: bool,
    ) -> None:
        if not consent_confirmed:
            raise BiometricConsentRequiredError
        fernet = self._require_fernet()
        if len(embeddings) < 2:
            raise ValueError("At least two suitable enrollment photos are required.")

        encrypted = [
            fernet.encrypt(
                np.asarray(embedding, dtype="<f4").reshape(-1).tobytes()
            )
            for embedding in embeddings
        ]
        consented_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            person = connection.execute(
                "SELECT id, active FROM people WHERE id = ?",
                (person_id,),
            ).fetchone()
            if person is None:
                raise BiometricPersonNotFoundError(person_id)
            if not person["active"]:
                raise ValueError("Reactivate this person before enrollment.")

            connection.execute(
                "DELETE FROM person_embeddings WHERE person_id = ?",
                (person_id,),
            )
            connection.executemany(
                """
                INSERT INTO person_embeddings (
                    person_id, encrypted_embedding, created_at
                ) VALUES (?, ?, ?)
                """,
                [
                    (person_id, embedding, consented_at)
                    for embedding in encrypted
                ],
            )
            connection.execute(
                "UPDATE people SET consented_at = ? WHERE id = ?",
                (consented_at, person_id),
            )

    def withdraw_consent(self, person_id: int) -> bool:
        with self._connect() as connection:
            person = connection.execute(
                "SELECT id FROM people WHERE id = ?",
                (person_id,),
            ).fetchone()
            if person is None:
                return False
            connection.execute(
                "DELETE FROM person_embeddings WHERE person_id = ?",
                (person_id,),
            )
            connection.execute(
                "UPDATE people SET consented_at = NULL, active = 0 WHERE id = ?",
                (person_id,),
            )
        return True

    def active_embeddings(self) -> list[dict]:
        fernet = self._require_fernet()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    person_embeddings.encrypted_embedding,
                    people.id AS person_id,
                    people.display_name,
                    people.category
                FROM person_embeddings
                JOIN people ON people.id = person_embeddings.person_id
                WHERE people.active = 1 AND people.consented_at IS NOT NULL
                """
            ).fetchall()
        results = []
        for row in rows:
            try:
                raw_embedding = fernet.decrypt(row["encrypted_embedding"])
            except InvalidToken as exc:
                raise BiometricEncryptionError(
                    "An encrypted enrollment record could not be decrypted."
                ) from exc
            results.append(
                {
                    "person_id": row["person_id"],
                    "display_name": row["display_name"],
                    "category": row["category"],
                    "embedding": np.frombuffer(raw_embedding, dtype="<f4").copy(),
                }
            )
        return results

    def record_recognition(
        self,
        camera_id: str,
        tracking_id: int,
        timestamp: datetime,
        status: RecognitionState,
        person_id: int | None,
        similarity: float | None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recognition_events (
                    camera_id, tracking_id, timestamp, status, person_id, similarity
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    tracking_id,
                    timestamp.isoformat(),
                    status,
                    person_id,
                    similarity,
                ),
            )

    def list_recognition_events(self, limit: int = 100) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    recognition_events.id,
                    recognition_events.camera_id,
                    recognition_events.tracking_id,
                    recognition_events.timestamp,
                    recognition_events.status,
                    recognition_events.similarity,
                    people.id AS person_id,
                    people.display_name AS person_name,
                    people.category AS person_category
                FROM recognition_events
                LEFT JOIN people ON people.id = recognition_events.person_id
                ORDER BY recognition_events.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def _require_fernet(self) -> Fernet:
        if self._fernet is None:
            raise BiometricEncryptionError(
                "Set SENTINEL_BIOMETRIC_KEY before enrolling or recognizing people."
            )
        return self._fernet

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
