from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.settings import settings


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    source: str = Field(
        min_length=1,
        description="Camera index such as '0' or a local/IP camera stream URL.",
    )


class ZonePoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ZoneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    polygon: list[ZonePoint] = Field(min_length=3, max_length=100)
    enabled: bool = True
    loiter_seconds: float = Field(
        default=settings.zone_loiter_seconds, gt=0, le=86400
    )
    event_cooldown_seconds: float = Field(
        default=settings.zone_event_cooldown_seconds, ge=0, le=86400
    )

    @field_validator("name")
    @classmethod
    def strip_zone_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field must contain non-whitespace characters.")
        return normalized


class ZoneUpdate(ZoneCreate):
    pass


PersonCategory = Literal["resident", "visitor", "authorized_staff"]
RecognitionState = Literal[
    "VERIFIED_RESIDENT",
    "VERIFIED_VISITOR",
    "VERIFIED_STAFF",
    "UNKNOWN",
    "UNVERIFIED",
    "FACE_NOT_VISIBLE",
    "IDENTIFYING",
]


class PersonCreate(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    category: PersonCategory

    @field_validator("display_name")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field must contain non-whitespace characters.")
        return normalized


class PersonUpdate(BaseModel):
    display_name: str = Field(min_length=1, max_length=100)
    category: PersonCategory
    active: bool

    @field_validator("display_name")
    @classmethod
    def strip_nonempty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field must contain non-whitespace characters.")
        return normalized


class ReviewEventCreate(BaseModel):
    camera_id: str = Field(min_length=1, max_length=100)
    camera_name: str = Field(min_length=1, max_length=100)
    timestamp: datetime
    detections: list["Detection"] = Field(min_length=1, max_length=100)


class ReviewEventUpdate(BaseModel):
    reviewed: bool
    person_id: int | None = None

    @field_validator("person_id")
    @classmethod
    def valid_person_id(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("person_id must be a positive integer.")
        return value


class Detection(BaseModel):
    class_name: str
    predicted_class: str
    temporal_candidate: str
    final_classification: str | None
    confidence: float
    classification_confidence: float | None
    temporal_support: float
    verification_status: Literal["verified", "primary_only", "uncertain"]
    verification_score: float | None = None
    verification_alternative: str | None = None
    observations: list[str] = Field(default_factory=list)
    object_id: str | None = None
    track_id: int | None = None
    first_seen_frame: int | None = None
    last_seen_frame: int | None = None
    track_observation_count: int = 0
    prediction_history: list[dict[str, str | float | int]] = Field(
        default_factory=list
    )
    movement_pixels: float | None = None
    recognition_status: RecognitionState = "UNVERIFIED"
    recognized_person_id: int | None = None
    recognized_name: str | None = None
    recognized_category: PersonCategory | None = None
    recognition_similarity: float | None = None
    box: list[float]


ReviewEventCreate.model_rebuild()


class CameraStatus(BaseModel):
    id: str
    name: str
    status: Literal["connecting", "online", "offline", "error", "stopped"]
    source_type: Literal["webcam", "network"]
    detections: list[Detection] = Field(default_factory=list)
    timestamp: datetime | None = None
    error: str | None = None
