import asyncio
import cv2
import ipaddress
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.schemas import (
    CameraCreate,
    PersonCreate,
    PersonUpdate,
    ReviewEventCreate,
    ReviewEventUpdate,
    ZoneCreate,
    ZoneUpdate,
)
from app.services.biometric_store import (
    BiometricConsentRequiredError,
    BiometricEncryptionError,
    BiometricPersonNotFoundError,
    BiometricStore,
)
from app.services.camera_manager import CameraManager, EventHub
from app.services.face_recognition import FaceQualityError, FaceRecognitionService
from app.services.people_store import (
    PeopleStore,
    PersonNotFoundError,
    ReviewEventNotFoundError,
)
from app.services.zone_store import ZoneStore
from app.settings import settings

logging.basicConfig(level=logging.INFO)
ALLOWED_FRONTEND_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
event_hub = EventHub()
camera_manager = CameraManager(event_hub)


@asynccontextmanager
async def lifespan(application: FastAPI):
    camera_manager.set_event_loop(asyncio.get_running_loop())
    store = PeopleStore(settings.database_path)
    store.initialize()
    biometric_store = BiometricStore(settings.database_path, settings.biometric_key)
    biometric_store.initialize()
    zone_store = ZoneStore(settings.database_path)
    zone_store.initialize()
    recognition = FaceRecognitionService(
        store=biometric_store,
        detector_path=settings.yunet_model_path,
        recognizer_path=settings.sface_model_path,
        match_threshold=settings.recognition_match_threshold,
        ambiguity_margin=settings.recognition_ambiguity_margin,
        minimum_face_size=settings.recognition_min_face_size,
        minimum_face_score=settings.recognition_min_face_score,
        device=settings.recognition_device,
    )
    camera_manager.set_recognition_service(recognition, biometric_store)
    camera_manager.set_zone_store(zone_store)
    application.state.people_store = store
    application.state.zone_store = zone_store
    application.state.biometric_store = biometric_store
    application.state.face_recognition = recognition
    yield
    camera_manager.stop_all()
    recognition.shutdown()


app = FastAPI(title="SentinelAI API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(ALLOWED_FRONTEND_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)


def get_people_store(request: Request) -> PeopleStore:
    return request.app.state.people_store


def require_local_access(
    request: Request,
) -> None:
    _require_loopback(request.client.host if request.client else None)


def _require_loopback(remote_host: str | None) -> None:
    try:
        is_loopback = remote_host is not None and ipaddress.ip_address(
            remote_host
        ).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise HTTPException(
            status_code=403,
            detail="This service is available only from this device.",
        )


@app.middleware("http")
async def restrict_to_loopback(request: Request, call_next):
    try:
        _require_loopback(request.client.host if request.client else None)
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    return await call_next(request)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/zones/defaults")
def zone_rule_defaults(_: None = Depends(require_local_access)) -> dict[str, int | float]:
    return {
        "min_track_age_seconds": settings.zone_min_track_age_seconds,
        "entry_frames": settings.zone_entry_frames,
        "exit_frames": settings.zone_exit_frames,
        "loiter_seconds": settings.zone_loiter_seconds,
        "event_cooldown_seconds": settings.zone_event_cooldown_seconds,
        "track_loss_tolerance_seconds": settings.zone_track_loss_tolerance_seconds,
        "track_match_distance_fraction": settings.zone_track_match_distance_fraction,
    }


@app.get("/api/cameras")
def list_cameras(_: None = Depends(require_local_access)) -> list[dict]:
    return [camera.status().model_dump(mode="json") for camera in camera_manager.list()]


@app.post("/api/cameras", status_code=201)
def add_camera(camera: CameraCreate, _: None = Depends(require_local_access)) -> dict:
    processor = camera_manager.add(camera)
    return processor.status().model_dump(mode="json")


@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: str, _: None = Depends(require_local_access)) -> dict:
    processor = camera_manager.get(camera_id)
    if processor is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    return processor.status().model_dump(mode="json")


@app.delete("/api/cameras/{camera_id}", status_code=204)
def remove_camera(
    camera_id: str,
    request: Request,
    _: None = Depends(require_local_access),
) -> None:
    if not camera_manager.remove(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found.")
    zone_store: ZoneStore = request.app.state.zone_store
    zone_store.delete_camera_zones(camera_id)


@app.get("/api/cameras/{camera_id}/zones")
def list_camera_zones(
    camera_id: str,
    request: Request,
    _: None = Depends(require_local_access),
) -> list[dict]:
    if camera_manager.get(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    zone_store: ZoneStore = request.app.state.zone_store
    return zone_store.list_zones(camera_id)


@app.post("/api/cameras/{camera_id}/zones", status_code=201)
def create_camera_zone(
    camera_id: str,
    zone: ZoneCreate,
    request: Request,
    _: None = Depends(require_local_access),
) -> dict:
    if camera_manager.get(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    zone_store: ZoneStore = request.app.state.zone_store
    try:
        return zone_store.create_zone(camera_id, zone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.put("/api/cameras/{camera_id}/zones/{zone_id}")
def update_camera_zone(
    camera_id: str,
    zone_id: str,
    zone: ZoneUpdate,
    request: Request,
    _: None = Depends(require_local_access),
) -> dict:
    if camera_manager.get(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera not found.")
    zone_store: ZoneStore = request.app.state.zone_store
    try:
        updated = zone_store.update_zone(camera_id, zone_id, zone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if updated is None:
        raise HTTPException(status_code=404, detail="Zone not found.")
    return updated


@app.delete("/api/cameras/{camera_id}/zones/{zone_id}", status_code=204)
def delete_camera_zone(
    camera_id: str,
    zone_id: str,
    request: Request,
    _: None = Depends(require_local_access),
) -> None:
    zone_store: ZoneStore = request.app.state.zone_store
    if not zone_store.delete_zone(camera_id, zone_id):
        raise HTTPException(status_code=404, detail="Zone not found.")


@app.get("/api/people")
def list_people(
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> list[dict]:
    return store.list_people()


@app.post("/api/people", status_code=201)
def create_person(
    person: PersonCreate,
    request: Request,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> dict:
    created = store.create_person(person)
    request.app.state.face_recognition.invalidate_profiles()
    return created


@app.get("/api/people/{person_id}")
def get_person(
    person_id: int,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> dict:
    person = store.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person record not found.")
    return person


@app.put("/api/people/{person_id}")
def update_person(
    person_id: int,
    update: PersonUpdate,
    request: Request,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> dict:
    person = store.update_person(
        person_id=person_id,
        display_name=update.display_name,
        category=update.category,
        active=update.active,
    )
    if person is None:
        raise HTTPException(status_code=404, detail="Person record not found.")
    request.app.state.face_recognition.invalidate_profiles()
    return person


@app.delete("/api/people/{person_id}", status_code=204)
def delete_person(
    person_id: int,
    request: Request,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> None:
    if not store.delete_person(person_id):
        raise HTTPException(status_code=404, detail="Person record not found.")
    request.app.state.face_recognition.invalidate_profiles()


@app.post("/api/people/{person_id}/enroll", status_code=200)
def enroll_person(
    person_id: int,
    request: Request,
    consent_confirmed: bool = Form(...),
    photos: list[UploadFile] = File(...),
    _: None = Depends(require_local_access),
) -> dict:
    if not consent_confirmed:
        raise HTTPException(
            status_code=422,
            detail="Informed consent is required before enrollment.",
        )
    if len(photos) < 2 or len(photos) > 8:
        raise HTTPException(
            status_code=422,
            detail="Upload between 2 and 8 enrollment photos.",
        )

    recognition: FaceRecognitionService = request.app.state.face_recognition
    biometric_store: BiometricStore = request.app.state.biometric_store
    if not recognition.status()["configured"]:
        raise HTTPException(
            status_code=503,
            detail="Face recognition is not configured. Check local model files and encryption key.",
        )

    embeddings = []
    try:
        for index, photo in enumerate(photos, start=1):
            image_bytes = photo.file.read(5 * 1024 * 1024 + 1)
            if len(image_bytes) > 5 * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"Enrollment photo {index} exceeds the 5 MB limit.",
                )
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise HTTPException(
                    status_code=422,
                    detail=f"Enrollment photo {index} is not a supported image.",
                )
            embeddings.append(recognition.enrollment_embedding(image))
        biometric_store.replace_embeddings(
            person_id=person_id,
            embeddings=embeddings,
            consent_confirmed=consent_confirmed,
        )
        recognition.invalidate_profiles()
    except FaceQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BiometricPersonNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person record not found.") from exc
    except BiometricConsentRequiredError as exc:
        raise HTTPException(status_code=422, detail="Informed consent is required.") from exc
    except BiometricEncryptionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail="The configured local face models could not be loaded.",
        ) from exc
    finally:
        for photo in photos:
            photo.file.close()

    person = request.app.state.people_store.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Person record not found.")
    return person


@app.post("/api/people/{person_id}/consent/withdraw")
def withdraw_person_consent(
    person_id: int,
    request: Request,
    _: None = Depends(require_local_access),
) -> dict:
    biometric_store: BiometricStore = request.app.state.biometric_store
    if not biometric_store.withdraw_consent(person_id):
        raise HTTPException(status_code=404, detail="Person record not found.")
    request.app.state.people_store.scrub_recognition_identity(person_id)
    request.app.state.face_recognition.invalidate_profiles()
    return request.app.state.people_store.get_person(person_id)


@app.get("/api/recognition/status")
def recognition_status(
    request: Request,
    _: None = Depends(require_local_access),
) -> dict:
    recognition: FaceRecognitionService = request.app.state.face_recognition
    return recognition.status()


@app.get("/api/recognition/events")
def recognition_events(
    request: Request,
    _: None = Depends(require_local_access),
) -> list[dict]:
    biometric_store: BiometricStore = request.app.state.biometric_store
    return biometric_store.list_recognition_events()


@app.get("/api/review-events")
def list_review_events(
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> list[dict]:
    return store.list_review_events()


@app.post("/api/review-events", status_code=201)
def create_review_event(
    event: ReviewEventCreate,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> dict:
    return store.create_review_event(
        camera_id=event.camera_id,
        camera_name=event.camera_name,
        timestamp=event.timestamp,
        detections=event.detections,
    )


@app.patch("/api/review-events/{event_id}")
def update_review_event(
    event_id: str,
    update: ReviewEventUpdate,
    _: None = Depends(require_local_access),
    store: PeopleStore = Depends(get_people_store),
) -> dict:
    if update.person_id is not None and not update.reviewed:
        raise HTTPException(
            status_code=422,
            detail="A person can only be associated with a reviewed event.",
        )
    try:
        return store.review_event(
            event_id=event_id,
            reviewed=update.reviewed,
            person_id=update.person_id,
        )
    except PersonNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Person record not found.") from exc
    except ReviewEventNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Review event not found.") from exc


@app.get("/api/cameras/{camera_id}/stream.mjpg")
async def stream_camera(camera_id: str, request: Request) -> StreamingResponse:
    origin = request.headers.get("origin")
    if origin and origin not in ALLOWED_FRONTEND_ORIGINS:
        raise HTTPException(status_code=403, detail="Camera stream origin is not allowed.")
    _require_loopback(request.client.host if request.client else None)
    processor = camera_manager.get(camera_id)
    if processor is None:
        raise HTTPException(status_code=404, detail="Camera not found.")

    async def frames():
        previous_frame: bytes | None = None
        while camera_manager.get(camera_id) is processor:
            frame = processor.latest_jpeg()
            if frame and frame is not previous_frame:
                previous_frame = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.05)

    return StreamingResponse(
        frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket) -> None:
    origin = websocket.headers.get("origin")
    if origin and origin not in ALLOWED_FRONTEND_ORIGINS:
        await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
        return
    remote_host = websocket.client.host if websocket.client else None
    try:
        _require_loopback(remote_host)
    except HTTPException:
        await websocket.close(
            code=1008,
            reason="This service is available only from this device.",
        )
        return
    await event_hub.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        event_hub.disconnect(websocket)
