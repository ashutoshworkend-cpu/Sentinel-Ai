import asyncio
import logging
import uuid

from fastapi import WebSocket

from app.schemas import CameraCreate
from app.services.biometric_store import BiometricStore
from app.services.face_recognition import FaceRecognitionService
from app.services.video_processor import VideoProcessor
from app.services.zone_store import ZoneStore

logger = logging.getLogger(__name__)


class EventHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._clients.discard(websocket)

    async def publish(self, event: dict) -> None:
        disconnected = []
        for client in tuple(self._clients):
            try:
                await client.send_json(event)
            except Exception:
                logger.exception("Failed to publish a websocket event")
                disconnected.append(client)
        for client in disconnected:
            self.disconnect(client)


class CameraManager:
    def __init__(self, event_hub: EventHub) -> None:
        self._event_hub = event_hub
        self._loop: asyncio.AbstractEventLoop | None = None
        self._cameras: dict[str, VideoProcessor] = {}
        self._recognition_service: FaceRecognitionService | None = None
        self._biometric_store: BiometricStore | None = None
        self._zone_store: ZoneStore | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_recognition_service(
        self,
        recognition_service: FaceRecognitionService,
        biometric_store: BiometricStore,
    ) -> None:
        self._recognition_service = recognition_service
        self._biometric_store = biometric_store

    def set_zone_store(self, zone_store: ZoneStore) -> None:
        self._zone_store = zone_store

    def add(self, camera: CameraCreate) -> VideoProcessor:
        if self._loop is None:
            raise RuntimeError("Camera manager has not been initialized.")
        camera_id = str(uuid.uuid4())

        def publish_from_thread(event: dict) -> None:
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._event_hub.publish(event),
                    self._loop,
                )

        processor = VideoProcessor(
            camera_id=camera_id,
            name=camera.name,
            source=camera.source,
            on_event=publish_from_thread,
            recognition_service=self._recognition_service,
            biometric_store=self._biometric_store,
            zone_store=self._zone_store,
        )
        self._cameras[camera_id] = processor
        processor.start()
        return processor

    def get(self, camera_id: str) -> VideoProcessor | None:
        return self._cameras.get(camera_id)

    def list(self) -> list[VideoProcessor]:
        return list(self._cameras.values())

    def remove(self, camera_id: str) -> bool:
        processor = self._cameras.pop(camera_id, None)
        if processor is None:
            return False
        processor.stop()
        return True

    def stop_all(self) -> None:
        for processor in self._cameras.values():
            processor.stop()
        self._cameras.clear()
