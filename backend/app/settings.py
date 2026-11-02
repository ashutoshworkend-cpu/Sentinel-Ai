import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    host = os.getenv("SENTINEL_HOST", "127.0.0.1")
    port = int(os.getenv("SENTINEL_PORT", "8000"))
    database_path = Path(
        os.getenv("SENTINEL_DATABASE_PATH")
        or str(Path(__file__).resolve().parents[1] / "data" / "sentinelai.sqlite3")
    )
    zone_min_track_age_seconds = float(
        os.getenv("SENTINEL_ZONE_MIN_TRACK_AGE_SECONDS", "1")
    )
    zone_entry_frames = int(os.getenv("SENTINEL_ZONE_ENTRY_FRAMES", "2"))
    zone_exit_frames = int(os.getenv("SENTINEL_ZONE_EXIT_FRAMES", "2"))
    zone_loiter_seconds = float(os.getenv("SENTINEL_ZONE_LOITER_SECONDS", "30"))
    zone_event_cooldown_seconds = float(
        os.getenv("SENTINEL_ZONE_EVENT_COOLDOWN_SECONDS", "30")
    )
    zone_track_loss_tolerance_seconds = float(
        os.getenv("SENTINEL_ZONE_TRACK_LOSS_TOLERANCE_SECONDS", "2")
    )
    zone_track_match_distance_fraction = float(
        os.getenv("SENTINEL_ZONE_TRACK_MATCH_DISTANCE_FRACTION", "0.03")
    )
    biometric_key = os.getenv("SENTINEL_BIOMETRIC_KEY", "")
    yunet_model_path = Path(
        os.getenv(
            "SENTINEL_YUNET_MODEL_PATH",
            str(Path(__file__).resolve().parents[1] / "models" / "face_detection_yunet_2023mar.onnx"),
        )
    )
    sface_model_path = Path(
        os.getenv(
            "SENTINEL_SFACE_MODEL_PATH",
            str(Path(__file__).resolve().parents[1] / "models" / "face_recognition_sface_2021dec.onnx"),
        )
    )
    recognition_match_threshold = float(
        os.getenv("SENTINEL_RECOGNITION_MATCH_THRESHOLD", "0.45")
    )
    recognition_ambiguity_margin = float(
        os.getenv("SENTINEL_RECOGNITION_AMBIGUITY_MARGIN", "0.05")
    )
    recognition_device = os.getenv("SENTINEL_RECOGNITION_DEVICE", "cpu").lower()
    recognition_interval_frames = int(
        os.getenv("SENTINEL_RECOGNITION_INTERVAL_FRAMES", "15")
    )
    recognition_recheck_frames = int(
        os.getenv("SENTINEL_RECOGNITION_RECHECK_FRAMES", "90")
    )
    recognition_min_face_size = int(
        os.getenv("SENTINEL_RECOGNITION_MIN_FACE_SIZE", "80")
    )
    recognition_min_face_score = float(
        os.getenv("SENTINEL_RECOGNITION_MIN_FACE_SCORE", "0.8")
    )
    model_path = os.getenv("SENTINEL_MODEL", "yolo11n.pt")
    confidence = float(os.getenv("SENTINEL_CONFIDENCE", "0.35"))
    min_class_confidence = float(os.getenv("SENTINEL_MIN_CLASS_CONFIDENCE", "0.55"))
    image_size = int(os.getenv("SENTINEL_IMAGE_SIZE", "640"))
    nms_iou = float(os.getenv("SENTINEL_NMS_IOU", "0.7"))
    temporal_min_observations = int(os.getenv("SENTINEL_TEMPORAL_MIN_OBSERVATIONS", "3"))
    temporal_window = int(os.getenv("SENTINEL_TEMPORAL_WINDOW", "12"))
    temporal_min_support = float(os.getenv("SENTINEL_TEMPORAL_MIN_SUPPORT", "0.65"))
    frame_interval = float(os.getenv("SENTINEL_FRAME_INTERVAL", "0.1"))
    reconnect_delay = float(os.getenv("SENTINEL_RECONNECT_DELAY", "2"))
    verifier_model = os.getenv("SENTINEL_VERIFIER_MODEL", "").strip()
    verifier_min_score = float(os.getenv("SENTINEL_VERIFIER_MIN_SCORE", "0.23"))
    verifier_min_margin = float(os.getenv("SENTINEL_VERIFIER_MIN_MARGIN", "0.04"))
    verifier_cache_ttl_frames = int(
        os.getenv("SENTINEL_VERIFIER_CACHE_TTL_FRAMES", "90")
    )
    verifier_required_classes = frozenset(
        label.strip()
        for label in os.getenv(
            "SENTINEL_VERIFIER_REQUIRED_CLASSES",
            "bird,cat,dog,knife,baseball bat,scissors",
        ).split(",")
        if label.strip()
    )
    verifier_trigger_classes = frozenset(
        label.strip()
        for label in os.getenv(
            "SENTINEL_VERIFIER_TRIGGER_CLASSES",
            "bird,cat,dog,knife,baseball bat,scissors",
        ).split(",")
        if label.strip()
    )


settings = Settings()
