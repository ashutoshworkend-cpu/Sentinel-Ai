from collections import OrderedDict

import numpy as np

from app.services.verification import VerificationResult
from app.services.video_processor import VideoProcessor


class Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class FakeBox:
    cls = [Scalar(14)]
    conf = [Scalar(0.9)]
    id = [Scalar(7)]

    class Coordinates:
        def tolist(self) -> list[float]:
            return [5.0, 5.0, 25.0, 35.0]

    xyxy = [Coordinates()]


class FakeResult:
    boxes = [FakeBox()]
    names = {14: "bird"}


class FakeVerifier:
    def __init__(self, agrees: bool) -> None:
        self.agrees = agrees
        self.calls = 0

    def verify(
        self,
        frame: np.ndarray,
        box: list[float],
        expected_class: str,
    ) -> VerificationResult:
        assert expected_class == "bird"
        self.calls += 1
        return VerificationResult(
            agrees=self.agrees,
            score=0.4,
            alternative="bird" if self.agrees else "broom",
        )


def make_processor() -> VideoProcessor:
    return VideoProcessor("camera-1", "Test", "0", lambda _: None)


def run_observation(
    processor: VideoProcessor,
    frame_index: int,
    verifier: FakeVerifier | None,
    cache: OrderedDict,
) -> dict:
    return processor._extract_detections(
        result=FakeResult(),
        frame=np.zeros((40, 40, 3), dtype=np.uint8),
        frame_index=frame_index,
        verifier=verifier,
        verification_cache=cache,
    )[0].model_dump()


def test_unverified_animal_is_never_returned_as_final_class() -> None:
    processor = make_processor()
    cache = OrderedDict()
    detections = [
        run_observation(processor, frame_index, None, cache)
        for frame_index in range(3)
    ]

    assert all(item["class_name"] == "unknown" for item in detections)
    assert all(item["final_classification"] is None for item in detections)
    assert detections[-1]["predicted_class"] == "bird"


def test_disagreeing_crop_verifier_leaves_track_unknown_and_is_cached() -> None:
    processor = make_processor()
    verifier = FakeVerifier(agrees=False)
    cache = OrderedDict()
    detections = [
        run_observation(processor, frame_index, verifier, cache)
        for frame_index in range(3)
    ]

    assert all(item["class_name"] == "unknown" for item in detections)
    assert detections[-1]["verification_alternative"] == "broom"
    assert verifier.calls == 1


def test_agreeing_verifier_waits_for_temporal_evidence() -> None:
    processor = make_processor()
    verifier = FakeVerifier(agrees=True)
    cache = OrderedDict()

    first = run_observation(processor, 0, verifier, cache)
    second = run_observation(processor, 1, verifier, cache)
    third = run_observation(processor, 2, verifier, cache)

    assert first["class_name"] == "unknown"
    assert second["class_name"] == "unknown"
    assert third["class_name"] == "bird"
    assert third["verification_status"] == "verified"
    assert verifier.calls == 1


def test_crop_verification_cache_is_refreshed_after_its_frame_ttl() -> None:
    processor = make_processor()
    verifier = FakeVerifier(agrees=False)
    cache = OrderedDict()

    run_observation(processor, 0, verifier, cache)
    run_observation(processor, 1, verifier, cache)
    run_observation(processor, 2, verifier, cache)
    run_observation(processor, 93, verifier, cache)

    assert verifier.calls == 2
