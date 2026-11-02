import numpy as np

from app.services.verification import ClipVerifier, VERIFICATION_LABELS


class FakeClassifier:
    def __init__(self, predictions: list[dict]) -> None:
        self.predictions = predictions

    def __call__(self, image: np.ndarray, **_: object) -> list[dict]:
        assert image.shape == (10, 10, 3)
        return self.predictions


def make_verifier(predictions: list[dict]) -> ClipVerifier:
    verifier = ClipVerifier.__new__(ClipVerifier)
    verifier._min_score = 0.23
    verifier._min_margin = 0.04
    verifier._classifier = FakeClassifier(predictions)
    return verifier


def test_verifier_contains_unseen_household_categories() -> None:
    assert "broom" in VERIFICATION_LABELS
    assert "mop" in VERIFICATION_LABELS
    assert "unidentified household object" in VERIFICATION_LABELS


def test_disagreement_does_not_replace_primary_prediction() -> None:
    verifier = make_verifier(
        [
            {"label": "broom", "score": 0.35},
            {"label": "bird", "score": 0.20},
        ]
    )

    result = verifier.verify(
        np.zeros((20, 20, 3), dtype=np.uint8),
        [0, 0, 10, 10],
        expected_class="bird",
    )

    assert result.agrees is False
    assert result.alternative == "broom"
