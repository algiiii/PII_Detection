"""Tests for CompositeDetector — running several detectors in one slot."""

from __future__ import annotations

from pii_detection.detection.composite import CompositeDetector
from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)


class _Fake:
    """Fake detector emitting a fixed set of ``(start, end, pii_type)`` spans."""

    detector_kind = DetectorKind.REGEX

    def __init__(self, detector_id: str, spans: list[tuple[int, int, str]]) -> None:
        self.detector_id = detector_id
        self._spans = spans

    def detect(self, text: str) -> list[PIICandidate]:
        return [
            PIICandidate(
                TextSpan(start, end),
                text[start:end],
                DetectionProvenance(self.detector_id, DetectorKind.REGEX, pii_type, 0.9),
            )
            for start, end, pii_type in self._spans
        ]


def test_composite_concatenates_in_order() -> None:
    a = _Fake("a", [(0, 3, "iban")])
    b = _Fake("b", [(5, 8, "email")])
    composite = CompositeDetector("comp", [a, b])

    candidates = composite.detect("abc  def")

    assert [c.provenance.detector_id for c in candidates] == ["a", "b"]
    assert [c.provenance.pii_type for c in candidates] == ["iban", "email"]


def test_composite_empty_returns_no_candidates() -> None:
    assert CompositeDetector("comp", []).detect("whatever") == []
