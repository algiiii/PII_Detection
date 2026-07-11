"""Tests for the detector contract and the optional base (Step 2)."""

from __future__ import annotations

import pytest

from pii_detection.detection.protocol import BaseDetector, PIIDetector
from pii_detection.detection.types import DetectorKind, PIICandidate, TextSpan


class _FakeDetector:
    """Satisfies PIIDetector by structural typing, without inheriting anything."""

    detector_id = "fake.v1"
    detector_kind = DetectorKind.REGEX

    def detect(self, text: str) -> list[PIICandidate]:
        return []


class TestProtocol:
    def test_structural_instance_without_inheritance(self) -> None:
        assert isinstance(_FakeDetector(), PIIDetector)

    def test_missing_detect_is_not_a_detector(self) -> None:
        class NoDetect:
            detector_id = "x"
            detector_kind = DetectorKind.NER

        assert not isinstance(NoDetect(), PIIDetector)

    def test_base_detector_satisfies_protocol_once_detect_added(self) -> None:
        class Concrete(BaseDetector):
            def detect(self, text: str) -> list[PIICandidate]:
                return []

        assert isinstance(Concrete("c.v1", DetectorKind.AI), PIIDetector)


class TestBuildCandidate:
    def _detector(self) -> BaseDetector:
        return BaseDetector("regex.iban_v1", DetectorKind.REGEX)

    def test_slices_text_and_stamps_provenance(self) -> None:
        text = "IBAN: IT60X0542811101000000123456 fine"
        span = TextSpan(6, 33)
        cand = self._detector().build_candidate(text, span, "iban", 0.6)

        assert cand.text == text[6:33]
        assert cand.span == span
        assert cand.provenance.detector_id == "regex.iban_v1"
        assert cand.provenance.detector_kind == DetectorKind.REGEX
        assert cand.provenance.pii_type == "iban"
        assert cand.provenance.confidence == 0.6

    def test_optional_provenance_fields(self) -> None:
        cand = self._detector().build_candidate(
            "aaaa", TextSpan(0, 4), "x", 0.5, checksum_validated=True
        )
        assert cand.provenance.checksum_validated is True

    def test_span_beyond_text_raises(self) -> None:
        with pytest.raises(ValueError):
            self._detector().build_candidate("short", TextSpan(0, 99), "x", 0.5)

    def test_invalid_confidence_propagates(self) -> None:
        with pytest.raises(ValueError):
            self._detector().build_candidate("aaaa", TextSpan(0, 4), "x", 1.5)
