"""Tests for the document scan CLI (B3 + B4).

``scan_document`` and ``format_matches`` are exercised with fake detectors on a
plain-text file, so no Presidio nor optional deps are needed.
"""

from __future__ import annotations

from pathlib import Path

from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    PIIMatch,
    TextSpan,
)
from pii_detection.scan import format_matches, scan_document


class _SubstringDetector:
    """Finds each given ``(pii_type, value)`` as a literal substring of the text."""

    detector_id = "fake.substr"
    detector_kind = DetectorKind.REGEX

    def __init__(self, targets: list[tuple[str, str]]) -> None:
        self._targets = targets

    def detect(self, text: str) -> list[PIICandidate]:
        found: list[PIICandidate] = []
        for pii_type, value in self._targets:
            index = text.find(value)
            if index >= 0:
                found.append(
                    PIICandidate(
                        span=TextSpan(index, index + len(value)),
                        text=value,
                        provenance=DetectionProvenance(
                            "fake.substr", DetectorKind.REGEX, pii_type, 0.9
                        ),
                    )
                )
        return found


class _EmptyDetector:
    detector_id = "fake.empty"
    detector_kind = DetectorKind.NER

    def detect(self, text: str) -> list[PIICandidate]:
        return []


class _AIDetector:
    """Fake AI detector: finds each ``(pii_type, value)`` as a substring, kind AI."""

    detector_id = "ai.fake"
    detector_kind = DetectorKind.AI

    def __init__(self, targets: list[tuple[str, str]]) -> None:
        self._targets = targets

    def detect(self, text: str) -> list[PIICandidate]:
        found: list[PIICandidate] = []
        for pii_type, value in self._targets:
            index = text.find(value)
            if index >= 0:
                found.append(
                    PIICandidate(
                        span=TextSpan(index, index + len(value)),
                        text=value,
                        provenance=DetectionProvenance("ai.fake", DetectorKind.AI, pii_type, 0.6),
                    )
                )
        return found


def test_scan_txt_returns_matches_in_order(tmp_path: Path) -> None:
    path = tmp_path / "note.txt"
    path.write_text(
        "Scrivi a mario@x.it, IBAN IT60X0542811101000000123456.", encoding="utf-8"
    )
    pattern = _SubstringDetector(
        [("email", "mario@x.it"), ("iban", "IT60X0542811101000000123456")]
    )

    matches = scan_document(path, pattern, _EmptyDetector())

    assert {(m.pii_type, m.text) for m in matches} == {
        ("email", "mario@x.it"),
        ("iban", "IT60X0542811101000000123456"),
    }
    assert [m.span.start for m in matches] == sorted(m.span.start for m in matches)


def test_ai_candidates_reach_the_output(tmp_path: Path) -> None:
    """An AI detector passed to scan_document contributes its discoveries."""
    path = tmp_path / "note.txt"
    path.write_text("Il paziente ha una cardiopatia nota.", encoding="utf-8")
    ai = _AIDetector([("health_data", "cardiopatia")])

    matches = scan_document(path, _EmptyDetector(), _EmptyDetector(), ai)

    discovered = [m for m in matches if m.confirmation_level is ConfirmationLevel.AI_DISCOVERED]
    assert len(discovered) == 1
    assert discovered[0].pii_type == "health_data"


def test_no_ai_detector_means_no_ai_candidates(tmp_path: Path) -> None:
    """Without an AI detector, no AI_DISCOVERED match appears (default off)."""
    path = tmp_path / "note.txt"
    path.write_text("Il paziente ha una cardiopatia nota.", encoding="utf-8")

    matches = scan_document(path, _EmptyDetector(), _EmptyDetector())

    assert all(m.confirmation_level is not ConfirmationLevel.AI_DISCOVERED for m in matches)


def test_format_matches_lists_type_value_and_level() -> None:
    provenance = DetectionProvenance("presidio.pattern", DetectorKind.REGEX, "email", 0.9)
    match = PIIMatch(
        span=TextSpan(0, 6),
        text="a@b.it",
        pii_type="email",
        confidence=0.9,
        confirmation_level=ConfirmationLevel.SINGLE_SOURCE,
        sources=[provenance],
        document_id="doc",
    )
    out = format_matches([match], document_id="doc")
    assert "a@b.it" in out
    assert "email" in out
    assert "single_source" in out
    assert "presidio.pattern" in out


def test_format_matches_empty() -> None:
    assert "No PII found" in format_matches([], document_id="doc")
