"""Tests for the end-to-end pipeline evaluation wiring.

Exercises ``evaluate_rendered_corpus`` with fake detectors (no Presidio): an
ideal substring detector finds every value in the clean text, and extraction can
only cost recall — never improve it — so the extraction tax is non-negative.
Needs ``fpdf2``/``fitz`` to build and read the tiny rendered corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fpdf")
pytest.importorskip("fitz")

from pii_detection.detection.types import (  # noqa: E402
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.evaluation.render import load_gold, render_corpus  # noqa: E402
from pii_detection.evaluation.run_pipeline import evaluate_rendered_corpus  # noqa: E402


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
    """A detector that never fires (the NER slot in this test)."""

    detector_id = "fake.empty"
    detector_kind = DetectorKind.NER

    def detect(self, text: str) -> list[PIICandidate]:
        return []


def test_extraction_tax_is_non_negative(tmp_path: Path) -> None:
    render_corpus(tmp_path, n=6, seed=42, formats=["pdf"])
    gold = load_gold(tmp_path / "gold.jsonl")
    targets = [pair for items in gold.values() for pair in items]

    clean_report, extracted_report = evaluate_rendered_corpus(
        tmp_path, _SubstringDetector(targets), _EmptyDetector(), file_format="pdf"
    )

    # The ideal detector finds every value present verbatim in the clean text.
    assert clean_report.overall.recall == 1.0
    # Extraction can only reshape text, so it never raises F1 above the clean run.
    assert extracted_report.overall.f1 <= clean_report.overall.f1
