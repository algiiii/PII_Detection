"""Tests for the Presidio-backed detectors.

Two levels:

* **unit** — a *fake* ``AnalyzerEngine`` returns canned results, so the wrapping
  logic of :class:`PresidioDetector` (REGEX/NER split by recognizer name, entity
  mapping, score→confidence, raw_label, zero-width guard) is checked
  deterministically and fast, without loading spaCy;
* **manual** — one test marked ``manual`` loads the real Italian analyzer and
  checks it actually finds names, validating the Italian NLP configuration (the
  "English by default → recall 0" trap). Deselected by default; run with
  ``pytest -m manual -s -k presidio``.
"""

from __future__ import annotations

import pytest

from pii_detection.detection.config import PresidioEntityModel
from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.presidio_detector import (
    NER_RECOGNIZER_NAMES,
    PresidioDetector,
    build_italian_analyzer,
    build_presidio_detectors,
)
from pii_detection.detection.types import DetectorKind


class _Result:
    """A stand-in for a Presidio ``RecognizerResult``."""

    def __init__(
        self, entity_type: str, start: int, end: int, score: float, recognizer: str
    ) -> None:
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score
        self.recognition_metadata = {"recognizer_name": recognizer}


class _Analyzer:
    """A stand-in for a Presidio ``AnalyzerEngine`` returning fixed results."""

    def __init__(self, results: list[_Result]) -> None:
        self._results = results

    def analyze(self, *, text: str, language: str) -> list[_Result]:
        return self._results


# A pattern (regex/checksum) result and a NER result over the text "IT60 Mario".
_PATTERN = _Result("IBAN_CODE", 0, 4, 0.80, "PatternRecognizer")
_NER = _Result("PERSON", 5, 10, 0.85, "SpacyRecognizer")
_TEXT = "IT60 Mario"
_MAP = {"IBAN_CODE": "iban", "PERSON": "person_name"}


def _detector(kind: DetectorKind, results: list[_Result]) -> PresidioDetector:
    # the fake analyzer stands in for AnalyzerEngine (structural, not a subclass)
    return PresidioDetector("presidio.test", kind, _Analyzer(results), _MAP)  # type: ignore[arg-type]


class TestTechniqueSplit:
    def test_regex_keeps_pattern_and_drops_ner(self) -> None:
        cands = _detector(DetectorKind.REGEX, [_PATTERN, _NER]).detect(_TEXT)
        assert [(c.provenance.pii_type, c.text) for c in cands] == [("iban", "IT60")]

    def test_ner_keeps_ner_and_drops_pattern(self) -> None:
        cands = _detector(DetectorKind.NER, [_PATTERN, _NER]).detect(_TEXT)
        assert [(c.provenance.pii_type, c.text) for c in cands] == [("person_name", "Mario")]


class TestMappingAndGuards:
    def test_entity_absent_from_map_is_dropped(self) -> None:
        unknown = _Result("MEDICAL_LICENSE", 0, 4, 0.9, "PatternRecognizer")
        assert _detector(DetectorKind.REGEX, [unknown]).detect(_TEXT) == []

    def test_zero_width_result_is_skipped(self) -> None:
        empty = _Result("IBAN_CODE", 4, 4, 0.9, "PatternRecognizer")
        assert _detector(DetectorKind.REGEX, [empty]).detect(_TEXT) == []


class TestProvenance:
    def test_regex_candidate_stamps_kind_score_and_no_raw_label(self) -> None:
        (cand,) = _detector(DetectorKind.REGEX, [_PATTERN]).detect(_TEXT)
        assert cand.provenance.detector_kind is DetectorKind.REGEX
        assert cand.provenance.confidence == pytest.approx(0.80)
        assert cand.provenance.raw_label is None
        assert cand.span.start == 0 and cand.span.end == 4

    def test_ner_candidate_carries_raw_label(self) -> None:
        (cand,) = _detector(DetectorKind.NER, [_NER]).detect(_TEXT)
        assert cand.provenance.detector_kind is DetectorKind.NER
        assert cand.provenance.raw_label == "PERSON"  # the original entity_type


class TestBuildPair:
    def test_builds_pattern_and_ner_sharing_the_analyzer(self) -> None:
        analyzer = _Analyzer([_PATTERN, _NER])
        entities = [
            PresidioEntityModel(entity="IBAN_CODE", pii_type="iban"),
            PresidioEntityModel(entity="PERSON", pii_type="person_name"),
        ]
        pattern, ner = build_presidio_detectors(entities, analyzer)  # type: ignore[arg-type]
        assert (pattern.detector_id, pattern.detector_kind) == ("presidio.pattern", DetectorKind.REGEX)
        assert (ner.detector_id, ner.detector_kind) == ("presidio.ner", DetectorKind.NER)
        # each keeps only its technique's result from the shared analyzer
        assert [c.provenance.pii_type for c in pattern.detect(_TEXT)] == ["iban"]
        assert [c.provenance.pii_type for c in ner.detect(_TEXT)] == ["person_name"]


class TestContract:
    def test_satisfies_pii_detector_protocol(self) -> None:
        assert isinstance(_detector(DetectorKind.REGEX, []), PIIDetector)

    def test_spacy_is_a_ner_recognizer_name(self) -> None:
        # guards the split: SpacyRecognizer must be classified as NER.
        assert "SpacyRecognizer" in NER_RECOGNIZER_NAMES


@pytest.mark.manual
def test_real_italian_analyzer_finds_names() -> None:
    """The real Italian analyzer must find person names (not English → recall 0).

    Loads spaCy ``it_core_news_lg``; run with ``pytest -m manual -s``.
    """
    analyzer = build_italian_analyzer()
    entities = [
        PresidioEntityModel(entity="PERSON", pii_type="person_name"),
        PresidioEntityModel(entity="LOCATION", pii_type="address"),
        PresidioEntityModel(entity="IBAN_CODE", pii_type="iban"),
    ]
    pattern, ner = build_presidio_detectors(entities, analyzer)
    text = "Il dipendente Mario Rossi risiede a Milano; IBAN IT60X0542811101000000123456."

    ner_hits = [(c.text, c.provenance.pii_type) for c in ner.detect(text)]
    pattern_hits = [(c.text, c.provenance.pii_type) for c in pattern.detect(text)]
    print(f"\nNER: {ner_hits}\npattern: {pattern_hits}")

    assert any(pii_type == "person_name" for _, pii_type in ner_hits)
