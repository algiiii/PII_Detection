"""Tests for the merge engine (block B4), on synthetic candidates.

The merge is document-agnostic, so it is exercised with hand-built
:class:`PIICandidate` instances (no Presidio, no real detectors): each test
places spans by offset and checks the resulting ``confirmation_level``,
``confidence`` and ``sources`` of the produced :class:`PIIMatch`.
"""

from __future__ import annotations

import pytest

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.types import (
    ConfirmationLevel,
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)

DOC = "doc-1"


def _cand(
    start: int,
    end: int,
    pii_type: str,
    kind: DetectorKind,
    *,
    confidence: float = 0.6,
) -> PIICandidate:
    """Build a synthetic candidate at ``[start, end)`` with the given identity."""
    return PIICandidate(
        span=TextSpan(start, end),
        text="*" * (end - start),
        provenance=DetectionProvenance(
            detector_id=f"{kind.value}.test",
            detector_kind=kind,
            pii_type=pii_type,
            confidence=confidence,
        ),
    )


def _regex(start: int, end: int, pii_type: str, *, confidence: float = 0.6) -> PIICandidate:
    return _cand(start, end, pii_type, DetectorKind.REGEX, confidence=confidence)


def _ner(start: int, end: int, pii_type: str, *, confidence: float = 0.6) -> PIICandidate:
    return _cand(start, end, pii_type, DetectorKind.NER, confidence=confidence)


def _ai(start: int, end: int, pii_type: str, *, confidence: float = 0.6) -> PIICandidate:
    return _cand(start, end, pii_type, DetectorKind.AI, confidence=confidence)


class TestDoubleConfirmed:
    def test_same_type_overlap_merges_into_one_match(self) -> None:
        """Regex and NER agreeing on ``pii_type`` over overlapping spans collapse
        to a single DOUBLE_CONFIRMED match carrying both provenances."""
        engine = MergeEngine()
        (match,) = engine.merge(
            [_regex(0, 10, "iban")], [_ner(0, 10, "iban")], document_id=DOC
        )
        assert match.confirmation_level is ConfirmationLevel.DOUBLE_CONFIRMED
        assert len(match.sources) == 2
        assert {p.detector_kind for p in match.sources} == {DetectorKind.REGEX, DetectorKind.NER}
        assert match.document_id == DOC

    def test_bonus_added_to_the_stronger_confidence(self) -> None:
        """Confidence of a double confirmation is ``max(the two) + bonus``."""
        engine = MergeEngine(double_confirmation_bonus=0.15)
        (match,) = engine.merge(
            [_regex(0, 10, "iban", confidence=0.6)],
            [_ner(0, 10, "iban", confidence=0.8)],
            document_id=DOC,
        )
        assert match.confidence == pytest.approx(0.95)

    def test_bonus_is_clamped_to_one(self) -> None:
        """The bonus can never push confidence above 1.0."""
        engine = MergeEngine(double_confirmation_bonus=0.15)
        (match,) = engine.merge(
            [_regex(0, 10, "iban", confidence=0.95)],
            [_ner(0, 10, "iban", confidence=0.9)],
            document_id=DOC,
        )
        assert match.confidence == pytest.approx(1.0)

    def test_regex_span_and_text_are_kept_as_representative(self) -> None:
        """On a double confirmation the regex span/text win (patterns are exact,
        NER spans can be loose)."""
        engine = MergeEngine()
        regex = _regex(2, 12, "iban")
        (match,) = engine.merge([regex], [_ner(0, 14, "iban")], document_id=DOC)
        assert match.span == regex.span
        assert match.text == regex.text


class TestConflicting:
    def test_overlap_with_disagreeing_type_keeps_both(self) -> None:
        """Overlapping spans but different ``pii_type``: no arbitration, both are
        kept as CONFLICTING (recall-first), one provenance each, for B5."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "person_name")], [_ner(0, 10, "address")], document_id=DOC
        )
        assert len(matches) == 2
        assert all(m.confirmation_level is ConfirmationLevel.CONFLICTING for m in matches)
        assert all(len(m.sources) == 1 for m in matches)
        assert {m.pii_type for m in matches} == {"person_name", "address"}


class TestSingleSource:
    def test_regex_without_partner_is_single_source(self) -> None:
        engine = MergeEngine()
        (match,) = engine.merge([_regex(0, 10, "iban")], [], document_id=DOC)
        assert match.confirmation_level is ConfirmationLevel.SINGLE_SOURCE
        assert len(match.sources) == 1

    def test_ner_without_partner_is_single_source(self) -> None:
        engine = MergeEngine()
        (match,) = engine.merge([], [_ner(0, 10, "person_name")], document_id=DOC)
        assert match.confirmation_level is ConfirmationLevel.SINGLE_SOURCE

    def test_overlap_below_threshold_does_not_confirm(self) -> None:
        """IoU under ``min_overlap_ratio`` is not the same PII: both candidates
        survive independently as SINGLE_SOURCE, never merged nor dropped."""
        engine = MergeEngine(min_overlap_ratio=0.5)
        # [0,10) vs [8,20): intersection 2, union 20 -> IoU 0.1 < 0.5.
        matches = engine.merge([_regex(0, 10, "iban")], [_ner(8, 20, "iban")], document_id=DOC)
        assert len(matches) == 2
        assert all(m.confirmation_level is ConfirmationLevel.SINGLE_SOURCE for m in matches)

    def test_lower_threshold_lets_the_same_pair_confirm(self) -> None:
        """The threshold is honored: the pair that stayed single at 0.5 confirms
        once the threshold drops below its IoU."""
        engine = MergeEngine(min_overlap_ratio=0.05)
        (match,) = engine.merge([_regex(0, 10, "iban")], [_ner(8, 20, "iban")], document_id=DOC)
        assert match.confirmation_level is ConfirmationLevel.DOUBLE_CONFIRMED


class TestBestPartner:
    def test_regex_pairs_with_the_highest_overlap_ner(self) -> None:
        """When several NER candidates overlap a regex one, the best IoU wins and
        the others are not consumed (they fall through as SINGLE_SOURCE)."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "iban")],
            [_ner(0, 5, "iban"), _ner(0, 10, "iban")],  # IoU 0.5 vs 1.0
            document_id=DOC,
        )
        levels = sorted(m.confirmation_level.value for m in matches)
        assert levels == ["double_confirmed", "single_source"]

    def test_one_ner_is_not_reused_by_two_regex(self) -> None:
        """A consumed NER candidate cannot double-confirm a second regex one."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "iban"), _regex(0, 10, "iban")],
            [_ner(0, 10, "iban")],
            document_id=DOC,
        )
        levels = sorted(m.confirmation_level.value for m in matches)
        assert levels == ["double_confirmed", "single_source"]


class TestAIAbsorb:
    """The AI pass as confirmer/discoverer — the six rows of the design table."""

    def test_ai_covering_a_new_span_is_discovered(self) -> None:
        """No overlap: an AI candidate touching no match is added AI_DISCOVERED."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "iban")], [], [_ai(50, 60, "health_data")], document_id=DOC
        )
        discovered = [m for m in matches if m.confirmation_level is ConfirmationLevel.AI_DISCOVERED]
        assert len(discovered) == 1
        assert discovered[0].pii_type == "health_data"

    def test_ai_confirms_a_single_source_into_double(self) -> None:
        """Same type over a SINGLE_SOURCE match: escalates to DOUBLE_CONFIRMED,
        the AI provenance is appended and the confidence gains the bonus."""
        engine = MergeEngine(double_confirmation_bonus=0.15)
        (match,) = engine.merge(
            [_regex(0, 10, "iban", confidence=0.6)], [], [_ai(0, 10, "iban")], document_id=DOC
        )
        assert match.confirmation_level is ConfirmationLevel.DOUBLE_CONFIRMED
        assert [p.detector_kind for p in match.sources] == [DetectorKind.REGEX, DetectorKind.AI]
        assert match.confidence == pytest.approx(0.75)

    def test_ai_confirms_a_double_confirmed_as_third_source(self) -> None:
        """Same type over a DOUBLE_CONFIRMED match: stays DOUBLE_CONFIRMED, now
        carrying three provenances, confidence bumped (clamped)."""
        engine = MergeEngine(double_confirmation_bonus=0.15)
        (match,) = engine.merge(
            [_regex(0, 10, "iban", confidence=0.6)],
            [_ner(0, 10, "iban", confidence=0.6)],
            [_ai(0, 10, "iban")],
            document_id=DOC,
        )
        assert match.confirmation_level is ConfirmationLevel.DOUBLE_CONFIRMED
        assert len(match.sources) == 3
        assert {p.detector_kind for p in match.sources} == {
            DetectorKind.REGEX, DetectorKind.NER, DetectorKind.AI
        }

    def test_ai_agreeing_with_a_conflicting_match_stays_conflicting(self) -> None:
        """Same type over a CONFLICTING match: the AI provenance is appended but
        the type disagreement is not resolved here — it stays CONFLICTING (B5)."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "person_name")],
            [_ner(0, 10, "address")],
            [_ai(0, 10, "person_name")],
            document_id=DOC,
        )
        assert all(m.confirmation_level is ConfirmationLevel.CONFLICTING for m in matches)
        confirmed = next(m for m in matches if m.pii_type == "person_name")
        assert DetectorKind.AI in {p.detector_kind for p in confirmed.sources}

    def test_ai_disagreeing_with_a_single_source_demotes_both(self) -> None:
        """Different type over a SINGLE_SOURCE match: the existing one is demoted
        to CONFLICTING and the AI is kept as its own CONFLICTING match."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "person_name")], [], [_ai(0, 10, "address")], document_id=DOC
        )
        assert len(matches) == 2
        assert all(m.confirmation_level is ConfirmationLevel.CONFLICTING for m in matches)
        assert {m.pii_type for m in matches} == {"person_name", "address"}

    def test_ai_disagreeing_with_a_double_confirmed_does_not_demote_it(self) -> None:
        """Different type over a DOUBLE_CONFIRMED match: the regex+NER agreement is
        not undone by a single AI opinion; the AI is kept as CONFLICTING."""
        engine = MergeEngine()
        matches = engine.merge(
            [_regex(0, 10, "iban")], [_ner(0, 10, "iban")], [_ai(0, 10, "person_name")], document_id=DOC
        )
        kept = next(m for m in matches if m.pii_type == "iban")
        ai_match = next(m for m in matches if m.pii_type == "person_name")
        assert kept.confirmation_level is ConfirmationLevel.DOUBLE_CONFIRMED
        assert ai_match.confirmation_level is ConfirmationLevel.CONFLICTING

    def test_ai_below_threshold_is_discovered_not_confirmed(self) -> None:
        """An AI candidate overlapping a match but under the IoU threshold is not
        a confirmation: it survives on its own as AI_DISCOVERED (recall-first)."""
        engine = MergeEngine(min_overlap_ratio=0.5)
        # [0,10) vs [8,20): IoU 0.1 < 0.5 — not the same PII.
        matches = engine.merge(
            [_regex(0, 10, "iban")], [], [_ai(8, 20, "iban")], document_id=DOC
        )
        assert len(matches) == 2
        assert any(m.confirmation_level is ConfirmationLevel.AI_DISCOVERED for m in matches)


class TestEdges:
    def test_empty_input_yields_no_matches(self) -> None:
        assert MergeEngine().merge([], [], document_id=DOC) == []

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_min_overlap_ratio_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            MergeEngine(min_overlap_ratio=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_bonus_raises(self, bad: float) -> None:
        with pytest.raises(ValueError):
            MergeEngine(double_confirmation_bonus=bad)
