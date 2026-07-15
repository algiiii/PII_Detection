"""Tests for the detector scoring (Step 10)."""

from __future__ import annotations

import pytest

from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.evaluation.corpus import AnnotatedDocument, GroundTruthSpan
from pii_detection.evaluation.scoring import EvaluationReport, Metrics, evaluate


def _cand(start: int, end: int, pii_type: str) -> PIICandidate:
    """A detection at ``[start, end)`` of the given type."""
    provenance = DetectionProvenance(
        detector_id="x",
        detector_kind=DetectorKind.REGEX,
        pii_type=pii_type,
        confidence=0.9,
    )
    return PIICandidate(span=TextSpan(start, end), text="?" * (end - start), provenance=provenance)


class _FixedDetector:
    """Detector returning a fixed list, ignoring the text (structural PIIDetector)."""

    detector_id = "fixed"
    detector_kind = DetectorKind.REGEX

    def __init__(self, candidates: list[PIICandidate]) -> None:
        self._candidates = candidates

    def detect(self, text: str) -> list[PIICandidate]:
        return list(self._candidates)


def _doc(*spans: GroundTruthSpan) -> AnnotatedDocument:
    """A document whose text is irrelevant (the fixed detector ignores it)."""
    return AnnotatedDocument("d", "." * 200, tuple(spans))


def _report(
    cands: list[PIICandidate], spans: list[GroundTruthSpan], min_overlap: float = 0.5
) -> EvaluationReport:
    return evaluate(_FixedDetector(cands), [_doc(*spans)], min_overlap=min_overlap)


class TestCounting:
    def test_perfect_match(self) -> None:
        r = _report([_cand(0, 10, "iban")], [GroundTruthSpan(0, 10, "iban")])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (1, 0, 0)
        assert r.overall.precision == 1.0
        assert r.overall.recall == 1.0

    def test_type_mismatch_is_fp_and_fn(self) -> None:
        r = _report([_cand(0, 10, "credit_card")], [GroundTruthSpan(0, 10, "iban")])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (0, 1, 1)
        assert r.per_category["credit_card"].fp == 1
        assert r.per_category["iban"].fn == 1

    def test_missing_detection_is_fn(self) -> None:
        r = _report([], [GroundTruthSpan(0, 10, "iban")])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (0, 0, 1)
        assert r.overall.recall == 0.0

    def test_extra_detection_is_fp(self) -> None:
        r = _report([_cand(0, 10, "iban")], [])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (0, 1, 0)
        assert r.overall.precision == 0.0


class TestOverlapThreshold:
    def test_below_threshold_no_match(self) -> None:
        # IoU = 4/10 = 0.4 < 0.5
        r = _report([_cand(0, 4, "iban")], [GroundTruthSpan(0, 10, "iban")])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (0, 1, 1)

    def test_above_threshold_matches(self) -> None:
        # IoU = 6/10 = 0.6 >= 0.5
        r = _report([_cand(0, 6, "iban")], [GroundTruthSpan(0, 10, "iban")])
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (1, 0, 0)

    def test_threshold_is_configurable(self) -> None:
        gt = [GroundTruthSpan(0, 10, "iban")]  # IoU of a [0,6) detection is 0.6
        assert _report([_cand(0, 6, "iban")], gt, min_overlap=0.7).overall.tp == 0
        assert _report([_cand(0, 6, "iban")], gt, min_overlap=0.5).overall.tp == 1


class TestOneToOne:
    def test_two_detections_one_truth(self) -> None:
        # both overlap the single truth; only the best matches, the other is FP
        r = _report(
            [_cand(0, 10, "iban"), _cand(0, 9, "iban")], [GroundTruthSpan(0, 10, "iban")]
        )
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (1, 1, 0)

    def test_one_detection_two_truths(self) -> None:
        r = _report(
            [_cand(0, 10, "iban")],
            [GroundTruthSpan(0, 10, "iban"), GroundTruthSpan(0, 9, "iban")],
        )
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (1, 0, 1)


class TestWorkedExample:
    def test_matches_the_numbers_in_scoring_md(self) -> None:
        cands = [_cand(0, 20, "iban"), _cand(30, 40, "email"), _cand(50, 62, "credit_card")]
        gts = [
            GroundTruthSpan(0, 20, "iban"),
            GroundTruthSpan(30, 45, "email"),
            GroundTruthSpan(50, 62, "phone"),
        ]
        r = _report(cands, gts)
        assert (r.overall.tp, r.overall.fp, r.overall.fn) == (2, 1, 1)
        assert r.overall.precision == pytest.approx(2 / 3)
        assert r.overall.recall == pytest.approx(2 / 3)
        assert r.overall.f1 == pytest.approx(2 / 3)


class TestMetrics:
    def test_division_guards_return_zero(self) -> None:
        assert Metrics(0, 0, 0).precision == 0.0
        assert Metrics(0, 0, 0).recall == 0.0
        assert Metrics(0, 0, 0).f1 == 0.0

    def test_values(self) -> None:
        m = Metrics(tp=1, fp=1, fn=0)
        assert m.precision == pytest.approx(0.5)
        assert m.recall == 1.0
        assert m.f1 == pytest.approx(2 * 0.5 * 1 / (0.5 + 1))
        assert m.support == 1
