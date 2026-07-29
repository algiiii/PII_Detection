"""Scoring of a detector against the annotated corpus (block B4, Step 10).

Turns a detector's raw output into **precision / recall / F1** by comparing the
detected spans with the ground-truth spans of the corpus. The full theory —
what a true/false positive means here, the span-matching rule, every formula,
and a worked example — is documented in ``doc/scoring.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pii_detection.detection.protocol import PIIDetector
from pii_detection.detection.types import PIICandidate, TextSpan
from pii_detection.evaluation.corpus import AnnotatedDocument, GroundTruthSpan

#: Default minimum IoU for a detection to count as matching a ground-truth span.
DEFAULT_MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class Metrics:
    """Precision/recall/F1 derived from raw counts.

    :ivar tp: true positives — detections correctly matched to a ground-truth PII.
    :ivar fp: false positives — detections with no matching ground-truth PII.
    :ivar fn: false negatives — ground-truth PII left undetected.
    """

    tp: int
    fp: int
    fn: int

    @property
    def support(self) -> int:
        """:returns: number of ground-truth items for this row (``tp + fn``)."""
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        """Of what was flagged, how much was right.

        :returns: ``tp / (tp + fp)``; ``0.0`` when nothing was detected.
        """
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        """Of what existed, how much was found.

        :returns: ``tp / (tp + fn)``; ``0.0`` when there was nothing to find.
        """
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        """Balance of the two.

        :returns: harmonic mean of precision and recall; ``0.0`` if both are 0.
        """
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated result of evaluating a detector over a corpus.

    :ivar overall: micro-averaged metrics (all categories and documents pooled).
    :ivar per_category: metrics broken down by ``pii_type``, sorted by name.
    """

    overall: Metrics
    per_category: dict[str, Metrics]


def _overlap(candidate: TextSpan, truth: GroundTruthSpan) -> float:
    """IoU between a detected span and a ground-truth span."""
    return candidate.overlap_ratio(TextSpan(truth.start, truth.end))


def _match_document(
    detected: Sequence[PIICandidate],
    truth: Sequence[GroundTruthSpan],
    min_overlap: float,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Greedy one-to-one matching of detections to ground-truth spans.

    A detection matches a ground-truth span when they share the ``pii_type`` and
    their spans overlap with IoU >= ``min_overlap``. Each detection and each
    ground-truth span is used at most once; the highest-overlap pairs win first.

    :returns: three counters keyed by ``pii_type`` — true positives, false
        positives, false negatives.
    """
    # All admissible (detection, ground-truth) pairs, best overlap first.
    pairs: list[tuple[float, int, int]] = []
    for di, det in enumerate(detected):
        for ti, gt in enumerate(truth):
            if det.provenance.pii_type != gt.pii_type:
                continue
            iou = _overlap(det.span, gt)
            if iou >= min_overlap:
                pairs.append((iou, di, ti))
    pairs.sort(reverse=True)

    used_det: set[int] = set()
    used_gt: set[int] = set()
    tp: Counter[str] = Counter()
    for _iou, di, ti in pairs:
        if di in used_det or ti in used_gt:
            continue
        used_det.add(di)
        used_gt.add(ti)
        tp[detected[di].provenance.pii_type] += 1

    fp: Counter[str] = Counter(
        det.provenance.pii_type for di, det in enumerate(detected) if di not in used_det
    )
    fn: Counter[str] = Counter(
        gt.pii_type for ti, gt in enumerate(truth) if ti not in used_gt
    )
    return tp, fp, fn


def evaluate(
    detector: PIIDetector,
    corpus: Iterable[AnnotatedDocument],
    *,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> EvaluationReport:
    """Run a detector over a corpus and score it.

    :param detector: the detector under test.
    :param corpus: annotated documents (clean text + ground-truth spans).
    :param min_overlap: minimum IoU for a detection to match a ground-truth span.
    :returns: overall (micro-averaged) and per-category metrics.
    """
    tp_by: Counter[str] = Counter()
    fp_by: Counter[str] = Counter()
    fn_by: Counter[str] = Counter()
    for doc in corpus:
        detected = detector.detect(doc.text)
        tp, fp, fn = _match_document(detected, doc.spans, min_overlap)
        tp_by.update(tp)
        fp_by.update(fp)
        fn_by.update(fn)

    return _report_from_counters(tp_by, fp_by, fn_by)


def _report_from_counters(
    tp_by: Counter[str], fp_by: Counter[str], fn_by: Counter[str]
) -> EvaluationReport:
    """Assemble an :class:`EvaluationReport` from per-category tp/fp/fn counters.

    Shared by the span scorer (:func:`evaluate`) and the value scorer
    (:func:`evaluate_values`): both only differ in how they count, not in how the
    report is built.

    :returns: overall (micro-averaged) plus per-category metrics.
    """
    categories = set(tp_by) | set(fp_by) | set(fn_by)
    per_category = {
        cat: Metrics(tp_by[cat], fp_by[cat], fn_by[cat]) for cat in sorted(categories)
    }
    overall = Metrics(sum(tp_by.values()), sum(fp_by.values()), sum(fn_by.values()))
    return EvaluationReport(overall=overall, per_category=per_category)


def _normalize_value(value: str) -> str:
    """Collapse whitespace and casefold a PII value for comparison.

    Extraction can insert newlines inside a value (a name wrapped across lines)
    and formats differ in case; normalizing both sides avoids counting such a
    value as missed when it is really there.

    :param value: raw value string.
    :returns: the normalized comparison key.
    """
    return " ".join(value.split()).casefold()


def _match_values(
    detected: Iterable[tuple[str, str]], gold: Iterable[tuple[str, str]]
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Multiset-match ``(pii_type, value)`` detections against gold, one document.

    A detection matches a gold item sharing the ``pii_type`` and the same value
    after :func:`_normalize_value`. Multiplicities are respected: a value present
    twice needs two detections to be fully covered.

    :param detected: detected ``(pii_type, value)`` pairs of one document.
    :param gold: ground-truth ``(pii_type, value)`` pairs of the same document.
    :returns: true-positive, false-positive and false-negative counters, keyed by
        ``pii_type``.
    """
    det: Counter[tuple[str, str]] = Counter(
        (pii_type, _normalize_value(value)) for pii_type, value in detected
    )
    gld: Counter[tuple[str, str]] = Counter(
        (pii_type, _normalize_value(value)) for pii_type, value in gold
    )
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for key in set(det) | set(gld):
        pii_type = key[0]
        matched = min(det[key], gld[key])
        tp[pii_type] += matched
        fp[pii_type] += det[key] - matched
        fn[pii_type] += gld[key] - matched
    return tp, fp, fn


def evaluate_values(
    detected_by_doc: Mapping[str, Iterable[tuple[str, str]]],
    gold_by_doc: Mapping[str, Iterable[tuple[str, str]]],
) -> EvaluationReport:
    """Score detected values against a value-based gold, document by document.

    The Tier-2 (post-extraction) scorer: once extraction has shifted the
    character offsets, matching is by ``(pii_type, normalized value)`` instead of
    by span. Documents are paired by id; a document present on only one side
    contributes only false negatives (gold only) or false positives (detections
    only).

    :param detected_by_doc: detected ``(pii_type, value)`` pairs per ``document_id``.
    :param gold_by_doc: ground-truth ``(pii_type, value)`` pairs per ``document_id``.
    :returns: the same :class:`EvaluationReport` the span scorer produces, so
        :func:`format_report` renders both tiers identically.
    """
    tp_by: Counter[str] = Counter()
    fp_by: Counter[str] = Counter()
    fn_by: Counter[str] = Counter()
    for doc_id in set(gold_by_doc) | set(detected_by_doc):
        tp, fp, fn = _match_values(
            detected_by_doc.get(doc_id, ()), gold_by_doc.get(doc_id, ())
        )
        tp_by.update(tp)
        fp_by.update(fp)
        fn_by.update(fn)
    return _report_from_counters(tp_by, fp_by, fn_by)


def format_report(report: EvaluationReport) -> str:
    """Render a report as a fixed-width table (one row per category + overall).

    :param report: the report to render.
    :returns: a multi-line string ready to ``print``.
    """
    header = (
        f"{'category':<16}{'prec':>7}{'recall':>8}{'f1':>7}"
        f"{'tp':>5}{'fp':>5}{'fn':>5}"
    )
    rule = "-" * len(header)
    lines = [header, rule]
    for cat, m in report.per_category.items():
        lines.append(
            f"{cat:<16}{m.precision:>7.2f}{m.recall:>8.2f}{m.f1:>7.2f}"
            f"{m.tp:>5}{m.fp:>5}{m.fn:>5}"
        )
    o = report.overall
    lines.append(rule)
    lines.append(
        f"{'OVERALL':<16}{o.precision:>7.2f}{o.recall:>8.2f}{o.f1:>7.2f}"
        f"{o.tp:>5}{o.fp:>5}{o.fn:>5}"
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MIN_OVERLAP",
    "Metrics",
    "EvaluationReport",
    "evaluate",
    "evaluate_values",
    "format_report",
]
