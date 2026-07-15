"""Scoring of a detector against the annotated corpus (block B4, Step 10).

Turns a detector's raw output into **precision / recall / F1** by comparing the
detected spans with the ground-truth spans of the corpus. The full theory —
what a true/false positive means here, the span-matching rule, every formula,
and a worked example — is documented in ``doc/scoring.md``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
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

    categories = set(tp_by) | set(fp_by) | set(fn_by)
    per_category = {
        cat: Metrics(tp_by[cat], fp_by[cat], fn_by[cat]) for cat in sorted(categories)
    }
    overall = Metrics(sum(tp_by.values()), sum(fp_by.values()), sum(fn_by.values()))
    return EvaluationReport(overall=overall, per_category=per_category)


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
    "format_report",
]
