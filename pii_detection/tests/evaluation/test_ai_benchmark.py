"""Tests for the multi-model AI benchmark (fake detectors, no codecarbon/Ollama)."""

from __future__ import annotations

from pii_detection.detection.types import (
    DetectionProvenance,
    DetectorKind,
    PIICandidate,
    TextSpan,
)
from pii_detection.evaluation.corpus import AnnotatedDocument, GroundTruthSpan
from pii_detection.evaluation.run_ai_benchmark import (
    BenchmarkRow,
    MergedPipelineDetector,
    format_benchmark,
    format_per_category,
    run_benchmark,
)
from pii_detection.evaluation.scoring import EvaluationReport, Metrics


class _Sub:
    """Substring detector of a given kind — emits a candidate per found value."""

    def __init__(self, targets: list[tuple[str, str]], kind: DetectorKind) -> None:
        self.detector_id = f"{kind.value}.fake"
        self.detector_kind = kind
        self._targets = targets

    def detect(self, text: str) -> list[PIICandidate]:
        out: list[PIICandidate] = []
        for pii_type, value in self._targets:
            index = text.find(value)
            if index >= 0:
                out.append(
                    PIICandidate(
                        TextSpan(index, index + len(value)),
                        value,
                        DetectionProvenance(self.detector_id, self.detector_kind, pii_type, 0.8),
                    )
                )
        return out


def _empty(kind: DetectorKind) -> _Sub:
    return _Sub([], kind)


def test_merged_pipeline_detector_returns_merged_spans() -> None:
    text = "Mario Rossi, IBAN IT60X0542811101000000123456"
    pattern = _Sub([("iban", "IT60X0542811101000000123456")], DetectorKind.REGEX)
    ner = _Sub([("person_name", "Mario Rossi")], DetectorKind.NER)
    ai = _Sub([("person_name", "Mario Rossi")], DetectorKind.AI)  # confirms the NER

    candidates = MergedPipelineDetector(pattern, ner, ai).detect(text)

    assert {c.provenance.pii_type for c in candidates} == {"iban", "person_name"}


def test_baseline_has_no_ai_contribution() -> None:
    text = "Mario Rossi here"
    ner = _Sub([("person_name", "Mario Rossi")], DetectorKind.NER)
    # ai=None -> baseline: only what pattern+ner find.
    candidates = MergedPipelineDetector(_empty(DetectorKind.REGEX), ner, None).detect(text)
    assert [c.provenance.pii_type for c in candidates] == ["person_name"]


def test_run_benchmark_rows_and_no_energy() -> None:
    corpus = [
        AnnotatedDocument("d1", "Mario Rossi", (GroundTruthSpan(0, 11, "person_name"),)),
    ]
    pattern, ner = _empty(DetectorKind.REGEX), _empty(DetectorKind.NER)

    def ai_for(model: str) -> _Sub:
        return _Sub([("person_name", "Mario Rossi")], DetectorKind.AI)

    rows = run_benchmark(["m1", "m2"], corpus, pattern, ner, ai_for, energy=False)

    assert [row.label for row in rows] == [
        "presidio (pattern+ner)",
        "ai:m1",
        "union:m1",
        "ai:m2",
        "union:m2",
    ]
    assert all(row.wh_per_doc is None for row in rows)  # energy off -> no codecarbon
    # The AI alone recovers the name the empty baseline misses.
    baseline = next(row for row in rows if row.label == "presidio (pattern+ner)")
    ai_row = next(row for row in rows if row.label == "ai:m1")
    assert baseline.report.overall.recall == 0.0
    assert ai_row.report.overall.recall == 1.0


def test_format_benchmark_marks_missing_energy() -> None:
    row = BenchmarkRow("ai:m", EvaluationReport(Metrics(1, 0, 0), {}), 1.5, None)
    out = format_benchmark([row])
    assert "ai:m" in out
    assert "—" in out  # em dash where energy is unavailable
    assert "1.50" in out or "1.500" in out  # seconds per doc rendered


def test_format_per_category_expands_each_row() -> None:
    report = EvaluationReport(
        overall=Metrics(1, 0, 1),
        per_category={"person_name": Metrics(1, 0, 0), "iban": Metrics(0, 0, 1)},
    )
    rows = [BenchmarkRow("union:m", report, 1.0, None)]

    out = format_per_category(rows)

    assert "=== union:m ===" in out  # each row is labelled
    assert "person_name" in out and "iban" in out  # per-category breakdown printed
    assert "OVERALL" in out  # the report's overall line is kept


class TestSkippedChunksAreReported:
    """A row measured while chunks were dropped must say so, not look healthy."""

    @staticmethod
    def _row(label: str, seen: int, failed: int) -> BenchmarkRow:
        report = EvaluationReport(overall=Metrics(1, 0, 0), per_category={})
        return BenchmarkRow(label, report, 1.0, 0.1, seen, failed)

    def test_clean_row_shows_no_skips_and_no_warning(self) -> None:
        out = format_benchmark([self._row("ai:m", 10, 0)])
        assert "0/10" in out
        assert "WARNING" not in out

    def test_baseline_row_has_no_chunk_column(self) -> None:
        out = format_benchmark([self._row("presidio (pattern+ner)", 0, 0)])
        assert "WARNING" not in out

    def test_damaged_row_warns_with_share(self) -> None:
        out = format_benchmark([self._row("ai:big", 128, 34)])
        assert "34/128" in out
        assert "WARNING" in out
        assert "27%" in out


class TestAiOnlyMode:
    """Passing no base detectors measures the ai: rows alone (memory headroom)."""

    def test_ai_only_drops_baseline_and_union(self) -> None:
        corpus = [
            AnnotatedDocument("d1", "chiama 3401234567", (GroundTruthSpan(7, 17, "phone"),))
        ]
        rows = run_benchmark(
            ["m1", "m2"], corpus, None, None, lambda _m: _empty(DetectorKind.AI), energy=False
        )
        assert [r.label for r in rows] == ["ai:m1", "ai:m2"]

    def test_full_mode_still_has_baseline_and_union(self) -> None:
        corpus = [
            AnnotatedDocument("d1", "chiama 3401234567", (GroundTruthSpan(7, 17, "phone"),))
        ]
        rows = run_benchmark(
            ["m1"],
            corpus,
            _empty(DetectorKind.REGEX),
            _empty(DetectorKind.NER),
            lambda _m: _empty(DetectorKind.AI),
            energy=False,
        )
        assert [r.label for r in rows] == ["presidio (pattern+ner)", "ai:m1", "union:m1"]
