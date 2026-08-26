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
from pii_detection.evaluation.run_pipeline import (  # noqa: E402
    evaluate_enterprise_corpus,
    evaluate_rendered_corpus,
)


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


def _write_enterprise(root: Path) -> None:
    """Build a minimal enterprise-corpus layout: gold + sources + tree."""
    (root / "tree" / "HR").mkdir(parents=True)
    (root / "tree" / "HR" / "nota.txt").write_text(
        "Contatto: mario@example.com in copia.", encoding="utf-8"
    )
    (root / "tree" / "HR" / "vuoto.txt").write_text("Nessun dato qui.", encoding="utf-8")
    (root / "gold.jsonl").write_text(
        '{"document_id": "HR/nota.txt", "pii": [{"pii_type": "email", "value": "mario@example.com"}]}\n'
        '{"document_id": "HR/vuoto.txt", "pii": []}\n',
        encoding="utf-8",
    )
    (root / "sources.jsonl").write_text(
        '{"document_id": "HR/nota.txt", "annotated": "Contatto: {{email:mario@example.com}} in copia."}\n'
        '{"document_id": "HR/vuoto.txt", "annotated": "Nessun dato qui."}\n',
        encoding="utf-8",
    )


def test_enterprise_corpus_scores_both_tiers(tmp_path: Path) -> None:
    """The tree layout scores like the rendered one: ids carry their own extension."""
    _write_enterprise(tmp_path)
    targets = [("email", "mario@example.com")]

    clean_report, extracted_report = evaluate_enterprise_corpus(
        tmp_path, _SubstringDetector(targets), _EmptyDetector()
    )

    # The value is in the clean text and survives .txt extraction untouched.
    assert clean_report.overall.recall == 1.0
    assert extracted_report.overall.recall == 1.0
    # The PII-free document contributes no false positive.
    assert clean_report.overall.fp == 0


def test_enterprise_unreadable_file_costs_recall_without_crashing(tmp_path: Path) -> None:
    """A file B3 cannot read is a Tier-2 miss, not an aborted run."""
    _write_enterprise(tmp_path)
    # Replace the note with a PDF that is not a PDF: extraction must fail on it.
    (tmp_path / "tree" / "HR" / "nota.txt").unlink()
    (tmp_path / "tree" / "HR" / "nota.pdf").write_text("non e' un pdf", encoding="utf-8")
    for name in ("gold.jsonl", "sources.jsonl"):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace("nota.txt", "nota.pdf"), "utf-8")

    clean_report, extracted_report = evaluate_enterprise_corpus(
        tmp_path, _SubstringDetector([("email", "mario@example.com")]), _EmptyDetector()
    )

    assert clean_report.overall.recall == 1.0
    assert extracted_report.overall.recall == 0.0


def test_enterprise_missing_sources_raises(tmp_path: Path) -> None:
    _write_enterprise(tmp_path)
    (tmp_path / "sources.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_enterprise_corpus(tmp_path, _EmptyDetector(), _EmptyDetector())
