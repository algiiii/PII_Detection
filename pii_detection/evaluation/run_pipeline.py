"""End-to-end pipeline evaluation: the extraction tax (Tier 1 vs Tier 2).

Executable entry point — ``python -m pii_detection.evaluation.run_pipeline`` —
that closes the loop over a rendered corpus. For each document it runs the same
detection pipeline (Presidio pattern + NER, merged by
:class:`~pii_detection.detection.pipeline.MergeEngine`) **twice**:

- on the clean text (Tier 1, no extraction), and
- on the text extracted from the rendered PDF/DOCX by B3 (Tier 2),

then scores both against the value-based gold with
:func:`~pii_detection.evaluation.scoring.evaluate_values`. The gap between the two
F1 scores is the **cost of going through extraction**.

The Presidio setup mirrors :mod:`~pii_detection.evaluation.run_presidio_baseline`;
``--gliner`` swaps spaCy for GLiNER (heavy, container only). ``swiss_avs`` has no
recognizer here, so it shows up as missed in both tiers (as in the baseline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pii_detection.detection.pipeline import MergeEngine
from pii_detection.detection.presidio_detector import build_default_detectors
from pii_detection.detection.protocol import PIIDetector
from pii_detection.evaluation.render import default_rendered_dir, load_gold
from pii_detection.evaluation.scoring import EvaluationReport, evaluate_values, format_report
from pii_detection.extraction import extract_document


def _detect_values(
    text: str,
    pattern: PIIDetector,
    ner: PIIDetector,
    merge: MergeEngine,
    document_id: str,
    ai: PIIDetector | None = None,
) -> list[tuple[str, str]]:
    """Run the pattern+NER (+optional AI) detectors, merge, and return ``(pii_type, value)``.

    The ``ai`` slot lets the benchmark (Step 6) score the union pipeline including
    the generative pass; when ``None`` the merge receives no AI candidates.
    """
    matches = merge.merge(
        pattern.detect(text),
        ner.detect(text),
        ai.detect(text) if ai is not None else (),
        document_id=document_id,
    )
    return [(match.pii_type, match.text) for match in matches]


def evaluate_rendered_corpus(
    directory: Path,
    pattern: PIIDetector,
    ner: PIIDetector,
    *,
    file_format: str = "pdf",
    merge: MergeEngine | None = None,
) -> tuple[EvaluationReport, EvaluationReport]:
    """Score a rendered corpus on both the clean and the extracted text.

    Reads the value gold and, per document, the clean text (``clean/<id>.txt``)
    and the text extracted from ``<id>.<file_format>`` (B3), running the same
    detect+merge on each.

    :param directory: a rendered-corpus directory (``gold.jsonl`` + ``clean/`` +
        the rendered files), as produced by
        :func:`~pii_detection.evaluation.render.render_corpus`.
    :param pattern: the pattern/regex detector.
    :param ner: the NER detector.
    :param file_format: rendered extension to read for the extracted tier.
    :param merge: merge engine to use; a default one is built when omitted.
    :returns: ``(clean_report, extracted_report)``.
    """
    merge = merge if merge is not None else MergeEngine()
    gold = load_gold(directory / "gold.jsonl")
    clean_by_doc: dict[str, list[tuple[str, str]]] = {}
    extracted_by_doc: dict[str, list[tuple[str, str]]] = {}
    for doc_id in sorted(gold):
        clean_text = (directory / "clean" / f"{doc_id}.txt").read_text(encoding="utf-8")
        extracted_text = extract_document(directory / f"{doc_id}.{file_format}").text
        clean_by_doc[doc_id] = _detect_values(clean_text, pattern, ner, merge, doc_id)
        extracted_by_doc[doc_id] = _detect_values(extracted_text, pattern, ner, merge, doc_id)
    return evaluate_values(clean_by_doc, gold), evaluate_values(extracted_by_doc, gold)


def main() -> None:
    """Build the Presidio detectors, evaluate both tiers on the corpus, print."""
    parser = argparse.ArgumentParser(description="End-to-end pipeline evaluation (extraction tax).")
    parser.add_argument("--dir", type=Path, default=default_rendered_dir())
    parser.add_argument("--format", choices=["pdf", "docx"], default="pdf", dest="file_format")
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="use GLiNER for the NER instead of spaCy (heavy; container only)",
    )
    args = parser.parse_args()

    pattern, ner = build_default_detectors(use_gliner=args.gliner)

    clean_report, extracted_report = evaluate_rendered_corpus(
        args.dir, pattern, ner, file_format=args.file_format
    )

    print("=== Tier 1 — clean text (detection only) ===")
    print(format_report(clean_report))
    print(f"\n=== Tier 2 — extracted from {args.file_format.upper()} (B3 + detection) ===")
    print(format_report(extracted_report))
    tax = clean_report.overall.f1 - extracted_report.overall.f1
    print(f"\nExtraction tax (F1 clean - F1 extracted): {tax:+.3f}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_rendered_corpus", "main"]
